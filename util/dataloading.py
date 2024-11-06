import json
import numpy as np
import cv2
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from copy import deepcopy
import skimage
import scipy
import basicpy

from .box_ops_numpy import mask_to_boxes, cxcywh_to_xyxy


base_datadir = Path('/ictstr01/groups/shared/users/lion.gleiter/organoid_sam/original_data/')


def compute_median_size_image(boxes):
    boxes = np.array(boxes).astype(float)
    heights = boxes[:, 2] - boxes[:, 0]
    widths = boxes[:, 3] - boxes[:, 1]
    median_size = np.median(np.concatenate([heights, widths], axis=0), axis=0)
    return median_size


def compute_median_size(ds):
    sizes = []
    for im, mask, boxes, im_path, im_ID in ds:
        boxes = np.array(boxes).astype(float)
        if boxes.shape[0]==0: # No boxes
            continue
        heights = boxes[:, 2] - boxes[:, 0]
        widths = boxes[:, 3] - boxes[:, 1]
        sizes.append(heights)
        sizes.append(widths)
    median_size = np.median(np.concatenate(sizes, axis=0), axis=0)
    return median_size


def compute_patch_size(median_size):
    patch_size = int(median_size * 8)
    return patch_size


def normalize(image, smoothness=10):
    # Normalize dtype
    image = image.astype(float)

    # Normalize channels
    if image.ndim == 2:  # Convert to RGB
        image = np.stack([image, image, image], axis=-1)
    else:  # Assert RGB image
        assert image.ndim==3, image.shape
        if image.shape[2] == 4:
            # Assuming 4th channel is alpha and almost 1 everywhere
            assert np.max(np.abs(image[:, :, 3] / (image[:, :, 3]).max() - 1)) < 0.01, np.unique(image[:, :, 3])
            image = image[:, :, :3]
        assert image.shape[2]==3, image.shape

    # Normalize range
    image /= np.maximum(np.quantile(image, 0.99, axis=(0,1), keepdims=True), 1e-3)

    # Apply background correction
    basic_successful = True
    channels = []
    flatfields = []
    try:
        for i in range(image.shape[2]):
            basic = basicpy.BaSiC(get_darkfield=False, smoothness_flatfield=smoothness)
            basic.fit(image[None, :, :, i])
            images_transformed = basic.transform(deepcopy(image[None, :, :, i]))
            channels.append(images_transformed[0])
            flatfields.append(basic.flatfield)
        flatfields = np.stack(flatfields, axis=-1)
        if np.any(np.isnan(flatfields)):
            basic_successful = False
    except RuntimeError:
        basic_successful = False

    if not basic_successful:
        # Instead use gaussian blur for background correction
        channels = []
        flatfields = []
        for i in range(image.shape[2]):
            bg = scipy.ndimage.gaussian_filter(image[:, :, i], sigma = (128, 128))
            images_transformed = deepcopy(image[:, :, i]) - (bg - bg.mean())
            channels.append(images_transformed)
            flatfields.append(bg)
        flatfields = np.stack(flatfields, axis=-1)
    image = np.stack(channels, axis=-1)

    # Normalize quantiles
    lower = np.quantile(image, 0.001, axis=(0,1), keepdims=True)
    upper = np.quantile(image, 0.999, axis=(0,1), keepdims=True)
    image = (image - lower) / np.maximum(upper - lower, 1e-3)
    image = np.clip(image * 255, 0, 255).astype(np.uint8)
    return image, flatfields


def filter_boxes_yxyx(boxes, ymin, xmin, ymax, xmax, threshold=10):
    """Boxes should be [[ymin, xmin, ymax, xmax]]. Threshold determines the necessary intersection area to include a box."""
    boxes = np.array(boxes).astype(float)
    assert boxes.ndim == 2, boxes.shape
    assert boxes.shape[1] == 4, boxes.shape
    
    y_left = np.maximum(ymin, boxes[:, 0])
    y_right = np.minimum(ymax, boxes[:, 2])
    x_left = np.maximum(xmin, boxes[:, 1])
    x_right = np.minimum(xmax, boxes[:, 3])
    
    intersection = np.maximum(x_right - x_left, 0) * np.maximum(y_right - y_left, 0)

    has_overlap = intersection >= threshold

    boxes_out = deepcopy(boxes)
    boxes_out = boxes_out[has_overlap]
    boxes_out = np.minimum(boxes_out, np.array([[ymax, xmax, ymax, xmax]]))
    boxes_out = np.maximum(boxes_out, np.array([[ymin, xmin, ymin, xmin]]))
    boxes_out -= np.array([[ymin, xmin, ymin, xmin]], dtype=float)

    return boxes_out


def patch(im, mask, boxes, size=512):
    H, W = im.shape[:2]
    padding = int(size / 12)

    # Avoid patching if the patch size is almost the image size:
    if size >= 0.8 * max(H, W):
        yield im, mask, boxes, 0, 0
        return

    # Compute number of patches per side
    n_patches_x = 1
    if W > size:
        n_patches_x += int(np.round((W-size) / (size - 2*padding) + 0.3))
    n_patches_y = 1
    if H > size:
        n_patches_y += int(np.round((H-size) / (size - 2*padding) + 0.3))

    size_x = int(np.ceil((W + n_patches_x * 2 * padding) / n_patches_x))
    size_y = int(np.ceil((H + n_patches_y * 2 * padding) / n_patches_y))
    size = max(size_x, size_y)

    grid_x = np.round(np.linspace(0, W-size, n_patches_x)).astype(int).tolist()
    grid_y = np.round(np.linspace(0, H-size, n_patches_y)).astype(int).tolist()

    # Iteratively yield patches
    for im_start_x in grid_x:
        for im_start_y in grid_y:
            requires_zero_background = False
            im_end_x = im_start_x + size
            im_end_y = im_start_y + size

            if W < im_end_x:
                requires_zero_background = True
                im_end_x = W
                # im_start_x = max(0, im_end_x - size)
                # else:
                #     im_end_x = im_start_x + size
            # else:
            #     im_end_x = im_start_x + size
            
            
            if H < im_end_y:
                requires_zero_background = True
                im_end_y = H
                    # im_start_y = max(0, im_end_y - size)
                # else:
                #     im_end_y = im_start_y + size
            # else:
            #     im_end_y = im_start_y + size

            # print(i, j, n_patches_x, n_patches_y)
            # print(H, W, im_start_x, im_start_y, im_end_x, im_end_y)
            # Crop image and generate grid
            mask_crop = None
            if not requires_zero_background:
                im_crop = im[int(im_start_y):int(im_end_y), int(im_start_x):int(im_end_x)]
                if mask is not None:
                    mask_crop = mask[int(im_start_y):int(im_end_y), int(im_start_x):int(im_end_x)]
            else:
                if im.ndim == 2:
                    im_crop = np.zeros((size, size), dtype=im.dtype)
                    im_crop[:int(im_end_y)-int(im_start_y), :int(im_end_x)-int(im_start_x)] = im[int(im_start_y):int(im_end_y), int(im_start_x):int(im_end_x)]
                else:
                    im_crop = np.zeros((size, size, im.shape[2]), dtype=im.dtype)
                    im_crop[:int(im_end_y)-int(im_start_y), :int(im_end_x)-int(im_start_x), :] = im[int(im_start_y):int(im_end_y), int(im_start_x):int(im_end_x)]
                
                if mask is not None:
                    mask_crop = np.zeros((size, size), dtype=mask.dtype)
                    mask_crop[:int(im_end_y)-int(im_start_y), :int(im_end_x)-int(im_start_x)] = mask[int(im_start_y):int(im_end_y), int(im_start_x):int(im_end_x)]

            boxes_crop = filter_boxes_yxyx(boxes, ymin=im_start_y, xmin=im_start_x, ymax=im_end_y, xmax=im_end_x, threshold=10)

            yield im_crop, mask_crop, boxes_crop, int(im_start_x), int(im_start_y)



class OrgaQuant:
    def __init__(self, split=None, data_folder = base_datadir):
        self.split = split
        self.train_annotations = pd.read_csv(data_folder / "Intestinal Organoid Dataset/Intestinal Organoid Dataset/train_labels.csv", header=None)
        self.train_annotations.columns = ["image_path", "x1", "y1", "x2", "y2", "class_name"]
        self.test_annotations = pd.read_csv(data_folder / "Intestinal Organoid Dataset/Intestinal Organoid Dataset/test_labels.csv")

        self.train_annotations['name'] = self.train_annotations['image_path'].apply(lambda x: Path(x).name)
        self.test_annotations['name'] = self.test_annotations['image_path'].apply(lambda x: Path(x).name)

        train_organoids = sorted(list(data_folder.glob("Intestinal Organoid Dataset/Intestinal Organoid Dataset/train/*.*")))
        test_organoids = sorted(list(data_folder.glob("Intestinal Organoid Dataset/Intestinal Organoid Dataset/test/*.*")))

        if split == 'train':
            self.images = train_organoids
            self.annotations = self.train_annotations
        elif split == 'test':
            self.images = test_organoids
            self.annotations = self.test_annotations
        else:
            raise ValueError(split)

    def __getitem__(self, i):
        im_path = self.images[i]
        im_ID = im_path.stem

        boxes = self.annotations[self.annotations['name'] == im_path.name]
        boxes = [[b.y1, b.x1, b.y2, b.x2] for b in boxes.itertuples(index=False)]
        boxes = np.array(boxes).astype(float).reshape((-1, 4))
        
        im = skimage.io.imread(im_path, as_gray=False)

        mask = None

        return im, mask, boxes, im_path, im_ID
    
    def __str__(self):
        return 'OrgaQuant'

    def __len__(self):
        return len(self.images)
    

class NeurIPSCellSeg:
    def __init__(self, split=None, data_folder = base_datadir):
        self.split = split
        train_neurips = sorted(list(data_folder.glob("NeurIPS_CellSeg/Training-labeled/Training-labeled/images/*.*")))

        val_neurips = sorted(list(data_folder.glob("NeurIPS_CellSeg/Tuning/Tuning/images/*.*")))[:-1]

        test_neurips = sorted(list(data_folder.glob("NeurIPS_CellSeg/Testing/Testing/Public/images/*.*")))
        # test_neurips_wsi = sorted(list(data_folder.glob("NeurIPS_CellSeg/Testing/Testing/Public/WSI/*.*")))
        
        train_annotations = pd.read_csv(data_folder / "NeurIPS_CellSeg/Training-labeled/Training-labeled/boxes.csv")
        val_annotations = pd.read_csv(data_folder / "NeurIPS_CellSeg/Tuning/Tuning/boxes.csv")
        test_annotations = pd.read_csv(data_folder / "NeurIPS_CellSeg/Testing/Testing/Public/boxes.csv")
        # test_annotations_wsi = pd.read_csv(data_folder / "NeurIPS_CellSeg/Training-labeled/Training-labeled/boxes_WSI.csv")

        
        if split == 'train':
            self.images = train_neurips
            self.annotations = train_annotations
        elif split == 'val':
            self.images = val_neurips
            self.annotations = val_annotations
        elif split == 'test':
            self.images = test_neurips
            self.annotations = test_annotations
        # elif split == 'test_wsi':
        #     self.images = test_neurips_wsi
        #     self.annotations = test_annotations_wsi
        # elif split is None:
        #     self.images = train_neurips + val_neurips + test_neurips  # + test_neurips_wsi
        #     self.annotations = pd.concat([train_annotations, 
        #                                   val_annotations,
        #                                   test_annotations,
        #                                 #   test_annotations_wsi
        #                                   ], axis=0, ignore_index=True)
        else:
            raise ValueError(split)

    def __getitem__(self, i):
        im_path = self.images[i]
        im_ID = im_path.stem

        # Boxes
        boxes = self.annotations[np.logical_and(
            self.annotations['name'] == im_path.name,
            self.annotations['image_path'].map(lambda x: Path(x).parent.parent.name == im_path.parent.parent.name)
        )]
        boxes = [[b.y1, b.x1, b.y2, b.x2] for b in boxes.itertuples(index=False)]
        boxes = np.array(boxes).astype(float).reshape((-1, 4))
        
        # Image
        im = skimage.io.imread(im_path, as_gray=False)

        # Instance masks
        mask_path = im_path.parent.parent / 'labels' / f"{im_path.stem}_label.tiff"
        mask = skimage.io.imread(mask_path, as_gray=True)

        return im, mask, boxes, im_path, im_ID
    
    def __str__(self):
        return 'NeurIPSCellSeg'
        
    def __len__(self):
        return len(self.images)


class OrganoID:
    def __init__(self, split=None, data_folder = base_datadir):
        self.split = split
        train = sorted(list(data_folder.glob("OrganoID/OriginalData/training/images/*.*")))
        train_orig = sorted(list(data_folder.glob("OrganoID/OriginalData/training/pre_augmented/images/*.*")))

        val = sorted(list(data_folder.glob("OrganoID/OriginalData/validation/images/*.*")))

        test = sorted(list(data_folder.glob("OrganoID/OriginalData/testing/images/PDAC*.*")))
        test_ACC = sorted(list(data_folder.glob("OrganoID/OriginalData/testing/images/ACC*.*")))
        test_Lung = sorted(list(data_folder.glob("OrganoID/OriginalData/testing/images/Lung*.*")))
        test_C = sorted(list(data_folder.glob("OrganoID/OriginalData/testing/images/C*.*")))
        test_mouse = sorted(list(data_folder.glob("OrganoID/MouseOrganoids/testing/images/*.*"))) + \
            sorted(list(data_folder.glob("OrganoID/MouseOrganoids/validation/images/*.*"))) + \
            sorted(list(data_folder.glob("OrganoID/MouseOrganoids/training/pre_augmented/images/*.*")))

        
        if split == 'train':
            self.images = train
        elif split == 'train_orig':
            self.images = train_orig
        elif split == 'val':
            self.images = val
        elif split == 'test':
            self.images = test
        elif split == 'test_ACC':
            self.images = test_ACC
        elif split == 'test_Lung':
            self.images = test_Lung
        elif split == 'test_C':
            self.images = test_C
        elif split == 'test_mouse':
            self.images = test_mouse
        else:
            raise ValueError(split)

    def __getitem__(self, i):
        im_path = self.images[i]
        im_ID = im_path.stem
        
        # Image
        im = skimage.io.imread(im_path, as_gray=False)

        # Instance masks
        mask_path = im_path.parent.parent / 'segmentations' / f"{im_path.stem}.png"
        mask = skimage.io.imread(mask_path, as_gray=True)
        mask = (mask > 0.5).astype(np.uint8)
        mask = skimage.measure.label(mask, return_num=False, connectivity=1)

        # Boxes
        boxes = mask_to_boxes(mask)
        boxes = np.array(boxes).astype(float).reshape((-1, 4))

        return im, mask, boxes, im_path, im_ID
    
    def __str__(self):
        return 'OrganoID'
        
    def __len__(self):
        return len(self.images)


class OrgaExtractor:
    def __init__(self, split=None, data_folder = base_datadir):
        self.split = split
        train = sorted(list(data_folder.glob("OrgaExtractor/train/input_*.*")))

        # val = sorted(list(data_folder.glob("OrganoID/OriginalData/validation/images/*.*")))

        test = sorted(list(data_folder.glob("OrgaExtractor/test/input_*.*")))

        
        if split == 'train':
            self.images = train
        # elif split == 'val':
        #     self.images = val
        elif split == 'test':
            self.images = test
        else:
            raise ValueError(split)

    def __getitem__(self, i):
        im_path = self.images[i]
        im_ID = im_path.stem
        
        # Image
        im = skimage.io.imread(im_path, as_gray=False)

        # Instance masks
        mask_path = im_path.parent / f"label_{im_path.stem[6:]}.tif"
        mask = skimage.io.imread(mask_path, as_gray=True)
        mask = (mask > 0.5).astype(np.uint8)
        mask = skimage.measure.label(mask, return_num=False, connectivity=1)

        # Boxes
        boxes = mask_to_boxes(mask)
        boxes = np.array(boxes).astype(float).reshape((-1, 4))

        return im, mask, boxes, im_path, im_ID
    
    def __str__(self):
        return 'OrgaExtractor'
        
    def __len__(self):
        return len(self.images)
    

class OrgaSegment:
    def __init__(self, split=None, data_folder = base_datadir):
        self.split = split
        train = sorted(list(data_folder.glob("Orgasegment/OrganoidBasic_v20211206/train/*_img.*")))

        val = sorted(list(data_folder.glob("Orgasegment/OrganoidBasic_v20211206/val/*_img.*")))

        test = sorted(list(data_folder.glob("Orgasegment/OrganoidBasic_v20211206/eval/*_img.*")))

        
        if split == 'train':
            self.images = train
        elif split == 'val':
            self.images = val
        elif split == 'test':
            self.images = test
        else:
            raise ValueError(split)

    def __getitem__(self, i):
        im_path = self.images[i]
        im_ID = im_path.stem
        
        # Image
        im = skimage.io.imread(im_path, as_gray=False)

        # Instance masks
        mask_path = im_path.parent / f"{im_path.stem[:-4]}_masks_organoid.png"
        mask = skimage.io.imread(mask_path, as_gray=True)
        # print(np.unique(mask), mask.dtype)
        # print(mask)
        # mask = (mask > 0.5).astype(np.uint8)
        # mask = skimage.measure.label(mask, return_num=False, connectivity=1)

        # Boxes
        boxes = mask_to_boxes(mask)
        boxes = np.array(boxes).astype(float).reshape((-1, 4))

        return im, mask, boxes, im_path, im_ID
    
    def __str__(self):
        return 'OrgaSegment'
        
    def __len__(self):
        return len(self.images)


class NewData:
    def __init__(self, split=None, data_folder = base_datadir):
        self.split = split
        train = sorted(list(data_folder.glob("own_data/organoid_img/*.*")))

        if split == 'train':
            self.images = train
        else:
            raise ValueError(split)

    def __getitem__(self, i):
        im_path = self.images[i]
        im_ID = im_path.stem
        
        # Image
        im = skimage.io.imread(im_path, as_gray=False)

        # Instance masks
        mask_path = im_path.parent.parent / 'labels' / f"{im_path.stem}_labels.tif"
        mask = skimage.io.imread(mask_path, as_gray=True)
        # print(np.unique(mask), mask.dtype)
        # print(mask)
        # mask = (mask > 0.5).astype(np.uint8)
        # mask = skimage.measure.label(mask, return_num=False, connectivity=1)

        # Boxes
        boxes = mask_to_boxes(mask)
        boxes = np.array(boxes).astype(float).reshape((-1, 4))

        return im, mask, boxes, im_path, im_ID
    
    def __str__(self):
        return 'NewData'
        
    def __len__(self):
        return len(self.images)


class Tellu:
    def __init__(self, split=None, data_folder = base_datadir):
        self.split = split
        self.train_annotations = pd.read_csv(data_folder / "Intestinal Organoid Dataset/Intestinal Organoid Dataset/train_labels.csv", header=None)
        
        self.train_annotations.columns = ["image_path", "x1", "y1", "x2", "y2", "class_name"]
        self.test_annotations = pd.read_csv(data_folder / "Intestinal Organoid Dataset/Intestinal Organoid Dataset/test_labels.csv")

        self.train_annotations['name'] = self.train_annotations['image_path'].apply(lambda x: Path(x).name)
        self.test_annotations['name'] = self.test_annotations['image_path'].apply(lambda x: Path(x).name)

        train_organoids = sorted(list(data_folder.glob("Tellu/OrganoidDataset/train/images/*.*")))
        val_organoids = sorted(list(data_folder.glob("Tellu/OrganoidDataset/val/images/*.*")))

        if split == 'train':
            self.images = train_organoids
        elif split == 'val':
            self.images = val_organoids
        else:
            raise ValueError(split)

    def __getitem__(self, i):
        im_path = self.images[i]
        im_ID = im_path.stem

        boxes = pd.read_csv(im_path.parent.parent / 'labels' / f'{im_path.stem}.txt', header=None, sep='\s+')
        boxes.columns = ["id", "x1", "y1", "w", "h"]
        boxes = [[b.y1, b.x1, b.h, b.w] for b in boxes.itertuples(index=False)]
        boxes = np.array(boxes).astype(float).reshape((-1, 4))
        
        im = skimage.io.imread(im_path, as_gray=False)
        H, W = im.shape[:2]

        boxes = cxcywh_to_xyxy(boxes) 
        boxes = boxes * np.array([[H, W, H, W]])

        mask = None

        return im, mask, boxes, im_path, im_ID
    
    def __str__(self):
        return 'Tellu'

    def __len__(self):
        return len(self.images)
    


class MultiOrg:
    def __init__(self, split=None, data_folder = base_datadir):
        self.split = split
        train_normal = sorted(list(data_folder.glob("MultiOrg/train/Normal/Plate_*/image_*/image_*.tiff")))
        train_macros = sorted(list(data_folder.glob("MultiOrg/train/Macros/Plate_*/image_*/image_*.tiff")))
        test_normal = sorted(list(data_folder.glob("MultiOrg/test/Normal/Plate_*/image_*/image_*.tiff")))
        test_macros = sorted(list(data_folder.glob("MultiOrg/test/Macros/Plate_*/image_*/image_*.tiff")))

        if split == 'train_normal':
            self.images = train_normal
        elif split == 'train_macros':
            self.images = train_macros
        elif split == 'test_normal':
            self.images = test_normal
        elif split == 'test_macros':
            self.images = test_macros
        else:
            raise ValueError(split)

    def __getitem__(self, i):
        im_path = self.images[i]
        im_ID = f'{im_path.parent.parent.name}_{im_path.parent.name}'
        
        # Image
        im = skimage.io.imread(im_path, as_gray=False)

        # Instance masks
        mask = None

        # Boxes
        label_path = list(im_path.parent.glob(f'{im_path.stem}_*.json'))[0]
        with open(str(label_path), 'r') as f:
            labels = json.load(f)
        boxes = [[box[0][0], box[0][1], box[2][0], box[2][1]] for k, box in labels.items()]
        boxes = np.array(boxes).astype(float).reshape((-1, 4))

        return im, mask, boxes, im_path, im_ID
    
    def __str__(self):
        return 'MultiOrg'
        
    def __len__(self):
        return len(self.images)

