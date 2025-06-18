import os
import sys
import PIL
import PIL.Image
import cv2
import numpy as np
import torchvision
from tqdm import tqdm
from typing import Literal, List
from copy import deepcopy
from pathlib import Path
from PIL import Image
from omegaconf import ListConfig, OmegaConf


import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler, SubsetRandomSampler, WeightedRandomSampler
from torchvision.models.detection.faster_rcnn import FasterRCNN_ResNet50_FPN_V2_Weights, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.ssd import SSD300_VGG16_Weights
from monai import transforms as tfs

from util import box_ops_numpy

path_detr = '/home/icb/lion.gleiter/projects/organoid_sam/detr'
if path_detr not in sys.path:
    sys.path.append(path_detr)
from detr.datasets import coco


def compute_weights_working(metadata: pd.DataFrame, groups, max_oversampling = 10.0, max_ratio_0_objects=0.05):
    """
    Compute per-sample weights for a dataset, balancing multiple subgroups of data.

    The subgroups are determined as unique value combinations of the variables in 'groups', e.g.
    if groups = ['dataset', 'patch_size'], then data from NeurIPS_train with patchsize 2048 would 
    be sampled equally often as data from OrgaSegment_train with patchsize 512.

    Args:
        metadata (pd.DataFrame): 
            Contains the following columns, and each row is one available datapoint: ('dataset', 'n_objects', 'n_objects_grouped', 'patch_size')
        groups (list[str] | None): 
            Keys for determining unique groups for weighting. 'None' means equal weight for all datapoints.
        max_oversampling (float):
            Specifies how much any sample weight may be maximally increased compared to the default weight.

    Returns:
        np.ndarray: 
            A NumPy array of weights for each sample in the dataset. The weights are normalized within each subset.
    """
    print("start compute weights")
    total_length = len(metadata)
    max_weight = max_oversampling / total_length

    # Initialize with equal weight for all points.
    w = np.ones(len(metadata), dtype=float) / total_length

    if groups is None:
        return w

    # Each group gets the same total weight and distributes it equally between its members.
    grouped_metadata = metadata.groupby(groups, as_index=False)
    n_groups = len(grouped_metadata)
    total_weight_per_group = 1.0 / n_groups

    for name, df in grouped_metadata:
        weight = total_weight_per_group / len(df)
        ids = df[['idx']].values.tolist()
        w[ids] = weight
        print(name, 'weight', weight)

    # Enforce max weight:
    w = np.minimum(w, max_weight)

    total_weight = np.sum(w)
    zero_objects = (metadata[['n_objects']].values == 0).flatten()
    weight_zero_objects = np.sum(w[zero_objects])
    if weight_zero_objects / total_weight > max_ratio_0_objects:
        w[zero_objects] = w[zero_objects] * (max_ratio_0_objects / (1-max_ratio_0_objects)) * (total_weight - weight_zero_objects) / weight_zero_objects

    # Check that calculations were correct
    total_weight = np.sum(w)
    weight_zero_objects = np.sum(w[zero_objects])
    assert weight_zero_objects / total_weight <= max_ratio_0_objects + 1e-3, weight_zero_objects / total_weight

    return w



def compute_weights(metadata: pd.DataFrame, groups, max_oversampling = 10.0, max_ratio_0_objects=0.05, weight_open_images=0.5):
    """
    Compute per-sample weights for a dataset, balancing multiple subgroups of data.

    The subgroups are determined as unique value combinations of the variables in 'groups', e.g.
    if groups = ['dataset', 'patch_size'], then data from NeurIPS_train with patchsize 2048 would 
    be sampled equally often as data from OrgaSegment_train with patchsize 512.

    Args:
        metadata (pd.DataFrame): 
            Contains the following columns, and each row is one available datapoint: ('dataset', 'n_objects', 'n_objects_grouped', 'patch_size')
        groups (list[str] | None): 
            Keys for determining unique groups for weighting. 'None' means equal weight for all datapoints.
        max_oversampling (float):
            Specifies how much any sample weight may be maximally increased compared to the default weight.

    Returns:
        np.ndarray: 
            A NumPy array of weights for each sample in the dataset. The weights are normalized within each subset.
    """
    print("start compute weights")
    total_length = len(metadata)
    max_weight = max_oversampling / total_length

    # Initialize with equal weight for all points.
    w = np.ones(len(metadata), dtype=float) / total_length

    # Assign weight Open Images
    total_available_weight = 1.0
    if np.any(metadata[['dataset']] == 'open_images'):
        print(f"\n\nFound Open Images samples in the data, assigning weight {weight_open_images}\n\n")
        is_oi = (metadata[['dataset']]=='open_images').values.flatten()
        is_not_oi = (metadata[['dataset']]!='open_images').values.flatten()
        oi_ids = metadata.loc[is_oi, ['idx']].values.tolist()
        other_ids = metadata.loc[is_not_oi, ['idx']].values.tolist()
        w[oi_ids] = weight_open_images / len(oi_ids)
        w[other_ids] = (1 - weight_open_images) / len(other_ids) if len(other_ids) > 0 else 1 
        total_available_weight = 1 - weight_open_images

    if groups is None:
        return w
    
    if np.all(metadata[['dataset']] == 'open_images'):
        return w

    # Each group outside Open Images gets the same total weight and distributes it equally between its members.
    is_not_oi = (metadata[['dataset']]!='open_images').values.flatten()
    metadata_wo_oi = metadata.loc[is_not_oi, :]
    grouped_metadata = metadata_wo_oi.groupby(groups, as_index=False)
    n_groups = len(grouped_metadata)
    total_weight_per_group = total_available_weight / n_groups

    for name, df in grouped_metadata:
        weight = total_weight_per_group / len(df)
        ids = df[['idx']].values.tolist()
        w[ids] = weight
        print(name, 'weight', weight)

    # Enforce max weight:
    w = np.minimum(w, max_weight)

    total_weight = np.sum(w)
    zero_objects = (metadata[['n_objects']].values == 0).flatten()
    weight_zero_objects = np.sum(w[zero_objects])
    if weight_zero_objects / total_weight > max_ratio_0_objects:
        w[zero_objects] = w[zero_objects] * (max_ratio_0_objects / (1-max_ratio_0_objects)) * (total_weight - weight_zero_objects) / weight_zero_objects

    # Check that calculations were correct
    total_weight = np.sum(w)
    weight_zero_objects = np.sum(w[zero_objects])
    assert weight_zero_objects / total_weight <= max_ratio_0_objects + 1e-3, weight_zero_objects / total_weight

    return w



def collate_fn(batch):
    """Take a batch of data samples and seperate images and targets into two seperate lists."""
    images = [b[0] for b in batch]
    targets = [b[1] for b in batch]
    img_ids = [b[2] for b in batch]
    offsets = [b[3] for b in batch]
    patch_sizes = [b[4] for b in batch]
    patch_numbers = [b[5] for b in batch]
    return images, targets, img_ids, offsets, patch_sizes, patch_numbers


# TODO: add backbone-specific padding to avoid padding artifacts during training. (always pad only the bottom and right, to avoid issues with bboxes.)

class RandRotate90(tfs.RandomizableTransform):
    def __init__(self, prob = 1, do_transform = True, axes=(-2, -1)):
        super().__init__(prob, do_transform)
        self.axes = axes

    def randomize(self):
        super().randomize(None)
        self.k = self.R.choice(a=4)

    def __call__(self, data):
        self.randomize()

        img, boxes = data['img'], data['boxes']
        H = img.shape[self.axes[0]]
        W = img.shape[self.axes[1]]

        if self._do_transform:
            img_rotate = tfs.Rotate90(k=self.k, spatial_axes=self.axes)
            img = img_rotate(img)

            # img = tfs.spatial.functional.rotate90(img, axes=self.axes, k=self.k, lazy=False, transform_info=None)
            
            boxes = self._rotate90k(boxes, H, W, k=self.k)
        return {'img': img, 'boxes': boxes}
    
    def _rotate90k(self, boxes, H, W, k):
        """Rotates k-times 90 degrees counterclockwise."""
        for _ in range(k):
            boxes = self._rotate90(boxes, H, W)

            # Swap H and W
            W_new, H_new = H, W
            H, W = H_new, W_new 

        return boxes
    
    def _rotate90(self, boxes, H, W):
        """Rotates 90 degrees counterclockwise.
        
        'boxes' must be of shape [N, 4] with coords in [x, y, x, y] in px, 
        where x corresponds to axis[1] and y to axis[0].
        """
        xmin = boxes[:, 1]
        xmax = boxes[:, 3]

        ymin = W - boxes[:, 2]
        ymax = W - boxes[:, 0]

        return np.stack((xmin, ymin, xmax, ymax), axis=1)
    
    # def _rotate90reverse(self, boxes, H, W):
    #     """Rotates 90 degrees clockwise.
        
    #     'boxes' must be of shape [N, 4] with coords in [x, y, x, y] in px, 
    #     where x corresponds to axis[1] and y to axis[0].
    #     """
    #     xmin = H - boxes[:, 3]
    #     xmax = H - boxes[:, 1]

    #     ymin = boxes[:, 0]
    #     ymax = boxes[:, 2]

    #     return np.stack((xmin, ymin, xmax, ymax), axis=1)


class RandFlip(tfs.RandomizableTransform):
    def __init__(self, prob = 1, do_transform = True, x_axis=-1):
        super().__init__(prob, do_transform)
        self.x_axis = x_axis
        self.img_flip = tfs.Flip(spatial_axis=self.x_axis, )

    def randomize(self):
        super().randomize(None)
        self.flip = self.R.choice(a=[True, False])

    def __call__(self, data):
        self.randomize()

        img, boxes = data['img'], data['boxes']
        W = img.shape[self.x_axis]

        if self._do_transform:
            if self.flip:
                img = self.img_flip(img)
                # img = tfs.spatial.functional.flip(img, sp_axes=self.x_axis, lazy=False, transform_info=None) #.as_tensor()
                boxes = self._flip(boxes, W)
        return {'img': img, 'boxes': boxes}
    
    def _flip(self, boxes, W):
        """Flips boxes along x axis.
        
        'boxes' must be of shape [N, 4] with coords in [x, y, x, y] in px.
        """
        xmin = W - boxes[:, 2]
        xmax = W - boxes[:, 0]

        ymin = boxes[:, 1]
        ymax = boxes[:, 3]

        return np.stack((xmin, ymin, xmax, ymax), axis=1)


class RandInvertIntensity(tfs.RandomizableTransform):
    def __init__(self, prob = 0.5, do_transform = True):
        super().__init__(prob, do_transform)

    def randomize(self):
        super().randomize(None)
        self.invert = self.R.choice(a=[True, False])

    def __call__(self, img):
        self.randomize()

        if self.invert:
            img = 1 - img

        return img


class Augmentation():
    def __init__(self):
        self.image_augmentation = tfs.OneOf(
            (
                tfs.Identity(),
                tfs.Compose((
                    tfs.RandGaussianSmooth(prob=0.45, sigma_x=(0.25, 1.5), sigma_y=(0.25, 1.5)),
                    tfs.RandGaussianSmooth(prob=0.05, sigma_x=(1.5, 5), sigma_y=(1.5, 5)),
                    tfs.RandAdjustContrast(prob=0.05, gamma=(0.5, 4.5)),
                    tfs.RandAdjustContrast(prob=0.05, gamma=(0.2, 8)),
                    RandInvertIntensity(prob=0.1),
                    tfs.RandGaussianNoise(prob=0.05, std=0.1),
                    tfs.RandGaussianNoise(prob=0.05, std=0.3),
                ))
            ),
            weights=(0.5, 0.5)
        )
        self.img_label_augmentation = tfs.Compose((
            RandRotate90(),
            RandFlip(),
        ))

    def __call__(self, img: torch.Tensor, boxes: torch.Tensor):
        # boxes are [x, y, x, y] in px
        img = self.image_augmentation(img)
        output = self.img_label_augmentation({'img': img, 'boxes': boxes})
        img, boxes = output['img'], output['boxes']

        return img, boxes


class Normalize():
    def __call__(self, image: PIL.Image):
        arr = torchvision.transforms.functional.pil_to_tensor(image)

        arr = arr / arr.max()

        return arr


def prepare_coco_targets(image: PIL.Image, target):
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]

        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        target["image_id"] = image_id
        if keypoints is not None:
            target["keypoints"] = keypoints

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])

        return image, target




class CachedDataset(Dataset):
    def __init__(self, 
                 dataset):
        super().__init__()
        self.dataset = dataset
        self.cache = []
        self.n_samples = 10
        self.metadata = self.dataset.metadata

        for i in range(self.n_samples):
            self.cache.append(self.dataset[i])

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        idx = idx % self.n_samples

        return self.cache[idx]






class ImageDataset(Dataset):
    def __init__(self, 
                 data_split: Literal["train", "test", "val"], 
                 data_split_dirs: List[str],
                 backbone_name: Literal["FRCNN", "FRCNNv2", "SSD", "DETR", "DETR_own_implementation",
                                        "resnet50", "SAM_base_images", "SAM_large_images"], 
                 decoder = None,
                 base_dir: str = "/ictstr01/groups/shared/users/lion.gleiter/organoid_sam/patched_data_multiscale_miccai",
                 augmentation: bool = True,
                 min_overlap: float = 0.99, # 0.9,
                 min_box_side: float = 0.0,
                 **kwargs):
        super().__init__()
        self.base_dir = Path(base_dir)
        self.metadata = []  # idx, dataset, n_objects, n_objects_grouped, patch_size
        
        # Dataset parameters
        self.data_split = data_split
        self.data_dirs = data_split_dirs
        self.use_coco_format = True if backbone_name=='DETR' else False
        
        # Pretrained transformer model parameters
        self.backbone = backbone_name
        self.decoder = decoder
        
        # Set parameters for filtering out objects that are too small or that do not have enough overlap with the image.
        self.min_overlap = min_overlap
        self.min_box_side = min_box_side

        # Set model for detection
        self.transforms = None
        if self.backbone == 'FRCNN':
            self.transforms = FasterRCNN_ResNet50_FPN_Weights.COCO_V1.transforms()
        elif self.backbone == 'FRCNNv2':
            self.transforms = FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1.transforms()
        elif self.backbone == 'SSD':
            self.transforms = SSD300_VGG16_Weights.COCO_V1.transforms()
        elif self.backbone == 'DETR':
            self.prepare = coco.ConvertCocoPolysToMask(False)
            self.transforms = coco.make_coco_transforms('train' if self.data_split=='train' else 'val')
        elif self.backbone == 'DETR_own_implementation':
            self.transforms = FasterRCNN_ResNet50_FPN_Weights.COCO_V1.transforms()
        elif self.backbone in ["resnet50", "SAM_base_images", "SAM_large_images"]:
            self.transforms = Normalize()
        else: 
            raise ValueError(f"backbone {self.backbone} is not supported.")

        # Set if data augmentations are activated
        if augmentation:
            self.augmentation = Augmentation()
        else:
            self.augmentation = None

        # Gets sorted paths of image embeddings
        open_image_dirs = [d for d in data_split_dirs if d.startswith('open_images_v4_5/original_data')]
        organoid_dirs = [d for d in data_split_dirs if not d.startswith('open_images_v4_5/original_data')]
        assert len(open_image_dirs) <= 1, open_image_dirs  # Should just be ['open_images_v4_5/original_data']

        # Get paths to precalculated embeddings for image crops
        embed_dir = self.base_dir / 'patch_images' / self.data_split
        embed_files = []
        for dataset in organoid_dirs:
            embed_files += list(embed_dir.glob(f'{dataset}/*/*.png'))

        # Sort paths to embeddings (for filtering later on)
        def sorting_key(file: Path):
            im_dir = file.parent.as_posix()
            patch_number = int(file.stem.replace('patch_', ''))
            return (im_dir, patch_number)
        self.image_files: list[Path] = sorted(embed_files, key=sorting_key)

        # Get paths to ground truth bboxes
        bbox_gt_dir = self.base_dir / 'patch_bbox_gt' / self.data_split
        bbox_gt_files = []
        for dataset in organoid_dirs:
            # Files named `patch_28.npy` but not `offsets.npy` or `patch_28_overlap.npy`
            bbox_gt_files += list(bbox_gt_dir.glob(f'{dataset}/*/patch_*[0-9].npy'))

        # Sort paths to bbox
        def sorting_key_bbox(file: Path):
            im_dir = file.parent.as_posix()
            patch_number = int(file.stem.replace('patch_', ''))
            return (im_dir, patch_number)
        self.label_files: list[Path] = sorted(bbox_gt_files, key=sorting_key_bbox)


        # Checks that the image files and label files match.
        assert len(self.image_files) == len(self.label_files), f"Mismatch between number of embeddings ({len(self.image_files)}) and labels ({len(self.label_files)})"
        for img_file, label_file in zip(self.image_files, self.label_files):
            # Get path infos
            img_path_cleaned = img_file.parent.as_posix().replace(f'patch_images', '')
            label_path_cleaned = label_file.parent.as_posix().replace(f'patch_bbox_gt', '')

            img_path_cleaned = os.path.join(img_path_cleaned, img_file.stem)
            label_path_cleaned = os.path.join(label_path_cleaned, label_file.stem)

            # Compare
            assert img_path_cleaned == label_path_cleaned, f"Filename mismatch: {img_file} and {label_file} are not the right img_patch and gt_label pair!"



        # Add metadata for non open-images data:
        current_count_file = None
        current_count_data = None
        for i, img_file in tqdm(enumerate(self.image_files)):
            dataset = img_file.parent.parent.name
            patch_number = int(img_file.stem.replace('patch_', ''))
            if self.data_split != 'test':
                patch_size = int(img_file.parent.name.split('_')[-1])
            else:
                patch_size = None
            if (current_count_data is None) or (current_count_file is None):
                current_count_file = bbox_gt_dir / dataset / img_file.parent.name / 'box_counts.npy'
                current_count_data = np.load(current_count_file, allow_pickle=False)
            elif current_count_file != bbox_gt_dir / dataset / img_file.parent.name / 'box_counts.npy':
                current_count_file = bbox_gt_dir / dataset / img_file.parent.name / 'box_counts.npy'
                current_count_data = np.load(current_count_file, allow_pickle=False)
            object_count = current_count_data[patch_number]
            object_count_category = int(np.ceil(object_count / 10.0 - 0.05)) * 10  # 0 for 0 objects, 10 for 1-10 objects, 20 for 11-20 objects, ...

            self.metadata.append([i, dataset, object_count, object_count_category, patch_size])


        # Appends Open Images file paths to self.image_files but not to self.label_files
        if self.data_split == 'train':
            oi_split = 'train'
        elif self.data_split == 'val':
            oi_split = 'validation'
        elif self.data_split == 'test':
            oi_split = 'test'
        else:
            raise ValueError(self.data_split)
        for oi_path in open_image_dirs:
            path = Path('/ictstr01/groups/shared/users/lion.gleiter') / oi_path / oi_split
            self.image_files += sorted(list(path.glob('*.jpg')))

        self.oi_annotation = pd.read_csv(f'/ictstr01/groups/shared/users/lion.gleiter/open_images_v4_5/original_data/{oi_split}-annotations-bbox.csv')

        # Update metadata.
        # self.is_empty = np.zeros(len(self.image_files), dtype=bool)
        for i in range(len(self.metadata), len(self.image_files)):
            self.metadata.append([i, 'open_images', 1000, 1000, 1000]) # dummy values to create a separate group.

        self.metadata = pd.DataFrame(data=self.metadata, columns=['idx', 'dataset', 'n_objects', 'n_objects_grouped', 'patch_size'])

    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Load image
        image_path = self.image_files[idx]
        # if self.backbone == 'DETR':
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        H, W = image.shape[:2]
        # else:
        #     image = Image.open(image_path).convert('RGB')
        #     H, W = image.size
        #     # image = self.transforms(image)
            
            
        # --- Load targets (BBs) and transform to right format
        # If it's an open images path
        if idx >= len(self.label_files):
            assert 'open_images' in str(image_path), image_path
            bboxes = self.oi_annotation[self.oi_annotation['ImageID']==image_path.stem]
            # img_original = cv2.imread(str(Path('/ictstr01/groups/shared/users/lion.gleiter/open_images_v4_5/original_data/') / image_path.parent.name / f'{image_path.stem}.jpg'))
            # H, W = img_original.shape[:2]
            targets = [[(b.YMin + b.YMax) / 2, (b.XMin + b.XMax) / 2, b.YMax - b.YMin, b.XMax - b.XMin] for b in bboxes.itertuples(index=False)]
            targets = np.array(targets).reshape((-1, 4))
            targets = targets * np.array([[H, W, H, W]]) / max(H, W)
            offsets = None
            patch_shape = None
            img_id = None
            patch_number = None
        # If it's not an open images path
        else:
            assert 'open_images' not in str(image_path), image_path
            label_path = self.label_files[idx]
            targets = np.load(label_path)

            # Filter boxes with low overlap with the current patch
            overlap_path = label_path.with_stem(f'{label_path.stem}_overlap')
            if overlap_path.exists():
                overlap = np.load(overlap_path)
                targets = targets[overlap > self.min_overlap]
                
            patch_number = int(label_path.stem.replace('patch_', ''))
            offsets_path = label_path.parent / 'offsets.npy'
            offsets = np.load(offsets_path, allow_pickle=False)[patch_number]
            patch_shape = (H, W)
            img_id = image_path.parent.name

        # Filter boxes that are too small
        targets = targets[
            (targets[:, 2] > self.min_box_side) & 
            (targets[:, 3] > self.min_box_side)
        ]
        
        # Convert BB format from cycxhw to xyxy
        targets = box_ops_numpy.cxcywh_to_xyxy(targets) * max(H, W)
        targets = np.stack((targets[:, 1], targets[:, 0], targets[:, 3], targets[:, 2]), axis=1)

        # print('targets after conversion to xyxy', targets)

        # Rescale image and boxes to max side length 1024
        max_side_length = 1024
        factor = float(max_side_length) / max(H, W) 
        image = cv2.resize(image, None, fx=factor, fy=factor, interpolation=cv2.INTER_LINEAR)  # resize to 1024
        image = image[:min(image.shape[0], max_side_length), 
                      :min(image.shape[1], max_side_length)]  # ensure maximum 1024
        out_H, out_W = image.shape[:2]
        targets *= np.array([[float(out_W) / W, 
                              float(out_H) / H, 
                              float(out_W) / W, 
                              float(out_H) / H]])  # Approx * factor

        image = PIL.Image.fromarray(image)

        # DETR
        if self.backbone == 'DETR':
            targets = np.concatenate([targets[:, :2], targets[:, 2:] - targets[:, :2]], axis=1)
            annotations = {
                'image_id': idx,
                'annotations': [
                    {
                        'bbox': box,
                        'is_crowd': False,
                        'area': box[2] * box[3],
                        'category_id': 1,  # Label
                    } for box in targets
                ]
            }
            targets = self.prepare(image, annotations)
            image, targets = self.transforms(image, targets)
            return image, targets, img_id, offsets, patch_shape, patch_number
        

        image = self.transforms(image)

        # Augmentation (targets are [x, y, x, y] in px)
        if self.augmentation is not None:
            image, targets = self.augmentation(image, targets)

        targets = torch.tensor(targets, dtype=torch.float32)

        # Pad with zeros to 1024, 1024 for DETR implementations and set labels to 0.
        if (self.decoder == "DETR_own_image_based") or (self.decoder.startswith("AnchorDETR")):
            image = torchvision.transforms.functional.pad(
                image, padding=(0, 0, 1024 - image.shape[-1], 1024 - image.shape[-2]),
                fill=0, padding_mode='constant'
            )
            assert image.shape[-1]==1024, image.shape
            assert image.shape[-2]==1024, image.shape

            labels = torch.zeros((targets.shape[0],), dtype=torch.int64)
        else:
            labels = torch.ones((targets.shape[0],), dtype=torch.int64)
        image_id = torch.tensor([idx])
        area = (targets[:, 3] - targets[:, 1]) * (targets[:, 2] - targets[:, 0])
        is_crowd = torch.zeros((targets.shape[0],), dtype=torch.bool)
        orig_size = torch.as_tensor([int(H), int(W)])
        size = torch.as_tensor([int(H), int(W)])

        targets_out = {'boxes': targets, 
                       'labels': labels, 
                       'image_id': image_id, 
                       'image_path': image_path,
                       'area': area, 
                       'is_crowd': is_crowd, 
                       'orig_size': orig_size, 
                       'size': size}        
        
        return image, targets_out, img_id, offsets, patch_shape, patch_number


class ImageDataModule(pl.LightningDataModule):
    def __init__(self, 
                 train_dirs: List[str],
                 val_dirs: List[str],
                 test_dirs: List[str],
                 batch_size: int, 
                 batches_per_epoch: int,
                 use_sampler: bool,
                 sampling_groups,
                 max_oversampling: float,
                 n_validation_samples: int | None,
                 cache_10_samples: bool = False,
                 **kwargs):
        super().__init__()
        self.n_workers = int(os.environ.get('SLURM_CPUS_PER_TASK', 0)) - 1
        self.kwargs = kwargs
        self.cache_10_samples = cache_10_samples
        
        # Data directories 
        self.train_dir_names = train_dirs
        self.val_dir_names = val_dirs
        self.test_dir_names = test_dirs

        # Set training parameters
        self.batch_size = batch_size
        self.batches_per_epoch = batches_per_epoch

        # Set data loading parameters
        self.use_sampler = use_sampler
        if isinstance(sampling_groups, ListConfig):
            self.sampling_groups = OmegaConf.to_object(sampling_groups)
        else:
            self.sampling_groups = sampling_groups
        self.max_oversampling = max_oversampling
        self.n_validation_samples = n_validation_samples
        
        
    def setup(self, stage: Literal["fit", "validate", None] = None):
        # Set datasets depending on the stage
        if stage == 'fit' or stage is None:
            if self.cache_10_samples:
                self.train_dataset = CachedDataset(ImageDataset(data_split_dirs=self.train_dir_names, 
                                                data_split="train", 
                                                augmentation=True,
                                                **self.kwargs))
            else:
                self.train_dataset = ImageDataset(data_split_dirs=self.train_dir_names, 
                                                data_split="train", 
                                                augmentation=True,
                                                **self.kwargs)
            print('len(train_dataset)', len(self.train_dataset))
        
        if (stage == 'fit') or (stage == 'validate') or (stage is None):
            if self.cache_10_samples:
                self.val_datasets = [CachedDataset(ImageDataset(
                    data_split_dirs=[val_dir_name],
                    data_split="val", 
                    augmentation=False,
                    **self.kwargs
                )) for val_dir_name in self.val_dir_names]
            else:
                self.val_datasets = [ImageDataset(
                    data_split_dirs=[val_dir_name],
                    data_split="val", 
                    augmentation=False,
                    **self.kwargs
                ) for val_dir_name in self.val_dir_names]
            print('len(val_datasets)', [len(ds) for ds in self.val_datasets])



    def train_dataloader(self):
        """
        This method sets up a DataLoader for the training dataset, either with a sampler or 
        with shuffling, depending on the configuration. It supports dataset balancing 
        by computing weights for the samples and optionally balances new data with a 
        specified ratio.
        """

        # assert not (self.balance_datasets and (self.balance_newdata > -0.5))
        if self.use_sampler:
            w = compute_weights(self.train_dataset.metadata, 
                                self.sampling_groups,
                                max_oversampling=self.max_oversampling)

            sampler = WeightedRandomSampler(
                weights=w,
                replacement=True,
                num_samples=self.batches_per_epoch * self.batch_size
            )

            return DataLoader(self.train_dataset, 
                              batch_size=self.batch_size,
                              sampler=sampler, 
                              num_workers=self.n_workers,
                              collate_fn=collate_fn)
        else:
            return DataLoader(self.train_dataset, 
                              batch_size=self.batch_size, 
                              shuffle=True, 
                              num_workers=self.n_workers,
                              collate_fn=collate_fn)

    
    def val_dataloader(self):
        """
        Creates and returns a DataLoader for the validation dataset.
        Depending on the configuration, it may use a sampler for balanced sampling or 
        load the data in a standard sequential manner.
        """
        if (self.n_validation_samples is not None) and (self.n_validation_samples > 0):
            dataloaders = []
            for val_dataset in self.val_datasets:
                sampler = RandomSampler(
                    val_dataset,
                    replacement=(len(val_dataset) < self.n_validation_samples),
                    num_samples=self.n_validation_samples
                )
                dataloaders.append(DataLoader(val_dataset, 
                                              batch_size=self.batch_size, 
                                              sampler=sampler, 
                                              num_workers=self.n_workers,
                                              collate_fn=collate_fn))
            return dataloaders
        else: 
            return [DataLoader(val_dataset, 
                               batch_size=self.batch_size, 
                               num_workers=self.n_workers,
                               collate_fn=collate_fn) for val_dataset in self.val_datasets]


    def get_test_dataloaders(self):
        """
        Creates and returns a DataLoader for the validation dataset.
        Depending on the configuration, it may use a sampler for balanced sampling or 
        load the data in a standard sequential manner.
        """
        print('self.test_dir_names', self.test_dir_names)
        if self.cache_10_samples:
            test_datasets = [CachedDataset(ImageDataset(
                data_split_dirs=[test_dir_name],
                data_split="test", 
                augmentation=False,
                **self.kwargs
            )) for test_dir_name in self.test_dir_names]
        else:
            test_datasets = [ImageDataset(
                data_split_dirs=[test_dir_name],
                data_split="test", 
                augmentation=False,
                **self.kwargs
            ) for test_dir_name in self.test_dir_names]
        print('test_datasets', test_datasets)

        return [DataLoader(test_dataset, 
                           batch_size=1, 
                           num_workers=self.n_workers,
                           collate_fn=collate_fn) for test_dataset in test_datasets]

