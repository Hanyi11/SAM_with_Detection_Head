# import io
# import json
import os
import numpy as np
import cv2
# import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
# import matplotlib.patches as patches
# from copy import deepcopy
# import skimage
# from skimage.io import imread
# from sklearn import metrics
# import scipy
# import basicpy
# import tifffile
import torch
from tqdm import tqdm
# from descartes import PolygonPatch
# import geopandas as gpd
# import torchmetrics.detection


import sys
sys.path.append('/home/icb/lion.gleiter/projects/organoid_sam/SAM_with_Detection_Head')

# from typing import Literal
# import rasterio
# from rasterio.features import shapes as rio_shapes
# import shapely
# from cellpose import models as cp_models
# from util import dataloading as dl
# from util import postprocessing as pp
from util import box_ops_numpy as bxn


# import detection_head_model as dhm

# Import SAM1
sam1_path = '/home/icb/lion.gleiter/projects/organoid_sam/segment-anything/segment-anything'
if sam1_path not in sys.path:
    sys.path.append(sam1_path)
from segment_anything import build_sam_vit_l, predictor

# Import SAM2
sam2_path = '/home/icb/lion.gleiter/projects/organoid_sam/sam2'
if sam2_path not in sys.path:
    sys.path.append(sam2_path)
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor






# def show_points(coords, labels, ax, marker_size=375):
#     pos_points = coords[labels==1]
#     neg_points = coords[labels==0]
#     ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
#     ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax, color='red'):
    y_min, x_min, y_max, x_max = box
    
    # Calculate width and height of the box
    width = x_max - x_min
    height = y_max - y_min

    ax.add_patch(plt.Rectangle((x_min, y_min), width, height, edgecolor=color, facecolor=(0,0,0,0), lw=2))

# def show_mask(contour, ax, random_color=False):
#     if random_color:
#         color = np.random.random(3)
#     else:
#         color = np.array([30/255, 184/255, 255/255])
#     edgecolor = np.concatenate([color, np.array([1.0])], axis=0)
#     facecolor = np.concatenate([color, np.array([0.4])], axis=0)

#     contour = gpd.GeoSeries(contour)
#     contour.plot(edgecolor=edgecolor, facecolor=facecolor, ax=ax)

# def visualize(im, contours, boxes, random_color=True, box_color='red', format='yxyx_px', ax=None, **fig_kwargs):
#     # Format boxes
#     H, W = im.shape[:2]

#     if format=='cycxhw_01':
#         # boxes are in cycxhw format normalized by the longest image side. Converts to min/max coordinates in pixels.
#         original_boxes = cxcywh_to_xyxy(boxes) * max(H, W)
#     elif format=='cycxhw_px':
#         # boxes are in cycxhw format in pixels. Converts to min/max coordinates in pixels.
#         original_boxes = cxcywh_to_xyxy(boxes)
#     elif format=='yxyx_01':
#         # boxes are in min/max coordinates normalized by the longest image side. Converts to min/max coordinates in pixels.
#         original_boxes = boxes * max(H, W)
#     elif format=='yxyx_px':
#         # boxes are in min/max coordinates in pixels. No conversion necessary.
#         original_boxes = boxes
#     else:
#         raise ValueError(format)

#     if ax is None:
#         fig, ax = plt.subplots(1, 1, **fig_kwargs)
    
#     ax.imshow(im)
#     ax.axis(False)
#     for i in range(original_boxes.shape[0]):
#         show_mask(contours[i], ax, random_color=random_color)
#         show_box(original_boxes[i], ax, color=box_color)





# Image dataset for parallel data loading:
class Images(torch.utils.data.Dataset):
    def __init__(self, embed_name):
        super().__init__()
        self.base = Path('/ictstr01/groups/shared/users/lion.gleiter/organoid_sam/patched_data_multiscale_miccai/patch_images')
        self.img_list = sorted(list(self.base.glob('*/*/*/*.png')))  # [:150000]
        print('len(self.img_list)', len(self.img_list))
        self.embed_name = embed_name

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx) -> tuple[np.ndarray, np.ndarray, Path]:
        img_file = self.img_list[idx]
        img_name = img_file.parent.name
        ds_name = img_file.parent.parent.name
        data_split = img_file.parent.parent.parent.name
        boxes_file = self.base.parent / 'patch_bbox_gt' / data_split / ds_name / img_name / f'{img_file.stem}.npy'
        boxes = np.load(boxes_file, allow_pickle=False)
        embed_file = self.base.parent / 'patch_embed' / self.embed_name / data_split / ds_name / img_name / f'{img_file.stem}.pt'
        img = cv2.imread(str(img_file), cv2.IMREAD_COLOR)
        return img, boxes, embed_file


def predict_mask(sam_predictor, box, H, W, sam_version, device):
    """`box` is assumed to be in y, x, y, x format unnormalized in px of the whole image
    """
    # device = self.detection_head.device
    box = box.flatten()
    assert box.shape[0]==4, box.shape

    # Forward
    if sam_version=='sam1':
        # Normalize box to [0, 1]
        input_box = torch.from_numpy(box / max(H, W))
        # [y x y x] --> [x y x y] in [0, 1024]
        transformed_boxes = torch.tensor([[input_box[1], input_box[0], input_box[3], input_box[2]]], device=device) * 1024
        masks, _, _ = sam_predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=False,
        )
        masks = masks.squeeze(1).cpu().numpy()
        masks = masks.astype(np.uint8)

    elif sam_version=='sam2':
        # [y x y x] --> [x y x y] unnormalized
        transformed_boxes = np.array([[box[1], box[0], box[3], box[2]]])
        masks, scs, _ = sam_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=transformed_boxes,
            multimask_output=False,
        )
        masks = (masks > 0.5).astype(np.uint8)
    assert np.all(masks.shape[-2:] == np.array([H, W], dtype=int)), f'{masks.shape}, {H}, {W}'
    assert masks.shape[0] == 1, masks.shape
    mask = masks[0]
    return mask


def forward_one_patch(patch, sam_version, sam_predictor):
    with torch.inference_mode():
        sam_predictor.set_image(patch)
        if sam_version=='sam1':
            image_embedding = {
                'original_size': sam_predictor.original_size,
                'input_size': sam_predictor.input_size,
                'features': sam_predictor.features,
                'is_image_set': True,
            }
        elif sam_version=='sam2':
            image_embedding = {
                '_orig_hw': sam_predictor._orig_hw,  # Original size of the image
                '_is_batch': sam_predictor._is_batch,        # Flag indicating if batch or not
                '_features': sam_predictor._features,            # Features extracted from the image
                '_is_image_set': True,                      # Flag indicating the image has been set
                'mask_threshold': sam_predictor.mask_threshold
            }
        else:
            raise ValueError(sam_version)
    return image_embedding


def main(sam_version = 'sam1'):
    
    results_dir = Path('/ictstr01/groups/shared/users/lion.gleiter/organoid_sam/results')
    plot_dir = results_dir / 'test_embedding_computation'
    plot_dir.mkdir(exist_ok=True)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    if sam_version=='sam1':
        sam_model = build_sam_vit_l(checkpoint='/ictstr01/groups/shared/users/lion.gleiter/organoid_sam/checkpoints/sam_vit_l_0b3195.pth')
        sam_predictor = predictor.SamPredictor(sam_model=sam_model.to(device=device))
    elif sam_version=='sam2':
        sam2_checkpoint = "/ictstr01/groups/shared/users/lion.gleiter/organoid_sam/checkpoints/sam2.1_hiera_large.pt"
        model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
        sam2 = build_sam2(model_cfg, sam2_checkpoint, device=device)
        sam_predictor = SAM2ImagePredictor(sam2)
    else:
        raise ValueError(f'SAM version {sam_version} is currently not supported')
    
    
    if sam_version=='sam1':
        dataset = Images(embed_name='SAM_large')
    elif sam_version=='sam2':
        dataset = Images(embed_name='SAM2_large')
    else:
        raise ValueError(sam_version)

    dataloader = torch.utils.data.DataLoader(dataset, num_workers=int(os.environ.get('SLURM_CPUS_PER_TASK', 0)) - 1,
                                             batch_size=1, collate_fn=lambda x: x)

    for i, batch in tqdm(enumerate(dataloader)):
        img, boxes, embed_file = batch[0]
        if embed_file.exists():
            continue
        H, W = img.shape[:2]
        image_embedding = forward_one_patch(img, sam_version=sam_version, sam_predictor=sam_predictor)

        embed_file.parent.mkdir(exist_ok=True, parents=True)
        torch.save(image_embedding, embed_file)

        # if (i % 100 == 20) and (i < 2000) and (boxes.shape[0]>0):
        #     boxes = bxn.cxcywh_to_xyxy(boxes) * max(H, W)
        #     mask = predict_mask(sam_predictor, boxes[-1, :], H, W, sam_version=sam_version, device=device)
        #     print("H, W", H, W)
        #     print('box', boxes[-1, :])
        #     # Visualize predictions
        #     plt.figure()
        #     plt.imshow(img)
        #     plt.axis('off')
        #     plt.imshow(mask.astype(np.uint8) * 255, alpha=0.5)
        #     show_box(boxes[-1], plt.gca())
        #     plt.savefig(plot_dir / f'{sam_version}_example_{i}.png')
        #     plt.close('all')


if __name__=='__main__':
    main(sam_version='sam2')