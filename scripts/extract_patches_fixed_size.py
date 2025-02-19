#!/usr/bin/env python

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

import sys
sys.path.append('/home/icb/lion.gleiter/projects/organoid_sam/SAM_with_Detection_Head')

from util.box_ops_numpy import mask_to_boxes, cxcywh_to_xyxy, xyxy_to_cxcywh, plot_boxes
from util import dataloading as dl


patched_data_base = Path('/ictstr01/groups/shared/users/lion.gleiter/organoid_sam/patched_data_multiscale_miccai')
patched_data_base.mkdir(exist_ok=True)

img_folder_base = patched_data_base / 'patch_images'
box_folder_base = patched_data_base / 'patch_bbox_gt'
seg_folder_base = patched_data_base / 'patch_seg_gt'



if __name__=="__main__":

    # # Train and validation data
    # data_sets = [
    #     (dl.OrgaQuant(split='train'), 'train', True),
    #     (dl.OrgaSegment(split='train'), 'train', True),
    #     (dl.Tellu(split='train'), 'train', True),
    #     (dl.MultiOrg(split='train_normal'), 'train', False),
    #     (dl.MultiOrg(split='train_macros'), 'train', False),

    #     (dl.OrgaQuant(split='val'), 'val', True),
    #     (dl.OrgaSegment(split='val'), 'val', True),
    #     (dl.Tellu(split='val'), 'val', True),
    #     (dl.MultiOrg(split='val_normal'), 'val', False),
    #     (dl.MultiOrg(split='val_macros'), 'val', False),
        
    #     (dl.NeurIPSCellSeg(split='train'), 'train', False),
    #     (dl.NeurIPSCellSeg(split='val'), 'val', False),
    # ]
    # for ds, split, correct_bg in data_sets:
    #     subdir = f'{str(ds)}_{ds.split}'

    #     for im, mask, boxes, im_path, im_ID in ds:
    #         im, _ = dl.normalize(im, correct_bg=correct_bg)
    #         H, W = im.shape[:2]

    #         for patch_size in [512, 1024, 2048, 4096]:
    #             if (patch_size > 2* max(H, W)) and (patch_size > 512):
    #                 # Stop extracting patches, if patch_size is much larger than the image.
    #                 continue

    #             img_folder = img_folder_base / split / subdir / f'im_{im_ID}_{patch_size}'
    #             img_folder.mkdir(exist_ok=True, parents=True)

    #             box_folder = box_folder_base / split / subdir / f'im_{im_ID}_{patch_size}'
    #             box_folder.mkdir(exist_ok=True, parents=True)

    #             seg_folder = seg_folder_base / split / subdir / f'im_{im_ID}_{patch_size}'
    #             seg_folder.mkdir(exist_ok=True, parents=True)

    #             offsets = []
    #             box_counts = []
    #             for i, (im_crop, 
    #                     mask_crop, 
    #                     boxes_crop, 
    #                     overlap, 
    #                     offset_x, 
    #                     offset_y) in enumerate(dl.patch_fixed(im, mask, boxes, size=patch_size, 
    #                                                           overlap=1/2, min_box_overlap=0.99)):
                    
    #                 # Filter only sufficiently large boxes for all but the smallest patch_size
    #                 if patch_size > 512:
    #                     boxes_crop = dl.filter_boxes_by_side_length(im_crop, boxes_crop, min_ratio=0.02)
                    
    #                 num_boxes = boxes_crop.shape[0]
    #                 # # Filter background crops
    #                 # if dl.is_empty(im_crop) and (boxes_crop.shape[0]==0):
    #                 #     continue

    #                 cv2.imwrite(str(img_folder / f'patch_{i}.png'), im_crop)
    #                 if mask_crop is not None:
    #                     np.save(seg_folder / f'patch_{i}.npy', mask_crop, allow_pickle=False)

    #                 H_crop, W_crop = im_crop.shape[:2]
    #                 boxes_crop = xyxy_to_cxcywh(boxes_crop / max(H_crop, W_crop))
    #                 np.save(box_folder / f'patch_{i}.npy', boxes_crop, allow_pickle=False)
        
    #                 # Don't save overlap, as it is > 0.99 anyways.
    #                 # np.save(box_folder / f'patch_{i}_overlap.npy', overlap, allow_pickle=False)

    #                 offsets.append([i, offset_y, offset_x])
    #                 box_counts.append(num_boxes)

    #             offsets = np.array(offsets)
    #             np.save(box_folder / f'offsets.npy', offsets, allow_pickle=False)

    #             box_counts = np.array(box_counts)
    #             np.save(box_folder / f'box_counts.npy', box_counts, allow_pickle=False)

    # Test data
    # Don't use the patch size in the image name for test set images, since we 
    # need to combine predictions from all patch sizes.
    data_sets = [
        (dl.OrgaQuant(split='test'), 'test', True),
        (dl.OrgaSegment(split='test'), 'test', True),
        (dl.Tellu(split='test'), 'test', True),
        (dl.MultiOrg(split='test_normal'), 'test', False),
        (dl.MultiOrg(split='test_macros'), 'test', False),
        (dl.NewData(split='all'), 'test', True),
        (dl.OrgaExtractor(split='all'), 'test', True),
        (dl.OrganoID(split='test'), 'test', True),
        (dl.OrganoID(split='test_C'), 'test', True),
        (dl.OrganoID(split='test_ACC'), 'test', True),
        (dl.OrganoID(split='test_Lung'), 'test', True),
        (dl.OrganoID(split='test_only_mouse'), 'test', True),
    ]
    for ds, split, correct_bg in data_sets:
        subdir = f'{str(ds)}_{ds.split}'

        for im, mask, boxes, im_path, im_ID in ds:
            im, _ = dl.normalize(im, correct_bg=correct_bg)
            H, W = im.shape[:2]

            img_folder = img_folder_base / split / subdir / f'im_{im_ID}'
            img_folder.mkdir(exist_ok=True, parents=True)

            box_folder = box_folder_base / split / subdir / f'im_{im_ID}'
            box_folder.mkdir(exist_ok=True, parents=True)

            seg_folder = seg_folder_base / split / subdir / f'im_{im_ID}'
            seg_folder.mkdir(exist_ok=True, parents=True)

            offsets = []
            box_counts = []

            i = 0
            for patch_size in [512, 1024, 2048, 4096]:
                if (patch_size > 2* max(H, W)) and (patch_size > 512):
                    # Stop extracting patches, if patch_size is much larger than the image.
                    continue

                for _, (im_crop, 
                        mask_crop, 
                        boxes_crop, 
                        overlap, 
                        offset_x, 
                        offset_y) in enumerate(dl.patch_fixed(im, mask, boxes, size=patch_size, 
                                                              overlap=1/2, min_box_overlap=0.99)):
                    
                    # No filtering for test data
                    
                    num_boxes = boxes_crop.shape[0]

                    cv2.imwrite(str(img_folder / f'patch_{i}.png'), im_crop)
                    if mask_crop is not None:
                        np.save(seg_folder / f'patch_{i}.npy', mask_crop, allow_pickle=False)

                    H_crop, W_crop = im_crop.shape[:2]
                    boxes_crop = xyxy_to_cxcywh(boxes_crop / max(H_crop, W_crop))
                    np.save(box_folder / f'patch_{i}.npy', boxes_crop, allow_pickle=False)

                    offsets.append([i, offset_y, offset_x])
                    box_counts.append(num_boxes)

                    i += 1

            offsets = np.array(offsets)
            np.save(box_folder / f'offsets.npy', offsets, allow_pickle=False)

            box_counts = np.array(box_counts)
            np.save(box_folder / f'box_counts.npy', box_counts, allow_pickle=False)


