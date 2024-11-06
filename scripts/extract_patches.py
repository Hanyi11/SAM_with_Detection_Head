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


img_folder_base = Path('/ictstr01/groups/shared/users/lion.gleiter/organoid_sam/patch_images') 
box_folder_base = Path('/ictstr01/groups/shared/users/lion.gleiter/organoid_sam/patch_bbox_gt')
seg_folder_base = Path('/ictstr01/groups/shared/users/lion.gleiter/organoid_sam/patch_seg_gt')



if __name__=="__main__":
    # test_sets = [
    #     dl.OrganoID(split='test'),
    #     dl.OrganoID(split='test_Lung'),
    #     dl.OrganoID(split='test_ACC'),
    #     dl.OrganoID(split='test_C'),
    #     dl.OrganoID(split='test_mouse'),
    # ]
    # for ds in test_sets:
    #     subdir = f'{str(ds)}_{ds.split}'

    #     for im, mask, boxes, im_path, im_ID in ds:
    #         im, flatfield = dl.normalize(im, smoothness=10)


    #         img_folder = img_folder_base / 'test' / subdir / f'im_{im_ID}'
    #         img_folder.mkdir(exist_ok=True, parents=True)

    #         box_folder = box_folder_base / 'test' / subdir / f'im_{im_ID}'
    #         box_folder.mkdir(exist_ok=True, parents=True)

    #         seg_folder = seg_folder_base / 'test' / subdir / f'im_{im_ID}'
    #         seg_folder.mkdir(exist_ok=True, parents=True)

    #         # Choose 4 patches overlapping at least by 1/6 for all test data
    #         H, W = im.shape[:2]
    #         patch_size = int(np.ceil(max(H, W) / (2 - 1/6)))

    #         offsets = []
    #         for i, (im_crop, mask_crop, boxes_crop, offset_x, offset_y) in enumerate(dl.patch(im, mask, boxes, size=patch_size)):
    #             cv2.imwrite(str(img_folder / f'patch_{i}.png'), im_crop)
    #             if mask_crop is not None:
    #                 np.save(seg_folder / f'patch_{i}.npy', mask_crop, allow_pickle=False)

    #             H_crop, W_crop = im_crop.shape[:2]
    #             boxes_crop = xyxy_to_cxcywh(boxes_crop / max(H_crop, W_crop))
    #             np.save(box_folder / f'patch_{i}.npy', boxes_crop, allow_pickle=False)

    #             offsets.append([offset_y, offset_x])
    #         offsets = np.array(offsets)
    #         np.save(box_folder / f'offsets.npy', offsets, allow_pickle=False)
    #         np.save(box_folder / f'boxes_original.npy', np.array(boxes), allow_pickle=False)
    #         if mask is not None:
    #             np.save(seg_folder / f'mask_original.npy', mask, allow_pickle=False)



    # # Patch size per dataset
    # val_sets = [
    #     dl.OrgaQuant(split='test'),
    #     dl.OrgaExtractor(split='test'),
    #     dl.OrgaSegment(split='val'),
    #     # dl.NeurIPSCellSeg(split='val'),
    # ]
    # for ds in val_sets:
    #     subdir = f'{str(ds)}_{ds.split}'
    #     patch_size = dl.compute_patch_size(dl.compute_median_size(ds))

    #     for im, mask, boxes, im_path, im_ID in ds:
    #         im, flatfield = dl.normalize(im, smoothness=10)


    #         img_folder = img_folder_base / 'val' / subdir / f'im_{im_ID}'
    #         img_folder.mkdir(exist_ok=True, parents=True)

    #         box_folder = box_folder_base / 'val' / subdir / f'im_{im_ID}'
    #         box_folder.mkdir(exist_ok=True, parents=True)

    #         seg_folder = seg_folder_base / 'val' / subdir / f'im_{im_ID}'
    #         seg_folder.mkdir(exist_ok=True, parents=True)

    #         # Choose 4 patches overlapping at least by 1/6 for all test data
    #         H, W = im.shape[:2]

    #         offsets = []
    #         for i, (im_crop, mask_crop, boxes_crop, offset_x, offset_y) in enumerate(dl.patch(im, mask, boxes, size=patch_size)):
    #             cv2.imwrite(str(img_folder / f'patch_{i}.png'), im_crop)
    #             if mask_crop is not None:
    #                 np.save(seg_folder / f'patch_{i}.npy', mask_crop, allow_pickle=False)

    #             H_crop, W_crop = im_crop.shape[:2]
    #             boxes_crop = xyxy_to_cxcywh(boxes_crop / max(H_crop, W_crop))
    #             np.save(box_folder / f'patch_{i}.npy', boxes_crop, allow_pickle=False)

    #             offsets.append([offset_y, offset_x])
    #         offsets = np.array(offsets)
    #         np.save(box_folder / f'offsets.npy', offsets, allow_pickle=False)

    #         # np.save(box_folder / f'boxes_original.npy', np.array(boxes), allow_pickle=False)
    #         # if mask is not None:
    #         #     np.save(seg_folder / f'mask_original.npy', mask, allow_pickle=False)

            

    # # Patch size per dataset
    # train_sets = [
    #     dl.OrgaQuant(split='train'),
    #     dl.OrgaExtractor(split='train'),
    #     dl.OrgaSegment(split='train'),
    # ]
    # for ds in train_sets:
    #     subdir = f'{str(ds)}_{ds.split}'
    #     patch_size = dl.compute_patch_size(dl.compute_median_size(ds))

    #     for im, mask, boxes, im_path, im_ID in ds:
    #         im, flatfield = dl.normalize(im, smoothness=10)


    #         img_folder = img_folder_base / 'train' / subdir / f'im_{im_ID}'
    #         img_folder.mkdir(exist_ok=True, parents=True)

    #         box_folder = box_folder_base / 'train' / subdir / f'im_{im_ID}'
    #         box_folder.mkdir(exist_ok=True, parents=True)

    #         seg_folder = seg_folder_base / 'train' / subdir / f'im_{im_ID}'
    #         seg_folder.mkdir(exist_ok=True, parents=True)

    #         # Choose 4 patches overlapping at least by 1/6 for all test data
    #         H, W = im.shape[:2]

    #         offsets = []
    #         for i, (im_crop, mask_crop, boxes_crop, offset_x, offset_y) in enumerate(dl.patch(im, mask, boxes, size=patch_size)):
    #             cv2.imwrite(str(img_folder / f'patch_{i}.png'), im_crop)
    #             if mask_crop is not None:
    #                 np.save(seg_folder / f'patch_{i}.npy', mask_crop, allow_pickle=False)

    #             H_crop, W_crop = im_crop.shape[:2]
    #             boxes_crop = xyxy_to_cxcywh(boxes_crop / max(H_crop, W_crop))
    #             np.save(box_folder / f'patch_{i}.npy', boxes_crop, allow_pickle=False)

    #             offsets.append([offset_y, offset_x])
    #         offsets = np.array(offsets)
    #         np.save(box_folder / f'offsets.npy', offsets, allow_pickle=False)



    # Patch size per image
    val_sets = [
        # dl.OrganoID(split='val'),
        dl.NeurIPSCellSeg(split='val'),
        dl.Tellu(split='val'),
        dl.MultiOrg(split='test_normal'),
        dl.MultiOrg(split='test_macros'),
    ]
    for ds in val_sets:
        subdir = f'{str(ds)}_{ds.split}'

        for im, mask, boxes, im_path, im_ID in ds:
            im, flatfield = dl.normalize(im, smoothness=10)


            img_folder = img_folder_base / 'val' / subdir / f'im_{im_ID}'
            img_folder.mkdir(exist_ok=True, parents=True)

            box_folder = box_folder_base / 'val' / subdir / f'im_{im_ID}'
            box_folder.mkdir(exist_ok=True, parents=True)

            seg_folder = seg_folder_base / 'val' / subdir / f'im_{im_ID}'
            seg_folder.mkdir(exist_ok=True, parents=True)

            # Choose 4 patches overlapping at least by 1/6 for all test data
            H, W = im.shape[:2]
            patch_size = dl.compute_patch_size(dl.compute_median_size_image(boxes))

            offsets = []
            for i, (im_crop, mask_crop, boxes_crop, offset_x, offset_y) in enumerate(dl.patch(im, mask, boxes, size=patch_size)):
                cv2.imwrite(str(img_folder / f'patch_{i}.png'), im_crop)
                if mask_crop is not None:
                    np.save(seg_folder / f'patch_{i}.npy', mask_crop, allow_pickle=False)

                H_crop, W_crop = im_crop.shape[:2]
                boxes_crop = xyxy_to_cxcywh(boxes_crop / max(H_crop, W_crop))
                np.save(box_folder / f'patch_{i}.npy', boxes_crop, allow_pickle=False)

                offsets.append([offset_y, offset_x])
            offsets = np.array(offsets)
            np.save(box_folder / f'offsets.npy', offsets, allow_pickle=False)



    # Patch size per image
    train_sets = [
        dl.NeurIPSCellSeg(split='train'),
        dl.OrganoID(split='train'),
        dl.NewData(split='train'),
        dl.Tellu(split='train'),
        dl.MultiOrg(split='train_normal'),
        dl.MultiOrg(split='train_macros'),
    ]
    for ds in train_sets:
        subdir = f'{str(ds)}_{ds.split}'

        for im, mask, boxes, im_path, im_ID in ds:
            im, flatfield = dl.normalize(im, smoothness=10)


            img_folder = img_folder_base / 'train' / subdir / f'im_{im_ID}'
            img_folder.mkdir(exist_ok=True, parents=True)

            box_folder = box_folder_base / 'train' / subdir / f'im_{im_ID}'
            box_folder.mkdir(exist_ok=True, parents=True)

            seg_folder = seg_folder_base / 'train' / subdir / f'im_{im_ID}'
            seg_folder.mkdir(exist_ok=True, parents=True)

            # Choose 4 patches overlapping at least by 1/6 for all test data
            H, W = im.shape[:2]
            patch_size = dl.compute_patch_size(dl.compute_median_size_image(boxes))

            offsets = []
            for i, (im_crop, mask_crop, boxes_crop, offset_x, offset_y) in enumerate(dl.patch(im, mask, boxes, size=patch_size)):
                cv2.imwrite(str(img_folder / f'patch_{i}.png'), im_crop)
                if mask_crop is not None:
                    np.save(seg_folder / f'patch_{i}.npy', mask_crop, allow_pickle=False)

                H_crop, W_crop = im_crop.shape[:2]
                boxes_crop = xyxy_to_cxcywh(boxes_crop / max(H_crop, W_crop))
                np.save(box_folder / f'patch_{i}.npy', boxes_crop, allow_pickle=False)

                offsets.append([offset_y, offset_x])
            offsets = np.array(offsets)
            np.save(box_folder / f'offsets.npy', offsets, allow_pickle=False)
