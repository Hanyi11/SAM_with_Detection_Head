from pathlib import Path
import sys
from typing import Literal
import cv2
import numpy as np
import skimage
import torch
import rasterio
from rasterio.features import shapes as rio_shapes
import shapely
import json
from . import dataloading as dl
from . import postprocessing as pp
from . import box_ops_numpy as bxn


# Load SAM and detection head
class PredictorFromMasks():
    def __init__(self, model: Literal['OrgaSegment', 'OrganoID', 'AnyStar'], scales = (10,)):
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.base_dir = Path('/ictstr01/groups/shared/users/lion.gleiter/organoid_sam/orgasegment_test')

        self.model = model
        if self.model == 'OrgaSegment':
            self.mask_dir = self.base_dir / 'OrganoidBasic20211215'
        elif self.model == 'OrganoID':
            self.mask_dir = self.base_dir / 'OrganoID_outputs'
        elif self.model == 'AnyStar':
            self.mask_dir = self.base_dir / 'anystar-mix'
            self.scales = scales
        else:
            raise ValueError(model)

        self.nms_thres = 0.5
        self.pred_boxes = []
        self.pred_scores = []
        self.pred_contours = []
        self.default_thres = 0.5

    def reset_image(self):
        self.pred_boxes = []
        self.pred_scores = []
        self.pred_contours = []

    def predict_cellsam(self, patch_name, offset_x=0, offset_y=0, scale = None):

        if self.model == 'OrgaSegment':
            mask = cv2.imread(self.mask_dir / f'{patch_name}_masks_class-1.png', cv2.IMREAD_GRAYSCALE)
            if mask is None:
                print(self.mask_dir / f'{patch_name}_masks_class-1.png')
                return np.array([]), np.array([]).reshape((-1, 4)), []
        elif self.model == 'OrganoID':
            mask = skimage.io.imread(self.mask_dir / f'{patch_name}_id-labeled.tif')
        elif self.model == "AnyStar":
            mask = cv2.imread(self.mask_dir / f'scale_{scale}_{scale}_100_percent' / f'{patch_name}.png')
            if mask is None:
                print(self.mask_dir / f'scale_{scale}_{scale}_100_percent' / f'{patch_name}.png')
            mask = mask[:, :, 0]
        else:
            raise ValueError(self.model)


        # print(mask.shape, mask)
        boxes = bxn.mask_to_boxes(mask) # [N, 4] in center y, center x, h, w in [0, 1]
        # masks = pp.convert_mask_to_binary(mask) # [N, H, B], 0/1
        scores = np.ones((boxes.shape[0],), dtype=float) # [N,] constant ones if they don't exist


        boxes = boxes + np.array([[offset_y, offset_x, offset_y, offset_x]])

        # mask = mask.astype(np.uint8)
        transform = rasterio.Affine(1, 0, offset_x, 0, 1, offset_y)
        polygons = []
        for v in sorted(np.unique(mask).astype(int).tolist()):
            if v==0:
                continue
            outlines = []
            for p, v in rio_shapes((mask==v).astype(np.uint8),
                                   mask=(mask==v).astype(np.uint8),
                                   connectivity=8,
                                   transform=transform):
                if v==1:
                    outlines.append(shapely.from_geojson(json.dumps(p)))
                else:
                    raise RuntimeError(f'value: {v}, polygon: {p}')
            # print(outlines)
            polygons.append(shapely.union_all(outlines))
        return scores, boxes, polygons

    def filter_diameter(self, min_diameter):
        diams_x = np.abs(self.pred_boxes[:, 3] - self.pred_boxes[:, 1])
        diams_y = np.abs(self.pred_boxes[:, 2] - self.pred_boxes[:, 0])
        keep_masks = (diams_x >= min_diameter) & (diams_y >= min_diameter)

        self.pred_contours = [c for c, keep in zip(self.pred_contours, keep_masks) if keep]
        self.pred_boxes = self.pred_boxes[keep_masks]
        self.pred_scores = self.pred_scores[keep_masks]

    def nms(self, boxes, scores, contours):
        kept_indices = pp.non_max_suppression(boxes, scores, threshold=self.nms_thres)
        if len(kept_indices) == 0:
            return np.array([], dtype=float).reshape((0, 4)), np.array([], dtype=float).reshape((0,)), []

        boxes = boxes[kept_indices]
        contours = [contours[ii] for ii in kept_indices]
        scores = scores[kept_indices]
        return boxes, scores, contours

    def forward_one_patch(self, patch_name, offset_x, offset_y, scale=None):
        pred_scores_patch, pred_boxes_patch, pred_contours_patch = self.predict_cellsam(patch_name, 
                                                                                        offset_x=offset_x, 
                                                                                        offset_y=offset_y, 
                                                                                        scale=scale)


        return pred_boxes_patch, pred_scores_patch, pred_contours_patch

    def forward(self, ds_name, image_ID, patch_size=1024, predict_masks=True, min_diameter=5):
        self.reset_image()

        patch_name = f'{ds_name}__{image_ID}'

        # Convert patch_size to tuple | list
        # Predict patches
        if self.model == 'AnyStar':
            for scale in self.scales:
                boxes_patch, scores_patch, contours = \
                    self.forward_one_patch(patch_name,
                                            offset_x=0,
                                            offset_y=0,
                                            scale=scale)
                self.pred_contours.extend(contours)
                self.pred_boxes.append(boxes_patch)
                self.pred_scores.append(scores_patch)

            self.pred_boxes = np.concatenate(self.pred_boxes, axis=0)
            self.pred_scores = np.concatenate(self.pred_scores, axis=0)
        else:
            boxes_patch, scores_patch, contours = \
                self.forward_one_patch(patch_name,
                                        offset_x=0,
                                        offset_y=0)
            self.pred_contours = contours
            self.pred_boxes = boxes_patch
            self.pred_scores = scores_patch


        # self.pred_boxes = np.concatenate(self.pred_boxes, axis=0)
        # self.pred_scores = np.concatenate(self.pred_scores, axis=0)

        # Postprocessing
        self.filter_diameter(min_diameter)
        
        self.pred_boxes, self.pred_scores, self.pred_contours = \
            self.nms(self.pred_boxes, 
                     self.pred_scores, 
                     self.pred_contours)
        
        if self.pred_boxes.shape[0] == 0:
            return self.pred_contours, self.pred_boxes, self.pred_scores

        # Threshold boxes
        masks, boxes, scores = self.set_threshold(self.default_thres)
        return masks, boxes, scores
    
    def set_threshold(self, conf_thres, predict_masks=True):
        keep_indices = self.pred_scores >= conf_thres
        pred_boxes = self.pred_boxes[keep_indices]
        pred_scores = self.pred_scores[keep_indices]
        pred_contours = [c for c, keep in zip(self.pred_contours, keep_indices) if keep]
        return pred_contours, pred_boxes, pred_scores
