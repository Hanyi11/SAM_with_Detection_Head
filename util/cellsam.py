import sys
import cv2
import numpy as np
import torch
import rasterio
from rasterio.features import shapes as rio_shapes
import shapely
import json
from . import dataloading as dl
from . import postprocessing as pp
from . import box_ops_numpy as bxn

from cellSAM import segment_cellular_image, get_model

# Load SAM and detection head
class CellSAM():
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # self.cellsam = 

        self.nms_thres = 0.5
        self.offsets = []
        self.pred_boxes = []
        self.pred_scores = []
        self.pred_contours = []
        self.default_thres = 0.5

    def reset_image(self):
        self.offsets = []
        self.pred_boxes = []
        self.pred_scores = []
        self.pred_contours = []
        self.pred_masks = []

        self.manual_boxes = []
        self.manual_contours = []

    def predict_cellsam(self, patch, offset_x, offset_y):
        H, W = patch.shape[:2]

        mask, embedding, bounding_boxes = segment_cellular_image(patch, device=str(self.device))
        # print('mask.shape', mask.shape)
        mask = cv2.resize(mask, (H, W), interpolation=cv2.INTER_NEAREST)

        # print('mask.shape', mask.shape)

        boxes = bxn.mask_to_boxes(mask) # bxn.mask_to_boxes(mask) # [N, 4] in center y, center x, h, w in [0, 1]
        # print('bounding_boxes', bounding_boxes.cpu().numpy().astype(int))
        # print('boxes', boxes.astype(int))
        # print('boxes', boxes.shape)

        # print('np.unique(mask)', np.unique(mask))
        # masks = pp.convert_mask_to_binary(mask) # [N, H, B], 0/1
        # print(masks.shape)
        scores = np.ones((boxes.shape[0],), dtype=float) # [N,] constant ones if they don't exist


        # boxes = bxn.cxcywh_to_xyxy(boxes) * max(H, W) + np.array([[offset_y, offset_x, offset_y, offset_x]])
        boxes = boxes / 1024 * np.array([[H, W, H, W]]) + np.array([[offset_y, offset_x, offset_y, offset_x]])

        # masks = masks.astype(np.uint8)
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
            return np.array([], dtype=float).reshape((0, 4)), np.array([], dtype=float).reshape((0,)), [], np.array([], dtype=float).reshape((0,2)), np.array([], dtype=float).reshape((0,))

        boxes = boxes[kept_indices]
        contours = [contours[ii] for ii in kept_indices]
        scores = scores[kept_indices]
        return boxes, scores, contours

    def forward_one_patch(self, patch, offset_x, offset_y):
        H, W = patch.shape[:2]
        with torch.inference_mode():


            pred_scores_patch, pred_boxes_patch, pred_contours_patch = self.predict_cellsam(patch, offset_x=offset_x, offset_y=offset_y)

            # pred_scores_patch = pred_scores_patch.cpu().numpy()
            # pred_boxes_patch = pred_boxes_patch.cpu().numpy()


        return pred_boxes_patch, pred_scores_patch, pred_contours_patch

    def forward(self, image, patch_size=1024, predict_masks=True, min_diameter=5):
        self.reset_image()

        # Convert patch_size to tuple | list
        H, W = image.shape[0], image.shape[1]
        if patch_size is None:
            patch_size = (max(H, W), )
        if not (isinstance(patch_size, list) or isinstance(patch_size, tuple)):
            patch_size = (patch_size, )
        
        # Predict patches
        i = 0
        for psize in patch_size:
            # print(psize)
            for im_crop, offset_x, offset_y, size in dl.patch_image(image, size=psize):
                boxes_patch, scores_patch, contours = \
                    self.forward_one_patch(patch=im_crop,
                                           offset_x=offset_x,
                                           offset_y=offset_y)
                self.pred_contours.extend(contours)
                self.pred_boxes.append(boxes_patch)
                self.pred_scores.append(scores_patch)
                print('patch', i)
                i += 1

        self.pred_boxes = np.concatenate(self.pred_boxes, axis=0)
        self.pred_scores = np.concatenate(self.pred_scores, axis=0)

        print(len(self.pred_boxes), len(self.pred_scores), len(self.pred_contours))

        # Postprocessing
        self.filter_diameter(min_diameter)
        
        print(len(self.pred_boxes), len(self.pred_scores), len(self.pred_contours))

        self.pred_boxes, self.pred_scores, self.pred_contours = \
            self.nms(self.pred_boxes, 
                     self.pred_scores, 
                     self.pred_contours)
        
        print(len(self.pred_boxes), len(self.pred_scores), len(self.pred_contours))
        
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
