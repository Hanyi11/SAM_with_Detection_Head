import sys
from typing import Literal
import numpy as np
import torch
import rasterio
from rasterio.features import shapes as rio_shapes
import shapely
import json
from cellpose import models as cp_models
from . import dataloading as dl
from . import postprocessing as pp
from . import box_ops_numpy as bxn


import detection_head_model as dhm
sys.path.append('/home/icb/lion.gleiter/projects/organoid_sam/segment-anything/segment-anything')
from segment_anything import build_sam_vit_l, predictor


# Load SAM and detection head
class Cellpose():
    def __init__(self, diameter=[None]):
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.diameter = diameter
        self.cellpose = cp_models.Cellpose(model_type='cyto3', device=device)

        self.pred_boxes = []
        self.pred_scores = []
        self.pred_contours = []
        self.nms_thres = 0.5

    def reset_image(self):
        self.pred_boxes = []
        self.pred_scores = []
        self.pred_contours = []

    def predict_image(self, image, diameter):
        masks, _, _, _ = self.cellpose.eval([image], diameter=diameter, channels=[[0, 0]])
        masks = masks[0]
        boxes = bxn.mask_to_boxes(masks)
        masks = pp.convert_mask_to_binary(masks).astype(np.uint8)
        scores = np.ones((masks.shape[0],), dtype=float)

        # Convert masks to contours
        if boxes.shape[0]==0:
            contours = []
        else:
            polygons = []
            for mask in masks:
                outlines = []
                for p, v in rio_shapes(mask,
                                       mask=mask,
                                       connectivity=8):
                    if v==1:
                        outlines.append(shapely.from_geojson(json.dumps(p)))
                    else:
                        raise RuntimeError(f'value: {v}, polygon: {p}')
                polygons.append(shapely.union_all(outlines))
            contours = polygons

        return contours, boxes, scores

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

    def forward(self, image, patch_size=1024, predict_masks=True, min_diameter=5):
        self.reset_image()

        H, W = image.shape[0], image.shape[1]
        for diameter in self.diameter:
            contours, boxes, scores = self.predict_image(image, diameter=diameter)

            self.pred_contours.extend(contours)
            self.pred_boxes.append(boxes)
            self.pred_scores.append(scores)
        self.pred_boxes = np.concatenate(self.pred_boxes, axis=0)
        self.pred_scores = np.concatenate(self.pred_scores, axis=0)

        # Postprocessing
        self.filter_diameter(min_diameter)
        
        # if predict_masks:
        #     keep_masks = [contour.area >= 100 for contour in self.pred_contours]
        #     self.pred_contours = [c for c, keep in zip(self.pred_contours, keep_masks) if keep]
        #     self.pred_boxes = self.pred_boxes[keep_masks]
        #     self.pred_scores = self.pred_scores[keep_masks]
        

        self.pred_boxes, self.pred_scores, self.pred_contours = self.nms(self.pred_boxes, 
                                                                         self.pred_scores, 
                                                                         self.pred_contours)
        
        if self.pred_boxes.shape[0] == 0:
            return self.pred_contours, self.pred_boxes, self.pred_scores

        # Threshold boxes
        contours, boxes, scores = self.set_threshold(0.5)
        return contours, boxes, scores
    
    def set_threshold(self, conf_thres, predict_masks=True):
        keep_indices = self.pred_scores >= conf_thres
        pred_boxes = self.pred_boxes[keep_indices]
        pred_scores = self.pred_scores[keep_indices]
        pred_contours = [c for c, keep in zip(self.pred_contours, keep_indices) if keep]
        return pred_contours, pred_boxes, pred_scores





# Load SAM and detection head
class CellposeWithSAM():
    def __init__(self, diameter=None, prompt_sam: Literal['box', 'point'] | None = None):
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.diameter = diameter
        self.cellpose = cp_models.Cellpose(model_type='cyto3', device=device)
        self.cellpose.eval()

        self.prompt_sam = prompt_sam
        sam_model = build_sam_vit_l(checkpoint='/ictstr01/groups/shared/users/lion.gleiter/organoid_sam/checkpoints/sam_vit_l_0b3195.pth')
        self.sam_predictor = predictor.SamPredictor(sam_model=sam_model.to(device=self.detection_head.device))

        self.nms_thres = 0.5
        self.image_embeddings = []
        self.offsets = []
        self.pred_patch_numbers = []
        self.pred_boxes = []
        self.pred_scores = []
        self.pred_contours = []
        self.pred_masks = []
        self.default_thres = 0.5

        self.manual_boxes = []
        self.manual_contours = []

    def reset_image(self):
        self.image_embeddings = []
        self.offsets = []
        self.pred_patch_numbers = []
        self.pred_boxes = []
        self.pred_scores = []
        self.pred_contours = []
        self.pred_masks = []

        self.manual_boxes = []
        self.manual_contours = []

    def predict_boxes(self, image):
        device = self.cellpose.device
        masks, _, _, _ = self.cellpose.eval([image], diameter=self.diameter, channels=[[0, 0]])
        masks = masks[0]
        boxes = bxn.mask_to_boxes(masks)
        masks = pp.convert_mask_to_binary(masks)
        scores = np.ones((masks.shape[0],), dtype=float)
        return scores, boxes, masks

    def predict_masks(self, boxes, offset_x=0, offset_y=0, image_embedding=None):
        """`boxes` are assumed to be in y_center, x_center, h, w format normalized to [0, 1]
        """
        # device = self.detection_head.device
        if boxes.shape[0]==0:
            return []

        if image_embedding is not None:
            # Set image embedding
            for k, v in image_embedding.items():
                setattr(self.sam_predictor, k, v)

        # [cy cx h w] --> [x y x y]
        transformed_boxes = torch.stack((
            boxes[:, 1] - boxes[:, 3] / 2,
            boxes[:, 0] - boxes[:, 2] / 2,
            boxes[:, 1] + boxes[:, 3] / 2,
            boxes[:, 0] + boxes[:, 2] / 2
        ), dim=1) * 1024

        # Forward
        masks, _, _ = self.sam_predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=False,
        )
        masks = masks.squeeze(1).cpu().numpy().astype(np.uint8)
        transform = rasterio.Affine(1, 0, offset_x, 0, 1, offset_y)
        polygons = []
        for mask in masks:
            outlines = []
            for p, v in rio_shapes(mask,
                                   mask=mask,
                                   connectivity=8,
                                   transform=transform):
                if v==1:
                    outlines.append(shapely.from_geojson(json.dumps(p)))
                else:
                    raise RuntimeError(f'value: {v}, polygon: {p}')
            # print(outlines)
            polygons.append(shapely.union_all(outlines))
        return polygons

    def filter_diameter(self, min_diameter, predict_masks):
        diams_x = np.abs(self.pred_boxes[:, 3] - self.pred_boxes[:, 1])
        diams_y = np.abs(self.pred_boxes[:, 2] - self.pred_boxes[:, 0])
        keep_masks = (diams_x >= min_diameter) & (diams_y >= min_diameter)

        if predict_masks:
            self.pred_contours = [c for c, keep in zip(self.pred_contours, keep_masks) if keep]
        self.pred_boxes = self.pred_boxes[keep_masks]
        self.pred_patch_numbers = self.pred_patch_numbers[keep_masks]
        self.offsets = self.offsets[keep_masks]
        self.pred_scores = self.pred_scores[keep_masks]

    def nms(self, boxes, scores, contours, patch_numbers, offsets, predict_masks=True):
        kept_indices = pp.non_max_suppression(boxes, scores, threshold=self.nms_thres)
        if len(kept_indices) == 0:
            return np.array([], dtype=float).reshape((0, 4)), np.array([], dtype=float).reshape((0,)), [], np.array([], dtype=float).reshape((0,2)), np.array([], dtype=float).reshape((0,))

        boxes = boxes[kept_indices]
        if predict_masks:
            contours = [contours[ii] for ii in kept_indices]
        patch_numbers = patch_numbers[kept_indices]
        offsets = offsets[kept_indices]
        scores = scores[kept_indices]
        return boxes, scores, contours, offsets, patch_numbers

    def forward_one_patch(self, patch, offset_x, offset_y, size, patch_idx, predict_masks, min_diameter):
        H, W = patch.shape[:2]
        with torch.inference_mode():
            self.sam_predictor.set_image(patch)
            image_embedding = {
                'original_size': self.sam_predictor.original_size,
                'input_size': self.sam_predictor.input_size,
                'features': self.sam_predictor.features,
                'is_image_set': True,
            }
            if patch_idx == len(self.image_embeddings):
                self.image_embeddings.append(image_embedding)
            elif patch_idx < len(self.image_embeddings):
                self.image_embeddings[patch_idx] = image_embedding
            else:
                raise RuntimeError(f"patch_idx {patch_idx} does not work with self.image_embeddings of size {len(self.image_embeddings)}")
            
            pred_scores_patch, pred_boxes_patch = self.predict_boxes(image_embedding)

            if predict_masks:
                pred_contours = self.predict_masks(pred_boxes_patch, offset_x=offset_x, offset_y=offset_y)
            else:
                pred_contours = None
        pred_scores_patch = pred_scores_patch.cpu().numpy()
        pred_boxes_patch = pred_boxes_patch.cpu().numpy()

        pred_boxes_patch = bxn.cxcywh_to_xyxy(pred_boxes_patch) * size + np.array([[offset_y, offset_x, offset_y, offset_x]])

        patch_numbers = np.ones(pred_scores_patch.shape, dtype=int) * patch_idx

        offsets = np.ones(pred_boxes_patch.shape, dtype=int) * np.array([[offset_y, offset_x, offset_y + H, offset_x + W]])
        return pred_boxes_patch, pred_scores_patch, pred_contours, offsets, patch_numbers

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
            for im_crop, offset_x, offset_y, size in dl.patch_image(image, size=psize):
                boxes_patch, scores_patch, contours, offsets, patch_numbers = \
                    self.forward_one_patch(patch=im_crop,
                                           offset_x=offset_x,
                                           offset_y=offset_y,
                                           size=size,
                                           patch_idx=i,
                                           predict_masks=predict_masks,
                                           min_diameter=min_diameter)
                if predict_masks:
                    self.pred_contours.extend(contours)
                self.pred_boxes.append(boxes_patch)
                self.pred_scores.append(scores_patch)
                self.pred_patch_numbers.append(patch_numbers)
                self.offsets.append(offsets)
                print('patch', i)
                i += 1

        self.pred_boxes = np.concatenate(self.pred_boxes, axis=0)
        self.pred_scores = np.concatenate(self.pred_scores, axis=0)
        self.pred_patch_numbers = np.concatenate(self.pred_patch_numbers, axis=0)
        self.offsets = np.concatenate(self.offsets, axis=0)

        # Postprocessing
        self.filter_diameter(min_diameter, predict_masks=predict_masks)
        
        if predict_masks:
            keep_masks = [contour.area >= 100 for contour in self.pred_contours]
            self.pred_contours = [c for c, keep in zip(self.pred_contours, keep_masks) if keep]
            self.pred_boxes = self.pred_boxes[keep_masks]
            self.pred_patch_numbers = self.pred_patch_numbers[keep_masks]
            self.offsets = self.offsets[keep_masks]
            self.pred_scores = self.pred_scores[keep_masks]
        
        # TODO: Remove boxes at patch borders

        self.pred_boxes, self.pred_scores, self.pred_contours, self.offsets, self.pred_patch_numbers = \
            self.nms(self.pred_boxes, 
                     self.pred_scores, 
                     self.pred_contours, 
                     self.pred_patch_numbers, 
                     self.offsets, 
                     predict_masks=predict_masks)
        if self.pred_boxes.shape[0] == 0:
            return self.pred_contours, self.pred_boxes, self.pred_scores

        # Threshold boxes
        masks, boxes, scores = self.set_threshold(self.default_thres, predict_masks=predict_masks)
        return masks, boxes, scores
    
    def forward_patches(self, patches, offsets, predict_masks=True, min_diameter=5):
        self.reset_image()
        H, W = patches.shape[0], patches.shape[1]
        size = max(H, W)
        
        # Predict patches
        for i, (im_crop, offset) in enumerate(zip(patches, offsets)):
            offset_x = offset[0]
            offset_y = offset[1]

            boxes_patch, scores_patch, contours, offsets, patch_numbers = \
                self.forward_one_patch(patch=im_crop,
                                        offset_x=offset_x,
                                        offset_y=offset_y,
                                        size=size,
                                        patch_idx=i,
                                        predict_masks=predict_masks,
                                        min_diameter=min_diameter)
            if predict_masks:
                self.pred_contours.extend(contours)
            self.pred_boxes.append(boxes_patch)
            self.pred_scores.append(scores_patch)
            self.pred_patch_numbers.append(patch_numbers)
            self.offsets.append(offsets)

        self.pred_boxes = np.concatenate(self.pred_boxes, axis=0)
        self.pred_scores = np.concatenate(self.pred_scores, axis=0)
        self.pred_patch_numbers = np.concatenate(self.pred_patch_numbers, axis=0)
        self.offsets = np.concatenate(self.offsets, axis=0)

        # Postprocessing
        self.filter_diameter(min_diameter, predict_masks=predict_masks)
        
        if predict_masks:
            keep_masks = [contour.area >= 100 for contour in self.pred_contours]
            self.pred_contours = [c for c, keep in zip(self.pred_contours, keep_masks) if keep]
            self.pred_boxes = self.pred_boxes[keep_masks]
            self.pred_patch_numbers = self.pred_patch_numbers[keep_masks]
            self.offsets = self.offsets[keep_masks]
            self.pred_scores = self.pred_scores[keep_masks]
        
        # TODO: Remove boxes at patch borders

        self.pred_boxes, self.pred_scores, self.pred_contours, self.offsets, self.pred_patch_numbers = \
            self.nms(self.pred_boxes, 
                     self.pred_scores, 
                     self.pred_contours, 
                     self.pred_patch_numbers, 
                     self.offsets, 
                     self.predict_masks)
        if self.pred_boxes.shape[0] == 0:
            return self.pred_contours, self.pred_boxes, self.pred_scores

        # Threshold boxes
        masks, boxes, scores = self.set_threshold(self.default_thres, predict_masks=predict_masks)
        return masks, boxes, scores

    def set_threshold(self, conf_thres, predict_masks=True):
        keep_indices = self.pred_scores >= conf_thres
        pred_boxes = self.pred_boxes[keep_indices]
        pred_patch_numbers = self.pred_patch_numbers[keep_indices]
        pred_offsets = self.offsets[keep_indices]
        pred_scores = self.pred_scores[keep_indices]
        if predict_masks:
            pred_contours = [c for c, keep in zip(self.pred_contours, keep_indices) if keep]
            pred_masks, pred_boxes, pred_scores = pp.stitch_contours(pred_contours, 
                                                                     pred_patch_numbers, 
                                                                     pred_offsets, 
                                                                     pred_boxes, 
                                                                     pred_scores)
        else:
            pred_masks = None
        return pred_masks, pred_boxes, pred_scores

    def add_boxes(self, boxes):
        pass
