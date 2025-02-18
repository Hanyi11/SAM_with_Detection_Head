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


# Load SAM and detection head
class GroundTruthSAM():
    def __init__(self, sam_version: Literal['sam1', 'sam2'] = 'sam1',
                 box_noise_std: int = 0, box_noise_bias: int = 0):
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.sam_version = sam_version
        if sam_version=='sam1':
            sam_model = build_sam_vit_l(checkpoint='/ictstr01/groups/shared/users/lion.gleiter/organoid_sam/checkpoints/sam_vit_l_0b3195.pth')
            self.sam_predictor = predictor.SamPredictor(sam_model=sam_model.to(device=self.device))
        elif sam_version=='sam2':
            sam2_checkpoint = "/ictstr01/groups/shared/users/lion.gleiter/organoid_sam/checkpoints/sam2.1_hiera_large.pt"
            model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
            sam2 = build_sam2(model_cfg, sam2_checkpoint, device=self.device)
            self.sam_predictor = SAM2ImagePredictor(sam2)
        else:
            raise ValueError(f'SAM version {sam_version} is currently not supported')
        
        self.box_noise_std = box_noise_std
        self.box_noise_bias = box_noise_bias
        self.rng = np.random.default_rng(2024)

        self.nms_thres = 0.5
        self.image_embeddings = []
        self.pred_boxes = []
        self.pred_scores = []
        self.pred_contours = []
        self.embed_offsets = []
        self.default_thres = 0.5

    def reset_image(self):
        self.image_embeddings.clear()
        self.pred_boxes = []
        self.pred_scores = []
        self.pred_contours.clear()
        self.embed_offsets.clear()

    def predict_boxes(self, boxes: np.ndarray, H, W):
        scores = np.ones((boxes.shape[0],), dtype=float)
        noise = np.clip(self.rng.normal(0, self.box_noise_std, boxes.shape), 
                        -2*self.box_noise_std, 
                        2*self.box_noise_std)
        noisy_boxes = boxes + noise + np.array([[-self.box_noise_bias, 
                                                 -self.box_noise_bias, 
                                                 self.box_noise_bias, 
                                                 self.box_noise_bias]], dtype=boxes.dtype)
        
        # Ensure noisy min values are smaller than noisy max values of the box coordinates
        noisy_boxes = np.concatenate([
            np.minimum(noisy_boxes[:, :2], noisy_boxes[:, 2:]-1),
            np.maximum(noisy_boxes[:, :2]+1, noisy_boxes[:, 2:]),
        ], axis=1)
        
        # Ensure box is within image range
        noisy_boxes = np.clip(noisy_boxes, 0, np.array([[H-1, W-1, H-1, W-1]]))  
        return scores, noisy_boxes

    def predict_mask(self, box):
        """`box` is assumed to be in y, x, y, x format unnormalized in px of the whole image
        """
        # device = self.detection_head.device
        box = box.flatten()
        assert box.shape[0]==4, box.shape

        # Find smallest image embedding which completely contains the box
        contains_box = []
        for offsets in self.embed_offsets:
            if (offsets[:2] <= box[:2]).all() and (box[2:] <=offsets[2:]).all():
                contains_box.append(True)
            else:
                contains_box.append(False)
        
        if not np.any(contains_box):
            raise RuntimeError(f'SAM-based mask prediction failed because there is no patch embedding which fully covers the box {box}.')
        
        embed_index = contains_box.index(True)
        image_embedding = self.image_embeddings[embed_index]
        # print('image_embedding', image_embedding)
        offsets = self.embed_offsets[embed_index]
        off_y, off_x = offsets[:2]

        # Subtract offset from box
        input_box = box - np.array([off_y, off_x, off_y, off_x])

        # Set image embedding
        for k, v in image_embedding.items():
            setattr(self.sam_predictor, k, v)


        # Forward
        H, W = offsets[2:] - offsets[:2]
        if self.sam_version=='sam1':
            # Normalize box to [0, 1]
            input_box = torch.from_numpy(input_box / max(H, W))
            # [y x y x] --> [x y x y] in [0, 1024]
            transformed_boxes = torch.tensor([[input_box[1], input_box[0], input_box[3], input_box[2]]], device=self.device) * 1024
            # print('transformed_boxes', transformed_boxes)
            masks, _, _ = self.sam_predictor.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=transformed_boxes,
                multimask_output=False,
            )
            # print('masks.shape', masks.shape)
            masks = masks.squeeze(1).cpu().numpy()
            # print(masks.min(), masks.max(), masks.dtype)
            masks = masks.astype(np.uint8)
        elif self.sam_version=='sam2':
            # [y x y x] --> [x y x y] unnormalized
            transformed_boxes = np.array([[input_box[1], input_box[0], input_box[3], input_box[2]]])
            masks, scs, _ = self.sam_predictor.predict(
                point_coords=None,
                point_labels=None,
                box=transformed_boxes,
                multimask_output=False,
            )
            masks = (masks > 0.5).astype(np.uint8)
        assert np.all(masks.shape[-2:] == np.array([H, W], dtype=int)), f'{masks.shape}, {H}, {W}'
        assert masks.shape[0] == 1, masks.shape
        mask = masks[0]
        transform = rasterio.Affine(1, 0, off_x, 0, 1, off_y)
        outlines = []
        for p, v in rio_shapes(mask,
                               mask=mask,
                               connectivity=8,
                               transform=transform):
            if v==1:
                outlines.append(shapely.from_geojson(json.dumps(p)))
            else:
                raise RuntimeError(f'value: {v}, polygon: {p}')
        polygon = shapely.union_all(outlines)
        return polygon

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

    def nms(self, boxes, scores, contours):
        kept_indices = pp.non_max_suppression(boxes, scores, threshold=self.nms_thres)
        if len(kept_indices) == 0:
            return np.array([], dtype=float).reshape((0, 4)), np.array([], dtype=float).reshape((0,)), []

        boxes = boxes[kept_indices]
        contours = [contours[ii] for ii in kept_indices]
        scores = scores[kept_indices]
        return boxes, scores, contours

    def forward_one_patch(self, patch, offset_x, offset_y, predict_masks=None, min_diameter=None):
        H, W = patch.shape[:2]
        # print(patch.min(), patch.max(), patch.shape, patch.dtype)
        with torch.inference_mode():
            self.sam_predictor.set_image(patch)
            if self.sam_version=='sam1':
                image_embedding = {
                    'original_size': self.sam_predictor.original_size,
                    'input_size': self.sam_predictor.input_size,
                    'features': self.sam_predictor.features,
                    'is_image_set': True,
                }
            elif self.sam_version=='sam2':
                image_embedding = {
                    '_orig_hw': self.sam_predictor._orig_hw,  # Original size of the image
                    '_is_batch': self.sam_predictor._is_batch,        # Flag indicating if batch or not
                    '_features': self.sam_predictor._features,            # Features extracted from the image
                    '_is_image_set': True,                      # Flag indicating the image has been set
                    'mask_threshold': self.sam_predictor.mask_threshold
                }
                # print('computed image_embedding', image_embedding)
        self.image_embeddings.append(image_embedding)
        self.embed_offsets.append(np.array([offset_y, offset_x, offset_y + H, offset_x + W]))

    def forward(self, image, gt_boxes, patch_size=1024, predict_masks=True, min_diameter=5):
        self.reset_image()

        H, W = image.shape[0], image.shape[1]
        if patch_size is None:
            patch_size = (max(H, W), )
        if not (isinstance(patch_size, list) or isinstance(patch_size, tuple)):
            patch_size = (patch_size, )
        
        # Predict patches
        i = 0
        for psize in patch_size:
            for im_crop, offset_x, offset_y, size in dl.patch_image(image, size=psize, overlap = 1/2):
                self.forward_one_patch(patch=im_crop,
                                       offset_x=offset_x,
                                       offset_y=offset_y)
                

        self.pred_scores, self.pred_boxes = self.predict_boxes(gt_boxes, H=H, W=W)
        for box in self.pred_boxes:
            self.pred_contours.append(self.predict_mask(box))

        # Postprocessing (not necessary if GT boxes are provided)
        # self.filter_diameter(min_diameter, predict_masks=predict_masks)
        
        # self.pred_boxes, self.pred_scores, self.pred_contours = self.nms(self.pred_boxes, 
        #                                                                  self.pred_scores, 
        #                                                                  self.pred_contours)
        if self.pred_boxes.shape[0] == 0:
            return self.pred_contours, self.pred_boxes, self.pred_scores

        # Threshold boxes
        contours, boxes, scores = self.set_threshold(self.default_thres)
        return contours, boxes, scores

    def set_threshold(self, conf_thres, predict_masks=True):
        keep_indices = self.pred_scores >= conf_thres
        pred_boxes = self.pred_boxes[keep_indices]
        pred_scores = self.pred_scores[keep_indices]
        pred_contours = [c for c, keep in zip(self.pred_contours, keep_indices) if keep]
        
        return pred_contours, pred_boxes, pred_scores
