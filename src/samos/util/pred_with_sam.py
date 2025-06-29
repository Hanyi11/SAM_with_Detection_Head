from pathlib import Path
import sys
import time
from typing import Literal
import cv2
import numpy as np
import torch
import rasterio
from rasterio.features import shapes as rio_shapes
import shapely
import json

from tqdm import tqdm
# from cellpose import models as cp_models
# from . import dataloading as dl
from . import postprocessing as pp
# from . import box_ops_numpy as bxn


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


# Load SAM and detection head
class PredictionSAM():
    def __init__(self, logging_name,
                 sam_version: Literal['sam1', 'sam2'] = 'sam1',
                 max_detections = 400):
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.logging_name = logging_name
        self.sam_version = sam_version
        self.max_detections = max_detections
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
        
        self.nms_thres = 0.5
        self.pred_boxes = []
        self.pred_scores = []
        self.pred_contours = []
        self.default_thres = 0.5

        self.base_patches = Path('/ictstr01/groups/shared/users/lion.gleiter/organoid_sam/patched_data_multiscale_miccai')
        self.base_preds = Path('/ictstr01/groups/shared/users/lion.gleiter/organoid_sam/testset_predictions/trained') / self.logging_name

    def reset_image(self):
        self.pred_boxes = []
        self.pred_scores = []
        self.pred_contours.clear()

    # def predict_mask(self, box, H, W):
    #     """`box` is assumed to be in y, x, y, x format unnormalized in px of the whole image
    #     """
    #     # device = self.detection_head.device
    #     box = box.flatten()
    #     assert box.shape[0]==4, box.shape

    #     # Subtract offset from box
    #     input_box = box

    #     # Set image embedding
    #     for k, v in image_embedding.items():
    #         setattr(self.sam_predictor, k, v)


    #     # Forward
    #     # H, W = offsets[2:] - offsets[:2]
    #     if self.sam_version=='sam1':
    #         # Normalize box to [0, 1]
    #         input_box = torch.from_numpy(input_box / max(H, W))
    #         # [y x y x] --> [x y x y] in [0, 1024]
    #         transformed_boxes = torch.tensor([[input_box[1], input_box[0], input_box[3], input_box[2]]], device=self.device) * 1024
    #         # print('transformed_boxes', transformed_boxes)
    #         masks, _, _ = self.sam_predictor.predict_torch(
    #             point_coords=None,
    #             point_labels=None,
    #             boxes=transformed_boxes,
    #             multimask_output=False,
    #         )
    #         # print('masks.shape', masks.shape)
    #         masks = masks.squeeze(1).cpu().numpy()
    #         # print(masks.min(), masks.max(), masks.dtype)
    #         masks = masks.astype(np.uint8)
    #     elif self.sam_version=='sam2':
    #         # [y x y x] --> [x y x y] unnormalized
    #         transformed_boxes = np.array([[input_box[1], input_box[0], input_box[3], input_box[2]]])
    #         masks, scs, _ = self.sam_predictor.predict(
    #             point_coords=None,
    #             point_labels=None,
    #             box=transformed_boxes,
    #             multimask_output=False,
    #         )
    #         masks = (masks > 0.5).astype(np.uint8)
    #     assert np.all(masks.shape[-2:] == np.array([H, W], dtype=int)), f'{masks.shape}, {H}, {W}'
    #     assert masks.shape[0] == 1, masks.shape
    #     mask = masks[0]
    #     transform = rasterio.Affine(1, 0, off_x, 0, 1, off_y)
    #     outlines = []
    #     for p, v in rio_shapes(mask,
    #                            mask=mask,
    #                            connectivity=8,
    #                            transform=transform):
    #         if v==1:
    #             outlines.append(shapely.from_geojson(json.dumps(p)))
    #         else:
    #             raise RuntimeError(f'value: {v}, polygon: {p}')
    #     polygon = shapely.union_all(outlines)
    #     return polygon

    def filter_diameter(self, boxes, scores, contours, patch_boxes, patch_ids, min_diameter):
        diams_x = np.abs(boxes[:, 3] - boxes[:, 1])
        diams_y = np.abs(boxes[:, 2] - boxes[:, 0])
        keep_masks = (diams_x >= min_diameter) & (diams_y >= min_diameter)

        contours = [c for c, keep in zip(contours, keep_masks) if keep]
        boxes = boxes[keep_masks]
        scores = scores[keep_masks]
        patch_boxes = patch_boxes[keep_masks]
        patch_ids = patch_ids[keep_masks]
        return boxes, scores, contours, patch_boxes, patch_ids

    def nms(self, boxes, scores, contours, boxes_patches, patch_ids):
        kept_indices = pp.non_max_suppression(boxes, scores, threshold=self.nms_thres)
        if len(kept_indices) == 0:
            return np.array([], dtype=float).reshape((0, 4)), np.array([], dtype=float).reshape((0,)), []

        boxes = boxes[kept_indices]
        contours = [contours[ii] for ii in kept_indices]
        scores = scores[kept_indices]
        boxes_patches = boxes_patches[kept_indices]
        patch_ids = patch_ids[kept_indices]
        return boxes, scores, contours, boxes_patches, patch_ids

    def load_boxes_one_patch(self, patched_image, pred_boxes_dir, off_x, off_y):
        boxes = np.load(pred_boxes_dir / f'{patched_image.stem}_boxes.npy', allow_pickle=False)
        scores = np.load(pred_boxes_dir / f'{patched_image.stem}_scores.npy', allow_pickle=False)
        contours = [None] * scores.shape[0]

        # xyxy to yxyx and coordinates of the full image
        boxes_patch = boxes.copy()




        boxes_full_image = np.stack([boxes[:, 1], boxes[:, 0], boxes[:, 3], boxes[:, 2]], axis=1)
        boxes_full_image += np.array([[off_y, off_x, off_y, off_x]])
        return contours, boxes_full_image, scores, boxes_patch


    def predict_masks_one_patch(self, patched_image, patched_embeddings_dir, boxes_patch, off_x, off_y):
        patch = cv2.imread(patched_image)
        H, W = patch.shape[:2]

        embed_file = patched_embeddings_dir / f'{patched_image.stem}.pt'
        try:
            # Set pre-computed image embedding
            image_embedding = torch.load(embed_file, map_location=self.device)
            for k, v in image_embedding.items():
                setattr(self.sam_predictor, k, v)
        except Exception as e:
            print(f'\n\nException {e} occurred for embedding file {embed_file}. New embedding will be saved.\n\n')
            # Compute and save image embedding
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
            torch.save(image_embedding, embed_file)

        with torch.inference_mode():
            # Predict boxes
            contours = []
            for input_box in boxes_patch:
                
                # Forward
                if self.sam_version=='sam1':
                    # Normalize box to [0, 1]
                    input_box = torch.from_numpy(input_box / max(H, W))
                    # To [x y x y] in [0, 1024]
                    transformed_boxes = torch.tensor([[input_box[0], input_box[1], input_box[2], input_box[3]]], device=self.device) * 1024
                    # print('transformed_boxes', transformed_boxes)
                    masks, _, _ = self.sam_predictor.predict_torch(
                        point_coords=None,
                        point_labels=None,
                        boxes=transformed_boxes,
                        multimask_output=False,
                    )
                    masks = masks.squeeze(1).cpu().numpy()
                    masks = masks.astype(np.uint8)
                elif self.sam_version=='sam2':
                    # To [x y x y] unnormalized
                    transformed_boxes = np.array([[input_box[0], input_box[1], input_box[2], input_box[3]]])
                    masks, _, _ = self.sam_predictor.predict(
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
                contours.append(polygon)
        return contours


    def forward(self, ds_name, image_ID, patch_size=1024, predict_masks=True, min_diameter=5):
        self.reset_image()

        if self.sam_version=='sam1':
            embed_name = 'SAM_large'
        elif self.sam_version=='sam2':
            embed_name = 'SAM2_large'

        patched_images_dir = self.base_patches / 'patch_images' / 'test' / ds_name / f'im_{image_ID}'
        patched_embeddings_dir = self.base_patches / 'patch_embed' / embed_name / 'test' / ds_name / f'im_{image_ID}'
        pred_boxes_dir = self.base_preds / ds_name / f'im_{image_ID}'

        
        offsets = np.load(self.base_patches / 'patch_bbox_gt' / 'test' / ds_name / f'im_{image_ID}' / 'offsets.npy', allow_pickle=False)

        patched_images = sorted(list(patched_images_dir.glob('patch_*.png')))

        all_boxes_patch = []
        patched_image_ids = []
        for i, patched_image in tqdm(enumerate(patched_images)):
            patch_num = int(patched_image.stem.replace('patch_', ''))
            offset = offsets[patch_num]
            contours, boxes, scores, boxes_patch = self.load_boxes_one_patch(patched_image=patched_image, 
                                                                             pred_boxes_dir=pred_boxes_dir,
                                                                             off_x=offset[2],
                                                                             off_y=offset[1])
            self.pred_scores.append(scores)
            self.pred_boxes.append(boxes)
            self.pred_contours.extend(contours)
            all_boxes_patch.append(boxes_patch)
            patched_image_ids.append(np.ones(scores.shape, dtype=int) * i)

        self.pred_scores = np.concatenate(self.pred_scores, axis=0)
        self.pred_boxes = np.concatenate(self.pred_boxes, axis=0)
        all_boxes_patch = np.concatenate(all_boxes_patch, axis=0)
        patched_image_ids = np.concatenate(patched_image_ids, axis=0)

        print('before filter_diameter', time.time())
        # Postprocessing
        self.pred_boxes, self.pred_scores, self.pred_contours, all_boxes_patch, patched_image_ids = \
            self.filter_diameter(self.pred_boxes, self.pred_scores, self.pred_contours, all_boxes_patch, patched_image_ids, min_diameter)



        # Only keep top max_detections
        print('len(self.pred_scores)', self.pred_scores.shape[0])
        sorted_indices = np.argsort(self.pred_scores)[::-1]
        sorted_indices = sorted_indices[:5000]

        self.pred_boxes = self.pred_boxes[sorted_indices]
        self.pred_scores = self.pred_scores[sorted_indices]
        all_boxes_patch = all_boxes_patch[sorted_indices]
        patched_image_ids = patched_image_ids[sorted_indices]
        self.pred_contours = [self.pred_contours[i] for i in sorted_indices]



        print('after filter_diameter', time.time())
        self.pred_boxes, self.pred_scores, self.pred_contours, all_boxes_patch, patched_image_ids = \
            self.nms(self.pred_boxes, self.pred_scores, self.pred_contours, all_boxes_patch, patched_image_ids)
        
        print('after nms', time.time())
        
        if self.pred_boxes.shape[0] == 0:
            return [], self.pred_boxes, self.pred_scores
        
        # Only keep top max_detections
        sorted_indices = np.argsort(self.pred_scores)[::-1]
        sorted_indices = sorted_indices[:self.max_detections]

        self.pred_boxes = self.pred_boxes[sorted_indices]
        self.pred_scores = self.pred_scores[sorted_indices]
        all_boxes_patch = all_boxes_patch[sorted_indices]
        patched_image_ids = patched_image_ids[sorted_indices]
        self.pred_contours = [self.pred_contours[i] for i in sorted_indices]
        
        # Predicts masks for boxes only after NMS to save computation.
        if predict_masks:
            self.pred_contours.clear()

            # Needed to reorder boxes and scores, since contours are predicted in different order:
            boxes = []  
            scores = []
            for i, patched_image in enumerate(patched_images):
                selected_ids = patched_image_ids == i
                boxes_patch = all_boxes_patch[selected_ids]
                boxes.append(self.pred_boxes[selected_ids].copy())
                scores.append(self.pred_scores[selected_ids].copy())
                if boxes_patch.shape[0]==0:
                    continue

                patch_num = int(patched_image.stem.replace('patch_', ''))
                offset = offsets[patch_num]
                # print('offset', offset)
                contours = self.predict_masks_one_patch(patched_image=patched_image, 
                                                        patched_embeddings_dir=patched_embeddings_dir,
                                                        boxes_patch=boxes_patch,
                                                        off_x=offset[2],
                                                        off_y=offset[1])
                self.pred_contours.extend(contours)
            self.pred_boxes = np.concatenate(boxes)
            self.pred_scores = np.concatenate(scores)
        
        assert len(self.pred_contours)==self.pred_boxes.shape[0], (len(self.pred_contours), self.pred_boxes.shape[0])

        print('after predict masks', time.time())
        # Threshold boxes
        contours, boxes, scores = self.set_threshold(self.default_thres)

        print('after set_threshold', time.time())

        return contours, boxes, scores

    def set_threshold(self, conf_thres, predict_masks=True):
        keep_indices = self.pred_scores >= conf_thres
        pred_boxes = self.pred_boxes[keep_indices]
        pred_scores = self.pred_scores[keep_indices]
        pred_contours = [c for c, keep in zip(self.pred_contours, keep_indices) if keep]
        
        return pred_contours, pred_boxes, pred_scores
