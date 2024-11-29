import sys
import numpy as np
import torch
import rasterio
from rasterio.features import shapes as rio_shapes
import shapely
import json
from . import dataloading as dl
from . import postprocessing as pp
from . import box_ops_numpy as bxn

import detection_head_model as dhm
sys.path.append('/home/icb/lion.gleiter/projects/organoid_sam/segment-anything/segment-anything')
from segment_anything import build_sam_vit_l, predictor

# Load SAM and detection head
class SAMOS():
    def __init__(self, checkpoint_path, default_thres=0.96):
        self.detection_head = dhm.DetectionHead.load_from_checkpoint(checkpoint_path=checkpoint_path)
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
        self.default_thres = default_thres

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


    def predict_boxes(self, image_embedding):
        # print(image_embedding.shape)
        
        device = self.detection_head.device
        image_embedding = image_embedding['features'].to(device)
        pos_embedding = self.detection_head.position_embedding(image_embedding) # bs x 256 x 64 x 64

        # forward
        outputs = self.detection_head.forward(
            query_embedding=self.detection_head.query_embed.weight, #.to(device), 
            image_embedding=image_embedding, 
            pos_embedding=pos_embedding
        )
        scores = torch.nn.functional.softmax(outputs['pred_logits'], dim=-1)
        # print('scores', scores)
        scores = scores[0, :, 0]
        boxes = outputs['pred_boxes'][0]
        return scores, boxes #.cpu().numpy()


    def predict_masks(self, boxes, offset_x=0, offset_y=0, image_embedding=None):
        """`boxes` are assumed to be in y_center, x_center, h, w format normalized to [0, 1]
        """
        # device = self.detection_head.device

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
        # print(transformed_boxes)

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

    def forward(self, image, patch_size=1024, predict_masks=True):
        self.reset_image()
        H, W = image.shape[0], image.shape[1]

        if patch_size is None:
            patch_size = (max(H, W), )
        if not (isinstance(patch_size, list) or isinstance(patch_size, tuple)):
            patch_size = (patch_size, )
        
        # self.pred_boxes = []
        # self.pred_scores = []
        # self.pred_masks = []
        # self.pred_patch_numbers = []
        # self.offsets = []
        for psize in patch_size:
            for i, (im_crop, offset_x, offset_y, size) in enumerate(dl.patch_image(image, size=psize)):
                with torch.inference_mode():
                    self.sam_predictor.set_image(im_crop)
                    image_embedding = {
                        'original_size': self.sam_predictor.original_size,
                        'input_size': self.sam_predictor.input_size,
                        'features': self.sam_predictor.features,
                        'is_image_set': True,
                    }
                    self.image_embeddings.append(image_embedding)
                    pred_scores_patch, pred_boxes_patch = self.predict_boxes(image_embedding)

                    if predict_masks:
                        self.pred_contours.extend(
                            self.predict_masks(pred_boxes_patch, offset_x=offset_x, offset_y=offset_y)
                        )

                pred_scores_patch = pred_scores_patch.cpu().numpy()
                pred_boxes_patch = pred_boxes_patch.cpu().numpy()
                # if predict_masks:
                #     pred_masks_patch = pred_masks_patch.cpu().numpy()

                pred_boxes_patch = bxn.cxcywh_to_xyxy(pred_boxes_patch) * size + np.array([[offset_y, offset_x, offset_y, offset_x]])
                self.pred_boxes.append(pred_boxes_patch)

                self.pred_scores.append(pred_scores_patch)

                # if predict_masks:
                #     labels = np.zeros((pred_masks_patch.shape[0], H, W), dtype=pred_masks_patch.dtype)
                #     labels[:, offset_y:offset_y+size, offset_x:offset_x+size] = pred_masks_patch
                #     self.pred_masks.append(labels)

                self.pred_patch_numbers.append(np.ones(pred_scores_patch.shape, dtype=int) * i)
                self.offsets.append(np.ones(pred_boxes_patch.shape, dtype=int) * np.array([[offset_y, offset_x, offset_y + size, offset_x + size]]))

        self.pred_boxes = np.concatenate(self.pred_boxes, axis=0)
        self.pred_scores = np.concatenate(self.pred_scores, axis=0)
        # if predict_masks:
        #     self.pred_masks = np.concatenate(self.pred_masks, axis=0)
        self.pred_patch_numbers = np.concatenate(self.pred_patch_numbers, axis=0)
        self.offsets = np.concatenate(self.offsets, axis=0)


        # Postprocessing
        if predict_masks:
            keep_masks = [contour.area >= 100 for contour in self.pred_contours]
                        #   self.pred_masks.sum(axis=(1, 2)) >= 100
            self.pred_contours = [c for c, keep in zip(self.pred_contours, keep_masks) if keep]
            self.pred_boxes = self.pred_boxes[keep_masks]
            # self.pred_masks = self.pred_masks[keep_masks]
            self.pred_patch_numbers = self.pred_patch_numbers[keep_masks]
            self.offsets = self.offsets[keep_masks]
            self.pred_scores = self.pred_scores[keep_masks]


        # # Postprocessing aspect ratio
        # keep_masks = self.pred_masks.sum(axis=(1, 2)) >= 100
        # self.pred_boxes = self.pred_boxes[keep_masks]
        # self.pred_masks = self.pred_masks[keep_masks]
        # self.pred_patch_numbers = self.pred_patch_numbers[keep_masks]
        # self.offsets = self.offsets[keep_masks]
        # self.pred_scores = self.pred_scores[keep_masks]
        
        
        # TODO: Remove boxes at patch borders

        kept_indices = pp.non_max_suppression(self.pred_boxes, self.pred_scores, threshold=self.nms_thres)
        # print(kept_indices)
        if len(kept_indices) == 0:
            # return np.array([], dtype=float).reshape((0, H, W)), np.array([], dtype=float).reshape((0, 4)), np.array([], dtype=float).reshape((0,))
            return [], np.array([], dtype=float).reshape((0, 4)), np.array([], dtype=float).reshape((0,))
        # print('after nms')

        self.pred_boxes = self.pred_boxes[kept_indices]
        if predict_masks:
            self.pred_contours = [self.pred_contours[ii] for ii in kept_indices]
            # self.pred_masks = self.pred_masks[kept_indices]
        self.pred_patch_numbers = self.pred_patch_numbers[kept_indices]
        self.offsets = self.offsets[kept_indices]
        self.pred_scores = self.pred_scores[kept_indices]

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
            # pred_masks = self.pred_masks[keep]
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
