import sys
import numpy as np
import torch
import rasterio
from rasterio.features import shapes as rio_shapes
import shapely
import json
from typing import Literal
from PIL import Image
from torchvision.models.detection.faster_rcnn import FasterRCNN_ResNet50_FPN_V2_Weights, FasterRCNN_ResNet50_FPN_Weights
from . import dataloading as dl
from . import postprocessing as pp
from . import box_ops_numpy as bxn

# import detection_head_model as dhm
from FasterRCNN_model import FasterRCNN_model as frcnn
sys.path.append('/home/icb/lion.gleiter/projects/organoid_sam/segment-anything/segment-anything')
from segment_anything import build_sam_vit_l, predictor

# Load SAM and detection head
class FasterRCNNPredictor():
    def __init__(self, checkpoint_path, version_FasterRCNN: Literal["v1", "v2"]="v2", default_thres=0.5):
        # Initilaize Faster-RCNN from pretrained checkpoint 
        self.version_FasterRCNN = version_FasterRCNN
        self.frcnn_model = frcnn.load_from_checkpoint(checkpoint_path=checkpoint_path)
        self.frcnn_model.eval()
        
        # Load transformations
        if self.version_FasterRCNN == "v1":
            self.transforms = FasterRCNN_ResNet50_FPN_Weights.COCO_V1.transforms()
        elif self.version_FasterRCNN == "v2":
            self.transforms = FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1.transforms()

        # Initialize pretrained SAM large model
        sam_model = build_sam_vit_l(checkpoint='/ictstr01/groups/shared/users/lion.gleiter/organoid_sam/checkpoints/sam_vit_l_0b3195.pth')
        self.sam_predictor = predictor.SamPredictor(sam_model=sam_model.to(device=self.frcnn_model.device))
        self.nms_thres = 0.5

        # Initialize empty lists for storage
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
        """Initializex / Resets empty lists for storage."""
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
        """
        Forward pass of pretrained FasterRCNN. Images as inputs and outputs predicted boxes with their confidence score.
        """
        H, W = image.shape[:2]
        print('predict_boxes image.shape', image.shape)
        image = Image.fromarray(image)
        image = self.transforms(image)
        image = image.to(self.frcnn_model.device)

        with torch.inference_mode():
            outputs = self.frcnn_model.forward([image])
        boxes = outputs['pred_boxes']#.cpu().numpy()
        # [xyxy] in px to [cy cx h w] in [0, 1]
        boxes = torch.stack((
            (boxes[:, 1] + boxes[:, 3]) / 2, 
            (boxes[:, 0] + boxes[:, 2]) / 2,
            (boxes[:, 3] - boxes[:, 1]), 
            (boxes[:, 2] - boxes[:, 0]),
        ), dim=1) / max(H, W)
        scores = outputs['pred_scores']#.cpu().numpy()

        return scores, boxes #.cpu().numpy()

    def predict_masks(self, boxes, offset_x=0, offset_y=0, image_embedding=None):
        """
        Predicts segmentation masks from FRCNN predicted boxes. 
        `boxes` are assumed to be in y_center, x_center, h, w format normalized to [0, 1]
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

        print('transformed_boxes.shape', transformed_boxes.shape)

        # SAM forward pass with FasterRCNN detection boxes as queries
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
        """
        Only predictions with a minimum side length of the detetcion box are kept. 
        """
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
        """
        Non maximum supression.
        """
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
        """
        Process a single image patch to predict bounding boxes, scores, and optionally masks.

        Parameters:
            patch (np.ndarray): The input image patch to process.
            offset_x (int): Horizontal offset of the patch in the original image.
            offset_y (int): Vertical offset of the patch in the original image.
            size (int): Scaling factor for predicted boxes.
            patch_idx (int): Index of the current patch being processed.
            predict_masks (bool): Whether to predict segmentation masks for the patch.
            min_diameter (float): Minimum diameter constraint for objects (currently unused).

        Returns:
            tuple: Contains the following elements:
                - pred_boxes_patch (np.ndarray): Predicted bounding boxes in the original image coordinates.
                - pred_scores_patch (np.ndarray): Confidence scores for the predicted boxes.
                - pred_contours (Optional[list]): Predicted segmentation masks (if enabled).
                - offsets (np.ndarray): Array defining the offsets of the patch in the original image.
                - patch_numbers (np.ndarray): Array indicating the patch index for each prediction.
        """
        # Get dimensions of the patch
        H, W = patch.shape[:2]

        # Disable gradient computation for inference
        with torch.inference_mode():
            # Set up SAM predictor with the current patch
            self.sam_predictor.set_image(patch)
            image_embedding = {
                'original_size': self.sam_predictor.original_size,
                'input_size': self.sam_predictor.input_size,
                'features': self.sam_predictor.features,
                'is_image_set': True,
            }

            # Update or append the image embedding for the current patch index
            if patch_idx == len(self.image_embeddings):
                self.image_embeddings.append(image_embedding)
            elif patch_idx < len(self.image_embeddings):
                self.image_embeddings[patch_idx] = image_embedding
            else:
                raise RuntimeError(f"patch_idx {patch_idx} does not work with self.image_embeddings of size {len(self.image_embeddings)}")

            # Predict bounding boxes and scores for the patch
            pred_scores_patch, pred_boxes_patch = self.predict_boxes(patch)

            # Optionally predict masks for the detected bounding boxes
            if predict_masks:
                pred_contours = self.predict_masks(pred_boxes_patch, offset_x=offset_x, offset_y=offset_y)
            else:
                pred_contours = None

        # Convert predictions to numpy arrays for further processing
        pred_scores_patch = pred_scores_patch.cpu().numpy()
        pred_boxes_patch = pred_boxes_patch.cpu().numpy()

        # Convert bounding boxes to xyxy format and scale to original image coordinates
        pred_boxes_patch = bxn.cxcywh_to_xyxy(pred_boxes_patch) * size + np.array([[offset_y, offset_x, offset_y, offset_x]])

        # Associate predictions with the current patch index
        patch_numbers = np.ones(pred_scores_patch.shape, dtype=int) * patch_idx

        # Calculate offsets for the patch within the original image
        offsets = np.ones(pred_boxes_patch.shape, dtype=int) * np.array([[offset_y, offset_x, offset_y + H, offset_x + W]])

        return pred_boxes_patch, pred_scores_patch, pred_contours, offsets, patch_numbers


    def forward(self, image, patch_size=1024, predict_masks=True, min_diameter=5):
        """
        Process the entire image to predict bounding boxes, scores, and optionally masks by dividing it into patches.

        Parameters:
            image (np.ndarray): Input image to process.
            patch_size (int or tuple): Size of patches for processing. Defaults to 1024.
            predict_masks (bool): Whether to predict segmentation masks for each patch. Defaults to True.
            min_diameter (float): Minimum diameter constraint for objects. Defaults to 5.

        Returns:
            tuple: Contains the following elements:
                - masks (list): Final filtered masks (if enabled).
                - boxes (np.ndarray): Final filtered bounding boxes.
                - scores (np.ndarray): Final filtered confidence scores.
        """
        # Reset predictions and state for the new image
        self.reset_image()

        # Determine the dimensions of the image
        H, W = image.shape[0], image.shape[1]

        # Convert patch_size to a tuple if not already in that form
        if patch_size is None:
            patch_size = (max(H, W), )
        if not (isinstance(patch_size, list) or isinstance(patch_size, tuple)):
            patch_size = (patch_size, )

        # Process the image patch-by-patch
        i = 0
        for psize in patch_size:
            for im_crop, offset_x, offset_y, size in dl.patch_image(image, size=psize):
                # Process each patch and get predictions
                boxes_patch, scores_patch, contours, offsets, patch_numbers = \
                    self.forward_one_patch(
                        patch=im_crop,
                        offset_x=offset_x,
                        offset_y=offset_y,
                        size=size,
                        patch_idx=i,
                        predict_masks=predict_masks,
                        min_diameter=min_diameter
                    )

                # Store predictions
                if predict_masks:
                    self.pred_contours.extend(contours)
                self.pred_boxes.append(boxes_patch)
                self.pred_scores.append(scores_patch)
                self.pred_patch_numbers.append(patch_numbers)
                self.offsets.append(offsets)
                print('patch', i)
                i += 1

        # Combine predictions across all patches
        self.pred_boxes = np.concatenate(self.pred_boxes, axis=0)
        self.pred_scores = np.concatenate(self.pred_scores, axis=0)
        self.pred_patch_numbers = np.concatenate(self.pred_patch_numbers, axis=0)
        self.offsets = np.concatenate(self.offsets, axis=0)

        # Postprocess to filter predictions by diameter
        self.filter_diameter(min_diameter, predict_masks=predict_masks)

        # Filter out small masks based on area
        if predict_masks:
            keep_masks = [contour.area >= 100 for contour in self.pred_contours]
            self.pred_contours = [c for c, keep in zip(self.pred_contours, keep_masks) if keep]
            self.pred_boxes = self.pred_boxes[keep_masks]
            self.pred_patch_numbers = self.pred_patch_numbers[keep_masks]
            self.offsets = self.offsets[keep_masks]
            self.pred_scores = self.pred_scores[keep_masks]

        # TODO: Remove bounding boxes at patch borders

        # Apply non-maximum suppression (NMS) to finalize predictions
        self.pred_boxes, self.pred_scores, self.pred_contours, self.offsets, self.pred_patch_numbers = \
            self.nms(
                self.pred_boxes,
                self.pred_scores,
                self.pred_contours,
                self.pred_patch_numbers,
                self.offsets,
                predict_masks=predict_masks
            )

        # Return early if no predictions remain
        if self.pred_boxes.shape[0] == 0:
            return self.pred_contours, self.pred_boxes, self.pred_scores

        # Apply a threshold to filter final predictions
        masks, boxes, scores = self.set_threshold(self.default_thres, predict_masks=predict_masks)

        return masks, boxes, scores

    
    def forward_patches(self, patches, offsets, predict_masks=True, min_diameter=5):
        """
        Process multiple image patches to predict bounding boxes, scores, and optionally masks.

        Parameters:
            patches (list or ndarray): List or array of image patches to process.
            offsets (list of tuples): List of (x, y) offsets corresponding to each patch.
            predict_masks (bool, optional): Whether to predict masks for the detected objects. Defaults to True.
            min_diameter (int, optional): Minimum diameter for filtering detected objects. Defaults to 5.

        Returns:
            tuple: A tuple containing masks, bounding boxes, and scores.
        """
        self.reset_image()
        
        # Determine size from patch dimensions
        H, W = patches.shape[0], patches.shape[1]
        size = max(H, W)

        # Iterate over patches and their corresponding offsets
        for i, (im_crop, offset) in enumerate(zip(patches, offsets)):
            offset_x = offset[0]
            offset_y = offset[1]

            # Forward one patch and collect predictions
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

        # Concatenate predictions across all patches
        self.pred_boxes = np.concatenate(self.pred_boxes, axis=0)
        self.pred_scores = np.concatenate(self.pred_scores, axis=0)
        self.pred_patch_numbers = np.concatenate(self.pred_patch_numbers, axis=0)
        self.offsets = np.concatenate(self.offsets, axis=0)

        # Filter predictions based on diameter
        self.filter_diameter(min_diameter, predict_masks=predict_masks)

        if predict_masks:
            # Retain masks with sufficient area
            keep_masks = [contour.area >= 100 for contour in self.pred_contours]
            self.pred_contours = [c for c, keep in zip(self.pred_contours, keep_masks) if keep]
            self.pred_boxes = self.pred_boxes[keep_masks]
            self.pred_patch_numbers = self.pred_patch_numbers[keep_masks]
            self.offsets = self.offsets[keep_masks]
            self.pred_scores = self.pred_scores[keep_masks]

        # TODO: Remove boxes at patch borders

        # Apply non-maximum suppression (NMS)
        self.pred_boxes, self.pred_scores, self.pred_contours, self.offsets, self.pred_patch_numbers = \
            self.nms(self.pred_boxes, 
                    self.pred_scores, 
                    self.pred_contours, 
                    self.pred_patch_numbers, 
                    self.offsets, 
                    self.predict_masks)

        if self.pred_boxes.shape[0] == 0:
            return self.pred_contours, self.pred_boxes, self.pred_scores

        # Apply thresholding to final predictions
        masks, boxes, scores = self.set_threshold(self.default_thres, predict_masks=predict_masks)
        return masks, boxes, scores

    def set_threshold(self, conf_thres, predict_masks=True):
        """
        Filters predictions based on a confidence threshold and optionally processes masks.

        Args:
            conf_thres (float): Confidence threshold for filtering predictions.
            predict_masks (bool, optional): If True, processes predicted masks. Defaults to True.

        Returns:
            tuple: 
                - pred_masks (list or None): Processed masks if predict_masks is True, otherwise None.
                - pred_boxes (numpy.ndarray): Filtered bounding boxes.
                - pred_scores (numpy.ndarray): Filtered confidence scores.
        """
        # Filter predictions based on confidence threshold
        keep_indices = self.pred_scores >= conf_thres
        pred_boxes = self.pred_boxes[keep_indices]
        pred_patch_numbers = self.pred_patch_numbers[keep_indices]
        pred_offsets = self.offsets[keep_indices]
        pred_scores = self.pred_scores[keep_indices]

        if predict_masks:
            # Filter and stitch contours if masks are predicted
            pred_contours = [c for c, keep in zip(self.pred_contours, keep_indices) if keep]
            pred_masks, pred_boxes, pred_scores = pp.stitch_contours(
                pred_contours, 
                pred_patch_numbers, 
                pred_offsets, 
                pred_boxes, 
                pred_scores
            )
        else:
            pred_masks = None

        return pred_masks, pred_boxes, pred_scores


    def add_boxes(self, boxes):
        pass
