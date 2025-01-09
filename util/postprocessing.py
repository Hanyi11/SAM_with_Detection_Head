








# import os
# import time
# import numpy as np
import json
from scipy.optimize import linear_sum_assignment
from skimage.measure import label

# import os
# from PIL import Image
import numpy as np
import cv2
# from copy import deepcopy
# from typing import Tuple
from pathlib import Path
# import cv2
# import numpy as np
import rasterio
from rasterio.features import shapes as rio_shapes
import scipy
import shapely

from .box_ops_numpy import cxcywh_to_xyxy



organoid_sam = Path('/ictstr01/groups/shared/users/lion.gleiter/organoid_sam')


def convert_mask_to_binary(mask):
    # if len(np.unique(mask)) == 0:
    #     return mask.reshape((-1, mask.shape[0], mask.shape[1]))
    masks = []
    for j in sorted(np.unique(mask).astype(int).tolist()):
        if j == 0:  # background
            continue
        # print('j', j)
        masks.append((mask == j).reshape((-1, mask.shape[0], mask.shape[1])))
    if len(masks) == 0:
        return np.array([], dtype=bool).reshape((-1, mask.shape[0], mask.shape[1]))
    else:
        return np.stack(masks, axis=0).reshape((-1, mask.shape[0], mask.shape[1]))

def get_pseudo_predictions(split, ds_name, im, mask, boxes, im_path, im_ID):
    box_folder = organoid_sam / 'patch_bbox_gt' / split / ds_name / f'im_{im_ID}'
    img_folder = organoid_sam / 'patch_images' / split / ds_name / f'im_{im_ID}'
    mask_folder = organoid_sam / 'patch_seg_gt' / split / ds_name / f'im_{im_ID}'
    assert box_folder.exists(), box_folder
    assert img_folder.exists(), img_folder
    assert mask_folder.exists(), mask_folder

    offsets = np.load(box_folder / 'offsets.npy')

    pred_boxes = []
    pred_scores = []
    pred_masks = []
    for patch_number, offset_y, offset_x in offsets:
        patch_img = cv2.imread(str(img_folder / f'patch_{patch_number}.png'))
        H, W = patch_img.shape[:2]
        pred_boxes_patch = np.load(box_folder / f'patch_{patch_number}.npy')
        pred_boxes_patch = cxcywh_to_xyxy(pred_boxes_patch) * max(H, W) + np.array([[offset_y, offset_x, offset_y, offset_x]])
        pred_boxes.append(pred_boxes_patch)

        pred_scores.append(np.random.uniform(0, 1, pred_boxes_patch.shape[0]) + 1)

        pred_masks_patch = np.load(mask_folder / f'patch_{patch_number}.npy')
        labels = np.zeros((im.shape[0], im.shape[1]), dtype=pred_masks_patch.dtype)
        labels[offset_y:offset_y+pred_masks_patch.shape[0], offset_x:offset_x+pred_masks_patch.shape[1]] = pred_masks_patch
        # for i in sorted(np.unique(labels).astype(int).tolist()):
        #     if i == 0:  # background
        #         continue
        #     mask = labels == i
        #     pred_masks.append(mask)
        pred_masks.append(convert_mask_to_binary(labels))

    pred_boxes = np.concatenate(pred_boxes, axis=0)
    pred_scores = np.concatenate(pred_scores, axis=0)
    pred_masks = np.concatenate(pred_masks, axis=0)
    # pred_masks = np.stack(pred_masks, axis=0)

    return offsets, pred_boxes, pred_scores, pred_masks


# /ictstr01/groups/shared/users/lion.gleiter/organoid_sam/patch_predictions/SAM_large/test/OrganoID_test/DetectionHead_SAM_large_OrganoID_train_default_best
def get_predictions(split, ds_name, pred_folder, im, im_ID):
    box_folder = organoid_sam / 'patch_bbox_gt' / split / ds_name / f'im_{im_ID}'
    img_folder = organoid_sam / 'patch_images' / split / ds_name / f'im_{im_ID}'
    assert box_folder.exists(), box_folder
    assert img_folder.exists(), img_folder

    offsets = np.load(box_folder / 'offsets.npy')

    pred_boxes = []
    pred_scores = []
    pred_masks = []
    pred_patch_numbers = []
    pred_offsets = []
    for patch_number, offset_y, offset_x in offsets:
        patch_img = cv2.imread(str(img_folder / f'patch_{patch_number}.png'))
        H, W = patch_img.shape[:2]


        pred_boxes_patch = np.load(pred_folder / f'im_{im_ID}' / f'pred_bbs_patch_{patch_number}.npy')
        # print('pred_boxes_patch.shape', pred_boxes_patch.shape)
        pred_boxes_patch = cxcywh_to_xyxy(pred_boxes_patch) * max(H, W) + np.array([[offset_y, offset_x, offset_y, offset_x]])
        pred_boxes.append(pred_boxes_patch)

        pred_scores_patch = np.load(pred_folder / f'im_{im_ID}' / f'pred_score_patch_{patch_number}.npy')
        # print('pred_scores_patch.shape', pred_scores_patch.shape)
        pred_scores.append(pred_scores_patch)

        pred_masks_patch = np.load(pred_folder / f'im_{im_ID}' / f'pred_mask_patch_{patch_number}.npy')
        # print('pred_masks_patch.shape', pred_masks_patch.shape)
        pred_masks_patch = pred_masks_patch.squeeze(1)

        labels = np.zeros((pred_masks_patch.shape[0], im.shape[0], im.shape[1]), dtype=pred_masks_patch.dtype)
        labels[:, offset_y:offset_y+pred_masks_patch.shape[1], offset_x:offset_x+pred_masks_patch.shape[2]] = pred_masks_patch
        
        pred_masks.append(labels)
        # pred_masks.append(convert_mask_to_binary(labels))

        pred_patch_numbers.append(np.ones(pred_scores_patch.shape, dtype=int) * patch_number)
        pred_offsets.append(np.ones(pred_boxes_patch.shape, dtype=int) * np.array([[offset_y, offset_x, offset_y + H, offset_x + W]]))

    pred_boxes = np.concatenate(pred_boxes, axis=0)
    pred_scores = np.concatenate(pred_scores, axis=0)
    pred_masks = np.concatenate(pred_masks, axis=0)
    pred_patch_numbers = np.concatenate(pred_patch_numbers, axis=0)
    pred_offsets = np.concatenate(pred_offsets, axis=0)

    return pred_boxes, pred_scores, pred_masks, pred_patch_numbers, pred_offsets


def get_predictions2(split, ds_name, pred_folder, im, im_ID, cthres = 0.96):
    box_folder = organoid_sam / 'patch_bbox_gt' / split / ds_name / f'im_{im_ID}'
    img_folder = organoid_sam / 'patch_images' / split / ds_name / f'im_{im_ID}'
    assert box_folder.exists(), box_folder
    assert img_folder.exists(), img_folder

    offsets = np.load(box_folder / 'offsets.npy')

    pred_boxes = []
    pred_scores = []
    pred_masks = []
    pred_patch_numbers = []
    pred_offsets = []
    for patch_number, offset_y, offset_x in offsets:
        patch_img = cv2.imread(str(img_folder / f'patch_{patch_number}.png'))
        H, W = patch_img.shape[:2]

        pred_scores_patch = np.load(pred_folder / f'im_{im_ID}' / f'pred_score_patch_{patch_number}.npy')

        pred_boxes_patch = np.load(pred_folder / f'im_{im_ID}' / f'pred_bbs_patch_{patch_number}.npy')
        pred_boxes_patch = cxcywh_to_xyxy(pred_boxes_patch) * max(H, W) + np.array([[offset_y, offset_x, offset_y, offset_x]])


        pred_masks_patch = np.load(pred_folder / f'im_{im_ID}' / f'pred_mask_patch_{patch_number}.npy')
        pred_masks_patch = pred_masks_patch.squeeze(1)

        labels = np.zeros((pred_masks_patch.shape[0], im.shape[0], im.shape[1]), dtype=pred_masks_patch.dtype)
        labels[:, offset_y:offset_y+pred_masks_patch.shape[1], offset_x:offset_x+pred_masks_patch.shape[2]] = pred_masks_patch
        
        keep = pred_scores_patch > cthres
        pred_boxes_patch = pred_boxes_patch[keep]
        pred_scores_patch = pred_scores_patch[keep]
        labels = labels[keep]

        pred_boxes.append(pred_boxes_patch)
        pred_scores.append(pred_scores_patch)
        pred_masks.append(labels)
        # pred_masks.append(convert_mask_to_binary(labels))

        pred_patch_numbers.append(np.ones(pred_scores_patch.shape, dtype=int) * patch_number)
        pred_offsets.append(np.ones(pred_boxes_patch.shape, dtype=int) * np.array([[offset_y, offset_x, offset_y + H, offset_x + W]]))

    pred_boxes = np.concatenate(pred_boxes, axis=0)
    pred_scores = np.concatenate(pred_scores, axis=0)
    pred_masks = np.concatenate(pred_masks, axis=0)
    pred_patch_numbers = np.concatenate(pred_patch_numbers, axis=0)
    pred_offsets = np.concatenate(pred_offsets, axis=0)

    return pred_boxes, pred_scores, pred_masks, pred_patch_numbers, pred_offsets



def stitch_contours(contours, patch_ids, offsets, boxes, scores):
    adjacancy = np.zeros((len(contours), len(contours)), dtype=bool)
    # print('offsets', offsets)
    # print('patch_ids', patch_ids)

    for i in range(len(contours)):
        contour = contours[i]
        patch_id = patch_ids[i]
        offset = offsets[i]
        
        for other_patch_id in np.unique(patch_ids):
            if patch_id == other_patch_id:
                # Same patch
                continue

            # Indices for contours from the other patch
            other_indices = patch_ids == other_patch_id

            if not np.any(other_indices):
                continue

            for j in np.nonzero(other_indices)[0]:
                offsets_other = offsets[j]
        
                y_left = np.maximum(offset[0], offsets_other[0])
                y_right = np.minimum(offset[2], offsets_other[2])
                x_left = np.maximum(offset[1], offsets_other[1])
                x_right = np.minimum(offset[3], offsets_other[3])
                overlap = shapely.Polygon((
                    (x_left, y_left),
                    (x_left, y_right),
                    (x_right, y_right),
                    (x_right, y_left),
                    (x_left, y_left),
                ))
                # print('overlap', overlap)
            
                contour_other = contours[j]

                contour_union = shapely.union(contour, contour_other)
                # print('contour_union', contour_union.area)

                contour_intersection = shapely.intersection(contour, contour_other)
                # print('contour_intersection', contour_intersection.area)

                union = shapely.intersection(overlap, contour_union).area
                intersection = shapely.intersection(overlap, contour_intersection).area

                # print('i, j', i, j)
                # print('contour', contour)
                # print('contour_other', contour_other)
                # print('patch_id', patch_id)
                # print('other_patch_id', other_patch_id)
                # print('contour.area, contour_other.area, intersection, union', contour.area, contour_other.area, intersection, union)
                # if (union > 0):
                #     print('intersection / union', intersection / union)
                if (union > 0) and (intersection / union > 0.9):
                    adjacancy[i, j] = True
                    adjacancy[j, i] = True

    n_components, connected_component_labels = scipy.sparse.csgraph.connected_components(adjacancy, directed=False)

    contours_out = []
    boxes_out = np.zeros((n_components, 4), dtype=float)
    scores_out = np.zeros((n_components,), dtype=float)
    for k in range(n_components):
        # print('k', k+1, 'of', n_components)
        cc_ids = connected_component_labels==k
        # print('cc_ids', cc_ids)
        # print('boxes', boxes[cc_ids, :])
        cc_contours = [contours[cc_id] for cc_id in np.nonzero(cc_ids)[0]]
        contours_out.append(shapely.union_all(cc_contours))
        boxes_out[k] = np.array([np.min(boxes[cc_ids, 0]), np.min(boxes[cc_ids, 1]), 
                                 np.max(boxes[cc_ids, 2]), np.max(boxes[cc_ids, 3])])
        scores_out[k] = np.max(scores[cc_ids])

    return contours_out, boxes_out, scores_out





def stitch(masks, patch_ids, offsets, boxes, scores):
    adjacancy = np.zeros((masks.shape[0], masks.shape[0]), dtype=bool)

    for i in range(masks.shape[0]):
        mask = masks[i]
        patch_id = patch_ids[i]
        offset = offsets[i]
        
        for other_patch_id in np.unique(patch_ids):
            if patch_id == other_patch_id:
                continue
            other_indices = patch_ids == other_patch_id

            if not np.any(other_indices):
                continue

            masks_other = masks[other_indices]
            patch_ids_other = patch_ids[other_indices]
            offsets_other = offsets[other_indices]
    
            y_left = np.maximum(offset[0], offsets_other[0, 0])
            y_right = np.minimum(offset[2], offsets_other[0, 2])
            x_left = np.maximum(offset[1], offsets_other[0, 1])
            x_right = np.minimum(offset[3], offsets_other[0, 3])

            ious = compute_iou_masks_array(mask[y_left:y_right, x_left:x_right], 
                                           masks_other[:, y_left:y_right, x_left:x_right])
            for j in range(ious.shape[0]):
                original_indices = np.nonzero(other_indices)[0]
                assert original_indices.ndim == 1, original_indices.shape
                assert original_indices.shape[0] == int(np.sum(other_indices)), original_indices
                j_original = int(original_indices[j])
                if ious[j] > 0.9:
                    adjacancy[i, j_original] = True
                    adjacancy[j_original, i] = True

    n_components, connected_component_labels = scipy.sparse.csgraph.connected_components(adjacancy, directed=False)


    masks_out = np.zeros((n_components, masks.shape[1], masks.shape[2]), dtype=bool)
    boxes_out = np.zeros((n_components, 4), dtype=float)
    scores_out = np.zeros((n_components,), dtype=float)

    for k in range(n_components):
        cc_ids = connected_component_labels==k
        masks_out[k] = np.any(masks[cc_ids], axis=0)
        boxes_out[k] = np.array([np.min(boxes[cc_ids, 0]), np.min(boxes[cc_ids, 1]), 
                                 np.max(boxes[cc_ids, 2]), np.max(boxes[cc_ids, 3])])
        scores_out[k] = np.max(scores[cc_ids])


    return masks_out, boxes_out, scores_out




# NMS
def iou_box_array(box, boxes):

    box = np.array(box).astype(float)
    boxes = np.array(boxes).astype(float)

    assert boxes.ndim==2, boxes.shape
    assert boxes.shape[1]==4, boxes.shape
    x_left = np.maximum(box[0], boxes[:, 0])
    x_right = np.minimum(box[2], boxes[:, 2])
    y_left = np.maximum(box[1], boxes[:, 1])
    y_right = np.minimum(box[3], boxes[:, 3])

    intersection = np.maximum(x_right - x_left, 0) * np.maximum(y_right - y_left, 0)

    area = (box[2] - box[0]) * (box[3] - box[1]) + (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

    assert np.all(area > 1), area
    # print(box, boxes)
    # print('np.minimum(box[2], boxes[:, 2])', np.minimum(box[2], boxes[:, 2]))
    # print('intersection', intersection)
    # print('area', area)
    _iou = np.where(area - intersection > 0, intersection / (area - intersection), 0.0)
    assert np.all(_iou >= 0), _iou
    assert np.all(_iou <= 1), _iou
    return _iou



def non_max_suppression(bboxes, scores, threshold):
    """Perform Non-Maximum Suppression (NMS) on bounding boxes.
    
    Args:
    bboxes (numpy.ndarray): Array of bounding boxes in the format (x1, y1, x2, y2).
    scores (numpy.ndarray): Array of scores for each bounding box.
    threshold (float): IoU threshold for suppression.

    Returns:
    numpy.ndarray: Array of indices of bounding boxes to keep.
    """
    # Sort the bounding boxes by the scores in descending order
    indices = np.argsort(scores)[::-1]
    
    keep = []
    while len(indices) > 0:
        current = indices[0]
        keep.append(current)
        
        if len(indices) == 1:
            break
        
        current_box = bboxes[current]
        remaining_boxes = bboxes[indices[1:]]
        
        # Compute IoU of the current box with the rest
        # ious = np.array([compute_iou_box(current_box, box) for box in remaining_boxes])
        ious = iou_box_array(current_box, remaining_boxes)
        
        # Select boxes with IoU less than the threshold
        indices = indices[1:][ious < threshold]
    
    return np.array(keep)

def compute_iou_box(box1, box2):
    x1_max = max(box1[0], box2[0])
    y1_max = max(box1[1], box2[1])
    x2_min = min(box1[2], box2[2])
    y2_min = min(box1[3], box2[3])

    intersection_area = max(0, x2_min - x1_max) * max(0, y2_min - y1_max)

    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union_area = box1_area + box2_area - intersection_area

    if union_area == 0:
        return 0.0

    iou = intersection_area / union_area
    return iou


def compute_iou_masks(pred_instance, gt_instance):
    
    intersection = np.logical_and(pred_instance, gt_instance).sum()
    union = np.logical_or(pred_instance, gt_instance).sum()
    
    if union == 0:
        return 0.0
    else:
        return intersection / union
    

def compute_iou_masks_array(pred_instance, gt_instances):
    
    intersection = np.logical_and(pred_instance[None, ...], gt_instances).sum(axis=(1, 2))
    union = np.logical_or(pred_instance[None, ...], gt_instances).sum(axis=(1, 2))
    

    iou = np.where(union > 0, intersection / union, 0.0)
    del intersection, union
    return iou



def compute_metrics_segmentation(pred_masks, gt_masks, iou_threshold=0.5):
    # pred_labels = np.unique(pred_mask)
    # gt_labels = np.unique(gt_mask)
    
    # # Remove background label (0)
    # pred_labels = pred_labels[pred_labels != 0]
    # gt_labels = gt_labels[gt_labels != 0]
    
    num_preds = pred_masks.shape[0]
    num_gts = gt_masks.shape[0]
    
    # Create the IoU matrix
    iou_matrix = np.zeros((num_preds, num_gts))
    
    # print('before iou', time.time())
    for i, pred_mask in enumerate(pred_masks):
        iou_matrix[i, :] = compute_iou_masks_array(pred_instance=pred_mask, gt_instances=gt_masks)
        # for j, gt_mask in enumerate(gt_masks):
        #     iou_matrix[i, j] = compute_iou_masks(pred_mask, gt_mask)
    
    # print('before lap', time.time())
    # Hungarian matching
    row_ind, col_ind = linear_sum_assignment(-iou_matrix)
    
    # print('after lap', time.time())

    tp = []
    fp = []
    fn = []
    
    matched_gt = set()
    matched_pred = set()
    
    for i, j in zip(row_ind, col_ind):
        if iou_matrix[i, j] >= iou_threshold:
            tp.append(iou_matrix[i, j])
            matched_gt.add(j)
            matched_pred.add(i)
    
    for i in range(num_preds):
        if i not in matched_pred:
            fp.append(i)
    
    for j in range(num_gts):
        if j not in matched_gt:
            fn.append(j)
    
    # Calculate PQ
    pq = np.sum(tp) / (len(tp) + 0.5 * len(fp) + 0.5 * len(fn))
    
    # Calculate precision, recall, and F1-score
    precision = len(tp) / (len(tp) + len(fp)) if (len(tp) + len(fp)) > 0 else 0
    recall = len(tp) / (len(tp) + len(fn)) if (len(tp) + len(fn)) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Calculate mean IoU
    mean_iou = np.mean(tp) if len(tp) > 0 else 0
    
    # Calculate Dice coefficient
    dice_coefficient = np.mean([2 * iou / (1 + iou) for iou in tp]) if len(tp) > 0 else 0
    
    return len(tp), len(fp), len(fn), pq, precision, recall, f1_score, mean_iou, dice_coefficient




def compute_metrics_detection(pred_boxes, gt_boxes, iou_threshold=0.5):
    
    num_preds = pred_boxes.shape[0]
    num_gts = gt_boxes.shape[0]
    
    # Create the IoU matrix
    iou_matrix = np.zeros((num_preds, num_gts))
    
    for i, pred_box in enumerate(pred_boxes):
        iou_matrix[i, :] = iou_box_array(pred_box, gt_boxes)
        # for j, gt_box in enumerate(gt_boxes):
        #     iou_matrix[i, j] = compute_iou_box(pred_box, gt_box)
    
    # Hungarian matching
    row_ind, col_ind = linear_sum_assignment(-iou_matrix)
    
    tp = []
    fp = []
    fn = []
    
    matched_gt = set()
    matched_pred = set()
    
    for i, j in zip(row_ind, col_ind):
        if iou_matrix[i, j] >= iou_threshold:
            tp.append(iou_matrix[i, j])
            matched_gt.add(j)
            matched_pred.add(i)
    
    for i in range(num_preds):
        if i not in matched_pred:
            fp.append(i)
    
    for j in range(num_gts):
        if j not in matched_gt:
            fn.append(j)
    
    
    # Calculate precision, recall, and F1-score
    precision = len(tp) / (len(tp) + len(fp)) if (len(tp) + len(fp)) > 0 else 0
    recall = len(tp) / (len(tp) + len(fn)) if (len(tp) + len(fn)) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Calculate mean IoU
    mean_iou = np.mean(tp) if len(tp) > 0 else 0
    
    # # Calculate Dice coefficient
    # dice_coefficient = np.mean([2 * iou / (1 + iou) for iou in tp]) if len(tp) > 0 else 0
    dice_coefficient = 0.0

    # # Calculate PQ
    # pq = np.sum(tp) / (len(tp) + 0.5 * len(fp) + 0.5 * len(fn))
    pq = 0.0
    
    return len(tp), len(fp), len(fn), pq, precision, recall, f1_score, mean_iou, dice_coefficient




def compute_iou_matrix_detection(pred_boxes, gt_boxes):
    
    num_preds = pred_boxes.shape[0]
    num_gts = gt_boxes.shape[0]
    
    # Create the IoU matrix
    iou_matrix = np.zeros((num_preds, num_gts))
    
    for i, pred_box in enumerate(pred_boxes):
        iou_matrix[i, :] = iou_box_array(pred_box, gt_boxes)

    return iou_matrix
    


def compute_metrics_detection_from_iou_matrix(iou_matrix, iou_threshold=0.5):
    
    num_preds = iou_matrix.shape[0]
    num_gts = iou_matrix.shape[1]
    
    # Hungarian matching
    row_ind, col_ind = linear_sum_assignment(-iou_matrix)
    
    tp = []
    fp = []
    fn = []
    
    matched_gt = set()
    matched_pred = set()
    
    for i, j in zip(row_ind, col_ind):
        if iou_matrix[i, j] >= iou_threshold:
            tp.append(iou_matrix[i, j])
            matched_gt.add(j)
            matched_pred.add(i)
    
    for i in range(num_preds):
        if i not in matched_pred:
            fp.append(i)
    
    for j in range(num_gts):
        if j not in matched_gt:
            fn.append(j)
    
    
    # Calculate precision, recall, and F1-score
    precision = len(tp) / (len(tp) + len(fp)) if (len(tp) + len(fp)) > 0 else 0
    recall = len(tp) / (len(tp) + len(fn)) if (len(tp) + len(fn)) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Calculate mean IoU
    mean_iou = np.mean(tp) if len(tp) > 0 else 0
    
    # # Calculate Dice coefficient
    # dice_coefficient = np.mean([2 * iou / (1 + iou) for iou in tp]) if len(tp) > 0 else 0
    dice_coefficient = 0.0

    # # Calculate PQ
    # pq = np.sum(tp) / (len(tp) + 0.5 * len(fp) + 0.5 * len(fn))
    pq = 0.0
    
    return len(tp), len(fp), len(fn), pq, precision, recall, f1_score, mean_iou, dice_coefficient




def compute_iou_matrix_segmentation(pred_masks, gt_masks):
    num_preds = pred_masks.shape[0]
    num_gts = gt_masks.shape[0]
    
    # Create the IoU matrix
    iou_matrix = np.zeros((num_preds, num_gts))
    
    for i, pred_mask in enumerate(pred_masks):
        iou_matrix[i, :] = compute_iou_masks_array(pred_instance=pred_mask, gt_instances=gt_masks)
        
    return iou_matrix


def mask_to_contour(mask):
    # Converts a binary mask into a shapely MultiPolygon.
    mask = mask.astype(np.uint8)
    outlines = []
    for p, v in rio_shapes(mask,
                           mask=mask,
                           connectivity=8):
        if v==1:
            outlines.append(shapely.from_geojson(json.dumps(p)))
        else:
            raise RuntimeError(f'value: {v}, polygon: {p}')
    return shapely.union_all(outlines)


def compute_iou_matrix_segmentation_contours(pred_contours, gt_masks):
    num_preds = len(pred_contours)
    num_gts = gt_masks.shape[0]
    print('gt_masks.shape', gt_masks.shape)
    
    # Create the IoU matrix
    iou_matrix = np.zeros((num_preds, num_gts))
    
    for j, gt_mask in enumerate(gt_masks):
        print('gt_mask.shape, gt_mask.dtype', gt_mask.shape, gt_mask.dtype)
        gt_contour = mask_to_contour(gt_mask)
        for i, pred_contour in enumerate(pred_contours):
            intersection = shapely.intersection(gt_contour, pred_contour).area
            union = shapely.union(gt_contour, pred_contour).area
            if union > 0:
                iou_matrix[i, j] = intersection / union
            else:
                iou_matrix[i, j] = 0.0
        
    return iou_matrix
    

def compute_metrics_segmentation_from_iou_matrix(iou_matrix, iou_threshold=0.5):
    # pred_labels = np.unique(pred_mask)
    # gt_labels = np.unique(gt_mask)
    
    # # Remove background label (0)
    # pred_labels = pred_labels[pred_labels != 0]
    # gt_labels = gt_labels[gt_labels != 0]
    
    num_preds = iou_matrix.shape[0]
    num_gts = iou_matrix.shape[1]


    # print('before lap', time.time())
    # Hungarian matching
    row_ind, col_ind = linear_sum_assignment(-iou_matrix)
    
    # print('after lap', time.time())

    tp = []
    fp = []
    fn = []
    
    matched_gt = set()
    matched_pred = set()
    
    for i, j in zip(row_ind, col_ind):
        if iou_matrix[i, j] >= iou_threshold:
            tp.append(iou_matrix[i, j])
            matched_gt.add(j)
            matched_pred.add(i)
    
    for i in range(num_preds):
        if i not in matched_pred:
            fp.append(i)
    
    for j in range(num_gts):
        if j not in matched_gt:
            fn.append(j)
    
    # Calculate PQ
    pq = np.sum(tp) / (len(tp) + 0.5 * len(fp) + 0.5 * len(fn))
    
    # Calculate precision, recall, and F1-score
    precision = len(tp) / (len(tp) + len(fp)) if (len(tp) + len(fp)) > 0 else 0
    recall = len(tp) / (len(tp) + len(fn)) if (len(tp) + len(fn)) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Calculate mean IoU
    mean_iou = np.mean(tp) if len(tp) > 0 else 0
    
    # Calculate Dice coefficient
    dice_coefficient = np.mean([2 * iou / (1 + iou) for iou in tp]) if len(tp) > 0 else 0
    
    return len(tp), len(fp), len(fn), pq, precision, recall, f1_score, mean_iou, dice_coefficient


