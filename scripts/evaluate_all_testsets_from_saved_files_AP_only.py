import io
import json
import numpy as np
import cv2
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from copy import deepcopy
import skimage
from skimage.io import imread
from sklearn import metrics
import scipy
import basicpy
import tifffile
import torch
from tqdm import tqdm
# from descartes import PolygonPatch
import geopandas as gpd
import basicpy
import torchmetrics.detection

import warnings
warnings.filterwarnings("ignore", "You are using `torch.load` with `weights_only=False`*.")


import sys
sys.path.append('/home/icb/lion.gleiter/projects/organoid_sam/SAM_with_Detection_Head')

from util.box_ops_numpy import mask_to_boxes, cxcywh_to_xyxy, xyxy_to_cxcywh, plot_boxes
from util import dataloading as dl
from util import postprocessing as pp
from util.pred_with_sam import PredictionSAM

base_datadir = Path('/ictstr01/groups/shared/users/lion.gleiter/organoid_sam/original_data/')
results_dir = Path('/ictstr01/groups/shared/users/lion.gleiter/organoid_sam/results')


def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax, color='red'):
    y_min, x_min, y_max, x_max = box
    
    # Calculate width and height of the box
    width = x_max - x_min
    height = y_max - y_min

    ax.add_patch(plt.Rectangle((x_min, y_min), width, height, edgecolor=color, facecolor=(0,0,0,0), lw=2))

def show_mask(contour, ax, random_color=False):
    if random_color:
        color = np.random.random(3)
    else:
        color = np.array([30/255, 184/255, 255/255])
    edgecolor = np.concatenate([color, np.array([1.0])], axis=0)
    facecolor = np.concatenate([color, np.array([0.4])], axis=0)

    contour = gpd.GeoSeries(contour)
    contour.plot(edgecolor=edgecolor, facecolor=facecolor, ax=ax)

def visualize(im, contours, boxes, random_color=True, box_color='red', format='yxyx_px', ax=None, **fig_kwargs):
    # Format boxes
    H, W = im.shape[:2]

    if format=='cycxhw_01':
        # boxes are in cycxhw format normalized by the longest image side. Converts to min/max coordinates in pixels.
        original_boxes = cxcywh_to_xyxy(boxes) * max(H, W)
    elif format=='cycxhw_px':
        # boxes are in cycxhw format in pixels. Converts to min/max coordinates in pixels.
        original_boxes = cxcywh_to_xyxy(boxes)
    elif format=='yxyx_01':
        # boxes are in min/max coordinates normalized by the longest image side. Converts to min/max coordinates in pixels.
        original_boxes = boxes * max(H, W)
    elif format=='yxyx_px':
        # boxes are in min/max coordinates in pixels. No conversion necessary.
        original_boxes = boxes
    else:
        raise ValueError(format)

    if ax is None:
        fig, ax = plt.subplots(1, 1, **fig_kwargs)
    
    ax.imshow(im)
    ax.axis(False)
    for i in range(original_boxes.shape[0]):
        show_mask(contours[i], ax, random_color=random_color)
        show_box(original_boxes[i], ax, color=box_color)

def convert_to_dicts(pred_list, bboxes=False):
    ''' Convert to dict format which will be used when computing metric. '''
    pred_dict = {}
    for idx, item in enumerate(pred_list):
        if bboxes:
            y1, x1, y2, x2 = item
            pred_dict[idx] = [y1.item(), x1.item(), y2.item(), x2.item()]
        else: pred_dict[idx] = item.item()
    return str(pred_dict)

def get_unique_id_from_img_path(image_path):
    """ Take the image path and create a unique ID from that in the form EXPTYPE_PLATE_IMG. """
    return '_'.join(image_path.as_posix().split('/')[-4:-1])



if __name__=='__main__':
    # Configuration

    # Parameters
    thresholds = [0.975]
    iou_thres = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]

    # use_fixed_patch_for_organoID = False  # If true, uses 4 patches per image and adjusts their size correspondingly.
    fixed_patch_size = (512, 2048)  # If None, uses 4 patches per image and adjusts their size correspondingly.
    evaluate_with_stitching = False  # should be False with the new training
    evaluate_segmentation = False  # True  # can be False with the new training, might speed up evaluation?
    # save_below_AP = 0.85

    # # Metrics
    # mAP_metric = torchmetrics.detection.MeanAveragePrecision(class_metrics=True, extended_summary=False, backend='faster_coco_eval')
    # mAP_metric.warn_on_many_detections = False

    for model_name, sam_version, logging_name in [
        # ('SSD_objects_2024', 'sam1', 'SSD_default_MultiOrg_train_normal_MultiOrg_train_macros_OrgaSegment_train_Tellu_train_OrgaQuant_train_20022025_baseline_ssd_sampling_n_objects_True_8_200_2024'),
        # ('SSD_objects_2025', 'sam1', 'SSD_default_MultiOrg_train_normal_MultiOrg_train_macros_OrgaSegment_train_Tellu_train_OrgaQuant_train_20022025_baseline_ssd_sampling_n_objects_True_8_200_2025'),
        # ('SSD_objects_2026', 'sam1', 'SSD_default_MultiOrg_train_normal_MultiOrg_train_macros_OrgaSegment_train_Tellu_train_OrgaQuant_train_20022025_baseline_ssd_sampling_n_objects_True_8_200_2026'),
        
        ('SSD_objects_dataset_2024', 'sam1', 'SSD_default_MultiOrg_train_normal_MultiOrg_train_macros_OrgaSegment_train_Tellu_train_OrgaQuant_train_20022025_baseline_ssd_sampling_n_objects_dataset_True_8_200_2024'),
        ('SSD_objects_dataset_2025', 'sam1', 'SSD_default_MultiOrg_train_normal_MultiOrg_train_macros_OrgaSegment_train_Tellu_train_OrgaQuant_train_20022025_baseline_ssd_sampling_n_objects_dataset_True_8_200_2025'),
        ('SSD_objects_dataset_2026', 'sam1', 'SSD_default_MultiOrg_train_normal_MultiOrg_train_macros_OrgaSegment_train_Tellu_train_OrgaQuant_train_20022025_baseline_ssd_sampling_n_objects_dataset_True_8_200_2026'),
        
        # ('SSD_objects_dataset_patchsize_2024', 'sam1', 'SSD_default_MultiOrg_train_normal_MultiOrg_train_macros_OrgaSegment_train_Tellu_train_OrgaQuant_train_20022025_baseline_ssd_sampling_n_objects_dataset_patch_size_True_8_200_2024'),
        # ('SSD_objects_dataset_patchsize_2025', 'sam1', 'SSD_default_MultiOrg_train_normal_MultiOrg_train_macros_OrgaSegment_train_Tellu_train_OrgaQuant_train_20022025_baseline_ssd_sampling_n_objects_dataset_patch_size_True_8_200_2025'),
        # ('SSD_objects_dataset_patchsize_2026', 'sam1', 'SSD_default_MultiOrg_train_normal_MultiOrg_train_macros_OrgaSegment_train_Tellu_train_OrgaQuant_train_20022025_baseline_ssd_sampling_n_objects_dataset_patch_size_True_8_200_2026'),
        
        # ('FRCNNv2_bs5_objects_2024', 'sam1', 'FRCNNv2_default_MultiOrg_train_normal_MultiOrg_train_macros_OrgaSegment_train_Tellu_train_OrgaQuant_train_20022025_baseline_frcnn_v2_sampling_n_objects_5_batches_per_epoch_True_8_5_2024'),
        # ('FRCNNv2_bs5_objects_2025', 'sam1', 'FRCNNv2_default_MultiOrg_train_normal_MultiOrg_train_macros_OrgaSegment_train_Tellu_train_OrgaQuant_train_20022025_baseline_frcnn_v2_sampling_n_objects_5_batches_per_epoch_True_8_5_2025'),
        # ('FRCNNv2_bs5_objects_2026', 'sam1', 'FRCNNv2_default_MultiOrg_train_normal_MultiOrg_train_macros_OrgaSegment_train_Tellu_train_OrgaQuant_train_20022025_baseline_frcnn_v2_sampling_n_objects_5_batches_per_epoch_True_8_5_2026'),
        
        # ('FRCNNv2_bs10_objects_2024', 'sam1', 'FRCNNv2_default_MultiOrg_train_normal_MultiOrg_train_macros_OrgaSegment_train_Tellu_train_OrgaQuant_train_20022025_baseline_frcnn_v2_sampling_n_objects_10_batches_per_epoch_True_8_10_2024'),
        # ('FRCNNv2_bs10_objects_2025', 'sam1', 'FRCNNv2_default_MultiOrg_train_normal_MultiOrg_train_macros_OrgaSegment_train_Tellu_train_OrgaQuant_train_20022025_baseline_frcnn_v2_sampling_n_objects_10_batches_per_epoch_True_8_10_2025'),
        # ('FRCNNv2_bs10_objects_2026', 'sam1', 'FRCNNv2_default_MultiOrg_train_normal_MultiOrg_train_macros_OrgaSegment_train_Tellu_train_OrgaQuant_train_20022025_baseline_frcnn_v2_sampling_n_objects_10_batches_per_epoch_True_8_10_2026'),
        
        # ('FRCNNv2_bs20_objects_2024', 'sam1', 'FRCNNv2_default_MultiOrg_train_normal_MultiOrg_train_macros_OrgaSegment_train_Tellu_train_OrgaQuant_train_20022025_baseline_frcnn_v2_sampling_n_objects_20_batches_per_epoch_True_8_20_2024'),
        # ('FRCNNv2_bs20_objects_2025', 'sam1', 'FRCNNv2_default_MultiOrg_train_normal_MultiOrg_train_macros_OrgaSegment_train_Tellu_train_OrgaQuant_train_20022025_baseline_frcnn_v2_sampling_n_objects_20_batches_per_epoch_True_8_20_2025'),
        # ('FRCNNv2_bs20_objects_2026', 'sam1', 'FRCNNv2_default_MultiOrg_train_normal_MultiOrg_train_macros_OrgaSegment_train_Tellu_train_OrgaQuant_train_20022025_baseline_frcnn_v2_sampling_n_objects_20_batches_per_epoch_True_8_20_2026'),

        # ('FRCNNv2_bs50_objects_2024', 'sam1', 'FRCNNv2_default_MultiOrg_train_normal_MultiOrg_train_macros_OrgaSegment_train_Tellu_train_OrgaQuant_train_20022025_baseline_frcnn_v2_sampling_n_objects_50_batches_per_epoch_True_8_50_2024'),
        # ('FRCNNv2_bs50_objects_2025', 'sam1', 'FRCNNv2_default_MultiOrg_train_normal_MultiOrg_train_macros_OrgaSegment_train_Tellu_train_OrgaQuant_train_20022025_baseline_frcnn_v2_sampling_n_objects_50_batches_per_epoch_True_8_50_2025'),
        # ('FRCNNv2_bs50_objects_2026', 'sam1', 'FRCNNv2_default_MultiOrg_train_normal_MultiOrg_train_macros_OrgaSegment_train_Tellu_train_OrgaQuant_train_20022025_baseline_frcnn_v2_sampling_n_objects_50_batches_per_epoch_True_8_50_2026'),

        ('FRCNNv2_bs50_objects_dataset_2024', 'sam1', 'FRCNNv2_default_MultiOrg_train_normal_MultiOrg_train_macros_OrgaSegment_train_Tellu_train_OrgaQuant_train_20022025_baseline_frcnn_v2_sampling_n_objects_dataset_50_batches_per_epoch_True_8_50_2024'),
        ('FRCNNv2_bs50_objects_dataset_2025', 'sam1', 'FRCNNv2_default_MultiOrg_train_normal_MultiOrg_train_macros_OrgaSegment_train_Tellu_train_OrgaQuant_train_20022025_baseline_frcnn_v2_sampling_n_objects_dataset_50_batches_per_epoch_True_8_50_2025'),
        ('FRCNNv2_bs50_objects_dataset_2026', 'sam1', 'FRCNNv2_default_MultiOrg_train_normal_MultiOrg_train_macros_OrgaSegment_train_Tellu_train_OrgaQuant_train_20022025_baseline_frcnn_v2_sampling_n_objects_dataset_50_batches_per_epoch_True_8_50_2026'),

        # ('FRCNNv2_bs50_objects_dataset_patchsize_2024', 'sam1', 'FRCNNv2_default_MultiOrg_train_normal_MultiOrg_train_macros_OrgaSegment_train_Tellu_train_OrgaQuant_train_20022025_baseline_frcnn_v2_sampling_n_objects_dataset_patch_size_50_batches_per_epoch_True_8_50_2024'),
        # ('FRCNNv2_bs50_objects_dataset_patchsize_2025', 'sam1', 'FRCNNv2_default_MultiOrg_train_normal_MultiOrg_train_macros_OrgaSegment_train_Tellu_train_OrgaQuant_train_20022025_baseline_frcnn_v2_sampling_n_objects_dataset_patch_size_50_batches_per_epoch_True_8_50_2025'),
        # ('FRCNNv2_bs50_objects_dataset_patchsize_2026', 'sam1', 'FRCNNv2_default_MultiOrg_train_normal_MultiOrg_train_macros_OrgaSegment_train_Tellu_train_OrgaQuant_train_20022025_baseline_frcnn_v2_sampling_n_objects_dataset_patch_size_50_batches_per_epoch_True_8_50_2026'),

        # ('FRCNNv2_bs100_objects_2024', 'sam1', 'FRCNNv2_default_MultiOrg_train_normal_MultiOrg_train_macros_OrgaSegment_train_Tellu_train_OrgaQuant_train_20022025_baseline_frcnn_v2_sampling_n_objects_100_batches_per_epoch_True_8_100_2024'),

        ('FRCNNv1_bs50_objects_dataset_2024', 'sam1', 'FRCNN_default_MultiOrg_train_normal_MultiOrg_train_macros_OrgaSegment_train_Tellu_train_OrgaQuant_train_20022025_baseline_frcnn_sampling_n_objects_dataset_True_8_50_2024'),
        ('FRCNNv1_bs50_objects_dataset_2025', 'sam1', 'FRCNN_default_MultiOrg_train_normal_MultiOrg_train_macros_OrgaSegment_train_Tellu_train_OrgaQuant_train_20022025_baseline_frcnn_sampling_n_objects_dataset_True_8_50_2025'),
        # ('FRCNNv1_bs50_objects_dataset_2026', 'sam1', 'FRCNN_default_MultiOrg_train_normal_MultiOrg_train_macros_OrgaSegment_train_Tellu_train_OrgaQuant_train_20022025_baseline_frcnn_sampling_n_objects_dataset_True_8_50_2026'),
        
        ('pretrained_sam1_small_stage1', 'sam1', 'DETR_own_implementation_SAM_large_open_images_v4_5_original_data_NeurIPSCellSeg_train_DETR_own_SAM1_small_head_pretraining_True_4_400_2024'),
    ]:
        model_name = f"{model_name}_{sam_version}"
        model = PredictionSAM(logging_name=logging_name, sam_version=sam_version)
        for ds_idx, ds in enumerate([
            dl.OrganoID(split='test'),
            dl.OrganoID(split='test_C'),
            dl.OrganoID(split='test_Lung'),
            dl.OrganoID(split='test_ACC'),
            dl.OrganoID(split='test_only_mouse'),
            dl.OrgaExtractor(split='all'),
            dl.NewData(split='all'),
            dl.OrgaSegment(split='test'),
            dl.OrgaQuant(split='test'),
            dl.Tellu(split='test'),
            dl.MultiOrg(split='test_macros'),
            dl.MultiOrg(split='test_normal'),
        ]):
            print(f'\n\n\n reached dataset {str(ds)} {ds.split} ====== \n\n\n')
            # if (results_dir / f'{model_name}' / f'mean_detection_AP_{str(ds)}_{ds.split}.csv').exists():
            #     continue


            detection_mAP = []
            for idx in tqdm(range(len(ds))):
                im, gt_mask, gt_boxes, im_path, im_ID = ds[idx]
                # im, flatfield = dl.normalize(im)
                print('gt_boxes.shape', gt_boxes.shape)

                # Prediction with optimal threshold
                contours, boxes, scores = model.forward(f'{str(ds)}_{ds.split}', im_ID, 
                                                        predict_masks=False,
                                                        min_diameter=30/1.29 if str(ds).startswith('MultiOrg') else 10)
                
                # Sets no threshold for computing the mAP. 
                if not evaluate_with_stitching:
                    contours, boxes, scores = model.set_threshold(conf_thres=0.0, 
                                                                predict_masks=evaluate_segmentation or evaluate_with_stitching)

                # Detection
                iou_matrix = pp.compute_iou_matrix_detection(boxes, gt_boxes)
                mAP_scores, pq_scores, iou_scores, dice_scores, f1_scores, prec_scores, recall_scores = pp.compute_metrics_detection_all(
                    iou_matrix, iou_thres, scores, thresholds
                )
                detection_mAP.append(mAP_scores)

            (results_dir / f'{model_name}').mkdir(exist_ok=True)

            detection_mAP = pd.DataFrame(data=detection_mAP, columns=iou_thres)
            detection_mAP.to_csv(results_dir / f'{model_name}' / f'detection_AP_{str(ds)}_{ds.split}.csv', index=False)
            detection_mAP.mean().to_csv(results_dir / f'{model_name}' / f'mean_detection_AP_{str(ds)}_{ds.split}.csv', index=False)

