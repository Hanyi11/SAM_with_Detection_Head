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
# from util.samos import SAMOS
# from util.ssd import SSDPredictor
# from util.FasterRCNN import FasterRCNNPredictor
# from util.cellpose import Cellpose
from util.umamba import Umamba

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

    # model_name = 'SAMOS_last'
    # model = SAMOS(checkpoint_path='/ictstr01/groups/shared/users/lion.gleiter/organoid_sam/checkpoints_trained/DetectionHead_SAM_large_OrganoID_train_MultiOrg_train_macros_MultiOrg_train_normal_OrgaExtractor_train_OrgaQuant_train_OrgaSegment_train_Tellu_train_NewData_train_ablation_dataset_5_pre_OI_NeurIPS_True_32_200/DetectionHead_SAM_large_OrganoID_train_MultiOrg_train_macros_MultiOrg_train_normal_OrgaExtractor_train_OrgaQuant_train_OrgaSegment_train_Tellu_train_NewData_train_ablation_dataset_5_pre_OI_NeurIPS_True_32_200-last_epoch=499-val_loss=5.44.ckpt')

    # model_name = 'retrained_last'
    # model = SAMOS(checkpoint_path='/ictstr01/groups/shared/users/lion.gleiter/organoid_sam/checkpoints_trained/DetectionHead_SAM_large_OrganoID_train_MultiOrg_train_macros_MultiOrg_train_normal_OrgaExtractor_train_OrgaQuant_train_OrgaSegment_train_Tellu_train_NewData_train_added_eval_True_32_200/DetectionHead_SAM_large_OrganoID_train_MultiOrg_train_macros_MultiOrg_train_normal_OrgaExtractor_train_OrgaQuant_train_OrgaSegment_train_Tellu_train_NewData_train_added_eval_True_32_200-last_epoch=799-val_loss=4.93.ckpt')
    
    # model_name = 'retrained_best'
    # model = SAMOS(checkpoint_path='/ictstr01/groups/shared/users/lion.gleiter/organoid_sam/checkpoints_trained/DetectionHead_SAM_large_OrganoID_train_MultiOrg_train_macros_MultiOrg_train_normal_OrgaExtractor_train_OrgaQuant_train_OrgaSegment_train_Tellu_train_NewData_train_added_eval_True_32_200/DetectionHead_SAM_large_OrganoID_train_MultiOrg_train_macros_MultiOrg_train_normal_OrgaExtractor_train_OrgaQuant_train_OrgaSegment_train_Tellu_train_NewData_train_added_eval_True_32_200-best_epoch=649-val_loss=4.83.ckpt')
    
    # model_name = 'SSD_full_best'
    # model = SSDPredictor('/ictstr01/groups/shared/users/lion.gleiter/organoid_sam/checkpoints_trained/SSD/SSD_OrganoID_train_MultiOrg_train_macros_MultiOrg_train_normal_OrgaExtractor_train_OrgaQuant_train_OrgaSegment_train_Tellu_train_NewData_train_full_data_09012025_True_32_200/SSD_OrganoID_train_MultiOrg_train_macros_MultiOrg_train_normal_OrgaExtractor_train_OrgaQuant_train_OrgaSegment_train_Tellu_train_NewData_train_full_data_09012025_True_32_200-best_epoch=274-val_loss=2.81.ckpt')

    # model_name = 'SSD_full_last_better_thres'
    # model = SSDPredictor('/ictstr01/groups/shared/users/lion.gleiter/organoid_sam/checkpoints_trained/SSD/SSD_OrganoID_train_MultiOrg_train_macros_MultiOrg_train_normal_OrgaExtractor_train_OrgaQuant_train_OrgaSegment_train_Tellu_train_NewData_train_full_data_09012025_True_32_200/SSD_OrganoID_train_MultiOrg_train_macros_MultiOrg_train_normal_OrgaExtractor_train_OrgaQuant_train_OrgaSegment_train_Tellu_train_NewData_train_full_data_09012025_True_32_200-last_epoch=799-val_loss=2.85.ckpt')

    # model_name = 'SSD_multiorg_last_multiscale'
    # model = SSDPredictor('/ictstr01/groups/shared/users/lion.gleiter/organoid_sam/checkpoints_trained/SSD/SSD_MultiOrg_train_macros_MultiOrg_train_normal_multiorg_with_090_overlap_True_32_200/SSD_MultiOrg_train_macros_MultiOrg_train_normal_multiorg_with_090_overlap_True_32_200-last_epoch=799-val_loss=3.13.ckpt')

    # model_name = 'legacy_FasterRCNN_v2_other_testsets_batch16_last'
    # model = FasterRCNNPredictor(checkpoint_path='/ictstr01/groups/shared/users/lion.gleiter/organoid_sam/checkpoints_trained/FasterRCNN/FasterRCNNv2_OrganoID_train_MultiOrg_train_macros_MultiOrg_train_normal_OrgaExtractor_train_OrgaQuant_train_OrgaSegment_train_Tellu_train_NewData_train_FasterRCNN_16012025_True_16_200/FasterRCNNv2_OrganoID_train_MultiOrg_train_macros_MultiOrg_train_normal_OrgaExtractor_train_OrgaQuant_train_OrgaSegment_train_Tellu_train_NewData_train_FasterRCNN_16012025_True_16_200-last_epoch=649-val_loss=0.28.ckpt',
    #                             version_FasterRCNN='v2')
    
    

    # Parameters
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.975]
    iou_thres = [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]

    # use_fixed_patch_for_organoID = False  # If true, uses 4 patches per image and adjusts their size correspondingly.
    fixed_patch_size = (1024, 2048)  # If None, uses 4 patches per image and adjusts their size correspondingly.
    evaluate_with_stitching = False  # should be False with the new training
    evaluate_segmentation = True  # can be False with the new training, might speed up evaluation?
    save_below_AP = 0.85

    # Metrics
    mAP_metric = torchmetrics.detection.MeanAveragePrecision(class_metrics=True, extended_summary=False, backend='faster_coco_eval')
    mAP_metric.warn_on_many_detections = False

    for model_name, model in [
        ('umamba_enc', Umamba('enc')),
        ('umamba_bot', Umamba('bot')),
    ]:
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
            if (results_dir / f'{model_name}' / f'mean_segmentation_metrics_{str(ds)}_{ds.split}.csv').exists():
                continue

            detection_mAP = []
            segmentation_mAP = []
            detection_metrics = []
            segmentation_metrics = []
            det_data = []
            seg_data = []
            mAP_metric.reset()
            for idx in tqdm(range(len(ds))):
                im, gt_mask, gt_boxes, im_path, im_ID = ds[idx]
                im, flatfield = dl.normalize(im)

                if fixed_patch_size is None:
                    H, W = im.shape[:2]
                    patch_size = int(np.ceil(max(H, W) * 7 / 12))
                    print(patch_size)
                else:
                    patch_size = fixed_patch_size

                # Prediction with optimal threshold
                contours, boxes, scores = model.forward(ds, idx,
                                                        patch_size=patch_size, 
                                                        predict_masks=evaluate_segmentation or evaluate_with_stitching,
                                                        min_diameter=30/1.29 if str(ds).startswith('MultiOrg') else 10)
                
                # Sets no threshold for computing the mAP. 
                if not evaluate_with_stitching:
                    contours, boxes, scores = model.set_threshold(conf_thres=0.0, 
                                                                predict_masks=evaluate_segmentation or evaluate_with_stitching)

                # Detection
                mAP_metric.update(preds=[{'boxes': torch.from_numpy(boxes), 
                                        'scores': torch.from_numpy(scores), 
                                        'labels': torch.zeros(scores.shape, dtype=torch.int)}], 
                                target=[{'boxes': torch.from_numpy(gt_boxes), 
                                        'labels': torch.zeros((gt_boxes.shape[0],), dtype=torch.int)}])
                iou_matrix = pp.compute_iou_matrix_detection(boxes, gt_boxes)
                mAP_scores, pq_scores, iou_scores, dice_scores, f1_scores, prec_scores, recall_scores = pp.compute_metrics_detection_all(
                    iou_matrix, iou_thres, scores, thresholds
                )
                detection_mAP.append(mAP_scores)
                detection_metrics.append(pd.DataFrame({
                    'thres': map(lambda x: f'{x:.3f}', thresholds), 
                    'pq': pq_scores,
                    'iou': iou_scores,
                    'dice': dice_scores,
                    'f1': f1_scores,
                    'precision': prec_scores,
                    'recall': recall_scores, 
                }))
                
                # Visualize with threshold 0.5
                if mAP_scores[0] < save_below_AP:
                    fig, ax = plt.subplots(1, 1, figsize=(12*4, 12*4), dpi=200)
                    plot_boxes(im, gt_boxes, format='yxyx_px', ax=ax, color='blue')
                    plot_boxes(im, boxes[scores>0.5], format='yxyx_px', ax=ax, show_image=False, color='red')
                    plot_dir = results_dir / 'plots' / f'{str(ds)}_{model_name}_{iou_thres[0]}'
                    results_dir.mkdir(exist_ok=True)
                    (results_dir / 'plots').mkdir(exist_ok=True)
                    plot_dir.mkdir(exist_ok=True)
                    plt.savefig(plot_dir / f'{str(ds)}_{ds.split}_{idx}_thres_50_ap50_{mAP_scores[0]:.3f}.png', dpi=200)
                    plt.close('all')


                # Segmentation
                if (gt_mask is not None) and evaluate_segmentation:
                    gt_masks = pp.convert_mask_to_binary(gt_mask)
                    iou_matrix_seg = pp.compute_iou_matrix_segmentation_contours(contours, gt_masks=gt_masks)
                    mAP_scores, hausdorff_scores, hausdorff_95_scores, masd_scores, assd_scores, iou_scores_no_thres, dice_scores_no_thres, pq_scores, iou_scores, dice_scores, f1_scores, prec_scores, recall_scores = pp.compute_metrics_segmentation_all(
                        iou_matrix_seg, iou_thres, scores, thresholds, gt_masks, contours
                    )

                    segmentation_mAP.append(pd.DataFrame({
                        'iou_thres': map(lambda x: f'{x:.3f}', iou_thres), 
                        'AP': mAP_scores, 
                        'hausdorff_scores': hausdorff_scores, 
                        'hausdorff_95_scores': hausdorff_95_scores, 
                        'masd_scores': masd_scores, 
                        'assd_scores': assd_scores,
                        'iou_scores_no_thres': iou_scores_no_thres,
                        'dice_scores_no_thres': dice_scores_no_thres,
                    }))
                    segmentation_metrics.append(pd.DataFrame({
                        'thres': map(lambda x: f'{x:.3f}', thresholds), 
                        'pq': pq_scores,
                        'iou': iou_scores,
                        'dice': dice_scores,
                        'f1': f1_scores,
                        'precision': prec_scores,
                        'recall': recall_scores, 
                    }))


                if evaluate_with_stitching:
                    for thres in thresholds:
                        contours, boxes, scores = model.set_threshold(conf_thres=thres, predict_masks=True)
                        
                        iou_matrix = pp.compute_iou_matrix_detection(boxes, gt_boxes)
                        tp, fp, fn, pq, precision, recall, f1_score, mean_iou, dice = \
                            pp.compute_metrics_detection_from_iou_matrix(iou_matrix=iou_matrix)
                        det_data.append((f'{thres:4.2f}', pq, f1_score, precision, recall, mean_iou))
                        
                        if (gt_mask is not None) and evaluate_segmentation:
                            iou_matrix = pp.compute_iou_matrix_segmentation_contours(contours, gt_masks=pp.convert_mask_to_binary(gt_mask))
                            tp, fp, fn, pq, precision, recall, f1_score, mean_iou, dice = \
                                pp.compute_metrics_segmentation_from_iou_matrix(iou_matrix=iou_matrix)
                            seg_data.append((f'{thres:4.2f}', pq, f1_score, precision, recall, mean_iou))


            (results_dir / f'{model_name}').mkdir(exist_ok=True)

            det_data = pd.DataFrame(data=det_data, columns=["thres", "pq", "f1_score", "precision", "recall", "iou"])
            det_data.to_csv(results_dir / f'{model_name}' / f'stiched_detection_metrics_{str(ds)}_{ds.split}.csv', index=False)

            seg_data = pd.DataFrame(data=seg_data, columns=["thres", "pq", "f1_score", "precision", "recall", "iou"])
            seg_data.to_csv(results_dir / f'{model_name}' / f'stiched_segmentation_metrics_{str(ds)}_{ds.split}.csv', index=False)

            mean_metrics = det_data.groupby('thres', as_index=False).mean()
            mean_metrics.to_csv(results_dir / f'{model_name}' / f'stiched_detection_mean_metrics_{str(ds)}_{ds.split}.csv', index=False)

            mean_metrics = seg_data.groupby('thres', as_index=False).mean()
            mean_metrics.to_csv(results_dir / f'{model_name}' / f'stiched_segmentation_mean_metrics_{str(ds)}_{ds.split}.csv', index=False)

            print('With stitching:\n', det_data.groupby('thres').mean())

            detection_mAP = pd.DataFrame(data=detection_mAP, columns=iou_thres)
            detection_mAP.to_csv(results_dir / f'{model_name}' / f'detection_AP_{str(ds)}_{ds.split}.csv', index=False)
            detection_mAP.mean().to_csv(results_dir / f'{model_name}' / f'mean_detection_AP_{str(ds)}_{ds.split}.csv', index=False)

            detection_metrics = pd.concat(detection_metrics, axis=0)
            detection_metrics.to_csv(results_dir / f'{model_name}' / f'detection_metrics_{str(ds)}_{ds.split}.csv', index=False)
            mean_metrics = detection_metrics.groupby('thres', as_index=False).mean()
            mean_metrics.to_csv(results_dir / f'{model_name}' / f'mean_detection_metrics_{str(ds)}_{ds.split}.csv', index=False)

            res = mAP_metric.compute()
            mAP_metric.reset()
            with open(results_dir / f'{model_name}' / f'mean_detection_torch_mAP_{str(ds)}_{ds.split}.csv', 'w') as f:
                json.dump({k: v.item() for k, v in res.items()}, f)

            if len(segmentation_mAP) > 0:
                segmentation_mAP = pd.concat(segmentation_mAP, axis=0)
                segmentation_mAP.to_csv(results_dir / f'{model_name}' / f'segmentation_AP_{str(ds)}_{ds.split}.csv', index=False)
                mean_metrics = segmentation_mAP.groupby('iou_thres', as_index=False).mean()
                mean_metrics.to_csv(results_dir / f'{model_name}' / f'mean_segmentation_AP_{str(ds)}_{ds.split}.csv', index=False)

                segmentation_metrics = pd.concat(segmentation_metrics, axis=0)
                segmentation_metrics.to_csv(results_dir / f'{model_name}' / f'segmentation_metrics_{str(ds)}_{ds.split}.csv', index=False)
                mean_metrics = segmentation_metrics.groupby('thres', as_index=False).mean()
                mean_metrics.to_csv(results_dir / f'{model_name}' / f'mean_segmentation_metrics_{str(ds)}_{ds.split}.csv', index=False)

