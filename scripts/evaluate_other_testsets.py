import io
import numpy as np
import cv2
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from copy import deepcopy
import skimage
from skimage.io import imread
import scipy
import basicpy
import tifffile
from tqdm import tqdm
# from descartes import PolygonPatch
import geopandas as gpd
import basicpy

import sys
sys.path.append('/home/icb/lion.gleiter/projects/organoid_sam/SAM_with_Detection_Head')

from util.box_ops_numpy import mask_to_boxes, cxcywh_to_xyxy, xyxy_to_cxcywh, plot_boxes
from util import dataloading as dl
from util import postprocessing as pp
from util.samos import SAMOS
from util.ssd import SSDPredictor

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
        color = np.concatenate([np.random.random(3), np.array([0.4])], axis=0)
    else:
        color = np.array([30/255, 184/255, 255/255, 0.4])

    contour = gpd.GeoSeries(contour)
    contour.plot(color=color, ax=ax)

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
    # model_name = 'SAMOS_last'
    # samos = SAMOS(checkpoint_path='/ictstr01/groups/shared/users/lion.gleiter/organoid_sam/checkpoints_trained/DetectionHead_SAM_large_OrganoID_train_MultiOrg_train_macros_MultiOrg_train_normal_OrgaExtractor_train_OrgaQuant_train_OrgaSegment_train_Tellu_train_NewData_train_ablation_dataset_5_pre_OI_NeurIPS_True_32_200/DetectionHead_SAM_large_OrganoID_train_MultiOrg_train_macros_MultiOrg_train_normal_OrgaExtractor_train_OrgaQuant_train_OrgaSegment_train_Tellu_train_NewData_train_ablation_dataset_5_pre_OI_NeurIPS_True_32_200-last_epoch=499-val_loss=5.44.ckpt')

    model_name = 'retrained_last'
    samos = SAMOS(checkpoint_path='/ictstr01/groups/shared/users/lion.gleiter/organoid_sam/checkpoints_trained/DetectionHead_SAM_large_OrganoID_train_MultiOrg_train_macros_MultiOrg_train_normal_OrgaExtractor_train_OrgaQuant_train_OrgaSegment_train_Tellu_train_NewData_train_added_eval_True_32_200/DetectionHead_SAM_large_OrganoID_train_MultiOrg_train_macros_MultiOrg_train_normal_OrgaExtractor_train_OrgaQuant_train_OrgaSegment_train_Tellu_train_NewData_train_added_eval_True_32_200-last_epoch=799-val_loss=4.93.ckpt')
    
    # model_name = 'retrained_best'
    # samos = SAMOS(checkpoint_path='/ictstr01/groups/shared/users/lion.gleiter/organoid_sam/checkpoints_trained/DetectionHead_SAM_large_OrganoID_train_MultiOrg_train_macros_MultiOrg_train_normal_OrgaExtractor_train_OrgaQuant_train_OrgaSegment_train_Tellu_train_NewData_train_added_eval_True_32_200/DetectionHead_SAM_large_OrganoID_train_MultiOrg_train_macros_MultiOrg_train_normal_OrgaExtractor_train_OrgaQuant_train_OrgaSegment_train_Tellu_train_NewData_train_added_eval_True_32_200-best_epoch=649-val_loss=4.83.ckpt')
    
    # model_name = 'SSD_full_best'
    # samos = SSDPredictor('/ictstr01/groups/shared/users/lion.gleiter/organoid_sam/checkpoints_trained/SSD/SSD_OrganoID_train_MultiOrg_train_macros_MultiOrg_train_normal_OrgaExtractor_train_OrgaQuant_train_OrgaSegment_train_Tellu_train_NewData_train_full_data_09012025_True_32_200/SSD_OrganoID_train_MultiOrg_train_macros_MultiOrg_train_normal_OrgaExtractor_train_OrgaQuant_train_OrgaSegment_train_Tellu_train_NewData_train_full_data_09012025_True_32_200-best_epoch=274-val_loss=2.81.ckpt')

    # model_name = 'SSD_full_last_better_thres'
    # samos = SSDPredictor('/ictstr01/groups/shared/users/lion.gleiter/organoid_sam/checkpoints_trained/SSD/SSD_OrganoID_train_MultiOrg_train_macros_MultiOrg_train_normal_OrgaExtractor_train_OrgaQuant_train_OrgaSegment_train_Tellu_train_NewData_train_full_data_09012025_True_32_200/SSD_OrganoID_train_MultiOrg_train_macros_MultiOrg_train_normal_OrgaExtractor_train_OrgaQuant_train_OrgaSegment_train_Tellu_train_NewData_train_full_data_09012025_True_32_200-last_epoch=799-val_loss=2.85.ckpt')

    # thresholds = [0.025, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4]
    # thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    thresholds = [0.2, 0.4, 0.5, 0.6, 0.75, 0.85, 0.9, 0.95, 0.96, 0.975, 0.99]

    for ds in [
        dl.OrganoID(split='test'),
        dl.OrganoID(split='test_C'),
        dl.OrganoID(split='test_Lung'),
        dl.OrganoID(split='test_ACC'),
        dl.OrganoID(split='test_only_mouse'),
    ]:
        # ids = {f'{t:4.2f}': [] for t in thresholds}
        # bboxes_all = {f'{t:4.2f}': [] for t in thresholds}
        # scores_all = {f'{t:4.2f}': [] for t in thresholds}
        # labels_all = {f'{t:4.2f}': [] for t in thresholds}

        pq_data = []
        seg_data = []
        for idx in tqdm(range(len(ds))):  # len(ds))):
            im, gt_mask, gt_boxes, im_path, im_ID = ds[idx]
            im, flatfield = dl.normalize(im)
            # im = (im / im.max() * 255).astype(np.uint8)
            # im = np.stack((im, im, im), axis=2)

            H, W = im.shape[:2]
            patch_size = int(np.ceil(max(H, W) * 7 / 12))
            print(patch_size)

            contours, boxes, scores = samos.forward(im, patch_size=patch_size, predict_masks=True, min_diameter=30/1.29)
            for thres in thresholds:
                contours, boxes, scores = samos.set_threshold(conf_thres=thres, predict_masks=True)
                
                iou_matrix = pp.compute_iou_matrix_detection(boxes, gt_boxes)
                tp, fp, fn, pq, precision, recall, f1_score, mean_iou, dice = \
                    pp.compute_metrics_detection_from_iou_matrix(iou_matrix=iou_matrix)
                pq_data.append((f'{thres:4.2f}', pq, f1_score, precision, recall, mean_iou))
                
                iou_matrix = pp.compute_iou_matrix_segmentation_contours(contours, gt_masks=pp.convert_mask_to_binary(gt_mask))
                tp, fp, fn, pq, precision, recall, f1_score, mean_iou, dice = \
                    pp.compute_metrics_segmentation_from_iou_matrix(iou_matrix=iou_matrix)
                seg_data.append((f'{thres:4.2f}', pq, f1_score, precision, recall, mean_iou))


                fig, ax = plt.subplots(1, 1, figsize=(12*4, 12*4), dpi=200)
                plot_boxes(im, gt_boxes, format='yxyx_px', ax=ax, color='blue')
                plot_boxes(im, boxes, format='yxyx_px', ax=ax, show_image=False, color='red')
                plot_dir = results_dir / 'plots' / f'{str(ds)}_{model_name}'
                plot_dir.mkdir(exist_ok=True)
                plt.savefig(plot_dir / f'{str(ds)}_{ds.split}_{idx}_thres_{int(thres*100)}.png', dpi=200)
                plt.close('all')

                # unique_id = get_unique_id_from_img_path(im_path)

                # boxes = boxes.astype(int)
                # boxes = np.stack((boxes[:, 1], boxes[:, 0], boxes[:, 3], boxes[:, 2]), axis=1)

                # labels = np.zeros((scores.shape[0],), dtype=int)
                # boxes = convert_to_dicts(boxes, bboxes=True)
                # scores = convert_to_dicts(scores)
                # labels = convert_to_dicts(labels)
                
                # ids[f'{thres:4.2f}'].append(unique_id)
                # bboxes_all[f'{thres:4.2f}'].append(boxes)
                # scores_all[f'{thres:4.2f}'].append(scores)
                # labels_all[f'{thres:4.2f}'].append(labels)

        # for t in thresholds:    
        #     # create a data frame to save predictions
        #     df = pd.DataFrame(data={'ID': ids[f'{t:4.2f}'], 
        #                             'Predicted Boxes': bboxes_all[f'{t:4.2f}'], 
        #                             'Model scores': scores_all[f'{t:4.2f}'], 
        #                             'Predicted Labels': labels_all[f'{t:4.2f}']})
        #     df.to_csv(results_dir / f'{model_name}_{str(ds)}_t_{t:4.2f}_submission.csv', index=False)

        pq_data = pd.DataFrame(data=pq_data, columns=["thres", "pq", "f1_score", "precision", "recall", "iou"])
        pq_data.to_csv(results_dir / f'{model_name}_test_metrics_{str(ds)}_{ds.split}.csv', index=False)

        seg_data = pd.DataFrame(data=seg_data, columns=["thres", "pq", "f1_score", "precision", "recall", "iou"])
        seg_data.to_csv(results_dir / f'{model_name}_segmentation_test_metrics_{str(ds)}_{ds.split}.csv', index=False)

        mean_metrics = pq_data.groupby('thres', as_index=False).mean()
        mean_metrics.to_csv(results_dir / f'{model_name}_mean_test_metrics_{str(ds)}_{ds.split}.csv', index=False)

        mean_metrics = seg_data.groupby('thres', as_index=False).mean()
        mean_metrics.to_csv(results_dir / f'{model_name}_segmentation_mean_test_metrics_{str(ds)}_{ds.split}.csv', index=False)

        print(pq_data.groupby('thres').mean())



    for ds in [
        dl.OrgaSegment(split='test'),
        dl.OrgaQuant(split='test'),
        dl.OrgaExtractor(split='test'),
        dl.Tellu(split='test'),
        dl.NewData(split='test'),
    ]:
        # Fixed patch size of 512
        patch_size = 512
        print(patch_size)

        pq_data = []
        seg_data = []
        for idx in tqdm(range(len(ds))):  # len(ds))):
            im, gt_mask, gt_boxes, im_path, im_ID = ds[idx]
            im, flatfield = dl.normalize(im)
            # im = (im / im.max() * 255).astype(np.uint8)
            # im = np.stack((im, im, im), axis=2)

            H, W = im.shape[:2]

            contours, boxes, scores = samos.forward(im, patch_size=patch_size, predict_masks=True, min_diameter=30/1.29)
            for thres in thresholds:
                contours, boxes, scores = samos.set_threshold(conf_thres=thres, predict_masks=True)
                
                iou_matrix = pp.compute_iou_matrix_detection(boxes, gt_boxes)
                tp, fp, fn, pq, precision, recall, f1_score, mean_iou, dice = \
                    pp.compute_metrics_detection_from_iou_matrix(iou_matrix=iou_matrix)
                pq_data.append((f'{thres:4.2f}', pq, f1_score, precision, recall, mean_iou))
                
                if gt_mask is not None:
                    iou_matrix = pp.compute_iou_matrix_segmentation_contours(contours, gt_masks=pp.convert_mask_to_binary(gt_mask))
                    tp, fp, fn, pq, precision, recall, f1_score, mean_iou, dice = \
                        pp.compute_metrics_segmentation_from_iou_matrix(iou_matrix=iou_matrix)
                    seg_data.append((f'{thres:4.2f}', pq, f1_score, precision, recall, mean_iou))


                fig, ax = plt.subplots(1, 1, figsize=(12*4, 12*4), dpi=200)
                plot_boxes(im, gt_boxes, format='yxyx_px', ax=ax, color='blue')
                plot_boxes(im, boxes, format='yxyx_px', ax=ax, show_image=False, color='red')
                plot_dir = results_dir / 'plots' / f'{str(ds)}_{model_name}'
                plot_dir.mkdir(exist_ok=True)
                plt.savefig(plot_dir / f'{str(ds)}_{ds.split}_{idx}_thres_{int(thres*100)}_ps_{patch_size}.png', dpi=200)
                plt.close('all')

        pq_data = pd.DataFrame(data=pq_data, columns=["thres", "pq", "f1_score", "precision", "recall", "iou"])
        pq_data.to_csv(results_dir / f'{model_name}_test_metrics_{str(ds)}_{ds.split}_ps_{patch_size}.csv', index=False)
        
        mean_metrics = pq_data.groupby('thres', as_index=False).mean()
        mean_metrics.to_csv(results_dir / f'{model_name}_mean_test_metrics_{str(ds)}_{ds.split}_ps_{patch_size}.csv', index=False)

        if len(seg_data) > 0:
            seg_data = pd.DataFrame(data=seg_data, columns=["thres", "pq", "f1_score", "precision", "recall", "iou"])
            seg_data.to_csv(results_dir / f'{model_name}_segmentation_test_metrics_{str(ds)}_{ds.split}_ps_{patch_size}.csv', index=False)

            mean_metrics = seg_data.groupby('thres', as_index=False).mean()
            mean_metrics.to_csv(results_dir / f'{model_name}_segmentation_mean_test_metrics_{str(ds)}_{ds.split}_ps_{patch_size}.csv', index=False)

        print(pq_data.groupby('thres').mean())
        