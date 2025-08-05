from pathlib import Path
import cv2
import numpy as np
import torch
from torch import nn
import pytorch_lightning as pl
import matplotlib.pyplot as plt

from torchmetrics.detection import GeneralizedIntersectionOverUnion, MeanAveragePrecision
import wandb

from ..util.box_ops_numpy import plot_boxes


# def show_box(box, ax, color='red'):
#     y_min, x_min, y_max, x_max = box
    
#     # Calculate width and height of the box
#     width = x_max - x_min
#     height = y_max - y_min

#     ax.add_patch(plt.Rectangle((x_min, y_min), width, height, edgecolor=color, facecolor=(0,0,0,0), lw=2))



class TrainingModule(pl.LightningModule):
    def __init__(self, 
                 model,
                 logging_name,
                 val_dirs,
                 learning_rate: float = 1e-4, 
                 weight_decay: float = 1e-4,
                 lr_drop: float = 200,
                 max_epochs: int = 500,
                 **kwargs
                ):
        super().__init__()
        self.save_hyperparameters()
        self.model = model
        self.current_test_set_name = None
        self.logging_name = logging_name
        self.val_dirs = val_dirs
        self.output_dir_base = Path('/ictstr01/groups/shared/users/lion.gleiter/organoid_sam/testset_predictions/trained') / self.logging_name

        # Parameters for optimizer and scheduler
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.lr_drop = lr_drop
        
        self.max_epochs = max_epochs

        self.training_step_outputs = []
        self.val_step_outputs = []

        self.val_giou_metrics = nn.ModuleList([
            GeneralizedIntersectionOverUnion() for _ in self.val_dirs
        ])
        self.val_map_metrics = nn.ModuleList([
            MeanAveragePrecision(extended_summary=False, backend='faster_coco_eval') for _ in self.val_dirs
        ])


    def configure_optimizers(self):
        n_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print("Num trainable params (total):", n_parameters)
        if isinstance(self.model.backbone, nn.Module):
            n_parameters_backbone = sum(p.numel() for p in self.model.backbone.parameters() if p.requires_grad)
            print("Num trainable params (backbone):", n_parameters_backbone)
        
        if hasattr(self.model, 'configure_optimizers'):
            return self.model.configure_optimizers()
        
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.lr_drop)
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        if batch_idx==0:
        # if True:
            images = batch[0]
            targets = batch[1]
            image = images[0].cpu().numpy().transpose((1, 2, 0))
            target = targets[0]
            with torch.no_grad():
                self.model.eval()
                pred = self.model.forward(images)
                self.model.train()
            if 'image_path' in target.keys():
                self.visualize_prediction(image,  # target['image_path'], 
                                          pred[0], 
                                          target['boxes'],
                                          f'example_pred@0.5/training')
        self.model.train()
        total_loss, loss_dict, metrics_dict = self.model.forward_train(batch)
        self.log('train_loss', total_loss, on_step=False, on_epoch=True)
        self.log_dict({f'train_{k}': v for k, v in loss_dict.items()}, on_step=False, on_epoch=True)
        self.log_dict({f'train_{k}': v for k, v in metrics_dict.items()}, on_step=False, on_epoch=True)

        
        # # if batch_idx==0:
        # if True:
        #     images = batch[0]
        #     targets = batch[1]
        #     with torch.no_grad():
        #         pred = self.model.forward(images)
        #     target = targets[0]
        #     if 'image_path' in target.keys():
        #         self.visualize_prediction(target['image_path'], pred[0], target['boxes'],
        #                                   f'example_pred@0.5/training')
        
        # self.training_step_outputs.append({'loss': total_loss, 'giou': metrics_dict['giou']}) #.detach().cpu()

        return {'loss': total_loss}

    def on_train_epoch_end(self):
        # avg_train_loss = torch.stack([x['loss'] for x in self.training_step_outputs]).mean()
        # avg_train_iou = torch.stack([x['giou'] for x in self.training_step_outputs]).mean()

        # self.log('train_loss', avg_train_loss)
        # self.log('train_giou', avg_train_iou)

        self.training_step_outputs.clear()
        torch.cuda.empty_cache()

    def visualize_prediction(self, im: np.ndarray,  # image_path: Path, 
                             pred, gt_boxes, name: str):
        # im = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        im = (im - im.min()) / (im.max() - im.min())
        H, W = im.shape[:2]
        # print('H, W', H, W)
        # im = cv2.resize(im, None, fx=1024/max(H,W), fy=1024/max(H,W), interpolation=cv2.INTER_LINEAR)

        boxes_pred = pred['boxes'].cpu().numpy().copy()
        scores_pred = pred['scores'].cpu().numpy().copy()
        gt_boxes = gt_boxes.cpu().numpy().copy()
        # print('gt_boxes', gt_boxes)
        # print('scores', scores_pred[:10])
        # print('boxes_pred', boxes_pred[:10])
        boxes_pred = boxes_pred[scores_pred>0.5]
        scores_pred = scores_pred[scores_pred>0.5]

        # Only keep top max_detections
        sorted_indices = np.argsort(scores_pred)[::-1]
        sorted_indices = sorted_indices[:100]

        boxes_pred = boxes_pred[sorted_indices]
        scores_pred = scores_pred[sorted_indices]
        # print('scores', scores_pred[:10])
        # print('boxes_pred', boxes_pred[:10])
        
        fig, ax = plt.subplots(1, 1, figsize=(12*4, 12*4), dpi=50)
        plot_boxes(im, gt_boxes, format='xyxy_px', ax=ax, show_image=True, color='blue', linewidth=15)
        plot_boxes(im, boxes_pred[:3], format='xyxy_px', ax=ax, show_image=False, color='red', linewidth=15)
        plot_boxes(im, boxes_pred[3:], format='xyxy_px', ax=ax, show_image=False, color='orange', linewidth=15)
        plt.axis(True)

        # Log with wandb
        wandb.log({name: wandb.Image(fig)})

        plt.close('all')


    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        self.model.eval()
        
        # Compute gIoU
        images = batch[0]
        targets = batch[1]
        pred = self.model.forward(images)  # Assumes output in the format [xyxy] in absolute pixel coordinates of the image coordinate system
        self.val_giou_metrics[dataloader_idx].update(pred, targets)
        self.val_map_metrics[dataloader_idx].update(pred, targets)

        if batch_idx==0:
            target = targets[0]
            image = images[0].cpu().numpy().transpose((1, 2, 0))
            if 'image_path' in target.keys():
                self.visualize_prediction(image,  # target['image_path'], 
                                          pred[0], target['boxes'],
                                          f'example_pred@0.5/{self.val_dirs[dataloader_idx]}')


        total_loss, loss_dict, metrics_dict = self.model.forward_eval(batch)

        # Metrics per validation set
        self.log(f'val_loss/{self.val_dirs[dataloader_idx]}', total_loss, on_step=False, on_epoch=True, add_dataloader_idx=False, prog_bar=False)
        self.log_dict({f'val_{k}/{self.val_dirs[dataloader_idx]}': v for k, v in loss_dict.items()}, on_step=False, on_epoch=True, add_dataloader_idx=False, prog_bar=False)
        self.log_dict({f'val_{k}/{self.val_dirs[dataloader_idx]}': v for k, v in metrics_dict.items()}, on_step=False, on_epoch=True, add_dataloader_idx=False, prog_bar=False)

        # Mean over all validation sets
        self.val_step_outputs.append({'loss': total_loss, 'giou': metrics_dict['giou']}) # .detach().cpu()

        return {'val_loss': total_loss}

    def on_validation_epoch_end(self):
        avg_val_loss = torch.stack([x['loss'] for x in self.val_step_outputs]).mean()
        avg_val_iou = torch.stack([x['giou'] for x in self.val_step_outputs]).mean()

        self.log('val_loss', avg_val_loss, prog_bar=False)
        self.log('val_giou', avg_val_iou, prog_bar=False)

        for i in range(len(self.val_giou_metrics)):
            metric = self.val_giou_metrics[i].compute()
            self.log(f'val_giou/{self.val_dirs[i]}', metric['giou'].cpu().item(), prog_bar=False)
            self.val_giou_metrics[i].reset()

            metric = self.val_map_metrics[i].compute()
            self.log(f'val_mAP/{self.val_dirs[i]}', metric['map'].cpu().item(), prog_bar=False)
            self.log(f'val_mAP@50/{self.val_dirs[i]}', metric['map_50'].cpu().item(), prog_bar=False)
            self.log(f'val_mAP@75/{self.val_dirs[i]}', metric['map_75'].cpu().item(), prog_bar=False)
            self.log(f'val_mAP_small/{self.val_dirs[i]}', metric['map_small'].cpu().item(), prog_bar=False)
            self.log(f'val_mAP_medium/{self.val_dirs[i]}', metric['map_medium'].cpu().item(), prog_bar=False)
            self.log(f'val_mAP_large/{self.val_dirs[i]}', metric['map_large'].cpu().item(), prog_bar=False)
            self.val_map_metrics[i].reset()

        self.val_step_outputs.clear()
        torch.cuda.empty_cache()

    # def on_test_epoch_start(self):
    #     # offsets_file = 
    #     # self.offsets = 
    #     return super().on_test_epoch_start()

    def test_step(self, batch, batch_idx):
        self.model.eval()
        images = batch[0]
        image_ids = batch[2]
        image_sizes = batch[4]
        patch_numbers = batch[5]
        preds = self.model.forward(images)
        for pred, size, img_id, patch_number in zip(preds, image_sizes, image_ids, patch_numbers):
            # Assumes output in the format [xyxy] in absolute pixel coordinates of the image coordinate system
            boxes = pred['boxes'].detach().cpu().numpy()
            scores = pred['scores'].detach().cpu().numpy()

            # Transform boxes to the original image scale
            max_side_length = 1024.0
            orig_H, orig_W = size
            boxes *= max(orig_H, orig_W) / max_side_length
            boxes = np.minimum(boxes, np.array([[orig_W, orig_H, orig_W, orig_H]]))

            # Save predictions
            scores_file = self.output_dir_base / self.current_test_set_name / img_id / f'patch_{patch_number}_scores.npy'
            boxes_file = self.output_dir_base / self.current_test_set_name / img_id / f'patch_{patch_number}_boxes.npy'
            assert not scores_file.exists(), scores_file
            assert not boxes_file.exists(), boxes_file
            scores_file.parent.mkdir(exist_ok=True, parents=True)
            np.save(scores_file, scores, allow_pickle=False)
            np.save(boxes_file, boxes, allow_pickle=False)

    def on_test_epoch_end(self):
        self.current_test_set_name = None
        torch.cuda.empty_cache()
        return super().on_test_epoch_end()
