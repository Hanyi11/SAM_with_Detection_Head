from pathlib import Path
import numpy as np
import torch
from torch import nn
import pytorch_lightning as pl


class TrainingModule(pl.LightningModule):
    def __init__(self, 
                 model,
                 logging_name,
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
        self.output_dir_base = Path('/ictstr01/groups/shared/users/lion.gleiter/organoid_sam/testset_predictions/trained') / self.logging_name

        # Parameters for optimizer and scheduler
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.lr_drop = lr_drop
        
        self.max_epochs = max_epochs

        self.training_step_outputs = []
        self.val_step_outputs = []


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
        self.model.train()
        total_loss, loss_dict, metrics_dict = self.model.forward_train(batch)
        self.log('train_loss', total_loss, on_step=False, on_epoch=True)
        self.log_dict({f'train_{k}': v for k, v in loss_dict.items()}, on_step=False, on_epoch=True)
        self.log_dict({f'train_{k}': v for k, v in metrics_dict.items()}, on_step=False, on_epoch=True)
        
        # self.training_step_outputs.append({'loss': total_loss, 'giou': metrics_dict['giou']}) #.detach().cpu()

        return {'loss': total_loss}

    def on_train_epoch_end(self):
        # avg_train_loss = torch.stack([x['loss'] for x in self.training_step_outputs]).mean()
        # avg_train_iou = torch.stack([x['giou'] for x in self.training_step_outputs]).mean()

        # self.log('train_loss', avg_train_loss)
        # self.log('train_giou', avg_train_iou)

        self.training_step_outputs.clear()
        torch.cuda.empty_cache()

    def validation_step(self, batch, batch_idx):
        self.model.eval()
        total_loss, loss_dict, metrics_dict = self.model.forward_eval(batch)

        # Metrics per validation set
        self.log('val_loss', total_loss, on_step=False, on_epoch=True, add_dataloader_idx=True)
        self.log_dict({f'val_{k}': v for k, v in loss_dict.items()}, on_step=False, on_epoch=True, add_dataloader_idx=True)
        self.log_dict({f'val_{k}': v for k, v in metrics_dict.items()}, on_step=False, on_epoch=True, add_dataloader_idx=True)

        # Mean over all validation sets
        self.val_step_outputs.append({'loss': total_loss, 'giou': metrics_dict['giou']}) # .detach().cpu()

        return {'val_loss': total_loss}

    def on_validation_epoch_end(self):
        avg_val_loss = torch.stack([x['loss'] for x in self.val_step_outputs]).mean()
        avg_val_iou = torch.stack([x['giou'] for x in self.val_step_outputs]).mean()

        self.log('val_loss', avg_val_loss)
        self.log('val_giou', avg_val_iou)

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
        for image, size, img_id, patch_number in zip(images, image_sizes, image_ids, patch_numbers):
            pred = self.model.forward([image])  # Assumes output in the format [xyxy] in absolute pixel coordinates of the image coordinate system

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
            np.save(scores_file, scores, allow_pickle=False)
            np.save(boxes_file, boxes, allow_pickle=False)

    def on_test_epoch_end(self):
        self.current_test_set_name = None
        torch.cuda.empty_cache()
        return super().on_test_epoch_end()
