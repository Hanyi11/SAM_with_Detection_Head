import torch
import pytorch_lightning as pl
from torch import nn, Tensor
from torchvision.models.detection import ssd300_vgg16
from torchvision.models.detection.ssd import SSD300_VGG16_Weights
from torchvision.models.vgg import VGG16_Weights
from transformer_layers import TransformerDecoder, MLP, PositionEmbeddingSine
from matcher import HungarianMatcher
from losses import SetCriterion

class SSD(nn.Module):
    def __init__(self,
                backbone = None,
                num_classes: int = 2,
                **kwargs):
        super().__init__()
        self.backbone = backbone

        # Define the SSD net
        self.model = ssd300_vgg16(weights_backbone=VGG16_Weights.IMAGENET1K_FEATURES,
                                  num_classes=num_classes)
        
    def forward(self, images, sizes=None, filter_class=None):
        detections = self.model(images)
        if filter_class is not None:
            out = []
            for detection in detections:
                boxes = detection['boxes']
                labels = detection['labels']
                scores = detection['scores']

                # Select only the results for one class (filter_class) 
                is_foreground = labels == filter_class
                boxes = boxes[is_foreground]
                labels = labels[is_foreground]
                scores = scores[is_foreground]
                out.append({'scores': scores, 'boxes': boxes, 'labels': labels})
            return out
   
        return detections

    def forward_train(self, batch):
        images, targets = batch[0], batch[1]
        device = images[0].device

        try:
            loss_dict: dict = self.model(images, targets)
        except Exception as e:
            print('targets', targets)
            raise e

        total_loss = sum(loss_dict[k] for k in loss_dict.keys())
        loss_dict.update({'loss': total_loss})

        # IoU
        # giou = self.criterion.compute_giou(outputs, processed_targets)

        # Move IoU to CPU for logging purposes
        # giou = giou.detach().cpu()

        return total_loss, loss_dict, {'giou': torch.tensor(0.0, device='cpu')}

    # def on_train_epoch_end(self):
    #     avg_train_loss = torch.stack([x['loss'] for x in self.training_step_outputs]).mean()
    #     avg_train_bbox_regression = torch.stack([x['bbox_regression'] for x in self.training_step_outputs]).mean()
    #     avg_train_classification = torch.stack([x['classification'] for x in self.training_step_outputs]).mean()
    #     # avg_train_iou = torch.stack([x['giou'] for x in self.training_step_outputs]).mean()

    #     self.log('train_loss', avg_train_loss)
    #     self.log('train_loss_bbox_regression', avg_train_bbox_regression)
    #     self.log('train_loss_classification', avg_train_classification)
    #     # self.log('train_giou', avg_train_iou)

    #     self.training_step_outputs.clear()

    def forward_eval(self, batch):
        images, targets = batch[0], batch[1]
        device = images[0].device

        training = self.model.training
        # print('training', training)  # False, needs to be True to compute the loss values
        self.model.training = True
        with torch.inference_mode():
            loss_dict = self.model(images, targets)
            print('loss_dict', loss_dict)
        self.model.training = training

        total_loss = sum(loss_dict[k] for k in loss_dict.keys())
        loss_dict.update({'loss': total_loss})

        # IoU
        # giou = self.criterion.compute_giou(outputs, processed_targets)

        # Move IoU to CPU for logging purposes
        # giou = giou.detach().cpu()

        return total_loss, loss_dict, {'giou': torch.tensor(0.0, device='cpu')}

    # def on_validation_epoch_end(self):
    #     avg_val_loss = torch.stack([x['loss'] for x in self.val_step_outputs]).mean()
    #     avg_val_bbox_regression = torch.stack([x['bbox_regression'] for x in self.val_step_outputs]).mean()
    #     avg_val_classification = torch.stack([x['classification'] for x in self.val_step_outputs]).mean()
    #     # avg_val_iou = torch.stack([x['giou'] for x in self.val_step_outputs]).mean()

    #     self.log('val_loss', avg_val_loss)
    #     self.log('val_loss_bbox_regression', avg_val_bbox_regression)
    #     self.log('val_loss_classification', avg_val_classification)
    #     # self.log('val_giou', avg_val_iou)

    #     self.val_step_outputs.clear()
