from typing import Literal
import pytorch_lightning as pl


# torchvision libraries
import torch
from torch import nn
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, FasterRCNN_ResNet50_FPN_V2_Weights, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models import ResNet50_Weights


class FasterRCNN_model(nn.Module):
    def __init__(self, 
                num_classes: int = 2, 
                decoder: Literal["FRCNN", "FRCNNv2"] = "FRCNNv2", 
                backbone = None,
                **kwargs,  # For unused config params.
                ):
        super().__init__()

        # Parameters for model initialization
        self.num_classes = num_classes
        # self.version_FasterRCNN = version_FasterRCNN
        self.decoder_name = decoder
        self.backbone = backbone

        # Parameters for optimizer and scheduler
        # self.learning_rate = learning_rate
        # self.weight_decay = weight_decay
        # self.lr_drop = lr_drop
        # self.max_epochs = max_epochs
     
        # Define the FasterRCNN net
        self.model = self.get_pretrained_FasterRCNN()
    
        # Initialize lists to store training and validation metrics
        self.training_step_outputs = []
        self.val_step_outputs = []
    
    
    def get_pretrained_FasterRCNN(self):
        # load a model pre-trained on COCO
        if self.decoder_name == "FRCNN":
            model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
                box_detections_per_img=200,
                # weights=FasterRCNN_ResNet50_FPN_Weights.COCO_V1
                weights_backbone="ResNet50_Weights.IMAGENET1K_V1"
                )
        elif self.decoder_name == "FRCNNv2":
            model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(
                box_detections_per_img=200,
                # weights=FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1
                weights_backbone="ResNet50_Weights.IMAGENET1K_V1"
                )
        else:
            raise ValueError(f'decoder {self.decoder_name}')


        if self.backbone is not None:
            raise NotImplementedError()
            # Modify the first convolution layer to accept n-channel input
            if self.encoder_name.startswith('SAM'):
                in_channels = 256  # New input channels
            else:
                raise ValueError(f'Only SAM variants currently supported as encoder, not {self.encoder_name}')
            old_conv = model.backbone.body.conv1

            # Create a new conv layer with same parameters but 256 input channels
            new_conv = nn.Conv2d(
                in_channels, 
                old_conv.out_channels, 
                kernel_size=old_conv.kernel_size, 
                stride=old_conv.stride, 
                padding=old_conv.padding, 
                bias=(old_conv.bias is not None)
            )

            # Initialize new weights (e.g., by averaging the original 3-channel weights)
            with torch.no_grad():
                new_conv.weight[:, :3] = old_conv.weight  # Copy RGB weights
                if in_channels > 3:
                    new_conv.weight[:, 3:] = old_conv.weight[:, :1].repeat(1, in_channels - 3, 1, 1)  # Copy first channel weights

            # Replace the original conv layer
            model.backbone.body.conv1 = new_conv

        
        # get number of input features for the classifier
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        
        # replace the pre-trained head with a new one
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, self.num_classes) 

        return model
    

    def forward(self, image, size=None):

        # Do a forward pass in FasterRCNN
        detections = self.model(image)
        detections = detections[0]
        boxes = detections['boxes']
        labels = detections['labels']
        scores = detections['scores']

        # Select only the results for foreground (1) predictions 
        is_foreground = labels == 1 # Creates a boolean tensor: True where labels == 1, False where labels == 0
        boxes = boxes[is_foreground]
        labels = labels[is_foreground]
        scores = scores[is_foreground]
        out = {'pred_scores': scores, 'pred_boxes': boxes}
   
        return out

    # def configure_optimizers(self):
    #     optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
    #     scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.lr_drop)
    #     return [optimizer], [scheduler]

    def forward_train(self, batch):
        images, targets = batch[0], batch[1]
        device = images[0].device

        # Do a forward pass of FasterRCNN
        try:
            # print(f"self.model(images, targets): {self.model(images, targets)}")
            loss_dict: dict = self.model(images, targets)
        except Exception as e:
            print('targets', targets)
            raise e
       
        if not loss_dict:
            raise ValueError("The loss_dict is empty. Please check the input or the loss computation.")

        total_loss = sum(loss_dict[k] for k in loss_dict.keys())
        loss_dict.update({'loss': total_loss})
    
        return total_loss, loss_dict, {'giou': torch.tensor(0.0, device='cpu')}

    # def on_train_epoch_end(self):
    #     avg_train_loss = torch.stack([x['loss'] for x in self.training_step_outputs]).mean()
    #     avg_train_bbox_regression = torch.stack([x['loss_box_reg'] for x in self.training_step_outputs]).mean()
    #     avg_train_classification = torch.stack([x['loss_classifier'] for x in self.training_step_outputs]).mean()
    #     avg_train_rpn_objectness = torch.stack([x['loss_objectness'] for x in self.training_step_outputs]).mean()
    #     avg_train_loss_rpn_box_reg = torch.stack([x['loss_rpn_box_reg'] for x in self.training_step_outputs]).mean()
        

    #     # avg_train_iou = torch.stack([x['giou'] for x in self.training_step_outputs]).mean()

    #     self.log('train_loss', avg_train_loss)
    #     self.log('train_loss_bbox_regression', avg_train_bbox_regression)
    #     self.log('train_loss_classification', avg_train_classification)
    #     self.log('train_loss_rpn_objectness', avg_train_rpn_objectness)
    #     self.log('train_loss_rpn_box_reg', avg_train_loss_rpn_box_reg)
    #     # self.log('train_giou', avg_train_iou)

    #     self.training_step_outputs.clear()

    def forward_eval(self, batch):
        # print("Validation Try begin")
        # print('batch_idx', batch_idx)
        images, targets = batch[0], batch[1]
        device = images[0].device
        # print(f"validation step targets: {targets}")
        # print('len(images), len(targets)', len(images), len(targets))
        
        training = self.model.training
        # print('training', training)  # False, needs to be True to compute the loss values
        
        with torch.inference_mode():
            self.model.training = True # False, needs to be True to compute the loss values
            self.model.rpn.training = True # False, needs to be True to compute the loss values
            self.model.roi_heads.training = True # False, needs to be True to compute the loss values
            # print(f"val: model.rpn.training =  {self.model.rpn.training}")
            # print(f"val: model.roi_heads.training =  {self.model.roi_heads.training}")

            loss_dict: dict = self.model(images, targets)
            if not loss_dict:
                raise ValueError("The loss_dict is empty. Please check the input or the loss computation.")

            # print('loss_dict', loss_dict)

        total_loss = sum(loss_dict[k] for k in loss_dict.keys())
        loss_dict.update({'loss': total_loss})

        self.val_step_outputs.append(loss_dict)

        return total_loss, loss_dict, {'giou': torch.tensor(0.0, device='cpu')}

    # def on_validation_epoch_end(self):
    #     avg_val_loss = torch.stack([x['loss'] for x in self.val_step_outputs]).mean()
    #     avg_val_bbox_regression = torch.stack([x['loss_box_reg'] for x in self.val_step_outputs]).mean()
    #     avg_val_classification = torch.stack([x['loss_classifier'] for x in self.val_step_outputs]).mean()
    #     avg_val_rpn_objectness = torch.stack([x['loss_objectness'] for x in self.val_step_outputs]).mean()
    #     avg_val_loss_rpn_box_reg = torch.stack([x['loss_rpn_box_reg'] for x in self.val_step_outputs]).mean()

    #     self.log('val_loss', avg_val_loss)
    #     self.log('val_loss_bbox_regression', avg_val_bbox_regression)
    #     self.log('val_loss_classification', avg_val_classification)
    #     self.log('val_lavg_val_rpn_objectnessoss', avg_val_rpn_objectness)
    #     self.log('val_loss_rpn_box_reg', avg_val_loss_rpn_box_reg)



    #     # self.log('val_giou', avg_val_iou)

    #     self.val_step_outputs.clear()
