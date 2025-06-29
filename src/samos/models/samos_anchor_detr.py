import os
from typing import Literal
import torch
import pytorch_lightning as pl
from torch import nn, Tensor
import torch.nn.functional as F
from torchvision.models.feature_extraction import create_feature_extractor
# from transformer_layers import TransformerDecoder, MLP, PositionEmbeddingSine
# from matcher import HungarianMatcher
# from losses import SetCriterion
import copy
from omegaconf import OmegaConf

from AnchorDETR.models.anchor_detr import AnchorDETR, PostProcess, SetCriterion
from AnchorDETR.models.backbone import build_backbone
from AnchorDETR.models.matcher import build_matcher
from AnchorDETR.models.transformer import build_transformer
from AnchorDETR.util.misc import NestedTensor
# from AnchorDETR.util.misc import (NestedTensor, nested_tensor_from_tensor_list,
#                                   accuracy, get_world_size, interpolate,
#                                   is_dist_avail_and_initialized)
from segment_anything import sam_model_registry, SamPredictor
from segment_anything.modeling.image_encoder import window_partition, get_rel_pos, window_unpartition


# ------------------------------------------------------------------------
# Modified from Anchor DETR (https://github.com/megvii-research/AnchorDETR/blob/main/models/anchor_detr.py)
# Copyright (c) 2021 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------




def build_sam_encoder(
       model_size: Literal["large", "base"] =  "large" ,
       checkpoint_dir: str =  "/ictstr01/groups/shared/users/lion.gleiter/organoid_sam/checkpoints"
    ):
    """
    Loads a Segment Anything Model (SAM) with pretrained weights, either 'large' or 'base', 
    and returns a SamPredictor object initialized on the specified device.
    
    Args:
        model_size (Literal["large", "base"], optional): Model size to load ('large' or 'base'). Default is 'large'.
        checkpoint_dir (str, optional): Directory path containing model checkpoints. Default is the specified path.
        device (str, optional): Device to load the model on ('cuda' or 'cpu'). Default is 'cuda'.
    
    Returns:
        SamPredictor: A configured SamPredictor instance with the loaded model.
    """
    # Define path to MedSAM checkpoint file & specify model type 
    if model_size == "large":
        sam_checkpoint = os.path.join(checkpoint_dir, "sam_vit_l_0b3195.pth")
        model_type = "vit_l"
        embed_dim = 1024
        encoder_global_attn_indexes=[5, 11, 17, 23]
    elif model_size == "base":
        sam_checkpoint = os.path.join(checkpoint_dir, "sam_vit_b_01ec64.pth") 
        model_type = "vit_b"
        embed_dim = 768
        encoder_global_attn_indexes=[2, 5, 8, 11]

    # Load SAM model with specified checkpoint and initialize it on the given device
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    return sam.image_encoder, embed_dim, encoder_global_attn_indexes

class SAMBackbone(nn.Module):
    def __init__(self, get_intermediate_layers = (2, 5, 8, 11), model_size='base'):
        super().__init__()
        self.image_encoder_sam, self.embed_dim, self.encoder_global_attn_indexes = build_sam_encoder(model_size=model_size)
        
        # Normalize colors similar to Sam class in segment_anything (we have 0, 1 as input range here instead of 0, 255).
        pixel_mean=[123.675 / 255, 116.28 / 255, 103.53 / 255]
        pixel_std=[58.395 / 255, 57.12 / 255, 57.375 / 255]
        # pixel_mean=[123.675, 116.28, 103.53]
        # pixel_std=[58.395, 57.12, 57.375]
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

        self.get_intermediate_layers = get_intermediate_layers
        features = {
            f'blocks.{i}': f'intermediate_{i}' for i in get_intermediate_layers
        }
        features.update({'neck': 'final_embed'})
        self.feature_layers = [f'intermediate_{i}' for i in get_intermediate_layers] + ['final_embed']

        # print('self.image_encoder_sam', self.image_encoder_sam)
        print('features', features)
        self.sam = create_feature_extractor(
            model=self.image_encoder_sam,
            return_nodes=features,
            tracer_kwargs={'autowrap_functions': [window_partition, get_rel_pos, window_unpartition]}
        )

        self.strides = self.feature_layers  # Only len(backbone.strides) is used and should equal the number of returned features maps
        self.num_channels = [
            self.embed_dim for _ in self.get_intermediate_layers
        ] + [256]  # Output dimension of the neck, same as the prompt_embed_dim


    def forward(self, x):
        if isinstance(x, NestedTensor):
            x, mask = x.decompose()
            
        # print(x.dtype)
        # print('x.min(), x.max()', x[:,0,:,:].min(), x[:,0,:,:].max())
        # print('x.min(), x.max()', x[:,1,:,:].min(), x[:,1,:,:].max())
        # print('x.min(), x.max()', x[:,2,:,:].min(), x[:,2,:,:].max())

        # Normalize colors similar to Sam class in segment_anything (we have 0, 1 as input range here instead of 0, 255).
        x = (x - self.pixel_mean) / self.pixel_std

        # print(x.dtype)
        # print(x.shape)
        # print('x.min(), x.max()', x[:,0,:,:].min(), x[:,0,:,:].max())
        # print('x.min(), x.max()', x[:,1,:,:].min(), x[:,1,:,:].max())
        # print('x.min(), x.max()', x[:,2,:,:].min(), x[:,2,:,:].max())
        feats = self.sam(x)
        outputs = []
        for k in self.feature_layers:
            if k.startswith('intermediate_'):
                feat = feats[k].permute(0, 3, 1, 2)
            else:
                feat = feats[k]
            b, _, h, w = feat.shape
            mask = torch.zeros((b, h, w), dtype=torch.bool, device=feat.device)
            outputs.append(NestedTensor(feat, mask))
        return outputs


def build(args):
    # num_classes = 2
    # device = torch.device(args.device)

    if args.backbone_name.startswith('SAM'):
        backbone = SAMBackbone(get_intermediate_layers=args.backbone_get_intermediate_layers,
                               model_size=args.backbone_model_size)
    else:
        backbone = build_backbone(args)

    if args.freeze_backbone:
        # Freeze the backbone parameters
        for param in backbone.parameters():
            param.requires_grad = False
    # Else, layer 2, 3, and 4 of resnet50 are trained.
    # else:
    #     # Unfreeze the backbone parameters
    #     for name, param in backbone.named_parameters():
    #         print(name, param.requires_grad)
    #         # assert param.requires_grad == True, name

    transformer = build_transformer(args)
    model = AnchorDETR(
        backbone,
        transformer,
        num_feature_levels=args.num_feature_levels,
        aux_loss=args.aux_loss
    )
    if args.finetune_only_class_linear:
        # Freeze the all parameters except transformer.class_embed
        for name, param in model.named_parameters():
            if not name.startswith('transformer.class_embed'):
                param.requires_grad = False
            else:
                print(f'Trainable parameter: {name}: {param.requires_grad}')

    matcher = build_matcher(args)
    weight_dict = {'loss_ce': args.cls_loss_coef, 'loss_bbox': args.bbox_loss_coef}
    weight_dict['loss_giou'] = args.giou_loss_coef
    if args.masks:
        weight_dict["loss_mask"] = args.mask_loss_coef
        weight_dict["loss_dice"] = args.dice_loss_coef
    # TODO this is a hack
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        aux_weight_dict.update({k + f'_enc': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    losses = ['labels', 'boxes']
    if args.masks:
        losses += ["masks"]
    # num_classes, matcher, weight_dict, losses, focal_alpha=0.25
    criterion = SetCriterion(args.num_classes, matcher, weight_dict, losses, focal_alpha=args.focal_alpha)
    # criterion.to(device)
    postprocessors = {'bbox': PostProcess()}
    return model, criterion, postprocessors



class DetectionTransformer(nn.Module):
    def __init__(self, 
                #  backbone_name,
                 **args
                ):
        super().__init__()
        # print(type(args))
        # print(args)
        args = OmegaConf.create(args)
        self.backbone_name = args.backbone_name
        self.model, self.criterion, self.postprocessors = build(args)

        # For comparing model parameters
        self.backbone = self.model.backbone

    def forward(self, images):
        """Predicts boxes and scores for a batch of image embeddings

        Args:
            images: a torch.Tensor of shape [B, 1, 1024, 1024].

        Returned boxes are expected to be in the format [x1, y1, x2, y2] in pixel coordinates.
        """
        output = self.forward_images(images)

        boxes = output['pred_boxes']
        logits = output['pred_logits']
        # print(logits.shape)
        scores = logits.sigmoid()
        # scores_softmax = logits.softmax(dim=-1)
        # print('scores', scores.shape)
        # print('scores', scores)
        # print('scores_softmax', scores_softmax.shape)
        # print('scores_softmax', scores_softmax)
        scores = scores[:, :, 0]  # class 0 is organoids, 1 is background
        # print(scores.shape)
        labels = torch.zeros(scores.shape, dtype=torch.int64, device=scores.device)

        # Transform boxes:  [cx cy w h] in [0, 1] range  -->  [x y x y] in [0, 1024] px
        boxes = torch.stack([
            boxes[:, :, 0] - (boxes[:, :, 2] / 2),
            boxes[:, :, 1] - (boxes[:, :, 3] / 2),
            boxes[:, :, 0] + (boxes[:, :, 2] / 2),
            boxes[:, :, 1] + (boxes[:, :, 3] / 2),
        ], dim=2) * 1024

        predictions = []
        for batch_idx in range(scores.shape[0]):
            predictions.append({'scores': scores[batch_idx],
                                'boxes': boxes[batch_idx],
                                'labels': labels[batch_idx]})

        return predictions

    def forward_images(self, images: torch.Tensor):
        # print('mem 1:', torch.cuda.memory_allocated())

        # device = images.device
        # batch_size = images.size(0)

        pred = self.model(images)

        return pred

    def forward_batch(self, batch):
        images, targets = batch[0], batch[1]

        # Convert target boxes from [x, y, x, y] in [0, 1024] px to [cx, cy, w, h] in [0, 1] range
        targets_updated = []
        for t in targets:
            boxes = t['boxes']
            boxes = torch.stack([
                (boxes[:, 0] + boxes[:, 2]) / 2,  # cx
                (boxes[:, 1] + boxes[:, 3]) / 2,  # cy
                boxes[:, 2] - boxes[:, 0],  # w
                boxes[:, 3] - boxes[:, 1]  # h
            ], dim=1) / 1024.0
            t['boxes'] = boxes
            targets_updated.append(t)

        # print('targets_updated', targets_updated[0]['boxes'])

        # Forward
        outputs = self.model(images)
        # boxes = outputs['pred_boxes']
        # print('boxes after forward', boxes[:1, :4, :], boxes.min(), boxes.max())

        # Loss
        # print('outputs', outputs)
        # print('targets', targets_updated)
        loss_dict = self.criterion(outputs, targets_updated)
        weight_dict = self.criterion.weight_dict
        total_loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # # Process targets: filter out all-zero entries and create dictionary
        # # targets are bounding boxes of shape [bs x num_queries x 4]
        # processed_targets = []
        # for target in targets:
        #     boxes = target['boxes']
        #     labels = target['labels']
        #     non_zero_indices = (boxes > -0.5).any(axis=1)
        #     # device = boxes.device
            
        #     # Transform boxes:  [x y x y] in [0, 1024] px  -->  [cy cx h w] in [0, 1] range
        #     boxes = torch.stack([
        #         (boxes[:, 1] + boxes[:, 3]) / 2,
        #         (boxes[:, 0] + boxes[:, 2]) / 2,
        #         boxes[:, 3] - boxes[:, 1],
        #         boxes[:, 2] - boxes[:, 0],
        #     ], dim=1) / 1024

        #     filtered_boxes = boxes[non_zero_indices]#.to(device)
        #     filtered_labels = labels[non_zero_indices]#.to(device)
        #     num_boxes = filtered_boxes.size(0)
        #     # print('num_boxes', num_boxes)
        #     processed_targets.append({
        #         'boxes': filtered_boxes,
        #         'labels': filtered_labels.to(dtype=torch.int64)  # Here, labels are 0 = ground truth, 1 = no object
        #     })
        
        # loss_dict = self.criterion(outputs, processed_targets)
        # weight_dict = self.criterion.weight_dict
        # total_loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # IoU
        # giou = self.criterion.compute_giou(outputs, processed_targets)

        # Move IoU to CPU for logging purposes
        giou = torch.tensor(0.0, device='cpu')  # giou.detach().cpu()

        return total_loss, loss_dict, {'giou': giou}
    
    def forward_train(self, batch):
        self.train()
        return self.forward_batch(batch)

    def forward_eval(self, batch):
        self.eval()
        return self.forward_batch(batch)

