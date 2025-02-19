import torch
import pytorch_lightning as pl
from torch import nn, Tensor
from transformer_layers import TransformerDecoder, MLP, PositionEmbeddingSine
from matcher import HungarianMatcher
from losses import SetCriterion


path_detr = '/home/icb/lion.gleiter/projects/organoid_sam/detr'
if path_detr not in sys.path:
    sys.path.append(path_detr)
from detr.models import build_model
from detr.models.backbone import build_backbone


class DETRDecoder(nn.Module):
    def __init__(self, 
                 backbone,
                 **kwargs
                ):
        super().__init__()
        self.backbone = backbone
        self.in_features = transformer_dim  # Expected number of features of the backbone output
        self.in_dim_fixed = None  # Arbitrary width / height possible as output of the backbone
        self.adaptor = nn.Conv2d(backbone.num_channels, self.in_features, kernel_size=1)

        # Define matcher and loss here
        self.matcher = HungarianMatcher(cost_class=set_cost_class, cost_bbox=set_cost_bbox, cost_giou=set_cost_giou)

        weight_dict = {'loss_ce': 1, 'loss_bbox': bbox_loss_coef, 'loss_giou': giou_loss_coef}
        
        if aux_loss:
            aux_weight_dict = {}
            for i in range(num_layers - 1):
                aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)

        # for loss computation
        losses = ['labels', 'boxes', 'cardinality']
        self.criterion = SetCriterion(num_classes=1, 
                                      matcher=self.matcher, 
                                      weight_dict=weight_dict,
                                      eos_coef=eos_coef, 
                                      losses=losses)
        
        self.max_epochs = max_epochs
        self.num_queries = num_queries
        self.query_embed = nn.Embedding(num_queries, transformer_dim)

        # Define positional embedding
        N_steps = transformer_dim // 2
        self.position_embedding = PositionEmbeddingSine(N_steps, normalize=True)

        # Define the transformer module
        self.transformer_decoder = TransformerDecoder(
            transformer_dim=transformer_dim,
            nheads=nheads,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            pre_norm=pre_norm,
            return_intermediate=aux_loss, # if use auxiliary loss, must return intermediate outputs
        )

        self.class_embed = nn.Linear(transformer_dim, 2) # Binary classification: Object or No object
        self.bbox_embed = MLP(transformer_dim, transformer_dim, 4, 3)

        self.aux_loss = aux_loss  # Ensure aux_loss is stored

    def _set_aux_loss(self, outputs_class, outputs_coord):
        return [{'pred_logits': a, 'pred_boxes': b} for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]

    def forward(self, image_embeddings):
        output = self.forward_images(image_embeddings)

        boxes = output['pred_boxes']
        scores = output['pred_logits'].softmax(dim=-1)
        scores = scores[:, :, 0]  # class 0 is organoids, 1 is background

        predictions = []
        for batch_idx in range(scores.shape[0]):
            predictions.append({'scores': scores[batch_idx],
                                'boxes': boxes[batch_idx]})

        return predictions

        # TODO: (filter positive predictions,) convert logits to scores and create a list[dict]


    def forward_images(self, image_embeddings: torch.Tensor):
        device = image_embeddings.device
        batch_size = image_embeddings.size(0)

        # Transformer input
        query_embedding = self.query_embed.weight.to(device)
        pos_embedding = self.position_embedding(image_embeddings) # bs x 256 x 64 x 64
        
        query_embedding = query_embedding.unsqueeze(0).expand(batch_size, -1, -1) # Add batch dimension and expand

        target = torch.zeros_like(query_embedding) # bs x num_queries x transformer_dim (256)
        image_embeddings = image_embeddings.flatten(2).permute(0, 2, 1) # bs x (64x64) x 256
        pos_embedding = pos_embedding.flatten(2).permute(0, 2, 1) # bs x (64x64) x 256

        # Use the transformer module
        target = self.transformer_decoder(target, query_embedding, image_embeddings, pos_embedding)

        # Feed transformer output into MLP to get class and bbox
        outputs_class = self.class_embed(target)
        outputs_coord = self.bbox_embed(target).sigmoid()
        # print('class and coord shapes', outputs_class.shape, outputs_coord.shape)  # [1, 1, 100, 2], [1, 1, 100, 4]
        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)
        return out

    def forward_batch(self, batch):
        image_embedding, targets = batch
        device = image_embedding.device

        # forward
        outputs = self.forward_images(image_embedding)
        
        # Process targets: filter out all-zero entries and create dictionary
        # targets are bounding boxes of shape [bs x num_queries x 4]
        processed_targets = []
        for target in targets:
            non_zero_indices = ~(target != 0).all(axis=1)
            filtered_boxes = target[non_zero_indices].to(device)
            num_boxes = filtered_boxes.size(0)
            print('num_boxes', num_boxes)
            processed_targets.append({
                'boxes': filtered_boxes,
                'labels': torch.zeros(num_boxes, dtype=torch.int64, device=device)  # Here, labels are 0 = ground truth, 1 = no object
            })
        
        # Loss
        loss_dict = self.criterion(outputs, processed_targets)
        weight_dict = self.criterion.weight_dict
        total_loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # IoU
        giou = self.criterion.compute_giou(outputs, processed_targets)

        # Move IoU to CPU for logging purposes
        giou = giou.detach().cpu()

        return total_loss, loss_dict, {'giou': giou}

    def forward_train(self, batch):
        self.train()
        return self.forward_batch(batch)
    
    def forward_eval(self, batch):
        self.eval()
        return self.forward_batch(batch)

