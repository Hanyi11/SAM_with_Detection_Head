import argparse
from pathlib import Path
import random
import time
# import lightning as pl
# from lightning import loggers as pl_loggers
# from lightning.callbacks import LearningRateMonitor, ModelCheckpoint
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
import torch
from torch.profiler import profile, record_function, ProfilerActivity
from samos import models
# from detection_head_datamodule import DetectionHeadDataModule
# from detection_head_model import DetectionHead
import wandb
import os
from omegaconf import OmegaConf
import hydra
from omegaconf import DictConfig
from omegaconf import open_dict

import sys
import_dir = '/home/icb/lion.gleiter/projects/organoid_sam/external'
if import_dir not in sys.path:
    sys.path.append(import_dir)
import_dir = '/home/icb/lion.gleiter/projects/organoid_sam/segment-anything/segment-anything'
if import_dir not in sys.path:
    sys.path.append(import_dir)


from samos.models import detr_own_impl_frcnn_bb_model
from samos.models import detr_own_impl_model
from samos.models import embeddings_datamodule
from samos.models import training_module
from samos.models import faster_rcnn_model
from samos.models import image_datamodule
from samos.models import ssd_model
from samos.models import samos_anchor_detr

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


# For loading previously saved checkpoints:
sys.modules['models'] = models


def initialize_model_and_dataset(args: dict):
    if args.backbone_name == "DETR":
        datamodule = models.image_datamodule.ImageDataModule(**args)
        raise NotImplementedError()
    elif args.backbone_name == "FRCNN":
        datamodule = models.image_datamodule.ImageDataModule(**args)
        # raise NotImplementedError()
    elif args.backbone_name == "FRCNNv2":
        datamodule = models.image_datamodule.ImageDataModule(**args)
        # raise NotImplementedError()
    elif args.backbone_name == "SSD":
        datamodule = models.image_datamodule.ImageDataModule(**args)
        raise NotImplementedError()
    elif args.backbone_name == "SAM_base_images":
        datamodule = models.image_datamodule.ImageDataModule(**args)
    elif args.backbone_name == "SAM_large_images":
        datamodule = models.image_datamodule.ImageDataModule(**args)
    elif args.backbone_name == "resnet50":
        datamodule = models.image_datamodule.ImageDataModule(**args)
    elif args.backbone_name == "SAM_large":
        datamodule = models.embeddings_datamodule.EmbeddingDataModule(**args)
    elif args.backbone_name == "SAM2_large":
        datamodule = models.embeddings_datamodule.EmbeddingDataModule(**args)
    elif args.backbone_name == "Cellpose":
        raise NotImplementedError()
    elif args.backbone_name == "FM_concat":
        datamodule = models.embeddings_datamodule.EmbeddingDataModule(**args)
    elif args.backbone_name == 'default':
        # Don't instantiate any backbone and change the name to the same as the decoder (necessary for image data module).
        backbone = None
        with open_dict(args):
            args.backbone_name = args.decoder
        datamodule = models.image_datamodule.ImageDataModule(**args)
    else:
        raise ValueError(f'args.backbone_name: {args.backbone_name} is invalid / not supported.')
    
    # with open_dict(args):
    #     args.pop('backbone')  # Avoid multiple kwargs named backbone

    if args.decoder == "DETR_own_implementation":
        decoder = models.detr_own_impl_model.DetectionTransformer(**args)
    elif args.decoder == "DETR_own_image_based":
        decoder = models.detr_own_impl_frcnn_bb_model.DetectionTransformer(**args)
    elif args.decoder == "AnchorDETR":
        decoder = models.samos_anchor_detr.DetectionTransformer(**args)
    elif args.decoder == "DETR":
        raise NotImplementedError()
    elif args.decoder == "FRCNN":
        decoder = models.faster_rcnn_model.FasterRCNN_model(backbone=backbone, **args)
    elif args.decoder == "FRCNNv2":
        decoder = models.faster_rcnn_model.FasterRCNN_model(backbone=backbone, **args)
    elif args.decoder == "SSD":
        decoder = models.ssd_model.SSD(backbone=backbone, **args)
    else:
        raise ValueError(f'args.decoder: {args.decoder} is invalid / not supported.')
    
    model = models.training_module.TrainingModule(model=decoder, **args)
    return datamodule, model



def test(args) -> None:
    # Print all parameters before training
    print("Training Parameters:")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")
            
    # For logging
    train_dirs_str = "_".join(args.train_dirs)
    logging_name = f"{args.decoder}_{args.backbone_name}_{train_dirs_str.replace('/', '_')}_{args.sub_name}_{args.use_sampler}_{args.batch_size}_{args.batches_per_epoch}_{args.run}"
    with open_dict(args):
        args.logging_name = logging_name

    # Force evaluation of variables in the checkpoint path before initializing 
    # the model, since the model initialization might change the backbone name.
    if args.checkpoint_path is not None:
        with open_dict(args):
            args.checkpoint_path = str(args.checkpoint_path)

    # Determinism
    random.seed(0)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    pl.seed_everything(args.seed)
        
    # Model and dataset initialization
    data_module, model = initialize_model_and_dataset(args)

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        gradient_clip_val=args.gradient_clip_val,
    )

    # Testing of the last checkpoint model
    ckpt_eval = Path("/ictstr01/groups/shared/users/lion.gleiter/organoid_sam/checkpoints_for_evaluation") / args.custom_ckpt_file
    assert ckpt_eval.exists(), ckpt_eval

    # Set output directory name separately from checkpoint name
    output_dir_name = args.custom_output_name
    model.output_dir_base = Path('/ictstr01/groups/shared/users/lion.gleiter/organoid_sam/testset_predictions/trained') / output_dir_name
    assert not model.output_dir_base.exists(), model.output_dir_base
    model.output_dir_base.mkdir()

    for ds_name, test_dataloader in zip(data_module.test_dir_names,
                                        data_module.get_test_dataloaders()):
        print('\n\nds_name', ds_name, '\n\n')
        print('\n\ntest_dataloader', test_dataloader, '\n\n')
        model.current_test_set_name = ds_name
        trainer.test(model, test_dataloader, ckpt_path=ckpt_eval)



@hydra.main(config_path="./configs", config_name="default", version_base=None)
def main(cfg: DictConfig):
    with open_dict(cfg):
        if cfg.seed is None:
            cfg.seed = 2024

    print('Final config:\n\n', OmegaConf.to_yaml(cfg), '\n')

    # Start training
    test(cfg)


if __name__ == "__main__":
    main()