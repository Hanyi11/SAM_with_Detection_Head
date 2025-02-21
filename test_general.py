# import argparse
from pathlib import Path
import random
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
import torch
import models
import wandb
import os
from omegaconf import OmegaConf
import hydra
from omegaconf import DictConfig
from omegaconf import open_dict

import models.training_module
import models.faster_rcnn_model
import models.image_datamodule
import models.ssd_model

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"



def initialize_model_and_dataset(args: dict):
    if args.backbone_name == "DETR":
        backbone = torch.nn.Identity()
        datamodule = models.image_datamodule.ImageDataModule(**args)
        raise NotImplementedError()
    elif args.backbone_name == "FRCNN":
        datamodule = models.image_datamodule.ImageDataModule(**args)
        raise NotImplementedError()
    elif args.backbone_name == "FRCNNv2":
        datamodule = models.image_datamodule.ImageDataModule(**args)
        raise NotImplementedError()
    elif args.backbone_name == "SSD":
        datamodule = models.image_datamodule.ImageDataModule(**args)
        raise NotImplementedError()
    elif args.backbone_name == "SAM_large":
        backbone = torch.nn.Identity()  # Replace with adaptor layers
        raise NotImplementedError()
    elif args.backbone_name == "SAM2_large":
        raise NotImplementedError()
    elif args.backbone_name == "Cellpose":
        raise NotImplementedError()
    elif args.backbone_name == "FM_concat":
        raise NotImplementedError()
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
        decoder = models.detection_head_model.DetectionTransformer(backbone=backbone, **args)
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
    val_dirs_str = "_".join(args.val_dirs)
    logging_name = f"{args.decoder}_{args.backbone_name}_{train_dirs_str.replace('/', '_')}_{args.sub_name}_{args.use_sampler}_{args.batch_size}_{args.batches_per_epoch}_{args.seed}"
    with open_dict(args):
        args.logging_name = logging_name


    # Determinism
    random.seed(0)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    pl.seed_everything(args.seed)
        
    data_module, model = initialize_model_and_dataset(args)


    # Load last training checkpoint
    ckpt_path = Path("/ictstr01/groups/shared/users/lion.gleiter/organoid_sam/checkpoints_trained_miccai/")
    ckpt_last = ckpt_path / logging_name / f"{logging_name}-last.ckpt"
    assert ckpt_last.exists(), ckpt_last

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        gradient_clip_val=args.gradient_clip_val,
    )

    for ds_name, test_dataloader in zip(data_module.test_dir_names,
                                        data_module.get_test_dataloaders()):
        model.current_test_set_name = ds_name
        trainer.test(model, test_dataloader, ckpt_path=ckpt_last)



@hydra.main(config_path="./configs", config_name="default", version_base=None)
def main(cfg: DictConfig):

    print('Final config:\n\n', OmegaConf.to_yaml(cfg), '\n')

    test(cfg)


if __name__ == "__main__":
    main()