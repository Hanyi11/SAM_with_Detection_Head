import argparse
from pathlib import Path
import random
# import lightning as pl
# from lightning import loggers as pl_loggers
# from lightning.callbacks import LearningRateMonitor, ModelCheckpoint
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
import torch
import models
# from detection_head_datamodule import DetectionHeadDataModule
# from detection_head_model import DetectionHead
import wandb
import os
from omegaconf import OmegaConf
import hydra
from omegaconf import DictConfig
from omegaconf import open_dict

import models.detr_own_impl_frcnn_bb_model
import models.detr_own_impl_model
import models.embeddings_datamodule
import models.training_module
import models.faster_rcnn_model
import models.image_datamodule
import models.ssd_model

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"



# # --train_dir="" --val_dir="" --sub_name="SAM_large"
# def get_args_parser():
#     parser = argparse.ArgumentParser(description='Set Detection Head', add_help=False)

#     # Directories for training and validation datasets
#     parser.add_argument('--train_dirs', type=str, nargs='+', required=True, help='List of directories containing the training dataset.')
#     parser.add_argument('--val_dirs', type=str, nargs='+', required=True, help='List of directories containing the validation dataset.')
#     parser.add_argument('--model_config', type=str, required=True, help='Decoder architecture config')
#     #                     '["FRCNN" (Faster R-CNN), "FRCNNv2" (Faster R-CNN v2), "DETR" (DETR transformer decoder), "SSD" (SSD decoder)]')
#     # parser.add_argument('--backbone', type=str, required=False, default='default', help="""
#     #                         Backbone which computes embeddings used as decoder input: [
#     #                             "SSD" (SSD backbone), 
#     #                             "FRCNN" (Faster R-CNN ResNet incl. FPN), 
#     #                             "FRCNNv2", 
#     #                             "DETR" (DETR backbone),
#     #                             "SAM_large" (SAM large ViT), 
#     #                             "SAM2_large" (SAM2 large ViT),
#     #                             "Cellpose" (Cellpose features ???),
#     #                             "FM_concat" (SAM + SAM2 + Cellpose features concatenated),
#     #                         ]
#     #                     """)
#     parser.add_argument('--batch_size', type=int, 
#                         help='Number of samples in each batch.')
#     parser.add_argument('--batches_per_epoch', type=int, 
#                         help="""Define how many batches are used during training of each epoch. The data 
#                         is then sampled by a RandomSample instead of using shuffle in the data loader""")
#     parser.add_argument('--use_sampler', action='store_true', 
#                         help='If true the model is trained with a fixed size of batches per epoch'
#                         'instead of using possibly all data in the datset each epoch. Is useful if you want to train models on multiple datasets and compare them.')
#     parser.add_argument('--max_oversampling', type=float, 
#                         help='Limits how often training samples may be drawn compared to no group-based sampling. If other groups are undersampled, the difference in sampling frequency might be larger.')
#     parser.add_argument('--n_validation_samples', type=int, 
#                         help='If > 0, only evaluates this amount of validation samples from each val set during training.')
    
#     # Learning rate and optimizer parameters
#     parser.add_argument('--learning_rate', type=float, help='Learning rate for the optimizer.')
#     parser.add_argument('--weight_decay', type=float, help='Weight decay for the optimizer.')
#     parser.add_argument('--lr_drop', type=int, help='Number of epochs before dropping the learning rate.')

#     # Training parameters
#     parser.add_argument('--max_epochs', type=int, help='Maximum number of epochs for training.')
#     parser.add_argument('--gradient_clip_val', type=float, help='Gradient clipping value.')

#     # Logging and checkpointing parameters
#     parser.add_argument('--sub_name', type=str, default='default', help='Sub-name for detailed identification of the checkpoint.')

#     # Fine-tuning specific parameters
#     parser.add_argument('--checkpoint_path', type=str, help='Path to the pre-trained checkpoint for fine-tuning.')

#     return parser



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



def train(args) -> None:
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
        

    # Model and dataset initialization
    data_module, model = initialize_model_and_dataset(args)


    # Configure model checkpointing
    ckpt_frequency = 25

    ckpt_path = Path("/ictstr01/groups/shared/users/lion.gleiter/organoid_sam/checkpoints_trained_miccai/")
    ckpt_path.mkdir(exist_ok=True)

    # Resume from previous training checkpoint and pretraining initialization
    (ckpt_path / logging_name).mkdir(exist_ok=True)
    ckpt_last = ckpt_path / logging_name / f"{logging_name}-last.ckpt"
    if ckpt_last.exists():
        ckpt_file_resume = ckpt_last
        args.checkpoint_path = None  # Don't load pretrained checkpoint
    else:
        ckpt_file_resume = None
        if (args.checkpoint_path is not None) and (args.checkpoint_path != ""):
            # Load pretrained checkpoint. Absolute path or file name?
            if args.checkpoint_path.startswith('/'):
                args.checkpoint_path = Path(args.checkpoint_path)
            else:
                args.checkpoint_path = ckpt_path / '-'.join(args.checkpoint_path.split('-')[:-1]) / args.checkpoint_path 
            assert args.checkpoint_path.exists(), args.checkpoint_path

    if (args.checkpoint_path is not None) and (args.checkpoint_path != ""):
        # Load the pre-trained checkpoint 
        model = model.__class__.load_from_checkpoint(**args)  # (args already contains the keyword checkpoint_path)


    # Logging
    wandb_logger = pl_loggers.WandbLogger(
        name=f"{args.project_name}_{args.decoder}_{args.backbone_name}_{args.sub_name}",
        project=f"{args.project_name}_{train_dirs_str}",
        log_model=True,
        save_dir=args.log_dir
    )

    wandb_logger.experiment.config.update({
        "train_dirs": args.train_dirs,
        "val_dirs": args.val_dirs,
        "decoder": args.decoder,
        "backbone": args.backbone_name,
        "sub_name": args.sub_name,
    })

    # Checkpointing
    checkpoint_callback_last = ModelCheckpoint(
        save_last=1,
        every_n_epochs=1,
        dirpath=ckpt_path / logging_name,
        filename=f"{logging_name}-last",
        verbose=True,
        save_on_train_epoch_end=True  # Ensures correct handling of ckpt_frequency
    )
    
    checkpoint_callback_best = ModelCheckpoint(
        save_top_k=1,
        monitor='val_loss',
        every_n_epochs=ckpt_frequency,
        dirpath=ckpt_path / logging_name,
        filename=f"{logging_name}-best_{{epoch}}_{{val_loss:.2f}}",
        verbose=True,
        save_on_train_epoch_end=False  # Ensures correct handling of ckpt_frequency
    )

    # Training
    lr_monitor = LearningRateMonitor(logging_interval="epoch", log_momentum=False)

    trainer = pl.Trainer(
        logger=[wandb_logger],
        callbacks=[checkpoint_callback_last, checkpoint_callback_best, lr_monitor],
        max_epochs=args.max_epochs,
        gradient_clip_val=args.gradient_clip_val,
        check_val_every_n_epoch=ckpt_frequency,
        accumulate_grad_batches=args.accumulate_grad_batches,
    )

    trainer.fit(model, data_module, ckpt_path=ckpt_file_resume)


    # Testing of the last checkpoint model
    ckpt_last = ckpt_path / logging_name / f"{logging_name}-last.ckpt"
    assert ckpt_last.exists(), ckpt_last

    print(len(data_module.test_dir_names))
    print(data_module.test_dir_names)
    for ds_name, test_dataloader in zip(data_module.test_dir_names,
                                        data_module.get_test_dataloaders()):
        print('\n\nds_name', ds_name, '\n\n')
        print('\n\ntest_dataloader', test_dataloader, '\n\n')
        model.current_test_set_name = ds_name
        trainer.test(model, test_dataloader, ckpt_path=ckpt_last)



@hydra.main(config_path="./configs", config_name="default", version_base=None)
def main(cfg: DictConfig):
    # # Parse command-line arguments
    # parser = argparse.ArgumentParser('Detection model training script', parents=[get_args_parser()])
    # args = parser.parse_args()

    # default_config = OmegaConf.load("configs/default.yaml")
    # model_config = OmegaConf.load(cfg.model_config)

    # args overwrite model_config which overwrites default_config
    # final_config = OmegaConf.merge(default_config, model_config,)

    print('Final config:\n\n', OmegaConf.to_yaml(cfg), '\n')

    # Initialize wandb
    train_dirs_str = "_".join(cfg.train_dirs)
    wandb.init(project=f"{cfg.project_name}", name=f"{cfg.project_name}_{cfg.decoder}_{cfg.backbone_name}_{train_dirs_str}_{cfg.sub_name}_{cfg.seed}")

    # Start training
    train(cfg)

    # Finish the wandb run
    wandb.finish()


if __name__ == "__main__":
    main()