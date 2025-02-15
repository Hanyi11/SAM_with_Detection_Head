import argparse
from pathlib import Path
# import lightning as pl
# from lightning import loggers as pl_loggers
# from lightning.callbacks import LearningRateMonitor, ModelCheckpoint
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
import torch
import models
# from detection_head_datamodule import DetectionHeadDataModule
# from detection_head_model import DetectionHead
import wandb
import os

import models.FasterRCNN_model
import models.detection_head_model

# --train_dir="" --val_dir="" --sub_name="SAM_large"
def get_args_parser():
    parser = argparse.ArgumentParser(description='Set Detection Head', add_help=False)

    # Directories for training and validation datasets
    parser.add_argument('--train_dirs', type=str, nargs='+', required=True, help='List of directories containing the training dataset.')
    parser.add_argument('--val_dirs', type=str, nargs='+', required=True, help='List of directories containing the validation dataset.')
    parser.add_argument('--encoder_name', type=str, required=True, help='Encoder used for calculating embeddings: '
                        '["SAM_base", "MedSAM", "CellSAM", "SAM_large", "MicroSAM_huge", "SAM2_large"]')
    parser.add_argument('--decoder_arch', type=str, required=True, help='Decoder architecture which predicts boxes and scores: '
                        '["FRCNN" (Faster R-CNN), "FRCNN_emb" (Faster R-CNN based on embeddings), "DETR_frcnn" (DETr based on Faster R-CNN backbone), "DETR_emb" (DETr based on embeddings)]')
    parser.add_argument('--batch_size', type=int, default=4, help='Number of samples in each batch.')
    parser.add_argument('--batches_per_epoch', type=int, default=500, help='Define how many batches are used during training of each epoch.'
                        ' The data is then sampled by a RandomSample instead of using shuffle in the data loader')
    parser.add_argument('--use_sampler', action='store_true', help='If true the model is trained with a fixed size of batches per epoch'
                        'instead of using possibly all data in the datset each epoch. Is useful if you want to train models on multiple datasets and compare them.')
    parser.add_argument('--balance_datasets', action='store_true', help='If true the different datasets (OrgaSegment, OrganoID, ...) will be equally likely sampled during training and validation.')
    parser.add_argument('--balance_validation', action='store_true', help='If true the different datasets (OrgaSegment, OrganoID, ...) will be equally likely sampled during training and validation.')
    
    parser.add_argument('--balance_newdata', type=float, default=0.05, help='balances the likelyhood of including a sample from our new data during training.')

    # Learning rate and optimizer parameters
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate for the optimizer.')
    parser.add_argument('--weight_decay', default=1e-4, type=float, help='Weight decay for the optimizer.')
    parser.add_argument('--lr_drop', default=200, type=int, help='Number of epochs before dropping the learning rate.')

    # Matcher coefficients for computing the matching cost
    parser.add_argument('--matching', type=str, required=True, help='Matching used to assign ground truth to predictions: '
                        '["hungarian", "greedy"]')
    parser.add_argument('--set_cost_class', default=1, type=float, help="Class coefficient in the matching cost.")
    parser.add_argument('--set_cost_bbox', default=5, type=float, help="L1 box coefficient in the matching cost.")
    parser.add_argument('--set_cost_giou', default=2, type=float, help="GIoU box coefficient in the matching cost.")

    # Loss coefficients for computing the loss
    parser.add_argument('--loss', type=str, required=True, help='Loss: '
                        '["DETR", "FRCNN"]')
    parser.add_argument('--bbox_loss_coef', default=5, type=float, help="Coefficient for bounding box loss.")
    parser.add_argument('--giou_loss_coef', default=2, type=float, help="Coefficient for GIoU loss.")
    parser.add_argument('--eos_coef', default=0.1, type=float, help="Relative classification weight of the no-object class.")

    # Model and training parameters
    parser.add_argument('--max_epochs', type=int, default=500, help='Maximum number of epochs for training.')
    parser.add_argument('--num_queries', type=int, default=200, help='Maximum number of queries.')
    parser.add_argument('--transformer_dim', type=int, default=256, help='Dimension of transformer embeddings.')
    parser.add_argument('--dropout', default=0.1, type=float, help="Dropout applied in the transformer.")
    parser.add_argument('--nheads', type=int, default=8, help='Number of heads in the multihead attention mechanism.')
    parser.add_argument('--dim_feedforward', type=int, default=512, help='Dimension of the feedforward network in transformer.')
    parser.add_argument('--num_layers', type=int, default=6, help='Number of layers in the transformer.')
    parser.add_argument('--pre_norm', type=bool, default=True, help='Whether to use pre-normalization in layers.')

    # Logging and checkpointing parameters
    parser.add_argument('--project_name', type=str, default='DetectionHead', help='Name of the project for logging purposes.')
    parser.add_argument('--sub_name', type=str, default='default', help='Sub-name for detailed identification of the checkpoint.')
    parser.add_argument('--log_dir', type=str, default='logs/', help='Directory to store logs.')
    parser.add_argument('--gradient_clip_val', type=float, default=0.1, help='Gradient clipping value.')

    # Additional training settings
    parser.add_argument('--aux_loss', type=bool, default=True, help='Whether to use auxiliary loss.')

    # Fine-tuning specific parameters
    parser.add_argument('--checkpoint_path', type=str, default=None, required=False, help='Path to the pre-trained checkpoint for fine-tuning.')

    return parser



def initialize_model(args):
    if args.decoder_arch == "DETR_emb":
        backbone = torch.nn.Identity()
        decoder = models.detection_head_model.DetectionTransformer(backbone=backbone, **args)
        model = models.detection_head_model.TrainingModule(model=decoder, **args)
    elif args.decoder_arch == "DETR_frcnn":
        model = models.detection_head_model.DetectionHead(**args)
    elif args.decoder_arch == "FRCNN":
        model = models.FasterRCNN_model.FasterRCNN_model(**args)
    elif args.decoder_arch == "FRCNN_emb":
        model = models.FasterRCNN_model.FasterRCNN_model(**args)
    else:
        raise ValueError(f'args.decoder_arch: {args.decoder_arch} is invalid / not supported.')
    return model







def train(args) -> None:
    # Print all parameters before training
    print("Training Parameters:")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")
        
    model = initialize_model(args)

    # Initialize the data module with training and validation data paths and batch size
    data_module = DetectionHeadDataModule(
                                        encoder_name=args.encoder_name, 
                                        batch_size=args.batch_size, 
                                        train_dir_names=args.train_dirs,
                                        val_dir_names=args.val_dirs,
                                        batches_per_epoch=args.batches_per_epoch,
                                        use_sampler=args.use_sampler,
                                        balance_datasets=args.balance_datasets,
                                        balance_validation=args.balance_validation,
                                        balance_newdata=args.balance_newdata
                                        )


    # Configure model checkpointing
    ckpt_frequency = 50

    ckpt_path = Path("/ictstr01/groups/shared/users/lion.gleiter/organoid_sam/checkpoints_trained/")
    ckpt_path.mkdir(exist_ok=True)

    # For logging
    train_dirs_str = "_".join(args.train_dirs)
    val_dirs_str = "_".join(args.val_dirs)
    logging_name = f"{args.project_name}_{args.encoder_name}_{train_dirs_str.replace('/', '_')}_{args.sub_name}_{args.use_sampler}_{args.batch_size}_{args.batches_per_epoch}"

    # Resume from previous training checkpoint
    (ckpt_path / logging_name).mkdir(exist_ok=True)
    ckpt_files = sorted(list((ckpt_path / logging_name).glob(f"{logging_name}-last_*-val_loss*.ckpt")))
    if len(ckpt_files) > 0:
        ckpt_file_resume = ckpt_files[-1]
    else:
        ckpt_file_resume = None

    # Pretraining initialization
    if (args.checkpoint_path is not None) and (args.checkpoint_path != ""):
        if ckpt_file_resume is not None:
            # Don't load from pretrained checkpoint, instead resume previous training run
            args.checkpoint_path = None
        else:
            # Absolute path or file name?
            if args.checkpoint_path.startswith('/'):
                args.checkpoint_path = Path(args.checkpoint_path)
            else:
                args.checkpoint_path = ckpt_path / '-'.join(args.checkpoint_path.split('-')[:-2]) / args.checkpoint_path 
            assert args.checkpoint_path.exists(), args.checkpoint_path

    if (args.checkpoint_path is not None) and (args.checkpoint_path != ""):
        # Load the pre-trained checkpoint
        # TODO: replace with Trainer.fit(model, ckpt_path=args.checkpoint_path)
        model = model.load_from_checkpoint(
            args.checkpoint_path,
        )
    else:
        # Create new model
        model = DetectionHead(
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            lr_drop=args.lr_drop,
            set_cost_class=args.set_cost_class,
            set_cost_bbox=args.set_cost_bbox,
            set_cost_giou=args.set_cost_giou,
            max_epochs=args.max_epochs,
            num_queries=args.num_queries,
            transformer_dim=args.transformer_dim,
            nheads=args.nheads,
            dim_feedforward=args.dim_feedforward,
            num_layers=args.num_layers,
            dropout=args.dropout,
            pre_norm=args.pre_norm,
            bbox_loss_coef=args.bbox_loss_coef,
            giou_loss_coef=args.giou_loss_coef,
            eos_coef=args.eos_coef,
            aux_loss=args.aux_loss
        )

    # Initialize a CSV logger to record training progress into a CSV file at specified directory
    csv_logger = pl_loggers.CSVLogger(os.path.join(args.log_dir, args.encoder_name, train_dirs_str,''))
    csv_logger.log_hyperparams({"train_dirs": args.train_dirs, "val_dirs": args.val_dirs})
    

    # Logging
    wandb_logger = pl_loggers.WandbLogger(
        name=f"{args.project_name}_{args.encoder_name}_{args.sub_name}",
        project=f"{args.project_name}_{train_dirs_str}",
        log_model=True,
        save_dir=args.log_dir
    )

    wandb_logger.experiment.config.update({
        "train_dirs": args.train_dirs,
        "val_dirs": args.val_dirs,
        "encoder_name": args.encoder_name,
        "sub_name": args.sub_name
    })

    # Checkpointing
    checkpoint_callback_last = ModelCheckpoint(
        save_last=1,
        every_n_epochs=ckpt_frequency,
        dirpath=ckpt_path / logging_name,
        filename=f"{logging_name}-last_{{epoch}}-{{val_loss:.2f}}",
        verbose=True,
        save_on_train_epoch_end=False  # Ensures correct handling of ckpt_frequency
    )
    
    checkpoint_callback_best = ModelCheckpoint(
        save_top_k=1,
        monitor='val_loss',
        every_n_epochs=ckpt_frequency,
        dirpath=ckpt_path / logging_name,
        filename=f"{logging_name}-best_{{epoch}}-{{val_loss:.2f}}",
        verbose=True,
        save_on_train_epoch_end=False  # Ensures correct handling of ckpt_frequency
    )

    # Training
    lr_monitor = LearningRateMonitor(logging_interval="epoch", log_momentum=False)

    trainer = pl.Trainer(
        logger=[csv_logger, wandb_logger],
        callbacks=[checkpoint_callback_last, checkpoint_callback_best, lr_monitor],
        max_epochs=args.max_epochs,
        gradient_clip_val=args.gradient_clip_val,
        check_val_every_n_epoch=ckpt_frequency,
    )

    trainer.fit(model, data_module, ckpt_path=ckpt_file_resume)


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser('Detection Head training script', parents=[get_args_parser()])
    args = parser.parse_args()
    train_dirs_str = "_".join(args.train_dirs)

    # Initialize wandb
    wandb.init(project=f"{args.project_name}", name=f"{args.project_name}_{args.encoder_name}_{train_dirs_str}_{args.sub_name}")

    # Start training
    train(args)

    # Finish the wandb run
    wandb.finish()
