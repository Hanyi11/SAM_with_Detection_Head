import argparse
from pathlib import Path
# import lightning as pl
# from lightning import loggers as pl_loggers
# from lightning.callbacks import LearningRateMonitor, ModelCheckpoint
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from detection_head_datamodule_copy import DetectionHeadDataModule
from detection_head_model import DetectionHead
import wandb
import os

# --train_dir="" --val_dir="" --sub_name="SAM_large"
def get_args_parser():
    parser = argparse.ArgumentParser(description='Set Detection Head', add_help=False)

    # Directories for training and validation datasets
    parser.add_argument('--train_dirs', type=str, nargs='+', required=True, help='List of directories containing the training dataset.')
    parser.add_argument('--val_dirs', type=str, nargs='+', required=True, help='List of directories containing the validation dataset.')
    parser.add_argument('--encoder_name', type=str, required=True, help='Encoder used for calculating embeddings: '
                        '["SAM_base", "MedSAM", "CellSAM", "SAM_large", "MicroSAM_huge", "SAM2_large"]')
    parser.add_argument('--batch_size', type=int, default=4, help='Number of samples in each batch.')
    parser.add_argument('--batches_per_epoch', type=int, default=500, help='Define how many batches are used during training of each epoch.'
                        ' The data is then sampled by a RandomSample instead of using shuffle in the data loader')
    parser.add_argument('--use_sampler', action='store_true', help='If true the model is trained with a fixed size of batches per epoch'
                        'instead of using possibly all data in the datset each epoch. Is useful if you want to train models on multiple datasets and compare them.')

    # Learning rate and optimizer parameters
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate for the optimizer.')
    parser.add_argument('--weight_decay', default=1e-4, type=float, help='Weight decay for the optimizer.')
    parser.add_argument('--lr_drop', default=200, type=int, help='Number of epochs before dropping the learning rate.')

    # Matcher coefficients for computing the matching cost
    parser.add_argument('--set_cost_class', default=1, type=float, help="Class coefficient in the matching cost.")
    parser.add_argument('--set_cost_bbox', default=5, type=float, help="L1 box coefficient in the matching cost.")
    parser.add_argument('--set_cost_giou', default=2, type=float, help="GIoU box coefficient in the matching cost.")

    # Loss coefficients for computing the loss
    parser.add_argument('--bbox_loss_coef', default=5, type=float, help="Coefficient for bounding box loss.")
    parser.add_argument('--giou_loss_coef', default=2, type=float, help="Coefficient for GIoU loss.")
    parser.add_argument('--eos_coef', default=0.1, type=float, help="Relative classification weight of the no-object class.")

    # Model and training parameters
    parser.add_argument('--max_epochs', type=int, default=500, help='Maximum number of epochs for training.')
    parser.add_argument('--num_queries', type=int, default=100, help='Maximum number of queries.')
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

    return parser

def train(args) -> None:
    # Print all parameters before training
    print("Training Parameters:")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")

    ckpt_frequency = 50

    # Initialize the data module with training and validation data paths and batch size
    data_module = DetectionHeadDataModule(
                                        encoder_name=args.encoder_name, 
                                        batch_size=args.batch_size, 
                                        train_dir_names=args.train_dirs,
                                        val_dir_names=args.val_dirs,
                                        batches_per_epoch=args.batches_per_epoch,
                                        use_sampler=args.use_sampler
                                        )

    # Create the model with specified configurations like learning rate and architecture parameters
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

    # for logging:
    train_dirs_str = "_".join(args.train_dirs)
    val_dirs_str = "_".join(args.val_dirs)
    # Initialize a CSV logger to record training progress into a CSV file at specified directory
    # csv_logger = pl_loggers.CSVLogger(args.log_dir)
    csv_logger = pl_loggers.CSVLogger(os.path.join(args.log_dir, args.encoder_name, train_dirs_str,''))
    csv_logger.log_hyperparams({"train_dirs": args.train_dirs, "val_dirs": args.val_dirs})
    

    # Initialize the Wandb logger for experiment tracking and logging
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

    # Prepare a unique checkpointing name combining project and subproject names
    checkpointing_name = f"{args.project_name}_{args.encoder_name}_{train_dirs_str.replace('/', '_')}_{args.sub_name}_{args.use_sampler}_{args.batch_size}_{args.batches_per_epoch}"

    # Configure model checkpointing to save all models every 50 epochs with specific filename format
    ckpt_path = Path("/ictstr01/groups/shared/users/lion.gleiter/organoid_sam/checkpoints_trained/")
    ckpt_path.mkdir(exist_ok=True)
    (ckpt_path / checkpointing_name).mkdir(exist_ok=True)
    checkpoint_callback_regular = ModelCheckpoint(
        save_top_k=1,
        save_last=3,
        every_n_epochs=ckpt_frequency,
        dirpath=ckpt_path / checkpointing_name,
        filename=f"{checkpointing_name}-{{epoch}}-{{val_loss:.2f}}",
        verbose=True,
        save_on_train_epoch_end=False  # Ensures correct handling of ckpt_frequency
    )

    # Set up a monitor to log learning rate changes at each epoch
    lr_monitor = LearningRateMonitor(logging_interval="epoch", log_momentum=False)

    # Configure the PyTorch Lightning trainer with loggers and callbacks, including gradient clipping
    trainer = pl.Trainer(
        logger=[csv_logger, wandb_logger],
        callbacks=[checkpoint_callback_regular, lr_monitor],
        max_epochs=args.max_epochs,
        gradient_clip_val=args.gradient_clip_val,
        check_val_every_n_epoch=ckpt_frequency,
    )

    # Start the model training process with the defined model and data module
    trainer.fit(model, data_module)

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
