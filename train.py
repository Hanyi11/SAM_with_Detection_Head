import argparse
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from detection_head_datamodule import DetectionHeadDataModule
from detection_head_model import DetectionHead
import wandb
# --train_dir="" --val_dir="" --sub_name="SAM_large"
def get_args_parser():
    parser = argparse.ArgumentParser(description='Set Detection Head', add_help=False)

    # Directories for training and validation datasets
    parser.add_argument('--train_dir', type=str, required=True, help='Directory containing the training dataset.')
    parser.add_argument('--val_dir', type=str, required=True, help='Directory containing the validation dataset.')
    parser.add_argument('--batch_size', type=int, default=4, help='Number of samples in each batch.')

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

    # Initialize the data module with training and validation data paths and batch size
    data_module = DetectionHeadDataModule(train_dir=args.train_dir, val_dir=args.val_dir, batch_size=args.batch_size)

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

    # Initialize a CSV logger to record training progress into a CSV file at specified directory
    csv_logger = pl_loggers.CSVLogger(args.log_dir)

    # Initialize the Wandb logger for experiment tracking and logging
    wandb_logger = pl_loggers.WandbLogger(
        name=f"{args.project_name}_{args.sub_name}",
        project=args.project_name,
        log_model=True,
        save_dir=args.log_dir
    )

    # Prepare a unique checkpointing name combining project and subproject names
    checkpointing_name = f"{args.project_name}_{args.sub_name}"

    # Configure model checkpointing to save all models every 50 epochs with specific filename format
    checkpoint_callback_regular = ModelCheckpoint(
        save_top_k=-1,
        every_n_epochs=50,
        dirpath="checkpoints/",
        filename=f"{checkpointing_name}-{{epoch}}-{{val_loss:.2f}}",
        verbose=True
    )

    # Set up a monitor to log learning rate changes at each epoch
    lr_monitor = LearningRateMonitor(logging_interval="epoch", log_momentum=False)

    # Configure the PyTorch Lightning trainer with loggers and callbacks, including gradient clipping
    trainer = pl.Trainer(
        logger=[csv_logger, wandb_logger],
        callbacks=[checkpoint_callback_regular, lr_monitor],
        max_epochs=args.max_epochs,
        gradient_clip_val=args.gradient_clip_val,
    )

    # Start the model training process with the defined model and data module
    trainer.fit(model, data_module)

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser('Detection Head training script', parents=[get_args_parser()])
    args = parser.parse_args()

    # Initialize wandb
    wandb.init(project=args.project_name, name=f"{args.project_name}_{args.sub_name}")

    # Start training
    train(args)

    # Finish the wandb run
    wandb.finish()
