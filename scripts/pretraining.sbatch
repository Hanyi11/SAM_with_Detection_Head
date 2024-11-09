#!/bin/bash

#SBATCH --job-name tmp
#SBATCH -o /home/icb/lion.gleiter/projects/celldeathpred_reanalysis/slurm_logs/%x/%j_output.log
#SBATCH -e /home/icb/lion.gleiter/projects/celldeathpred_reanalysis/slurm_logs/%x/%j_error.log

#SBATCH --qos gpu_max
#SBATCH --partition gpu_p

#SBATCH --cpus-per-task 30
#SBATCH --mem=80G

#SBATCH --gres=gpu:1

#SBATCH -t 02-00:00:00

#SBATCH --nice=10000  # Manual priority. Do not change this.

echo $HOME
source $HOME/.bashrc

# activate the conda enviroment
conda deactivate
source /lustre/groups/peng/datasets/lion/data/venvs/sam_dh/bin/activate

# python train_copy.py --use_sampler --encoder_name SAM_large  --train_dirs NeurIPSCellSeg_train --val_dirs NeurIPSCellSeg_val --batch_size 32 --sub_name pretraining --batches_per_epoch 200 --max_epochs 800

python train_copy.py --use_sampler --encoder_name SAM_large  --train_dirs "open_images_embeddings/validation" --val_dirs NeurIPSCellSeg_val --batch_size 32 --sub_name pretraining --batches_per_epoch 200 --max_epochs 800
