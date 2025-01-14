#!/bin/bash

#SBATCH --job-name added_eval
#SBATCH -o /ictstr01/groups/shared/users/lion.gleiter/organoid_sam/logs/slurm/%x/%j_output.log
#SBATCH -e /ictstr01/groups/shared/users/lion.gleiter/organoid_sam/logs/slurm/%x/%j_error.log

# #SBATCH --qos gpu_max
#SBATCH --qos gpu_normal
#SBATCH --partition gpu_p

#SBATCH --cpus-per-task 20
#SBATCH --mem=100G
#SBATCH --exclude=supergpu17

#SBATCH --gres=gpu:1

#SBATCH -t 01-00:00:00

#SBATCH --nice=10000  # Manual priority. Do not change this.

echo $HOME
source $HOME/.bashrc

# activate the conda enviroment
conda deactivate
source /lustre/groups/peng/datasets/lion/data/venvs/sam_dh/bin/activate


# python train_SSD.py --use_sampler --balance_validation --balance_newdata 0.05 --train_dirs MultiOrg_train_macros MultiOrg_train_normal --val_dirs MultiOrg_val_macros MultiOrg_val_normal --batch_size 32 --batches_per_epoch 200 --sub_name multiorg_data_09012025 --max_epochs 800
python train_SSD.py --use_sampler --balance_validation --balance_newdata 0.05 --train_dirs MultiOrg_train_macros MultiOrg_train_normal --val_dirs MultiOrg_val_macros MultiOrg_val_normal --batch_size 32 --batches_per_epoch 200 --sub_name multiorg_with_090_overlap --max_epochs 800

