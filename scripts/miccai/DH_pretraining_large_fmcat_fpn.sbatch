#!/bin/bash

#SBATCH --job-name extract_patches
#SBATCH -o /ictstr01/groups/shared/users/lion.gleiter/organoid_sam/logs/slurm/%x/%j_output.log
#SBATCH -e /ictstr01/groups/shared/users/lion.gleiter/organoid_sam/logs/slurm/%x/%j_error.log

### ---------- GPU ----------
# #SBATCH --exclude=supergpu17
#SBATCH --qos gpu_normal 
# #SBATCH --qos gpu_preemptible 
#SBATCH --partition gpu_p

#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=28
#SBATCH --mem=100G
# #SBATCH --constraint="a100_40gb|a100_80gb|h100_80gb"

#SBATCH --time=01-00:00:00 # Job runtime 02-00:00:00
#SBATCH --nice=10000  # Manual priority. Do not change this.

echo $HOME
source $HOME/.bashrc
echo 'Starting python script'

# Prints GPU info
nvidia-smi


# activate the conda enviroment
conda deactivate
source /lustre/groups/peng/datasets/lion/data/venvs/sam_dh/bin/activate

python train_general.py +baseline_models=detr_own_fmcat_large_head_fpn sub_name="DETR_own_FM_concat_large_head_FPN_pretraining" sampling_groups=["n_objects"] seed=2024 train_dirs=["open_images_v4_5/original_data","NeurIPSCellSeg_train"] val_dirs=["NeurIPSCellSeg_val"]
