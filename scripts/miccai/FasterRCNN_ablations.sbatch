#!/bin/bash

#SBATCH --job-name extract_patches
#SBATCH -o /ictstr01/groups/shared/users/lion.gleiter/organoid_sam/logs/slurm/%x/%j_output.log
#SBATCH -e /ictstr01/groups/shared/users/lion.gleiter/organoid_sam/logs/slurm/%x/%j_error.log

### ---------- GPU ----------
# #SBATCH --exclude=supergpu17
#SBATCH --qos gpu_normal  # gpu_normal 
#SBATCH --partition gpu_p

#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=50G

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

python train_general.py +baseline_models=frcnn sub_name="baseline_frcnn" seed=2024
# # python train_general.py +baseline_models=frcnn sub_name="baseline_frcnn" seed=2025
# # python train_general.py +baseline_models=frcnn sub_name="baseline_frcnn" seed=2026

python train_general.py +baseline_models=frcnn_v2 sub_name="baseline_frcnn_v2" seed=2024
python train_general.py +baseline_models=ssd sub_name="baseline_ssd" seed=2024

python train_general.py +baseline_models=frcnn_v2 sub_name="baseline_frcnn_v2_sampling_per_datasets" sampling_groups=["dataset"] seed=2024
python train_general.py +baseline_models=frcnn_v2 sub_name="baseline_frcnn_v2_sampling_per_object_count" sampling_groups=["n_objects_grouped"] seed=2024
python train_general.py +baseline_models=frcnn_v2 sub_name="baseline_frcnn_v2_sampling_per_datasets_and_object_count" sampling_groups=["dataset","n_objects_grouped"] seed=2024
