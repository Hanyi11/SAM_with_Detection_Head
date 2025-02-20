#!/bin/bash

#SBATCH --job-name extract_patches
#SBATCH -o /ictstr01/groups/shared/users/lion.gleiter/organoid_sam/logs/slurm/%x/%j_output.log
#SBATCH -e /ictstr01/groups/shared/users/lion.gleiter/organoid_sam/logs/slurm/%x/%j_error.log

#SBATCH --qos gpu_preemptible  
# #SBATCH --qos gpu_normal
#SBATCH --partition gpu_p

#SBATCH --cpus-per-task 4
#SBATCH --mem=20G

#SBATCH --gres=gpu:1

#SBATCH -t 01-00:00:00
#SBATCH --nice=10000

echo $HOME
source $HOME/.bashrc

# activate the right enviroment
conda deactivate
conda deactivate
conda deactivate
source /lustre/groups/peng/datasets/lion/data/venvs/sam_dh/bin/activate

python scripts/compute_embeddings.py
