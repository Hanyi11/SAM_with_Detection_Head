#!/bin/bash

#SBATCH --job-name extract_patches
#SBATCH -o /ictstr01/groups/shared/users/lion.gleiter/organoid_sam/logs/slurm/%x/%j_output.log
#SBATCH -e /ictstr01/groups/shared/users/lion.gleiter/organoid_sam/logs/slurm/%x/%j_error.log

#SBATCH --qos cpu_normal
#SBATCH --partition cpu_p

#SBATCH --cpus-per-task 2
#SBATCH --mem=80G

#SBATCH -t 02-00:00:00
#SBATCH --nice=10000

echo $HOME
source $HOME/.bashrc

# activate the right enviroment
conda deactivate
conda deactivate
conda deactivate
source /lustre/groups/peng/datasets/lion/data/venvs/sam_dh/bin/activate

python scripts/extract_patches.py
