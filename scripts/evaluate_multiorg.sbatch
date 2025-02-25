#!/bin/bash

#SBATCH --job-name extract_patches
#SBATCH -o /ictstr01/groups/shared/users/lion.gleiter/organoid_sam/logs/slurm/%x/%j_output.log
#SBATCH -e /ictstr01/groups/shared/users/lion.gleiter/organoid_sam/logs/slurm/%x/%j_error.log

# #SBATCH --qos gpu_preemptible  # gpu_normal  
# #SBATCH --qos gpu_long
# #SBATCH --partition gpu_p

#SBATCH --qos cpu_normal
#SBATCH --partition cpu_p

#SBATCH --cpus-per-task 8
#SBATCH --mem=150G

# #SBATCH --gres=gpu:1
# #SBATCH --constraint="a100_40gb|a100_80gb|h100_80gb"

# #SBATCH -t 01-00:00:00
#SBATCH -t 03-00:00:00
#SBATCH --nice=10000

echo $HOME
source $HOME/.bashrc

# activate the right enviroment
conda deactivate
conda deactivate
conda deactivate
source /lustre/groups/peng/datasets/lion/data/venvs/sam_dh/bin/activate

# python scripts/evaluate_multiorg.py
# python scripts/evaluate_other_testsets.py

# python scripts/evaluate_all_testsets.py
python scripts/evaluate_all_testsets_from_saved_files.py
python scripts/evaluate_sam_segmentation_with_gt.py
