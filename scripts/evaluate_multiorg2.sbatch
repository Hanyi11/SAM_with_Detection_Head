#!/bin/bash

#SBATCH --job-name added_eval
#SBATCH -o /ictstr01/groups/shared/users/lion.gleiter/organoid_sam/logs/slurm/%x/%j_output.log
#SBATCH -e /ictstr01/groups/shared/users/lion.gleiter/organoid_sam/logs/slurm/%x/%j_error.log

#SBATCH --qos gpu_normal  
# #SBATCH --qos gpu_long
#SBATCH --partition gpu_p

#SBATCH --cpus-per-task 4
#SBATCH --mem=100G

# #SBATCH --gres=gpu:1
# #SBATCH --constraint="a100_40gb|a100_80gb|h100_80gb"

#SBATCH -t 01-00:00:00
# #SBATCH -t 03-00:00:00
#SBATCH --nice=0

echo $HOME
source $HOME/.bashrc

# activate the right enviroment
conda deactivate
conda deactivate
conda deactivate
source /lustre/groups/peng/datasets/lion/data/venvs/cellsam_evaluation/bin/activate

# python scripts/evaluate_multiorg.py
# python scripts/evaluate_other_testsets.py

# python scripts/evaluate_all_testsets.py
# python scripts/evaluate_all_testsets_umamba.py
# python scripts/evaluate_all_testsets_orgasegment.py
# python scripts/evaluate_sam_segmentation_with_gt.py
python scripts/evaluate_all_testsets_from_saved_files.py
