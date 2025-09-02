#!/bin/bash

#SBATCH --job-name added_eval
#SBATCH -o /ictstr01/groups/shared/users/lion.gleiter/organoid_sam/logs/slurm/%x/%j_output.log
#SBATCH -e /ictstr01/groups/shared/users/lion.gleiter/organoid_sam/logs/slurm/%x/%j_error.log

#SBATCH --qos gpu_priority  # gpu_normal 
# #SBATCH --qos gpu_preemptible  # gpu_normal  
# #SBATCH --qos gpu_long
#SBATCH --partition gpu_p

# #SBATCH --qos cpu_normal
# #SBATCH --partition cpu_p
# #SBATCH --reservation=res_mojtaba.bahrami

#SBATCH --cpus-per-task 2
#SBATCH --mem=50G

#SBATCH --gres=gpu:1
# #SBATCH --constraint="a100_40gb|a100_80gb|h100_80gb"

# #SBATCH -t 01-00:00:00
#SBATCH -t 00-12:00:00
#SBATCH --nice=10000

echo $HOME
source $HOME/.bashrc

# activate the right enviroment
conda deactivate
source /lustre/groups/peng/datasets/lion/data/venvs/cellsam_evaluation/bin/activate

# python scripts/evaluate_multiorg.py
# python scripts/evaluate_other_testsets.py

# python scripts/evaluate_all_testsets.py
# python scripts/evaluate_all_testsets_umamba.py
# python scripts/evaluate_all_testsets.py
# python scripts/evaluate_all_testsets_orgasegment.py
# python scripts/evaluate_all_testsets_from_saved_files_AP_only.py
# python scripts/evaluate_sam_segmentation_with_gt.py
python scripts/evaluate_all_testsets_from_saved_files_segmentation.py

deactivate
