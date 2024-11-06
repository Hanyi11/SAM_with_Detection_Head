#!/bin/bash

#SBATCH --job-name tmp
#SBATCH -o /home/icb/lion.gleiter/projects/celldeathpred_reanalysis/slurm_logs/%x/%j_output.log
#SBATCH -e /home/icb/lion.gleiter/projects/celldeathpred_reanalysis/slurm_logs/%x/%j_error.log

#SBATCH --qos gpu_max
#SBATCH --partition gpu_p

#SBATCH --cpus-per-task 1
#SBATCH --mem=16G

#SBATCH -t 02-00:00:00


# #SBATCH --qos cpu_short
# #SBATCH --partition cpu_p

# #SBATCH --cpus-per-task 1
# #SBATCH --mem=16G

# #SBATCH -t 00-05:00:00

#SBATCH --nice=10000  # Manual priority. Do not change this.

echo $HOME
source $HOME/.bashrc

# activate the conda enviroment
conda deactivate
source /lustre/groups/peng/datasets/lion/data/venvs/sam_dh/bin/activate

python scripts/extract_patches.py
