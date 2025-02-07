#!/bin/bash

#SBATCH --job-name FasterRCNN_evaluation
#SBATCH -o /home/aih/maximilian.hoermann/SAM_with_Detection_Head/a_slurm_log/%x/%j_output.log
#SBATCH -e /home/aih/maximilian.hoermann/SAM_with_Detection_Head/a_slurm_log/%x/%j_error.log

### ---------- GPU ----------
#SBATCH --exclude=supergpu17
#SBATCH --qos gpu_normal  # gpu_normal 
#SBATCH --partition gpu_p

#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --mem=100G

#SBATCH --time=01-00:00:00 # Job runtime 02-00:00:00
#SBATCH --nice=10000  # Manual priority. Do not change this.

echo $HOME
source $HOME/.bashrc
echo 'Starting evaluate_multiorg_frcnn.sbatch python script'

# Prints GPU info
nvidia-smi

# get to folder
echo "Go to folder /home/aih/maximilian.hoermann/SAM_with_Detection_Head"
cd /home/aih/maximilian.hoermann/SAM_with_Detection_Head

# activate the conda enviroment
echo "Activate Conda environment FasterRCNN"
conda deactivate
conda activate FasterRCNN 
echo "Environment is activated."

# Predictions on Lions code
echo "Start the code!"
python /home/aih/maximilian.hoermann/SAM_with_Detection_Head/scripts/evaluate_multiorg_frcnn.py