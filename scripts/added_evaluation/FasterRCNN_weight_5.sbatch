#!/bin/bash

#SBATCH --job-name FasterRCNN
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
echo 'Starting FasterRCNN_weight_5.sh python script'

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
# python train_FasterRCNN.py --use_sampler --version_FasterRCNN v1 --balance_validation --balance_newdata 0.05 --train_dirs OrganoID_train NewData_train --val_dirs OrganoID_val NewData_val --batch_size 16 --batches_per_epoch 200 --sub_name FasterRCNN_16012025 --max_epochs 800
python train_FasterRCNN.py --use_sampler --version_FasterRCNN v2 --balance_validation --balance_newdata 0.05 --train_dirs OrganoID_train MultiOrg_train_macros MultiOrg_train_normal OrgaExtractor_train OrgaQuant_train OrgaSegment_train Tellu_train NewData_train --val_dirs OrganoID_val MultiOrg_val_macros MultiOrg_val_normal OrgaExtractor_val OrgaQuant_val OrgaSegment_val Tellu_val NewData_val --batch_size 16 --batches_per_epoch 200 --sub_name FasterRCNN_16012025 --max_epochs 800
