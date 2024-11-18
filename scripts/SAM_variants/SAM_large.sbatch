#!/bin/bash

#SBATCH --job-name tmp
#SBATCH -o /home/icb/lion.gleiter/projects/celldeathpred_reanalysis/slurm_logs/%x/%j_output.log
#SBATCH -e /home/icb/lion.gleiter/projects/celldeathpred_reanalysis/slurm_logs/%x/%j_error.log

# #SBATCH --qos gpu_max
#SBATCH --qos gpu_normal
#SBATCH --partition gpu_p

#SBATCH --cpus-per-task 10
#SBATCH --mem=80G

#SBATCH --gres=gpu:1

#SBATCH -t 02-00:00:00

#SBATCH --nice=10000  # Manual priority. Do not change this.

echo $HOME
source $HOME/.bashrc

# activate the conda enviroment
conda deactivate
source /lustre/groups/peng/datasets/lion/data/venvs/sam_dh/bin/activate


# python train_copy.py --encoder_name SAM_large  --train_dirs OrganoID_train --val_dirs OrganoID_val --batch_size 32 --sub_name finetune_specialist --max_epochs 500
# python train_copy.py --encoder_name SAM_large  --train_dirs OrganoID_train --val_dirs OrganoID_val --batch_size 32 --sub_name finetune_specialist_pre_OI --max_epochs 500 --checkpoint_path 'DetectionHead_SAM_large_open_images_embeddings_validation_pretraining_True_32_200-epoch=799-val_loss=29.34.ckpt'
python train_copy.py --encoder_name SAM_large  --train_dirs OrganoID_train --val_dirs OrganoID_val --batch_size 32 --sub_name finetune_specialist_pre_NeurIPS --max_epochs 500 --checkpoint_path 'DetectionHead_SAM_large_NeurIPSCellSeg_train_pretraining_True_32_200-last_epoch=799-val_loss=9.49.ckpt'
# python train_copy.py --encoder_name SAM_large  --train_dirs OrganoID_train --val_dirs OrganoID_val --batch_size 32 --sub_name finetune_specialist_pre_OI_NeurIPS --max_epochs 500 --checkpoint_path 'DetectionHead_SAM_large_NeurIPSCellSeg_train_pretraining_stage2_True_32_200-last_epoch=499-val_loss=8.78.ckpt'
