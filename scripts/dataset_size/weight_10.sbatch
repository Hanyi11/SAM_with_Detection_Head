#!/bin/bash

#SBATCH --job-name tmp
#SBATCH -o /home/icb/lion.gleiter/projects/celldeathpred_reanalysis/slurm_logs/%x/%j_output.log
#SBATCH -e /home/icb/lion.gleiter/projects/celldeathpred_reanalysis/slurm_logs/%x/%j_error.log

#SBATCH --qos gpu_max
# #SBATCH --qos gpu_normal
#SBATCH --partition gpu_p

#SBATCH --cpus-per-task 20
#SBATCH --mem=80G
#SBATCH --exclude=supergpu17

#SBATCH --gres=gpu:1

#SBATCH -t 02-00:00:00

#SBATCH --nice=10000  # Manual priority. Do not change this.

echo $HOME
source $HOME/.bashrc

# activate the conda enviroment
conda deactivate
source /lustre/groups/peng/datasets/lion/data/venvs/sam_dh/bin/activate


python train_copy.py --use_sampler --encoder_name SAM_large --balance_validation --balance_newdata 0.1 --train_dirs OrganoID_train MultiOrg_train_macros MultiOrg_train_normal OrgaExtractor_train OrgaQuant_train OrgaSegment_train Tellu_train NewData_train --val_dirs OrganoID_val OrgaExtractor_test OrgaQuant_test OrgaSegment_val Tellu_val --batch_size 32 --batches_per_epoch 200 --sub_name ablation_dataset_10_pre_OI_NeurIPS --max_epochs 800 --checkpoint_path 'DetectionHead_SAM_large_NeurIPSCellSeg_train_pretraining_stage2_True_32_200-last_epoch=499-val_loss=8.78.ckpt'

