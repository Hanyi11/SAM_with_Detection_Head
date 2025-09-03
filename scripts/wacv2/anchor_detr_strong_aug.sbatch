#!/bin/bash

#SBATCH --job-name train_detection
#SBATCH -o /ictstr01/groups/shared/users/lion.gleiter/organoid_sam/logs/slurm/%x/%j_output.log
#SBATCH -e /ictstr01/groups/shared/users/lion.gleiter/organoid_sam/logs/slurm/%x/%j_error.log

### ---------- GPU ----------
# #SBATCH --exclude=supergpu17
#SBATCH --qos gpu_priority
# #SBATCH --qos gpu_long
#SBATCH --partition gpu_p

#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=15
#SBATCH --mem=50G
#SBATCH --constraint=a100_80gb

#SBATCH --time=00-12:00:00 # Job runtime 02-00:00:00
# #SBATCH --begin=now+18hours
#SBATCH --nice=10000  # Manual priority. Do not change this.

# Notification
#SBATCH --mail-type=END,FAIL  # Send email on END,FAIL events
#SBATCH --mail-user=lion.gleiter@helmholtz-munich.de  # Email address to send notifications

echo $HOME
source $HOME/.bashrc
echo 'Starting python script'

# Prints GPU info
nvidia-smi


# activate the conda enviroment
conda deactivate
source ../venvs/anchor_detr/bin/activate

# # python train_general.py +baseline_models=anchor_detr sub_name='"${hydra:runtime.choices.baseline_models}_pretraining_NeurIPS_superlong_focal_alpha_09"' sampling_groups=["n_objects"] seed=null train_dirs=["NeurIPSCellSeg_train"] val_dirs=["NeurIPSCellSeg_val"] lr_drop=100000 max_epochs=2500

# python train_general.py +baseline_models=anchor_detr sub_name='"${hydra:runtime.choices.baseline_models}_finetuning_long"' sampling_groups=["n_objects","dataset"] lr_drop=900 max_epochs=1200 seed=null run=2024 checkpoint_path='"${decoder}_${backbone_name}_NeurIPSCellSeg_train_${hydra:runtime.choices.baseline_models}_pretraining_NeurIPS_superlong_focal_alpha_09_True_${batch_size}_${batches_per_epoch}_2024-last.ckpt"' finetune_only_class_linear=false
# # python train_general.py +baseline_models=anchor_detr sub_name='"${hydra:runtime.choices.baseline_models}_finetuning_long"' sampling_groups=["n_objects","dataset"] lr_drop=900 max_epochs=1200 seed=null run=2025 checkpoint_path='"${decoder}_${backbone_name}_NeurIPSCellSeg_train_${hydra:runtime.choices.baseline_models}_pretraining_NeurIPS_superlong_focal_alpha_09_True_${batch_size}_${batches_per_epoch}_2024-last.ckpt"' finetune_only_class_linear=false
# # python train_general.py +baseline_models=anchor_detr sub_name='"${hydra:runtime.choices.baseline_models}_finetuning_long"' sampling_groups=["n_objects","dataset"] lr_drop=900 max_epochs=1200 seed=null run=2026 checkpoint_path='"${decoder}_${backbone_name}_NeurIPSCellSeg_train_${hydra:runtime.choices.baseline_models}_pretraining_NeurIPS_superlong_focal_alpha_09_True_${batch_size}_${batches_per_epoch}_2024-last.ckpt"' finetune_only_class_linear=false

python test_custom_checkpoint.py +custom_output_name=anchor_detr_1200_best +custom_ckpt_file='"1200_epochs_best/anchor_detr_resnet50_strong_aug_finetuning_long_best_epoch=1074_val_loss=3.72.ckpt"' +baseline_models=anchor_detr sub_name='"${hydra:runtime.choices.baseline_models}_finetuning_long"' sampling_groups=["n_objects","dataset"] lr_drop=900 max_epochs=1200 seed=null run=2024 checkpoint_path='"${decoder}_${backbone_name}_NeurIPSCellSeg_train_${hydra:runtime.choices.baseline_models}_pretraining_NeurIPS_superlong_focal_alpha_09_True_${batch_size}_${batches_per_epoch}_2024-last.ckpt"' finetune_only_class_linear=false
