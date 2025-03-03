#!/bin/bash

#SBATCH --job-name extract_patches
#SBATCH -o /ictstr01/groups/shared/users/lion.gleiter/organoid_sam/logs/slurm/%x/%j_output.log
#SBATCH -e /ictstr01/groups/shared/users/lion.gleiter/organoid_sam/logs/slurm/%x/%j_error.log

### ---------- GPU ----------
# #SBATCH --exclude=supergpu17
#SBATCH --qos gpu_normal
# #SBATCH --qos gpu_preemptible 
#SBATCH --partition gpu_p

#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=25
#SBATCH --mem=80G
#SBATCH --reservation=res_mojtaba.bahrami
# #SBATCH --constraint="a100_40gb|a100_80gb|a100_20gb"

#SBATCH --time=01-00:00:00 # Job runtime 02-00:00:00
#SBATCH --nice=0  # Manual priority. Do not change this.

echo $HOME
source $HOME/.bashrc
echo 'Starting python script'

# Prints GPU info
nvidia-smi


# activate the conda enviroment
conda deactivate
source /lustre/groups/peng/datasets/lion/data/venvs/sam_dh/bin/activate

# python train_general.py +baseline_models=detr_own_sam1_small_head sub_name="DETR_own_SAM1_small_head_pretraining" sampling_groups=["n_objects"] seed=2024 train_dirs=["open_images_v4_5/original_data","NeurIPSCellSeg_train"] val_dirs=["NeurIPSCellSeg_val"]
# python train_general.py +baseline_models=detr_own_sam1_small_head sub_name="DETR_own_SAM1_small_head_pretraining_stage2" sampling_groups=["n_objects"] seed=2024 train_dirs=["open_images_v4_5/original_data","NeurIPSCellSeg_train"] val_dirs=["NeurIPSCellSeg_val"] checkpoint_path="DETR_own_implementation_SAM_large_open_images_v4_5_original_data_NeurIPSCellSeg_train_DETR_own_SAM1_small_head_pretraining_True_4_400_2024-last.ckpt"
# python train_general.py +baseline_models=detr_own_sam1_small_head sub_name="DETR_own_SAM1_small_head_pretraining_only_neurips" sampling_groups=["n_objects"] seed=2024 train_dirs=["NeurIPSCellSeg_train"] val_dirs=["NeurIPSCellSeg_val"]
# python train_general.py +baseline_models=detr_own_sam1_small_head sub_name="DETR_own_SAM1_small_head_pretraining_only_neurips_fixed_lr" sampling_groups=["n_objects"] seed=2024 train_dirs=["NeurIPSCellSeg_train"] val_dirs=["NeurIPSCellSeg_val"] lr_drop=10000 learning_rate=1e-5
# python train_general.py +baseline_models=detr_own_sam1_small_head sub_name="DETR_own_SAM1_small_head_pretraining_only_neurips_fixed_lr_1e4" sampling_groups=["n_objects"] seed=2024 train_dirs=["NeurIPSCellSeg_train"] val_dirs=["NeurIPSCellSeg_val"] max_epochs=2000 lr_drop=10000 learning_rate=1e-4

# python train_general.py +baseline_models=detr_own_sam1_small_head sub_name='"${hydra:runtime.choices.baseline_models}_pretraining_stage1_OI"' sampling_groups=["n_objects"] seed=2024 train_dirs=["open_images_v4_5/original_data"] val_dirs=["NeurIPSCellSeg_val"] lr_drop=200 max_epochs=300
# python train_general.py +baseline_models=detr_own_sam1_small_head sub_name='"${hydra:runtime.choices.baseline_models}_pretraining_stage2_NeurIPS"' sampling_groups=["n_objects"] seed=2024 train_dirs=["NeurIPSCellSeg_train"] val_dirs=["NeurIPSCellSeg_val"] lr_drop=200 max_epochs=300 checkpoint_path='"${decoder}_${backbone_name}_open_images_v4_5_original_data_${hydra:runtime.choices.baseline_models}_pretraining_stage1_OI_True_${batch_size}_${batches_per_epoch}_2024-last.ckpt"'

# python train_general.py +baseline_models=detr_own_sam1_small_head sub_name='"${hydra:runtime.choices.baseline_models}_finetuning_long_extended_bb_loss_version_2"' sampling_groups=["n_objects","dataset"] lr_drop=150 max_epochs=350 seed=2024 checkpoint_path='"DETR_own_implementation_SAM_large_MultiOrg_train_normal_MultiOrg_train_macros_OrgaSegment_train_Tellu_train_OrgaQuant_train_detr_own_sam1_small_head_finetuning_long_version_2_True_16_400_2024-last.ckpt"' bbox_loss_coef=20
# python train_general.py +baseline_models=detr_own_sam1_small_head sub_name='"${hydra:runtime.choices.baseline_models}_finetuning_long_extended_bb_loss_version_2"' sampling_groups=["n_objects","dataset"] lr_drop=150 max_epochs=350 seed=2025 checkpoint_path='"DETR_own_implementation_SAM_large_MultiOrg_train_normal_MultiOrg_train_macros_OrgaSegment_train_Tellu_train_OrgaQuant_train_detr_own_sam1_small_head_finetuning_long_version_2_True_16_400_2025-last.ckpt"' bbox_loss_coef=20
python train_general.py +baseline_models=detr_own_sam1_small_head sub_name='"${hydra:runtime.choices.baseline_models}_finetuning_long_extended_bb_loss_version_2"' sampling_groups=["n_objects","dataset"] lr_drop=150 max_epochs=350 seed=2026 checkpoint_path='"DETR_own_implementation_SAM_large_MultiOrg_train_normal_MultiOrg_train_macros_OrgaSegment_train_Tellu_train_OrgaQuant_train_detr_own_sam1_small_head_finetuning_long_version_2_True_16_400_2026-last.ckpt"' bbox_loss_coef=20
