import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset, RandomSampler, SubsetRandomSampler
import torch
import os
import numpy as np
from typing import Literal, List

def get_subdirectories_from_dir(directory: str, dir_beginning: str = '', keywords: List[str] = []) -> List[str]:
    """
    Get subdirectories from a directory with specific criteria.

    Args:
        directory (str): Path to the directory.
        dir_beginning (str, optional): Beginning of the directory names you are looking for. Defaults to ''.
        keywords (List[str], optional): List of keywords that have to be in the directory name. Defaults to [].

    Returns:
        matching_dirs (List[str]): Subdirectories in the directory which meet the specified criteria.
    """
    matching_dirs = []
    
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        if os.path.isdir(item_path) and item.startswith(dir_beginning):
            if all(keyword in item for keyword in keywords):
                matching_dirs.append(item_path)
    
    return matching_dirs

def get_files_from_dir(directory: str, file_ending: str = '.png', file_beginning: str = '', keyword: str = '') -> List[str]:
    """
    Get files from a directory with specific criteria.

    Args:
        directory (str): Path to directory with files.
        file_ending (str, optional): Ending of file. Defaults to '.png'.
        file_beginning (str, optional): Beginning of the files you are looking for. Defaults to ''.
        keyword (str, optional): Keyword that has to be in the file name. Defaults to ''.

    Returns:
        matching_files (List[str]): Files in the directory which meet the specified criteria.
    """
    matching_files = []
    
    for filename in os.listdir(directory):
        if (filename.startswith(file_beginning) and
            filename.endswith(file_ending) and
            keyword in filename):
            matching_files.append(os.path.join(directory, filename))
    
    return matching_files



class DetectionHeadDataset(Dataset):
    def __init__(self, 
                 data_split_dirs: List[str],
                 data_split: Literal["train", "test", "val"] = "train", 
                 encoder_name: Literal["SAM_base", "MedSAM", "CellSAM", 
                                       "SAM_large", "MicroSAM_huge", "SAM2_large"] = "SAM_base",
                 num_queries: int = 300, 
                 base_dir: str = "/ictstr01/groups/shared/users/lion.gleiter/organoid_sam/"):
        super().__init__()
        
        # Set attributes
        self.data_split_dirs = data_split_dirs
        self.base_dir = base_dir
        self.num_queries = num_queries
        self.data_split = data_split
        self.encoder_name = encoder_name

        # ---- Get sorted paths of Image embeddings ----
        # Define the parent directory
        parent_dir: str = os.path.join(self.base_dir, "patch_embeddings", self.encoder_name, self.data_split)
        datasets_dirs: List[str] = [os.path.join(parent_dir, data_split_dir) for data_split_dir in data_split_dirs]

        # Get all datasets directories and their subdirectories in a single step
        merged_image_dirs: List[str] = [
            subdir for datasets_dir in datasets_dirs
            for subdir in get_subdirectories_from_dir(datasets_dir)
        ]

        # Collect all image paths from subdirectories
        merged_image_paths: List[str] = [
            image_path for image_dir in merged_image_dirs
            for image_path in get_files_from_dir(image_dir, file_ending=".npy")
        ]

        # Sort all .npy files from the merged image paths
        self.image_files: List[str] = sorted(merged_image_paths)


        # ---- Get sorted paths of GT-BBox labels ----
        parent_dir_gt: str = os.path.join(self.base_dir, "patch_bbox_gt", self.data_split)
        datasets_dirs_gt: List[str] = [os.path.join(parent_dir_gt, data_split_dir) for data_split_dir in data_split_dirs]

        # Get all datasets directories and their subdirectories in a single step
        merged_gt_dirs: List[str] = [
            subdir for dataset_dir in datasets_dirs_gt 
            for subdir in get_subdirectories_from_dir(dataset_dir)
        ]

        # Collect all gt paths from subdirectories
        merged_gt_paths: List[str] = [
            gt_path for gt_dir in merged_gt_dirs 
            for gt_path in get_files_from_dir(gt_dir, file_beginning="patch" ,file_ending=".npy")
        ]
        
        # Sort all .npy files from merged gt paths
        self.label_files: List[str] = sorted(merged_gt_paths)
        

        # ---- Ensure that the image files and label files match
        assert len(self.image_files) == len(self.label_files), f"Mismatch between number of images ({len(self.image_files)}) and labels ({len(self.label_files)})"
        for img_file, label_file in zip(self.image_files, self.label_files):
            # Get path infos
            img_base, img_name = os.path.split(img_file)
            label_base, label_name = os.path.split(label_file)

            # Get rid of encoder_name part in img_base:
            img_base_cleaned = img_base.replace("patch_embeddings/" + self.encoder_name, "")
            label_base_cleaned = label_base.replace("patch_bbox_gt", "")

            # "/ictstr01/groups/shared/users/lion.gleiter/organoid_sam/patch_embeddings/CellSAM/test/OrganoID_test/im_PDAC1/images_patches_emb_0000.npy"
            # "/ictstr01/groups/shared/users/lion.gleiter/organoid_sam/patch_bbox_gt/test/OrganoID_test/im_PDAC1/patch_0.npy"
            
            # Convert img_name from image-nameing-convention to label-naming-convention
            patch_number = int(img_name.split('_')[-1].split('.')[0])  # Extracts '0000' and converts to integer: 0000 -> 0
            img_name_converted = f"patch_{patch_number}.npy"

            # Reconstruct cleaned paths for comparison
            img_path_cleaned = os.path.join(img_base_cleaned, img_name_converted)
            label_path_cleaned = os.path.join(label_base_cleaned, label_name)
            
            # Compare
            img_path_cleaned == label_path_cleaned, f"Filename mismatch: {img_file} and {label_file} are not the right img_patch and gt_label pair!"

    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        label_path = self.label_files[idx]

        # Load image
        image = np.load(image_path)
        
        # Load targets
        targets = np.load(label_path)

        # Remove the first dimension if it exists
        if image.shape[0] == 1:
            image = np.squeeze(image, axis=0)
        
        # Ensure image is of shape (256, 64, 64)
        if image.shape != (256, 64, 64):
            image = image.transpose((2, 0, 1))
            if image.shape != (256, 64, 64):
                raise ValueError(f"Unexpected image shape: {image.shape}, expected (256, 64, 64)")

        # Pad targets to ensure they are of shape (num_queries, 4)
        num_boxes = targets.shape[0]
        if num_boxes < self.num_queries:
            pad_size = self.num_queries - num_boxes
            padded_targets = np.pad(targets, ((0, pad_size), (0, 0)), mode='constant', constant_values=0)
        else:
            padded_targets = targets[:self.num_queries]

        # Convert to torch tensors
        image = torch.tensor(image, dtype=torch.float32)
        padded_targets = torch.tensor(padded_targets, dtype=torch.float32)
        
        return image, padded_targets

class DetectionHeadDataModule(pl.LightningDataModule):
    def __init__(self, 
                 train_dir_names: List[str],
                 val_dir_names: List[str],
                 encoder_name: Literal["SAM_base", "MedSAM", "CellSAM", 
                                       "SAM_large", "MicroSAM_huge", "SAM2_large"] = "SAM_base",
                 batch_size: int = 32, 
                 num_queries: int = 300,
                 batches_per_epoch: int = 411,
                 use_sampler: bool = False
                 ):
        super().__init__()

        self.train_dir_names = train_dir_names
        self.val_dir_names = val_dir_names
        self.encoder_name = encoder_name
        self.batch_size = batch_size
        self.num_queries = num_queries
        self.batches_per_epoch = batches_per_epoch
        self.use_sampler = use_sampler

        
    def setup(self, stage: Literal["fit", "validate", None] = None):
        if stage == 'fit' or stage is None:
            self.train_dataset = DetectionHeadDataset(data_split_dirs=self.train_dir_names, 
                                                      data_split="train", 
                                                      encoder_name=self.encoder_name, 
                                                      num_queries=self.num_queries)
            
            self.val_dataset = DetectionHeadDataset(data_split_dirs=self.val_dir_names,
                                                    data_split="val", 
                                                    encoder_name=self.encoder_name, 
                                                    num_queries=self.num_queries)
        
        if stage == 'validate' or stage is None:
            self.val_dataset = DetectionHeadDataset(data_split_dirs=self.val_dir_names, 
                                                    data_split="val", 
                                                    encoder_name=self.encoder_name, 
                                                    num_queries=self.num_queries)

    def train_dataloader(self):
        if self.use_sampler == True:
            sampler = RandomSampler(
                                    data_source=self.train_dataset,
                                    replacement=True,
                                    num_samples=self.batches_per_epoch * self.batch_size
                                    )

            return DataLoader(self.train_dataset, 
                              batch_size=self.batch_size, 
                              sampler=sampler, 
                              num_workers=4)
        else: 
            return DataLoader(self.train_dataset, 
                              batch_size=self.batch_size, 
                              shuffle=True, 
                              num_workers=4)

    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, 
                          batch_size=self.batch_size, 
                          num_workers=4)

if __name__ == "__main__":
    from tqdm import tqdm 

    print("Start DataLoader test!")
    
    data_split: Literal["train", "test", "val"] = "val"
    data_split_dirs: List[str] = ["NeurIPSCellSeg_val", "MultiOrg_test_macros", "MultiOrg_test_normal", 
                                  "OrgaQuant_test", "OrgaSegment_val","OrganoID_val", "Tellu_val"]

    print(f"data split: {data_split}")
    # encoder_name:  Literal["SAM_base", "MedSAM", "CellSAM", "SAM_large", "MicroSAM_huge", "SAM2_large"] = "SAM_base"
    encoder_names: List = ["SAM_base", "MedSAM", "CellSAM", "SAM_large", "MicroSAM_huge", "SAM2_large"]
    

    for encoder_name in encoder_names:
        print(f"encoder name: {encoder_name}")
        dataset = DetectionHeadDataset(
            data_split_dirs=data_split_dirs,
            encoder_name=encoder_name, 
            data_split=data_split)

        i = 0 
        for  image, padded_targets in tqdm(dataset):
            i += 1
            if i %100 == 0:
                print(f"{i}th image is calculated")
                print(f"image shape is: {image.size()}")
                print(f"padded targets is: {padded_targets.size()}")

        
        print(f"Dataset has {i} images.")
        print()
    print(f"image embedding of {data_split} worked")