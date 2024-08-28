import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
import torch
import os
import numpy as np

class DetectionHeadDataset(Dataset):
    def __init__(self, data_dir, num_queries=300):
        super().__init__()
        self.data_dir = data_dir
        self.num_queries = num_queries

        self.images_dir = os.path.join(data_dir, 'images_patches_emb')
        self.labels_dir = os.path.join(data_dir, 'targets')

        self.image_files = sorted([f for f in os.listdir(self.images_dir) if f.endswith('.npy')])
        self.label_files = sorted([f for f in os.listdir(self.labels_dir) if f.endswith('.npy')])

        # Ensure that the number of image files and label files match
        assert len(self.image_files) == len(self.label_files), "Mismatch between number of images and labels"
        for img_file, label_file in zip(self.image_files, self.label_files):
            img_name = os.path.splitext(img_file)[0]
            label_name = os.path.splitext(label_file)[0]
            assert img_name == label_name.replace('_label_', '_img_'), f"Filename mismatch: {img_file} and {label_file}"

    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        image_path = os.path.join(self.images_dir, self.image_files[idx])
        label_path = os.path.join(self.labels_dir, self.label_files[idx])

        # Load image
        image = np.load(image_path)
        
        # Load targets
        targets = np.load(label_path)

        # Remove the first dimension if it exists
        if image.shape[0] == 1:
            image = np.squeeze(image, axis=0)
        
        # Ensure image is of shape (256, 64, 64)
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
    def __init__(self, train_dir, val_dir, batch_size=32, num_queries=300):
        super().__init__()
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.batch_size = batch_size
        self.num_queries = num_queries

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_dataset = DetectionHeadDataset(self.train_dir, num_queries=self.num_queries)
            self.val_dataset = DetectionHeadDataset(self.val_dir, num_queries=self.num_queries)
        
        if stage == 'validate' or stage is None:
            self.val_dataset = DetectionHeadDataset(self.val_dir, num_queries=self.num_queries)
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=4)
