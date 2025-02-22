
import os
from copy import deepcopy
import cv2
from monai import transforms as tfs
import numpy as np
from omegaconf import ListConfig, OmegaConf
import pandas as pd
from pathlib import Path
import PIL
import PIL.Image
import pytorch_lightning as pl
import sys
import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler, SubsetRandomSampler, WeightedRandomSampler
from torchvision.models.detection.faster_rcnn import FasterRCNN_ResNet50_FPN_V2_Weights, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.ssd import SSD300_VGG16_Weights
from tqdm import tqdm
from typing import Literal, List

import warnings
warnings.filterwarnings("ignore", "You are using `torch.load` with `weights_only=False`*.")

from util import box_ops_numpy

path_detr = '/home/icb/lion.gleiter/projects/organoid_sam/detr'
if path_detr not in sys.path:
    sys.path.append(path_detr)
from detr.datasets import coco


def compute_weights(metadata: pd.DataFrame, groups, max_oversampling = 10.0, max_ratio_0_objects=0.05, weight_open_images=0.5):
    """
    Compute per-sample weights for a dataset, balancing multiple subgroups of data.

    The subgroups are determined as unique value combinations of the variables in 'groups', e.g.
    if groups = ['dataset', 'patch_size'], then data from NeurIPS_train with patchsize 2048 would 
    be sampled equally often as data from OrgaSegment_train with patchsize 512.

    Args:
        metadata (pd.DataFrame): 
            Contains the following columns, and each row is one available datapoint: ('dataset', 'n_objects', 'n_objects_grouped', 'patch_size')
        groups (list[str] | None): 
            Keys for determining unique groups for weighting. 'None' means equal weight for all datapoints.
        max_oversampling (float):
            Specifies how much any sample weight may be maximally increased compared to the default weight.

    Returns:
        np.ndarray: 
            A NumPy array of weights for each sample in the dataset. The weights are normalized within each subset.
    """
    print("start compute weights")
    total_length = len(metadata)
    max_weight = max_oversampling / total_length

    # Initialize with equal weight for all points.
    w = np.ones(len(metadata), dtype=float) / total_length

    # Assign weight Open Images
    total_available_weight = 1.0
    if np.any(metadata[['dataset']] == 'open_images'):
        print(f"\n\nFound Open Images samples in the data, assigning weight {weight_open_images}\n\n")
        is_oi = (metadata[['dataset']]=='open_images').values.flatten()
        is_not_oi = (metadata[['dataset']]!='open_images').values.flatten()
        oi_ids = metadata.loc[is_oi, ['idx']].values.tolist()
        other_ids = metadata.loc[is_not_oi, ['idx']].values.tolist()
        w[oi_ids] = weight_open_images / len(oi_ids)
        w[other_ids] = (1 - weight_open_images) / len(other_ids)
        total_available_weight = 1 - weight_open_images

    if groups is None:
        return w

    # Each group outside Open Images gets the same total weight and distributes it equally between its members.
    is_not_oi = (metadata[['dataset']]!='open_images').values.flatten()
    metadata_wo_oi = metadata.loc[is_not_oi, :]
    grouped_metadata = metadata_wo_oi.groupby(groups, as_index=False)
    n_groups = len(grouped_metadata)
    total_weight_per_group = total_available_weight / n_groups

    for name, df in grouped_metadata:
        weight = total_weight_per_group / len(df)
        ids = df[['idx']].values.tolist()
        w[ids] = weight
        print(name, 'weight', weight)

    # Enforce max weight:
    w = np.minimum(w, max_weight)

    total_weight = np.sum(w)
    zero_objects = (metadata[['n_objects']].values == 0).flatten()
    weight_zero_objects = np.sum(w[zero_objects])
    if weight_zero_objects / total_weight > max_ratio_0_objects:
        w[zero_objects] = w[zero_objects] * (max_ratio_0_objects / (1-max_ratio_0_objects)) * (total_weight - weight_zero_objects) / weight_zero_objects

    # Check that calculations were correct
    total_weight = np.sum(w)
    weight_zero_objects = np.sum(w[zero_objects])
    assert weight_zero_objects / total_weight <= max_ratio_0_objects + 1e-3, weight_zero_objects / total_weight

    return w



def collate_fn(batch):
    """Take a batch of data samples and seperate images and targets into two seperate lists."""
    if isinstance(batch[0][0], list) or isinstance(batch[0][0], tuple):
        features0 = torch.cat([b[0][0] for b in batch], dim=0)
        features1 = torch.cat([b[0][1] for b in batch], dim=0)
        features2 = torch.cat([b[0][2] for b in batch], dim=0)
        features = [features0, features1, features2]
    else:
        features = torch.cat([b[0] for b in batch], dim=0)
    targets = [b[1] for b in batch]
    img_ids = [b[2] for b in batch]
    offsets = [b[3] for b in batch]
    patch_sizes = [b[4] for b in batch]
    patch_numbers = [b[5] for b in batch]
    return features, targets, img_ids, offsets, patch_sizes, patch_numbers


# TODO: add backbone-specific padding to avoid padding artifacts during training. (always pad only the bottom and right, to avoid issues with bboxes.)


def prepare_coco_targets(image: PIL.Image, target):
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]

        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        target["image_id"] = image_id
        if keypoints is not None:
            target["keypoints"] = keypoints

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])

        return image, target



class EmbeddingDataset(Dataset):
    def __init__(self, 
                 data_split: Literal["train", "test", "val"], 
                 data_split_dirs: List[str],
                 backbone_name: Literal["SAM_large", "SAM2_large", "FM_concat"], 
                 decoder: Literal["FRCNN", "FRCNNv2", "SSD", "DETR", "DETR_own_implementation"],
                 base_dir: str = "/ictstr01/groups/shared/users/lion.gleiter/organoid_sam/patched_data_multiscale_miccai",
                 min_overlap: float = 0.99, # 0.9,
                 min_box_side: float = 0.0,
                 num_queries = None,
                 **kwargs):
        """
        Returns:
            features: 
                For SAM_large: torch.Tensor of shape [1, 256, 64, 64]. 
                For SAM2_large: Tuple of 3 torch.Tensors of shape [1, 256, 64, 64], [1, 64, 128, 128], [1, 32, 256, 256]
                For FM_concat: Tuple of 3 torch.Tensors of shape [1, 512, 64, 64], [1, 64, 128, 128], [1, 32, 256, 256], where 
                    the low-res features of SAM and SAM2 are concatenated
        """
        super().__init__()
        self.base_dir = Path(base_dir)
        self.metadata = []  # idx, dataset, n_objects, n_objects_grouped, patch_size
        
        # Dataset parameters
        self.data_split = data_split
        self.data_dirs = data_split_dirs
        self.use_coco_format = True if backbone_name=='DETR' else False
        
        # Pretrained transformer model parameters
        self.backbone = backbone_name
        self.decoder = decoder

        self.num_queries = num_queries
        if self.decoder == "DETR_own_implementation":
            assert self.num_queries is not None, self.num_queries
        
        # Set parameters for filtering out objects that are too small or that do not have enough overlap with the image.
        self.min_overlap = min_overlap
        self.min_box_side = min_box_side

        # Gets sorted paths of image embeddings
        open_image_dirs = [d for d in data_split_dirs if d.startswith('open_images_v4_5/original_data')]
        organoid_dirs = [d for d in data_split_dirs if not d.startswith('open_images_v4_5/original_data')]
        assert len(open_image_dirs) <= 1, open_image_dirs  # Should just be ['open_images_v4_5/original_data']

        # Get paths to precalculated embeddings for image crops
        embed_dir = self.base_dir / 'patch_images' / self.data_split
        embed_files = []
        for dataset in organoid_dirs:
            embed_files += list(embed_dir.glob(f'{dataset}/*/*.png'))

        # Sort paths to embeddings (for filtering later on)
        def sorting_key(file: Path):
            im_dir = file.parent.as_posix()
            patch_number = int(file.stem.replace('patch_', ''))
            return (im_dir, patch_number)
        self.image_files: list[Path] = sorted(embed_files, key=sorting_key)

        # Get paths to ground truth bboxes
        bbox_gt_dir = self.base_dir / 'patch_bbox_gt' / self.data_split
        bbox_gt_files = []
        for dataset in organoid_dirs:
            # Files named `patch_28.npy` but not `offsets.npy` or `patch_28_overlap.npy`
            bbox_gt_files += list(bbox_gt_dir.glob(f'{dataset}/*/patch_*[0-9].npy'))

        # Sort paths to bbox
        def sorting_key_bbox(file: Path):
            im_dir = file.parent.as_posix()
            patch_number = int(file.stem.replace('patch_', ''))
            return (im_dir, patch_number)
        self.label_files: list[Path] = sorted(bbox_gt_files, key=sorting_key_bbox)


        # Checks that the image files and label files match.
        assert len(self.image_files) == len(self.label_files), f"Mismatch between number of embeddings ({len(self.image_files)}) and labels ({len(self.label_files)})"
        for img_file, label_file in zip(self.image_files, self.label_files):
            # Get path infos
            img_path_cleaned = img_file.parent.as_posix().replace(f'patch_images', '')
            label_path_cleaned = label_file.parent.as_posix().replace(f'patch_bbox_gt', '')

            img_path_cleaned = os.path.join(img_path_cleaned, img_file.stem)
            label_path_cleaned = os.path.join(label_path_cleaned, label_file.stem)

            # Compare
            assert img_path_cleaned == label_path_cleaned, f"Filename mismatch: {img_file} and {label_file} are not the right img_patch and gt_label pair!"



        # Add metadata for non open-images data:
        current_count_file = None
        current_count_data = None
        for i, img_file in tqdm(enumerate(self.image_files)):
            dataset = img_file.parent.parent.name
            patch_number = int(img_file.stem.replace('patch_', ''))
            if self.data_split != 'test':
                patch_size = int(img_file.parent.name.split('_')[-1])
            else:
                patch_size = None
            if (current_count_data is None) or (current_count_file is None):
                current_count_file = bbox_gt_dir / dataset / img_file.parent.name / 'box_counts.npy'
                current_count_data = np.load(current_count_file, allow_pickle=False)
            elif current_count_file != bbox_gt_dir / dataset / img_file.parent.name / 'box_counts.npy':
                current_count_file = bbox_gt_dir / dataset / img_file.parent.name / 'box_counts.npy'
                current_count_data = np.load(current_count_file, allow_pickle=False)
            object_count = current_count_data[patch_number]
            object_count_category = int(np.ceil(object_count / 10.0 - 0.05)) * 10  # 0 for 0 objects, 10 for 1-10 objects, 20 for 11-20 objects, ...

            self.metadata.append([i, dataset, object_count, object_count_category, patch_size])


        # Appends Open Images file paths to self.image_files but not to self.label_files
        for oi_path in open_image_dirs:
            path = Path('/ictstr01/groups/shared/users/lion.gleiter') / oi_path / 'validation'
            self.image_files += sorted(list(path.glob('*.jpg')))

        self.oi_annotation = pd.read_csv('/ictstr01/groups/shared/users/lion.gleiter/open_images_v4_5/original_data/validation-annotations-bbox.csv')

        # Update metadata.
        # self.is_empty = np.zeros(len(self.image_files), dtype=bool)
        for i in range(len(self.metadata), len(self.image_files)):
            self.metadata.append([i, 'open_images', 1000, 1000, 1000]) # dummy values to create a separate group.

        self.metadata = pd.DataFrame(data=self.metadata, columns=['idx', 'dataset', 'n_objects', 'n_objects_grouped', 'patch_size'])

    def __len__(self):
        return len(self.image_files)
    
    def _load_embedding(self, img_file, is_open_images):
        if is_open_images:
            embed_file_sam1 = Path('/ictstr01/groups/shared/users/lion.gleiter/organoid_sam/open_images_embeddings') / 'SAM_large' / 'validation' / f'{img_file.stem}.pt'
            embed_file_sam2 = Path('/ictstr01/groups/shared/users/lion.gleiter/organoid_sam/open_images_embeddings') / 'SAM2_large' / 'validation' / f'{img_file.stem}.pt'
        else:  # Not Open Images
            img_name = img_file.parent.name
            ds_name = img_file.parent.parent.name
            ds_split = img_file.parent.parent.parent.name
            embed_file_sam1 = Path('/ictstr01/groups/shared/users/lion.gleiter/organoid_sam/patched_data_multiscale_miccai/patch_embed') / 'SAM_large' / ds_split / ds_name / img_name / f'{img_file.stem}.pt'
            embed_file_sam2 = Path('/ictstr01/groups/shared/users/lion.gleiter/organoid_sam/patched_data_multiscale_miccai/patch_embed') / 'SAM2_large' / ds_split / ds_name / img_name / f'{img_file.stem}.pt'
    
        if self.backbone == 'SAM_large':
            embed = torch.load(embed_file_sam1, map_location=torch.device('cpu'))
            features = embed['features']
            orig_size = embed['original_size']

        elif self.backbone == 'SAM2_large':
            embed = torch.load(embed_file_sam2, map_location=torch.device('cpu'))
            features = (
                embed['_features']['image_embed'],
                embed['_features']['high_res_feats'][1],
                embed['_features']['high_res_feats'][0],
            )
            orig_size = embed['_orig_hw'][0]

        elif self.backbone == 'FM_concat':
            embed1 = torch.load(embed_file_sam1, map_location=torch.device('cpu'))
            features1 = embed1['features']
            orig_size = embed1['original_size']

            embed2 = torch.load(embed_file_sam2, map_location=torch.device('cpu'))
            features = (
                torch.cat((features1, embed2['_features']['image_embed']), dim=1),
                embed2['_features']['high_res_feats'][1],
                embed2['_features']['high_res_feats'][0],
            )
            orig_size2 = embed2['_orig_hw'][0]
            assert orig_size[0] == orig_size2[0], (orig_size, orig_size2)
            assert orig_size[1] == orig_size2[1], (orig_size, orig_size2)
        else:
            raise ValueError(self.backbone)

        return features, orig_size

    def __getitem__(self, idx):
        # Load image
        image_path = self.image_files[idx]
            
        # Load features and targets (BBs)
        if idx >= len(self.label_files):  # It's an open images path
            assert 'open_images' in str(image_path), image_path
            features, (H, W) = self._load_embedding(image_path, is_open_images=True)

            bboxes = self.oi_annotation[self.oi_annotation['ImageID']==image_path.stem]
            targets = [[(b.YMin + b.YMax) / 2, (b.XMin + b.XMax) / 2, b.YMax - b.YMin, b.XMax - b.XMin] for b in bboxes.itertuples(index=False)]
            targets = np.array(targets).reshape((-1, 4))

            # H, W = img_original.shape[:2]
            targets = targets * np.array([[H, W, H, W]]) / max(H, W)
            offsets = np.array([0, 0, 0])
            patch_shape = (H, W)
            img_id = image_path.stem
            patch_number = None

        else:  # It's not an open images path
            assert 'open_images' not in str(image_path), image_path
            features, (H, W) = self._load_embedding(image_path, is_open_images=False)

            label_path = self.label_files[idx]
            targets = np.load(label_path)

            # Filter boxes with low overlap with the current patch
            overlap_path = label_path.with_stem(f'{label_path.stem}_overlap')
            if overlap_path.exists():
                overlap = np.load(overlap_path)
                targets = targets[overlap > self.min_overlap]
                
            patch_number = int(label_path.stem.replace('patch_', ''))
            offsets_path = label_path.parent / 'offsets.npy'
            offsets = np.load(offsets_path, allow_pickle=False)[patch_number]
            patch_shape = (H, W)
            img_id = image_path.parent.name

        # Filter boxes that are too small
        targets = targets[
            (targets[:, 2] > self.min_box_side) & 
            (targets[:, 3] > self.min_box_side)
        ]
        
        # Convert BB format from cycxhw to xyxy
        targets = box_ops_numpy.cxcywh_to_xyxy(targets) * max(H, W)
        targets = np.stack((targets[:, 1], targets[:, 0], targets[:, 3], targets[:, 2]), axis=1)

        # Rescale image and boxes to max side length 1024
        max_side_length = 1024
        factor = float(max_side_length) / max(H, W) 
        out_H, out_W = min(max_side_length, H*factor), min(max_side_length, W*factor)
        targets *= np.array([[float(out_W) / W, 
                              float(out_H) / H, 
                              float(out_W) / W, 
                              float(out_H) / H]])  # Approx * factor
        
        
        if self.decoder == "DETR_own_implementation":
            # Pad targets to ensure they are of shape (num_queries, 4)
            num_boxes = targets.shape[0]
            if num_boxes < self.num_queries:
                pad_size = self.num_queries - num_boxes
                targets = np.pad(targets, ((0, pad_size), (0, 0)), mode='constant', constant_values=-1)
            else:
                targets = targets[:self.num_queries]


        # DETR
        if self.decoder == 'DETR':
            # TODO: adjust similar to image_datamodule.py
            targets = np.concatenate([targets[:, :2], targets[:, 2:] - targets[:, :2]], axis=1)
            annotations = {
                'image_id': idx,
                'annotations': [
                    {
                        'bbox': box,
                        'is_crowd': False,
                        'area': box[2] * box[3],
                        'category_id': 0,  # Label
                    } for box in targets
                ]
            }
            targets = self.prepare(image, annotations)
            image, targets = self.transforms(image, targets)
            return image, targets, img_id, offsets, patch_shape, patch_number


        targets = torch.tensor(targets, dtype=torch.float32)
        labels = torch.zeros((targets.shape[0],), dtype=torch.int64)
        image_id = torch.tensor([idx])
        area = (targets[:, 3] - targets[:, 1]) * (targets[:, 2] - targets[:, 0])
        is_crowd = torch.zeros((targets.shape[0],), dtype=torch.bool)
        orig_size = torch.as_tensor([int(H), int(W)])
        size = torch.as_tensor([int(H), int(W)])

        targets_out = {'boxes': targets, 
                       'labels': labels, 
                       'image_id': image_id, 
                       'area': area, 
                       'is_crowd': is_crowd, 
                       'orig_size': orig_size, 
                       'size': size}        
        
        return features, targets_out, img_id, offsets, patch_shape, patch_number


class EmbeddingDataModule(pl.LightningDataModule):
    def __init__(self, 
                 train_dirs: List[str],
                 val_dirs: List[str],
                 test_dirs: List[str],
                 batch_size: int, 
                 batches_per_epoch: int,
                 use_sampler: bool,
                 sampling_groups,
                 max_oversampling: float,
                 n_validation_samples: int | None,
                 **kwargs):
        super().__init__()
        self.n_workers = int(os.environ.get('SLURM_CPUS_PER_TASK', 0)) - 1
        self.kwargs = kwargs
        
        # Data directories 
        self.train_dir_names = train_dirs
        self.val_dir_names = val_dirs
        self.test_dir_names = test_dirs

        # Set training parameters
        self.batch_size = batch_size
        self.batches_per_epoch = batches_per_epoch

        # Set data loading parameters
        self.use_sampler = use_sampler
        if isinstance(sampling_groups, ListConfig):
            self.sampling_groups = OmegaConf.to_object(sampling_groups)
        else:
            self.sampling_groups = sampling_groups
        self.max_oversampling = max_oversampling
        self.n_validation_samples = n_validation_samples
        
    def setup(self, stage: Literal["fit", "validate", None] = None):
        # Set datasets depending on the stage
        if stage == 'fit' or stage is None:
            self.train_dataset = EmbeddingDataset(data_split_dirs=self.train_dir_names, 
                                                  data_split="train", 
                                                  **self.kwargs)
        
        if (stage == 'fit') or (stage == 'validate') or (stage is None):
            self.val_datasets = [EmbeddingDataset(
                data_split_dirs=[val_dir_name],
                data_split="val", 
                **self.kwargs
            ) for val_dir_name in self.val_dir_names]

    def train_dataloader(self):
        """
        This method sets up a DataLoader for the training dataset, either with a sampler or 
        with shuffling, depending on the configuration. It supports dataset balancing 
        by computing weights for the samples and optionally balances new data with a 
        specified ratio.
        """

        # assert not (self.balance_datasets and (self.balance_newdata > -0.5))
        if self.use_sampler:
            w = compute_weights(self.train_dataset.metadata, 
                                self.sampling_groups,
                                max_oversampling=self.max_oversampling)

            sampler = WeightedRandomSampler(
                weights=w,
                replacement=True,
                num_samples=self.batches_per_epoch * self.batch_size
            )

            return DataLoader(self.train_dataset, 
                              batch_size=self.batch_size,
                              sampler=sampler, 
                              num_workers=self.n_workers,
                              collate_fn=collate_fn)
        else:
            return DataLoader(self.train_dataset, 
                              batch_size=self.batch_size, 
                              shuffle=True, 
                              num_workers=self.n_workers,
                              collate_fn=collate_fn)
    
    def val_dataloader(self):
        """
        Creates and returns a DataLoader for the validation dataset.
        Depending on the configuration, it may use a sampler for balanced sampling or 
        load the data in a standard sequential manner.
        """
        if (self.n_validation_samples is not None) and (self.n_validation_samples > 0):
            dataloaders = []
            for val_dataset in self.val_datasets:
                sampler = RandomSampler(
                    val_dataset,
                    replacement=(len(val_dataset) < self.n_validation_samples),
                    num_samples=self.n_validation_samples
                )
                dataloaders.append(DataLoader(val_dataset, 
                                              batch_size=self.batch_size, 
                                              sampler=sampler, 
                                              num_workers=self.n_workers,
                                              collate_fn=collate_fn))
            return dataloaders
        else: 
            return [DataLoader(val_dataset, 
                               batch_size=self.batch_size, 
                               num_workers=self.n_workers,
                               collate_fn=collate_fn) for val_dataset in self.val_datasets]

    def get_test_dataloaders(self):
        """Creates and returns DataLoaders for the test datasets.
        """
        print('self.test_dir_names', self.test_dir_names)
        test_datasets = [EmbeddingDataset(
            data_split_dirs=[test_dir_name],
            data_split="test",
            **self.kwargs
        ) for test_dir_name in self.test_dir_names]

        return [DataLoader(test_dataset, 
                           batch_size=1, 
                           num_workers=self.n_workers,
                           collate_fn=collate_fn) for test_dataset in test_datasets]

