from copy import deepcopy
from pathlib import Path
import cv2
import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset, RandomSampler, SubsetRandomSampler, WeightedRandomSampler
import torch
import os
import numpy as np
from typing import Literal, List
from tqdm import tqdm
import multiprocessing as mp
from PIL import Image
from torchvision.models.detection.ssd import SSD300_VGG16_Weights

from util import box_ops_numpy


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


def compute_weights_new_data_only(dataset, factor_new_data, factor_neurips=1, factor_OI=1, factor_empty_patches=0.01):
    print("start compute weights new data only")
    w = np.zeros(len(dataset), dtype=float)

    # Reduce weight of empty patches:
    is_empty = deepcopy(dataset.is_empty)

    print_ids = []
    print_id_datasets = []
    print_id_empty = []
    # Reweight each dataset
    for ds_name in [
        'NewData',
        'NeurIPS',
        'open_images',
        None, # All others
    ]:
        has_printed = False
        has_printed_empty = False
        if ds_name is not None:
            _w_not_empty = np.zeros_like(w)
            _w_empty = np.zeros_like(w)
            for i, img_path in tqdm(enumerate(dataset.image_files)):
                if ds_name in str(img_path):
                    if is_empty[i]:
                        _w_empty[i] = 1.0
                        if not has_printed_empty:
                            print_ids.append(i)
                            print_id_empty.append(True)
                            print_id_datasets.append(ds_name)
                            has_printed_empty = True
                    else:
                        _w_not_empty[i] = 1.0
                        if not has_printed:
                            print_ids.append(i)
                            print_id_empty.append(False)
                            print_id_datasets.append(ds_name)
                            has_printed = True
                    assert w[i] < 1e-16, img_path
        else:
            _w_not_empty = np.zeros_like(w)
            _w_empty = np.zeros_like(w)
            for i, img_path in tqdm(enumerate(dataset.image_files)):
                if ('NewData' in str(img_path)) or ('NeurIPS' in str(img_path)) or ('open_images' in str(img_path)):
                    continue
                else:
                    if is_empty[i]:
                        _w_empty[i] = 1.0
                        if not has_printed_empty:
                            print_ids.append(i)
                            print_id_empty.append(True)
                            print_id_datasets.append('other')
                            has_printed_empty = True
                    else:
                        _w_not_empty[i] = 1.0
                        if not has_printed:
                            print_ids.append(i)
                            print_id_empty.append(False)
                            print_id_datasets.append('other')
                            has_printed = True
                    assert w[i] < 1e-16, img_path

        
        if _w_not_empty.sum() >= 1:
            _w_not_empty = (_w_not_empty / _w_not_empty.sum()) * (1.0 - factor_empty_patches)
        if _w_empty.sum() >= 1:
            _w_empty = (_w_empty / _w_empty.sum()) * factor_empty_patches

        # _w_empty and _w_not_empty sum up together to probability 1. Now reweigh depending on dataset

        if ds_name is None:
            _w_empty *= (1.0 - factor_new_data)
            _w_not_empty *= (1.0 - factor_new_data)
        elif ds_name=='NewData':
            _w_empty *= factor_new_data
            _w_not_empty *= factor_new_data
        elif ds_name=='NeurIPS':
            _w_empty *= factor_neurips
            _w_not_empty *= factor_neurips
        elif ds_name=='open_images':
            _w_empty *= factor_OI
            _w_not_empty *= factor_OI
        else:
            raise RuntimeError(ds_name)
        
        w = np.maximum(w, _w_empty)
        w = np.maximum(w, _w_not_empty)

    print("done compute weights new data only")
    for i, ds_name, empty in zip(print_ids, print_id_datasets, print_id_empty):
        print('weight', w[i], ds_name, empty)

    return np.maximum(w, 0.0)


def compute_weights(dataset, factor_neurips=7, factor_OI=7, factor_empty_patches=0.01):
    print("start compute weights")
    w = np.zeros(len(dataset), dtype=float)

    # Reduce weight of empty patches:
    is_empty = deepcopy(dataset.is_empty)


    print_ids = []
    print_id_datasets = []
    print_id_empty = []
    # Reweight each dataset
    for ds_name in [
        'OrganoID',
        'OrgaSegment',
        'OrgaQuant',
        'OrgaExtractor',
        'Tellu',
        'MultiOrg',
        'NewData',
        'NeurIPS',
        'open_images',
        None, # All others
    ]:
        has_printed = False
        has_printed_empty = False

        if ds_name is not None:
            _w_not_empty = np.zeros_like(w)
            _w_empty = np.zeros_like(w)
            for i, img_path in tqdm(enumerate(dataset.image_files)):
                if ds_name in str(img_path):
                    if is_empty[i]:
                        _w_empty[i] = 1.0

                        if not has_printed_empty:
                            print_ids.append(i)
                            print_id_empty.append(True)
                            print_id_datasets.append(ds_name)
                            has_printed_empty = True

                    else:
                        _w_not_empty[i] = 1.0
                        if not has_printed:
                            print_ids.append(i)
                            print_id_empty.append(False)
                            print_id_datasets.append(ds_name)
                            has_printed = True
                    assert w[i] < 1e-16, img_path
        else:
            _w_not_empty = np.zeros_like(w)
            _w_empty = np.zeros_like(w)
            for i, img_path in tqdm(enumerate(dataset.image_files)):
                if ('OrganoID' in str(img_path)) or \
                    ('OrgaSegment' in str(img_path)) or \
                    ('OrgaQuant' in str(img_path)) or \
                    ('OrgaExtractor' in str(img_path)) or \
                    ('Tellu' in str(img_path)) or \
                    ('MultiOrg' in str(img_path)) or \
                    ('NewData' in str(img_path)) or \
                    ('NeurIPS' in str(img_path)) or \
                    ('open_images' in str(img_path)):
                    continue
                else:
                    if is_empty[i]:
                        _w_empty[i] = 1.0
                        if not has_printed_empty:
                            print_ids.append(i)
                            print_id_empty.append(True)
                            print_id_datasets.append('other')
                            has_printed_empty = True
                    else:
                        _w_not_empty[i] = 1.0
                        if not has_printed:
                            print_ids.append(i)
                            print_id_empty.append(False)
                            print_id_datasets.append('other')
                            has_printed = True
                    assert w[i] < 1e-16, img_path

        
        if _w_not_empty.sum() >= 1:
            _w_not_empty = (_w_not_empty / _w_not_empty.sum()) * (1.0 - factor_empty_patches)
        if _w_empty.sum() >= 1:
            _w_empty = (_w_empty / _w_empty.sum()) * factor_empty_patches

        # _w_empty and _w_not_empty sum up together to probability 1. Now reweigh depending on dataset

        if ds_name is None:
            _w_empty *= 1.0
            _w_not_empty *= 1.0
        elif ds_name=='NeurIPS':
            _w_empty *= factor_neurips
            _w_not_empty *= factor_neurips
        elif ds_name=='open_images':
            _w_empty *= factor_OI
            _w_not_empty *= factor_OI
        else:
            _w_empty *= 1.0
            _w_not_empty *= 1.0
        
        w = np.maximum(w, _w_empty)
        w = np.maximum(w, _w_not_empty)

    print("done compute weights")
    for i, ds_name, empty in zip(print_ids, print_id_datasets, print_id_empty):
        print('weight', w[i], ds_name, empty)
    return np.maximum(w, 0.0)


def collate_fn(batch):
    images = [b[0] for b in batch]
    targets = [b[1] for b in batch]
    return images, targets


class SSDDataset(Dataset):
    def __init__(self, 
                 data_split_dirs: List[str],
                 data_split: Literal["train", "test", "val"] = "train", 
                 encoder_name: Literal["SAM_base", "MedSAM", "CellSAM", 
                                       "SAM_large", "MicroSAM_huge", "SAM2_large"] = "SAM_base",
                 num_queries: int = 300, 
                 base_dir: str = "/ictstr01/groups/shared/users/lion.gleiter/organoid_sam/"):
        super().__init__()
        
        self.data_split_dirs = data_split_dirs
        self.base_dir = Path(base_dir)
        self.num_queries = num_queries
        self.data_split = data_split
        self.encoder_name = encoder_name
        self.transforms = SSD300_VGG16_Weights.COCO_V1.transforms()

        # Gets sorted paths of image embeddings
        open_image_dirs = [d for d in data_split_dirs if d.startswith('open_images_v4_5/original_data')]
        organoid_dirs = [d for d in data_split_dirs if not d.startswith('open_images_v4_5/original_data')]
        assert len(open_image_dirs) <= 1, open_image_dirs  # Should just be ['open_images_v4_5/original_data']


        # Files for image crops
        embed_dir = self.base_dir / 'patched_data' / 'patch_images' / self.data_split

        embed_files = []
        for dataset in organoid_dirs:
            embed_files += list(embed_dir.glob(f'{dataset}/*/*.png'))

        # Sort embed files
        def sorting_key(file: Path):
            im_dir = file.parent.as_posix()
            patch_number = int(file.stem.replace('patch_', ''))
            return (im_dir, patch_number)
        
        self.image_files: list[Path] = sorted(embed_files, key=sorting_key)


        # Files for ground truth bboxes
        bbox_gt_dir = self.base_dir / 'patched_data' / 'patch_bbox_gt' / self.data_split

        bbox_gt_files = []
        for dataset in organoid_dirs:
            # Files named `patch_28.npy` but not `offsets.npy` or `patch_28_overlap.npy`
            bbox_gt_files += list(bbox_gt_dir.glob(f'{dataset}/*/patch_*[0-9].npy'))

        # Sort embed files
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


        # Appends Open Images file paths to self.image_files but not to self.label_files
        for oi_path in open_image_dirs:
            path = Path('/ictstr01/groups/shared/users/lion.gleiter') / oi_path / 'validation'
            self.image_files += sorted(list(path.glob('*.jpg')))

        self.oi_annotation = pd.read_csv('/ictstr01/groups/shared/users/lion.gleiter/open_images_v4_5/original_data/validation-annotations-bbox.csv')

        # Creates a boolean vector indicating which images have 0 bbox annotations.
        self.is_empty = np.zeros(len(self.image_files), dtype=bool)
        for i, label_path in tqdm(enumerate(self.label_files)):
            targets = np.load(label_path)
            if targets.shape[0] == 0:
                self.is_empty[i] = True

    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Load image
        image_path = self.image_files[idx]
        image = Image.open(image_path).convert('RGB')
        image = self.transforms(image)
        H, W = image.shape[-2:]
        # print('image in __getitem__', image)
        # print('image.shape', image.shape)
        # image = image.squeeze(0)  # .permute(1, 2, 0).cpu().numpy()

        # image = np.load(image_path)

        # Load targets
        if idx >= len(self.label_files):
            # It's an open images path
            assert 'open_images' in str(image_path), image_path
            # load bounding box
            bboxes = self.oi_annotation[self.oi_annotation['ImageID']==image_path.stem]
            # img_original = cv2.imread(str(Path('/ictstr01/groups/shared/users/lion.gleiter/open_images_v4_5/original_data/') / image_path.parent.name / f'{image_path.stem}.jpg'))
            # H, W = img_original.shape[:2]
            targets = [[(b.YMin + b.YMax) / 2, (b.XMin + b.XMax) / 2, b.YMax - b.YMin, b.XMax - b.XMin] for b in bboxes.itertuples(index=False)]
            targets = np.array(targets).reshape((-1, 4))
            targets = targets * np.array([[H, W, H, W]]) / max(H, W)
        else:
            # It's not an open images path
            assert 'open_images' not in str(image_path), image_path
            label_path = self.label_files[idx]
            targets = np.load(label_path)

        
        # Convert cycxhw to xyxy
        targets = box_ops_numpy.cxcywh_to_xyxy(targets) * max(H, W)
        targets = np.stack((targets[:, 1], targets[:, 0], targets[:, 3], targets[:, 2]), axis=1)

        # Pad targets to ensure they are of shape (num_queries, 4)
        # num_boxes = targets.shape[0]
        # if num_boxes < self.num_queries:
        #     pad_size = self.num_queries - num_boxes
        #     padded_targets = np.pad(targets, ((0, pad_size), (0, 0)), mode='constant', constant_values=0)
        # else:
        #     padded_targets = targets[:self.num_queries]

        # Convert to torch tensors
        # image = torch.tensor(image, dtype=torch.float32)
        targets = torch.tensor(targets, dtype=torch.float32)
        labels = torch.ones((targets.shape[0],), dtype=torch.int64)
        

        # # Prints from which dataset we sampled and its sampling weight for debugging 
        # w = compute_weights_new_data_only(self, factor_new_data=0.2)
        # for ds_name in [
        #     'OrganoID',
        #     'OrgaSegment',
        #     'OrgaQuant',
        #     'OrgaExtractor',
        #     'Tellu',
        #     'MultiOrg',
        #     'NewData',
        #     'NeurIPS',
        #     'open_images'
        # ]:
        #     if ds_name in str(image_path):
        #         print('idx:', idx, "ds", ds_name, 'weight', w[idx], num_boxes)
        
        return image, {'boxes': targets, 'labels': labels}

class DataModule(pl.LightningDataModule):
    def __init__(self, 
                 train_dir_names: List[str],
                 val_dir_names: List[str],
                 encoder_name: Literal["SAM_base", "MedSAM", "CellSAM", 
                                       "SAM_large", "MicroSAM_huge", "SAM2_large"] = "SAM_base",
                 batch_size: int = 32, 
                 num_queries: int = 300,
                 batches_per_epoch: int = 411,
                 use_sampler: bool = False,
                 balance_datasets: bool = False,
                 balance_validation: bool = False,
                 balance_newdata: float = 0.05,
                 ):
        super().__init__()
        self.n_workers = int(os.environ.get('SLURM_CPUS_PER_TASK', 0)) - 1
        # self.n_workers = 0

        self.train_dir_names = train_dir_names
        self.val_dir_names = val_dir_names
        self.encoder_name = encoder_name
        self.batch_size = batch_size
        self.num_queries = num_queries
        self.batches_per_epoch = batches_per_epoch
        self.use_sampler = use_sampler
        self.balance_datasets = balance_datasets
        self.balance_newdata = balance_newdata
        self.balance_validation = balance_validation

        
    def setup(self, stage: Literal["fit", "validate", None] = None):
        if stage == 'fit' or stage is None:
            self.train_dataset = SSDDataset(data_split_dirs=self.train_dir_names, 
                                            data_split="train", 
                                            encoder_name=self.encoder_name, 
                                            num_queries=self.num_queries)
            
            self.val_dataset = SSDDataset(data_split_dirs=self.val_dir_names,
                                          data_split="val", 
                                          encoder_name=self.encoder_name, 
                                          num_queries=self.num_queries)
        
        if stage == 'validate' or stage is None:
            self.val_dataset = SSDDataset(data_split_dirs=self.val_dir_names, 
                                          data_split="val", 
                                          encoder_name=self.encoder_name, 
                                          num_queries=self.num_queries)




    def train_dataloader(self):
        assert not (self.balance_datasets and (self.balance_newdata > -0.5))
        if self.use_sampler:
            if self.balance_datasets:
                w = compute_weights(self.train_dataset)
                sampler = WeightedRandomSampler(
                    weights=w,
                    replacement=True,
                    num_samples=self.batches_per_epoch * self.batch_size
                )
            elif self.balance_newdata > -0.5:
                print('using self.balance_newdata', self.balance_newdata)
                w = compute_weights_new_data_only(self.train_dataset, self.balance_newdata)
                sampler = WeightedRandomSampler(
                    weights=w,
                    replacement=True,
                    num_samples=self.batches_per_epoch * self.batch_size
                )
            else:
                sampler = RandomSampler(
                    data_source=self.train_dataset,
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
        if self.use_sampler and (self.balance_datasets or self.balance_validation):
            w = compute_weights(self.val_dataset)
            sampler = WeightedRandomSampler(
                weights=w,
                replacement=True,
                num_samples=self.batch_size * 20
            )
            return DataLoader(self.val_dataset, 
                              batch_size=self.batch_size, 
                              sampler=sampler, 
                              num_workers=self.n_workers,
                              collate_fn=collate_fn)
        else: 
            return DataLoader(self.val_dataset, 
                              batch_size=self.batch_size, 
                              num_workers=self.n_workers,
                              collate_fn=collate_fn)

