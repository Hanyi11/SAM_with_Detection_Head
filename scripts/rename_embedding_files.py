#!/usr/bin/env python

import numpy as np
import cv2
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from copy import deepcopy
import skimage
import scipy
import basicpy


img_folder_base = Path('/ictstr01/groups/shared/users/lion.gleiter/organoid_sam/patch_images') 
box_folder_base = Path('/ictstr01/groups/shared/users/lion.gleiter/organoid_sam/patch_bbox_gt')
seg_folder_base = Path('/ictstr01/groups/shared/users/lion.gleiter/organoid_sam/patch_seg_gt')



if __name__=="__main__":
    model = 'MicroSAM_huge'
    embed_base = Path('/ictstr01/groups/shared/users/lion.gleiter/organoid_sam/patch_embeddings2') 
    for i, file in enumerate(embed_base.glob(f'{model}/*/*/*/*')):
        stem = file.stem
        if stem.startswith('images_patches_emb_'):
            continue
        patch_num = int(stem.removeprefix(prefix='patch_'))
        new_stem = f'images_patches_emb_{patch_num:04d}'
        new_file = file.with_stem(new_stem)
        file.rename(new_file)
        print(file)
        print(new_file)
        if i >= 0:
            break