import os
from typing import Literal, List
import numpy as np
import torch
from segment_anything import sam_model_registry, SamPredictor


def load_sam_predictor(
       model_size: Literal["large", "base"] =  "large" ,
       checkpoint_dir: str =  "/ictstr01/groups/shared/users/lion.gleiter/organoid_sam/checkpoints",
       device: str = "cuda"
    ) -> "SamPredictor":
    
    if model_size == "large":
        sam_checkpoint = os.path.join(checkpoint_dir, "sam_vit_l_0b3195.pth")
        model_type = "vit_l"
    elif model_size == "base":
        sam_checkpoint = os.path.join(checkpoint_dir, "sam_vit_b_01ec64.pth") 
        model_type = "vit_b"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    sam.eval()
    return SamPredictor(sam)

def load_medsam_predictor(
       checkpoint_dir: str =  "/ictstr01/groups/shared/users/lion.gleiter/organoid_sam/checkpoints",
       device: str = "cuda"
    ) -> "SamPredictor":
    
    medsam_checkpoint = os.path.join(checkpoint_dir, "medsam_vit_b.pth") 
    model_type = "vit_b"

    medsam = sam_model_registry[model_type](checkpoint=medsam_checkpoint)
    medsam.to(device=device)
    medsam.eval()
    return SamPredictor(medsam)

def load_cellsam_predictor(
        checkpoint_dir: str =  "/ictstr01/groups/shared/users/lion.gleiter/organoid_sam/checkpoints",
        device: str = "cuda"
    ) -> "SamPredictor":
    
    cellsam_checkpoint = os.path.join(checkpoint_dir, "cellsam_bbox.pth") 
    model_type = "vit_b"

    cellsam = sam_model_registry[model_type](checkpoint=cellsam_checkpoint)
    cellsam.to(device)
    cellsam.eval()
    return SamPredictor(cellsam)

def load_microsam_predictor(
        model_size: Literal["huge", "large", "base" ] = "large",
        device: str = "cuda",    
        checkpoint_dir: str =  "/ictstr01/groups/shared/users/lion.gleiter/organoid_sam/checkpoints"
    ) -> "SamPredictor":
    
    if model_size == "huge":
        checkpoint_name = "microsam_vit_h_lm.pth"
        model_type = "vit_h"
    elif model_size == "large":
        checkpoint_name = "microsam_vit_l_lm.pt"
        model_type = "vit_l"
    elif model_size == "base":
        checkpoint_name = "microsam_vit_b_lm.pth"
        model_type = "vit_b"
    
    microsam_checkpoint = os.path.join(checkpoint_dir, checkpoint_name) 
    
    microsam = sam_model_registry[model_type](checkpoint=microsam_checkpoint)
    microsam.to(device)
    microsam.eval()
    return SamPredictor(microsam)

def sam_predict_embedding(
        sam_predictor: "SamPredictor",
        image: np.ndarray
    ) -> np.ndarray:

    sam_predictor.set_image(image)
    embedding = sam_predictor.get_image_embedding().squeeze(0)
    embedding = embedding.permute(1, 2, 0) # 1xCxHxW -> HxWxC
    embedding = embedding.cpu().numpy() # torch.tensor -> np.ndarray
    return embedding

def sam_predict_and_safe_embedding(
        sam_predictor: "SamPredictor",
        image: np.ndarray,
        save_name: str = "embedding",
        save_dir: str = "/home/aih/maximilian.hoermann/projects/segment-anything/saved_embeddings"  
    ):

    sam_predictor.set_image(image)

    # Save for loading with SAM decoder
    save_path_sam = os.path.join(save_dir, save_name + ".pt")
    sam_predictor.save_image_embedding(save_path_sam)

    # Save as .npy for DETR
    embedding = sam_predictor.get_image_embedding().squeeze(0)
    embedding = embedding.permute(1, 2, 0) # 1xCxHxW -> HxWxC
    embedding = embedding.cpu().numpy() # torch.tensor -> np.ndarray
    save_path_np = os.path.join(save_dir, save_name + ".npy")
    np.save(save_path_np, embedding)

    # --- Test ---
    # test = np.load(save_path_np)
    # residuum = np.sum(embedding - test)
    # print(f"Array saved succesfully and reloaded. The sumed residuum is: {residuum}")

def sam_predict_masks_from_bbs(
        bounding_boxes: List[np.ndarray],
        sam_predictor: "SamPredictor",
        embedding_path: str = "/home/aih/maximilian.hoermann/projects/segment-anything/saved_embeddings/embedding.pt", # image in HWC uint8 format, with pixel values in [0, 255].
         # each bounding_box (np.ndarray): A length 4 array given a box prompt to the model, in XYXY format. 
    ) -> np.ndarray:
    
    masks_pro_img = []
    scores_pro_img = []
    logits_pro_img = []

    with torch.inference_mode():
        sam_predictor.load_image_embedding(embedding_path)
        for bounding_box in bounding_boxes:
            masks_pro_bb, scores_pro_bb, logits_pro_bb = sam_predictor.predict(
                point_coords=None,
                point_labels=None,
                box=bounding_box[None, :], # 
                multimask_output=True,
                )
            
            sorted_ind = np.argsort(scores_pro_bb)[::-1]
            masks_pro_bb = masks_pro_bb[sorted_ind]
            scores_pro_bb = scores_pro_bb[sorted_ind]
            logits_pro_bb = logits_pro_bb[sorted_ind]
            
            masks_pro_img.append(masks_pro_bb)
            scores_pro_img.append(scores_pro_bb)
            logits_pro_img.append(logits_pro_bb)
    
    return masks_pro_img, scores_pro_img, logits_pro_img

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

if __name__ == "__main__":
    import cv2
    import os
    import re
    from tqdm import tqdm

    # Define function for embedding each image
    def process_image(image_path: str, predictor: "SamPredictor"):
        # load image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # extract image name & image_dir
        image_dir, image_name = os.path.split(image_path)
        
        # extract image number
        number = re.search(r'\d+', image_name)
        if number:
            extracted_number = int(number.group())
        else:
            print("No number found in the string.")
                
        # construct saving parameters
        save_name = f"images_patches_emb_{extracted_number:04}"
        relative_path = image_dir.replace(base_dir, "").lstrip("/")
        save_dir = os.path.join(save_parent_dir, encoder_name, relative_path)
        print(f"Savedir: {save_dir}")
        os.makedirs(save_dir, exist_ok=True)

        # calculate embeddings
        sam_predict_and_safe_embedding(predictor, image, save_name, save_dir)

    # Define function for processing each directory
    def process_directory(image_dir:str, predictor: "SamPredictor"):
        # get paths of all files in image_dir
        image_paths = get_files_from_dir(image_dir, file_ending=".png")
        image_paths = sorted(image_paths)

        # calculate embeddings for each image and save image
        for i, image_path in enumerate(image_paths):
            process_image(image_path, predictor)
   
    # Paramaters 
    # "/ictstr01/groups/shared/users/lion.gleiter/organoid_sam/patch_images/test/OrganoID_mouse/im_18/patch_3.png"
    base_dir = "/ictstr01/groups/shared/users/lion.gleiter/organoid_sam/patch_images"
    save_parent_dir = "/ictstr01/groups/shared/users/lion.gleiter/organoid_sam/patch_embeddings"
    encoder_name = "CellSAM" 
    predictor = load_cellsam_predictor()
   
    traintestvals = ["test"]
    
    # sam predictor gets initialized with weights
    # if encoder_name == "SAM_large":
    #     predictor = load_sam_predictor("large")
    # elif encoder_name == "SAM_base":
    #     predictor = load_sam_predictor("base")
    # elif encoder_name == "MedSAM":
    #     predictor = load_medsam_predictor()
    # elif encoder_name == "CellSAM":
    #     predictor = load_cellsam_predictor()
    # elif encoder_name == "MicroSAM_huge":
    #     predictor = load_microsam_predictor("huge")  
    # elif encoder_name == "MicroSAM_large":
    #     predictor = load_microsam_predictor("large")  
    # elif encoder_name == "MicroSAM_base":
    #     predictor = load_microsam_predictor("base") 
    # else:
    #     raise ValueError(f"The enconder_name {encoder_name} does not exist!!")  

    
    # datasets = ["OrganoID_mouse", "Intestinal Organoid Dataset", "NeurIPS_CellSeg", "OrgaExtractor"]
    datasets = ["OrganoID_test", "OrganoID_test_ACC", "OrganoID_test_C", "OrganoID_test_Lung", "OrganoID_test_mouse"]
    parent_dirs = [os.path.join(base_dir, traintestval, dataset) for traintestval in traintestvals for dataset in datasets]
    

    
    for parent_dir in parent_dirs:
        image_dirs = get_subdirectories_from_dir(parent_dir)
        print(f"The image dirs are: {image_dirs}")
        for image_dir in tqdm(image_dirs):
            print(f"The image dir is : {image_dir}")
            process_directory(image_dir, predictor)
            


    

    # Get list of image directories
    
    ### ----- Primitive For-Loop
    # --- Set backbone ---
    # predictor = load_sam_predictor("large")
    # predictor = load_medsam_predictor()
    # predictor = load_cellsam_predictor()
    # predictor = load_microsam_predictor("huge")
    
    # Get image paths
    # parent_dir = "/ictstr01/groups/shared/users/lion.gleiter/organoid_sam/patch_images"
    # image_dirs = get_subdirectories_from_dir(parent_dir)
    # for image_dir in image_dirs:
    #     image_paths = get_files_from_dir(image_dir, file_ending=".png")
    #     image_paths = sorted(image_paths)
    #     for i, image_path in enumerate(image_paths):
    #         image = cv2.imread(image_path)
    #         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #         save_name = f"images_patches_emb_{i:04}"
    #         save_dir = "/ictstr01/groups/shared/users/lion.gleiter/organoid_sam/patch_embeddings"
    #         os.makedirs(save_dir, exist_ok=1)
    #         sam_predict_and_safe_embedding(predictor, image, save_name, save_dir)



    # ----- Simple Study -------

    # image = cv2.imread("/home/aih/maximilian.hoermann/datasets/TestSet/20210413_A549_Apoptosis_loop006_xy2_512_0/cycle_0001.png")
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # predictor = load_sam_predictor("large")
    # predictor = load_medsam_predictor()
    # predictor = load_cellsam_predictor()
    # predictor = load_microsam_predictor("huge")

    # sam_predict_and_safe_embedding(predictor, image)


    # masks, _, _ = sam_predict_masks_from_bbs(
    #     bounding_boxes = [np.array([210, 340, 240, 370])], 
    #     sam_predictor = predictor
    #                                                                             )
    
    # # # ---- Test ----
    # # mask = masks[0] # 3xHxW
    # # mask = mask.astype(np.uint8) 
    # # mask = mask * 255
    # # mask = np.transpose(mask, (1, 2, 0))
    # # overlayed_image = cv2.addWeighted(image, 0.7, mask, 0.3, 0)  # 0.7 and 0.3 are weights
    # # cv2.imwrite("/home/aih/maximilian.hoermann/projects/segment-anything/saved_embeddings/overlayed_image.png", overlayed_image)