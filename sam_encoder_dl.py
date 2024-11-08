import os
import re
import cv2
from typing import Literal, List
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader

from segment_anything import sam_model_registry, SamPredictor


# ---- Main Functions ----
def load_sam_predictor(
       model_size: Literal["large", "base"] =  "large" ,
       checkpoint_dir: str =  "/ictstr01/groups/shared/users/lion.gleiter/organoid_sam/checkpoints",
       device: str = "cuda"
    ) -> "SamPredictor":
    """
    Loads a Segment Anything Model (SAM) with pretrained weights, either 'large' or 'base', 
    and returns a SamPredictor object initialized on the specified device.
    
    Args:
        model_size (Literal["large", "base"], optional): Model size to load ('large' or 'base'). Default is 'large'.
        checkpoint_dir (str, optional): Directory path containing model checkpoints. Default is the specified path.
        device (str, optional): Device to load the model on ('cuda' or 'cpu'). Default is 'cuda'.
    
    Returns:
        SamPredictor: A configured SamPredictor instance with the loaded model.
    """
    # Define path to MedSAM checkpoint file & specify model type 
    if model_size == "large":
        sam_checkpoint = os.path.join(checkpoint_dir, "sam_vit_l_0b3195.pth")
        model_type = "vit_l"
    elif model_size == "base":
        sam_checkpoint = os.path.join(checkpoint_dir, "sam_vit_b_01ec64.pth") 
        model_type = "vit_b"

    # Load SAM model with specified checkpoint and initialize it on the given device
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    sam.eval()

    return SamPredictor(sam)

def load_medsam_predictor(
       checkpoint_dir: str = "/ictstr01/groups/shared/users/lion.gleiter/organoid_sam/checkpoints",
       device: str = "cuda"
    ) -> "SamPredictor":
    """
    Loads a MedSAM model with pretrained weights and returns a SamPredictor object initialized on the specified device.
    
    Args:
        checkpoint_dir (str, optional): Directory path containing MedSAM model checkpoints. Default is the specified path.
        device (str, optional): Device to load the model on ('cuda' or 'cpu'). Default is 'cuda'.
    
    Returns:
        SamPredictor: A configured SamPredictor instance with the loaded MedSAM model.
    """
    
    # Define path to MedSAM checkpoint file & specify model type 
    medsam_checkpoint = os.path.join(checkpoint_dir, "medsam_vit_b.pth")
    model_type = "vit_b"  
    
    # Load MedSAM model with specified checkpoint, initialize it on the given device and set to evaluation mode
    medsam = sam_model_registry[model_type](checkpoint=medsam_checkpoint)
    medsam.to(device=device)
    medsam.eval()  
    
    return SamPredictor(medsam)

def load_cellsam_predictor(
        checkpoint_dir: str = "/ictstr01/groups/shared/users/lion.gleiter/organoid_sam/checkpoints",
        device: str = "cuda"
    ) -> "SamPredictor":
    """
    Loads a CellSAM model with pretrained weights and returns a SamPredictor object initialized on the specified device.
    
    Args:
        checkpoint_dir (str, optional): Directory path containing CellSAM model checkpoints. Default is the specified path.
        device (str, optional): Device to load the model on ('cuda' or 'cpu'). Default is 'cuda'.
    
    Returns:
        SamPredictor: A configured SamPredictor instance with the loaded CellSAM model.
    """
    
    # Define path to CellSAM checkpoint file & specify model type 
    cellsam_checkpoint = os.path.join(checkpoint_dir, "cellsam_bbox.pth")
    model_type = "vit_b"  
    
    # Load CellSAM model with specified checkpoint, initialize it on the given device and set to evaluation mode
    cellsam = sam_model_registry[model_type](checkpoint=cellsam_checkpoint)
    cellsam.to(device)
    cellsam.eval() 
    
    return SamPredictor(cellsam)

def load_microsam_predictor(
        model_size: Literal["huge", "large", "base"] = "large",
        device: Literal["cuda", "cpu"] = "cuda",
        checkpoint_dir: str = "/ictstr01/groups/shared/users/lion.gleiter/organoid_sam/checkpoints"
    ) -> "SamPredictor":
    """
    Loads a MicroSAM model of specified size with pretrained weights and returns a SamPredictor object initialized on the given device.
    
    Args:
        model_size (Literal["huge", "large", "base"], optional): The model size to load. Options are "huge", "large", or "base". Default is "large".
        device (Literal["cuda", "cpu"], optional): Device to load the model on ("cuda" or "cpu"). Default is "cuda".
        checkpoint_dir (str, optional): Directory path containing MicroSAM model checkpoints. Default is the specified path.
    
    Returns:
        SamPredictor: A configured SamPredictor instance with the loaded MicroSAM model.
    """
    
    # Determine checkpoint file and model type based on model size
    if model_size == "huge":
        checkpoint_name = "microsam_vit_h_lm.pth"
        model_type = "vit_h"
    elif model_size == "large":
        checkpoint_name = "microsam_vit_l_lm.pt"
        model_type = "vit_l"
    elif model_size == "base":
        checkpoint_name = "microsam_vit_b_lm.pth"
        model_type = "vit_b"
    
    # Construct full path to the checkpoint file
    microsam_checkpoint = os.path.join(checkpoint_dir, checkpoint_name)
    
    # Load MicroSAM model with the specified checkpoint, initialize on the given device & set to evaluation mode
    microsam = sam_model_registry[model_type](checkpoint=microsam_checkpoint)
    microsam.to(device)
    microsam.eval() 
    
    return SamPredictor(microsam)

def save_image_embedding(predictor: "SamPredictor", path: str):
    """
    Saves the image embedding (original size, input size, and features) of a SamPredictor object 
    to a specified file path.

    Args:
        predictor (SamPredictor): The SamPredictor object containing the image and embedding data.
        path (str): The file path where the embedding data will be saved.

    Raises:
        RuntimeError: If no image has been set in the SamPredictor using .set_image().
    """
    # Check if an image has been set; if not, raise an error.
    if not predictor.is_image_set:
        raise RuntimeError("An image must be set with .set_image(...) before embedding saving.")
    
    # Prepare the dictionary of data to be saved
    res = {
        'original_size': predictor.original_size,  # Original size of the image
        'input_size': predictor.input_size,        # Input size (after preprocessing) of the image
        'features': predictor.features,            # Features extracted from the image
        'is_image_set': True,                      # Flag indicating the image has been set
    }
    
    # Save the data dictionary to the specified path as a PyTorch file
    torch.save(res, path)

def load_image_embedding(predictor: "SamPredictor", path: str):
    """
    Loads the image embedding (original size, input size, and features) from a saved file 
    into a SamPredictor object.

    Args:
        predictor (SamPredictor): The SamPredictor object into which the embedding data will be loaded.
        path (str): The file path where the embedding data is stored.

    """
    # Load the embedding data from the specified path and ensure it's placed on the correct device
    res = torch.load(path, predictor.device)
    
    # Iterate over each key-value pair in the loaded dictionary and set them as attributes of the predictor
    for k, v in res.items():
        setattr(predictor, k, v)

def sam_predict_and_save_embedding(
        sam_predictor: "SamPredictor",
        image: np.ndarray,
        save_name: str = "embedding",
        save_dir: str = "/home/aih/maximilian.hoermann/projects/segment-anything/saved_embeddings"
    ):
    """
    Generates and saves the embedding for a given image using the SamPredictor model. 
    The embedding is saved in both PyTorch tensor (.pt) and NumPy array (.npy) formats.

    Args:
        sam_predictor (SamPredictor): The SamPredictor object used to process the image and generate the embedding.
        image (np.ndarray): The input image to process, expected as a NumPy array.
        save_name (str, optional): The base name for the saved embedding files. Defaults to "embedding".
        save_dir (str, optional): The directory to save the generated embeddings. Defaults to a predefined path.

    Saves:
        - Embedding as a PyTorch tensor (.pt) for use with SAM decoder.
        - Embedding as a NumPy array (.npy) for use with DETR.
    """
    
    # Set the input image for the predictor
    sam_predictor.set_image(image)

    # Save the embedding in PyTorch format (.pt) for use with SAM decoder
    save_path_sam = os.path.join(save_dir, save_name + ".pt")
    save_image_embedding(sam_predictor, save_path_sam)

    # Extract and format the embedding as a NumPy array
    embedding = sam_predictor.get_image_embedding().squeeze(0)  # Remove unnecessary dimensions
    embedding = embedding.permute(1, 2, 0)  # Change shape from (1, C, H, W) to (H, W, C)
    embedding = embedding.cpu().numpy()  # Convert from torch tensor to NumPy array

    # Save the embedding as a NumPy array (.npy) for DETR
    save_path_np = os.path.join(save_dir, save_name + ".npy")
    np.save(save_path_np, embedding)

    # --- Optional Test ---
    # The following code can be used to verify if the saved embedding can be reloaded correctly.
    # test = np.load(save_path_np)
    # residuum = np.sum(embedding - test)
    # print(f"Array saved successfully and reloaded. The summed residuum is: {residuum}")

def process_image(image: np.ndarray, image_path: str, predictor: "SamPredictor", encoder_name: str):
    """
    Extract embeddings from an image and saves them in a structured directory.

    Args:
        image (np.ndarray): The input image as a numpy array.
        image_path (str): The full path to the input image file, used to determine the save location.
        predictor (SamPredictor): An instance of SamPredictor for calculating image embeddings.
        encoder_name (str): Name of the encoder used, included in the save path for organization.

    Steps:
        1. Extracts the image name and directory from `image_path`.
        2. Extracts a numeric identifier from the image name.
        3. Constructs a save directory path based on `base_dir`, `save_parent_dir`, and `encoder_name`.
        4. Generates a standardized save name for the embedding file.
        5. Calls `sam_predict_and_safe_embedding` to calculate and save the embeddings.

    """

    # extract image name & image_dir
    image_dir, orig_image_name = os.path.split(image_path)
    
    # extract image number
    number = re.search(r'\d+', orig_image_name)
    if number:
        extracted_number = int(number.group())
    else:
        print("No number found in the string.")
            
    # Construct saving directory
    relative_path = image_dir.replace(base_dir, "").lstrip("/")
    save_dir = os.path.join(save_parent_dir, encoder_name, relative_path)
    os.makedirs(save_dir, exist_ok=True)

    # Construct name for saving
    save_name = f"images_patches_emb_{extracted_number:04}"

    # Calculate embeddings
    sam_predict_and_save_embedding(predictor, image, save_name, save_dir)


# ---- Helper Functions ----
def sam_predict_masks_from_bbs(
        bounding_boxes: List[np.ndarray],
        sam_predictor: "SamPredictor",
        embedding_path: str = "/home/aih/maximilian.hoermann/projects/segment-anything/saved_embeddings/embedding.pt", 
    ) -> np.ndarray:
    """
    Predicts segmentation masks for given bounding boxes using a precomputed image embedding.

    Args:
        bounding_boxes (List[np.ndarray]): List of bounding boxes in XYXY format (each of length 4) to use as input for mask prediction.
        sam_predictor (SamPredictor): The SamPredictor object used for mask prediction.
        embedding_path (str, optional): Path to the precomputed image embedding file (.pt). Defaults to a predefined path.

    Returns:
        Tuple of lists: 
        - masks_pro_img (List[np.ndarray]): A list of predicted masks for each bounding box.
        - scores_pro_img (List[np.ndarray]): A list of scores corresponding to the predicted masks.
        - logits_pro_img (List[np.ndarray]): A list of logits for each predicted mask.
    
    Notes:
        - The bounding boxes are processed one by one and the masks are sorted by their scores in descending order.
        - The function assumes that the image embedding has been precomputed and saved at the provided path.
    """
    
    # Lists to store results for all bounding boxes in the image
    masks_pro_img = []
    scores_pro_img = []
    logits_pro_img = []

    with torch.inference_mode():
        # Load the precomputed embedding for the image
        load_image_embedding(sam_predictor, embedding_path)
        
        # Loop through each bounding box and predict masks
        for bounding_box in bounding_boxes:
            masks_pro_bb, scores_pro_bb, logits_pro_bb = sam_predictor.predict(
                point_coords=None,
                point_labels=None,
                box=bounding_box[None, :],  # Provide bounding box in batch format
                multimask_output=True,      # Get multiple possible masks
            )
            
            # Sort masks, scores, and logits by score in descending order
            sorted_ind = np.argsort(scores_pro_bb)[::-1]
            masks_pro_bb = masks_pro_bb[sorted_ind]
            scores_pro_bb = scores_pro_bb[sorted_ind]
            logits_pro_bb = logits_pro_bb[sorted_ind]
            
            # Append results for this bounding box to respective lists
            masks_pro_img.append(masks_pro_bb)
            scores_pro_img.append(scores_pro_bb)
            logits_pro_img.append(logits_pro_bb)
    
    # Return all predicted masks, scores, and logits
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


# ---- Dataset for DataLoader ----
class OrganoidImageDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        return image, image_path

if __name__ == "__main__":
    # --- Parameters to set ---
    save_parent_dir = "/ictstr01/groups/shared/users/lion.gleiter/organoid_sam/patch_embeddings"
    encoder_names =  ["MicroSAM_huge"] # ["SAM_base", "MedSAM", "CellSAM", "SAM_large", "MicroSAM_huge" ]
    data_split = "train"
    print(encoder_names)
    print(data_split)


    # Get all paths of images you want to embed
    base_dir = "/ictstr01/groups/shared/users/lion.gleiter/organoid_sam/patch_images"
    datasets = os.listdir(os.path.join(base_dir, data_split))
    datasets = datasets[::-1]
    parent_dirs = [os.path.join(base_dir, data_split, dataset) for dataset in datasets]
    
    all_img_paths = []
    for parent_dir in parent_dirs:
        image_dirs = get_subdirectories_from_dir(parent_dir)
        for image_dir in image_dirs:
            image_paths = get_files_from_dir(image_dir, file_ending=".png")
            image_paths = sorted(image_paths)
            all_img_paths.extend(image_paths)

    # Instantiate the data loader
    organoid_dataset = OrganoidImageDataset(all_img_paths, transform=None)
    data_loader = DataLoader(organoid_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

    # Grab the right encoder & embed the images
    for encoder_name in encoder_names: 
        if encoder_name == "SAM_large":
            predictor = load_sam_predictor("large")
        elif encoder_name == "SAM_base":
            predictor = load_sam_predictor("base")
        elif encoder_name == "MedSAM":
            predictor = load_medsam_predictor()
        elif encoder_name == "CellSAM":
            predictor = load_cellsam_predictor()
        elif encoder_name == "MicroSAM_huge":
            predictor = load_microsam_predictor("huge")  
        elif encoder_name == "MicroSAM_large":
            predictor = load_microsam_predictor("large")  
        elif encoder_name == "MicroSAM_base":
            predictor = load_microsam_predictor("base") 
        
        # Embed the images & safe them
        for image, image_path in data_loader:
            process_image(image.squeeze(0).numpy(), image_path=image_path[0], predictor=predictor, encoder_name=encoder_name)
