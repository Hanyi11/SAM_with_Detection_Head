import numpy as np
import torch

from segment_anything.modeling import Sam

from typing import Optional, Tuple

from .utils.transforms import ResizeLongestSide


class SamEncoder:
    def __init__(
        self,
        sam_model: Sam,
        device,
    ) -> None:
        """
        Uses SAM to calculate the image embedding for an image.

        Arguments:
          sam_model (Sam): The model to use for mask prediction.
        """
        super().__init__()
        self.model = sam_model
        self.transform = ResizeLongestSide(sam_model.image_encoder.img_size)
        self.reset_image()
        self.device = device

    def set_image(
        self,
        image: np.ndarray,
        image_format: str = "RGB",
    ) -> None:
        """
        Calculates the image embeddings for the provided image, allowing
        masks to be predicted with the 'predict' method.

        Arguments:
          image (np.ndarray): The image for calculating masks. Expects an
            image in HWC uint8 format, with pixel values in [0, 255].
          image_format (str): The color format of the image, in ['RGB', 'BGR'].
        """
        assert image_format in [
            "RGB",
            "BGR",
        ], f"image_format must be in ['RGB', 'BGR'], is {image_format}."
        if image_format != self.model.image_format:
            image = image[..., ::-1]

        # Transform the image to the form expected by the model
        input_image = self.transform.apply_image(image)
        input_image_torch = torch.as_tensor(input_image, device=self.device)
        input_image_torch = input_image_torch.permute(2, 0, 1).contiguous()[None, :, :, :]

        features = self.set_torch_image(input_image_torch, image.shape[:2])
        return features

    @torch.no_grad()
    def set_torch_image(
        self,
        transformed_image: torch.Tensor,
        original_image_size: Tuple[int, ...],
    ) -> None:
        """
        Calculates the image embeddings for the provided image, allowing
        masks to be predicted with the 'predict' method. Expects the input
        image to be already transformed to the format expected by the model.

        Arguments:
          transformed_image (torch.Tensor): The input image, with shape
            1x3xHxW, which has been transformed with ResizeLongestSide.
          original_image_size (tuple(int, int)): The size of the image
            before transformation, in (H, W) format.
        """
        assert (
            len(transformed_image.shape) == 4
            and transformed_image.shape[1] == 3
            and max(*transformed_image.shape[2:]) == self.model.image_encoder.img_size
        ), f"set_torch_image input must be BCHW with long side {self.model.image_encoder.img_size}."
        self.reset_image()

        self.original_size = original_image_size
        self.input_size = tuple(transformed_image.shape[-2:])
        input_image = self.model.preprocess(transformed_image)
        #self.features = self.model.image_encoder(input_image)
        #self.is_image_set = True
        return self.model.image_encoder(input_image)
        

    def reset_image(self) -> None:
        """Resets the currently set image."""
        self.is_image_set = False
        self.features = None
        self.orig_h = None
        self.orig_w = None
        self.input_h = None
        self.input_w = None
    
    def set_box_coordinates(self, label: np.array):  # This function can only deal with one GT each time
        unique_values = np.unique(label)
        bboxes = []
        original_size = label.shape

        for value in unique_values:
            if value == 0:  # Skip the background
                continue

            mask = label == value
            # Find bounding box
            positions = np.argwhere(mask)
            x_min, y_min = positions.min(axis=0)
            x_max, y_max = positions.max(axis=0)
            bbox = np.array([x_min, y_min, x_max, y_max])
            bboxes.append(bbox)

        # Convert list of bounding boxes to a numpy array for efficient sorting
        bboxes = np.array(bboxes)

        # Transform the bounding boxes
        transformed_bboxes = self.transform.apply_boxes(bboxes, original_size)

        # Inline conversion of bounding boxes to (cx, cy, w, h) format
        x0, y0, x1, y1 = transformed_bboxes[:, 0], transformed_bboxes[:, 1], transformed_bboxes[:, 2], transformed_bboxes[:, 3]
        cx = (x0 + x1) / 2
        cy = (y0 + y1) / 2
        w = x1 - x0
        h = y1 - y0
        transformed_bboxes = np.stack((cx, cy, w, h), axis=-1)

        return transformed_bboxes

    def trans_box_coordinates(self, bboxes: np.array, original_size):  # This function can only deal with one GT each time
        
        # Transform the bounding boxes
        transformed_bboxes = self.transform.apply_boxes(bboxes, original_size)

        # Inline conversion of bounding boxes to (cx, cy, w, h) format
        x0, y0, x1, y1 = transformed_bboxes[:, 0], transformed_bboxes[:, 1], transformed_bboxes[:, 2], transformed_bboxes[:, 3]
        cx = (x0 + x1) / 2
        cy = (y0 + y1) / 2
        w = x1 - x0
        h = y1 - y0
        transformed_bboxes = np.stack((cx, cy, w, h), axis=-1)

        return transformed_bboxes
    