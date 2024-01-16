import numpy as np
import torch
from typing import Optional, Tuple

from segment_anything.modeling import Sam
from segment_anything import SamPredictor


class MyPredictor(SamPredictor):
    '''
    my predictor for sam model
    support training mask decoder w/o class predicting
    '''
    def __init__(
        self,
        sam_model: Sam,
    ) -> None:
        super().__init__(sam_model)
        self.model.image_encoder.eval()
        self.model.prompt_encoder.eval()
        self.model.mask_decoder.train()     

    @torch.no_grad()
    def set_image(
        self,
        image: np.ndarray,
        image_embedding: Optional[np.ndarray] = None,
        image_format: str = "RGB",
    ) -> None:
        """
        allowing masks to be predicted with the 'predict' method.

        Arguments:
          image (np.ndarray): The image for calculating masks. Expects an
            image in HWC uint8 format, with pixel values in [0, 255].
          image_embedding (np.ndarray): The image embedding for calculating masks.
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

        # self.set_torch_image(input_image_torch, image.shape[:2])
        transformed_image, original_image_size = \
            input_image_torch, image.shape[:2]
        self.reset_image()
        self.original_size = original_image_size
        self.input_size = tuple(transformed_image.shape[-2:])

        if(image_embedding is not None):
            self.features = image_embedding.to(self.device)
        else:
            input_image = self.model.preprocess(transformed_image)
            self.features = self.model.image_encoder(input_image)

        self.is_image_set = True

    def my_predict(
        self,
        point_coords: Optional[np.ndarray] = None,
        point_labels: Optional[np.ndarray] = None,
        box: Optional[np.ndarray] = None,
        mask_input: Optional[np.ndarray] = None,
        multimask_output: bool = True,
        return_logits: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        difference from method predict() of parent:
        1. support batch prompts,such as point_coords: (batch_size, num_points, 2)
        2. return tensors instead of numpy arrays
        3. constrain mask output logits to [0,1]
        4. can support predicting class logits when return_logits=True
        """
        if not self.is_image_set:
            raise RuntimeError("An image must be set with .set_image(...) before mask prediction.")

        # Transform input prompts
        coords_torch, labels_torch, box_torch, mask_input_torch = None, None, None, None
        if point_coords is not None:
            assert (
                point_labels is not None
            ), "point_labels must be supplied if point_coords is supplied."
            point_coords = self.transform.apply_coords(point_coords, self.original_size)
            coords_torch = torch.as_tensor(point_coords, dtype=torch.float, device=self.device)
            labels_torch = torch.as_tensor(point_labels, dtype=torch.int, device=self.device)
            if(coords_torch.dim()==2):
                coords_torch, labels_torch = coords_torch[None, :, :], labels_torch[None, :]
        if box is not None:
            box = self.transform.apply_boxes(box, self.original_size)
            box_torch = torch.as_tensor(box, dtype=torch.float, device=self.device)
            if(box_torch.dim()==1):
                box_torch = box_torch[None, :]
        if mask_input is not None:
            mask_input_torch = torch.as_tensor(mask_input, dtype=torch.float, device=self.device)
            if(mask_input_torch.dim()==3):
                mask_input_torch = mask_input_torch[None, :, :, :]

        masks, iou_predictions, low_res_masks, label_pred_logits = self.my_predict_torch(
            coords_torch,
            labels_torch,
            box_torch,
            mask_input_torch,
            multimask_output,
            return_logits=return_logits,
        )
        if(return_logits):
            masks = torch.functional.F.sigmoid(masks)
        # print(label_pred_logits.shape,label_pred_logits)
        return masks, iou_predictions, low_res_masks, label_pred_logits


    def my_predict_torch(
        self,
        point_coords: Optional[torch.Tensor],
        point_labels: Optional[torch.Tensor],
        boxes: Optional[torch.Tensor] = None,
        mask_input: Optional[torch.Tensor] = None,
        multimask_output: bool = True,
        return_logits: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        difference from method predict_torch() of parent:
        1. without the decorator @torch.no_grad()
        2. can pred class logits when return_logits=True
        """
        if not self.is_image_set:
            raise RuntimeError("An image must be set with .set_image(...) before mask prediction.")

        if point_coords is not None:
            points = (point_coords, point_labels)
        else:
            points = None

        # Embed prompts
        with torch.no_grad():
            sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
                points=points,
                boxes=boxes,
                masks=mask_input,
            )

        # Predict masks
        low_res_masks, iou_predictions, label_pred_logits = self.model.mask_decoder(
            image_embeddings=self.features,
            image_pe=self.model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output,
        )

        # Upscale the masks to the original image resolution
        masks = self.model.postprocess_masks(low_res_masks, self.input_size, self.original_size)

        if not return_logits:
            masks = masks > self.model.mask_threshold

        return masks, iou_predictions, low_res_masks, label_pred_logits


