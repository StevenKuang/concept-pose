"""
Data preprocessing utilities for HouseCat6D dataset.

Provides functions for image/mask/depth resizing, padding, and preprocessing
to maintain consistency across RGB images, masks, and depth maps.

Extracted from datasets/ds_housecat.py and build_3d_saliency_model.py.
"""

import numpy as np
import cv2
import torch
from PIL import Image
from typing import Tuple, Optional


def resize_and_pad_image(
    img: Image.Image,
    target_size: int = 384
) -> Tuple[torch.Tensor, Tuple[int, int, int, int, int, int]]:
    """
    Resize and pad image to square target size while preserving aspect ratio.

    Extracted from ds_housecat.py:222-247.

    Args:
        img: PIL Image in RGB format
        target_size: Target square size (default 384)

    Returns:
        img_tensor: (3, target_size, target_size) tensor in [0, 1]
        coords: (paste_x, paste_y, paste_x_end, paste_y_end, orig_w, orig_h)
                Coordinates of the resized image within the padded square

    Example:
        >>> img = Image.open('image.jpg').convert('RGB')
        >>> img_tensor, coords = resize_and_pad_image(img, 384)
        >>> paste_x, paste_y, paste_x_end, paste_y_end, orig_w, orig_h = coords
    """
    orig_w, orig_h = img.size

    # Calculate new dimensions preserving aspect ratio
    aspect = orig_w / orig_h
    if aspect > 1:
        new_w = target_size
        new_h = int(target_size / aspect)
    else:
        new_h = target_size
        new_w = int(target_size * aspect)

    # Resize image
    img = img.resize((new_w, new_h), Image.BILINEAR)

    # Create padded image with black borders
    padded_img = Image.new('RGB', (target_size, target_size), (0, 0, 0))
    paste_x = (target_size - new_w) // 2
    paste_y = (target_size - new_h) // 2
    padded_img.paste(img, (paste_x, paste_y))

    # Convert to tensor [0, 1]
    img_tensor = torch.from_numpy(np.array(padded_img)).float() / 255.0
    img_tensor = img_tensor.permute(2, 0, 1)  # (H, W, 3) -> (3, H, W)

    # Return coordinates for inverse mapping
    paste_x_end = paste_x + new_w
    paste_y_end = paste_y + new_h
    coords = (paste_x, paste_y, paste_x_end, paste_y_end, orig_w, orig_h)

    return img_tensor, coords


def resize_and_pad_mask(
    mask: np.ndarray,
    target_size: int = 384,
    coords: Optional[Tuple[int, int, int, int, int, int]] = None
) -> torch.Tensor:
    """
    Resize and pad mask to match image preprocessing.

    Extracted from ds_housecat.py:409-432.

    Args:
        mask: Binary mask array (H, W) with values 0-255
        target_size: Target square size (default 384)
        coords: Optional precomputed coordinates from resize_and_pad_image.
                If provided, uses these coordinates directly.
                Format: (paste_x, paste_y, paste_x_end, paste_y_end, orig_w, orig_h)

    Returns:
        mask_tensor: (target_size, target_size) tensor in [0, 1]

    Example:
        >>> # Using with precomputed coords
        >>> img_tensor, coords = resize_and_pad_image(img)
        >>> mask_tensor = resize_and_pad_mask(mask, coords=coords)
    """
    mask_pil = Image.fromarray(mask)

    if coords is not None:
        # Use precomputed coordinates
        paste_x, paste_y, paste_x_end, paste_y_end, orig_w, orig_h = coords
        # Ensure all values are integers (coords may come from float32 tensors)
        paste_x, paste_y = int(paste_x), int(paste_y)
        paste_x_end, paste_y_end = int(paste_x_end), int(paste_y_end)
        orig_w, orig_h = int(orig_w), int(orig_h)
        new_w = paste_x_end - paste_x
        new_h = paste_y_end - paste_y
    else:
        # Compute coordinates (same logic as image)
        orig_w, orig_h = mask_pil.size
        aspect = orig_w / orig_h
        if aspect > 1:
            new_w = target_size
            new_h = int(target_size / aspect)
        else:
            new_h = target_size
            new_w = int(target_size * aspect)
        paste_x = (target_size - new_w) // 2
        paste_y = (target_size - new_h) // 2

    # Resize mask (use NEAREST to preserve binary values)
    mask_pil = mask_pil.resize((new_w, new_h), Image.NEAREST)

    # Create padded mask with zeros
    padded_mask = Image.new('L', (target_size, target_size), 0)
    padded_mask.paste(mask_pil, (paste_x, paste_y))

    # Convert to tensor [0, 1]
    mask_tensor = torch.from_numpy(np.array(padded_mask)).float() / 255.0

    return mask_tensor


def preprocess_depth_map(
    depth_img: np.ndarray,
    coords: Tuple[int, int, int, int, int, int],
    target_size: int = 384,
    depth_scale: float = 1000.0
) -> np.ndarray:
    """
    Preprocess depth map to match RGB image preprocessing.

    Extracted from build_3d_saliency_model.py:446-465.

    Args:
        depth_img: Depth map (H, W) in dataset-specific units
        coords: Coordinates from resize_and_pad_image
                Format: (paste_x, paste_y, paste_x_end, paste_y_end, orig_w, orig_h)
        target_size: Target square size (default 384)
        depth_scale: Scale factor to convert to meters (default 1000.0 for mm->m)

    Returns:
        depth_padded: Preprocessed depth in meters (target_size, target_size) as float32

    Example:
        >>> img_tensor, coords = resize_and_pad_image(img)
        >>> depth_preprocessed = preprocess_depth_map(depth_raw, coords, depth_scale=1000.0)
    """
    # Ensure all values are integers (coords may come from float32 tensors)
    paste_x, paste_y, paste_x_end, paste_y_end, orig_w, orig_h = coords
    paste_x, paste_y = int(paste_x), int(paste_y)
    paste_x_end, paste_y_end = int(paste_x_end), int(paste_y_end)
    orig_w, orig_h = int(orig_w), int(orig_h)

    # Calculate resize dimensions (same as image/mask)
    aspect = orig_w / orig_h
    if aspect > 1:
        new_w = target_size
        new_h = int(target_size / aspect)
    else:
        new_h = target_size
        new_w = int(target_size * aspect)

    # Resize depth (use NEAREST to preserve depth values)
    depth_resized = cv2.resize(
        depth_img,
        (int(new_w), int(new_h)),
        interpolation=cv2.INTER_NEAREST
    )

    # Create padded depth with zeros
    depth_padded = np.zeros((target_size, target_size), dtype=np.uint16)
    depth_padded[int(paste_y):int(paste_y_end), int(paste_x):int(paste_x_end)] = depth_resized

    # Convert to meters using dataset-specific scale
    return depth_padded.astype(np.float32) / depth_scale


def extract_object_mask_from_instance(
    instance_mask: np.ndarray,
    instance_id: int,
    coords: Tuple[int, int, int, int, int, int],
    target_size: int = 384
) -> torch.Tensor:
    """
    Extract and preprocess binary mask for a specific object from instance mask.

    Combines extraction + resize_and_pad_mask.
    Extracted from ds_housecat.py:403-432.

    Args:
        instance_mask: Instance segmentation mask (H, W) with instance IDs
        instance_id: ID of the object to extract
        coords: Coordinates from resize_and_pad_image
        target_size: Target square size (default 384)

    Returns:
        mask_tensor: Binary mask tensor (target_size, target_size) in [0, 1]

    Example:
        >>> # Extract mask for object with instance_id=3
        >>> object_mask = extract_object_mask_from_instance(
        ...     instance_mask, instance_id=3, coords=coords
        ... )
    """
    # Create binary mask for this object
    object_mask = (instance_mask == instance_id).astype(np.uint8) * 255

    # Apply same resize/padding as images
    return resize_and_pad_mask(object_mask, target_size, coords)


def get_unpadded_region(
    padded_tensor: torch.Tensor,
    coords: Tuple[int, int, int, int, int, int]
) -> torch.Tensor:
    """
    Extract the unpadded region from a padded tensor.

    Args:
        padded_tensor: Padded tensor (C, H, W) or (H, W)
        coords: Coordinates from resize_and_pad_image
                Format: (paste_x, paste_y, paste_x_end, paste_y_end, orig_w, orig_h)

    Returns:
        unpadded: Cropped tensor containing only the valid region

    Example:
        >>> img_tensor, coords = resize_and_pad_image(img)
        >>> unpadded = get_unpadded_region(img_tensor, coords)
    """
    paste_x, paste_y, paste_x_end, paste_y_end, orig_w, orig_h = coords

    if padded_tensor.ndim == 3:
        # (C, H, W)
        return padded_tensor[:, paste_y:paste_y_end, paste_x:paste_x_end]
    elif padded_tensor.ndim == 2:
        # (H, W)
        return padded_tensor[paste_y:paste_y_end, paste_x:paste_x_end]
    else:
        raise ValueError(f"Unsupported tensor shape: {padded_tensor.shape}")


def batch_resize_and_pad_images(
    images: list,
    target_size: int = 384
) -> Tuple[torch.Tensor, list]:
    """
    Batch process multiple images.

    Args:
        images: List of PIL Images
        target_size: Target square size

    Returns:
        image_batch: (N, 3, target_size, target_size) tensor
        coords_list: List of coordinate tuples for each image

    Example:
        >>> images = [Image.open(f'img{i}.jpg') for i in range(10)]
        >>> batch_tensor, coords_list = batch_resize_and_pad_images(images)
    """
    image_tensors = []
    coords_list = []

    for img in images:
        img_tensor, coords = resize_and_pad_image(img, target_size)
        image_tensors.append(img_tensor)
        coords_list.append(coords)

    image_batch = torch.stack(image_tensors)
    return image_batch, coords_list
