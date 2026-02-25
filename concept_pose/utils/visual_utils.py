"""
Visual Utility Functions
========================

Utility functions for image and saliency map processing.
Extracted from vlm_pose/visual_util.py for self-contained package.
"""

import numpy as np
import torch
from typing import Optional, Tuple, Union


def pad_and_resize_saliency_map(saliency_map, original_coords, target_size):
    """
    Pad and resize the saliency map to the target size.

    Args:
        saliency_map: Tensor of shape (C, H, W) with saliency scores
        original_coords: Bounding box coordinates [x1, y1, x2, y2, ...]
        target_size: Target image size (H, W) for output

    Returns:
        Tensor of shape (C, target_H, target_W) with padded and resized saliency
    """
    bbox = np.array(original_coords[:4]).astype(np.int32)
    ret_map = torch.zeros(saliency_map.shape[0], target_size[0], target_size[1])
    resize_shape = (bbox[3] - bbox[1], bbox[2] - bbox[0])
    resized_saliency_map = torch.nn.functional.interpolate(
        saliency_map.unsqueeze(1),
        size=resize_shape,
        mode='bilinear'
    )

    for i in range(saliency_map.shape[0]):
        ret_map[i, bbox[1]:bbox[3], bbox[0]:bbox[2]] = resized_saliency_map[i].squeeze(1)

    return ret_map


def resize_saliency_with_padding(
    saliency: Union[np.ndarray, torch.Tensor],
    target_size: Tuple[int, int],
    padding_coords: Optional[Union[np.ndarray, Tuple]] = None,
    input_format: str = 'CHW'
) -> Union[np.ndarray, torch.Tensor]:
    """
    Generic function to resize saliency maps with optional padding handling.

    This function correctly handles the resizing of saliency maps that correspond to
    padded images. It ensures that saliency values are only placed in the actual
    content region, not the padded regions.

    Workflow:
    1. If padding_coords provided:
       - Extract the content region coordinates [x1, y1, x2, y2]
       - Resize saliency to match content size: (y2-y1, x2-x1)
       - Create zero-filled output of target_size
       - Place resized saliency at correct position [y1:y2, x1:x2]

    2. If no padding_coords:
       - Simple resize to target_size

    Args:
        saliency: Saliency map
            - Shape: (C, H, W) or (1, C, H, W) if torch tensor
            - Shape: (C, H, W) if numpy array
        target_size: Target output size as (H, W)
        padding_coords: Optional padding coordinates [x1, y1, x2, y2, ...]
            - Specifies where actual content is within the padded image
            - If None, assumes no padding (simple resize)
        input_format: Format of input ('CHW' or 'BCHW' for batch)

    Returns:
        Resized saliency map in same format as input
            - If input is torch: returns torch tensor
            - If input is numpy: returns numpy array

    Example:
        >>> # Original image: 480x640, padded to 640x640
        >>> # Content region: y=80 to y=560, x=0 to x=640
        >>> coords = [0, 80, 640, 560, 640, 480]  # [x1, y1, x2, y2, orig_w, orig_h]
        >>> saliency_384 = extract_saliency(...)  # (C, 384, 384) from SigLIP
        >>> saliency_640 = resize_saliency_with_padding(
        ...     saliency_384, target_size=(640, 640), padding_coords=coords
        ... )
        >>> # Result: (C, 640, 640) with zeros in padded regions
    """
    is_numpy = isinstance(saliency, np.ndarray)

    # Convert to torch if needed
    if is_numpy:
        saliency = torch.from_numpy(saliency)

    # Handle input shape
    squeeze_batch = False
    if saliency.dim() == 3:  # (C, H, W)
        saliency = saliency.unsqueeze(0)  # (1, C, H, W)
        squeeze_batch = True

    B, C, H_sal, W_sal = saliency.shape
    target_H, target_W = target_size

    if padding_coords is not None:
        # Extract padding coordinates
        if isinstance(padding_coords, (list, tuple)):
            padding_coords = np.array(padding_coords)

        x1, y1, x2, y2 = padding_coords[:4].astype(np.int32)
        content_h = y2 - y1
        content_w = x2 - x1

        # Create output filled with zeros (for padded regions)
        output = torch.zeros(B, C, target_H, target_W, dtype=saliency.dtype, device=saliency.device)

        # Resize saliency to content size
        resized_content = torch.nn.functional.interpolate(
            saliency,
            size=(content_h, content_w),
            mode='bilinear',
            align_corners=False
        )

        # Place resized content at correct position
        output[:, :, y1:y2, x1:x2] = resized_content
    else:
        # No padding - simple resize
        output = torch.nn.functional.interpolate(
            saliency,
            size=(target_H, target_W),
            mode='bilinear',
            align_corners=False
        )

    # Restore original batch dimension
    if squeeze_batch:
        output = output.squeeze(0)  # (C, H, W)

    # Convert back to numpy if needed
    if is_numpy:
        output = output.cpu().numpy()

    return output


def masks_to_bboxes(masks):
    """
    Convert binary masks to bounding boxes.

    Args:
        masks: Tensor of shape (N, H, W) with binary masks

    Returns:
        Tensor of shape (N, 4) with bounding boxes [x1, y1, x2, y2]
    """
    bboxes = []
    for i in range(masks.shape[0]):
        mask = masks[i]
        coords = mask.nonzero(as_tuple=True)
        if coords[0].numel() == 0:
            # Empty mask - return zero bbox
            bbox = [0, 0, 0, 0]
        else:
            # Bounding box: [x1, y1, x2, y2]
            bbox = [
                coords[1].min(),  # x1
                coords[0].min(),  # y1
                coords[1].max(),  # x2
                coords[0].max()   # y2
            ]
        bboxes.append(bbox)
    return torch.tensor(bboxes)
