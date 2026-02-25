"""
Normalization utilities for converting between object space and NOCS space.

NOCS (Normalized Object Coordinate Space) normalizes objects to [-0.5, 0.5]³
for consistent representation across different object instances.

Extracted from build_3d_saliency_model.py:567-600.
"""

import numpy as np
from typing import Tuple, Optional


def normalize_to_nocs(
    points_3d: np.ndarray,
    obj_scale: Optional[np.ndarray] = None,
    centroid: Optional[np.ndarray] = None,
    anchor_pose_mode: str = 'absolute'
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Normalize 3D points to NOCS unit cube [-0.5, 0.5]³.

    Args:
        points_3d: 3D points in object space (N, 3)
        obj_scale: Optional GT object scale (3D vector) for normalization.
                  If provided, uses this for normalization.
                  If None, normalizes to fit in [-0.5, 0.5] cube.
        centroid: Optional centroid for centering. If None, uses origin (0,0,0)
                  for 'absolute' mode, or center of mass for 'relative' mode.
        anchor_pose_mode: 'absolute' (GT pose used, points in object space) or
                         'relative' (no GT pose, points in camera space).
                         In 'relative' mode, centers at COM before scaling.

    Returns:
        normalized_points: Points in NOCS space (N, 3)
        used_centroid: Centroid that was used (3,)
        scale_factor: Scale factor applied (scalar)

    Example:
        >>> points = np.random.randn(1000, 3)  # Object-space points
        >>> nocs_points, centroid, scale = normalize_to_nocs(points)
        >>> assert nocs_points.min() >= -0.5 and nocs_points.max() <= 0.5
    """
    if len(points_3d) == 0:
        return points_3d, np.zeros(3), 1.0

    # Determine centroid for centering
    if centroid is None:
        if anchor_pose_mode == 'relative':
            # Compute center of mass (needed for camera-space points)
            centroid = points_3d.mean(axis=0)
        else:
            # Keep origin-centered (works for object-space points)
            centroid = np.zeros(3)

    # Center the points if in relative mode
    if anchor_pose_mode == 'relative':
        points_centered = points_3d - centroid
    else:
        points_centered = points_3d

    # Determine scale factor
    if obj_scale is not None:
        # Use GT scale to normalize to actual object size
        scale_factor = np.mean(obj_scale)  # Use average scale
        points_normalized = points_centered / scale_factor
    else:
        # Normalize to fit in [-0.5, 0.5] cube
        max_extent = np.abs(points_centered).max()
        scale_factor = 2 * max_extent  # Scale to unit cube
        if scale_factor > 0:
            points_normalized = points_centered / scale_factor
        else:
            points_normalized = points_centered

    return points_normalized, centroid, scale_factor


def denormalize_from_nocs(
    points_nocs: np.ndarray,
    centroid: np.ndarray,
    scale_factor: float
) -> np.ndarray:
    """
    Convert NOCS coordinates back to object space.

    Args:
        points_nocs: Points in NOCS space (N, 3)
        centroid: Original centroid (3,)
        scale_factor: Original scale factor (scalar)

    Returns:
        points_3d: Points in object space (N, 3)

    Example:
        >>> # Round-trip test
        >>> original = np.random.randn(100, 3)
        >>> nocs, c, s = normalize_to_nocs(original)
        >>> recovered = denormalize_from_nocs(nocs, c, s)
        >>> assert np.allclose(original, recovered)
    """
    return points_nocs * scale_factor + centroid


def compute_category_scale(
    scales_list: list,
    method: str = 'max'
) -> float:
    """
    Compute single scale factor for entire category from multiple object scales.

    Args:
        scales_list: List of GT scales (each is 3D vector or scalar)
        method: 'max' (ensures everything fits) or 'mean' (average)

    Returns:
        category_scale: Single scale factor for the category

    Example:
        >>> scales = [np.array([0.1, 0.12, 0.11]), np.array([0.15, 0.14, 0.13])]
        >>> cat_scale = compute_category_scale(scales)
        >>> print(cat_scale)  # 0.15 (max of all dimensions)
    """
    if not scales_list:
        return 1.0

    # Flatten all scales to handle both 3D vectors and scalars
    all_values = []
    for scale in scales_list:
        if isinstance(scale, (list, np.ndarray)):
            if np.ndim(scale) == 0:  # Scalar numpy array
                all_values.append(float(scale))
            else:
                all_values.extend(np.atleast_1d(scale))
        else:
            all_values.append(scale)

    if method == 'max':
        return max(all_values)
    elif method == 'mean':
        return np.mean(all_values)
    else:
        raise ValueError(f"Unknown method: {method}")


def get_nocs_bounds(points_nocs: np.ndarray) -> dict:
    """
    Get bounds and statistics of points in NOCS space.

    Args:
        points_nocs: Points in NOCS space (N, 3)

    Returns:
        Dictionary with bounds information

    Example:
        >>> bounds = get_nocs_bounds(nocs_points)
        >>> print(bounds['min'], bounds['max'])
    """
    if len(points_nocs) == 0:
        return {
            'min': np.array([-0.5, -0.5, -0.5]),
            'max': np.array([0.5, 0.5, 0.5]),
            'extent': np.zeros(3),
            'center': np.zeros(3)
        }

    return {
        'min': points_nocs.min(axis=0),
        'max': points_nocs.max(axis=0),
        'extent': points_nocs.max(axis=0) - points_nocs.min(axis=0),
        'center': points_nocs.mean(axis=0)
    }
