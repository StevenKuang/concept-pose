"""
3D-2D projection utilities for pose estimation.

Provides functions for projecting 3D points to 2D image coordinates
and back-projecting 2D+depth to 3D.

Extracted from build_3d_saliency_model.py:527-565.
"""

import numpy as np
from typing import Tuple, Optional


def backproject_to_3d(
    pixel_coords: np.ndarray,
    depth_values: np.ndarray,
    intrinsics: np.ndarray,
    R: Optional[np.ndarray] = None,
    t: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Back-project 2D pixels + depth to 3D points.

    Args:
        pixel_coords: (N, 2) array of (x, y) pixel coordinates
        depth_values: (N,) array of depth values
        intrinsics: (3, 3) camera intrinsics matrix
        R: Optional (3, 3) rotation to apply (camera to object)
        t: Optional (3,) translation to apply

    Returns:
        points_3d: (N, 3) array of 3D points

    Example:
        >>> pixels = np.array([[100, 150], [200, 250]])
        >>> depths = np.array([1.0, 1.2])
        >>> K = np.array([[500, 0, 256], [0, 500, 256], [0, 0, 1]])
        >>> points = backproject_to_3d(pixels, depths, K)
    """
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    # Unproject to camera space
    px, py = pixel_coords[:, 0], pixel_coords[:, 1]
    z = depth_values
    x = (px - cx) * z / fx
    y = (py - cy) * z / fy

    points_camera = np.stack([x, y, z], axis=1)

    # Transform to object space if R, t provided
    if R is not None and t is not None:
        points_world = points_camera - t
        points_object = (R.T @ points_world.T).T
        return points_object
    else:
        return points_camera


def project_to_2d(
    points_3d: np.ndarray,
    intrinsics: np.ndarray,
    R: Optional[np.ndarray] = None,
    t: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Project 3D points to 2D image coordinates.

    Args:
        points_3d: (N, 3) array of 3D points
        intrinsics: (3, 3) camera intrinsics matrix
        R: Optional (3, 3) rotation from object to camera
        t: Optional (3,) translation from object to camera

    Returns:
        pixel_coords: (N, 2) array of (x, y) pixel coordinates
        depths: (N,) array of depth values (z coordinates in camera frame)

    Example:
        >>> points = np.array([[0.1, 0.2, 1.0], [-0.1, -0.2, 1.5]])
        >>> K = np.array([[500, 0, 256], [0, 500, 256], [0, 0, 1]])
        >>> pixels, depths = project_to_2d(points, K)
    """
    # Transform to camera space if R, t provided
    if R is not None and t is not None:
        points_camera = (R @ points_3d.T).T + t
    else:
        points_camera = points_3d

    # Project using intrinsics
    points_2d_homo = points_camera @ intrinsics.T  # (N, 3)
    pixel_coords = points_2d_homo[:, :2] / points_2d_homo[:, 2:3]  # (N, 2)
    depths = points_camera[:, 2]  # (N,)

    return pixel_coords, depths


def filter_valid_projections(
    pixel_coords: np.ndarray,
    depths: np.ndarray,
    image_shape: Tuple[int, int],
    depth_min: float = 0.01
) -> np.ndarray:
    """
    Filter projected points to keep only those visible in image.

    Args:
        pixel_coords: (N, 2) pixel coordinates
        depths: (N,) depth values
        image_shape: (H, W) image dimensions
        depth_min: Minimum valid depth

    Returns:
        valid_mask: (N,) boolean array indicating valid projections

    Example:
        >>> valid = filter_valid_projections(pixels, depths, (480, 640))
        >>> visible_pixels = pixels[valid]
    """
    H, W = image_shape

    # Check depth
    valid_depth = depths > depth_min

    # Check image bounds
    in_bounds = (
        (pixel_coords[:, 0] >= 0) & (pixel_coords[:, 0] < W) &
        (pixel_coords[:, 1] >= 0) & (pixel_coords[:, 1] < H)
    )

    return valid_depth & in_bounds


def estimate_surface_normals(
    depth_map: np.ndarray,
    intrinsics: np.ndarray,
    window_size: int = 1
) -> np.ndarray:
    """
    Estimate surface normals from depth map using gradient.

    Args:
        depth_map: (H, W) depth map
        intrinsics: (3, 3) camera intrinsics
        window_size: Size of gradient estimation window

    Returns:
        normals: (H, W, 3) surface normal map

    Example:
        >>> normals = estimate_surface_normals(depth, K)
        >>> # Normals are in camera coordinate frame
    """
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]

    # Compute depth gradients
    dz_dy, dz_dx = np.gradient(depth_map)

    # Estimate normals in camera space
    # For a point at (x, y, z), the normal is proportional to:
    # (-fx * dz_dx / z, -fy * dz_dy / z, 1)
    H, W = depth_map.shape
    normals = np.zeros((H, W, 3))

    # Avoid division by zero
    valid = depth_map > 0

    normals[valid, 0] = -dz_dx[valid] * fx / depth_map[valid]
    normals[valid, 1] = -dz_dy[valid] * fy / depth_map[valid]
    normals[valid, 2] = 1.0

    # Normalize to unit length
    norms = np.linalg.norm(normals, axis=2, keepdims=True)
    normals = np.divide(normals, norms, where=norms > 0)

    return normals
