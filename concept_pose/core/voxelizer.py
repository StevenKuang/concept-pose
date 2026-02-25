"""
Voxelization utilities for converting 3D point clouds to voxel grids.

This module provides pure functions for voxelization, extracted from the original
SaliencyModelBuilder class to enable reuse and easier testing.
"""

import numpy as np
from typing import Tuple, Optional
from concept_pose.core.point_cloud import filter_statistical_outliers
from concept_pose.core.normalization import normalize_to_nocs


def voxelize_points(
    points_3d: np.ndarray,
    saliencies_3d: np.ndarray,
    voxel_resolution: int = 64,
    aggregation_method: str = 'mean',
    voxel_size: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert 3D points and saliencies to voxel grid.

    Extracted from build_3d_saliency_model.py:611-656.

    Args:
        points_3d: Normalized 3D points in [-0.5, 0.5]³ space (N, 3)
        saliencies_3d: Semantic saliency vectors per point (N, num_labels)
        voxel_resolution: Voxel grid resolution (creates resolution³ grid)
        aggregation_method: How to aggregate multiple points in same voxel ('mean' or 'median')
        voxel_size: Alternative name for voxel_resolution (for backward compatibility)

    Returns:
        voxel_grid: (res, res, res, num_labels) array of aggregated saliencies
        voxel_counts: (res, res, res) array with number of points per voxel
        valid_voxels: (res, res, res) boolean mask of voxels with data

    Example:
        >>> points = np.random.rand(1000, 3) - 0.5  # Points in [-0.5, 0.5]
        >>> saliencies = np.random.rand(1000, 15)  # 15 semantic labels
        >>> voxel_grid, counts, valid = voxelize_points(points, saliencies, 64)
        >>> print(voxel_grid.shape)  # (64, 64, 64, 15)
    """
    if voxel_size is not None:
        voxel_resolution = voxel_size

    if len(points_3d) == 0:
        # Handle empty point cloud
        num_labels = saliencies_3d.shape[1] if len(saliencies_3d) > 0 else 0
        return (
            np.zeros((voxel_resolution, voxel_resolution, voxel_resolution, num_labels)),
            np.zeros((voxel_resolution, voxel_resolution, voxel_resolution)),
            np.zeros((voxel_resolution, voxel_resolution, voxel_resolution), dtype=bool)
        )

    # Initialize voxel grid
    num_labels = saliencies_3d.shape[1]
    voxel_grid = np.zeros((voxel_resolution, voxel_resolution, voxel_resolution, num_labels))
    voxel_counts = np.zeros((voxel_resolution, voxel_resolution, voxel_resolution))

    # Map points to voxel indices (NOCS is [-0.5, 0.5])
    voxel_indices = ((points_3d + 0.5) * voxel_resolution).astype(int)
    voxel_indices = np.clip(voxel_indices, 0, voxel_resolution - 1)

    # Aggregate saliencies per voxel
    for i, (vx, vy, vz) in enumerate(voxel_indices):
        voxel_grid[vx, vy, vz] += saliencies_3d[i]
        voxel_counts[vx, vy, vz] += 1

    # Average where multiple points exist
    valid_voxels = voxel_counts > 0
    if aggregation_method == 'mean':
        voxel_grid[valid_voxels] /= voxel_counts[valid_voxels, np.newaxis]
    elif aggregation_method == 'median':
        # For median, we'd need to store all values per voxel (more complex)
        # For now, fall back to mean
        voxel_grid[valid_voxels] /= voxel_counts[valid_voxels, np.newaxis]
    else:
        raise ValueError(f"Unknown aggregation method: {aggregation_method}")

    return voxel_grid, voxel_counts, valid_voxels


def devoxelize_grid(
    voxel_grid: np.ndarray,
    valid_voxels: np.ndarray,
    voxel_resolution: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract occupied voxels from grid back to point representation.

    Args:
        voxel_grid: (res, res, res, num_labels) voxel grid
        valid_voxels: (res, res, res) boolean mask
        voxel_resolution: Voxel grid resolution

    Returns:
        voxel_centers: (N, 3) positions of occupied voxels in NOCS space
        voxel_saliencies: (N, num_labels) saliency vectors

    Example:
        >>> # Extract compact representation from voxel grid
        >>> centers, saliencies = devoxelize_grid(voxel_grid, valid_voxels, 64)
    """
    occupied_indices = np.where(valid_voxels)

    # Convert voxel indices to NOCS coordinates [-0.5, 0.5]
    voxel_centers = np.stack(occupied_indices, axis=1).astype(np.float64)
    voxel_centers = (voxel_centers + 0.5) / voxel_resolution - 0.5

    # Extract saliencies for occupied voxels
    voxel_saliencies = voxel_grid[occupied_indices]

    return voxel_centers, voxel_saliencies


def compute_voxel_occupancy_stats(valid_voxels: np.ndarray) -> dict:
    """
    Compute statistics about voxel grid occupancy.

    Args:
        valid_voxels: (res, res, res) boolean mask

    Returns:
        Dictionary with occupancy statistics

    Example:
        >>> stats = compute_voxel_occupancy_stats(valid_voxels)
        >>> print(stats['occupancy_percentage'])
    """
    total_voxels = valid_voxels.size
    occupied_voxels = np.sum(valid_voxels)
    occupancy_percentage = (occupied_voxels / total_voxels) * 100

    return {
        'total_voxels': int(total_voxels),
        'occupied_voxels': int(occupied_voxels),
        'empty_voxels': int(total_voxels - occupied_voxels),
        'occupancy_percentage': float(occupancy_percentage)
    }


def voxelize_point_cloud_with_params(
    points_3d: np.ndarray,
    saliencies: np.ndarray,
    nocs_scale: float,
    nocs_centroid: np.ndarray = None,
    voxel_resolution: int = 64,
    aggregation_method: str = 'mean',
    apply_outlier_filter: bool = True,
    nb_neighbors: int = 20,
    std_ratio: float = 2.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Complete pipeline for voxelizing point cloud with given normalization parameters.

    This function encapsulates the full voxelization pipeline used in both anchor
    and query frame processing:
    1. Optional outlier filtering
    2. Normalize to NOCS space using provided scale (and auto-computed centroid)
    3. Voxelize
    4. Extract compact representation
    5. Denormalize back to original space

    IMPORTANT: This function computes the centroid from the input points rather than
    using a provided centroid. This is correct for camera-space (relative mode)
    voxelization where each frame has its own camera-centric coordinate frame.
    Only the scale (object size) is shared between frames.

    Args:
        points_3d: (N, 3) 3D points in camera/object space
        saliencies: (N, C) semantic saliency vectors
        nocs_scale: Scale factor for NOCS normalization (from anchor model, represents object size)
        nocs_centroid: DEPRECATED - Centroid is computed from points_3d automatically
        voxel_resolution: Voxel grid resolution (default: 64)
        aggregation_method: 'mean' or 'median' for voxel aggregation
        apply_outlier_filter: Whether to filter statistical outliers before voxelization
        nb_neighbors: Number of neighbors for outlier detection
        std_ratio: Std ratio threshold for outlier detection

    Returns:
        voxelized_points: (M, 3) voxel centers in original space (M <= N)
        voxelized_saliencies: (M, C) aggregated saliencies per voxel

    Example:
        >>> # Anchor: build model and save scale
        >>> anchor_points, anchor_sal = backproject_depth(...)
        >>> voxel_points, voxel_sal, scale, _ = build_voxel_model(anchor_points, anchor_sal)
        >>>
        >>> # Query: voxelize using anchor's scale only (centroid computed automatically)
        >>> query_points, query_sal = backproject_depth(...)
        >>> query_voxels, query_voxel_sal = voxelize_point_cloud_with_params(
        ...     query_points, query_sal, nocs_scale=scale, voxel_resolution=64
        ... )
    """
    if len(points_3d) == 0:
        return np.array([]), np.array([])

    # Step 1: Optional outlier filtering
    if apply_outlier_filter:
        points_3d, saliencies, _ = filter_statistical_outliers(
            points_3d, saliencies,
            nb_neighbors=nb_neighbors,
            std_ratio=std_ratio
        )

        if len(points_3d) == 0:
            return np.array([]), np.array([])

    # Step 2: Normalize to NOCS space using provided scale
    # IMPORTANT: In camera-space mode, we use anchor's scale but compute our own centroid
    # because the centroid is view-specific (different for each camera pose)
    query_centroid = np.mean(points_3d, axis=0)  # Compute query's own centroid
    points_nocs, used_centroid, used_scale = normalize_to_nocs(
        points_3d,
        obj_scale=nocs_scale,  # Use anchor's scale (object size)
        centroid=query_centroid,  # Use query's own centroid (view-specific)
        anchor_pose_mode='relative'  # Center at centroid, then scale
    )

    # Step 3: Voxelize in NOCS space
    voxel_grid, voxel_counts, valid_voxels = voxelize_points(
        points_nocs,
        saliencies,
        voxel_resolution=voxel_resolution,
        aggregation_method=aggregation_method
    )

    # Step 4: Extract compact representation (occupied voxels only)
    voxel_centers_nocs, voxel_saliencies = devoxelize_grid(
        voxel_grid,
        valid_voxels,
        voxel_resolution
    )

    if len(voxel_centers_nocs) == 0:
        return np.array([]), np.array([])

    # Step 5: Denormalize back to camera space
    # Reverse the normalization: points = nocs_points * scale + centroid
    voxel_centers_camera = voxel_centers_nocs * used_scale + used_centroid

    return voxel_centers_camera, voxel_saliencies
