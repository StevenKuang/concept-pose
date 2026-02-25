"""
Point cloud processing utilities.

Provides functions for filtering, transforming, and analyzing 3D point clouds.
Extracted from build_3d_saliency_model.py:372-419.
"""

import numpy as np
from sklearn.neighbors import NearestNeighbors
from typing import Tuple


def filter_statistical_outliers(
    points: np.ndarray,
    saliencies: np.ndarray,
    nb_neighbors: int = 20,
    std_ratio: float = 2.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Remove statistical outliers using k-NN based local density estimation.

    This is better for elongated objects than global centroid distance.
    Uses local density (mean distance to k nearest neighbors) to identify outliers.

    Extracted from build_3d_saliency_model.py:372-419.

    Args:
        points: 3D points array (N, 3)
        saliencies: Saliency vectors array (N, num_labels)
        nb_neighbors: Number of nearest neighbors to consider
        std_ratio: Standard deviation ratio for outlier threshold.
                  Points with mean distance > (global_mean + std_ratio * global_std)
                  are considered outliers.

    Returns:
        filtered_points: Cleaned points array
        filtered_saliencies: Corresponding saliencies array
        valid_mask: Boolean mask indicating which points are valid (inliers)

    Example:
        >>> points = np.random.randn(1000, 3)
        >>> # Add some outliers
        >>> points = np.vstack([points, np.random.randn(10, 3) * 10])
        >>> saliencies = np.random.rand(1010, 15)
        >>> clean_points, clean_sal, mask = filter_statistical_outliers(points, saliencies)
        >>> print(f"Removed {len(points) - len(clean_points)} outliers")
    """
    if len(points) == 0:
        return points, saliencies, np.ones(len(points), dtype=bool)

    if len(points) <= nb_neighbors:
        # Not enough points for filtering, return as is
        print(f"Warning: Only {len(points)} points, need >{nb_neighbors} for filtering")
        return points, saliencies, np.ones(len(points), dtype=bool)

    # Build k-NN index
    nbrs = NearestNeighbors(n_neighbors=nb_neighbors + 1, algorithm='auto').fit(points)
    distances, indices = nbrs.kneighbors(points)

    # Use mean distance to k nearest neighbors (excluding self at index 0)
    mean_distances = distances[:, 1:].mean(axis=1)

    # Define outlier threshold using global statistics of local densities
    global_mean = mean_distances.mean()
    global_std = mean_distances.std()
    threshold = global_mean + std_ratio * global_std

    # Create mask for valid (non-outlier) points
    valid_mask = mean_distances <= threshold

    # Filter both points and saliencies
    filtered_points = points[valid_mask]
    filtered_saliencies = saliencies[valid_mask]

    # Print filtering statistics
    num_removed = len(points) - len(filtered_points)
    removal_percentage = (num_removed / len(points)) * 100 if len(points) > 0 else 0
    print(f"Statistical outlier filtering: removed {num_removed}/{len(points)} points ({removal_percentage:.1f}%)")
    print(f"  Threshold: {threshold:.4f}, Mean distance: {mean_distances.mean():.4f} ± {mean_distances.std():.4f}")

    return filtered_points, filtered_saliencies, valid_mask


def compute_point_cloud_bounds(points: np.ndarray) -> dict:
    """
    Compute bounding box and statistics for point cloud.

    Args:
        points: 3D points (N, 3)

    Returns:
        Dictionary with bounds information

    Example:
        >>> bounds = compute_point_cloud_bounds(points)
        >>> print(bounds['extent'])  # (dx, dy, dz)
    """
    if len(points) == 0:
        return {
            'min': np.zeros(3),
            'max': np.zeros(3),
            'extent': np.zeros(3),
            'center': np.zeros(3)
        }

    return {
        'min': points.min(axis=0),
        'max': points.max(axis=0),
        'extent': points.max(axis=0) - points.min(axis=0),
        'center': points.mean(axis=0)
    }


def downsample_points(
    points: np.ndarray,
    saliencies: np.ndarray,
    target_count: int,
    method: str = 'random'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Downsample point cloud to target number of points.

    Args:
        points: 3D points (N, 3)
        saliencies: Saliency vectors (N, num_labels)
        target_count: Desired number of points
        method: 'random' or 'fps' (farthest point sampling)

    Returns:
        downsampled_points: (target_count, 3)
        downsampled_saliencies: (target_count, num_labels)

    Example:
        >>> # Reduce large point cloud
        >>> small_points, small_sal = downsample_points(points, saliencies, 10000)
    """
    if len(points) <= target_count:
        return points, saliencies

    if method == 'random':
        # Random sampling
        indices = np.random.choice(len(points), target_count, replace=False)
        return points[indices], saliencies[indices]

    elif method == 'fps':
        # Farthest point sampling (better coverage but slower)
        indices = farthest_point_sampling(points, target_count)
        return points[indices], saliencies[indices]

    else:
        raise ValueError(f"Unknown downsampling method: {method}")


def farthest_point_sampling(points: np.ndarray, num_samples: int) -> np.ndarray:
    """
    Sample points using farthest point sampling for better spatial coverage.

    Args:
        points: 3D points (N, 3)
        num_samples: Number of points to sample

    Returns:
        indices: Indices of sampled points
    """
    N = len(points)
    if num_samples >= N:
        return np.arange(N)

    # Initialize with random point
    sampled_indices = [np.random.randint(N)]
    distances = np.full(N, np.inf)

    for _ in range(1, num_samples):
        # Update distances to nearest sampled point
        last_point = points[sampled_indices[-1]]
        dist_to_last = np.linalg.norm(points - last_point, axis=1)
        distances = np.minimum(distances, dist_to_last)

        # Select farthest point
        farthest_idx = np.argmax(distances)
        sampled_indices.append(farthest_idx)

    return np.array(sampled_indices)
