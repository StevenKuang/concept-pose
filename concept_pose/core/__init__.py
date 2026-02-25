"""
Core algorithms for voxel-based pose estimation.

This module contains reusable, pure functions for:
- Voxelization and devoxelization
- NOCS normalization
- Point cloud processing and filtering
- 3D-2D projection utilities
"""

from concept_pose.core.voxelizer import (
    voxelize_points,
    devoxelize_grid,
    compute_voxel_occupancy_stats,
)

from concept_pose.core.normalization import (
    normalize_to_nocs,
    denormalize_from_nocs,
    compute_category_scale,
    get_nocs_bounds,
)

from concept_pose.core.point_cloud import (
    filter_statistical_outliers,
    compute_point_cloud_bounds,
    downsample_points,
    farthest_point_sampling,
)

from concept_pose.core.projection import (
    backproject_to_3d,
    project_to_2d,
    filter_valid_projections,
    estimate_surface_normals,
)

__all__ = [
    # Voxelization
    'voxelize_points',
    'devoxelize_grid',
    'compute_voxel_occupancy_stats',
    
    # Normalization
    'normalize_to_nocs',
    'denormalize_from_nocs',
    'compute_category_scale',
    'get_nocs_bounds',
    
    # Point cloud
    'filter_statistical_outliers',
    'compute_point_cloud_bounds',
    'downsample_points',
    'farthest_point_sampling',
    
    # Projection
    'backproject_to_3d',
    'project_to_2d',
    'filter_valid_projections',
    'estimate_surface_normals',
]
