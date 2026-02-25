"""
Demo module for ConceptPose in-the-wild inference.

Provides tools for pose estimation on arbitrary image pairs without dataset dependencies.
"""

__all__ = [
    'WildPoseEstimator',
    'estimate_relative_pose',
]


def __getattr__(name):
    if name in ('WildPoseEstimator', 'estimate_relative_pose'):
        from concept_pose.demo.wild_pose_estimator import (
            WildPoseEstimator,
            estimate_relative_pose,
        )
        if name == 'WildPoseEstimator':
            return WildPoseEstimator
        return estimate_relative_pose
    raise AttributeError(f"module 'concept_pose.demo' has no attribute {name}")
