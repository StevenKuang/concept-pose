"""
ConceptPose: Semantic Concept-Based 6D Pose Estimation
"""

__version__ = '1.0.0'
__author__ = 'ConceptPose Team'


def __getattr__(name):
    if name == 'OneShotPoseEstimator':
        from concept_pose.pose.one_shot_estimator import OneShotPoseEstimator
        return OneShotPoseEstimator
    raise AttributeError(f"module 'concept_pose' has no attribute {name}")


__all__ = [
    'OneShotPoseEstimator',
]
