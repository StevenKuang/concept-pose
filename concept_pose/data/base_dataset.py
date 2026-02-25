"""
Abstract base class for datasets.

All dataset implementations (HouseCat6D, NOCS, Toyota, etc.) should inherit
from this class and implement the required methods.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple
import numpy as np


class BaseDataset(ABC):
    """
    Abstract base class for pose estimation datasets.

    This class defines the interface that all dataset implementations must follow.
    It provides a common API for loading frames, querying objects, and accessing
    ground truth data across different dataset formats.
    """

    def __init__(self, data_dir: str, target_size: int = 518, **kwargs):
        """
        Initialize dataset.

        Args:
            data_dir: Root directory of the dataset
            target_size: Target image size for preprocessing
            **kwargs: Dataset-specific parameters
        """
        self.data_dir = data_dir
        self.target_size = target_size

    @abstractmethod
    def load_frame(
        self,
        frame_idx: int,
        object_name: str,
        mask_cache: Optional[Dict] = None
    ) -> Dict:
        """
        Load a single frame with all data needed for pose estimation.

        This method handles all preprocessing (resize, pad, normalization) and
        returns data in a standardized format.

        Args:
            frame_idx: Global frame index
            object_name: Object name to extract mask for
            mask_cache: Optional dict mapping (frame_idx, object_name) -> mask.
                       If provided, uses cached mask instead of loading GT.

        Returns:
            Dictionary with:
                - 'rgb': (H, W, 3) float32 [0-1]
                - 'mask': (H, W) float32 [0-1]
                - 'depth': (H, W) float32 meters
                - 'K': (3, 3) camera intrinsics (adjusted for padding)
                - 'pose': dict with 'R' (3, 3) and 't' (3,)
                - 'frame_info': metadata dict

        Raises:
            FileNotFoundError: If required files don't exist
            ValueError: If object not found in frame
        """
        pass

    @abstractmethod
    def get_valid_frames_for_object(self, object_name: str) -> List[int]:
        """
        Find all frames containing a specific object.

        Args:
            object_name: Object name (e.g., 'bottle-with-label-4')

        Returns:
            List of frame indices where object appears
        """
        pass

    @abstractmethod
    def get_all_objects(self, category: Optional[str] = None) -> List[str]:
        """
        Get all unique object instances in the dataset.

        Args:
            category: Optional category filter (e.g., 'cup', 'bottle').
                     If None, returns all objects.

        Returns:
            List of object names
        """
        pass

    @abstractmethod
    def supports_gt_masks(self) -> bool:
        """
        Check if dataset provides ground truth instance masks.

        Returns:
            True if GT masks available, False if mask generation needed
        """
        pass

    @abstractmethod
    def get_frame_info(self, frame_idx: int) -> Tuple[int, int]:
        """
        Get scene and frame metadata.

        Args:
            frame_idx: Global frame index

        Returns:
            Tuple of (scene_idx, frame_in_scene)
        """
        pass

    @abstractmethod
    def get_gt_poses(self) -> List[Dict]:
        """
        Get ground truth poses for all frames.

        Returns:
            List of dicts, one per frame, with keys:
                - 'model_names': List of object names in frame
                - 'rotations': List of (3, 3) rotation matrices
                - 'translations': List of (3,) translation vectors
                - 'instance_ids': List of instance IDs (for mask extraction)
        """
        pass

    @abstractmethod
    def get_intrinsics(self) -> np.ndarray:
        """
        Get camera intrinsics for all frames.

        Returns:
            Array of shape (N_frames, 3, 3) with camera matrices
        """
        pass

    def get_mesh_path(self, object_name: str) -> str:
        """
        Get path to 3D mesh file for object.

        Args:
            object_name: Object name

        Returns:
            Path to mesh file (e.g., .ply or .obj)

        Raises:
            NotImplementedError: If dataset doesn't provide meshes
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not provide mesh paths"
        )

    def get_model_diameter(self, object_name: str) -> float:
        """
        Get model diameter for metric computation.

        Args:
            object_name: Object name

        Returns:
            Diameter in meters

        Raises:
            NotImplementedError: If dataset doesn't provide diameters
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not provide model diameters"
        )

    def __len__(self) -> int:
        """
        Get total number of frames in dataset.

        Returns:
            Number of frames
        """
        return len(self.get_gt_poses())

    def __repr__(self) -> str:
        """String representation of dataset."""
        return (
            f"{self.__class__.__name__}("
            f"data_dir='{self.data_dir}', "
            f"num_frames={len(self)}, "
            f"target_size={self.target_size})"
        )
