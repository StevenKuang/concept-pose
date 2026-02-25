"""
DatasetReal275: NOCS Real275 test set loader.

Loads the NOCS Real275 real-world test set with 2,754 frames across 6 scenes.
Contains 18 object instances from 6 categories: bottle, bowl, camera, can, laptop, mug.
"""

import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import cv2
from PIL import Image

from .base_dataset import BaseDataset
from .preprocessing import (
    resize_and_pad_image,
    resize_and_pad_mask,
    preprocess_depth_map
)


def extract_pure_rotation(R: np.ndarray) -> np.ndarray:
    """
    Extract pure rotation from a matrix that may contain scale.

    Real275/NOCS datasets store GT poses with both rotation and scale.
    This function extracts only the rotation component via SVD.

    Args:
        R: (3, 3) rotation matrix (may contain scale)

    Returns:
        R_pure: (3, 3) pure rotation matrix (det=1, orthogonal)
    """
    det_R = np.linalg.det(R)

    # If R is already a pure rotation, return as-is
    if abs(det_R - 1.0) < 0.01:
        return R.astype(np.float32)

    # Extract pure rotation via SVD: R = U @ S @ V.T
    # Pure rotation is: R_pure = U @ V.T
    U, S, Vt = np.linalg.svd(R.astype(np.float64))
    R_pure = (U @ Vt).astype(np.float32)

    return R_pure


# NOCS category ID to name mapping
CATEGORY_MAP = {
    1: 'bottle',
    2: 'bowl',
    3: 'camera',
    4: 'can',
    5: 'laptop',
    6: 'mug'
}


class DatasetReal275(BaseDataset):
    """
    NOCS Real275 real-world test set loader.

    Dataset structure:
        real_test/
            scene_1/
                0000_color.png
                0000_depth.png
                0000_mask.png
                0000_coord.png
                0000_meta.txt  # Lists objects: instance_id category_id object_name
                ...
            scene_2/
                ...
        gts/real_test/
            results_real_test_scene_1_0000.pkl  # Contains gt_RTs array
            ...
        obj_models/real_test/
            bottle_red_stanford_norm.obj
            bottle_red_stanford_norm.txt  # Contains 3D extents
            ...

    Camera intrinsics (fixed for all frames):
        [[591.0125, 0, 322.525],
         [0, 590.16775, 244.11084],
         [0, 0, 1]]
    """

    def __init__(
        self,
        data_dir: str,
        target_size: int = 518,
        num_scenes: int = 6,
        **kwargs
    ):
        """
        Initialize Real275 dataset.

        Args:
            data_dir: Root directory (should contain real_test/, gts/, obj_models/)
            target_size: Target image size for preprocessing
            num_scenes: Number of scenes to load (default: 6)
        """
        super().__init__(data_dir, target_size)

        self.num_scenes = num_scenes
        self.scene_dir = os.path.join(data_dir, 'real_test')
        self.gt_dir = os.path.join(data_dir, 'gts', 'real_test')
        self.mesh_dir = os.path.join(data_dir, 'obj_models', 'real_test')

        # Fixed camera intrinsics for Real275
        self.camera_intrinsics = np.array([
            [591.0125, 0, 322.525],
            [0, 590.16775, 244.11084],
            [0, 0, 1]
        ], dtype=np.float32)

        # Build frame index: (scene_idx, frame_num) for each global frame_idx
        self.frame_index = []
        self.frame_to_scene = {}  # global_idx -> (scene_idx, frame_num)

        print(f"Indexing Real275 dataset from {self.scene_dir}...")

        for scene_idx in range(1, num_scenes + 1):
            scene_path = os.path.join(self.scene_dir, f'scene_{scene_idx}')
            if not os.path.exists(scene_path):
                continue

            # Find all meta files (one per frame)
            meta_files = sorted([
                f for f in os.listdir(scene_path)
                if f.endswith('_meta.txt')
            ])

            for meta_file in meta_files:
                frame_num = int(meta_file.split('_')[0])
                global_idx = len(self.frame_index)
                self.frame_index.append((scene_idx, frame_num))
                self.frame_to_scene[global_idx] = (scene_idx, frame_num)

        print(f"Loaded {len(self.frame_index)} frames from {num_scenes} scenes")

        # Build object index: object_name -> list of frame indices
        self._object_index = None
        self._all_objects = None
        self._gt_poses = None
        self._mesh_diameters = None

    def _build_object_index(self):
        """Build index of which objects appear in which frames."""
        if self._object_index is not None:
            return

        print("Building object index...")
        self._object_index = {}
        self._all_objects = set()

        for global_idx, (scene_idx, frame_num) in enumerate(self.frame_index):
            meta_path = os.path.join(
                self.scene_dir,
                f'scene_{scene_idx}',
                f'{frame_num:04d}_meta.txt'
            )

            if not os.path.exists(meta_path):
                continue

            # Parse meta file
            with open(meta_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    if len(parts) >= 3:
                        object_name = parts[2]
                        self._all_objects.add(object_name)

                        if object_name not in self._object_index:
                            self._object_index[object_name] = []
                        self._object_index[object_name].append(global_idx)

        self._all_objects = sorted(list(self._all_objects))
        print(f"Found {len(self._all_objects)} unique objects")

    def load_frame(
        self,
        frame_idx: int,
        object_name: str,
        mask_cache: Optional[Dict] = None
    ) -> Dict:
        """
        Load frame data for one-shot pose estimation.

        Args:
            frame_idx: Global frame index
            object_name: Object to extract (e.g., 'bowl_white_small_norm')
            mask_cache: Optional cached masks from Grounded SAM2

        Returns:
            Dictionary with preprocessed data
        """
        scene_idx, frame_num = self.frame_to_scene[frame_idx]
        scene_path = os.path.join(self.scene_dir, f'scene_{scene_idx}')

        # Load images
        rgb_path = os.path.join(scene_path, f'{frame_num:04d}_color.png')
        depth_path = os.path.join(scene_path, f'{frame_num:04d}_depth.png')
        mask_path = os.path.join(scene_path, f'{frame_num:04d}_mask.png')
        meta_path = os.path.join(scene_path, f'{frame_num:04d}_meta.txt')

        if not os.path.exists(rgb_path):
            raise FileNotFoundError(f"RGB not found: {rgb_path}")

        # Load RGB and get preprocessing coords
        rgb_raw = Image.open(rgb_path).convert('RGB')
        rgb_tensor, coords = resize_and_pad_image(rgb_raw, self.target_size)
        rgb = rgb_tensor.cpu().numpy().transpose(1, 2, 0)  # (3, H, W) -> (H, W, 3)

        # Parse meta file to find object instance info
        meta_lines = []
        with open(meta_path, 'r') as f:
            meta_lines = [line.strip() for line in f if line.strip()]

        instance_id = None
        pose_idx = None
        for idx, line in enumerate(meta_lines):
            parts = line.split()
            if len(parts) >= 3 and parts[2] == object_name:
                instance_id = int(parts[0])
                pose_idx = idx
                break

        if instance_id is None:
            raise ValueError(f"Object {object_name} not found in frame {frame_idx}")

        # Load GT mask or use cached mask
        if mask_cache and (frame_idx, object_name) in mask_cache:
            # Use pre-generated mask from cache (already preprocessed)
            mask = mask_cache[(frame_idx, object_name)].cpu().numpy()
        else:
            # Extract mask from instance segmentation
            mask_raw = np.array(Image.open(mask_path))
            object_mask = (mask_raw == instance_id).astype(np.uint8) * 255
            mask_tensor = resize_and_pad_mask(object_mask, self.target_size, coords)
            mask = mask_tensor.cpu().numpy()

        # Load and preprocess depth
        depth_raw = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)
        if depth_raw is None:
            raise RuntimeError(f"Failed to load depth: {depth_path}")
        depth = preprocess_depth_map(depth_raw, coords, self.target_size)

        # Load GT pose
        gt_pkl_path = os.path.join(
            self.gt_dir,
            f'results_real_test_scene_{scene_idx}_{frame_num:04d}.pkl'
        )

        with open(gt_pkl_path, 'rb') as f:
            gt_data = pickle.load(f)

        gt_RTs = gt_data['gt_RTs']  # (N_objects, 4, 4)

        if pose_idx is None or pose_idx >= len(gt_RTs):
            raise ValueError(f"Pose not found for {object_name} in frame {frame_idx}")

        RT = gt_RTs[pose_idx]
        R = extract_pure_rotation(RT[:3, :3])  # Clean rotation (remove scale)
        t = RT[:3, 3]

        # Adjust camera intrinsics for resize + padding
        # coords = (paste_x, paste_y, paste_x_end, paste_y_end, orig_w, orig_h)
        paste_x, paste_y, paste_x_end, paste_y_end, orig_w, orig_h = coords

        # Calculate resize scale factor
        resized_w = paste_x_end - paste_x
        resized_h = paste_y_end - paste_y
        scale = resized_w / orig_w  # Should be same as resized_h / orig_h

        # Step 1: Scale intrinsics for resize operation
        K_adjusted = self.camera_intrinsics.copy()
        K_adjusted[0] *= scale  # Scale entire first row (fx, cx)
        K_adjusted[1] *= scale  # Scale entire second row (fy, cy)

        # Step 2: Add padding offset to principal point
        K_adjusted[0, 2] += paste_x  # cx
        K_adjusted[1, 2] += paste_y  # cy

        return {
            'rgb': rgb,
            'mask': mask,
            'depth': depth,
            'K': K_adjusted,
            'pose': {'R': R, 't': t},
            'frame_info': {
                'frame_idx': frame_idx,
                'scene': scene_idx - 1,  # 0-indexed for evaluator compatibility
                'frame_num': frame_num,
                'object_name': object_name
            }
        }

    def get_valid_frames_for_object(self, object_name: str) -> List[int]:
        """Get all frames containing the specified object."""
        self._build_object_index()
        return self._object_index.get(object_name, [])

    def get_all_objects(self, category: Optional[str] = None) -> List[str]:
        """
        Get all object instances.

        Args:
            category: Optional filter by category (e.g., 'bottle', 'mug').
                     If None, returns all objects.

        Returns:
            List of object names (e.g., ['bowl_white_small_norm', ...])
        """
        self._build_object_index()

        if category is None:
            return self._all_objects

        # Filter by category name
        category_lower = category.lower()
        return [
            obj for obj in self._all_objects
            if obj.startswith(category_lower + '_')
        ]

    def supports_gt_masks(self) -> bool:
        """Real275 provides GT instance masks."""
        return True

    def get_frame_info(self, frame_idx: int) -> Tuple[int, int]:
        """Get scene and frame number."""
        return self.frame_to_scene[frame_idx]

    def get_frame_by_scene_and_num(self, scene_idx: int, frame_num: int) -> Optional[int]:
        """
        Get global frame index from scene index and frame number.

        Args:
            scene_idx: Scene index (1-6 for Real275)
            frame_num: Frame number within the scene (0-based)

        Returns:
            Global frame index, or None if not found
        """
        # Search through frame_index to find matching (scene, frame)
        for global_idx, (s_idx, f_num) in enumerate(self.frame_index):
            if s_idx == scene_idx and f_num == frame_num:
                return global_idx
        return None

    def get_gt_poses(self) -> List[Dict]:
        """
        Get ground truth poses for all frames.

        Returns:
            List of dicts with keys: 'model_names', 'rotations', 'translations', 'instance_ids'
        """
        if self._gt_poses is not None:
            return self._gt_poses

        print("Loading ground truth poses...")
        self._gt_poses = []

        for global_idx, (scene_idx, frame_num) in enumerate(self.frame_index):
            meta_path = os.path.join(
                self.scene_dir,
                f'scene_{scene_idx}',
                f'{frame_num:04d}_meta.txt'
            )

            gt_pkl_path = os.path.join(
                self.gt_dir,
                f'results_real_test_scene_{scene_idx}_{frame_num:04d}.pkl'
            )

            # Parse meta file
            model_names = []
            instance_ids = []
            with open(meta_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 3:
                        instance_ids.append(int(parts[0]))
                        model_names.append(parts[2])

            # Load GT poses
            with open(gt_pkl_path, 'rb') as f:
                gt_data = pickle.load(f)

            gt_RTs = gt_data['gt_RTs']  # (N, 4, 4)

            # Extract rotations and clean them (remove scale component)
            # Real275/NOCS datasets store R with both rotation and scale
            rotations = [extract_pure_rotation(RT[:3, :3]) for RT in gt_RTs[:len(model_names)]]
            translations = [RT[:3, 3] for RT in gt_RTs[:len(model_names)]]

            self._gt_poses.append({
                'model_names': model_names,
                'rotations': rotations,
                'translations': translations,
                'instance_ids': instance_ids
            })

        return self._gt_poses

    def get_intrinsics(self) -> np.ndarray:
        """
        Get camera intrinsics for all frames.

        Returns:
            Array of shape (N_frames, 3, 3) - same intrinsics for all frames
        """
        num_frames = len(self.frame_index)
        intrinsics = np.tile(self.camera_intrinsics[None, :, :], (num_frames, 1, 1))
        return intrinsics

    def get_mesh_path(self, object_name: str) -> str:
        """
        Get path to object mesh (.obj file).

        Args:
            object_name: Object name (e.g., 'bowl_white_small_norm')

        Returns:
            Path to .obj file
        """
        mesh_path = os.path.join(self.mesh_dir, f'{object_name}.obj')
        if not os.path.exists(mesh_path):
            raise FileNotFoundError(f"Mesh not found: {mesh_path}")
        return mesh_path

    def get_model_diameter(self, object_name: str) -> float:
        """
        Get model diameter for BOP metrics.

        Diameter is computed from 3D extents in the .txt file.

        Args:
            object_name: Object name

        Returns:
            Diameter in meters
        """
        if self._mesh_diameters is None:
            self._mesh_diameters = {}

        if object_name in self._mesh_diameters:
            return self._mesh_diameters[object_name]

        # Load extents from .txt file
        extent_path = os.path.join(self.mesh_dir, f'{object_name}.txt')
        if not os.path.exists(extent_path):
            raise FileNotFoundError(f"Extent file not found: {extent_path}")

        with open(extent_path, 'r') as f:
            lines = f.readlines()
            extents = [float(line.strip()) for line in lines if line.strip()]

        if len(extents) != 3:
            raise ValueError(f"Invalid extent file: {extent_path}")

        # Diameter = length of 3D diagonal
        diameter = np.sqrt(sum(e**2 for e in extents))
        self._mesh_diameters[object_name] = diameter

        return diameter

    def get_model_symmetries(self, object_name: str) -> Dict:
        """
        Get model symmetry information for Real275 objects.

        Following Oryon/Any6D convention:
        - bottle, bowl, can: continuous Z-rotation symmetry
        - camera, laptop, mug: no symmetries

        NOTE: The original Real275/NOCS dataset does NOT include symmetry annotations.
        Oryon generates symmetries via scripts/data/nocs_bop_models.py which hardcodes:
            if 'bottle' in objname or 'bowl' in objname or 'can' in objname:
                cur_info['symmetries_continuous'] = [{'axis': [0,0,1], 'offset': [0,0,0]}]
        This method implements the same logic directly without requiring
        external file generation.

        Args:
            object_name: Object name (e.g., "bottle_red_stanford_norm")

        Returns:
            Symmetry dict in BOP format
        """
        # Extract category from object name (e.g., "bottle_red_stanford_norm" -> "bottle")
        category = object_name.split('_')[0]

        # Define symmetric categories (continuous Z-rotation)
        if category in ['bottle', 'bowl', 'can']:
            return {
                'symmetries_discrete': [],
                'symmetries_continuous': [{
                    'axis': [0, 0, 1],  # Z-axis rotation
                    'offset': [0, 0, 0]
                }]
            }
        else:
            # No symmetries for camera, laptop, mug
            return {
                'symmetries_discrete': [],
                'symmetries_continuous': []
            }

    def get_symmetry_transformations(
        self,
        object_name: str,
        max_sym_disc_step: float = 0.05
    ) -> list:
        """
        Get symmetry transformations in BOP format.

        Matches Oryon's implementation exactly:
        - Returns list of dicts with 'R' and 't' keys
        - Discretizes continuous symmetries
        - Always includes identity transformation

        Args:
            object_name: Object name
            max_sym_disc_step: Maximum step for discretizing continuous symmetries

        Returns:
            List of symmetry transformations: [{'R': (3,3), 't': (3,1)}, ...]
        """
        import numpy as np

        model_info = self.get_model_symmetries(object_name)

        # Discrete symmetries (includes identity)
        trans_disc = [{'R': np.eye(3), 't': np.zeros((3, 1))}]
        if model_info.get('symmetries_discrete'):
            for sym in model_info['symmetries_discrete']:
                sym_4x4 = np.reshape(sym, (4, 4))
                R = sym_4x4[:3, :3]
                t = sym_4x4[:3, 3].reshape((3, 1))
                trans_disc.append({'R': R, 't': t})

        # Discretized continuous symmetries
        trans_cont = []
        if model_info.get('symmetries_continuous'):
            for sym in model_info['symmetries_continuous']:
                axis = np.array(sym['axis'])
                offset = np.array(sym['offset']).reshape((3, 1))

                # Discretize rotation around axis
                discrete_steps_count = int(np.ceil(np.pi / max_sym_disc_step))
                discrete_step = 2.0 * np.pi / discrete_steps_count

                for i in range(discrete_steps_count):
                    angle = i * discrete_step
                    # Rodrigues' rotation formula
                    K = np.array([
                        [0, -axis[2], axis[1]],
                        [axis[2], 0, -axis[0]],
                        [-axis[1], axis[0], 0]
                    ])
                    R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
                    t = -R @ offset + offset
                    trans_cont.append({'R': R, 't': t})

        # Combine discrete and continuous symmetries
        trans = []
        for tran_disc in trans_disc:
            if len(trans_cont):
                for tran_cont in trans_cont:
                    R = tran_cont['R'] @ tran_disc['R']
                    t = tran_cont['R'] @ tran_disc['t'] + tran_cont['t']
                    trans.append({'R': R, 't': t})
            else:
                trans.append(tran_disc)

        return trans
