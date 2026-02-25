"""
DatasetYCBV: YCB-Video BOP test set loader.

Loads the YCB-Video test set in BOP19 format with 92 scenes and 21 objects.
YCB-Video contains everyday objects captured in cluttered tabletop scenarios.
"""

import os
import json
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


class DatasetYCBV(BaseDataset):
    """
    YCB-Video BOP test set loader.

    Dataset structure:
        concept-pose/data/ycbv/
            test/
                000048/
                    rgb/000000.png
                    depth/000000.png  # PNG format, 0.1mm scale (uint16, 0.1 millimeters)
                    mask_visib/000000_000000.png
                    scene_gt.json  # BOP format GT poses
                    scene_gt_info.json  # BOP format metadata
                    scene_camera.json  # Camera intrinsics
                ...
                000059/
                    ...
            models/
                models/
                    obj_000001.ply
                    ...
                    obj_000021.ply
                    models_info.json  # Contains diameters and symmetries

    Camera intrinsics:
        Per-frame calibration loaded from scene_camera.json (BOP format)

    Key differences from TYOL:
        - Depth scale: 10000.0 (0.1mm units, NOT 1000.0)
        - 21 unique objects (not categories with instances)
        - Object names: category_id format (e.g., "can_1", "scissors_17")
        - Category mapping: Loaded from data/ycbv/ycbv_object_categories.txt
        - BOP names stored internally for reference (e.g., "002_master_chef_can")
    """

    def __init__(
        self,
        bop_root: str,
        category_mapping_path: str,
        target_size: int = 518,
        num_scenes: int = 92,
        depth_scale: float = 10000.0,
        **kwargs
    ):
        """
        Initialize YCB-Video dataset.

        Args:
            bop_root: Root directory for BOP data (contains ycbv/, ycbv_models/)
            category_mapping_path: Path to ycbv_object_categories.txt
            target_size: Target image size for preprocessing
            num_scenes: Number of scenes to load (default: 92, but test set has 12 scenes)
            depth_scale: Scale factor to convert depth to meters (default: 10000.0 for 0.1mm->m)
        """
        super().__init__(data_dir=bop_root, target_size=target_size)

        self.bop_root = bop_root
        self.num_scenes = num_scenes
        self.depth_scale = depth_scale
        self.test_dir = os.path.join(bop_root, 'test')
        self.models_dir = os.path.join(bop_root, 'models', 'models')

        # Load category mapping (obj_id -> category_name for partonomy)
        self._obj_id_to_names = {}  # {obj_id: {'object_name': ..., 'category_name': ...}}
        with open(category_mapping_path, 'r') as f:
            lines = f.readlines()[1:]  # Skip header
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                parts = [p.strip() for p in line.split('|')]
                if len(parts) >= 3:
                    obj_id = int(parts[0])
                    object_name = parts[1]
                    category_name = parts[2]
                    self._obj_id_to_names[obj_id] = {
                        'object_name': object_name,
                        'category_name': category_name
                    }

        # Load models info (diameters, symmetries)
        models_info_path = os.path.join(self.models_dir, 'models_info.json')
        with open(models_info_path, 'r') as f:
            self._models_info = json.load(f)

        # Build frame index: (scene_idx, frame_num) for each global frame_idx
        self.frame_index = []
        self.frame_to_scene = {}  # global_idx -> (scene_idx, frame_num)

        print(f"Indexing YCB-Video dataset from {self.test_dir}...")

        # Load all scenes and their GT poses
        self._scene_data = {}  # scene_idx -> {frame_idx: [{'obj_id': ..., 'R': ..., 't': ...}, ...]}
        self._scene_cameras = {}  # scene_idx -> {frame_idx: {'cam_K': ..., 'depth_scale': ...}}

        # YCB-Video test scenes: 48-59 (BOP convention)
        test_scene_ids = list(range(48, 60))  # 48, 49, ..., 59

        for scene_idx in test_scene_ids:
            scene_path = os.path.join(self.test_dir, f'{scene_idx:06d}')
            if not os.path.exists(scene_path):
                continue

            # Load scene_gt.json
            scene_gt_path = os.path.join(scene_path, 'scene_gt.json')
            if not os.path.exists(scene_gt_path):
                continue
            with open(scene_gt_path, 'r') as f:
                scene_gt = json.load(f)

            # Load scene_camera.json for per-frame intrinsics
            scene_camera_path = os.path.join(scene_path, 'scene_camera.json')
            with open(scene_camera_path, 'r') as f:
                scene_camera = json.load(f)

            # Parse GT data and camera data
            scene_data = {}
            scene_cam_data = {}
            for frame_str, objects in scene_gt.items():
                frame_num = int(frame_str)
                scene_data[frame_num] = []

                for obj_data in objects:
                    # BOP format: cam_R_m2c is flattened 3x3 rotation, cam_t_m2c is in mm
                    R_flat = np.array(obj_data['cam_R_m2c'], dtype=np.float32)
                    R = R_flat.reshape(3, 3)
                    t = np.array(obj_data['cam_t_m2c'], dtype=np.float32) / 1000.0  # mm to meters
                    obj_id = int(obj_data['obj_id'])

                    scene_data[frame_num].append({
                        'obj_id': obj_id,
                        'R': R,
                        't': t
                    })

                # Load camera intrinsics for this frame
                if frame_str in scene_camera:
                    cam_K_flat = scene_camera[frame_str]['cam_K']
                    cam_K = np.array(cam_K_flat, dtype=np.float32).reshape(3, 3)
                    scene_cam_data[frame_num] = {
                        'cam_K': cam_K,
                        'depth_scale': scene_camera[frame_str].get('depth_scale', 1.0)
                    }

                # Add to global frame index
                global_idx = len(self.frame_index)
                self.frame_index.append((scene_idx, frame_num))
                self.frame_to_scene[global_idx] = (scene_idx, frame_num)

            self._scene_data[scene_idx] = scene_data
            self._scene_cameras[scene_idx] = scene_cam_data

        print(f"Loaded {len(self.frame_index)} frames from {len(test_scene_ids)} test scenes")

        # Build object index: object_id -> list of frame indices
        self._object_index = None
        self._all_objects = None

    def _build_object_index(self):
        """Build index of which objects appear in which frames."""
        if self._object_index is not None:
            return

        print("Building object index...")
        self._object_index = {}
        self._all_objects = set()

        for global_idx, (scene_idx, frame_num) in enumerate(self.frame_index):
            scene_data = self._scene_data.get(scene_idx, {})
            if frame_num not in scene_data:
                continue

            for obj_data in scene_data[frame_num]:
                obj_id = obj_data['obj_id']
                self._all_objects.add(obj_id)

                if obj_id not in self._object_index:
                    self._object_index[obj_id] = []
                self._object_index[obj_id].append(global_idx)

        self._all_objects = sorted(list(self._all_objects))
        print(f"Found {len(self._all_objects)} unique objects")

    def _get_object_name(self, obj_id: int) -> str:
        """
        Get canonical object name from object ID.

        Returns name in format: <category_name>_<obj_id>
        This matches the expected format for partonomy pipeline.
        Example: "can_1", "scissors_17", "bottle_5"

        Args:
            obj_id: Object ID (1-21)

        Returns:
            Object name in category_id format (e.g., 'can_1', 'scissors_17')
        """
        if obj_id in self._obj_id_to_names:
            category = self._obj_id_to_names[obj_id]['category_name']
            return f"{category}_{obj_id}"
        else:
            # Fallback to generic name
            return f"obj_{obj_id}"

    def _get_bop_name(self, obj_id: int) -> str:
        """
        Get full BOP object name from object ID.

        Args:
            obj_id: Object ID (1-21)

        Returns:
            Full BOP name (e.g., '002_master_chef_can', '037_scissors')
        """
        if obj_id in self._obj_id_to_names:
            return self._obj_id_to_names[obj_id]['object_name']
        else:
            return f"obj_{obj_id:06d}"

    def _get_category_name(self, obj_id: int) -> str:
        """
        Get category name for partonomy/LLM pipeline.

        Args:
            obj_id: Object ID (1-21)

        Returns:
            Category name (e.g., 'can', 'bottle', 'box')
        """
        if obj_id in self._obj_id_to_names:
            return self._obj_id_to_names[obj_id]['category_name']
        else:
            return "unknown"

    def _get_object_id(self, object_name: str) -> Optional[int]:
        """
        Get object ID from object name.

        Supports multiple formats:
        - "can_1" (category_id format)
        - "002_master_chef_can" (full BOP name)
        - "can" (category name only)

        Args:
            object_name: Object name in any supported format

        Returns:
            Object ID or None if not found
        """
        # Try parsing category_id format (e.g., "can_1")
        if '_' in object_name:
            try:
                obj_id = int(object_name.split('_')[-1])
                if obj_id in self._obj_id_to_names:
                    return obj_id
            except ValueError:
                pass

        # Try exact match on full BOP name (e.g., "002_master_chef_can")
        for obj_id, names in self._obj_id_to_names.items():
            if names['object_name'] == object_name:
                return obj_id

        # Try match on category name only (e.g., "can")
        for obj_id, names in self._obj_id_to_names.items():
            if names['category_name'] == object_name:
                return obj_id

        return None

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
            object_name: Object to extract (e.g., 'can_1', 'scissors_17', or 'can')
            mask_cache: Optional cached masks from Grounded SAM2

        Returns:
            Dictionary with preprocessed data:
                - 'rgb': (H, W, 3) float32 [0-1], resized and padded
                - 'mask': (H, W) float32 [0-1], resized and padded
                - 'depth': (H, W) float32 meters, resized and padded
                - 'K': (3, 3) camera intrinsics (adjusted for padding)
                - 'pose': dict with 'R' (3, 3) and 't' (3,)
                - 'frame_info': metadata dict (includes bop_name for reference)
        """
        scene_idx, frame_num = self.frame_to_scene[frame_idx]
        scene_path = os.path.join(self.test_dir, f'{scene_idx:06d}')

        # Convert object name to ID
        obj_id = self._get_object_id(object_name)
        if obj_id is None:
            raise ValueError(f"Unknown object name: {object_name}")

        # Load RGB (keep as PIL Image for preprocessing)
        rgb_path = os.path.join(scene_path, 'rgb', f'{frame_num:06d}.png')
        rgb_raw = Image.open(rgb_path).convert('RGB')

        # Load depth (BOP format: PNG with 0.1mm scale - CRITICAL: use 10000.0)
        depth_path = os.path.join(scene_path, 'depth', f'{frame_num:06d}.png')
        depth_raw = np.array(Image.open(depth_path)).astype(np.float32)

        # Load mask
        if mask_cache is not None and (frame_idx, object_name) in mask_cache:
            # Use cached mask from Grounded SAM2
            mask = mask_cache[(frame_idx, object_name)]
        else:
            # Use GT mask
            # BOP format: mask_visib/{frame:06d}_{instance:06d}.png
            # Find the instance index for this object in GT data
            scene_data = self._scene_data[scene_idx][frame_num]
            instance_idx = None
            for idx, obj_data in enumerate(scene_data):
                if obj_data['obj_id'] == obj_id:
                    instance_idx = idx
                    break

            if instance_idx is None:
                raise ValueError(
                    f"Object {object_name} (id={obj_id}) not found in frame {frame_idx}"
                )

            # BOP mask filename: {frame:06d}_{instance:06d}.png
            mask_path = os.path.join(scene_path, 'mask_visib', f'{frame_num:06d}_{instance_idx:06d}.png')

            if not os.path.exists(mask_path):
                raise FileNotFoundError(
                    f"Mask file not found: {mask_path}\n"
                    f"  Expected for object {object_name} (id={obj_id}), instance {instance_idx}"
                )

            # Load mask (BOP masks are binary: 255 for object, 0 for background)
            mask = np.array(Image.open(mask_path))

        # Get GT pose for this object
        scene_data = self._scene_data[scene_idx][frame_num]
        pose = None
        for obj_data in scene_data:
            if obj_data['obj_id'] == obj_id:
                pose = {
                    'R': obj_data['R'].copy(),
                    't': obj_data['t'].copy()
                }
                break

        if pose is None:
            raise ValueError(
                f"Pose not found for object {object_name} (id={obj_id}) in frame {frame_idx}"
            )

        # Store original size for metadata
        orig_w, orig_h = rgb_raw.size

        # Preprocess: resize and pad
        rgb_tensor, coords = resize_and_pad_image(rgb_raw, self.target_size)
        rgb_processed = rgb_tensor.cpu().numpy().transpose(1, 2, 0)  # (3, H, W) -> (H, W, 3)

        # Preprocess mask and depth using same coords
        mask_tensor = resize_and_pad_mask(mask, self.target_size, coords)
        mask_processed = mask_tensor.cpu().numpy()
        depth_processed = preprocess_depth_map(depth_raw, coords, self.target_size, self.depth_scale)

        # Extract pad info from coords
        paste_x, paste_y, paste_x_end, paste_y_end, _, _ = coords
        scale = (paste_x_end - paste_x) / orig_w
        pad_info = {
            'scale': scale,
            'pad_left': paste_x,
            'pad_top': paste_y,
            'pad_right': self.target_size - paste_x_end,
            'pad_bottom': self.target_size - paste_y_end
        }

        # Get per-frame camera intrinsics from BOP data
        cam_K_original = self._scene_cameras[scene_idx][frame_num]['cam_K'].copy()

        # Adjust camera intrinsics for padding
        K_adjusted = cam_K_original.copy()
        K_adjusted[0, 0] *= pad_info['scale']  # fx
        K_adjusted[1, 1] *= pad_info['scale']  # fy
        K_adjusted[0, 2] = K_adjusted[0, 2] * pad_info['scale'] + pad_info['pad_left']  # cx
        K_adjusted[1, 2] = K_adjusted[1, 2] * pad_info['scale'] + pad_info['pad_top']   # cy

        # Metadata (scene_idx for YCB-V is 48-59)
        frame_info = {
            'scene': scene_idx,  # Keep original scene ID (48-59)
            'scene_idx': scene_idx,
            'frame': frame_num,
            'frame_num': frame_num,
            'global_idx': frame_idx,
            'object_name': self._get_object_name(obj_id),  # category_id format (e.g., "can_1")
            'object_id': obj_id,
            'category_name': self._get_category_name(obj_id),
            'bop_name': self._get_bop_name(obj_id),  # Full BOP name for reference
            'original_size': (orig_h, orig_w),
            'processed_size': rgb_processed.shape[:2],
            'pad_info': pad_info,
            'coords': coords  # Required by CategoryLevelEvaluator for saliency resizing
        }

        return {
            'rgb': rgb_processed,
            'mask': mask_processed,
            'depth': depth_processed,
            'K': K_adjusted,
            'pose': pose,
            'frame_info': frame_info
        }

    def get_valid_frames_for_object(self, object_name: str) -> List[int]:
        """
        Find all frames containing a specific object.

        Args:
            object_name: Object name (e.g., 'can_1', 'scissors_17', or 'can')

        Returns:
            List of frame indices where object appears
        """
        self._build_object_index()

        obj_id = self._get_object_id(object_name)
        if obj_id is None:
            return []

        return self._object_index.get(obj_id, [])

    def get_all_objects(self, category: Optional[str] = None) -> List[str]:
        """
        Get all unique objects in the dataset.

        Args:
            category: Optional category filter (e.g., 'can', 'bottle')

        Returns:
            List of object names in category_id format (e.g., ['can_1', 'can_4', ...])
        """
        self._build_object_index()

        if category is None:
            # Return all object names
            return [self._get_object_name(obj_id) for obj_id in self._all_objects]
        else:
            # Filter by category
            filtered = []
            for obj_id in self._all_objects:
                if self._get_category_name(obj_id) == category:
                    filtered.append(self._get_object_name(obj_id))
            return filtered

    def supports_gt_masks(self) -> bool:
        """
        Check if dataset provides ground truth instance masks.

        Returns:
            True (YCB-V provides GT masks)
        """
        return True

    def get_frame_info(self, frame_idx: int) -> Tuple[int, int]:
        """
        Get scene and frame metadata.

        Args:
            frame_idx: Global frame index

        Returns:
            Tuple of (scene_idx, frame_in_scene)
        """
        return self.frame_to_scene[frame_idx]

    def get_frame_by_scene_and_num(self, scene_idx: int, frame_num: int) -> Optional[int]:
        """
        Get global frame index from scene index and frame number.

        Args:
            scene_idx: Scene index (48-59 for test set)
            frame_num: Frame number within scene

        Returns:
            Global frame index or None if not found
        """
        for global_idx, (s_idx, f_num) in enumerate(self.frame_index):
            if s_idx == scene_idx and f_num == frame_num:
                return global_idx
        return None

    def get_gt_poses(self) -> List[Dict]:
        """
        Get ground truth poses for all frames.

        Returns:
            List of dicts, one per frame
        """
        gt_poses = []

        for global_idx, (scene_idx, frame_num) in enumerate(self.frame_index):
            scene_data = self._scene_data.get(scene_idx, {}).get(frame_num, [])

            # Collect all objects in this frame
            model_names = []
            rotations = []
            translations = []
            instance_ids = []

            for obj_data in scene_data:
                obj_id = obj_data['obj_id']
                model_names.append(self._get_object_name(obj_id))
                rotations.append(obj_data['R'])
                translations.append(obj_data['t'])
                instance_ids.append(obj_id)

            gt_poses.append({
                'model_names': model_names,
                'rotations': rotations,
                'translations': translations,
                'instance_ids': instance_ids
            })

        return gt_poses

    def get_intrinsics(self) -> np.ndarray:
        """
        Get camera intrinsics for all frames.

        Returns:
            Array of shape (N_frames, 3, 3) with camera matrices
        """
        num_frames = len(self.frame_index)
        intrinsics = []

        for frame_idx in range(num_frames):
            scene_idx, frame_num = self.frame_to_scene[frame_idx]
            cam_K = self._scene_cameras[scene_idx][frame_num]['cam_K']
            intrinsics.append(cam_K)

        return np.array(intrinsics)

    def get_mesh_path(self, object_name: str) -> str:
        """
        Get path to 3D mesh file for object.

        Args:
            object_name: Object name

        Returns:
            Path to .ply mesh file
        """
        obj_id = self._get_object_id(object_name)
        if obj_id is None:
            raise ValueError(f"Unknown object name: {object_name}")

        mesh_path = os.path.join(self.models_dir, f'obj_{obj_id:06d}.ply')
        if not os.path.exists(mesh_path):
            raise FileNotFoundError(f"Mesh not found: {mesh_path}")

        return mesh_path

    def get_model_diameter(self, object_name: str) -> float:
        """
        Get model diameter for metric computation.

        Args:
            object_name: Object name

        Returns:
            Diameter in meters
        """
        obj_id = self._get_object_id(object_name)
        if obj_id is None:
            raise ValueError(f"Unknown object name: {object_name}")

        # BOP models_info.json stores diameter in mm
        diameter_mm = self._models_info[str(obj_id)]['diameter']
        return diameter_mm / 1000.0  # Convert to meters

    def get_model_symmetries(self, object_name: str) -> Dict:
        """
        Get model symmetry information.

        Args:
            object_name: Object name

        Returns:
            Symmetry dict from BOP models_info
        """
        obj_id = self._get_object_id(object_name)
        if obj_id is None:
            raise ValueError(f"Unknown object name: {object_name}")

        model_info = self._models_info[str(obj_id)]

        # Extract symmetry info (BOP format)
        return {
            'symmetries_discrete': model_info.get('symmetries_discrete', []),
            'symmetries_continuous': model_info.get('symmetries_continuous', [])
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
