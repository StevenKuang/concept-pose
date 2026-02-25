"""
DatasetLMO: LINEMOD-Occlusion (LM-O) BOP test set loader.

LM-O is a variant of LINEMOD with heavy occlusion in a single test scene.
This dataset is specifically designed for occlusion vs performance studies.

Key differences from DatasetLM:
- Single test scene (scene 2) with heavy occlusion
- 8 objects instead of 15
- Includes occlusion metadata (visib_fract) from scene_gt_info.json
- Scans for existing scene directories instead of assuming 1-N
- Stores original instance_idx for correct mask loading
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


class DatasetLMO(BaseDataset):
    """
    LINEMOD-Occlusion BOP test set loader.

    Dataset structure:
        concept-pose/data/lmo/
            test/
                000002/  # Only scene 2 exists in LM-O
                    rgb/000000.png
                    depth/000000.png
                    mask_visib/000000_000000.png
                    scene_gt.json
                    scene_gt_info.json  # Contains visib_fract for occlusion
                    scene_camera.json
            lmo_models/
                models/
                    obj_000001.ply
                    ...
                    models_info.json
            lmo_base/
                test_targets_bop19.json

    Key features:
        - Single test scene with heavy occlusion
        - 8 objects (ape, can, cat, driller, duck, eggbox, glue, holepuncher)
        - Occlusion metadata (visib_fract) included in frame_info
        - Correct instance_idx tracking for mask loading
    """

    # BOP object names for LM-O (8 objects)
    OBJECT_NAMES = {
        1: "ape",
        5: "can",
        6: "cat",
        8: "driller",
        9: "duck",
        10: "eggbox",
        11: "glue",
        12: "holepuncher"
    }

    def __init__(
        self,
        bop_root: str,
        target_size: int = 518,
        depth_scale: float = 1000.0,
        use_bop_targets: bool = True,
        models_subdir: str = 'lmo_models/models',
        bop_targets_subdir: str = 'lmo_base',
        **kwargs
    ):
        """
        Initialize LM-O dataset.

        Args:
            bop_root: Root directory for BOP data (contains test/, lmo_models/, lmo_base/)
            target_size: Target image size for preprocessing
            depth_scale: Scale factor to convert depth to meters (default: 1000.0 for mm->m)
            use_bop_targets: If True, only load frames in BOP test targets (default: True)
            models_subdir: Subdirectory for models (default: 'lmo_models/models')
            bop_targets_subdir: Subdirectory for BOP targets (default: 'lmo_base')
        """
        super().__init__(data_dir=bop_root, target_size=target_size)

        self.bop_root = bop_root
        self.depth_scale = depth_scale
        self.use_bop_targets = use_bop_targets
        self.test_dir = os.path.join(bop_root, 'test')
        self.models_dir = os.path.join(bop_root, models_subdir)
        self.bop_targets_subdir = bop_targets_subdir

        # Load models info (diameters, symmetries)
        models_info_path = os.path.join(self.models_dir, 'models_info.json')
        with open(models_info_path, 'r') as f:
            self._models_info = json.load(f)

        # Load BOP test targets
        self._bop_targets = None
        if use_bop_targets:
            bop_targets_path = os.path.join(bop_root, bop_targets_subdir, 'test_targets_bop19.json')
            if os.path.exists(bop_targets_path):
                with open(bop_targets_path, 'r') as f:
                    targets = json.load(f)
                # Build set for fast lookup: (scene_id, im_id, obj_id)
                self._bop_targets = set()
                for target in targets:
                    self._bop_targets.add((
                        target['scene_id'],
                        target['im_id'],
                        target['obj_id']
                    ))
                print(f"Loaded {len(targets)} BOP test targets")

        # Build frame index
        self.frame_index = []
        self.frame_to_scene = {}

        print(f"Indexing LM-O dataset from {self.test_dir}...")

        # Load scene data and cameras
        self._scene_data = {}  # scene_idx -> {frame_idx: [{'obj_id': ..., 'R': ..., 't': ..., 'instance_idx': ...}, ...]}
        self._scene_cameras = {}
        self._scene_gt_info = {}  # scene_idx -> {frame_str: [{visib_fract: ..., ...}, ...]}

        # Scan for existing scene directories (LM-O only has scene 2)
        existing_scenes = []
        for scene_dir in sorted(os.listdir(self.test_dir)):
            scene_path = os.path.join(self.test_dir, scene_dir)
            if os.path.isdir(scene_path):
                try:
                    scene_idx = int(scene_dir)
                    existing_scenes.append(scene_idx)
                except ValueError:
                    continue

        print(f"Found scenes: {existing_scenes}")

        for scene_idx in existing_scenes:
            scene_path = os.path.join(self.test_dir, f'{scene_idx:06d}')

            # Load scene_gt.json
            scene_gt_path = os.path.join(scene_path, 'scene_gt.json')
            if not os.path.exists(scene_gt_path):
                continue
            with open(scene_gt_path, 'r') as f:
                scene_gt = json.load(f)

            # Load scene_gt_info.json for occlusion metadata
            scene_gt_info_path = os.path.join(scene_path, 'scene_gt_info.json')
            if os.path.exists(scene_gt_info_path):
                with open(scene_gt_info_path, 'r') as f:
                    self._scene_gt_info[scene_idx] = json.load(f)

            # Load scene_camera.json
            scene_camera_path = os.path.join(scene_path, 'scene_camera.json')
            with open(scene_camera_path, 'r') as f:
                scene_camera = json.load(f)

            # Get fixed camera intrinsics for this scene
            first_frame_str = list(scene_camera.keys())[0]
            cam_K_flat = scene_camera[first_frame_str]['cam_K']
            cam_K = np.array(cam_K_flat, dtype=np.float32).reshape(3, 3)
            depth_scale_val = scene_camera[first_frame_str].get('depth_scale', 1.0)

            self._scene_cameras[scene_idx] = {
                'cam_K': cam_K,
                'depth_scale': depth_scale_val
            }

            # Parse GT data with original instance_idx tracking
            scene_data = {}
            for frame_str, objects in scene_gt.items():
                frame_num = int(frame_str)
                scene_data[frame_num] = []

                for original_idx, obj_data in enumerate(objects):
                    # BOP format: cam_R_m2c is flattened 3x3 rotation, cam_t_m2c is in mm
                    R_flat = np.array(obj_data['cam_R_m2c'], dtype=np.float32)
                    R = R_flat.reshape(3, 3)
                    t = np.array(obj_data['cam_t_m2c'], dtype=np.float32) / 1000.0
                    obj_id = int(obj_data['obj_id'])

                    # Filter by BOP targets if enabled
                    if self._bop_targets is not None:
                        if (scene_idx, frame_num, obj_id) not in self._bop_targets:
                            continue

                    # Store original instance_idx for correct mask loading
                    scene_data[frame_num].append({
                        'obj_id': obj_id,
                        'R': R,
                        't': t,
                        'instance_idx': original_idx  # Original index in scene_gt.json
                    })

                # Only add frame if it has valid objects
                if scene_data[frame_num]:
                    global_idx = len(self.frame_index)
                    self.frame_index.append((scene_idx, frame_num))
                    self.frame_to_scene[global_idx] = (scene_idx, frame_num)

            self._scene_data[scene_idx] = scene_data

        print(f"Loaded {len(self.frame_index)} frames from {len(existing_scenes)} scenes")

        # Build object index
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
        """Get canonical object name from object ID."""
        return f"obj_{obj_id}"

    def _get_bop_name(self, obj_id: int) -> str:
        """Get full BOP object name from object ID."""
        return self.OBJECT_NAMES.get(obj_id, f"obj_{obj_id}")

    def _get_object_id(self, object_name: str) -> Optional[int]:
        """
        Get object ID from object name.

        Supports multiple formats:
        - "obj_1" (standard format)
        - "ape" (BOP name)
        - "1" (raw ID)
        """
        # Try parsing obj_id format
        if object_name.startswith('obj_'):
            try:
                obj_id = int(object_name.split('_')[1])
                if obj_id in self.OBJECT_NAMES:
                    return obj_id
            except (ValueError, IndexError):
                pass

        # Try parsing as raw integer
        try:
            obj_id = int(object_name)
            if obj_id in self.OBJECT_NAMES:
                return obj_id
        except ValueError:
            pass

        # Try matching BOP name
        for obj_id, bop_name in self.OBJECT_NAMES.items():
            if bop_name == object_name:
                return obj_id

        return None

    def get_occlusion_info(self, scene_idx: int, frame_num: int, instance_idx: int) -> Dict:
        """
        Get occlusion metadata for a specific object instance.

        Args:
            scene_idx: Scene index
            frame_num: Frame number
            instance_idx: Original instance index in scene_gt.json

        Returns:
            Dict with visib_fract, px_count_visib, px_count_all, etc.
        """
        if scene_idx not in self._scene_gt_info:
            return {}

        frame_str = str(frame_num)
        if frame_str not in self._scene_gt_info[scene_idx]:
            return {}

        instances = self._scene_gt_info[scene_idx][frame_str]
        if instance_idx >= len(instances):
            return {}

        return instances[instance_idx]

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
            object_name: Object to extract (e.g., 'obj_1', 'obj_5', or 'ape')
            mask_cache: Optional cached masks from Grounded SAM2

        Returns:
            Dictionary with preprocessed data including occlusion metadata:
                - 'rgb': (H, W, 3) float32 [0-1]
                - 'mask': (H, W) float32 [0-1]
                - 'depth': (H, W) float32 meters
                - 'K': (3, 3) camera intrinsics
                - 'pose': dict with 'R' (3, 3) and 't' (3,)
                - 'frame_info': metadata dict with visib_fract
        """
        scene_idx, frame_num = self.frame_to_scene[frame_idx]
        scene_path = os.path.join(self.test_dir, f'{scene_idx:06d}')

        # Convert object name to ID
        obj_id = self._get_object_id(object_name)
        if obj_id is None:
            raise ValueError(f"Unknown object name: {object_name}")

        # Load RGB
        rgb_path = os.path.join(scene_path, 'rgb', f'{frame_num:06d}.png')
        rgb_raw = Image.open(rgb_path).convert('RGB')

        # Load depth
        depth_path = os.path.join(scene_path, 'depth', f'{frame_num:06d}.png')
        depth_raw = np.array(Image.open(depth_path)).astype(np.float32)

        # Find object data with stored instance_idx
        scene_data = self._scene_data[scene_idx][frame_num]
        obj_data = None
        for data in scene_data:
            if data['obj_id'] == obj_id:
                obj_data = data
                break

        if obj_data is None:
            raise ValueError(
                f"Object {object_name} (id={obj_id}) not found in frame {frame_idx}"
            )

        # Use stored instance_idx for correct mask loading
        instance_idx = obj_data['instance_idx']

        # Load mask
        if mask_cache is not None and (frame_idx, object_name) in mask_cache:
            mask = mask_cache[(frame_idx, object_name)]
        else:
            # BOP mask filename: {frame:06d}_{instance:06d}.png
            mask_path = os.path.join(scene_path, 'mask_visib', f'{frame_num:06d}_{instance_idx:06d}.png')

            if not os.path.exists(mask_path):
                raise FileNotFoundError(
                    f"Mask file not found: {mask_path}\n"
                    f"  Expected for object {object_name} (id={obj_id}), instance {instance_idx}"
                )

            mask = np.array(Image.open(mask_path))

        # Get GT pose
        pose = {
            'R': obj_data['R'].copy(),
            't': obj_data['t'].copy()
        }

        # Get occlusion metadata
        occlusion_info = self.get_occlusion_info(scene_idx, frame_num, instance_idx)

        # Store original size
        orig_w, orig_h = rgb_raw.size

        # Preprocess: resize and pad
        rgb_tensor, coords = resize_and_pad_image(rgb_raw, self.target_size)
        rgb_processed = rgb_tensor.cpu().numpy().transpose(1, 2, 0)

        # Preprocess mask and depth
        mask_tensor = resize_and_pad_mask(mask, self.target_size, coords)
        mask_processed = mask_tensor.cpu().numpy()
        depth_processed = preprocess_depth_map(depth_raw, coords, self.target_size, self.depth_scale)

        # Extract pad info
        paste_x, paste_y, paste_x_end, paste_y_end, _, _ = coords
        scale = (paste_x_end - paste_x) / orig_w
        pad_info = {
            'scale': scale,
            'pad_left': paste_x,
            'pad_top': paste_y,
            'pad_right': self.target_size - paste_x_end,
            'pad_bottom': self.target_size - paste_y_end
        }

        # Get camera intrinsics
        cam_K_original = self._scene_cameras[scene_idx]['cam_K'].copy()

        # Adjust camera intrinsics for padding
        K_adjusted = cam_K_original.copy()
        K_adjusted[0, 0] *= pad_info['scale']
        K_adjusted[1, 1] *= pad_info['scale']
        K_adjusted[0, 2] = K_adjusted[0, 2] * pad_info['scale'] + pad_info['pad_left']
        K_adjusted[1, 2] = K_adjusted[1, 2] * pad_info['scale'] + pad_info['pad_top']

        # Metadata with occlusion info
        frame_info = {
            'scene': scene_idx,
            'scene_idx': scene_idx,
            'frame': frame_num,
            'frame_num': frame_num,
            'global_idx': frame_idx,
            'object_name': self._get_object_name(obj_id),
            'object_id': obj_id,
            'bop_name': self._get_bop_name(obj_id),
            'instance_idx': instance_idx,  # Original instance index
            'original_size': (orig_h, orig_w),
            'processed_size': rgb_processed.shape[:2],
            'pad_info': pad_info,
            'coords': coords,
            # Occlusion metadata
            'visib_fract': occlusion_info.get('visib_fract', 1.0),
            'px_count_visib': occlusion_info.get('px_count_visib'),
            'px_count_all': occlusion_info.get('px_count_all'),
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
        """Find all frames containing a specific object."""
        self._build_object_index()

        obj_id = self._get_object_id(object_name)
        if obj_id is None:
            return []

        return self._object_index.get(obj_id, [])

    def get_all_objects(self, category: Optional[str] = None) -> List[str]:
        """Get all unique objects in the dataset."""
        self._build_object_index()
        return [self._get_object_name(obj_id) for obj_id in self._all_objects]

    def get_frames_with_occlusion(self, object_name: str) -> List[Tuple[int, float]]:
        """
        Get all frames for an object with their occlusion levels.

        Args:
            object_name: Object name

        Returns:
            List of (frame_idx, visib_fract) tuples, sorted by visib_fract descending
        """
        self._build_object_index()

        obj_id = self._get_object_id(object_name)
        if obj_id is None:
            return []

        frames_with_occlusion = []
        for frame_idx in self._object_index.get(obj_id, []):
            scene_idx, frame_num = self.frame_to_scene[frame_idx]
            scene_data = self._scene_data[scene_idx][frame_num]

            # Find this object's instance_idx
            for obj_data in scene_data:
                if obj_data['obj_id'] == obj_id:
                    instance_idx = obj_data['instance_idx']
                    occlusion_info = self.get_occlusion_info(scene_idx, frame_num, instance_idx)
                    visib_fract = occlusion_info.get('visib_fract', 1.0)
                    frames_with_occlusion.append((frame_idx, visib_fract))
                    break

        # Sort by visibility (highest first = least occluded first)
        frames_with_occlusion.sort(key=lambda x: -x[1])
        return frames_with_occlusion

    def supports_gt_masks(self) -> bool:
        """Check if dataset provides ground truth instance masks."""
        return True

    def get_frame_info(self, frame_idx: int) -> Tuple[int, int]:
        """Get scene and frame metadata."""
        return self.frame_to_scene[frame_idx]

    def get_frame_by_scene_and_num(self, scene_idx: int, frame_num: int) -> Optional[int]:
        """Get global frame index from scene index and frame number."""
        for global_idx, (s_idx, f_num) in enumerate(self.frame_index):
            if s_idx == scene_idx and f_num == frame_num:
                return global_idx
        return None

    def get_gt_poses(self) -> List[Dict]:
        """Get ground truth poses for all frames."""
        gt_poses = []

        for global_idx, (scene_idx, frame_num) in enumerate(self.frame_index):
            scene_data = self._scene_data.get(scene_idx, {}).get(frame_num, [])

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
        """Get camera intrinsics for all frames."""
        intrinsics = []

        for frame_idx in range(len(self.frame_index)):
            scene_idx, frame_num = self.frame_to_scene[frame_idx]
            cam_K = self._scene_cameras[scene_idx]['cam_K']
            intrinsics.append(cam_K)

        return np.array(intrinsics)

    def get_mesh_path(self, object_name: str) -> str:
        """Get path to 3D mesh file for object."""
        obj_id = self._get_object_id(object_name)
        if obj_id is None:
            raise ValueError(f"Unknown object name: {object_name}")

        mesh_path = os.path.join(self.models_dir, f'obj_{obj_id:06d}.ply')
        if not os.path.exists(mesh_path):
            raise FileNotFoundError(f"Mesh not found: {mesh_path}")

        return mesh_path

    def get_model_diameter(self, object_name: str) -> float:
        """Get model diameter for metric computation."""
        obj_id = self._get_object_id(object_name)
        if obj_id is None:
            raise ValueError(f"Unknown object name: {object_name}")

        diameter_mm = self._models_info[str(obj_id)]['diameter']
        return diameter_mm / 1000.0

    def get_model_symmetries(self, object_name: str) -> Dict:
        """Get model symmetry information."""
        obj_id = self._get_object_id(object_name)
        if obj_id is None:
            raise ValueError(f"Unknown object name: {object_name}")

        model_info = self._models_info[str(obj_id)]

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

        Matches Oryon's implementation exactly.
        """
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

                discrete_steps_count = int(np.ceil(np.pi / max_sym_disc_step))
                discrete_step = 2.0 * np.pi / discrete_steps_count

                for i in range(discrete_steps_count):
                    angle = i * discrete_step
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
