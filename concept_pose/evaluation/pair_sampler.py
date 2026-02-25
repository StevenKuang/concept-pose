"""
Test Pair Sampler for Evaluation
=================================

Handles loading test pairs from JSON files and on-the-fly sampling for
pose estimation evaluation.
"""

import json
import random
import numpy as np
from typing import List, Dict, Tuple, Optional
from pathlib import Path

from concept_pose.data.base_dataset import BaseDataset


class PairSampler:
    """
    Test pair sampler for one-shot pose estimation evaluation.

    Supports:
    - Loading pre-generated pairs from JSON
    - Sampling pairs on-the-fly with various constraints
    - Saving sampled pairs to JSON
    """

    def __init__(self, dataset: BaseDataset):
        """
        Initialize pair sampler.

        Args:
            dataset: BaseDataset instance
        """
        self.dataset = dataset

    def load_pairs(self, pairs_file: str) -> Tuple[List[Dict], Dict]:
        """
        Load test pairs from JSON file.

        Args:
            pairs_file: Path to pairs JSON file

        Returns:
            Tuple of (pairs list, split_config dict)
        """
        print(f"Loading test pairs from: {pairs_file}")
        with open(pairs_file, 'r') as f:
            test_data = json.load(f)

        pairs = test_data['pairs']
        split_config = test_data['split_config']

        print(f"  Loaded {len(pairs)} pairs")
        print(f"  Split: {split_config['name']}")
        print(f"  Scene mode: {split_config['scene_mode']}")

        return pairs, split_config

    def sample_pairs(
        self,
        objects: List[str],
        num_pairs: int,
        scene_mode: str = 'mixed',
        min_frame_gap: int = 10,
        seed: Optional[int] = None
    ) -> List[Dict]:
        """
        Sample anchor-query pairs on-the-fly.

        Args:
            objects: List of object names to sample pairs for
            num_pairs: Total number of pairs to sample
            scene_mode: 'same_scene', 'cross_scene', or 'mixed'
            min_frame_gap: Minimum frame gap for same-scene pairs
            seed: Random seed for reproducibility

        Returns:
            List of pair dicts
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        print(f"\n{'='*60}")
        print(f"Sampling Test Pairs")
        print(f"{'='*60}")
        print(f"  Objects: {len(objects)}")
        print(f"  Num pairs: {num_pairs}")
        print(f"  Scene mode: {scene_mode}")
        print(f"  Min frame gap: {min_frame_gap}")
        print(f"  Random seed: {seed}")

        # Filter objects based on scene mode
        if scene_mode == 'cross_scene':
            # Only keep objects that appear in multiple scenes
            multi_scene_objects = []
            for obj_name in objects:
                valid_frames = self.dataset.get_valid_frames_for_object(obj_name)
                scenes = set()
                for frame_idx in valid_frames:
                    scene_idx, _ = self.dataset.get_frame_info(frame_idx)
                    scenes.add(scene_idx)
                if len(scenes) >= 2:
                    multi_scene_objects.append(obj_name)

            if len(multi_scene_objects) < len(objects):
                print(f"  Filtered to {len(multi_scene_objects)} objects with cross-scene instances")
                print(f"  Skipped {len(objects) - len(multi_scene_objects)} single-scene objects")

            objects = multi_scene_objects

        # Distribute pairs across objects
        pairs_per_object = max(1, num_pairs // len(objects))
        remainder = num_pairs % len(objects)

        all_pairs = []
        for idx, obj_name in enumerate(objects):
            # Give extra pairs to first N objects to reach target
            obj_pairs_count = pairs_per_object + (1 if idx < remainder else 0)

            obj_pairs = self._sample_pairs_for_object(
                object_name=obj_name,
                num_pairs=obj_pairs_count,
                scene_mode=scene_mode,
                min_frame_gap=min_frame_gap
            )
            all_pairs.extend(obj_pairs)

        # Shuffle and limit to requested number
        random.shuffle(all_pairs)
        all_pairs = all_pairs[:num_pairs]

        print(f"\nSampled {len(all_pairs)} total pairs")
        return all_pairs

    def _sample_pairs_for_object(
        self,
        object_name: str,
        num_pairs: int,
        scene_mode: str,
        min_frame_gap: int
    ) -> List[Dict]:
        """
        Sample pairs for a specific object.

        Args:
            object_name: Object name
            num_pairs: Number of pairs to sample
            scene_mode: 'same_scene', 'cross_scene', or 'mixed'
            min_frame_gap: Minimum frame gap for same-scene pairs

        Returns:
            List of pair dicts
        """
        # Get valid frames for this object
        valid_frames = self.dataset.get_valid_frames_for_object(object_name)

        if len(valid_frames) < 2:
            print(f"  Warning: {object_name} has only {len(valid_frames)} frames, skipping")
            return []

        print(f"  {object_name}: {len(valid_frames)} valid frames")

        pairs = []
        seen_pairs = set()
        attempts = 0
        max_attempts = num_pairs * 1000  # Allow many attempts

        while len(pairs) < num_pairs and attempts < max_attempts:
            attempts += 1

            # Sample two random frames
            idx_a, idx_q = random.sample(valid_frames, 2)

            # Get scene info
            scene_a, frame_in_scene_a = self.dataset.get_frame_info(idx_a)
            scene_q, frame_in_scene_q = self.dataset.get_frame_info(idx_q)

            # Check scene mode constraint
            same_scene = (scene_a == scene_q)

            if scene_mode == 'same_scene' and not same_scene:
                continue
            if scene_mode == 'cross_scene' and same_scene:
                continue

            # For same scene, enforce minimum frame gap
            # Note: frame_gap calculation is dataset-specific, so we use a simple check
            if same_scene and abs(frame_in_scene_a - frame_in_scene_q) < min_frame_gap:
                continue

            # Check if already sampled
            # Use ordered tuple: (A, B) and (B, A) are different pairs in relative pose estimation
            pair_key = (idx_a, idx_q)
            if pair_key in seen_pairs:
                continue

            # Create pair dict
            pairs.append({
                'anchor_frame': int(idx_a),
                'query_frame': int(idx_q),
                'object_name': object_name,
                'category': object_name.split('-')[0],
                'metadata': {
                    'same_scene': same_scene,
                    'anchor_scene': f'scene{scene_a+1:02d}',
                    'query_scene': f'scene{scene_q+1:02d}',
                    'anchor_frame_num': int(frame_in_scene_a),
                    'query_frame_num': int(frame_in_scene_q),
                    'frame_gap': int(abs(frame_in_scene_a - frame_in_scene_q)) if same_scene else -1
                }
            })
            seen_pairs.add(pair_key)

        if len(pairs) < num_pairs:
            print(f"  Warning: Only sampled {len(pairs)}/{num_pairs} pairs for {object_name}")

        return pairs

    def load_oryon_fixed_split(
        self,
        oryon_root: str,
        split_name: str = 'cross_scene_test',
        num_pairs: Optional[int] = None
    ) -> Tuple[List[Dict], Dict]:
        """
        Load pairs from oryon's fixed Real275 split.

        Args:
            oryon_root: Path to oryon data root (contains fixed_split directory)
            split_name: Split name ('cross_scene_test' or 'overfit_cross')
            num_pairs: Optional limit on number of pairs to load (takes first N pairs)

        Returns:
            Tuple of (pairs list, split_config dict)
        """
        from concept_pose.data.dataset_real275 import DatasetReal275

        # Validate dataset type
        if not isinstance(self.dataset, DatasetReal275):
            raise ValueError(
                "Oryon fixed split is only available for Real275 dataset, "
                f"but dataset is {type(self.dataset).__name__}"
            )

        # Load instance list from oryon's fixed split
        split_path = Path(oryon_root) / 'fixed_split' / split_name / 'instance_list.txt'
        if not split_path.exists():
            raise FileNotFoundError(f"Oryon split file not found: {split_path}")

        print(f"\n{'='*60}")
        print(f"Loading Oryon Fixed Split: {split_name}")
        print(f"{'='*60}")
        print(f"  Split file: {split_path}")

        pairs = []
        skipped = 0

        with open(split_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                # Parse line: "real_test, <scene_a> <frame_a>, <scene_q> <frame_q>, <obj_id> <obj_name>"
                parts = line.strip().split(',')
                if len(parts) != 4:
                    print(f"  Warning: Skipping malformed line {line_num}: {line.strip()}")
                    skipped += 1
                    continue

                _, idx_a, idx_q, obj_info = parts

                # Parse anchor scene and frame
                scene_a, frame_a = [int(x) for x in idx_a.strip().split()]

                # Parse query scene and frame
                scene_q, frame_q = [int(x) for x in idx_q.strip().split()]

                # Parse object info (category_id and object_name)
                obj_parts = obj_info.strip().split()
                obj_name = obj_parts[1]  # Object name

                # Convert (scene, frame) to global frame index
                global_idx_a = self.dataset.get_frame_by_scene_and_num(scene_a, frame_a)
                global_idx_q = self.dataset.get_frame_by_scene_and_num(scene_q, frame_q)

                if global_idx_a is None or global_idx_q is None:
                    print(f"  Warning: Could not find frames for pair {line_num}: "
                          f"scene{scene_a}_frame{frame_a} or scene{scene_q}_frame{frame_q}")
                    skipped += 1
                    continue

                # Check if object exists in both frames
                # (This validation happens during evaluation, so we just store the pair)

                # Create pair dict in our format
                pairs.append({
                    'anchor_frame': int(global_idx_a),
                    'query_frame': int(global_idx_q),
                    'object_name': obj_name,
                    'category': obj_name.split('_')[0],  # Extract category from object name
                    'metadata': {
                        'same_scene': (scene_a == scene_q),
                        'anchor_scene': f'scene{scene_a:02d}',
                        'query_scene': f'scene{scene_q:02d}',
                        'anchor_frame_num': int(frame_a),
                        'query_frame_num': int(frame_q),
                        'frame_gap': int(abs(frame_a - frame_q)) if scene_a == scene_q else -1,
                        'source': f'oryon_{split_name}'
                    }
                })

        total_loaded = len(pairs)
        print(f"  Loaded {total_loaded} pairs")
        if skipped > 0:
            print(f"  Skipped {skipped} pairs (malformed or missing frames)")

        # Apply num_pairs limit if specified (takes first N, deterministic)
        if num_pairs is not None and num_pairs < len(pairs):
            pairs = pairs[:num_pairs]
            print(f"  Limited to first {num_pairs} pairs (from {total_loaded} total)")

        # Count same-scene vs cross-scene
        same_scene_count = sum(1 for p in pairs if p['metadata']['same_scene'])
        cross_scene_count = len(pairs) - same_scene_count
        print(f"  Same-scene pairs: {same_scene_count}")
        print(f"  Cross-scene pairs: {cross_scene_count}")

        # Create split config
        split_config = {
            'name': f'oryon_{split_name}' + (f'_{num_pairs}' if num_pairs else ''),
            'total_pairs': len(pairs),
            'scene_mode': 'mixed' if same_scene_count > 0 and cross_scene_count > 0 else (
                'same_scene' if same_scene_count > 0 else 'cross_scene'
            ),
            'source': 'oryon_fixed_split',
            'split_file': str(split_path),
            'num_pairs_limit': num_pairs if num_pairs else total_loaded
        }

        return pairs, split_config

    def load_oryon_fixed_split_tyol(
        self,
        oryon_root: str,
        split_name: str = 'cross_scene_test',
        num_pairs: Optional[int] = None
    ) -> Tuple[List[Dict], Dict]:
        """
        Load pairs from oryon's fixed TYOL split.

        Args:
            oryon_root: Path to oryon data root (contains fixed_split directory)
            split_name: Split name ('cross_scene_test' or 'overfit_cross')
            num_pairs: Optional limit on number of pairs to load (takes first N pairs)

        Returns:
            Tuple of (pairs list, split_config dict)
        """
        from concept_pose.data.dataset_tyol import DatasetTyol

        # Validate dataset type
        if not isinstance(self.dataset, DatasetTyol):
            raise ValueError(
                "Oryon fixed TYOL split is only available for TYOL dataset, "
                f"but dataset is {type(self.dataset).__name__}"
            )

        # Load instance list from oryon's fixed split
        split_path = Path(oryon_root) / 'fixed_split' / split_name / 'instance_list.txt'
        if not split_path.exists():
            raise FileNotFoundError(f"Oryon TYOL split file not found: {split_path}")

        print(f"\n{'='*60}")
        print(f"Loading Oryon Fixed TYOL Split: {split_name}")
        print(f"{'='*60}")
        print(f"  Split file: {split_path}")

        pairs = []
        skipped = 0

        with open(split_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                # Parse line: "test, <scene_a> <frame_a>, <scene_q> <frame_q>, <cls_id>"
                # Example: "test, 20 15, 20 39, 20"
                parts = line.strip().split(',')
                if len(parts) != 4:
                    print(f"  Warning: Skipping malformed line {line_num}: {line.strip()}")
                    skipped += 1
                    continue

                _, idx_a, idx_q, cls_id_str = parts

                # Parse anchor scene and frame
                scene_a, frame_a = [int(x) for x in idx_a.strip().split()]

                # Parse query scene and frame
                scene_q, frame_q = [int(x) for x in idx_q.strip().split()]

                # Parse object ID
                cls_id = int(cls_id_str.strip())

                # Get object name from ID
                object_name = self.dataset._get_object_name(cls_id)

                # Convert (scene, frame) to global frame index
                global_idx_a = self.dataset.get_frame_by_scene_and_num(scene_a, frame_a)
                global_idx_q = self.dataset.get_frame_by_scene_and_num(scene_q, frame_q)

                if global_idx_a is None or global_idx_q is None:
                    print(f"  Warning: Could not find frames for pair {line_num}: "
                          f"scene{scene_a}_frame{frame_a} or scene{scene_q}_frame{frame_q}")
                    skipped += 1
                    continue

                # Create pair dict in our format
                pairs.append({
                    'anchor_frame': int(global_idx_a),
                    'query_frame': int(global_idx_q),
                    'object_name': object_name,
                    'category': object_name.split('_')[0],  # Extract category (e.g., "cup" from "cup_10")
                    'metadata': {
                        'same_scene': (scene_a == scene_q),
                        'anchor_scene': f'scene{scene_a:02d}',
                        'query_scene': f'scene{scene_q:02d}',
                        'anchor_frame_num': int(frame_a),
                        'query_frame_num': int(frame_q),
                        'frame_gap': int(abs(frame_a - frame_q)) if scene_a == scene_q else -1,
                        'source': f'oryon_tyol_{split_name}',
                        'object_id': cls_id
                    }
                })

        total_loaded = len(pairs)
        print(f"  Loaded {total_loaded} pairs")
        if skipped > 0:
            print(f"  Skipped {skipped} pairs (malformed or missing frames)")

        # Apply num_pairs limit if specified (takes first N, deterministic)
        if num_pairs is not None and num_pairs < len(pairs):
            pairs = pairs[:num_pairs]
            print(f"  Limited to first {num_pairs} pairs (from {total_loaded} total)")

        # Count same-scene vs cross-scene
        same_scene_count = sum(1 for p in pairs if p['metadata']['same_scene'])
        cross_scene_count = len(pairs) - same_scene_count
        print(f"  Same-scene pairs: {same_scene_count}")
        print(f"  Cross-scene pairs: {cross_scene_count}")

        # Create split config
        split_config = {
            'name': f'oryon_tyol_{split_name}' + (f'_{num_pairs}' if num_pairs else ''),
            'total_pairs': len(pairs),
            'scene_mode': 'mixed' if same_scene_count > 0 and cross_scene_count > 0 else (
                'same_scene' if same_scene_count > 0 else 'cross_scene'
            ),
            'source': 'oryon_fixed_tyol_split',
            'split_file': str(split_path),
            'num_pairs_limit': num_pairs if num_pairs else total_loaded
        }

        return pairs, split_config

    def sample_one2any_first_occurrence_ycbv(
        self,
        num_pairs: Optional[int] = None,
        seed: Optional[int] = None,
        use_bop_targets: bool = True
    ) -> Tuple[List[Dict], Dict]:
        """
        Sample pairs using One2Any's first occurrence strategy for YCB-Video.

        For each object:
            - For each scene, find the first frame where object appears
            - Use first frame as reference
            - All other frames with that object in same scene become query frames
            - **Filter to BOP test targets (900 frames) for fair comparison**

        This matches One2Any's exact sampling strategy:
        - Implicit same-scene constraint (multiple references per object)
        - First frame per (object, scene) pair becomes reference
        - Only uses BOP-curated test frames (default)
        - Deterministic given sorted frame order

        Args:
            num_pairs: Optional limit on number of pairs (samples randomly if specified)
            seed: Random seed for sampling when num_pairs is specified
            use_bop_targets: Filter to BOP test targets only (default: True)

        Returns:
            Tuple of (pairs list, split_config dict)
        """
        import json
        from pathlib import Path
        from concept_pose.data.dataset_ycbv import DatasetYCBV

        # Validate dataset type
        if not isinstance(self.dataset, DatasetYCBV):
            raise ValueError(
                "One2Any first occurrence sampling is only available for YCB-V dataset, "
                f"but dataset is {type(self.dataset).__name__}"
            )

        print(f"\n{'='*60}")
        print(f"Sampling One2Any First Occurrence Pairs (YCB-Video)")
        print(f"{'='*60}")

        # Load BOP test targets if filtering
        bop_targets_frames = None
        if use_bop_targets:
            bop_targets_path = Path(self.dataset.bop_root) / 'ycbv_base' / 'test_targets_bop19.json'
            if not bop_targets_path.exists():
                print(f"  Warning: BOP targets not found at {bop_targets_path}")
                print(f"  Falling back to all frames")
                use_bop_targets = False
            else:
                with open(bop_targets_path, 'r') as f:
                    bop_targets = json.load(f)

                # Build set of (scene_id, frame_id, obj_id) tuples from BOP targets
                bop_targets_frames = set()
                for target in bop_targets:
                    scene_id = target['scene_id']
                    im_id = target['im_id']
                    obj_id = target['obj_id']
                    bop_targets_frames.add((scene_id, im_id, obj_id))

                print(f"  BOP test targets loaded: {len(bop_targets)} targets")
                print(f"  Unique frames: {len(set((t['scene_id'], t['im_id']) for t in bop_targets))}")

        # Build object index if not already built
        self.dataset._build_object_index()

        # Get all objects in dataset
        all_objects = self.dataset.get_all_objects()
        print(f"  Total objects: {len(all_objects)}")

        pairs = []
        reference_frames = {}  # (object_name, scene) -> reference_frame_idx

        # For each object
        for obj_name in all_objects:
            # Get all frames containing this object
            valid_frames = self.dataset.get_valid_frames_for_object(obj_name)

            if len(valid_frames) == 0:
                continue

            # Get object ID for BOP filtering
            obj_id = self.dataset._get_object_id(obj_name)

            # Group frames by scene and filter to BOP targets
            scene_to_frames = {}
            for frame_idx in valid_frames:
                scene_idx, frame_num = self.dataset.get_frame_info(frame_idx)

                # Filter to BOP targets if enabled
                if use_bop_targets and bop_targets_frames is not None:
                    if (scene_idx, frame_num, obj_id) not in bop_targets_frames:
                        continue

                if scene_idx not in scene_to_frames:
                    scene_to_frames[scene_idx] = []
                scene_to_frames[scene_idx].append((frame_idx, frame_num))

            # For each scene, sort frames and pick first as reference
            for scene_idx, frame_list in scene_to_frames.items():
                # Sort by frame number to ensure first occurrence
                frame_list_sorted = sorted(frame_list, key=lambda x: x[1])

                if len(frame_list_sorted) < 2:
                    # Need at least 2 frames (1 ref + 1 query)
                    continue

                # First frame becomes reference
                ref_frame_idx, ref_frame_num = frame_list_sorted[0]
                reference_frames[(obj_name, scene_idx)] = ref_frame_idx

                # All other frames become queries
                for query_frame_idx, query_frame_num in frame_list_sorted[1:]:
                    pairs.append({
                        'anchor_frame': int(ref_frame_idx),
                        'query_frame': int(query_frame_idx),
                        'object_name': obj_name,
                        'category': self.dataset._get_category_name(
                            self.dataset._get_object_id(obj_name)
                        ),
                        'metadata': {
                            'same_scene': True,  # Always same scene for One2Any
                            'anchor_scene': f'scene{scene_idx:02d}',
                            'query_scene': f'scene{scene_idx:02d}',
                            'anchor_frame_num': int(ref_frame_num),
                            'query_frame_num': int(query_frame_num),
                            'frame_gap': int(query_frame_num - ref_frame_num),
                            'source': 'one2any_first_occurrence',
                            'is_first_occurrence_ref': True,
                            'bop_filtered': use_bop_targets
                        }
                    })

        total_pairs = len(pairs)
        print(f"  Generated {total_pairs} pairs from {len(reference_frames)} reference frames")
        print(f"  Reference frames: {len(reference_frames)} (one per object-scene)")

        # Apply num_pairs limit if specified (random sampling)
        if num_pairs is not None and num_pairs < len(pairs):
            if seed is not None:
                random.seed(seed)
            pairs = random.sample(pairs, num_pairs)
            print(f"  Randomly sampled {num_pairs} pairs from {total_pairs} total")

        # Count objects
        object_counts = {}
        for pair in pairs:
            obj = pair['object_name']
            object_counts[obj] = object_counts.get(obj, 0) + 1

        print(f"  Unique objects in pairs: {len(object_counts)}")
        print(f"  Pairs per object (mean): {len(pairs) / len(object_counts):.1f}")

        # Create split config
        split_config = {
            'name': f'one2any_first_occurrence_ycbv' + (f'_{num_pairs}' if num_pairs else '_bop900'),
            'total_pairs': len(pairs),
            'scene_mode': 'same_scene',  # Always same-scene for One2Any
            'source': 'one2any_first_occurrence',
            'sampling_strategy': 'first_occurrence_per_object_scene',
            'num_reference_frames': len(reference_frames),
            'total_available_pairs': total_pairs,
            'bop_filtered': use_bop_targets,
            'test_set': 'BOP-curated 900 frames' if use_bop_targets else 'All frames'
        }

        return pairs, split_config

    def sample_oryon_style_ycbv(
        self,
        num_pairs: int = 2000,
        scene_mode: str = 'cross_scene',
        seed: Optional[int] = 42,
        use_bop_targets: bool = True
    ) -> Tuple[List[Dict], Dict]:
        """
        Sample random anchor-query pairs for YCB-V (Oryon-style evaluation).

        Mimics Oryon's evaluation protocol:
        - Random sampling of 2k pairs (not first-occurrence like One2Any)
        - Cross-scene pairs by default (Oryon's typical approach for Real275)
        - BOP test targets filtering (900 frames) for fair comparison
        - Fixed seed for reproducibility
        - Fair distribution across objects

        This is YCB-V's equivalent to Oryon's published Real275/TYOL splits,
        but generated on-the-fly since Oryon didn't publish YCB-V pairs.

        Args:
            num_pairs: Total number of pairs to sample (default: 2000)
            scene_mode: 'cross_scene', 'same_scene', or 'mixed' (default: 'cross_scene')
            seed: Random seed for reproducibility (default: 42)
            use_bop_targets: Filter to BOP test targets only (default: True)

        Returns:
            Tuple of (pairs list, split_config dict)
        """
        import json
        from pathlib import Path
        from concept_pose.data.dataset_ycbv import DatasetYCBV

        # Validate dataset type
        if not isinstance(self.dataset, DatasetYCBV):
            raise ValueError(
                "Oryon-style sampling is only available for YCB-V dataset, "
                f"but dataset is {type(self.dataset).__name__}"
            )

        print(f"\n{'='*60}")
        print(f"Sampling Oryon-Style Pairs (YCB-Video)")
        print(f"{'='*60}")
        print(f"  Num pairs: {num_pairs}")
        print(f"  Scene mode: {scene_mode}")
        print(f"  BOP filtering: {use_bop_targets}")
        print(f"  Random seed: {seed}")

        # Load BOP test targets if filtering
        bop_targets_frames = None
        if use_bop_targets:
            bop_targets_path = Path(self.dataset.bop_root) / 'ycbv_base' / 'test_targets_bop19.json'
            if not bop_targets_path.exists():
                print(f"  Warning: BOP targets not found at {bop_targets_path}")
                print(f"  Falling back to all frames")
                use_bop_targets = False
            else:
                with open(bop_targets_path, 'r') as f:
                    bop_targets = json.load(f)

                # Build set of (scene_id, frame_id, obj_id) tuples from BOP targets
                bop_targets_frames = set()
                for target in bop_targets:
                    scene_id = target['scene_id']
                    im_id = target['im_id']
                    obj_id = target['obj_id']
                    bop_targets_frames.add((scene_id, im_id, obj_id))

                print(f"  BOP test targets loaded: {len(bop_targets)} targets")
                print(f"  Unique frames: {len(set((t['scene_id'], t['im_id']) for t in bop_targets))}")

        # Build object index if not already built
        self.dataset._build_object_index()

        # Get all objects in dataset
        all_objects = self.dataset.get_all_objects()
        print(f"  Total objects: {len(all_objects)}")

        # If BOP filtering is enabled, we need to temporarily filter valid frames
        # We'll do this by temporarily wrapping the dataset's get_valid_frames_for_object method
        original_get_valid_frames = None
        if use_bop_targets and bop_targets_frames is not None:
            original_get_valid_frames = self.dataset.get_valid_frames_for_object

            # Create BOP-filtered version
            def get_valid_frames_bop_filtered(obj_name: str):
                frames = original_get_valid_frames(obj_name)
                obj_id = self.dataset._get_object_id(obj_name)
                filtered = []
                for frame_idx in frames:
                    scene_idx, frame_num = self.dataset.get_frame_info(frame_idx)
                    if (scene_idx, frame_num, obj_id) in bop_targets_frames:
                        filtered.append(frame_idx)
                return filtered

            # Temporarily replace method
            self.dataset.get_valid_frames_for_object = get_valid_frames_bop_filtered

        # Filter objects that have enough frames for sampling
        objects_to_sample = []
        for obj_name in all_objects:
            valid_frames = self.dataset.get_valid_frames_for_object(obj_name)
            if len(valid_frames) >= 2:  # Need at least 2 frames to make a pair
                objects_to_sample.append(obj_name)

        print(f"  Objects with sufficient frames: {len(objects_to_sample)}")

        # Sample pairs using existing infrastructure
        pairs = self.sample_pairs(
            objects=objects_to_sample,
            num_pairs=num_pairs,
            scene_mode=scene_mode,
            min_frame_gap=10,
            seed=seed
        )

        # Restore original method if we wrapped it
        if original_get_valid_frames is not None:
            self.dataset.get_valid_frames_for_object = original_get_valid_frames

        # Fix category names using YCB-V's proper category extraction
        # (sample_pairs uses hyphen splitting which doesn't work for YCB-V's underscore format)
        for pair in pairs:
            obj_name = pair['object_name']
            obj_id = self.dataset._get_object_id(obj_name)
            if obj_id is not None:
                pair['category'] = self.dataset._get_category_name(obj_id)

        # Add BOP filtering metadata to pairs
        for pair in pairs:
            pair['metadata']['bop_filtered'] = use_bop_targets
            pair['metadata']['source'] = 'oryon_style_random'

        # Count statistics
        same_scene_count = sum(1 for p in pairs if p['metadata']['same_scene'])
        cross_scene_count = len(pairs) - same_scene_count

        print(f"\n  Sampled {len(pairs)} pairs:")
        print(f"    Same-scene: {same_scene_count}")
        print(f"    Cross-scene: {cross_scene_count}")

        # Count objects
        object_counts = {}
        for pair in pairs:
            obj = pair['object_name']
            object_counts[obj] = object_counts.get(obj, 0) + 1

        print(f"  Unique objects: {len(object_counts)}")
        print(f"  Pairs per object (mean): {len(pairs) / len(object_counts):.1f}")

        # Create split config
        split_config = {
            'name': f'oryon_style_ycbv_{num_pairs}',
            'total_pairs': len(pairs),
            'scene_mode': scene_mode,
            'source': 'oryon_style_random_sampling',
            'sampling_strategy': 'random_with_bop_filtering',
            'num_objects': len(object_counts),
            'bop_filtered': use_bop_targets,
            'test_set': 'BOP-curated 900 frames' if use_bop_targets else 'All frames',
            'same_scene_count': same_scene_count,
            'cross_scene_count': cross_scene_count,
            'seed': seed
        }

        return pairs, split_config

    def sample_oryon_style_lm(
        self,
        num_pairs: int = 2000,
        scene_mode: str = 'same_scene',
        seed: Optional[int] = 42,
        use_bop_targets: bool = True
    ) -> Tuple[List[Dict], Dict]:
        """
        Sample random anchor-query pairs for LINEMOD (Oryon-style evaluation).

        Mimics Oryon's evaluation protocol adapted for LINEMOD:
        - Random sampling of 2k pairs (not first-occurrence like One2Any)
        - Same-scene pairs by default (REQUIRED: each LINEMOD object appears in only one scene)
        - BOP test targets filtering (3000 frames) for fair comparison
        - Fixed seed for reproducibility
        - Fair distribution across objects

        This is LINEMOD's equivalent to Oryon's published Real275/TYOL splits,
        but generated on-the-fly since Oryon didn't publish LINEMOD pairs.

        Args:
            num_pairs: Total number of pairs to sample (default: 2000)
            scene_mode: 'same_scene' (default), 'cross_scene', or 'mixed'
                        NOTE: 'same_scene' is REQUIRED for LINEMOD since each object
                        appears in only one scene. Cross-scene will fail.
            seed: Random seed for reproducibility (default: 42)
            use_bop_targets: Filter to BOP test targets only (default: True)

        Returns:
            Tuple of (pairs list, split_config dict)
        """
        import json
        from pathlib import Path
        from concept_pose.data.dataset_lm import DatasetLM

        # Validate dataset type
        if not isinstance(self.dataset, DatasetLM):
            raise ValueError(
                "Oryon-style LM sampling is only available for LINEMOD dataset, "
                f"but dataset is {type(self.dataset).__name__}"
            )

        # Warn if cross-scene mode is requested (will likely fail)
        if scene_mode == 'cross_scene':
            print(f"\nWarning: Cross-scene mode requested for LINEMOD, but each object")
            print(f"         appears in only one scene. This will likely fail.")
            print(f"         Consider using scene_mode='same_scene' instead.")

        print(f"\n{'='*60}")
        print(f"Sampling Oryon-Style Pairs (LINEMOD)")
        print(f"{'='*60}")
        print(f"  Num pairs: {num_pairs}")
        print(f"  Scene mode: {scene_mode}")
        print(f"  BOP filtering: {use_bop_targets}")
        print(f"  Random seed: {seed}")

        # Load BOP test targets if filtering
        bop_targets_frames = None
        if use_bop_targets:
            bop_targets_path = Path(self.dataset.bop_root) / 'lm_base' / 'test_targets_bop19.json'
            if not bop_targets_path.exists():
                print(f"  Warning: BOP targets not found at {bop_targets_path}")
                print(f"  Falling back to all frames")
                use_bop_targets = False
            else:
                with open(bop_targets_path, 'r') as f:
                    bop_targets = json.load(f)

                # Build set of (scene_id, frame_id, obj_id) tuples from BOP targets
                bop_targets_frames = set()
                for target in bop_targets:
                    scene_id = target['scene_id']
                    im_id = target['im_id']
                    obj_id = target['obj_id']
                    bop_targets_frames.add((scene_id, im_id, obj_id))

                print(f"  BOP test targets loaded: {len(bop_targets)} targets")
                print(f"  Unique frames: {len(set((t['scene_id'], t['im_id']) for t in bop_targets))}")

        # Build object index if not already built
        self.dataset._build_object_index()

        # Get all objects in dataset
        all_objects = self.dataset.get_all_objects()
        print(f"  Total objects: {len(all_objects)}")

        # If BOP filtering is enabled, we need to temporarily filter valid frames
        # We'll do this by temporarily wrapping the dataset's get_valid_frames_for_object method
        original_get_valid_frames = None
        if use_bop_targets and bop_targets_frames is not None:
            original_get_valid_frames = self.dataset.get_valid_frames_for_object

            # Create BOP-filtered version
            def get_valid_frames_bop_filtered(obj_name: str):
                frames = original_get_valid_frames(obj_name)
                obj_id = self.dataset._get_object_id(obj_name)
                filtered = []
                for frame_idx in frames:
                    scene_idx, frame_num = self.dataset.get_frame_info(frame_idx)
                    if (scene_idx, frame_num, obj_id) in bop_targets_frames:
                        filtered.append(frame_idx)
                return filtered

            # Temporarily replace method
            self.dataset.get_valid_frames_for_object = get_valid_frames_bop_filtered

        # Filter objects that have enough frames for sampling
        objects_to_sample = []
        for obj_name in all_objects:
            valid_frames = self.dataset.get_valid_frames_for_object(obj_name)
            if len(valid_frames) >= 2:  # Need at least 2 frames to make a pair
                objects_to_sample.append(obj_name)

        print(f"  Objects with sufficient frames: {len(objects_to_sample)}")

        # Sample pairs using existing infrastructure
        pairs = self.sample_pairs(
            objects=objects_to_sample,
            num_pairs=num_pairs,
            scene_mode=scene_mode,
            min_frame_gap=10,
            seed=seed
        )

        # Restore original method if we wrapped it
        if original_get_valid_frames is not None:
            self.dataset.get_valid_frames_for_object = original_get_valid_frames

        # Fix category names using LINEMOD's BOP names (ape, can, duck, etc.)
        # (sample_pairs uses hyphen splitting which doesn't work for LINEMOD's obj_id format)
        for pair in pairs:
            obj_name = pair['object_name']
            obj_id = self.dataset._get_object_id(obj_name)
            if obj_id is not None:
                pair['category'] = self.dataset._get_bop_name(obj_id)

        # Add BOP filtering metadata to pairs
        for pair in pairs:
            pair['metadata']['bop_filtered'] = use_bop_targets
            pair['metadata']['source'] = 'oryon_style_random'

        # Count statistics
        same_scene_count = sum(1 for p in pairs if p['metadata']['same_scene'])
        cross_scene_count = len(pairs) - same_scene_count

        print(f"\n  Sampled {len(pairs)} pairs:")
        print(f"    Same-scene: {same_scene_count}")
        print(f"    Cross-scene: {cross_scene_count}")

        # Count objects
        object_counts = {}
        for pair in pairs:
            obj = pair['object_name']
            object_counts[obj] = object_counts.get(obj, 0) + 1

        print(f"  Unique objects: {len(object_counts)}")
        print(f"  Pairs per object (mean): {len(pairs) / len(object_counts):.1f}")

        # Create split config
        split_config = {
            'name': f'oryon_style_lm_{num_pairs}',
            'total_pairs': len(pairs),
            'scene_mode': scene_mode,
            'source': 'oryon_style_random_sampling',
            'sampling_strategy': 'random_with_bop_filtering',
            'num_objects': len(object_counts),
            'bop_filtered': use_bop_targets,
            'test_set': 'BOP-curated 3000 frames' if use_bop_targets else 'All frames',
            'same_scene_count': same_scene_count,
            'cross_scene_count': cross_scene_count,
            'seed': seed
        }

        return pairs, split_config

    def sample_lmo_occlusion_study(
        self,
        seed: Optional[int] = 42,
        max_queries_per_object: Optional[int] = None
    ) -> Tuple[List[Dict], Dict]:
        """
        Sample pairs for LM-O occlusion vs performance study.

        For each object:
        1. Find the frame with MINIMAL occlusion (highest visib_fract) → anchor
        2. All other frames become queries (uniformly sampled if max_queries_per_object set)
        3. Track visib_fract for correlation analysis

        This creates a controlled experiment where:
        - Anchor is always the "best" view (least occluded)
        - Query occlusion varies, allowing correlation analysis

        Args:
            seed: Random seed for reproducibility
            max_queries_per_object: If set, uniformly sample this many queries per object

        Returns:
            Tuple of (pairs list, split_config dict)
        """
        from concept_pose.data.dataset_lmo import DatasetLMO

        # Validate dataset type
        if not isinstance(self.dataset, DatasetLMO):
            raise ValueError(
                "LM-O occlusion study sampling requires DatasetLMO, "
                f"but dataset is {type(self.dataset).__name__}"
            )

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        print(f"\n{'='*60}")
        print(f"Sampling LM-O Occlusion Study Pairs")
        print(f"{'='*60}")
        print(f"  Strategy: Least-occluded anchor → all other queries")
        print(f"  Random seed: {seed}")
        if max_queries_per_object:
            print(f"  Max queries per object: {max_queries_per_object}")

        # Build object index
        self.dataset._build_object_index()

        # Get all objects
        all_objects = self.dataset.get_all_objects()
        print(f"  Total objects: {len(all_objects)}")

        pairs = []

        for obj_name in all_objects:
            obj_id = self.dataset._get_object_id(obj_name)
            bop_name = self.dataset._get_bop_name(obj_id)

            # Use DatasetLMO's get_frames_with_occlusion method
            # Returns list of (frame_idx, visib_fract) sorted by visib_fract descending
            frame_occlusion = self.dataset.get_frames_with_occlusion(obj_name)

            if len(frame_occlusion) < 2:
                print(f"  Skipping {obj_name}: only {len(frame_occlusion)} frames")
                continue

            # Anchor = least occluded frame (highest visib_fract)
            anchor_idx, anchor_visib = frame_occlusion[0]
            anchor_scene, anchor_frame_num = self.dataset.get_frame_info(anchor_idx)

            # All other frames are potential queries
            all_queries = frame_occlusion[1:]

            # Uniform sampling if max_queries_per_object is set
            if max_queries_per_object and len(all_queries) > max_queries_per_object:
                # Sample uniformly across occlusion range (sorted by visib_fract)
                indices = np.linspace(0, len(all_queries) - 1, max_queries_per_object, dtype=int)
                queries_to_use = [all_queries[i] for i in indices]
            else:
                queries_to_use = all_queries

            print(f"  {obj_name} ({bop_name}): anchor visib={anchor_visib:.3f}, "
                  f"{len(queries_to_use)} queries (from {len(all_queries)} total)")

            # Create pairs for selected queries
            for query_idx, query_visib in queries_to_use:
                query_scene, query_frame_num = self.dataset.get_frame_info(query_idx)

                pair = {
                    'anchor_frame': anchor_idx,
                    'query_frame': query_idx,
                    'object_name': obj_name,
                    'category': bop_name,
                    'metadata': {
                        'same_scene': anchor_scene == query_scene,
                        'anchor_scene': anchor_scene,
                        'query_scene': query_scene,
                        'anchor_frame_num': anchor_frame_num,
                        'query_frame_num': query_frame_num,
                        'anchor_visib_fract': anchor_visib,
                        'query_visib_fract': query_visib,
                        'source': 'lmo_occlusion_study'
                    }
                }
                pairs.append(pair)

        print(f"\n  Total pairs: {len(pairs)}")

        # Statistics
        visib_fracts = [p['metadata']['query_visib_fract'] for p in pairs]
        print(f"  Query visib_fract range: [{min(visib_fracts):.3f}, {max(visib_fracts):.3f}]")
        print(f"  Query visib_fract mean: {np.mean(visib_fracts):.3f}")

        # Create split config
        split_config = {
            'name': 'lmo_occlusion_study',
            'total_pairs': len(pairs),
            'scene_mode': 'same_scene',
            'source': 'lmo_occlusion_study',
            'sampling_strategy': 'least_occluded_anchor',
            'num_objects': len(all_objects),
            'seed': seed
        }

        return pairs, split_config

    def sample_icosphere(
        self,
        num_references: int = 2,
        num_pairs: Optional[int] = None,
        seed: Optional[int] = None,
        use_bop_targets: bool = False,
        per_scene: bool = True
    ) -> Tuple[List[Dict], Dict]:
        """
        Sample reference frames using icosphere-based viewpoint selection.

        Mimics FoundationPose/UA-Pose reference selection strategy:
        - Select num_references frames with maximum viewpoint diversity
        - Use geodesic distance on viewing sphere for diversity
        - All other frames become queries
        - Dataset-agnostic: works with any BaseDataset that provides GT poses

        For each object (and optionally each scene):
            1. Get all frames containing the object
            2. Compute viewing directions (camera-to-object unit vectors)
            3. Select num_references frames with maximum angular separation
            4. Create pairs: reference frames → all other frames

        Args:
            num_references: Number of reference frames per object/scene (default: 2)
            num_pairs: Optional limit on total pairs (random sample if exceeded)
            seed: Random seed for reproducibility
            use_bop_targets: For YCB-V, filter to BOP test targets only
            per_scene: If True, select references per (object, scene) pair.
                      If False, select references per object across all scenes.

        Returns:
            Tuple of (pairs list, split_config dict)
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        print(f"\n{'='*60}")
        print(f"Sampling Icosphere-Based Pairs")
        print(f"{'='*60}")
        print(f"  Num references: {num_references}")
        print(f"  Per scene: {per_scene}")
        print(f"  Use BOP targets: {use_bop_targets}")
        print(f"  Random seed: {seed}")

        # Handle BOP filtering for YCB-V
        bop_targets_frames = None
        if use_bop_targets:
            try:
                from concept_pose.data.dataset_ycbv import DatasetYCBV
                if isinstance(self.dataset, DatasetYCBV):
                    bop_targets_path = Path(self.dataset.bop_root) / 'ycbv_base' / 'test_targets_bop19.json'
                    if bop_targets_path.exists():
                        with open(bop_targets_path, 'r') as f:
                            bop_targets = json.load(f)
                        bop_targets_frames = set()
                        for target in bop_targets:
                            bop_targets_frames.add((target['scene_id'], target['im_id'], target['obj_id']))
                        print(f"  BOP test targets loaded: {len(bop_targets)} targets")
            except ImportError:
                pass

        # Build object index if not already built
        if hasattr(self.dataset, '_build_object_index'):
            self.dataset._build_object_index()

        # Get all objects
        all_objects = self.dataset.get_all_objects()
        print(f"  Total objects: {len(all_objects)}")

        pairs = []
        reference_frames = {}  # (object_name, scene) -> list of reference frame indices
        skipped_objects = []

        # For each object
        for obj_name in all_objects:
            valid_frames = self.dataset.get_valid_frames_for_object(obj_name)

            if len(valid_frames) < num_references + 1:
                skipped_objects.append(obj_name)
                continue

            # Get object ID for BOP filtering (if applicable)
            obj_id = None
            if hasattr(self.dataset, '_get_object_id'):
                obj_id = self.dataset._get_object_id(obj_name)

            # Group frames by scene if per_scene=True
            if per_scene:
                scene_to_frames = {}
                for frame_idx in valid_frames:
                    scene_idx, frame_num = self.dataset.get_frame_info(frame_idx)

                    # BOP filtering
                    if use_bop_targets and bop_targets_frames is not None and obj_id is not None:
                        if (scene_idx, frame_num, obj_id) not in bop_targets_frames:
                            continue

                    if scene_idx not in scene_to_frames:
                        scene_to_frames[scene_idx] = []
                    scene_to_frames[scene_idx].append(frame_idx)

                # Process each scene
                for scene_idx, frame_list in scene_to_frames.items():
                    if len(frame_list) < num_references + 1:
                        continue

                    # Select diverse reference frames
                    ref_indices = self._select_diverse_views(
                        frame_list,
                        obj_name,
                        num_references
                    )

                    # Skip if we couldn't get viewing directions
                    if len(ref_indices) == 0:
                        continue

                    reference_frames[(obj_name, scene_idx)] = ref_indices

                    # Create pairs: references -> all other frames
                    query_indices = [f for f in frame_list if f not in ref_indices]

                    for query_idx in query_indices:
                        scene_q, frame_num_q = self.dataset.get_frame_info(query_idx)

                        pairs.append({
                            'anchor_frame': int(ref_indices[0]),  # Primary reference
                            'query_frame': int(query_idx),
                            'object_name': obj_name,
                            'category': obj_name.split('_')[0] if '_' in obj_name else obj_name,
                            'metadata': {
                                'same_scene': True,
                                'reference_frames': [int(r) for r in ref_indices],
                                'num_references': len(ref_indices),
                                'query_scene': f'scene{scene_q:02d}',
                                'query_frame_num': int(frame_num_q),
                                'source': 'icosphere_sampling',
                                'bop_filtered': use_bop_targets
                            }
                        })
            else:
                # Global: select references across all scenes for this object
                # BOP filtering
                filtered_frames = valid_frames
                if use_bop_targets and bop_targets_frames is not None and obj_id is not None:
                    filtered_frames = []
                    for frame_idx in valid_frames:
                        scene_idx, frame_num = self.dataset.get_frame_info(frame_idx)
                        if (scene_idx, frame_num, obj_id) in bop_targets_frames:
                            filtered_frames.append(frame_idx)

                if len(filtered_frames) < num_references + 1:
                    skipped_objects.append(obj_name)
                    continue

                # Select diverse reference frames
                ref_indices = self._select_diverse_views(
                    filtered_frames,
                    obj_name,
                    num_references
                )

                # Skip if we couldn't get viewing directions
                if len(ref_indices) == 0:
                    skipped_objects.append(obj_name)
                    continue

                reference_frames[(obj_name, -1)] = ref_indices

                # Create pairs: references -> all other frames
                query_indices = [f for f in filtered_frames if f not in ref_indices]

                for query_idx in query_indices:
                    scene_q, frame_num_q = self.dataset.get_frame_info(query_idx)

                    pairs.append({
                        'anchor_frame': int(ref_indices[0]),  # Primary reference
                        'query_frame': int(query_idx),
                        'object_name': obj_name,
                        'category': obj_name.split('_')[0] if '_' in obj_name else obj_name,
                        'metadata': {
                            'same_scene': False,  # Cross-scene possible
                            'reference_frames': [int(r) for r in ref_indices],
                            'num_references': len(ref_indices),
                            'query_scene': f'scene{scene_q:02d}',
                            'query_frame_num': int(frame_num_q),
                            'source': 'icosphere_sampling',
                            'bop_filtered': use_bop_targets
                        }
                    })

        total_pairs = len(pairs)
        print(f"  Generated {total_pairs} pairs from {len(reference_frames)} reference sets")
        print(f"  Skipped {len(skipped_objects)} objects (insufficient frames)")

        # Apply num_pairs limit if specified
        if num_pairs is not None and num_pairs < len(pairs):
            pairs = random.sample(pairs, num_pairs)
            print(f"  Randomly sampled {num_pairs} pairs from {total_pairs} total")

        # Create split config
        split_config = {
            'name': f'icosphere_{num_references}ref' + ('_perscene' if per_scene else '_global') + (f'_{num_pairs}' if num_pairs else ''),
            'total_pairs': len(pairs),
            'scene_mode': 'same_scene' if per_scene else 'mixed',
            'source': 'icosphere_sampling',
            'num_references': num_references,
            'per_scene': per_scene,
            'num_reference_sets': len(reference_frames),
            'total_available_pairs': total_pairs,
            'bop_filtered': use_bop_targets,
            'sampling_strategy': 'farthest_point_on_viewing_sphere'
        }

        return pairs, split_config

    def _select_diverse_views(
        self,
        frame_indices: List[int],
        obj_name: str,
        num_views: int
    ) -> List[int]:
        """
        Select num_views frames with maximum viewpoint diversity.

        Uses farthest point sampling on the viewing sphere:
        1. Compute camera-to-object viewing directions for all frames
        2. Select first frame randomly
        3. Iteratively select frame with maximum minimum distance to selected frames

        Args:
            frame_indices: List of candidate frame indices
            obj_name: Object name
            num_views: Number of views to select

        Returns:
            List of selected frame indices with diverse viewpoints
        """
        if len(frame_indices) <= num_views:
            return frame_indices

        # Compute viewing directions for all frames
        viewing_directions = []
        valid_indices = []

        for frame_idx in frame_indices:
            direction = self._get_viewing_direction(frame_idx, obj_name)
            if direction is not None:
                viewing_directions.append(direction)
                valid_indices.append(frame_idx)

        if len(valid_indices) <= num_views:
            return valid_indices

        viewing_directions = np.array(viewing_directions)  # Shape: (N, 3)

        # Farthest point sampling
        selected_indices = []

        # Start with random frame
        first_idx = np.random.randint(len(valid_indices))
        selected_indices.append(first_idx)

        # Iteratively select frames with maximum minimum distance to selected
        for _ in range(num_views - 1):
            # Compute distances from all frames to all selected frames
            # Distance = 1 - dot product (since vectors are normalized)
            selected_dirs = viewing_directions[selected_indices]  # Shape: (k, 3)

            # Compute dot products: (N, 3) @ (3, k) = (N, k)
            dot_products = viewing_directions @ selected_dirs.T

            # For each candidate, get minimum distance to any selected frame
            # Distance on sphere = arccos(dot_product), but we can use 1 - dot_product as proxy
            min_distances = 1.0 - np.max(dot_products, axis=1)  # Max dot product = min angle

            # Mask already selected frames
            min_distances[selected_indices] = -np.inf

            # Select frame with maximum minimum distance
            next_idx = np.argmax(min_distances)
            selected_indices.append(next_idx)

        # Return frame indices
        return [valid_indices[i] for i in selected_indices]

    def _get_viewing_direction(
        self,
        frame_idx: int,
        obj_name: str
    ) -> Optional[np.ndarray]:
        """
        Get viewing direction (camera-to-object unit vector) for a frame.

        Args:
            frame_idx: Frame index
            obj_name: Object name

        Returns:
            Viewing direction as unit vector (3,), or None if pose unavailable
        """
        try:
            # Get frame info
            scene_idx, frame_num = self.dataset.get_frame_info(frame_idx)

            # Get object ID
            if not hasattr(self.dataset, '_get_object_id'):
                return None
            obj_id = self.dataset._get_object_id(obj_name)

            # Try YCB-V style access: _scene_data
            if hasattr(self.dataset, '_scene_data'):
                scene_data = self.dataset._scene_data.get(scene_idx, {})
                frame_data = scene_data.get(frame_num, [])

                # Find object in annotations
                for ann in frame_data:
                    if ann['obj_id'] == obj_id:
                        # 't' is object center in camera frame (already in meters for YCB-V)
                        t = ann['t']  # Shape: (3,)

                        # Normalize to get viewing direction
                        # (from camera origin to object center)
                        norm = np.linalg.norm(t)
                        if norm > 0:
                            return t / norm
                        else:
                            return None

            # Fallback: Try to access via frame_list structure (for other datasets)
            if hasattr(self.dataset, 'frame_list') and frame_idx < len(self.dataset.frame_list):
                frame_info = self.dataset.frame_list[frame_idx]
                if 'objects' in frame_info and obj_name in frame_info['objects']:
                    obj_info = frame_info['objects'][obj_name]
                    if 'pose' in obj_info:
                        pose = obj_info['pose']
                        if isinstance(pose, np.ndarray) and pose.shape == (4, 4):
                            translation = pose[:3, 3]
                            norm = np.linalg.norm(translation)
                            if norm > 0:
                                return translation / norm

            return None

        except Exception as e:
            # Silently fail and skip this frame
            return None

    def save_pairs(
        self,
        pairs: List[Dict],
        output_file: str,
        split_config: Optional[Dict] = None
    ):
        """
        Save pairs to JSON file.

        Args:
            pairs: List of pair dicts
            output_file: Output JSON path
            split_config: Optional split configuration metadata
        """
        if split_config is None:
            split_config = {
                'name': Path(output_file).stem,
                'total_pairs': len(pairs),
                'scene_mode': 'unknown'
            }

        output_data = {
            'pairs': pairs,
            'split_config': split_config
        }

        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)

        print(f"Saved {len(pairs)} pairs to: {output_file}")

    def export_oryon_format(
        self,
        pairs: List[Dict],
        split_config: Dict,
        output_dir: str,
        num_tracked: int = 10
    ):
        """
        Export pairs to Oryon's fixed split format.

        Creates directory structure matching Oryon's Real275/TYOL format:
            <output_dir>/<split_name>/
                instance_list.txt    (all pairs)
                tracked.txt          (num_tracked representative pairs)
                stats.json           (optional metadata)

        File format (follows Real275 style):
            test, <scene_a> <frame_a>, <scene_q> <frame_q>, <obj_id> <object_name>

        Example (YCB-V):
            test, 48 0, 49 15, 2 cracker_box
            test, 48 10, 50 5, 5 tomato_soup_can

        Example (LINEMOD):
            test, 1 15, 1 40, 1 ape
            test, 5 19, 5 63, 5 can

        Args:
            pairs: List of pair dicts from sampling methods
            split_config: Split configuration dict
            output_dir: Base output directory (e.g., 'data/oryon_data/ycbv/fixed_split')
            num_tracked: Number of pairs to include in tracked.txt (default: 10)
        """
        from concept_pose.data.dataset_ycbv import DatasetYCBV
        from concept_pose.data.dataset_lm import DatasetLM

        # Auto-detect dataset type and select appropriate category extraction method
        if isinstance(self.dataset, DatasetYCBV):
            get_category_name = lambda obj_id: self.dataset._get_category_name(obj_id)
        elif isinstance(self.dataset, DatasetLM):
            get_category_name = lambda obj_id: self.dataset._get_bop_name(obj_id)
        else:
            raise ValueError(
                f"Oryon format export not supported for dataset type: {type(self.dataset).__name__}. "
                f"Supported types: DatasetYCBV, DatasetLM"
            )

        # Get split name from config
        split_name = split_config.get('name', 'exported_split')

        # Create output directory
        split_dir = Path(output_dir) / split_name
        split_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"Exporting to Oryon Format")
        print(f"{'='*60}")
        print(f"  Output directory: {split_dir}")
        print(f"  Total pairs: {len(pairs)}")
        print(f"  Tracked pairs: {num_tracked}")

        # Convert pairs to Oryon format
        oryon_lines = []
        for pair in pairs:
            # Get scene and frame info
            anchor_scene, anchor_frame = self.dataset.get_frame_info(pair['anchor_frame'])
            query_scene, query_frame = self.dataset.get_frame_info(pair['query_frame'])

            # Get object info
            obj_name = pair['object_name']
            obj_id = self.dataset._get_object_id(obj_name)

            if obj_id is None:
                print(f"  Warning: Skipping pair with unknown object: {obj_name}")
                continue

            # Get category name (e.g., "cracker_box" for YCB-V, "ape" for LINEMOD)
            category_name = get_category_name(obj_id)

            # Format line: test, <scene_a> <frame_a>, <scene_q> <frame_q>, <obj_id> <category_name>
            line = f"test, {anchor_scene} {anchor_frame}, {query_scene} {query_frame}, {obj_id} {category_name}\n"
            oryon_lines.append(line)

        # Write instance_list.txt
        instance_list_path = split_dir / 'instance_list.txt'
        with open(instance_list_path, 'w') as f:
            f.writelines(oryon_lines)
        print(f"  Wrote {len(oryon_lines)} pairs to instance_list.txt")

        # Write tracked.txt (first num_tracked pairs)
        if num_tracked > 0:
            tracked_path = split_dir / 'tracked.txt'
            tracked_lines = oryon_lines[:min(num_tracked, len(oryon_lines))]
            with open(tracked_path, 'w') as f:
                f.writelines(tracked_lines)
            print(f"  Wrote {len(tracked_lines)} pairs to tracked.txt")

        # Write stats.json (metadata)
        stats = {
            'split_name': split_name,
            'total_pairs': len(oryon_lines),
            'scene_mode': split_config.get('scene_mode', 'unknown'),
            'num_objects': split_config.get('num_objects', 0),
            'same_scene_count': split_config.get('same_scene_count', 0),
            'cross_scene_count': split_config.get('cross_scene_count', 0),
            'test_set': split_config.get('test_set', 'unknown'),
            'bop_filtered': split_config.get('bop_filtered', False),
            'seed': split_config.get('seed', None),
            'source': split_config.get('source', 'unknown'),
            'sampling_strategy': split_config.get('sampling_strategy', 'unknown')
        }

        # Add per-object statistics
        object_counts = {}
        for pair in pairs:
            obj_name = pair['object_name']
            obj_id = self.dataset._get_object_id(obj_name)
            if obj_id is not None:
                category_name = get_category_name(obj_id)
                object_counts[category_name] = object_counts.get(category_name, 0) + 1

        stats['pairs_per_object'] = dict(sorted(object_counts.items()))

        stats_path = split_dir / 'stats.json'
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"  Wrote statistics to stats.json")

        print(f"\nExport complete!")
        print(f"  instance_list.txt: {instance_list_path}")
        print(f"  tracked.txt: {tracked_path if num_tracked > 0 else 'N/A'}")
        print(f"  stats.json: {stats_path}")

    def load_oryon_fixed_split_ycbv(
        self,
        oryon_root: str,
        split_name: str = 'cross_scene_test',
        num_pairs: Optional[int] = None
    ) -> Tuple[List[Dict], Dict]:
        """
        Load pairs from Oryon's fixed YCB-V split.

        Args:
            oryon_root: Path to oryon YCB-V data root (e.g., 'data/oryon_data/ycbv')
            split_name: Split name (default: 'cross_scene_test')
            num_pairs: Optional limit on number of pairs to load (takes first N pairs)

        Returns:
            Tuple of (pairs list, split_config dict)
        """
        from concept_pose.data.dataset_ycbv import DatasetYCBV

        # Validate dataset type
        if not isinstance(self.dataset, DatasetYCBV):
            raise ValueError(
                "Oryon fixed YCB-V split is only available for YCB-V dataset, "
                f"but dataset is {type(self.dataset).__name__}"
            )

        # Load instance list from oryon's fixed split
        split_path = Path(oryon_root) / 'fixed_split' / split_name / 'instance_list.txt'
        if not split_path.exists():
            raise FileNotFoundError(f"Oryon YCB-V split file not found: {split_path}")

        print(f"\n{'='*60}")
        print(f"Loading Oryon Fixed YCB-V Split: {split_name}")
        print(f"{'='*60}")
        print(f"  Split file: {split_path}")

        pairs = []
        skipped = 0

        with open(split_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                # Parse line: "test, <scene_a> <frame_a>, <scene_q> <frame_q>, <obj_id> <category_name>"
                # Example: "test, 48 0, 49 15, 2 cracker_box"
                parts = line.strip().split(',')
                if len(parts) != 4:
                    print(f"  Warning: Skipping malformed line {line_num}: {line.strip()}")
                    skipped += 1
                    continue

                _, idx_a, idx_q, obj_info = parts

                # Parse anchor scene and frame
                try:
                    scene_a, frame_a = [int(x) for x in idx_a.strip().split()]
                except ValueError:
                    print(f"  Warning: Skipping line {line_num} with invalid anchor format: {idx_a}")
                    skipped += 1
                    continue

                # Parse query scene and frame
                try:
                    scene_q, frame_q = [int(x) for x in idx_q.strip().split()]
                except ValueError:
                    print(f"  Warning: Skipping line {line_num} with invalid query format: {idx_q}")
                    skipped += 1
                    continue

                # Parse object info (obj_id and category_name)
                obj_parts = obj_info.strip().split()
                if len(obj_parts) < 2:
                    print(f"  Warning: Skipping line {line_num} with invalid object info: {obj_info}")
                    skipped += 1
                    continue

                obj_id = int(obj_parts[0])
                category_name = obj_parts[1]

                # Get full object name (e.g., "cracker_box_2" from obj_id=2)
                object_name = self.dataset._get_object_name(obj_id)

                # Convert (scene, frame) to global frame index
                global_idx_a = self.dataset.get_frame_by_scene_and_num(scene_a, frame_a)
                global_idx_q = self.dataset.get_frame_by_scene_and_num(scene_q, frame_q)

                if global_idx_a is None or global_idx_q is None:
                    print(f"  Warning: Could not find frames for pair {line_num}: "
                          f"scene{scene_a}_frame{frame_a} or scene{scene_q}_frame{frame_q}")
                    skipped += 1
                    continue

                # Create pair dict in our format
                pairs.append({
                    'anchor_frame': int(global_idx_a),
                    'query_frame': int(global_idx_q),
                    'object_name': object_name,
                    'category': category_name,
                    'metadata': {
                        'same_scene': (scene_a == scene_q),
                        'anchor_scene': f'scene{scene_a:02d}',
                        'query_scene': f'scene{scene_q:02d}',
                        'anchor_frame_num': int(frame_a),
                        'query_frame_num': int(frame_q),
                        'frame_gap': int(abs(frame_a - frame_q)) if scene_a == scene_q else -1,
                        'source': f'oryon_ycbv_{split_name}',
                        'object_id': obj_id
                    }
                })

        total_loaded = len(pairs)
        print(f"  Loaded {total_loaded} pairs")
        if skipped > 0:
            print(f"  Skipped {skipped} pairs (malformed or missing frames)")

        # Apply num_pairs limit if specified (takes first N, deterministic)
        if num_pairs is not None and num_pairs < len(pairs):
            pairs = pairs[:num_pairs]
            print(f"  Limited to first {num_pairs} pairs (from {total_loaded} total)")

        # Count same-scene vs cross-scene
        same_scene_count = sum(1 for p in pairs if p['metadata']['same_scene'])
        cross_scene_count = len(pairs) - same_scene_count
        print(f"  Same-scene pairs: {same_scene_count}")
        print(f"  Cross-scene pairs: {cross_scene_count}")

        # Create split config
        split_config = {
            'name': f'oryon_ycbv_{split_name}' + (f'_{num_pairs}' if num_pairs else ''),
            'total_pairs': len(pairs),
            'scene_mode': 'mixed' if same_scene_count > 0 and cross_scene_count > 0 else (
                'same_scene' if same_scene_count > 0 else 'cross_scene'
            ),
            'source': 'oryon_fixed_ycbv_split',
            'split_file': str(split_path),
            'num_pairs_limit': num_pairs if num_pairs else total_loaded
        }

        return pairs, split_config

    def load_oryon_fixed_split_lm(
        self,
        oryon_root: str,
        split_name: str = 'same_scene_test',
        num_pairs: Optional[int] = None
    ) -> Tuple[List[Dict], Dict]:
        """
        Load pairs from Oryon's fixed LINEMOD split.

        Args:
            oryon_root: Path to oryon LINEMOD data root (e.g., 'data/oryon_data/lm')
            split_name: Split name (default: 'same_scene_test')
            num_pairs: Optional limit on number of pairs to load (takes first N pairs)

        Returns:
            Tuple of (pairs list, split_config dict)
        """
        from concept_pose.data.dataset_lm import DatasetLM

        # Validate dataset type
        if not isinstance(self.dataset, DatasetLM):
            raise ValueError(
                "Oryon fixed LINEMOD split is only available for LINEMOD dataset, "
                f"but dataset is {type(self.dataset).__name__}"
            )

        # Load instance list from oryon's fixed split
        split_path = Path(oryon_root) / 'fixed_split' / split_name / 'instance_list.txt'
        if not split_path.exists():
            raise FileNotFoundError(f"Oryon LINEMOD split file not found: {split_path}")

        print(f"\n{'='*60}")
        print(f"Loading Oryon Fixed LINEMOD Split: {split_name}")
        print(f"{'='*60}")
        print(f"  Split file: {split_path}")

        pairs = []
        skipped = 0

        with open(split_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                # Parse line: "test, <scene_a> <frame_a>, <scene_q> <frame_q>, <obj_id> <category_name>"
                # Example: "test, 1 15, 1 40, 1 ape"
                parts = line.strip().split(',')
                if len(parts) != 4:
                    print(f"  Warning: Skipping malformed line {line_num}: {line.strip()}")
                    skipped += 1
                    continue

                _, idx_a, idx_q, obj_info = parts

                # Parse anchor scene and frame
                try:
                    scene_a, frame_a = [int(x) for x in idx_a.strip().split()]
                except ValueError:
                    print(f"  Warning: Skipping line {line_num} with invalid anchor format: {idx_a}")
                    skipped += 1
                    continue

                # Parse query scene and frame
                try:
                    scene_q, frame_q = [int(x) for x in idx_q.strip().split()]
                except ValueError:
                    print(f"  Warning: Skipping line {line_num} with invalid query format: {idx_q}")
                    skipped += 1
                    continue

                # Parse object info (obj_id and category_name)
                obj_parts = obj_info.strip().split()
                if len(obj_parts) < 2:
                    print(f"  Warning: Skipping line {line_num} with invalid object info: {obj_info}")
                    skipped += 1
                    continue

                obj_id = int(obj_parts[0])
                category_name = obj_parts[1]

                # Get full object name (e.g., "obj_1" from obj_id=1)
                object_name = self.dataset._get_object_name(obj_id)

                # Convert (scene, frame) to global frame index
                global_idx_a = self.dataset.get_frame_by_scene_and_num(scene_a, frame_a)
                global_idx_q = self.dataset.get_frame_by_scene_and_num(scene_q, frame_q)

                if global_idx_a is None or global_idx_q is None:
                    print(f"  Warning: Could not find frames for pair {line_num}: "
                          f"scene{scene_a}_frame{frame_a} or scene{scene_q}_frame{frame_q}")
                    skipped += 1
                    continue

                # Create pair dict in our format
                pairs.append({
                    'anchor_frame': int(global_idx_a),
                    'query_frame': int(global_idx_q),
                    'object_name': object_name,
                    'category': category_name,
                    'metadata': {
                        'same_scene': (scene_a == scene_q),
                        'anchor_scene': f'scene{scene_a:02d}',
                        'query_scene': f'scene{scene_q:02d}',
                        'anchor_frame_num': int(frame_a),
                        'query_frame_num': int(frame_q),
                        'frame_gap': int(abs(frame_a - frame_q)) if scene_a == scene_q else -1,
                        'source': f'oryon_lm_{split_name}',
                        'object_id': obj_id
                    }
                })

        total_loaded = len(pairs)
        print(f"  Loaded {total_loaded} pairs")
        if skipped > 0:
            print(f"  Skipped {skipped} pairs (malformed or missing frames)")

        # Apply num_pairs limit if specified (takes first N, deterministic)
        if num_pairs is not None and num_pairs < len(pairs):
            pairs = pairs[:num_pairs]
            print(f"  Limited to first {num_pairs} pairs (from {total_loaded} total)")

        # Count same-scene vs cross-scene
        same_scene_count = sum(1 for p in pairs if p['metadata']['same_scene'])
        cross_scene_count = len(pairs) - same_scene_count
        print(f"  Same-scene pairs: {same_scene_count}")
        print(f"  Cross-scene pairs: {cross_scene_count}")

        # Create split config
        split_config = {
            'name': f'oryon_lm_{split_name}' + (f'_{num_pairs}' if num_pairs else ''),
            'total_pairs': len(pairs),
            'scene_mode': 'mixed' if same_scene_count > 0 and cross_scene_count > 0 else (
                'same_scene' if same_scene_count > 0 else 'cross_scene'
            ),
            'source': 'oryon_fixed_lm_split',
            'split_file': str(split_path),
            'num_pairs_limit': num_pairs if num_pairs else total_loaded
        }

        return pairs, split_config

    def sample_one2any_first_occurrence_lm(
        self,
        num_pairs: Optional[int] = None,
        seed: Optional[int] = None,
        use_bop_targets: bool = True
    ) -> Tuple[List[Dict], Dict]:
        """
        Sample pairs using One2Any's first occurrence strategy for LINEMOD.

        For each object:
            - For each scene, find the first frame where object appears
            - Use first frame as reference
            - All other frames with that object in same scene become query frames
            - **Filter to BOP test targets (3000 frames) for fair comparison**

        This matches One2Any's exact sampling strategy:
        - Implicit same-scene constraint (multiple references per object)
        - First frame per (object, scene) pair becomes reference
        - Only uses BOP-curated test frames (default)
        - Deterministic given sorted frame order

        Args:
            num_pairs: Optional limit on number of pairs (samples randomly if specified)
            seed: Random seed for sampling when num_pairs is specified
            use_bop_targets: Filter to BOP test targets only (default: True)

        Returns:
            Tuple of (pairs list, split_config dict)
        """
        import json
        import random
        from pathlib import Path
        from concept_pose.data.dataset_lm import DatasetLM

        # Validate dataset type
        if not isinstance(self.dataset, DatasetLM):
            raise ValueError(
                "One2Any first occurrence sampling for LINEMOD is only available for LM dataset, "
                f"but dataset is {type(self.dataset).__name__}"
            )

        print(f"\n{'='*60}")
        print(f"Sampling One2Any First Occurrence Pairs (LINEMOD)")
        print(f"{'='*60}")

        # Load BOP test targets if filtering
        bop_targets_frames = None
        if use_bop_targets:
            bop_targets_path = Path(self.dataset.bop_root) / 'lm_base' / 'test_targets_bop19.json'
            if not bop_targets_path.exists():
                print(f"  Warning: BOP targets not found at {bop_targets_path}")
                print(f"  Falling back to all frames")
                use_bop_targets = False
            else:
                with open(bop_targets_path, 'r') as f:
                    bop_targets = json.load(f)

                # Build set of (scene_id, frame_id, obj_id) tuples from BOP targets
                bop_targets_frames = set()
                for target in bop_targets:
                    scene_id = target['scene_id']
                    im_id = target['im_id']
                    obj_id = target['obj_id']
                    bop_targets_frames.add((scene_id, im_id, obj_id))

                print(f"  BOP test targets loaded: {len(bop_targets)} targets")
                print(f"  Unique frames: {len(set((t['scene_id'], t['im_id']) for t in bop_targets))}")

        # Build object index if not already built
        self.dataset._build_object_index()

        # Get all objects in dataset
        all_objects = self.dataset.get_all_objects()
        print(f"  Total objects: {len(all_objects)}")

        pairs = []
        reference_frames = {}  # (object_name, scene) -> reference_frame_idx

        # For each object
        for obj_name in all_objects:
            # Get all frames containing this object
            valid_frames = self.dataset.get_valid_frames_for_object(obj_name)

            if len(valid_frames) == 0:
                continue

            # Get object ID for BOP filtering
            obj_id = self.dataset._get_object_id(obj_name)

            # Group frames by scene and filter to BOP targets
            scene_to_frames = {}
            for frame_idx in valid_frames:
                scene_idx, frame_num = self.dataset.get_frame_info(frame_idx)

                # Filter to BOP targets if enabled
                if use_bop_targets and bop_targets_frames is not None:
                    if (scene_idx, frame_num, obj_id) not in bop_targets_frames:
                        continue

                if scene_idx not in scene_to_frames:
                    scene_to_frames[scene_idx] = []
                scene_to_frames[scene_idx].append((frame_idx, frame_num))

            # For each scene, sort frames and pick first as reference
            for scene_idx, frame_list in scene_to_frames.items():
                # Sort by frame number to ensure first occurrence
                frame_list_sorted = sorted(frame_list, key=lambda x: x[1])

                if len(frame_list_sorted) < 2:
                    # Need at least 2 frames (1 ref + 1 query)
                    continue

                # First frame becomes reference
                ref_frame_idx, ref_frame_num = frame_list_sorted[0]
                reference_frames[(obj_name, scene_idx)] = ref_frame_idx

                # All other frames become queries
                for query_frame_idx, query_frame_num in frame_list_sorted[1:]:
                    pairs.append({
                        'anchor_frame': int(ref_frame_idx),
                        'query_frame': int(query_frame_idx),
                        'object_name': obj_name,
                        'category': self.dataset._get_bop_name(obj_id),  # Use BOP name as category
                        'metadata': {
                            'same_scene': True,  # Always same scene for One2Any
                            'anchor_scene': f'scene{scene_idx:02d}',
                            'query_scene': f'scene{scene_idx:02d}',
                            'anchor_frame_num': int(ref_frame_num),
                            'query_frame_num': int(query_frame_num),
                            'frame_gap': int(query_frame_num - ref_frame_num),
                            'source': 'one2any_first_occurrence',
                            'is_first_occurrence_ref': True,
                            'bop_filtered': use_bop_targets,
                            'bop_name': self.dataset._get_bop_name(obj_id)
                        }
                    })

        total_pairs = len(pairs)
        print(f"  Generated {total_pairs} pairs from {len(reference_frames)} reference frames")
        print(f"  Reference frames: {len(reference_frames)} (one per object-scene)")

        # Apply num_pairs limit if specified (random sampling)
        if num_pairs is not None and num_pairs < len(pairs):
            if seed is not None:
                random.seed(seed)
            pairs = random.sample(pairs, num_pairs)
            print(f"  Randomly sampled {num_pairs} pairs from {total_pairs} total")

        # Count objects
        object_counts = {}
        for pair in pairs:
            obj = pair['object_name']
            object_counts[obj] = object_counts.get(obj, 0) + 1

        print(f"  Unique objects in pairs: {len(object_counts)}")
        print(f"  Pairs per object (mean): {len(pairs) / len(object_counts):.1f}")

        # Create split config
        split_config = {
            'name': f'one2any_first_occurrence_lm' + (f'_{num_pairs}' if num_pairs else '_bop3000'),
            'total_pairs': len(pairs),
            'scene_mode': 'same_scene',  # Always same-scene for One2Any
            'source': 'one2any_first_occurrence',
            'sampling_strategy': 'first_occurrence_per_object_scene',
            'num_reference_frames': len(reference_frames),
            'total_available_pairs': total_pairs,
            'bop_filtered': use_bop_targets,
            'test_set': 'BOP-curated 3000 frames' if use_bop_targets else 'All frames'
        }

        return pairs, split_config
