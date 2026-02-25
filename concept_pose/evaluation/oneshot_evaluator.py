"""
One-Shot Pose Estimation Evaluator
===================================

Dataset-agnostic evaluator for one-shot pose estimation.
Extracted from test_housecat_batch.py to work with any BaseDataset implementation.
"""

import os
import numpy as np
import torch
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for headless environments
import matplotlib.pyplot as plt

from concept_pose.data.base_dataset import BaseDataset
from concept_pose.pose.one_shot_estimator import OneShotPoseEstimator
from concept_pose.pose.bop_metrics import BOPEvaluator
from concept_pose.pose.pose_metrics import compute_all_metrics
from .base_evaluator import BaseEvaluator


def project_points_to_image(
    points_3d: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    K: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Project 3D points to image coordinates.

    Args:
        points_3d: (N, 3) 3D points in object frame
        R: (3, 3) rotation matrix (object to camera)
        t: (3,) translation vector
        K: (3, 3) camera intrinsics

    Returns:
        (N, 2) pixel coordinates, (N,) valid mask
    """
    # Transform to camera frame
    points_cam = (R @ points_3d.T).T + t

    # Filter points behind camera
    valid_depth = points_cam[:, 2] > 0

    # Project to image
    points_hom = (K @ points_cam.T).T
    pixels = points_hom[:, :2] / points_hom[:, 2:3]

    return pixels, valid_depth


def draw_axes_3d(
    ax,
    R: np.ndarray,
    t: np.ndarray,
    K: np.ndarray,
    object_center: np.ndarray,
    scale: float = 0.1,
    linewidth: int = 3,
    alpha: float = 1.0
):
    """
    Draw 3D coordinate axes on image.

    Args:
        ax: Matplotlib axis
        R: (3, 3) rotation matrix (object to camera)
        t: (3,) translation vector
        K: (3, 3) camera intrinsics
        object_center: (3,) center of object in object frame
        scale: Length of axes in meters
        linewidth: Line width for axes
        alpha: Transparency (1.0 = opaque, 0.5 = semi-transparent)
    """
    # Define axes in object frame, centered at object center
    origin = object_center.astype(np.float32)
    x_axis = origin + np.array([scale, 0, 0], dtype=np.float32)
    y_axis = origin + np.array([0, scale, 0], dtype=np.float32)
    z_axis = origin + np.array([0, 0, scale], dtype=np.float32)

    # Project to image
    axes_3d = np.stack([origin, x_axis, y_axis, z_axis])
    axes_2d, valid = project_points_to_image(axes_3d, R, t, K)

    if not np.all(valid):
        return  # Skip if any point is behind camera

    origin_2d = axes_2d[0]
    x_2d = axes_2d[1]
    y_2d = axes_2d[2]
    z_2d = axes_2d[3]

    # Draw X axis (red)
    ax.plot([origin_2d[0], x_2d[0]], [origin_2d[1], x_2d[1]],
            'r-', linewidth=linewidth, alpha=alpha, label='X' if alpha == 1.0 else None)

    # Draw Y axis (green)
    ax.plot([origin_2d[0], y_2d[0]], [origin_2d[1], y_2d[1]],
            'g-', linewidth=linewidth, alpha=alpha, label='Y' if alpha == 1.0 else None)

    # Draw Z axis (blue)
    ax.plot([origin_2d[0], z_2d[0]], [origin_2d[1], z_2d[1]],
            'b-', linewidth=linewidth, alpha=alpha, label='Z' if alpha == 1.0 else None)


# Main Evaluator Class

class OneShotEvaluator(BaseEvaluator):
    """
    Dataset-agnostic one-shot pose estimation evaluator.

    Handles:
    - Mask generation (GT or Grounded SAM2)
    - Evaluation loop
    - Metrics computation
    - Visualization
    - Results aggregation
    """

    def __init__(
        self,
        dataset: BaseDataset,
        estimator: OneShotPoseEstimator,
        bop_evaluator: BOPEvaluator,
        semantic_labels: Union[List[str], Dict[str, List[str]]],
        device: str = 'cuda',
        anchor_pose_mode: str = 'absolute',
        use_mesh_visualization: bool = False
    ):
        """
        Initialize evaluator.

        Args:
            dataset: BaseDataset instance
            estimator: OneShotPoseEstimator instance
            bop_evaluator: BOPEvaluator instance
            semantic_labels: Semantic part labels
                - List[str]: Single label set for all objects (backward compat)
                - Dict[str, List[str]]: Per-object labels {object_name: [labels]}
            device: 'cuda' or 'cpu'
            anchor_pose_mode: Mode for anchor pose usage ('absolute' or 'relative')
            use_mesh_visualization: If True, visualize with mesh instead of anchor point cloud
        """
        self.dataset = dataset
        self.estimator = estimator
        self.bop_evaluator = bop_evaluator
        self.semantic_labels = semantic_labels
        self.use_mesh_visualization = use_mesh_visualization
        self.device = device
        self.anchor_pose_mode = anchor_pose_mode

    def evaluate(
        self,
        pairs: List[Dict],
        mask_cache: Optional[Dict] = None,
        viz_dir: Optional[Path] = None,
        max_pairs: Optional[int] = None,
        output_file: Optional[str] = None,
        eval_metadata: Optional[Dict] = None,
        debug_viz_dir: Optional[Path] = None
    ) -> List[Dict]:
        """
        Evaluate pose estimation on test pairs.

        Args:
            pairs: List of test pair dicts
            mask_cache: Optional pre-generated masks {(frame_idx, object_name): mask}
            viz_dir: Optional directory to save visualizations
            max_pairs: Optional limit on number of pairs to test (takes first N, deterministic)
            output_file: Optional path to save results incrementally (atomic writes)
            eval_metadata: Optional metadata to include in saved results (config, split info, etc.)

        Returns:
            List of result dicts
        """
        if max_pairs is not None:
            pairs = pairs[:max_pairs]

        results = []
        num_success = 0

        print(f"\n{'='*60}")
        print(f"Testing {len(pairs)} pairs")
        print(f"{'='*60}\n")

        # Warmup: Pre-load SigLIP2 model to exclude loading time from first pair's timing
        if len(pairs) > 0:
            print("Warming up SigLIP2 model (excluding from timing)...")
            first_pair = pairs[0]
            first_anchor_data = self.dataset.load_frame(
                first_pair['anchor_frame'],
                first_pair['object_name'],
                mask_cache=mask_cache
            )
            # Extract semantic labels for warmup
            if isinstance(self.semantic_labels, dict):
                warmup_labels = self.semantic_labels[first_pair['object_name']]
            else:
                warmup_labels = self.semantic_labels
            # Trigger model loading with a dummy saliency extraction
            _ = self.estimator._extract_saliency(
                first_anchor_data['rgb'],
                first_anchor_data['mask'],
                warmup_labels
            )
            print("  SigLIP2 model loaded ✓\n")

        # Warning for debug visualization mode
        if debug_viz_dir is not None:
            # Check if voxelization is enabled
            is_voxelized = (self.estimator.voxelize_anchor or self.estimator.voxelize_query)
            memory_estimate_gb = len(pairs) * (0.1 if is_voxelized else 0.05)  # Dense models skip correspondence data

            print(f"⚠️  DEBUG VISUALIZATION MODE ENABLED")
            print(f"   This will save detailed debug data for each pair to: {debug_viz_dir}")
            print(f"   Voxelization: {'Enabled' if is_voxelized else 'Disabled (correspondence data will be skipped to save VRAM)'}")
            print(f"   Estimated memory usage: ~{memory_estimate_gb:.1f} GB for {len(pairs)} pairs")
            if not is_voxelized:
                print(f"   Note: Dense non-voxelized models use less debug memory but more inference VRAM")
            print(f"   WARNING: Large evaluations may cause OOM. Consider using --max_pairs to limit.")
            print(f"   Recommended: Use debug mode with max_pairs <= 100 for 16GB RAM, <= 500 for 64GB RAM")
            print()

        for pair_idx, pair in enumerate(pairs):
            print(f"{'='*60}")
            print(f"Pair {pair_idx + 1}/{len(pairs)}")
            print(f"{'='*60}")

            anchor_idx = pair['anchor_frame']
            query_idx = pair['query_frame']
            object_name = pair['object_name']

            print(f"  Object: {object_name}")
            print(f"  Anchor frame: {anchor_idx}")
            print(f"  Query frame: {query_idx}")
            print(f"  Same scene: {pair['metadata']['same_scene']}")

            try:
                # Evaluate single pair
                result = self._evaluate_single_pair(pair, mask_cache, viz_dir, pair_idx, debug_viz_dir=debug_viz_dir)
                results.append(result)

                if result['success']:
                    num_success += 1

                # Clear GPU cache after each pair to prevent VRAM buildup (especially with debug_viz)
                if debug_viz_dir is not None and torch.cuda.is_available():
                    torch.cuda.empty_cache()

            except Exception as e:
                print(f"\nException: {e}")
                import traceback
                traceback.print_exc()

                results.append({
                    'pair_idx': pair_idx,
                    'anchor_frame': anchor_idx,
                    'query_frame': query_idx,
                    'object_name': object_name,
                    'category': pair['category'],
                    'success': False,
                    'error': str(e)
                })

            # Proactive memory cleanup after each pair to prevent OOM accumulation
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Incremental save after each pair (atomic write)
            if output_file is not None:
                try:
                    self._save_results_atomic(results, output_file, eval_metadata)
                    # Only show save confirmation every 10 pairs to avoid spam
                    if (pair_idx + 1) % 10 == 0 or (pair_idx + 1) == len(pairs):
                        print(f"  💾 Progress saved: {len(results)}/{len(pairs)} pairs → {output_file}")
                except Exception as save_error:
                    print(f"  ⚠️  Warning: Failed to save incremental results: {save_error}")
                    # Continue evaluation even if save fails

            print()

        print(f"{'='*60}")
        print("Batch Testing Complete")
        print(f"{'='*60}")
        print(f"Successful: {num_success}/{len(results)}")
        print(f"Success rate: {100*num_success/len(results):.1f}%")

        # Compute and report pose estimation timing statistics
        pose_times = [r['pose_estimation_time'] for r in results if 'pose_estimation_time' in r]
        if pose_times:
            avg_time = np.mean(pose_times)
            min_time = np.min(pose_times)
            max_time = np.max(pose_times)
            print(f"\nPose Estimation Timing (build + estimate, excludes metrics):")
            print(f"  Average: {avg_time:.2f}s per pair")
            print(f"  Min: {min_time:.2f}s")
            print(f"  Max: {max_time:.2f}s")

        # Save all debug visualization data to a single .npz file
        if debug_viz_dir is not None:
            debug_viz_dir.mkdir(parents=True, exist_ok=True)
            debug_npz_path = debug_viz_dir / "debug_viz.npz"

            # Collect all debug data from results
            all_debug_data = [r['debug_data'] for r in results if r.get('debug_data') is not None]

            if all_debug_data:
                try:
                    # Helper function to convert any value to numpy
                    def to_numpy(value, dtype=None):
                        if value is None:
                            return None
                        # Check if it's a PyTorch tensor
                        if hasattr(value, 'cpu') and hasattr(value, 'numpy'):
                            arr = value.cpu().numpy()
                            return arr.astype(dtype) if dtype is not None else arr
                        # Check if it's already a numpy array
                        if isinstance(value, (np.ndarray, np.generic)):
                            return value.astype(dtype) if dtype is not None else value
                        # For scalars and lists
                        if isinstance(value, (int, float, str, bool, list)):
                            return value
                        # Try np.asarray as fallback
                        arr = np.asarray(value)
                        return arr.astype(dtype) if dtype is not None else arr

                    # Convert all debug data and prepare for saving
                    # We'll store data as arrays indexed by pair
                    save_dict = {}

                    for i, debug_data in enumerate(all_debug_data):
                        prefix = f"pair_{i:04d}_"

                        # Convert with appropriate dtypes
                        save_dict[f'{prefix}pair_idx'] = debug_data['pair_idx']
                        save_dict[f'{prefix}object_name'] = str(debug_data['object_name'])
                        save_dict[f'{prefix}ref_frame_id'] = debug_data['ref_frame_id']
                        save_dict[f'{prefix}query_frame_id'] = debug_data['query_frame_id']
                        save_dict[f'{prefix}category'] = str(debug_data['category'])

                        save_dict[f'{prefix}rgb_ref'] = to_numpy(debug_data['rgb_ref'], np.uint8)
                        save_dict[f'{prefix}rgb_query'] = to_numpy(debug_data['rgb_query'], np.uint8)
                        save_dict[f'{prefix}mask_ref'] = to_numpy(debug_data['mask_ref'], np.uint8)
                        save_dict[f'{prefix}mask_query'] = to_numpy(debug_data['mask_query'], np.uint8)
                        save_dict[f'{prefix}bbox_ref'] = to_numpy(debug_data['bbox_ref'])
                        save_dict[f'{prefix}bbox_query'] = to_numpy(debug_data['bbox_query'])
                        save_dict[f'{prefix}saliency_ref'] = to_numpy(debug_data['saliency_ref'], np.float32)
                        save_dict[f'{prefix}saliency_query'] = to_numpy(debug_data['saliency_query'], np.float32)
                        save_dict[f'{prefix}labels'] = to_numpy(debug_data['labels'])
                        save_dict[f'{prefix}observed_3d'] = to_numpy(debug_data['observed_3d'], np.float32)
                        save_dict[f'{prefix}observed_saliency'] = to_numpy(debug_data['observed_saliency'], np.float32)
                        save_dict[f'{prefix}pixel_coords_query'] = to_numpy(debug_data['pixel_coords_query'], np.int16)

                        # Correspondence data (only save if available, skipped for dense non-voxelized models)
                        if debug_data['corr_observed_idx'] is not None:
                            save_dict[f'{prefix}corr_observed_idx'] = to_numpy(debug_data['corr_observed_idx'], np.int32)
                            save_dict[f'{prefix}corr_observed_3d'] = to_numpy(debug_data['corr_observed_3d'], np.float32)
                            save_dict[f'{prefix}corr_model_3d'] = to_numpy(debug_data['corr_model_3d'], np.float32)
                            save_dict[f'{prefix}corr_kl_divergence'] = to_numpy(debug_data['corr_kl_divergence'], np.float32)
                            save_dict[f'{prefix}corr_cosine_similarity'] = to_numpy(debug_data['corr_cosine_similarity'], np.float32)
                            save_dict[f'{prefix}corr_observed_saliency'] = to_numpy(debug_data['corr_observed_saliency'], np.float32)
                            save_dict[f'{prefix}corr_model_saliency'] = to_numpy(debug_data['corr_model_saliency'], np.float32)
                            save_dict[f'{prefix}corr_pixel_coords_ref'] = to_numpy(debug_data['corr_pixel_coords_ref'], np.float32)

                        save_dict[f'{prefix}inlier_mask'] = to_numpy(debug_data['inlier_mask'], np.bool_)
                        save_dict[f'{prefix}pose_ref_gt'] = to_numpy(debug_data['pose_ref_gt'])
                        save_dict[f'{prefix}pose_query_gt'] = to_numpy(debug_data['pose_query_gt'])
                        save_dict[f'{prefix}K_ref'] = to_numpy(debug_data['K_ref'], np.float32)
                        save_dict[f'{prefix}K_query'] = to_numpy(debug_data['K_query'], np.float32)

                        # Build estimated pose matrix from R and t
                        R_est_np = to_numpy(debug_data['pose_query_est_R'])
                        t_est_np = to_numpy(debug_data['pose_query_est_t'])
                        if R_est_np is not None and t_est_np is not None:
                            pose_est = np.vstack([
                                np.hstack([R_est_np, t_est_np.reshape(3, 1)]),
                                [0, 0, 0, 1]
                            ])
                        else:
                            pose_est = None
                        save_dict[f'{prefix}pose_query_est'] = pose_est

                        save_dict[f'{prefix}success'] = debug_data['success']
                        save_dict[f'{prefix}num_correspondences'] = debug_data['num_correspondences']
                        save_dict[f'{prefix}num_inliers'] = debug_data['num_inliers']
                        save_dict[f'{prefix}scale'] = debug_data['scale']

                        # Add pose metrics (only present if success=True)
                        save_dict[f'{prefix}rotation_error_deg'] = debug_data.get('rotation_error_deg', None)
                        save_dict[f'{prefix}translation_error_m'] = debug_data.get('translation_error_m', None)
                        save_dict[f'{prefix}add_error'] = debug_data.get('add_error', None)

                    # Add evaluation metadata if available
                    if eval_metadata is not None:
                        # Store eval_metadata as JSON string (numpy doesn't support nested dicts well)
                        import json
                        save_dict['eval_metadata_json'] = json.dumps(eval_metadata)
                        print(f"\n   Including evaluation metadata in debug file")

                    # Check if correspondence data was saved
                    has_correspondence = all_debug_data[0]['corr_observed_idx'] is not None if all_debug_data else False

                    # Save to single .npz file
                    np.savez_compressed(debug_npz_path, **save_dict)
                    print(f"\n💾 Saved debug visualization data: {debug_npz_path}")
                    print(f"   ({len(all_debug_data)} pairs)")
                    if not has_correspondence:
                        print(f"   ⚠️  Correspondence data skipped (non-voxelized model - saves VRAM)")

                    # Free memory after successful save (debug data can be >100MB per pair)
                    del save_dict
                    del all_debug_data
                    import gc
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    print(f"   Freed debug data from memory")

                except Exception as e:
                    print(f"  ❌ Error saving debug visualization data: {e}")
                    import traceback
                    traceback.print_exc()

            # Clear debug_data from results to free memory (whether save succeeded or not)
            for r in results:
                if 'debug_data' in r:
                    r['debug_data'] = None

        return results

    def _evaluate_single_pair(
        self,
        pair: Dict,
        mask_cache: Optional[Dict],
        viz_dir: Optional[Path],
        pair_idx: int,
        debug_viz_dir: Optional[Path] = None
    ) -> Dict:
        """Evaluate a single anchor-query pair."""
        anchor_idx = pair['anchor_frame']
        query_idx = pair['query_frame']
        object_name = pair['object_name']

        # Load anchor and query frames
        print("\nLoading frames...")
        anchor_data = self.dataset.load_frame(anchor_idx, object_name, mask_cache=mask_cache)
        query_data = self.dataset.load_frame(query_idx, object_name, mask_cache=mask_cache)

        # Scene indices are already correct (match directory names), no offset needed
        print(f"  Anchor: scene{anchor_data['frame_info']['scene']:02d}, "
              f"frame {anchor_data['frame_info']['frame_num']}")
        print(f"  Query:  scene{query_data['frame_info']['scene']:02d}, "
              f"frame {query_data['frame_info']['frame_num']}")

        # Build reference model
        print("\nBuilding reference model...")

        # Print object and category information
        print(f"  Object: {object_name}", end='')
        if 'category' in pair:
            print(f" (category: {pair['category']})")
        elif 'bop_name' in anchor_data['frame_info']:
            print(f" (category: {anchor_data['frame_info']['bop_name']})")
        elif 'category_name' in anchor_data['frame_info']:
            print(f" (category: {anchor_data['frame_info']['category_name']})")
        else:
            print()

        # Extract labels for this specific object (if dict format)
        if isinstance(self.semantic_labels, dict):
            object_labels = self.semantic_labels[object_name]
        else:
            # Backward compatibility: single list for all objects
            object_labels = self.semantic_labels

        # Generate visualization path if viz_dir is provided
        viz_building_path = None
        if viz_dir is not None:
            viz_building_path = str(viz_dir / f"building_{pair_idx:04d}_{object_name}.png")

        # Start timing pose estimation (includes saliency extraction, excludes metrics)
        import time
        pose_estimation_start = time.time()

        self.estimator.build_reference_model(
            ref_rgb=anchor_data['rgb'],
            ref_mask=anchor_data['mask'],
            ref_depth=anchor_data['depth'],
            ref_pose=anchor_data['pose'],
            K_ref=anchor_data['K'],
            semantic_labels=object_labels,
            visualize_building=(viz_dir is not None),
            viz_save_path=viz_building_path,
            anchor_pose_mode=self.anchor_pose_mode
        )

        # Estimate pose
        print("\nEstimating pose...")

        # Generate visualization path for query frame if viz_dir is provided
        viz_query_path = None
        if viz_dir is not None:
            viz_query_path = str(viz_dir / f"query_{pair_idx:04d}_{object_name}.png")

        R_est, t_est, info = self.estimator.estimate_pose(
            est_rgb=query_data['rgb'],
            est_mask=query_data['mask'],
            est_depth=query_data['depth'],
            K_est=query_data['K'],
            visualize_query=(viz_dir is not None),
            viz_save_path=viz_query_path,
            return_debug_info=(debug_viz_dir is not None)
        )

        # Stop timing pose estimation
        pose_estimation_time = time.time() - pose_estimation_start
        print(f"\n  Pose estimation time: {pose_estimation_time:.2f}s")

        # Prepare debug visualization data if requested (will be saved later)
        debug_data = None
        if debug_viz_dir is not None and 'debug_info' in info and info['debug_info'] is not None:
            import numpy as np

            debug_info = info['debug_info']
            semantic_labels = self.estimator.semantic_labels if hasattr(self.estimator, 'semantic_labels') else []

            # Prepare GT poses as 4x4 matrices
            ref_pose = anchor_data['pose'] if isinstance(anchor_data['pose'], np.ndarray) else np.vstack([
                np.hstack([anchor_data['pose']['R'], anchor_data['pose']['t'].reshape(3, 1)]),
                [0, 0, 0, 1]
            ])
            query_pose = query_data['pose'] if isinstance(query_data['pose'], np.ndarray) else np.vstack([
                np.hstack([query_data['pose']['R'], query_data['pose']['t'].reshape(3, 1)]),
                [0, 0, 0, 1]
            ])

            # Compute bounding boxes from masks
            def get_bbox_from_mask(mask):
                """Get bbox [x_min, y_min, x_max, y_max] from binary mask."""
                if mask is None:
                    return None
                y_coords, x_coords = np.where(mask > 0)
                if len(x_coords) == 0:
                    return None
                return np.array([int(x_coords.min()), int(y_coords.min()),
                               int(x_coords.max()), int(y_coords.max())])

            bbox_ref = get_bbox_from_mask(debug_info['mask_ref']) if debug_info['mask_ref'] is not None else None
            bbox_query = get_bbox_from_mask(query_data['mask'])

            # Project correspondence 3D model points to 2D reference pixels (skip for non-voxelized)
            corr_pixel_coords_ref = None
            if (debug_info['correspondences']['model_3d'] is not None and
                len(debug_info['correspondences']['model_3d']) > 0):
                model_3d_points = debug_info['correspondences']['model_3d']  # (N, 3) in ref camera space
                # Project: pixel = K @ point_3d
                K_ref = anchor_data['K']
                # Ensure 3D points are (N, 3) and K is (3, 3)
                if isinstance(model_3d_points, np.ndarray) and model_3d_points.ndim == 2:
                    # (N, 3) @ (3, 3).T = (N, 3)
                    projected = model_3d_points @ K_ref.T
                    # Normalize by depth (z coordinate)
                    depths = projected[:, 2:3]  # (N, 1)
                    # Avoid division by zero
                    valid_depth = np.abs(depths) > 1e-6
                    pixel_coords = np.zeros((len(model_3d_points), 2), dtype=np.float32)
                    pixel_coords[valid_depth.squeeze()] = (projected[valid_depth.squeeze(), :2] /
                                                           depths[valid_depth.squeeze()])
                    corr_pixel_coords_ref = pixel_coords  # (N, 2) [x, y]

            debug_data = {
                # Metadata
                'pair_idx': pair_idx,
                'object_name': object_name,
                'ref_frame_id': anchor_idx,
                'query_frame_id': query_idx,
                'category': pair.get('category', ''),

                # Images
                'rgb_ref': (debug_info['rgb_ref'] * 255) if debug_info['rgb_ref'] is not None else None,
                'rgb_query': (query_data['rgb'] * 255),
                'mask_ref': debug_info['mask_ref'] if debug_info['mask_ref'] is not None else None,
                'mask_query': query_data['mask'],
                'bbox_ref': bbox_ref,
                'bbox_query': bbox_query,

                # Saliency
                'saliency_ref': debug_info['saliency_ref'] if debug_info['saliency_ref'] is not None else None,
                'saliency_query': debug_info['saliency_query'],
                'labels': np.array(semantic_labels),

                # 3D data
                'observed_3d': debug_info['observed_3d'],
                'observed_saliency': debug_info['observed_saliency'],
                'pixel_coords_query': debug_info['pixel_coords_query'],

                # Correspondences
                'corr_observed_idx': debug_info['correspondences']['observed_idx'],
                'corr_observed_3d': debug_info['correspondences']['observed_3d'],
                'corr_model_3d': debug_info['correspondences']['model_3d'],
                'corr_kl_divergence': debug_info['correspondences']['kl_divergence'],
                'corr_cosine_similarity': debug_info['correspondences']['cosine_similarity'],
                'corr_observed_saliency': debug_info['correspondences']['observed_saliency'],
                'corr_model_saliency': debug_info['correspondences']['model_saliency'],
                'corr_pixel_coords_ref': corr_pixel_coords_ref,  # 2D pixels in reference frame
                'inlier_mask': debug_info['inlier_mask'],

                # Poses and camera intrinsics
                'pose_ref_gt': ref_pose,
                'pose_query_gt': query_pose,
                'pose_query_est_R': R_est if R_est is not None else None,
                'pose_query_est_t': t_est if R_est is not None else None,
                'K_ref': anchor_data['K'],
                'K_query': query_data['K'],

                # Metrics
                'success': info['success'],
                'num_correspondences': info['num_correspondences'],
                'num_inliers': info['num_inliers'],
                'scale': info['scale'],
            }

        if info['success']:
            # Compute metrics
            # Get mesh path from dataset (dataset-specific logic)
            try:
                mesh_path = self.dataset.get_mesh_path(object_name)
            except (AttributeError, NotImplementedError):
                mesh_path = None  # Fall back to default path construction

            model_points = self.bop_evaluator.get_model_points(object_name, num_points=1000, mesh_path=mesh_path)
            diameter = self.bop_evaluator.get_model_diameter(object_name, mesh_path=mesh_path)

            # Get symmetries for this object (if available)
            symmetries = None
            if self.bop_evaluator.symmetries and object_name in self.bop_evaluator.symmetries:
                symmetries = self.bop_evaluator.symmetries[object_name]

            pose_metrics = compute_all_metrics(
                R_est, t_est,
                query_data['pose']['R'], query_data['pose']['t'],
                model_points,
                diameter=diameter,
                symmetries=symmetries
            )

            # Add pose metrics to debug_data if it exists (ensure they're numpy)
            if debug_data is not None:
                import numpy as np
                debug_data['rotation_error_deg'] = float(pose_metrics['rotation_error_deg']) if not isinstance(pose_metrics['rotation_error_deg'], (int, float)) else pose_metrics['rotation_error_deg']
                debug_data['translation_error_m'] = float(pose_metrics['translation_error_m']) if not isinstance(pose_metrics['translation_error_m'], (int, float)) else pose_metrics['translation_error_m']
                debug_data['add_error'] = float(pose_metrics['add_error']) if not isinstance(pose_metrics['add_error'], (int, float)) else pose_metrics['add_error']

            # Compute BOP metrics
            print("\nComputing BOP metrics...")
            bop_metrics = self.bop_evaluator.evaluate_frame(
                R_est, t_est,
                query_data['pose']['R'], query_data['pose']['t'],
                query_data['K'],
                image_size=(query_data['rgb'].shape[0], query_data['rgb'].shape[1]),
                object_name=object_name
            )

            print(f"\n{'='*40}")
            print("Results:")
            print(f"{'='*40}")
            print(f"  Translation error: {pose_metrics['translation_error_m']*1000:.2f} mm")
            print(f"  Rotation error: {pose_metrics['rotation_error_deg']:.2f} deg")
            print(f"  ADD: {pose_metrics['add_error']*1000:.2f} mm")
            print(f"  3D IoU: {pose_metrics['iou_3d']:.3f}")
            print(f"  VSD: {bop_metrics.get('vsd', 'N/A')}")
            print(f"  MSSD: {bop_metrics.get('mssd', 'N/A')}")
            print(f"  MSPD: {bop_metrics.get('mspd', 'N/A')}")

            # Check BOP success criteria
            # Pose success thresholds (now computed in pose_metrics)
            success_5deg2cm = pose_metrics.get('success_5deg2cm', False)
            success_10deg5cm = pose_metrics.get('success_10deg5cm', False)

            if success_5deg2cm:
                print("  ✓ Success (5deg/2cm)")
            elif success_10deg5cm:
                print("  ~ Marginal (10deg/5cm)")
            else:
                print("  ✗ Failed")

            # Compute adaptive ADD(S) - matches Any6D/Oryon convention
            has_symmetry = (self.bop_evaluator.symmetries and
                          object_name in self.bop_evaluator.symmetries and
                          len(self.bop_evaluator.symmetries[object_name]) > 1)
            adds_adaptive_score = (pose_metrics.get('adds_score', False) if has_symmetry
                                  else pose_metrics.get('add_score', False))

            # Visualize if requested
            if viz_dir is not None:
                print("\nGenerating visualization...")

                # Create the pair visualization (GT vs Estimated)
                pair_viz_path = self._visualize_result(
                    anchor_data, query_data, R_est, t_est, info.get('scale', 1.0),
                    viz_dir, pair_idx, object_name
                )

                # Combine building, query, and pair visualizations into one image
                # (building and query were already saved during model building and pose estimation)
                building_path = viz_dir / f"building_{pair_idx:04d}_{object_name}.png"
                query_path = viz_dir / f"query_{pair_idx:04d}_{object_name}.png"

                # Check if individual visualizations exist before combining
                if building_path.exists() and query_path.exists():
                    combined_path = viz_dir / f"pair_{pair_idx:04d}_{object_name}.png"
                    self._combine_visualizations(
                        building_path, query_path, pair_viz_path, combined_path
                    )

                    # Clean up individual visualization files
                    building_path.unlink()  # Delete anchor building viz
                    query_path.unlink()     # Delete query building viz
                    # pair_viz_path is overwritten by combined_path (same name)

                    print(f"  Saved combined visualization to: {combined_path}")
                else:
                    # Fallback: just use pair visualization if building/query don't exist
                    print(f"  Saved to: {pair_viz_path}")

            # Extract query frame info for occlusion analysis (if available)
            query_info = {
                'scene': query_data['frame_info'].get('scene'),
                'frame_num': query_data['frame_info'].get('frame_num'),
                'visib_fract': query_data['frame_info'].get('visib_fract', 1.0),
                'px_count_visib': query_data['frame_info'].get('px_count_visib', 0),
                'px_count_all': query_data['frame_info'].get('px_count_all', 0),
            }

            return {
                'pair_idx': pair_idx,
                'anchor_frame': anchor_idx,
                'query_frame': query_idx,
                'object_name': object_name,
                'category': pair['category'],
                'same_scene': pair['metadata']['same_scene'],
                'success': True,
                'pose_estimation_time': pose_estimation_time,  # Time for build + estimate (excludes metrics)
                'rotation_error_deg': pose_metrics['rotation_error_deg'],
                'translation_error_m': pose_metrics['translation_error_m'],
                'add_error': pose_metrics['add_error'],
                'adds_error': pose_metrics.get('adds_error', pose_metrics['add_error']),  # ADD-S error
                'add_score': pose_metrics.get('add_score', False),  # ADD-10
                'adds_score': pose_metrics.get('adds_score', False),  # ADD-S-10
                'adds_adaptive_score': adds_adaptive_score,  # ADD(S)-10 adaptive
                'iou_3d': pose_metrics['iou_3d'],
                'iou_3d_50': pose_metrics.get('iou_3d_50', False),
                'debug_data': debug_data,  # Kept for .npz saving, filtered out before JSON save
                'iou_3d_75': pose_metrics.get('iou_3d_75', False),
                'vsd': bop_metrics.get('vsd'),
                'mssd': bop_metrics.get('mssd'),
                'mspd': bop_metrics.get('mspd'),
                'success_5deg2cm': success_5deg2cm,
                'success_5deg5cm': pose_metrics.get('success_5deg5cm', False),
                'success_10deg5cm': success_10deg5cm,
                'success_10deg10cm': pose_metrics.get('success_10deg10cm', False),
                'query_info': query_info  # Occlusion metadata for analysis
            }
        else:
            print(f"\nFailed: {info.get('error', 'Unknown error')}")

            # Extract query frame info even for failed cases
            query_info = {
                'scene': query_data['frame_info'].get('scene'),
                'frame_num': query_data['frame_info'].get('frame_num'),
                'visib_fract': query_data['frame_info'].get('visib_fract', 1.0),
                'px_count_visib': query_data['frame_info'].get('px_count_visib', 0),
                'px_count_all': query_data['frame_info'].get('px_count_all', 0),
            }

            return {
                'pair_idx': pair_idx,
                'anchor_frame': anchor_idx,
                'query_frame': query_idx,
                'object_name': object_name,
                'category': pair['category'],
                'same_scene': pair['metadata']['same_scene'],
                'success': False,
                'pose_estimation_time': pose_estimation_time,  # Time for build + estimate (even if failed)
                'error': info.get('error'),
                'query_info': query_info  # Occlusion metadata for analysis
            }

    def _combine_visualizations(
        self,
        building_path: Path,
        query_path: Path,
        pair_path: Path,
        output_path: Path
    ) -> None:
        """
        Combine three visualization images into one.

        Layout:
        - Top row: 4 anchor panels (from building visualization)
        - Middle row: 4 query panels (from query visualization)
        - Bottom row: 2 pair panels (from pair visualization)

        Args:
            building_path: Path to anchor building visualization
            query_path: Path to query building visualization
            pair_path: Path to pair visualization (GT vs Est)
            output_path: Path to save combined visualization

        Note:
            Delegates to BaseEvaluator._combine_images_vertically()
        """
        self._combine_images_vertically([building_path, query_path, pair_path], output_path, dpi=150)

    def _visualize_result(
        self,
        anchor_data: Dict,
        query_data: Dict,
        R_est: np.ndarray,
        t_est: np.ndarray,
        scale_est: float,
        viz_dir: Path,
        pair_idx: int,
        object_name: str
    ) -> Path:
        """Generate visualization of pose estimation result."""
        # Get visualization points (mesh or voxels)
        if self.use_mesh_visualization:
            # Load and sample mesh points (in object space)
            try:
                mesh_path = self.dataset.get_mesh_path(object_name)
            except (AttributeError, NotImplementedError):
                mesh_path = None

            # Sample ~2000 points from mesh surface (more detail than typical voxel grid)
            mesh_points_obj = self.bop_evaluator.get_model_points(
                object_name, num_points=2000, mesh_path=mesh_path
            )

            if mesh_points_obj is None:
                # Fallback to voxels if mesh not found
                print(f"  Warning: Mesh not found for {object_name}, using voxel points")
                use_mesh = False
            else:
                use_mesh = True
                # Mesh points are already in object space - use directly
                mesh_center = mesh_points_obj.mean(axis=0)
        else:
            use_mesh = False

        # Get poses first (needed for coordinate transformation in relative mode)
        R_anchor = anchor_data['pose']['R']
        t_anchor = anchor_data['pose']['t']
        R_query_gt = query_data['pose']['R']
        t_query_gt = query_data['pose']['t']

        # Get points in object space
        if use_mesh:
            # Mesh: already in object space, use directly
            voxel_obj_anchor = mesh_points_obj
            voxel_obj_est = mesh_points_obj * scale_est  # Apply estimated scale
            object_center = mesh_center
        else:
            # Voxels: denormalize from NOCS space first
            voxel_nocs = self.estimator.points_3d_nocs
            voxel_denorm_anchor = voxel_nocs * self.estimator.nocs_scale + self.estimator.nocs_centroid
            voxel_denorm_est = voxel_nocs * (self.estimator.nocs_scale * scale_est) + self.estimator.nocs_centroid
            object_center_denorm = self.estimator.nocs_centroid

            # In relative mode, denormalized points are in anchor camera space
            # Transform to object space for visualization
            if self.anchor_pose_mode == 'relative':
                # Transform from anchor camera space to object space
                # obj_pts = R_anchor.T @ (cam_pts - t_anchor)
                voxel_obj_anchor = (R_anchor.T @ (voxel_denorm_anchor - t_anchor).T).T
                voxel_obj_est = (R_anchor.T @ (voxel_denorm_est - t_anchor).T).T
                object_center = R_anchor.T @ (object_center_denorm - t_anchor)
            else:
                # In absolute mode, denormalized points are already in object space
                voxel_obj_anchor = voxel_denorm_anchor
                voxel_obj_est = voxel_denorm_est
                object_center = object_center_denorm
        K_anchor = anchor_data['K']
        K_query = query_data['K']

        # Get RGB images
        rgb_anchor = anchor_data['rgb']
        rgb_query = query_data['rgb']
        H, W = rgb_anchor.shape[:2]

        # Project voxels
        # For anchor and GT query: use anchor frame scale
        pixels_anchor, valid_anchor = project_points_to_image(voxel_obj_anchor, R_anchor, t_anchor, K_anchor)
        pixels_query_gt, valid_query_gt = project_points_to_image(voxel_obj_anchor, R_query_gt, t_query_gt, K_query)
        # For estimated query: use scaled point cloud
        pixels_query_est, valid_query_est = project_points_to_image(voxel_obj_est, R_est, t_est, K_query)

        # Filter valid points
        def in_bounds(pixels, H, W):
            return (pixels[:, 0] >= 0) & (pixels[:, 0] < W) & \
                   (pixels[:, 1] >= 0) & (pixels[:, 1] < H)

        valid_anchor = valid_anchor & in_bounds(pixels_anchor, H, W)
        valid_query_gt = valid_query_gt & in_bounds(pixels_query_gt, H, W)
        valid_query_est = valid_query_est & in_bounds(pixels_query_est, H, W)

        # Create figure
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))

        # Left panel: Anchor
        axes[0].imshow(rgb_anchor)
        if np.any(valid_anchor):
            axes[0].scatter(
                pixels_anchor[valid_anchor, 0],
                pixels_anchor[valid_anchor, 1],
                c='lime', s=2, alpha=0.6, label='GT Voxels'
            )
        draw_axes_3d(axes[0], R_anchor, t_anchor, K_anchor, object_center, scale=0.1, linewidth=3, alpha=1.0)
        axes[0].set_title(f'Anchor (GT)\n{object_name}\nFrame: {anchor_data["frame_info"]["frame_num"]}', fontsize=12)
        axes[0].axis('off')
        axes[0].legend(loc='upper right', fontsize=8)

        # Right panel: Query
        axes[1].imshow(rgb_query)
        if np.any(valid_query_gt):
            axes[1].scatter(
                pixels_query_gt[valid_query_gt, 0],
                pixels_query_gt[valid_query_gt, 1],
                c='lime', s=2, alpha=0.6, label='GT Voxels'
            )
        if np.any(valid_query_est):
            axes[1].scatter(
                pixels_query_est[valid_query_est, 0],
                pixels_query_est[valid_query_est, 1],
                c='red', s=2, alpha=0.6, label='Est Voxels'
            )
        draw_axes_3d(axes[1], R_query_gt, t_query_gt, K_query, object_center, scale=0.1, linewidth=3, alpha=1.0)
        draw_axes_3d(axes[1], R_est, t_est, K_query, object_center, scale=0.1, linewidth=4, alpha=0.7)
        axes[1].set_title(f'Query (GT=solid, Est=transparent)\n{object_name}\nFrame: {query_data["frame_info"]["frame_num"]}', fontsize=12)
        axes[1].axis('off')
        axes[1].legend(loc='upper right', fontsize=8)

        plt.tight_layout()

        # Save
        viz_dir.mkdir(parents=True, exist_ok=True)
        save_path = viz_dir / f'pair_{pair_idx:04d}_{object_name}.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

        return save_path

    @staticmethod
    def get_results_summary(results: List[Dict]) -> Dict:
        """
        Compute summary statistics from results.

        Args:
            results: List of result dicts

        Returns:
            Dict with summary statistics

        Note:
            Delegates to BaseEvaluator.get_results_summary() with 'total_pairs' key
        """
        return BaseEvaluator.get_results_summary(results, total_key='total_pairs')
