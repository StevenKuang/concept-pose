"""
One-Shot 6D Pose Estimation
============================

A modular, dataset-agnostic wrapper for one-shot 6D pose estimation using
semantic voxel models and 3D-3D registration.

This module provides a clean interface for:
1. Building a voxel model from a single reference RGBD frame
2. Estimating pose on target frames using semantic 3D-3D matching

Key Features:
- Dataset-agnostic (no HouseCat6D dependencies)
- Modular design (easy to swap components)
- Self-contained pipeline
- On-the-fly model building

Example Usage:
--------------
```python
from concept_pose.pose.one_shot_estimator import OneShotPoseEstimator

# Initialize estimator
estimator = OneShotPoseEstimator(
    voxel_resolution=64,
    ransac_iterations=50000,
    device='cuda'
)

# Build reference model from one RGBD frame
estimator.build_reference_model(
    ref_rgb=ref_rgb,           # (H, W, 3) RGB image
    ref_mask=ref_mask,         # (H, W) binary mask
    ref_depth=ref_depth,       # (H, W) depth in meters
    ref_pose={'R': R, 't': t}, # Reference pose
    K_ref=K_ref,               # (3, 3) camera intrinsics
    semantic_labels=['handle', 'rim', 'body', ...]
)

# Estimate pose on target frame
R_est, t_est, info = estimator.estimate_pose(
    est_rgb=target_rgb,
    est_mask=target_mask,
    est_depth=target_depth,
    K_est=K_target
)

print(f"Estimated R:\\n{R_est}")
print(f"Estimated t: {t_est}")
print(f"Inliers: {info['num_inliers']}/{info['num_correspondences']}")
```
"""

import numpy as np
import torch
import cv2
from typing import Dict, List, Tuple, Optional, Union

# Import core voxelization and normalization functions
from concept_pose.core import (
    voxelize_points,
    normalize_to_nocs,
    backproject_to_3d,
    filter_statistical_outliers
)

# Import 3D registration functions
from concept_pose.pose import registration_3d

# Import saliency generator
from concept_pose.saliency import SigLIP2SaliencyGenerator


class OneShotPoseEstimator:
    """
    One-shot 6D pose estimator using semantic voxel models.

    This class provides a self-contained pipeline for building a voxel model
    from a single reference RGBD frame and estimating pose on target frames.

    The pipeline uses:
    - Semantic saliency maps (SigLIP2) for part-based features
    - Voxelized NOCS representation for the reference model
    - 3D-3D registration (RANSAC + Umeyama + ICP) for pose estimation

    Attributes:
        voxel_resolution (int): Resolution of voxel grid (e.g., 64)
        device (str): Torch device ('cuda' or 'cpu')
        config (dict): Configuration parameters for registration

        # Model data (populated after build_reference_model)
        voxel_grid (np.ndarray): (R, R, R, C) voxelized saliency model
        valid_voxels (np.ndarray): (R, R, R) boolean mask of occupied voxels
        points_3d_nocs (np.ndarray): (N, 3) NOCS-normalized voxel centers
        saliencies_3d (np.ndarray): (N, C) semantic features per voxel
        nocs_scale (float): Scale factor for NOCS normalization
        nocs_centroid (np.ndarray): (3,) centroid for NOCS normalization
        semantic_labels (list): List of semantic part labels
    """

    def __init__(
        self,
        voxel_resolution: int = 64,
        ransac_iterations: int = 50000,
        ransac_threshold: float = 0.01,
        similarity_threshold: float = 0.6,
        max_correspondences: int = 500,
        use_icp: bool = True,
        icp_max_iters: int = 50,
        estimate_scale: bool = False,
        aggregation_method: str = 'mean',
        loss_method: str = 'kl_divergence',
        voxelize_anchor: bool = True,
        voxelize_query: bool = False,
        use_gpu_ransac: bool = True,
        ransac_batch_size: int = 1024,
        temperature: float = 1.0,
        lambda_reverse: float = 0.5,
        device: str = 'cuda',
        saliency_method: str = 'siglip',
        saliency_generator: Optional[SigLIP2SaliencyGenerator] = None,
        binarize_saliency: bool = False,
        binarize_threshold: float = 0.5
    ):
        """
        Initialize the one-shot pose estimator.

        Args:
            voxel_resolution: Voxel grid resolution (e.g., 64)
            ransac_iterations: Number of RANSAC iterations
            ransac_threshold: RANSAC inlier threshold in meters (e.g., 0.01 = 1cm)
            similarity_threshold: Semantic similarity threshold for matching
            max_correspondences: Maximum number of correspondences to find
            use_icp: Whether to use ICP refinement
            icp_max_iters: Maximum ICP iterations
            estimate_scale: Whether to estimate scale (False = use NOCS scale)
            aggregation_method: Voxel aggregation method ('mean' or 'max')
            loss_method: Semantic matching method ('kl_divergence', 'cosine', 'asymmetric')
            voxelize_anchor: Whether to voxelize anchor frame (default: True, disable for ablation studies)
            voxelize_query: Whether to voxelize query frame (default: False, enable for noisy/dense scenes)
            use_gpu_ransac: Use GPU-accelerated batched RANSAC (default: True, 50-100x faster, RECOMMENDED)
            ransac_batch_size: Batch size for GPU RANSAC (default: 1024, tune for GPU memory)
            temperature: Temperature for KL-based correspondence methods (default: 1.0)
            lambda_reverse: Weight for reverse KL in bidirectional method (default: 0.5)
            device: Torch device ('cuda' or 'cpu')
            saliency_method: Saliency generation method ('siglip' or 'clip')
            saliency_generator: Pre-initialized saliency generator (optional)
            binarize_saliency: If True, binarize saliency maps (ablation study)
            binarize_threshold: Threshold for binarization (default: 0.5)
        """
        self.voxel_resolution = voxel_resolution
        self.device = device
        self.aggregation_method = aggregation_method
        self.voxelize_anchor = voxelize_anchor
        self.voxelize_query = voxelize_query
        self.saliency_method = saliency_method
        self.binarize_saliency = binarize_saliency
        self.binarize_threshold = binarize_threshold

        # Configuration for 3D registration
        self.config = {
            'ransac_iterations_3d': ransac_iterations,
            'ransac_3d_threshold': ransac_threshold,
            'similarity_threshold': similarity_threshold,
            'max_correspondences': max_correspondences,
            'use_icp_3d': use_icp,
            'icp_max_iters': icp_max_iters,
            'estimate_scale': estimate_scale,
            'icp_convergence': 0.0001,
            'icp_distance_threshold': 0.02,
            'loss_method': loss_method,
            'voxelize_anchor': voxelize_anchor,
            'voxelize_query': voxelize_query,
            'use_gpu_ransac': use_gpu_ransac,
            'ransac_batch_size': ransac_batch_size,
            'temperature': temperature,
            'lambda_reverse': lambda_reverse
        }

        # Model data (populated by build_reference_model)
        self.voxel_grid = None
        self.valid_voxels = None
        self.points_3d_nocs = None
        self.saliencies_3d = None
        self.nocs_scale = None
        self.nocs_centroid = None
        self.semantic_labels = None

        # External saliency generator (optional, for reuse across multiple estimators)
        self._external_saliency_generator = saliency_generator
        self._internal_saliency_generator = None

    def _get_saliency_generator(self, semantic_labels: List[str]) -> SigLIP2SaliencyGenerator:
        """
        Get or create a saliency generator, updating labels if they changed.

        The generator is created once and cached. When labels change, we call
        set_labels() to update them without recreating the expensive model.

        Args:
            semantic_labels: List of semantic part labels

        Returns:
            SigLIP2SaliencyGenerator instance with updated labels
        """
        if self._external_saliency_generator is not None:
            # External generator provided - update its labels
            self._external_saliency_generator.set_labels(semantic_labels)
            return self._external_saliency_generator

        if self._internal_saliency_generator is None:
            # Create generator for the first time
            import torch
            device = self.device if torch.cuda.is_available() else None

            if self.saliency_method == 'clip':
                # Use CLIP generator
                from concept_pose.saliency import CLIPGradCAMGenerator
                if device:
                    self._internal_saliency_generator = CLIPGradCAMGenerator(
                        semantic_labels,
                        custom_device=device
                    )
                else:
                    self._internal_saliency_generator = CLIPGradCAMGenerator(semantic_labels)
                print(f"Created CLIP saliency generator")
            elif self.saliency_method == 'dinotxt':
                # Use DinoTxt generator
                from concept_pose.saliency import DinoTxtGradCAMGenerator
                if device:
                    self._internal_saliency_generator = DinoTxtGradCAMGenerator(
                        semantic_labels,
                        custom_device=device
                    )
                else:
                    self._internal_saliency_generator = DinoTxtGradCAMGenerator(semantic_labels)
                print(f"Created DinoTxt saliency generator")
            else:
                # Use SigLIP generator (default)
                if device:
                    self._internal_saliency_generator = SigLIP2SaliencyGenerator(
                        semantic_labels,
                        custom_device=device
                    )
                else:
                    self._internal_saliency_generator = SigLIP2SaliencyGenerator(semantic_labels)
                print(f"Created SigLIP2 saliency generator")
        else:
            # Generator exists - update labels (cheap operation if unchanged)
            self._internal_saliency_generator.set_labels(semantic_labels)

        return self._internal_saliency_generator

    def _cleanup_saliency_generator(self):
        """Clean up internal saliency generator if it exists."""
        if self._internal_saliency_generator is not None:
            self._internal_saliency_generator.cleanup()
            self._internal_saliency_generator = None

    def _preprocess_rgb(self, rgb: np.ndarray) -> np.ndarray:
        """
        Preprocess RGB image to standard format.

        Args:
            rgb: (H, W, 3) RGB image, uint8 [0-255] or float32 [0-1]

        Returns:
            (H, W, 3) RGB image as float32 [0-1]
        """
        if rgb.dtype == np.uint8:
            rgb = rgb.astype(np.float32) / 255.0
        elif rgb.dtype != np.float32:
            rgb = rgb.astype(np.float32)

        # Ensure [0, 1] range
        if rgb.max() > 1.0:
            rgb = rgb / 255.0

        return rgb

    def _preprocess_mask(self, mask: np.ndarray) -> np.ndarray:
        """
        Preprocess mask to binary format.

        Args:
            mask: (H, W) mask, any numeric type

        Returns:
            (H, W) binary mask as bool
        """
        if mask.dtype != bool:
            mask = mask > 0.5
        return mask

    def _preprocess_depth(self, depth: np.ndarray) -> np.ndarray:
        """
        Preprocess depth map to standard format.

        Args:
            depth: (H, W) depth map

        Returns:
            (H, W) depth map as float32 in meters
        """
        if depth.dtype != np.float32:
            depth = depth.astype(np.float32)
        return depth

    def _parse_camera_matrix(self, K: Union[np.ndarray, Dict]) -> np.ndarray:
        """
        Parse camera intrinsics to 3x3 matrix.

        Args:
            K: Either (3, 3) matrix or dict with {fx, fy, cx, cy}

        Returns:
            (3, 3) camera intrinsics matrix
        """
        if isinstance(K, dict):
            fx = K['fx']
            fy = K['fy']
            cx = K['cx']
            cy = K['cy']
            K_matrix = np.array([
                [fx, 0, cx],
                [0, fy, cy],
                [0, 0, 1]
            ], dtype=np.float32)
            return K_matrix
        else:
            return K.astype(np.float32)

    def _parse_pose(self, pose: Union[np.ndarray, Dict]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Parse pose to (R, t) tuple.

        NOTE: Assumes R is already a pure rotation matrix (det=1, orthogonal).
        Dataset-specific quirks (e.g., scale in R for NOCS/Real275) should be
        handled by the dataset class before returning poses.

        Args:
            pose: Either (4, 4) transformation matrix or dict with {'R': (3,3), 't': (3,)}

        Returns:
            R: (3, 3) rotation matrix
            t: (3,) translation vector
        """
        if isinstance(pose, dict):
            R = pose['R'].astype(np.float32)
            t = pose['t'].astype(np.float32)
        else:
            # Assume 4x4 matrix
            R = pose[:3, :3].astype(np.float32)
            t = pose[:3, 3].astype(np.float32)

        return R, t

    def _extract_saliency(
        self,
        rgb: np.ndarray,
        mask: np.ndarray,
        semantic_labels: List[str]
    ) -> np.ndarray:
        """
        Extract semantic saliency map from RGB image.

        IMPORTANT: This function crops to object bbox before SigLIP extraction,
        matching the exact workflow from dataset_housecat.py!

        Workflow (matches dataset_housecat.py extract_saliency_maps with crop_object=True):
        1. Compute bbox from mask using masks_to_bboxes()
        2. Crop RGB to object bbox
        3. Pass cropped region to SigLIP
        4. Use pad_and_resize_saliency_map() to place saliency back in full image

        Args:
            rgb: (H, W, 3) RGB image [0-1], may be padded
            mask: (H, W) binary mask, same size as rgb
            semantic_labels: List of semantic part labels

        Returns:
            (C, H, W) saliency map where C = len(semantic_labels)
        """
        from PIL import Image
        from concept_pose.utils import masks_to_bboxes, pad_and_resize_saliency_map

        generator = self._get_saliency_generator(semantic_labels)

        H, W = mask.shape

        # # === ABLATION: Apply mask before cropping ===
        # rgb = rgb * mask[:, :, np.newaxis]
        # # === END ABLATION ===

        # Step 1: Compute bbox from mask (same as dataset_housecat.py)
        mask_tensor = torch.from_numpy(mask).unsqueeze(0)  # (1, H, W)
        bboxes = masks_to_bboxes(mask_tensor)  # (1, 4) [x1, y1, x2, y2]
        bbox = bboxes[0].numpy()

        # Check for valid bbox
        if bbox[0] == 0 and bbox[1] == 0 and bbox[2] == 0 and bbox[3] == 0:
            # Empty mask - return zero saliency
            return np.zeros((len(semantic_labels), H, W), dtype=np.float32)

        # Step 2: Crop RGB to object bbox (same as dataset_housecat.py line 618)
        x1, y1, x2, y2 = bbox.astype(np.int32)
        rgb_cropped = rgb[y1:y2, x1:x2]

        # Step 3: Convert to PIL Image and pass to SigLIP
        rgb_uint8 = (rgb_cropped * 255).astype(np.uint8)
        pil_image = Image.fromarray(rgb_uint8)

        # Process frame with SigLIP (same as dataset_housecat.py line 627-628)
        saliency_tensor, _ = generator.process_frame(pil_image, visualize=False)

        # Handle both (1, C, H, W) and (C, H, W) formats
        # In dataset_housecat.py, process_frame returns (C, H, W) for saliency maps
        if saliency_tensor.dim() == 4:  # (1, C, H, W)
            saliency_tensor = saliency_tensor.squeeze(0)  # -> (C, H, W)
        elif saliency_tensor.dim() != 3:
            raise ValueError(f"Unexpected saliency tensor shape: {saliency_tensor.shape}")

        # Now saliency_tensor is (C, H', W') where H', W' = 384x384

        # Step 4: Resize and pad back to full image size (same as dataset_housecat.py line 631-633)
        # Pass directly like in dataset_housecat.py - no indexing needed
        saliency_resized = pad_and_resize_saliency_map(
            saliency_tensor,  # (C, H', W')
            bbox,  # [x1, y1, x2, y2]
            (H, W)  # target size
        )

        # Convert to numpy
        if isinstance(saliency_resized, torch.Tensor):
            saliency = saliency_resized.cpu().numpy()
        else:
            saliency = saliency_resized

        # Apply mask to zero out non-object regions (extra safety)
        mask_expanded = mask[np.newaxis, :, :]  # (1, H, W)
        saliency = saliency * mask_expanded

        # Ablation: Binarize saliency maps if enabled
        if self.binarize_saliency:
            saliency = (saliency > self.binarize_threshold).astype(np.float32)

        return saliency

    def _visualize_model_building(
        self,
        rgb: np.ndarray,
        mask: np.ndarray,
        depth: np.ndarray,
        saliency: np.ndarray,
        save_path: Optional[str] = None
    ):
        """
        Visualize the reference model building inputs.

        Shows RGB, mask, depth, and PCA-colored saliency map.

        Args:
            rgb: (H, W, 3) RGB image [0-1]
            mask: (H, W) binary mask
            depth: (H, W) depth map in meters
            saliency: (C, H, W) saliency map
            save_path: Optional path to save visualization
        """
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt
        from sklearn.decomposition import PCA

        fig, axes = plt.subplots(1, 4, figsize=(20, 5))

        # 1. RGB
        axes[0].imshow(rgb)
        axes[0].set_title('RGB', fontsize=14, fontweight='bold')
        axes[0].axis('off')

        # 2. Mask
        axes[1].imshow(mask, cmap='gray')
        axes[1].set_title(f'Mask ({np.sum(mask>0)} pixels)', fontsize=14, fontweight='bold')
        axes[1].axis('off')

        # 3. Depth
        depth_vis = depth.copy()
        depth_vis[depth_vis == 0] = np.nan
        im = axes[2].imshow(depth_vis, cmap='turbo')
        axes[2].set_title('Depth (meters)', fontsize=14, fontweight='bold')
        axes[2].axis('off')
        plt.colorbar(im, ax=axes[2], fraction=0.046)

        # 4. Saliency visualization using KL-divergence matching
        C, H, W = saliency.shape

        # Get mask pixels
        mask_pixels = mask > 0
        y_coords, x_coords = np.where(mask_pixels)
        saliency_vectors = saliency[:, mask_pixels].T  # (N, C)

        if len(saliency_vectors) > 0:
            # Check if we have stored anchor data for KL-based matching
            if hasattr(self, '_visualization_anchor') and self._visualization_anchor is not None:
                # Query mode: match to anchor using KL divergence
                anchor_data = self._visualization_anchor
                anchor_sal = anchor_data['saliency']  # (M, C)
                anchor_colors = anchor_data['colors']  # (M, 3)

                # Compute KL divergence: query (N) vs anchor (M)
                # KL(P||Q) = sum(P * log(P/Q))
                import torch
                import torch.nn.functional as F

                query_t = torch.from_numpy(saliency_vectors).float()  # (N, C)
                anchor_t = torch.from_numpy(anchor_sal).float()  # (M, C)

                # Apply softmax to get probability distributions
                query_probs = F.softmax(query_t, dim=-1)  # (N, C)
                anchor_probs = F.softmax(anchor_t, dim=-1)  # (M, C)

                # Compute KL divergence for all pairs (batched for memory efficiency)
                # Process in batches to avoid OOM
                batch_size = 1000
                best_matches = []

                for i in range(0, len(query_probs), batch_size):
                    q_batch = query_probs[i:i+batch_size].unsqueeze(1)  # (B, 1, C)
                    a_expanded = anchor_probs.unsqueeze(0)  # (1, M, C)

                    # KL divergence
                    kl = torch.sum(q_batch * torch.log(q_batch / (a_expanded + 1e-10) + 1e-10), dim=-1)  # (B, M)
                    best_idx = torch.argmin(kl, dim=1)  # (B,)
                    best_matches.append(best_idx.numpy())

                best_matches = np.concatenate(best_matches)

                # Assign colors from best matching anchor pixels
                pixel_colors = anchor_colors[best_matches]  # (N, 3)
                title_suffix = '\n(KL-matched to anchor)'
            else:
                # Anchor mode: color by spatial position (x->R, y->G, normalized)
                pixel_colors = np.zeros((len(saliency_vectors), 3))
                pixel_colors[:, 0] = x_coords / W  # Red = x position
                pixel_colors[:, 1] = y_coords / H  # Green = y position
                pixel_colors[:, 2] = 0.3  # Small blue for visibility

                # Store anchor data for query matching
                self._visualization_anchor = {
                    'saliency': saliency_vectors.copy(),
                    'colors': pixel_colors.copy(),
                    'coords': np.stack([x_coords, y_coords], axis=1)
                }
                title_suffix = '\n(Position-colored anchor)'

            # Create RGB image
            saliency_rgb = np.zeros((H, W, 3))
            saliency_rgb[mask_pixels] = pixel_colors

            axes[3].imshow(saliency_rgb)
            axes[3].set_title(
                f'Saliency (Semantic){title_suffix}',
                fontsize=14, fontweight='bold'
            )
        else:
            axes[3].imshow(np.zeros((H, W, 3)))
            axes[3].set_title('Saliency (No valid pixels)', fontsize=14, fontweight='bold')

        axes[3].axis('off')

        plt.suptitle('Reference Model Building Visualization', fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()

        if save_path:
            # Ensure directory exists
            from pathlib import Path
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)

            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"  Saved building visualization to: {save_path}")

        plt.close(fig)

    def _backproject_to_object_space(
        self,
        depth: np.ndarray,
        saliency: np.ndarray,
        K: np.ndarray,
        R_obj: np.ndarray,
        t_obj: np.ndarray,
        mask: Optional[np.ndarray] = None,
        return_pixel_coords: bool = False
    ):
        """
        Back-project depth map to 3D object space.

        Args:
            depth: (H, W) depth map in meters
            saliency: (C, H, W) saliency map
            K: (3, 3) camera intrinsics
            R_obj: (3, 3) object rotation (camera to object)
            t_obj: (3,) object translation
            mask: (H, W) optional binary mask (use this to determine valid pixels)
            return_pixel_coords: If True, return (points_3d, saliencies, pixel_coords)

        Returns:
            points_3d: (N, 3) 3D points in object frame
            saliencies: (N, C) semantic features
            pixel_coords: (N, 2) [x, y] pixel coordinates (if return_pixel_coords=True)
        """
        C, H, W = saliency.shape

        # Find valid pixels
        # Priority: use mask if provided, otherwise use saliency sparsity
        if mask is not None:
            valid_mask = (mask > 0) & (depth > 0)
        else:
            valid_mask = (np.any(saliency != 0, axis=0)) & (depth > 0)

        valid_coords = np.argwhere(valid_mask)  # (N, 2) [y, x]

        if len(valid_coords) == 0:
            if return_pixel_coords:
                return None, None, None
            return None, None

        # Extract pixel coordinates and depth
        pixel_coords = valid_coords[:, [1, 0]]  # (N, 2) [x, y]
        depth_values = depth[valid_coords[:, 0], valid_coords[:, 1]]  # (N,)

        # Back-project using core function (returns points in object frame)
        points_3d = backproject_to_3d(pixel_coords, depth_values, K, R_obj, t_obj)

        # Extract saliencies
        saliencies = saliency[:, valid_coords[:, 0], valid_coords[:, 1]].T  # (N, C)

        if return_pixel_coords:
            return points_3d, saliencies, pixel_coords
        return points_3d, saliencies

    def build_reference_model(
        self,
        ref_rgb: np.ndarray,
        ref_mask: np.ndarray,
        ref_depth: np.ndarray,
        ref_pose: Union[np.ndarray, Dict],
        K_ref: Union[np.ndarray, Dict],
        semantic_labels: List[str],
        filter_outliers: bool = True,
        outlier_nb_neighbors: int = 20,
        outlier_std_ratio: float = 2.0,
        visualize_building: bool = False,
        viz_save_path: Optional[str] = None,
        anchor_pose_mode: str = 'absolute'
    ):
        """
        Build voxel model from reference RGBD frame.

        Pipeline:
        1. Extract semantic saliency from ref RGB (crops to object bbox automatically)
        2. Back-project ref depth to 3D object space (or camera space if anchor_pose_mode='relative')
        3. (Optional) Filter statistical outliers
        4. Normalize to NOCS space
        5. Voxelize

        Args:
            ref_rgb: (H, W, 3) reference RGB image (may be padded)
            ref_mask: (H, W) reference mask
            ref_depth: (H, W) reference depth in meters
            ref_pose: Reference object pose (dict or 4x4 matrix)
            K_ref: Reference camera intrinsics (already adjusted for padding if applicable)
            semantic_labels: List of semantic part labels (e.g., ['handle', 'rim', ...])
            filter_outliers: Whether to filter statistical outliers
            outlier_nb_neighbors: Number of neighbors for outlier detection
            outlier_std_ratio: Std ratio threshold for outlier detection
            visualize_building: If True, visualize RGB, mask, depth, and saliency
            viz_save_path: If provided, save visualization to this path
            anchor_pose_mode: Mode for anchor pose usage:
                'absolute' (default): Use GT pose during building (anchor in object space)
                'relative': Don't use GT during building (anchor in camera space), compose later
        """
        print("\n=== Building Reference Model ===")
        print(f"Anchor pose mode: {anchor_pose_mode}")

        # Store semantic labels
        self.semantic_labels = semantic_labels

        # Preprocess inputs
        ref_rgb = self._preprocess_rgb(ref_rgb)
        ref_mask = self._preprocess_mask(ref_mask)
        ref_depth = self._preprocess_depth(ref_depth)
        K_ref = self._parse_camera_matrix(K_ref)
        R_obj, t_obj = self._parse_pose(ref_pose)

        # Store anchor pose mode and GT pose (for composition later if needed)
        self._anchor_pose_mode = anchor_pose_mode
        self._anchor_R = R_obj
        self._anchor_t = t_obj

        # Reset visualization data for consistent coloring between anchor and query
        self._visualization_pca = None
        self._visualization_anchor = None

        print(f"Reference image size: {ref_rgb.shape[:2]}")
        print(f"Semantic labels: {semantic_labels}")

        # Step 1: Extract semantic saliency
        print("\n[1/5] Extracting semantic saliency...")
        saliency = self._extract_saliency(ref_rgb, ref_mask, semantic_labels)
        print(f"  Saliency shape: {saliency.shape}")

        # Visualize building process if requested
        if visualize_building:
            self._visualize_model_building(
                ref_rgb, ref_mask, ref_depth, saliency, viz_save_path
            )

        # Step 2: Back-project to 3D space
        if anchor_pose_mode == 'absolute':
            print("\n[2/5] Back-projecting to 3D object space...")
            points_3d, saliencies_3d, ref_pixel_coords = self._backproject_to_object_space(
                ref_depth, saliency, K_ref, R_obj, t_obj, ref_mask, return_pixel_coords=True
            )
        else:  # 'relative'
            print("\n[2/5] Back-projecting to 3D camera space (no GT transform)...")
            points_3d, saliencies_3d, ref_pixel_coords = self._backproject_to_object_space(
                ref_depth, saliency, K_ref, None, None, ref_mask, return_pixel_coords=True
            )

        if points_3d is None or len(points_3d) == 0:
            raise ValueError("Failed to back-project points. Check depth and mask.")

        print(f"  Generated {len(points_3d)} 3D points in object space")

        # # Check object-space coordinates
        # extent_obj_x = points_3d[:, 0].max() - points_3d[:, 0].min()
        # extent_obj_y = points_3d[:, 1].max() - points_3d[:, 1].min()
        # extent_obj_z = points_3d[:, 2].max() - points_3d[:, 2].min()
        # print(f"  Object-space point cloud stats:")
        # print(f"    X range: [{points_3d[:, 0].min():.4f}, {points_3d[:, 0].max():.4f}] m (extent: {extent_obj_x*1000:.1f} mm)")
        # print(f"    Y range: [{points_3d[:, 1].min():.4f}, {points_3d[:, 1].max():.4f}] m (extent: {extent_obj_y*1000:.1f} mm)")
        # print(f"    Z range: [{points_3d[:, 2].min():.4f}, {points_3d[:, 2].max():.4f}] m (extent: {extent_obj_z*1000:.1f} mm)")

        # Step 3: Filter outliers (optional)
        # Two-stage filtering: local (statistical) + global (distance from center)
        if filter_outliers:
            print("\n[3/5] Filtering outliers (two-stage)...")

            # Stage 1: Local statistical outlier removal
            print("  Stage 1: Local statistical filtering...")
            points_3d, saliencies_3d, _ = filter_statistical_outliers(
                points_3d, saliencies_3d,
                nb_neighbors=outlier_nb_neighbors,
                std_ratio=outlier_std_ratio
            )
            print(f"    After local filtering: {len(points_3d)} points")

            # Stage 2: Global outlier removal (distance from center of mass)
            # Remove points more than 2.5 std deviations from center
            print("  Stage 2: Global distance filtering...")
            center = points_3d.mean(axis=0)
            distances = np.linalg.norm(points_3d - center, axis=1)
            mean_dist = distances.mean()
            std_dist = distances.std()
            global_threshold = mean_dist + 2.5 * std_dist

            global_inliers = distances < global_threshold
            n_removed_global = len(points_3d) - np.sum(global_inliers)

            if n_removed_global > 0:
                points_3d = points_3d[global_inliers]
                saliencies_3d = saliencies_3d[global_inliers]
                print(f"    Removed {n_removed_global} global outliers (>{global_threshold:.4f}m from center)")

            print(f"  After filtering: {len(points_3d)} points remain")
        else:
            print("\n[3/5] Skipping outlier filtering")

        # Step 4: Normalize to NOCS space
        print("\n[4/5] Normalizing to NOCS space...")

        # Compute extents for diagnostic
        extent_x = points_3d[:, 0].max() - points_3d[:, 0].min()
        extent_y = points_3d[:, 1].max() - points_3d[:, 1].min()
        extent_z = points_3d[:, 2].max() - points_3d[:, 2].min()
        max_extent = max(extent_x, extent_y, extent_z)

        # Choose scale based on coordinate frame
        if anchor_pose_mode == 'absolute':
            # Object-centric coordinates: use max absolute coordinate (original behavior)
            point_cloud_extent = np.abs(points_3d).max()
            nocs_scale = 2 * point_cloud_extent
        else:
            # Camera-space coordinates: use max extent * 2 (robust to position)
            # This measures the largest dimension of the bounding box
            nocs_scale = 2 * max_extent

        points_3d_nocs, nocs_centroid, nocs_scale_out = normalize_to_nocs(
            points_3d, nocs_scale, anchor_pose_mode=anchor_pose_mode
        )

        self.nocs_centroid = nocs_centroid
        self.nocs_scale = nocs_scale_out

        # Extents already computed above, just print diagnostics
        print(f"  NOCS scale: {self.nocs_scale:.6f} m = {self.nocs_scale*1000:.2f} mm")
        print(f"  NOCS centroid: {self.nocs_centroid}")
        print(f"  NOCS range: [{points_3d_nocs.min():.3f}, {points_3d_nocs.max():.3f}]")
        print(f"  Visible extent (X,Y,Z): ({extent_x*1000:.1f}, {extent_y*1000:.1f}, {extent_z*1000:.1f}) mm")
        print(f"  Max visible extent: {max_extent*1000:.1f} mm")

        # Step 5: Voxelize (optional, based on voxelize_anchor parameter)
        if self.voxelize_anchor:
            print(f"\n[5/5] Voxelizing to {self.voxel_resolution}³ grid...")
            voxel_grid, voxel_counts, valid_voxels = voxelize_points(
                points_3d_nocs, saliencies_3d,
                voxel_resolution=self.voxel_resolution,
                aggregation_method=self.aggregation_method
            )

            self.voxel_grid = voxel_grid
            self.valid_voxels = valid_voxels

            # Extract compact representation (only occupied voxels)
            occupied_indices = np.where(valid_voxels)
            voxel_centers = np.stack(occupied_indices, axis=1).astype(np.float64)
            voxel_centers = (voxel_centers + 0.5) / self.voxel_resolution - 0.5

            self.points_3d_nocs = voxel_centers
            self.saliencies_3d = voxel_grid[occupied_indices]

            occupied_voxels = np.sum(valid_voxels)
            occupancy_pct = (occupied_voxels / self.voxel_resolution**3) * 100

            print(f"  Occupied voxels: {occupied_voxels}/{self.voxel_resolution**3} ({occupancy_pct:.2f}%)")
        else:
            print(f"\n[5/5] Skipping voxelization (voxelize_anchor=False)")
            # Use raw normalized point cloud directly
            self.voxel_grid = None
            self.valid_voxels = None
            self.points_3d_nocs = points_3d_nocs
            self.saliencies_3d = saliencies_3d

            print(f"  Using {len(points_3d_nocs)} raw points (no voxelization)")

        # Store reference frame data for debug visualization
        self._ref_rgb = ref_rgb  # (H, W, 3)
        self._ref_mask = ref_mask  # (H, W)
        self._ref_depth = ref_depth  # (H, W)
        self._ref_saliency = saliency  # (C, H, W)
        self._ref_K = K_ref  # (3, 3)
        self._ref_pixel_coords = ref_pixel_coords  # (N, 2) [x, y] mapping to 3D points before voxelization

        print("\n=== Reference Model Built Successfully ===\n")

    def _voxelize_query_cloud(
        self,
        points_3d: np.ndarray,
        saliencies: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Voxelize query point cloud using anchor's normalization parameters.

        This ensures query frame has the same level of spatial abstraction as the
        anchor frame, which can help reduce false positive correspondences.

        Args:
            points_3d: (N, 3) query 3D points in camera space (numpy or torch)
            saliencies: (N, C) semantic saliency vectors (numpy or torch)

        Returns:
            voxelized_points: (M, 3) voxel centers in camera space (M <= N), same type as input
            voxelized_saliencies: (M, C) aggregated saliencies per voxel, same type as input
        """
        from concept_pose.core.voxelizer import voxelize_point_cloud_with_params

        if self.nocs_scale is None or self.nocs_centroid is None:
            raise RuntimeError(
                "Cannot voxelize query: anchor model normalization parameters not available. "
                "Call build_reference_model() first."
            )

        # Convert to numpy if needed (backproject_depth returns torch tensors)
        is_torch = isinstance(points_3d, torch.Tensor)
        if is_torch:
            points_np = points_3d.cpu().numpy()
            sal_np = saliencies.cpu().numpy()
        else:
            points_np = points_3d
            sal_np = saliencies

        # Use anchor's NOCS scale for consistent voxelization
        # Centroid is computed automatically from query points (view-specific)
        voxel_points, voxel_sal = voxelize_point_cloud_with_params(
            points_3d=points_np,
            saliencies=sal_np,
            nocs_scale=self.nocs_scale,
            voxel_resolution=self.voxel_resolution,
            aggregation_method=self.aggregation_method,
            apply_outlier_filter=False,  # Disable outlier filtering for query (may remove valid edge points)
            nb_neighbors=20,
            std_ratio=2.0
        )

        # Convert back to torch if input was torch
        if is_torch:
            voxel_points = torch.from_numpy(voxel_points).float().to(self.device)
            voxel_sal = torch.from_numpy(voxel_sal).float().to(self.device)

        return voxel_points, voxel_sal

    def estimate_pose(
        self,
        est_rgb: np.ndarray,
        est_mask: np.ndarray,
        est_depth: np.ndarray,
        K_est: Union[np.ndarray, Dict],
        visualize_query: bool = False,
        viz_save_path: Optional[str] = None,
        return_debug_info: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Estimate pose on target RGBD frame.

        Pipeline:
        1. Extract semantic saliency from target RGB (crops to object bbox automatically)
        2. Back-project target depth to 3D camera space
        3. Find semantic correspondences (3D-3D matching)
        4. RANSAC + Umeyama alignment
        5. ICP refinement

        Args:
            est_rgb: (H, W, 3) target RGB image (may be padded)
            est_mask: (H, W) target mask
            est_depth: (H, W) target depth in meters
            K_est: Target camera intrinsics (already adjusted for padding if applicable)
            visualize_query: If True, visualize query frame RGB, mask, depth, and saliency
            viz_save_path: If provided, save visualization to this path
            return_debug_info: If True, return extended debug information for visualization

        Returns:
            R: (3, 3) estimated rotation matrix
            t: (3,) estimated translation vector
            info: dict with diagnostic information
                - 'success': bool
                - 'num_correspondences': int
                - 'num_inliers': int
                - 'ransac_success': bool
                - 'scale': float
                - 'icp_converged': bool (if ICP used)
                - 'debug_info': dict (if return_debug_info=True) containing:
                    - 'saliency_query': (C, H, W) query saliency maps
                    - 'observed_3d': (N, 3) query 3D points in camera space
                    - 'observed_saliency': (N, C) query saliency vectors
                    - 'pixel_coords_query': (N, 2) query pixel coordinates
                    - 'model_3d': (M, 3) reference 3D points (transformed to world)
                    - 'model_saliency': (M, C) reference saliency vectors
                    - 'correspondences': dict with matched pairs data
                    - 'inlier_mask': (K,) boolean mask
        """
        if self.points_3d_nocs is None:
            raise RuntimeError("Reference model not built. Call build_reference_model() first.")

        print("\n=== Estimating Pose ===")

        # Preprocess inputs
        est_rgb = self._preprocess_rgb(est_rgb)
        est_mask = self._preprocess_mask(est_mask)
        est_depth = self._preprocess_depth(est_depth)
        K_est = self._parse_camera_matrix(K_est)

        # Step 1: Extract semantic saliency
        print("[1/4] Extracting semantic saliency...")
        saliency = self._extract_saliency(est_rgb, est_mask, self.semantic_labels)

        # Visualize query frame if requested
        if visualize_query:
            self._visualize_model_building(
                est_rgb, est_mask, est_depth, saliency, viz_save_path
            )

        # Step 2: Back-project to 3D camera space
        print("[2/4] Back-projecting depth to 3D camera space...")
        observed_3d, observed_sal, pixel_coords = registration_3d.backproject_depth(
            est_depth, saliency, K_est, device=self.device
        )

        if observed_3d is None or len(observed_3d) < 3:
            print("  Failed: Not enough valid 3D points")
            return None, None, {
                'success': False,
                'num_correspondences': 0,
                'num_inliers': 0,
                'ransac_success': False,
                'scale': None,
                'error': 'Not enough valid 3D points'
            }

        print(f"  Back-projected {len(observed_3d)} 3D points")

        # Step 2.5: Filter outliers from query frame (two-stage)
        print("  [2.5/4] Filtering query outliers (two-stage)...")

        # Convert to numpy if needed
        observed_3d_np = observed_3d.cpu().numpy() if isinstance(observed_3d, torch.Tensor) else observed_3d
        observed_sal_np = observed_sal.cpu().numpy() if isinstance(observed_sal, torch.Tensor) else observed_sal

        # Stage 1: Local statistical outlier removal
        observed_3d_np, observed_sal_np, stage1_mask = filter_statistical_outliers(
            observed_3d_np, observed_sal_np,
            nb_neighbors=20,
            std_ratio=1.5
        )

        # Apply stage 1 mask to pixel_coords
        if pixel_coords is not None:
            pixel_coords_np = pixel_coords.cpu().numpy() if isinstance(pixel_coords, torch.Tensor) else pixel_coords
            pixel_coords_np = pixel_coords_np[stage1_mask]

        # Stage 2: Global outlier removal (distance from center of mass)
        center = observed_3d_np.mean(axis=0)
        distances = np.linalg.norm(observed_3d_np - center, axis=1)
        mean_dist = distances.mean()
        std_dist = distances.std()
        global_threshold = mean_dist + 2.5 * std_dist

        global_inliers = distances < global_threshold
        n_removed_global = len(observed_3d_np) - np.sum(global_inliers)

        if n_removed_global > 0:
            observed_3d_np = observed_3d_np[global_inliers]
            observed_sal_np = observed_sal_np[global_inliers]
            if pixel_coords is not None:
                pixel_coords_np = pixel_coords_np[global_inliers]
            print(f"    Removed {n_removed_global} global outliers from query")

        print(f"    After filtering: {len(observed_3d_np)} query points")

        # Convert back to torch if needed
        if isinstance(observed_3d, torch.Tensor):
            observed_3d = torch.from_numpy(observed_3d_np).float().to(self.device)
            observed_sal = torch.from_numpy(observed_sal_np).float().to(self.device)
        else:
            observed_3d = observed_3d_np
            observed_sal = observed_sal_np

        # Update pixel_coords with filtered version
        if pixel_coords is not None:
            if isinstance(pixel_coords, torch.Tensor):
                pixel_coords = torch.from_numpy(pixel_coords_np).float().to(self.device)
            else:
                pixel_coords = pixel_coords_np

        # Check we still have enough points
        if len(observed_3d) < 3:
            print("  Failed: Not enough points after outlier filtering")
            return None, None, {
                'success': False,
                'num_correspondences': 0,
                'num_inliers': 0,
                'ransac_success': False,
                'scale': None,
                'error': 'Not enough points after outlier filtering'
            }

        # Step 2.6: Optionally voxelize query point cloud
        if self.voxelize_query:
            print("  [2.6/4] Voxelizing query point cloud...")
            num_points_before = len(observed_3d)
            observed_3d, observed_sal = self._voxelize_query_cloud(observed_3d, observed_sal)

            if observed_3d is None or len(observed_3d) < 3:
                print("  Failed: Not enough points after voxelization")
                return None, None, {
                    'success': False,
                    'num_correspondences': 0,
                    'num_inliers': 0,
                    'ransac_success': False,
                    'scale': None,
                    'error': 'Not enough points after voxelization'
                }

            print(f"  Voxelized: {num_points_before} → {len(observed_3d)} points")

            # Project voxel centers back to image to get pseudo-pixel coordinates
            # This allows visualizer to work with voxelized queries
            observed_3d_np = observed_3d.cpu().numpy() if isinstance(observed_3d, torch.Tensor) else observed_3d

            # Project 3D points to 2D: [x, y, z] -> [u, v]
            # K @ [X, Y, Z]^T = [u*Z, v*Z, Z]^T
            points_homo = (K_est @ observed_3d_np.T).T  # (N, 3)
            pixel_coords = points_homo[:, :2] / points_homo[:, 2:3]  # (N, 2) [x, y]

            print(f"  Computed pseudo-pixel coordinates for {len(pixel_coords)} voxel centers")
        else:
            print("  Skipping query voxelization (voxelize_query=False)")

        # Step 3: Get model 3D points (transform NOCS to world space)
        model_3d_nocs = torch.from_numpy(self.points_3d_nocs).float().to(self.device)
        nocs_scale_float = float(self.nocs_scale)
        nocs_centroid_tensor = torch.from_numpy(self.nocs_centroid).float().to(self.device)
        model_3d = model_3d_nocs * nocs_scale_float + nocs_centroid_tensor
        model_sal = torch.from_numpy(self.saliencies_3d).float().to(self.device)

        estimate_scale = self.config.get('estimate_scale', False)
        print(f"  Pre-scaled with extent-based scale={nocs_scale_float:.4f}m, Umeyama refine={estimate_scale}")

        # Step 4: RANSAC + Umeyama alignment
        print("[3/4] Running RANSAC + Umeyama alignment...")
        success, R, t, s, inlier_mask, correspondences = registration_3d.ransac_3d_registration(
            observed_3d, model_3d, observed_sal, model_sal,
            config=self.config,
            device=self.device
        )

        if not success:
            print("  Failed: RANSAC did not converge")
            return None, None, {
                'success': False,
                'num_correspondences': 0,
                'num_inliers': 0,
                'ransac_success': False,
                'scale': None,
                'error': 'RANSAC failed to converge'
            }

        num_inliers = np.sum(inlier_mask) if inlier_mask is not None else 0
        matched_observed, matched_model = correspondences if correspondences else (None, None)
        num_correspondences = len(matched_observed) if matched_observed is not None else 0

        print(f"  Found {num_correspondences} correspondences, {num_inliers} inliers")

        # Step 5: ICP refinement (optional)
        icp_converged = False
        icp_error = 0.0
        if self.config['use_icp_3d'] and num_inliers >= 3:
            print("[4/4] Running ICP refinement...")
            R_refined, t_refined, s_refined, icp_error = registration_3d.icp_refinement(
                observed_3d, model_3d, R, t, s,
                config=self.config
            )
            R, t, s = R_refined, t_refined, s_refined
            icp_converged = True
            print("  ICP refinement completed")
        else:
            print("[4/4] Skipping ICP refinement")

        # If in relative mode, compose with stored GT anchor pose
        if self._anchor_pose_mode == 'relative':
            print("\n[Composition] Composing relative transform with GT anchor pose...")
            # R, t is: anchor_camera → query_camera
            # Need: object → query_camera = (anchor_cam → query_cam) @ (object → anchor_cam)
            # where (object → anchor_cam) = (R_anchor, t_anchor)
            R_final = R @ self._anchor_R
            t_final = R @ self._anchor_t + t
            print(f"  Composed to absolute object pose in query frame")
            R, t = R_final, t_final

        print("\n=== Pose Estimation Complete ===")
        print(f"Estimated translation: {t}")
        print(f"Estimated scale: {s:.6f}")

        # Prepare info dict
        info = {
            'success': True,
            'num_correspondences': num_correspondences,
            'num_inliers': num_inliers,
            'ransac_success': True,
            'scale': float(s),
            'icp_converged': icp_converged
        }

        # Add debug info if requested
        if return_debug_info and matched_observed is not None:
            # Compute both KL divergence and cosine similarity for matched correspondences
            from concept_pose.pose.loss import compute_correspondence_scores

            # Get saliency vectors for matched points
            # matched_observed and matched_model are (K, 3) 3D coordinates
            # We need to map them back to saliency vectors
            # observed_sal is (N, C) and we need to find which indices in observed_3d correspond to matched_observed

            # Convert to tensors if needed
            obs_3d_tensor = torch.from_numpy(np.asarray(observed_3d)).float().to(self.device) if not isinstance(observed_3d, torch.Tensor) else observed_3d
            obs_sal_tensor = torch.from_numpy(np.asarray(observed_sal)).float().to(self.device) if not isinstance(observed_sal, torch.Tensor) else observed_sal
            model_sal_tensor = torch.from_numpy(self.saliencies_3d).float().to(self.device)

            # Find indices of matched points in original arrays
            # matched_observed and matched_model should be numpy arrays from RANSAC, but check first
            matched_obs_tensor = torch.from_numpy(matched_observed).float().to(self.device) if not isinstance(matched_observed, torch.Tensor) else matched_observed.float().to(self.device)
            matched_model_tensor = torch.from_numpy(matched_model).float().to(self.device) if not isinstance(matched_model, torch.Tensor) else matched_model.float().to(self.device)

            # Find nearest neighbors to get indices (since RANSAC returns point coordinates, not indices)
            # For observed points
            obs_dists = torch.cdist(matched_obs_tensor, obs_3d_tensor)  # (K, N)
            obs_indices = torch.argmin(obs_dists, dim=1)  # (K,)

            # For model points
            model_3d_tensor = torch.from_numpy(self.points_3d_nocs).float().to(self.device)
            model_3d_world = model_3d_tensor * nocs_scale_float + nocs_centroid_tensor
            model_dists = torch.cdist(matched_model_tensor, model_3d_world)  # (K, M)
            model_indices = torch.argmin(model_dists, dim=1)  # (K,)

            # Extract saliency vectors for correspondences
            corr_obs_sal = obs_sal_tensor[obs_indices]  # (K, C)
            corr_model_sal = model_sal_tensor[model_indices]  # (K, C)

            # Compute KL divergence (as used in matching)
            kl_scores = compute_correspondence_scores(
                corr_obs_sal, corr_model_sal,
                method='kl_divergence',
                temperature=self.config.get('temperature', 1.0),
                return_costs=False  # Return similarities (negative costs)
            )  # (K, K)
            kl_div_values = torch.diag(kl_scores).cpu().numpy()  # Extract diagonal for pairwise scores

            # Compute cosine similarity
            cosine_scores = compute_correspondence_scores(
                corr_obs_sal, corr_model_sal,
                method='cosine',
                return_costs=False
            )  # (K, K)
            cosine_sim_values = torch.diag(cosine_scores).cpu().numpy()

            # Helper to safely convert tensors to numpy
            def to_numpy(x):
                if x is None:
                    return None
                if isinstance(x, torch.Tensor):
                    return x.cpu().numpy()
                if isinstance(x, np.ndarray):
                    return x
                return np.asarray(x)

            # Prepare debug info
            debug_info = {
                # Configuration flags
                'voxelize_anchor': self.voxelize_anchor,
                'voxelize_query': self.voxelize_query,

                # Query frame data
                'saliency_query': to_numpy(saliency),  # (C, H, W)
                'observed_3d': to_numpy(observed_3d),  # (N, 3)
                'observed_saliency': to_numpy(observed_sal),  # (N, C)
                'pixel_coords_query': to_numpy(pixel_coords),  # (N, 2)

                # Reference frame data (from stored model building)
                'rgb_ref': to_numpy(self._ref_rgb) if hasattr(self, '_ref_rgb') and self._ref_rgb is not None else None,  # (H, W, 3)
                'mask_ref': to_numpy(self._ref_mask) if hasattr(self, '_ref_mask') and self._ref_mask is not None else None,  # (H, W)
                'depth_ref': to_numpy(self._ref_depth) if hasattr(self, '_ref_depth') and self._ref_depth is not None else None,  # (H, W)
                'saliency_ref': to_numpy(self._ref_saliency) if hasattr(self, '_ref_saliency') and self._ref_saliency is not None else None,  # (C, H, W)
                'pixel_coords_ref': to_numpy(self._ref_pixel_coords) if hasattr(self, '_ref_pixel_coords') and self._ref_pixel_coords is not None else None,  # (N, 2)

                # Reference model data (in world space, not NOCS)
                'model_3d': model_3d.cpu().numpy() if (self.voxelize_anchor or self.voxelize_query) else None,  # (M, 3) - skip for dense models
                'model_saliency': to_numpy(self.saliencies_3d) if (self.voxelize_anchor or self.voxelize_query) else None,  # (M, C) - skip for dense models

                # Correspondences (skip for dense non-voxelized models due to memory)
                'correspondences': {
                    'observed_idx': obs_indices.cpu().numpy() if (self.voxelize_anchor or self.voxelize_query) else None,
                    'model_idx': model_indices.cpu().numpy() if (self.voxelize_anchor or self.voxelize_query) else None,
                    'observed_3d': to_numpy(matched_observed) if (self.voxelize_anchor or self.voxelize_query) else None,
                    'model_3d': to_numpy(matched_model) if (self.voxelize_anchor or self.voxelize_query) else None,
                    'kl_divergence': to_numpy(kl_div_values) if (self.voxelize_anchor or self.voxelize_query) else None,
                    'cosine_similarity': to_numpy(cosine_sim_values) if (self.voxelize_anchor or self.voxelize_query) else None,
                    'observed_saliency': corr_obs_sal.cpu().numpy() if (self.voxelize_anchor or self.voxelize_query) else None,
                    'model_saliency': corr_model_sal.cpu().numpy() if (self.voxelize_anchor or self.voxelize_query) else None,
                },
                'inlier_mask': to_numpy(inlier_mask) if inlier_mask is not None else np.array([]),  # (K,)
            }

            info['debug_info'] = debug_info

        return R, t, info

    def cleanup(self):
        """Clean up resources (e.g., saliency generator)."""
        self._cleanup_saliency_generator()

    def __del__(self):
        """Destructor to ensure cleanup."""
        self.cleanup()


def estimate_pose_one_shot(
    ref_rgb: np.ndarray,
    ref_mask: np.ndarray,
    ref_depth: np.ndarray,
    ref_pose: Union[np.ndarray, Dict],
    K_ref: Union[np.ndarray, Dict],
    est_rgb: np.ndarray,
    est_mask: np.ndarray,
    est_depth: np.ndarray,
    K_est: Union[np.ndarray, Dict],
    semantic_labels: List[str],
    voxel_resolution: int = 64,
    device: str = 'cuda',
    **kwargs
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Convenience function for one-shot pose estimation.

    Builds model from reference frame and estimates pose on target frame
    in a single function call.

    Args:
        ref_rgb: (H, W, 3) reference RGB
        ref_mask: (H, W) reference mask
        ref_depth: (H, W) reference depth
        ref_pose: Reference object pose
        K_ref: Reference camera intrinsics
        est_rgb: (H, W, 3) target RGB
        est_mask: (H, W) target mask
        est_depth: (H, W) target depth
        K_est: Target camera intrinsics
        semantic_labels: List of semantic labels
        voxel_resolution: Voxel grid resolution
        device: Torch device
        **kwargs: Additional arguments for OneShotPoseEstimator

    Returns:
        R: (3, 3) rotation matrix
        t: (3,) translation vector
        info: dict with diagnostic information
    """
    estimator = OneShotPoseEstimator(
        voxel_resolution=voxel_resolution,
        device=device,
        **kwargs
    )

    # Build reference model
    estimator.build_reference_model(
        ref_rgb=ref_rgb,
        ref_mask=ref_mask,
        ref_depth=ref_depth,
        ref_pose=ref_pose,
        K_ref=K_ref,
        semantic_labels=semantic_labels
    )

    # Estimate pose
    R, t, info = estimator.estimate_pose(
        est_rgb=est_rgb,
        est_mask=est_mask,
        est_depth=est_depth,
        K_est=K_est
    )

    # Cleanup
    estimator.cleanup()

    return R, t, info
