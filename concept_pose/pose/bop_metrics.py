"""
BOP Challenge Metrics for 6D Object Pose Estimation
====================================================

Implements the three standard BOP metrics:
- VSD (Visible Surface Discrepancy): Depth-based metric accounting for occlusions
- MSSD (Maximum Symmetry-Aware Surface Distance): 3D distance metric
- MSPD (Maximum Symmetry-Aware Projection Distance): 2D reprojection metric

And evaluation protocol:
- Average Recall (AR): Average over multiple error thresholds
- BOP Score: Mean of AR_VSD, AR_MSSD, AR_MSPD

References:
- BOP Challenge: https://bop.felk.cvut.cz/
- BOP Toolkit: https://github.com/thodan/bop_toolkit
- Paper: https://arxiv.org/abs/1808.08319

Usage:
    # Initialize evaluator once
    evaluator = BOPEvaluator(mesh_dir='/path/to/meshes')

    # Evaluate each frame
    for frame in test_frames:
        metrics = evaluator.evaluate_frame(
            est_R, est_t, gt_R, gt_t, K, image_size, object_name
        )

    # Compute final scores
    ar_metrics = evaluator.compute_average_recall()
    bop_score = evaluator.compute_bop_score()
"""

import numpy as np
import trimesh
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import warnings


class BOPEvaluator:
    """
    BOP Challenge metrics evaluator for 6D pose estimation.

    Features:
    - Lazy mesh loading with caching
    - PyTorch3D rendering for VSD (optional)
    - Symmetry-aware metrics (MSSD, MSPD)
    - Per-frame evaluation + accumulated AR computation
    - Graceful degradation if dependencies missing

    Attributes:
        mesh_dir: Directory containing object mesh files
        symmetries: Dict mapping object names to symmetry transformations
        renderer: Rendering backend ('pytorch3d' or None for no VSD)
        device: torch device for computation
    """

    def __init__(
        self,
        mesh_dir: str,
        symmetries: Optional[Dict[str, List[Dict]]] = None,
        renderer: str = 'pytorch3d',
        device: str = 'cuda',
        mesh_scale: float = 0.001
    ):
        """
        Initialize BOP evaluator.

        Args:
            mesh_dir: Path to directory containing object meshes
            symmetries: Dict of {object_name: [{'R': rotation_matrix}]}
                       Empty list means asymmetric object
            renderer: Rendering backend ('pytorch3d' or None)
            device: Torch device for computation
            mesh_scale: Scale factor for mesh vertices (default 0.001 for BOP mm->m conversion)
        """
        self.mesh_dir = Path(mesh_dir)
        self.symmetries = symmetries or {}
        self.device = device
        self.mesh_scale = mesh_scale

        # Mesh cache: {object_name: trimesh.Trimesh}
        self.mesh_cache = {}
        self.model_points_cache = {}  # {object_name: np.ndarray (N, 3)}
        self.diameter_cache = {}  # {object_name: float}

        # Accumulated errors for AR computation
        self.vsd_errors = []
        self.mssd_errors = []
        self.mspd_errors = []
        self.model_diameters = []  # Store diameter for each evaluation

        # Initialize renderer
        self.depth_renderer = None
        if renderer == 'pytorch3d':
            try:
                self.depth_renderer = DepthRenderer(backend='pytorch3d', device=device)
                print(f"[BOP] PyTorch3D renderer initialized on {device}")
            except ImportError:
                warnings.warn(
                    "PyTorch3D not available. VSD metric will be disabled. "
                    "Install with: pip install pytorch3d"
                )
            except RuntimeError as e:
                # PyTorch3D installed but without CUDA support
                if 'Not compiled with GPU support' in str(e) or 'CUDA' in str(e):
                    warnings.warn(
                        f"PyTorch3D not compiled with GPU support. Falling back to CPU rendering for VSD. "
                        f"This will be slower. Error: {e}"
                    )
                    try:
                        self.depth_renderer = DepthRenderer(backend='pytorch3d', device='cpu')
                        print(f"[BOP] PyTorch3D renderer initialized on CPU (slow)")
                    except Exception as e2:
                        warnings.warn(f"Failed to initialize CPU renderer: {e2}. VSD will be disabled.")
                else:
                    raise

    def load_mesh(self, object_name: str, mesh_path: Optional[str] = None) -> Optional[trimesh.Trimesh]:
        """
        Load mesh for an object (with caching).

        Args:
            object_name: Object identifier (e.g., 'shoe-aqua_cyan_right')
            mesh_path: Optional explicit mesh path (for dataset-specific structures)

        Returns:
            trimesh.Trimesh object or None if not found
        """
        if object_name in self.mesh_cache:
            return self.mesh_cache[object_name]

        # If explicit mesh path provided, use it (dataset-specific)
        if mesh_path is not None:
            mesh_path = Path(mesh_path)
        else:
            # Default behavior: construct path from object name
            # Extract category from object name
            # HouseCat6D: shoe-aqua_cyan_right → shoe
            # Real275: bottle_shengjun_norm → bottle
            if '-' in object_name:
                category = object_name.split('-')[0]
            else:
                category = object_name.split('_')[0]

            mesh_path = self.mesh_dir / category / f"{object_name}.obj"

        if not mesh_path.exists():
            warnings.warn(f"Mesh not found: {mesh_path}")
            return None

        try:
            mesh = trimesh.load(str(mesh_path), force='mesh')
            # Apply scale factor (e.g., convert BOP meshes from mm to meters)
            if self.mesh_scale != 1.0:
                mesh.vertices *= self.mesh_scale
            self.mesh_cache[object_name] = mesh
            return mesh
        except Exception as e:
            warnings.warn(f"Failed to load mesh {mesh_path}: {e}")
            return None

    def get_model_points(self, object_name: str, num_points: int = 1000, mesh_path: Optional[str] = None) -> Optional[np.ndarray]:
        """
        Get uniformly sampled points from object mesh surface.

        Args:
            object_name: Object identifier
            num_points: Number of points to sample
            mesh_path: Optional explicit mesh path (for dataset-specific structures)

        Returns:
            (N, 3) array of 3D points or None
        """
        if object_name in self.model_points_cache:
            return self.model_points_cache[object_name]

        mesh = self.load_mesh(object_name, mesh_path=mesh_path)
        if mesh is None:
            return None

        # Sample points uniformly from mesh surface
        points, _ = trimesh.sample.sample_surface(mesh, num_points)
        self.model_points_cache[object_name] = points
        return points

    def get_model_diameter(self, object_name: str, mesh_path: Optional[str] = None) -> Optional[float]:
        """
        Compute model diameter (max distance between any two points).

        Args:
            object_name: Object identifier
            mesh_path: Optional explicit path to mesh file (dataset-specific)

        Returns:
            Diameter in meters or None
        """
        if object_name in self.diameter_cache:
            return self.diameter_cache[object_name]

        mesh = self.load_mesh(object_name, mesh_path=mesh_path)
        if mesh is None:
            return None

        # Diameter = max distance between vertices
        vertices = np.array(mesh.vertices)
        diameter = compute_model_diameter(mesh)
        self.diameter_cache[object_name] = diameter
        return diameter

    def evaluate_frame(
        self,
        est_R: np.ndarray,
        est_t: np.ndarray,
        gt_R: np.ndarray,
        gt_t: np.ndarray,
        K: np.ndarray,
        image_size: Tuple[int, int],
        object_name: str
    ) -> Dict[str, float]:
        """
        Evaluate BOP metrics for a single frame.

        Args:
            est_R: (3, 3) estimated rotation matrix
            est_t: (3,) estimated translation vector
            gt_R: (3, 3) ground truth rotation
            gt_t: (3,) ground truth translation
            K: (3, 3) camera intrinsics
            image_size: (height, width) tuple
            object_name: Object identifier

        Returns:
            Dictionary containing:
                - vsd: Visible Surface Discrepancy (if renderer available)
                - mssd: Maximum Symmetry-Aware Surface Distance
                - mspd: Maximum Symmetry-Aware Projection Distance
                - model_diameter: Object diameter (for reference)
        """
        metrics = {}

        # Get model data
        mesh = self.load_mesh(object_name)
        model_points = self.get_model_points(object_name)
        diameter = self.get_model_diameter(object_name)

        if mesh is None or model_points is None or diameter is None:
            warnings.warn(f"Cannot compute BOP metrics for {object_name} (mesh/data unavailable)")
            return metrics

        # Get symmetries for this object
        symmetries = self.symmetries.get(object_name, [])

        # Compute VSD (if renderer available)
        if self.depth_renderer is not None:
            try:
                # BOP/Any6D: compute VSD for multiple tau thresholds (5-50% of diameter)
                vsd_errors = compute_vsd(
                    est_R, est_t, gt_R, gt_t,
                    mesh, K, image_size,
                    self.depth_renderer,
                    delta=15.0,  # 15mm visibility tolerance
                    tau_fracs=None,  # Use default [0.05, 0.10, ..., 0.50]
                    diameter=diameter
                )
                # Store array of VSD errors (one per tau)
                self.vsd_errors.append(vsd_errors)
                # Report mean VSD for display (across all taus)
                metrics['vsd'] = np.mean(vsd_errors)
            except RuntimeError as e:
                # Try CPU rendering if GPU not supported
                if 'Not compiled with GPU support' in str(e) or 'CUDA' in str(e):
                    try:
                        cpu_renderer = DepthRenderer(backend='pytorch3d', device='cpu')
                        vsd_errors = compute_vsd(
                            est_R, est_t, gt_R, gt_t,
                            mesh, K, image_size,
                            cpu_renderer,
                            delta=15.0,
                            tau_fracs=None,
                            diameter=diameter
                        )
                        self.vsd_errors.append(vsd_errors)
                        metrics['vsd'] = np.mean(vsd_errors)
                        # Switch to CPU renderer permanently
                        self.depth_renderer = cpu_renderer
                        warnings.warn("PyTorch3D GPU rendering failed. Switched to CPU rendering (slower).")
                    except Exception as e2:
                        warnings.warn(f"VSD computation failed on CPU too: {e2}")
                else:
                    warnings.warn(f"VSD computation failed: {e}")
            except Exception as e:
                warnings.warn(f"VSD computation failed: {e}")

        # Compute MSSD
        try:
            mssd = compute_mssd(
                est_R, est_t, gt_R, gt_t,
                model_points, symmetries, diameter
            )
            metrics['mssd'] = mssd
            self.mssd_errors.append(mssd)
        except Exception as e:
            warnings.warn(f"MSSD computation failed: {e}")

        # Compute MSPD
        try:
            mspd = compute_mspd(
                est_R, est_t, gt_R, gt_t,
                model_points, K, symmetries
            )
            metrics['mspd'] = mspd
            self.mspd_errors.append(mspd)
        except Exception as e:
            warnings.warn(f"MSPD computation failed: {e}")

        # Store diameter for this frame
        self.model_diameters.append(diameter)
        metrics['model_diameter'] = diameter

        return metrics

    def compute_average_recall(
        self,
        vsd_thresholds: Optional[List[float]] = None,
        mssd_thresholds: Optional[List[float]] = None,
        mspd_thresholds: Optional[List[int]] = None
    ) -> Dict[str, float]:
        """
        Compute Average Recall (AR) over multiple error thresholds.

        AR follows BOP/Any6D methodology:
        - VSD: Averaged over all (frame, tau, theta) combinations
        - MSSD/MSPD: Averaged over all (frame, theta) combinations

        Args:
            vsd_thresholds: List of VSD theta thresholds (error fractions in [0, 1])
                           Default: [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
                           Note: VSD computed for multiple tau values internally
            mssd_thresholds: List of MSSD thresholds (fractions of diameter)
                            Default: [0.05, 0.10, ..., 0.50] (10 values, Any6D standard)
            mspd_thresholds: List of MSPD thresholds (pixels)
                            Default: [5, 10, ..., 50] (10 values, Any6D standard)

        Returns:
            Dictionary with:
                - ar_vsd: Average recall for VSD
                - ar_mssd: Average recall for MSSD
                - ar_mspd: Average recall for MSPD
                - num_frames: Number of evaluated frames
        """
        # Default thresholds matching Any6D/BOP (0.05 to 0.50 step 0.05)
        if vsd_thresholds is None:
            vsd_thresholds = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
        if mssd_thresholds is None:
            mssd_thresholds = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
        if mspd_thresholds is None:
            mspd_thresholds = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]

        results = {'num_frames': len(self.model_diameters)}

        # Compute AR for VSD (Any6D/BOP approach)
        # VSD errors are stored as (num_frames, num_taus) where each frame has
        # VSD computed for multiple tau values [0.05, 0.10, ..., 0.50]
        if len(self.vsd_errors) > 0:
            # Stack all VSD error arrays into 2D: (num_frames, num_taus)
            vsd_errors_2d = np.stack(self.vsd_errors, axis=0)  # (N, 10)

            # For each theta threshold, compute recall across all (frame, tau) pairs
            # This matches Any6D: all_vsd_recs = np.stack([vsd_errs < rec_i for rec_i in vsd_rec], axis=1)
            vsd_recalls = []
            for theta in vsd_thresholds:
                # Boolean array: which (frame, tau) pairs have error < theta
                correct_mask = vsd_errors_2d < theta  # (num_frames, num_taus)
                recall = np.mean(correct_mask)  # Average over all frames and taus
                vsd_recalls.append(recall)

            # AR_VSD: average recall across all theta thresholds
            results['ar_vsd'] = np.mean(vsd_recalls)

        # Compute AR for MSSD
        if len(self.mssd_errors) > 0:
            mssd_recalls = []
            for tau_frac in mssd_thresholds:
                correct = sum(
                    mssd < tau_frac * diam
                    for mssd, diam in zip(self.mssd_errors, self.model_diameters)
                )
                recall = correct / len(self.mssd_errors)
                mssd_recalls.append(recall)
            results['ar_mssd'] = np.mean(mssd_recalls)

        # Compute AR for MSPD
        if len(self.mspd_errors) > 0:
            mspd_recalls = []
            for tau_px in mspd_thresholds:
                correct = sum(mspd < tau_px for mspd in self.mspd_errors)
                recall = correct / len(self.mspd_errors)
                mspd_recalls.append(recall)
            results['ar_mspd'] = np.mean(mspd_recalls)

        return results

    def compute_bop_score(self) -> float:
        """
        Compute BOP score (average of AR_VSD, AR_MSSD, AR_MSPD).

        Returns:
            BOP score in [0, 1] or 0.0 if no metrics available
        """
        ar_metrics = self.compute_average_recall()

        # Average available AR metrics
        ar_values = [
            ar_metrics.get('ar_vsd'),
            ar_metrics.get('ar_mssd'),
            ar_metrics.get('ar_mspd')
        ]
        ar_values = [v for v in ar_values if v is not None]

        if len(ar_values) == 0:
            return 0.0

        return np.mean(ar_values)

    def reset(self):
        """Reset accumulated errors (call before new evaluation run)."""
        self.vsd_errors = []
        self.mssd_errors = []
        self.mspd_errors = []
        self.model_diameters = []


# =============================================================================
# Core Metric Functions
# =============================================================================

def compute_vsd(
    est_R: np.ndarray,
    est_t: np.ndarray,
    gt_R: np.ndarray,
    gt_t: np.ndarray,
    mesh: trimesh.Trimesh,
    K: np.ndarray,
    image_size: Tuple[int, int],
    renderer: 'DepthRenderer',
    delta: float = 15.0,
    tau_fracs: List[float] = None,
    diameter: float = 1.0,
    cost_type: str = 'step'
) -> np.ndarray:
    """
    Compute Visible Surface Discrepancy (VSD) for multiple tau thresholds.

    VSD measures the discrepancy between visible surfaces only, accounting
    for occlusions by rendering depth maps at both poses.

    Args:
        est_R, est_t: Estimated pose
        gt_R, gt_t: Ground truth pose
        mesh: Object mesh (trimesh.Trimesh)
        K: Camera intrinsics (3, 3)
        image_size: (height, width)
        renderer: DepthRenderer instance
        delta: Visibility mask tolerance (mm)
        tau_fracs: List of error thresholds as fractions of diameter
                  Default: [0.05, 0.10, ..., 0.50] (BOP standard)
        diameter: Model diameter (m)
        cost_type: 'step' (binary) or 'tlinear' (truncated linear)

    Returns:
        Array of VSD errors in [0, 1], one per tau threshold
    """
    if tau_fracs is None:
        # Default: BOP standard tau range (5% to 50% of diameter)
        tau_fracs = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]

    height, width = image_size

    # Render depth at estimated pose (only once!)
    depth_est = renderer.render(mesh, est_R, est_t, K, image_size)

    # Render depth at GT pose (only once!)
    depth_gt = renderer.render(mesh, gt_R, gt_t, K, image_size)

    # Convert depth to distance images
    dist_est = depth_to_distance_image(depth_est, K)
    dist_gt = depth_to_distance_image(depth_gt, K)

    # Estimate visibility masks
    visib_est = dist_est > 0
    visib_gt = dist_gt > 0

    # Compute union and intersection of visibility masks
    visib_union = visib_est | visib_gt
    visib_union_count = np.sum(visib_union)

    if visib_union_count == 0:
        # Maximum error if no visible pixels in either render
        return np.ones(len(tau_fracs))

    # Intersection: pixels visible in BOTH renders (refined by delta tolerance)
    visib_inter = estimate_visib_mask(dist_est, dist_gt, delta, visib_est & visib_gt)
    visib_inter_count = np.sum(visib_inter)

    # Complement: pixels in union but not in intersection (visibility mismatch penalty)
    complement_count = visib_union_count - visib_inter_count

    # Compute VSD for each tau threshold
    vsd_errors = []

    if visib_inter_count > 0:
        # Compute distance differences once
        dist_diff = np.abs(dist_est[visib_inter] - dist_gt[visib_inter])

        for tau_frac in tau_fracs:
            tau = tau_frac * diameter  # Convert to absolute threshold

            if cost_type == 'step':
                # Binary cost: 1 if distance exceeds tau, 0 otherwise
                costs = (dist_diff > tau).astype(float)
            elif cost_type == 'tlinear':
                # Truncated linear: normalized by tau, clamped to [0, 1]
                costs = np.minimum(dist_diff / tau, 1.0)
            else:
                raise ValueError(f"Unknown cost_type: {cost_type}")

            sum_costs = np.sum(costs)

            # Official BOP VSD formula
            vsd_error = (sum_costs + complement_count) / visib_union_count
            vsd_errors.append(vsd_error)
    else:
        # No intersection, only visibility mismatch penalty
        vsd_error = complement_count / visib_union_count
        vsd_errors = [vsd_error] * len(tau_fracs)

    return np.array(vsd_errors)


def compute_mssd(
    est_R: np.ndarray,
    est_t: np.ndarray,
    gt_R: np.ndarray,
    gt_t: np.ndarray,
    model_points: np.ndarray,
    symmetries: Optional[List[Dict]] = None,
    diameter: float = 1.0
) -> float:
    """
    Compute Maximum Symmetry-Aware Surface Distance (MSSD).

    MSSD measures the maximum 3D distance between transformed model points,
    accounting for object symmetries.

    Args:
        est_R, est_t: Estimated pose (t can be (3,) or (3,1))
        gt_R, gt_t: Ground truth pose (t can be (3,) or (3,1))
        model_points: (N, 3) array of model surface points
        symmetries: List of symmetry dicts with 'R' key (rotation matrices)
        diameter: Model diameter (for normalization)

    Returns:
        MSSD error (max distance across all points and symmetries)
    """
    # Ensure t vectors are (3,) for consistent broadcasting
    est_t = np.asarray(est_t).flatten()
    gt_t = np.asarray(gt_t).flatten()

    # Transform points to world space with estimated pose
    pts_est = (est_R @ model_points.T).T + est_t  # (N, 3)

    # If no symmetries, use identity
    if not symmetries:
        symmetries = [{'R': np.eye(3), 't': np.zeros((3, 1))}]

    # Try all symmetries and find minimum error
    errors = []
    for sym in symmetries:
        # Apply symmetry to GT pose
        R_gt_sym = gt_R @ sym['R']
        sym_t = np.asarray(sym.get('t', np.zeros((3, 1)))).flatten()
        t_gt_sym = gt_t + (gt_R @ sym_t)

        # Transform points with GT symmetry pose
        pts_gt = (R_gt_sym @ model_points.T).T + t_gt_sym  # (N, 3)

        # Compute maximum distance
        distances = np.linalg.norm(pts_est - pts_gt, axis=1)
        max_dist = np.max(distances)
        errors.append(max_dist)

    # Return minimum error across symmetries
    return min(errors)


def compute_mspd(
    est_R: np.ndarray,
    est_t: np.ndarray,
    gt_R: np.ndarray,
    gt_t: np.ndarray,
    model_points: np.ndarray,
    K: np.ndarray,
    symmetries: Optional[List[Dict]] = None
) -> float:
    """
    Compute Maximum Symmetry-Aware Projection Distance (MSPD).

    MSPD measures the maximum 2D reprojection error in pixels,
    accounting for object symmetries.

    Args:
        est_R, est_t: Estimated pose (t can be (3,) or (3,1))
        gt_R, gt_t: Ground truth pose (t can be (3,) or (3,1))
        model_points: (N, 3) array of model surface points
        K: Camera intrinsics (3, 3)
        symmetries: List of symmetry dicts

    Returns:
        MSPD error (max pixel distance)
    """
    # Ensure t vectors are (3,) for consistent broadcasting
    est_t = np.asarray(est_t).flatten()
    gt_t = np.asarray(gt_t).flatten()

    # Project points with estimated pose
    pixels_est = project_points(model_points, est_R, est_t, K)  # (N, 2)

    # If no symmetries, use identity
    if not symmetries:
        symmetries = [{'R': np.eye(3), 't': np.zeros((3, 1))}]

    # Try all symmetries
    errors = []
    for sym in symmetries:
        # Apply symmetry to GT pose
        R_gt_sym = gt_R @ sym['R']
        sym_t = np.asarray(sym.get('t', np.zeros((3, 1)))).flatten()
        t_gt_sym = gt_t + (gt_R @ sym_t)

        # Project with GT symmetry pose
        pixels_gt = project_points(model_points, R_gt_sym, t_gt_sym, K)  # (N, 2)

        # Compute maximum pixel distance
        pixel_diffs = np.linalg.norm(pixels_est - pixels_gt, axis=1)
        max_diff = np.max(pixel_diffs)
        errors.append(max_diff)

    return min(errors)


# =============================================================================
# Utility Functions
# =============================================================================

def compute_model_diameter(mesh: trimesh.Trimesh) -> float:
    """
    Compute model diameter (maximum distance between any two vertices).

    Args:
        mesh: trimesh.Trimesh object

    Returns:
        Diameter in meters
    """
    vertices = np.array(mesh.vertices)

    # Efficient approximation: diameter ≈ 2 * max distance from centroid
    centroid = vertices.mean(axis=0)
    distances = np.linalg.norm(vertices - centroid, axis=1)
    diameter = 2 * np.max(distances)

    return diameter


def depth_to_distance_image(depth: np.ndarray, K: np.ndarray) -> np.ndarray:
    """
    Convert depth image to distance image (depth along optical axis → 3D distance).

    Args:
        depth: (H, W) depth map (z-coordinate)
        K: (3, 3) camera intrinsics

    Returns:
        (H, W) distance image (Euclidean distance from camera center)
    """
    height, width = depth.shape
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    # Create pixel coordinate grids
    u, v = np.meshgrid(np.arange(width), np.arange(height))

    # Compute normalized coordinates
    x_norm = (u - cx) / fx
    y_norm = (v - cy) / fy

    # Distance = depth * sqrt(1 + x_norm^2 + y_norm^2)
    distance = depth * np.sqrt(1 + x_norm**2 + y_norm**2)

    return distance


def estimate_visib_mask(
    dist_est: np.ndarray,
    dist_gt: np.ndarray,
    delta: float,
    base_mask: np.ndarray
) -> np.ndarray:
    """
    Estimate visibility mask for VSD computation.

    Pixels are considered visible if:
    1. They are visible in base_mask
    2. Distance difference is within delta tolerance

    Args:
        dist_est: (H, W) distance image for estimated pose
        dist_gt: (H, W) distance image for GT pose
        delta: Tolerance in mm (converted to meters internally)
        base_mask: (H, W) boolean mask of initially visible pixels

    Returns:
        (H, W) boolean visibility mask
    """
    delta_m = delta / 1000.0  # Convert mm to meters

    # Pixels are visible if distance difference is small
    dist_diff = np.abs(dist_est - dist_gt)
    close_enough = dist_diff < delta_m

    return base_mask & close_enough


def project_points(
    points_3d: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    K: np.ndarray
) -> np.ndarray:
    """
    Project 3D points to 2D image coordinates.

    Args:
        points_3d: (N, 3) points in object space
        R: (3, 3) rotation matrix
        t: (3,) translation vector
        K: (3, 3) camera intrinsics

    Returns:
        (N, 2) pixel coordinates [u, v]
    """
    # Transform to camera space
    points_cam = (R @ points_3d.T).T + t  # (N, 3)

    # Project to image
    points_hom = (K @ points_cam.T).T  # (N, 3)
    pixels = points_hom[:, :2] / points_hom[:, 2:3]  # (N, 2)

    return pixels


def sample_model_points(mesh: trimesh.Trimesh, num_points: int = 1000) -> np.ndarray:
    """
    Sample points uniformly from mesh surface.

    Args:
        mesh: trimesh.Trimesh object
        num_points: Number of points to sample

    Returns:
        (N, 3) array of sampled points
    """
    points, _ = trimesh.sample.sample_surface(mesh, num_points)
    return points


# =============================================================================
# Depth Renderer
# =============================================================================

class DepthRenderer:
    """
    Depth renderer abstraction supporting PyTorch3D backend.
    """

    def __init__(self, backend: str = 'pytorch3d', device: str = 'cuda'):
        """
        Initialize depth renderer.

        Args:
            backend: 'pytorch3d' (only supported backend currently)
            device: torch device
        """
        self.backend = backend
        self.device = device
        self.renderer = None

        if backend == 'pytorch3d':
            self._init_pytorch3d()
        else:
            raise ValueError(f"Unsupported renderer backend: {backend}")

    def _init_pytorch3d(self):
        """Initialize PyTorch3D renderer."""
        try:
            from pytorch3d.structures import Meshes
            from pytorch3d.renderer import (
                RasterizationSettings,
                MeshRasterizer,
                PerspectiveCameras
            )
            self.pytorch3d_available = True
            # Renderer will be created per-render with specific camera params
        except ImportError:
            raise ImportError(
                "PyTorch3D not available. Install with: pip install pytorch3d"
            )

    def render(
        self,
        mesh: trimesh.Trimesh,
        R: np.ndarray,
        t: np.ndarray,
        K: np.ndarray,
        image_size: Tuple[int, int]
    ) -> np.ndarray:
        """
        Render depth map for object at given pose.

        Args:
            mesh: trimesh.Trimesh object
            R: (3, 3) rotation matrix (object to camera)
            t: (3,) translation vector
            K: (3, 3) camera intrinsics
            image_size: (height, width)

        Returns:
            (H, W) depth map (z-coordinates in meters, 0 for background)
        """
        if self.backend == 'pytorch3d':
            return self._render_pytorch3d(mesh, R, t, K, image_size)
        else:
            raise NotImplementedError(f"Backend {self.backend} not implemented")

    def _render_pytorch3d(
        self,
        mesh: trimesh.Trimesh,
        R: np.ndarray,
        t: np.ndarray,
        K: np.ndarray,
        image_size: Tuple[int, int]
    ) -> np.ndarray:
        """Render using PyTorch3D."""
        from pytorch3d.structures import Meshes
        from pytorch3d.renderer import (
            RasterizationSettings,
            MeshRasterizer,
            PerspectiveCameras
        )

        height, width = image_size

        # Convert trimesh to PyTorch3D format
        verts = torch.from_numpy(np.array(mesh.vertices)).float().to(self.device)
        faces = torch.from_numpy(np.array(mesh.faces)).long().to(self.device)

        # Transform mesh to camera frame
        R_tensor = torch.from_numpy(R).float().to(self.device)
        t_tensor = torch.from_numpy(t).float().to(self.device)
        verts_cam = (R_tensor @ verts.T).T + t_tensor

        # Create PyTorch3D mesh
        meshes = Meshes(verts=[verts_cam], faces=[faces])

        # Setup camera (PyTorch3D uses different convention than OpenCV)
        # PyTorch3D NDC space: convert intrinsics
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]

        # Convert to PyTorch3D NDC coordinates
        focal_length = torch.tensor([[fx, fy]], dtype=torch.float32, device=self.device)
        principal_point = torch.tensor([[cx, cy]], dtype=torch.float32, device=self.device)

        cameras = PerspectiveCameras(
            focal_length=focal_length,
            principal_point=principal_point,
            image_size=((height, width),),
            device=self.device,
            in_ndc=False
        )

        # Rasterization settings
        raster_settings = RasterizationSettings(
            image_size=(height, width),
            blur_radius=0.0,
            faces_per_pixel=1,
            perspective_correct=True,
            bin_size=0  # Use naive rasterization to avoid overflow for complex meshes
        )

        # Create rasterizer
        rasterizer = MeshRasterizer(
            cameras=cameras,
            raster_settings=raster_settings
        )

        # Rasterize
        fragments = rasterizer(meshes)

        # Extract depth (zbuf: z-coordinates in camera frame)
        depth = fragments.zbuf[0, :, :, 0].cpu().numpy()

        # PyTorch3D uses -1 for background, convert to 0
        depth[depth < 0] = 0

        return depth


# =============================================================================
# Example Usage
# =============================================================================

if __name__ == '__main__':
    # Example: Evaluate BOP metrics on a single frame

    # Initialize evaluator
    evaluator = BOPEvaluator(
        mesh_dir='data/HouseCat6D/obj_models_small_size_final',
        renderer='pytorch3d'
    )

    # Example data (replace with actual values)
    est_R = np.eye(3)
    est_t = np.array([0.0, 0.0, 0.5])
    gt_R = np.eye(3)
    gt_t = np.array([0.0, 0.0, 0.5])
    K = np.array([[500, 0, 256], [0, 500, 256], [0, 0, 1]])
    image_size = (512, 512)
    object_name = 'shoe-aqua_cyan_right'

    # Evaluate single frame
    metrics = evaluator.evaluate_frame(
        est_R, est_t, gt_R, gt_t, K, image_size, object_name
    )

    print("BOP Metrics:")
    print(f"  VSD: {metrics.get('vsd', 'N/A')}")
    print(f"  MSSD: {metrics.get('mssd', 'N/A')}")
    print(f"  MSPD: {metrics.get('mspd', 'N/A')}")

    # After evaluating all frames, compute AR and BOP score
    ar_metrics = evaluator.compute_average_recall()
    bop_score = evaluator.compute_bop_score()

    print(f"\nAverage Recall:")
    print(f"  AR_VSD: {ar_metrics.get('ar_vsd', 'N/A')}")
    print(f"  AR_MSSD: {ar_metrics.get('ar_mssd', 'N/A')}")
    print(f"  AR_MSPD: {ar_metrics.get('ar_mspd', 'N/A')}")
    print(f"\nBOP Score: {bop_score}")
