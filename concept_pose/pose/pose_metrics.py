"""
Pose Estimation Metrics
=======================

Core metrics for 6D pose estimation evaluation.

This module provides clean, self-contained implementations of standard
pose estimation metrics:
- Rotation and translation errors
- ADD (Average Distance of Model Points)
- ADD-S (Average Symmetric Distance)
- 3D IoU (3D Intersection over Union)
"""

import numpy as np
from typing import Dict, Optional


def compute_pose_errors(est_R: np.ndarray, est_t: np.ndarray,
                       gt_R: np.ndarray, gt_t: np.ndarray,
                       symmetries: Optional[list] = None) -> Dict[str, float]:
    """
    Compute basic rotation and translation errors with optional symmetry handling.

    Rotation error is symmetry-aware: tests all symmetry transformations and returns
    the minimum error (following BOP evaluation protocol). Translation error remains
    standard Euclidean distance (following NOCS convention).

    Args:
        est_R: (3, 3) estimated rotation matrix
        est_t: (3,) estimated translation vector
        gt_R: (3, 3) ground truth rotation matrix
        gt_t: (3,) ground truth translation vector
        symmetries: Optional list of symmetry transformations in BOP format:
                   [{'R': (3,3), 't': (3,1)}, ...]. If None or empty, no symmetry.

    Returns:
        Dictionary with:
            - translation_error_m: Translation error in meters
            - rotation_error_deg: Rotation error in degrees (min across symmetries)
            - success_5deg2cm, success_10deg5cm: Standard thresholds
            - success_5deg5cm, success_10deg10cm: Alternative thresholds
    """
    # Translation error (Euclidean distance, no symmetry consideration per NOCS)
    t_error = np.linalg.norm(est_t - gt_t)

    # Rotation error with symmetry handling
    if symmetries is None or len(symmetries) == 0:
        # No symmetry - use standard geodesic distance
        symmetries = [{'R': np.eye(3), 't': np.zeros((3, 1))}]

    min_r_error_deg = float('inf')
    for sym in symmetries:
        # Apply symmetry to GT rotation
        R_gt_sym = gt_R @ sym['R']

        # Compute rotation error (geodesic distance on SO(3))
        R_error = est_R @ R_gt_sym.T
        trace = np.trace(R_error)
        r_error = np.arccos(np.clip((trace - 1) / 2, -1, 1))
        r_error_deg = np.degrees(r_error)

        min_r_error_deg = min(min_r_error_deg, r_error_deg)

    # Use minimum rotation error for threshold checks
    r_error_deg = min_r_error_deg
    t_error_cm = t_error * 100  # Convert to cm

    return {
        'translation_error_m': float(t_error),
        'rotation_error_deg': float(r_error_deg),
        'success_5deg2cm': bool(r_error_deg < 5 and t_error_cm < 2),
        'success_5deg5cm': bool(r_error_deg < 5 and t_error_cm < 5),
        'success_10deg5cm': bool(r_error_deg < 10 and t_error_cm < 5),
        'success_10deg10cm': bool(r_error_deg < 10 and t_error_cm < 10)
    }


def compute_add_metric(est_R: np.ndarray, est_t: np.ndarray,
                      gt_R: np.ndarray, gt_t: np.ndarray,
                      model_points: np.ndarray,
                      diameter: Optional[float] = None,
                      symmetric: bool = False) -> Dict[str, float]:
    """
    Compute ADD (Average Distance of Model Points) metric.

    Standard metric for 6D pose estimation that measures the average distance
    between corresponding model points after transformation.

    Args:
        est_R: (3, 3) estimated rotation matrix
        est_t: (3,) estimated translation vector
        gt_R: (3, 3) ground truth rotation matrix
        gt_t: (3,) ground truth translation vector
        model_points: (N, 3) 3D model points in canonical frame
        diameter: Object diameter for ADD(-S) threshold computation
        symmetric: If True, use ADD-S (for symmetric objects)

    Returns:
        Dictionary with:
            - add_error: ADD error in meters
            - add_score: 1.0 if error < 0.1 * diameter, else 0.0
            - add_50: 1.0 if error < 0.05 * diameter, else 0.0
    """
    # Transform model points with estimated pose
    pts_est = (est_R @ model_points.T).T + est_t

    # Transform model points with GT pose
    pts_gt = (gt_R @ model_points.T).T + gt_t

    if symmetric:
        # ADD-S: Use closest point distance (for symmetric objects)
        from scipy.spatial import cKDTree
        tree = cKDTree(pts_est)
        distances, _ = tree.query(pts_gt, k=1)
        add_error = np.mean(distances)
    else:
        # ADD: Direct correspondence distance
        distances = np.linalg.norm(pts_est - pts_gt, axis=1)
        add_error = np.mean(distances)

    result = {
        'add_error': float(add_error)
    }

    # Compute threshold-based scores if diameter provided
    if diameter is not None:
        result['add_score'] = float(add_error < 0.1 * diameter)  # ADD-0.1d
        result['add_50'] = float(add_error < 0.05 * diameter)    # ADD-0.05d

    return result


def compute_3d_iou(est_R: np.ndarray, est_t: np.ndarray,
                   gt_R: np.ndarray, gt_t: np.ndarray,
                   model_points: np.ndarray,
                   symmetries: Optional[list] = None) -> Dict[str, float]:
    """
    Compute 3D bounding box IoU between estimated and GT poses with symmetry handling.

    This is a mAP-based metric that assesses 3D bounding box IoU,
    capturing both pose accuracy and object size consistency.

    For symmetric objects, tests all symmetry transformations and returns the
    maximum IoU (following NOCS evaluation protocol).

    The metric:
    1. Transforms model points with estimated pose → compute 3D bbox
    2. For each symmetry: transforms model points with GT pose + symmetry → compute 3D bbox
    3. Computes IoU between estimated bbox and each symmetry-transformed GT bbox
    4. Returns maximum IoU across all symmetries
    5. Reports IoU-50 and IoU-75 (standard thresholds)

    Args:
        est_R: (3, 3) estimated rotation matrix
        est_t: (3,) estimated translation vector
        gt_R: (3, 3) ground truth rotation matrix
        gt_t: (3,) ground truth translation vector
        model_points: (N, 3) 3D model points in canonical frame
        symmetries: Optional list of symmetry transformations in BOP format:
                   [{'R': (3,3), 't': (3,1)}, ...]. If None or empty, no symmetry.

    Returns:
        Dictionary with:
            - iou_3d: Maximum 3D bounding box IoU value [0, 1]
            - iou_3d_50: 1.0 if IoU >= 0.5, else 0.0
            - iou_3d_75: 1.0 if IoU >= 0.75, else 0.0
    """
    # Transform model points with estimated pose
    pts_est = (est_R @ model_points.T).T + est_t

    # Compute estimated bbox once (reused for all symmetries)
    min_est = np.min(pts_est, axis=0)  # (3,) [x_min, y_min, z_min]
    max_est = np.max(pts_est, axis=0)  # (3,) [x_max, y_max, z_max]
    est_dims = max_est - min_est
    est_volume = np.prod(est_dims)

    # Test all symmetries and find maximum IoU
    if symmetries is None or len(symmetries) == 0:
        # No symmetry - use standard computation
        symmetries = [{'R': np.eye(3), 't': np.zeros((3, 1))}]

    max_iou = 0.0
    for sym in symmetries:
        # Apply symmetry to GT pose
        R_gt_sym = gt_R @ sym['R']
        sym_t = np.asarray(sym.get('t', np.zeros((3, 1)))).flatten()
        t_gt_sym = gt_t + (gt_R @ sym_t)

        # Transform model points with symmetry-transformed GT pose
        pts_gt = (R_gt_sym @ model_points.T).T + t_gt_sym

        # Compute GT bbox for this symmetry
        min_gt = np.min(pts_gt, axis=0)
        max_gt = np.max(pts_gt, axis=0)

        # Compute intersection box
        min_inter = np.maximum(min_est, min_gt)
        max_inter = np.minimum(max_est, max_gt)

        # Compute intersection volume
        inter_dims = np.maximum(0, max_inter - min_inter)
        inter_volume = np.prod(inter_dims)

        # Compute GT volume
        gt_dims = max_gt - min_gt
        gt_volume = np.prod(gt_dims)

        # Compute union volume
        union_volume = est_volume + gt_volume - inter_volume

        # Compute IoU for this symmetry
        iou = inter_volume / union_volume if union_volume > 0 else 0.0
        max_iou = max(max_iou, iou)

    return {
        'iou_3d': float(max_iou),
        'iou_3d_50': float(max_iou >= 0.5),
        'iou_3d_75': float(max_iou >= 0.75)
    }


def compute_all_metrics(est_R: np.ndarray, est_t: np.ndarray,
                       gt_R: np.ndarray, gt_t: np.ndarray,
                       model_points: np.ndarray,
                       diameter: Optional[float] = None,
                       symmetries: Optional[list] = None) -> Dict[str, float]:
    """
    Compute all pose metrics in one call.

    Convenience function that computes rotation error, translation error,
    ADD, ADD-S, and 3D bounding box IoU metrics together.

    Args:
        est_R: (3, 3) estimated rotation matrix
        est_t: (3,) estimated translation vector
        gt_R: (3, 3) ground truth rotation matrix
        gt_t: (3,) ground truth translation vector
        model_points: (N, 3) 3D model points in canonical frame
        diameter: Object diameter for threshold-based metrics
        symmetries: Optional list of symmetry transformations in BOP format.
                   Used for symmetry-aware rotation error and 3D IoU computation.

    Returns:
        Dictionary with all metrics:
            - translation_error_m
            - rotation_error_deg (symmetry-aware if symmetries provided)
            - add_error: Regular ADD error (point-to-point distance)
            - add_score: ADD-10 success (if diameter provided)
            - add_50: ADD-0.05d success (if diameter provided)
            - adds_error: ADD-S error (nearest-neighbor distance for symmetric objects)
            - adds_score: ADD-S-10 success (if diameter provided)
            - adds_50: ADD-S-0.05d success (if diameter provided)
            - iou_3d (symmetry-aware if symmetries provided)
            - iou_3d_50
            - iou_3d_75

    Note: Both ADD and ADD-S are always computed. ADD-S implicitly handles
    symmetry via nearest-neighbor matching. Use rotation_error_deg and iou_3d
    with symmetries parameter for explicit symmetry handling.
    """
    metrics = {}

    # Basic pose errors (symmetry-aware rotation error)
    metrics.update(compute_pose_errors(est_R, est_t, gt_R, gt_t, symmetries=symmetries))

    # ADD metric (regular)
    add_metrics = compute_add_metric(
        est_R, est_t, gt_R, gt_t,
        model_points, diameter, symmetric=False
    )
    metrics.update(add_metrics)

    # ADD-S metric (symmetric)
    adds_metrics = compute_add_metric(
        est_R, est_t, gt_R, gt_t,
        model_points, diameter, symmetric=True
    )
    # Rename keys to distinguish from ADD
    metrics['adds_error'] = adds_metrics['add_error']
    if diameter is not None:
        metrics['adds_score'] = adds_metrics['add_score']
        metrics['adds_50'] = adds_metrics['add_50']

    # 3D bounding box IoU metric (symmetry-aware)
    metrics.update(compute_3d_iou(
        est_R, est_t, gt_R, gt_t,
        model_points, symmetries=symmetries
    ))

    return metrics


def compute_auc_from_errors(errors: list, max_val: float = 0.1) -> float:
    """
    Compute Area Under Curve (AUC) for pose estimation errors.

    This metric computes the area under the precision-recall curve up to a
    maximum error threshold. It provides a more comprehensive assessment than
    threshold-based metrics by considering performance across all error levels.

    Implementation follows the BOP toolkit and One2Any methodology:
    - Sort errors in ascending order
    - Compute precision at each recall level (recall = rank / total)
    - Filter to errors < max_val
    - Build monotonic precision-recall curve
    - Integrate area and normalize by max_val

    Args:
        errors: List of error values (ADD, ADD-S, etc.) in meters
        max_val: Maximum error threshold for AUC computation (default: 0.1m)

    Returns:
        AUC score between 0 and 1, where:
        - 1.0 = perfect (all errors are 0)
        - 0.0 = all errors exceed max_val
        - Higher is better

    Example:
        >>> errors = [0.02, 0.05, 0.08, 0.12, 0.15]  # 3/5 under 0.1m
        >>> auc = compute_auc_from_errors(errors, max_val=0.1)
        >>> # AUC ≈ 0.65 (accounts for error magnitudes, not just pass/fail)
    """
    if not errors or len(errors) == 0:
        return 0.0

    # Sort errors in ascending order
    rec = np.sort(np.array(errors))
    n = len(rec)

    # Compute precision at each recall level
    # recall = rank / total, precision = rank / total (since we sort by error)
    prec = np.arange(1, n + 1) / float(n)

    # Filter to errors within max_val threshold
    index = np.where(rec < max_val)[0]
    rec = rec[index]
    prec = prec[index]

    if len(rec) == 0:
        return 0.0

    # Build precision-recall curve endpoints
    # Start: (0, 0) - no recall, no precision
    # End: (max_val, prec[-1]) - extend last precision to max_val
    mrec = np.array([0, *list(rec), max_val])
    mpre = np.array([0, *list(prec), prec[-1]])

    # Make precision monotonic (ensures valid precision-recall curve)
    # Each precision value is max of itself and all previous values
    for i in range(1, len(mpre)):
        mpre[i] = max(mpre[i], mpre[i - 1])

    # Compute area under curve using trapezoidal integration
    # Find unique recall values (where curve changes)
    i = np.where(mrec[1:] != mrec[:-1])[0] + 1

    # Sum areas: (recall_diff × precision) for each segment
    ap = np.sum((mrec[i] - mrec[i - 1]) * mpre[i]) / max_val

    return float(ap)
