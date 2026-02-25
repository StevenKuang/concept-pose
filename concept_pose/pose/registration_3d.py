"""
3D-3D Point Cloud Registration for Pose Estimation

This module provides 3D-to-3D registration algorithms as an alternative to 2D-3D PnP
when depth maps are available. Uses semantic features for correspondence matching
and geometric algorithms (Umeyama, RANSAC, ICP) for robust pose estimation.

Main functions:
- backproject_depth: Convert depth map to 3D point cloud
- find_correspondences_3d: Semantic matching in 3D space
- ransac_3d_registration: Robust 3D-3D alignment with RANSAC
- umeyama_alignment: Closed-form 3D-3D alignment
- icp_refinement: Iterative refinement of pose
"""

import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial import cKDTree


def backproject_depth(depth_map, saliency_map, camera_matrix, device='cuda'):
    """
    Back-project 2D depth map to 3D point cloud.

    Args:
        depth_map: (H, W) depth in meters (numpy or torch)
        saliency_map: (C, H, W) semantic features (numpy or torch)
        camera_matrix: (3, 3) camera intrinsics
        device: torch device for computation

    Returns:
        points_3d: (N, 3) 3D points in camera frame
        saliencies: (N, C) semantic features
        pixel_coords: (N, 2) original pixel coordinates [x, y]
    """
    # Convert to torch if needed
    if not isinstance(depth_map, torch.Tensor):
        depth_map = torch.from_numpy(depth_map).float().to(device)
    elif depth_map.device != torch.device(device):
        depth_map = depth_map.to(device)

    if not isinstance(saliency_map, torch.Tensor):
        saliency_map = torch.from_numpy(saliency_map).float().to(device)
    elif saliency_map.device != torch.device(device):
        saliency_map = saliency_map.to(device)

    C, H, W = saliency_map.shape

    # Camera intrinsics
    if isinstance(camera_matrix, np.ndarray):
        camera_matrix = torch.from_numpy(camera_matrix).float().to(device)
    fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
    cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]

    # Find valid pixels (non-zero saliency AND valid depth > 0)
    valid_mask = (torch.any(saliency_map != 0, dim=0)) & (depth_map > 0)
    valid_coords = torch.nonzero(valid_mask, as_tuple=False)  # (N, 2) [y, x]

    if len(valid_coords) == 0:
        return None, None, None

    # Extract coordinates
    y_coords = valid_coords[:, 0].float()
    x_coords = valid_coords[:, 1].float()

    # Get depth and saliency at valid pixels
    depths = depth_map[valid_coords[:, 0], valid_coords[:, 1]]  # (N,)
    saliencies = saliency_map[:, valid_coords[:, 0], valid_coords[:, 1]].t()  # (N, C)

    # Back-project to 3D camera coordinates
    x_3d = (x_coords - cx) * depths / fx
    y_3d = (y_coords - cy) * depths / fy
    z_3d = depths

    points_3d = torch.stack([x_3d, y_3d, z_3d], dim=1)  # (N, 3)
    pixel_coords = torch.stack([x_coords, y_coords], dim=1)  # (N, 2) [x, y]

    return points_3d, saliencies, pixel_coords


def find_correspondences_3d(observed_3d, observed_sal, model_3d, model_sal,
                           similarity_threshold=-2.0, max_correspondences=500,
                           loss_method='kl_divergence', temperature=1.0,
                           lambda_reverse=0.5, device='cuda'):
    """
    Find 3D-3D correspondences using semantic similarity.

    Similar to 2D correspondence matching but works in 3D space.
    Uses same semantic matching logic (KL divergence) as 2D pipeline.

    Args:
        observed_3d: (N, 3) observed 3D points from depth
        observed_sal: (N, C) observed semantic features
        model_3d: (M, 3) model 3D points (voxels in NOCS space)
        model_sal: (M, C) model semantic features
        similarity_threshold: Threshold for semantic similarity
        max_correspondences: Max number of correspondences to return
        loss_method: Semantic matching method ('kl_divergence', 'reverse_kl', 'bidirectional_kl',
                     'jensen_shannon', 'weighted_cosine', 'cosine', 'asymmetric')
        temperature: Temperature for KL divergence softmax (lower = sharper, default: 1.0)
        lambda_reverse: Weight for reverse KL in bidirectional method (default: 0.5)
        device: torch device

    Returns:
        matched_observed: (K, 3) matched observed points
        matched_model: (K, 3) matched model points
        match_scores: (K,) semantic similarity scores
    """
    # Import here to avoid circular dependency
    from concept_pose.pose.loss import compute_correspondence_scores

    # Ensure tensors on correct device
    if not isinstance(observed_sal, torch.Tensor):
        observed_sal = torch.from_numpy(observed_sal).float().to(device)
    if not isinstance(model_sal, torch.Tensor):
        model_sal = torch.from_numpy(model_sal).float().to(device)
    if not isinstance(observed_3d, torch.Tensor):
        observed_3d = torch.from_numpy(observed_3d).float().to(device)
    if not isinstance(model_3d, torch.Tensor):
        model_3d = torch.from_numpy(model_3d).float().to(device)

    # Compute confidence scores for observed points
    strength = torch.norm(observed_sal, dim=1)  # (N,)
    max_channel = torch.max(observed_sal, dim=1)[0]
    mean_channel = torch.mean(observed_sal, dim=1)
    distinctiveness = max_channel / (mean_channel + 1e-8)
    confidence_scores = strength * distinctiveness

    # Select top points by confidence
    num_to_sample = min(max_correspondences, len(observed_3d))
    sorted_indices = torch.argsort(confidence_scores, descending=True)
    selected_indices = sorted_indices[:num_to_sample]

    selected_sal = observed_sal[selected_indices]  # (M, C)
    selected_3d = observed_3d[selected_indices]  # (M, 3)

    # Compute semantic similarity scores (batched for memory efficiency)
    batch_size = 260  # Same as 2D pipeline
    all_best_scores = []
    all_best_indices = []

    for batch_start in range(0, len(selected_sal), batch_size):
        batch_end = min(batch_start + batch_size, len(selected_sal))
        batch_sal = selected_sal[batch_start:batch_end]  # (B, C)

        # Compute scores for this batch
        batch_scores = compute_correspondence_scores(
            batch_sal,    # (B, C)
            model_sal,    # (M, C)
            method=loss_method,
            temperature=temperature,  # Pass temperature for KL divergence
            lambda_reverse=lambda_reverse,  # Pass lambda for bidirectional KL
            return_costs=False  # Return similarities
        )  # (B, M)

        # Find best match for each point
        batch_best_scores, batch_best_indices = torch.max(batch_scores, dim=1)
        all_best_scores.append(batch_best_scores)
        all_best_indices.append(batch_best_indices)

        del batch_scores
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Concatenate results
    best_scores = torch.cat(all_best_scores, dim=0)  # (M,)
    best_indices = torch.cat(all_best_indices, dim=0)  # (M,)

    # Debug: Print score statistics
    median_score = best_scores.median().item()
    print(f"    [Correspondence Matching]")
    print(f"      Similarity scores: min={best_scores.min().item():.4f}, "
          f"max={best_scores.max().item():.4f}, "
          f"mean={best_scores.mean().item():.4f}, "
          f"median={median_score:.4f}")

    # Auto-detect median threshold: if similarity_threshold >= 0, use median instead
    # (since valid KL divergence similarities are always negative, >= 0 is a sentinel)
    # Also support string "median" for clarity
    use_median = (isinstance(similarity_threshold, str) and similarity_threshold.lower() == 'median') or \
                 (isinstance(similarity_threshold, (int, float)) and similarity_threshold >= 0)

    if use_median:
        similarity_threshold = median_score
        print(f"      Similarity threshold: {similarity_threshold:.4f} (auto-median)")
    else:
        print(f"      Similarity threshold: {similarity_threshold:.4f}")

    # Threshold filtering (cost-based: higher score = lower cost = better)
    cost_threshold = abs(similarity_threshold) if similarity_threshold < 0 else 2.0
    passes_threshold = best_scores > -cost_threshold
    valid_matches = torch.nonzero(passes_threshold, as_tuple=False).squeeze(1)

    print(f"      Cost threshold: -{cost_threshold:.4f}")
    print(f"      Matches before filtering: {len(best_scores)}")
    print(f"      Matches after filtering: {len(valid_matches)} ({100*len(valid_matches)/len(best_scores):.1f}%)")

    if len(valid_matches) == 0:
        return None, None, None

    # Extract matched pairs
    matched_observed = selected_3d[valid_matches]  # (K, 3)
    matched_model_indices = best_indices[valid_matches]
    matched_model = model_3d[matched_model_indices]  # (K, 3)
    match_scores = best_scores[valid_matches]  # (K,)

    return matched_observed, matched_model, match_scores


def umeyama_alignment(src_points, dst_points, estimate_scale=False):
    """
    Umeyama algorithm: Closed-form 3D-3D point cloud alignment.

    Finds optimal R, t, s such that: dst = s * R @ src + t

    Args:
        src_points: (N, 3) source points (model)
        dst_points: (N, 3) destination points (observed)
        estimate_scale: If True, estimate scale; if False, scale = 1.0

    Returns:
        R: (3, 3) rotation matrix
        t: (3,) translation vector
        s: scalar scale factor
    """
    # Convert to numpy if needed
    if isinstance(src_points, torch.Tensor):
        src_points = src_points.cpu().numpy()
    if isinstance(dst_points, torch.Tensor):
        dst_points = dst_points.cpu().numpy()

    assert src_points.shape == dst_points.shape
    assert src_points.shape[0] >= 3, "Need at least 3 points"

    # Compute centroids
    src_mean = np.mean(src_points, axis=0)
    dst_mean = np.mean(dst_points, axis=0)

    # Center the points
    src_centered = src_points - src_mean
    dst_centered = dst_points - dst_mean

    # Compute scale
    if estimate_scale:
        src_scale = np.sqrt(np.mean(np.sum(src_centered ** 2, axis=1)))
        dst_scale = np.sqrt(np.mean(np.sum(dst_centered ** 2, axis=1)))
        s = dst_scale / (src_scale + 1e-10)
    else:
        s = 1.0
        src_scale = 1.0
        dst_scale = 1.0

    # Normalize for rotation estimation
    src_normalized = src_centered / (src_scale + 1e-10)
    dst_normalized = dst_centered / (dst_scale + 1e-10)

    # Compute cross-covariance matrix
    H = src_normalized.T @ dst_normalized  # (3, 3)

    # SVD
    U, S, Vt = np.linalg.svd(H)

    # Compute rotation (handle reflection)
    R = Vt.T @ U.T

    # Ensure proper rotation (det(R) = +1, not -1)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    # Compute translation
    t = dst_mean - s * R @ src_mean

    return R, t, s


def ransac_3d_registration(observed_3d, model_3d, observed_sal, model_sal, config, device='cuda'):
    """
    RANSAC for robust 3D-3D point cloud registration.

    Supports both CPU (original) and GPU-accelerated (batched) implementations.
    Use config['use_gpu_ransac'] = True for 50-100x speedup on GPU.

    Args:
        observed_3d: (N, 3) observed 3D points
        model_3d: (M, 3) model 3D points (NOCS space)
        observed_sal: (N, C) observed saliencies
        model_sal: (M, C) model saliencies
        config: Configuration dict with RANSAC parameters:
            - use_gpu_ransac (bool): Use GPU-accelerated batched RANSAC (default: False)
            - ransac_batch_size (int): Batch size for GPU RANSAC (default: 1024)
            - ransac_iterations_3d (int): Number of RANSAC iterations
            - ransac_3d_threshold (float): Inlier threshold in meters
            - estimate_scale (bool): Whether to estimate scale
        device: torch device

    Returns:
        success: bool
        R: (3, 3) rotation matrix
        t: (3,) translation vector
        s: scalar scale
        inlier_mask: (K,) boolean mask of inliers
        correspondences: tuple (matched_observed, matched_model)
    """
    # Check if GPU RANSAC is requested
    use_gpu_ransac = config.get('use_gpu_ransac', False)

    # Convert torch.device to string if needed
    device_str = str(device) if isinstance(device, torch.device) else device

    if use_gpu_ransac and 'cuda' in device_str:
        # Use GPU-accelerated batched RANSAC
        from concept_pose.pose.ransac_cuda import ransac_3d_registration_cuda
        return ransac_3d_registration_cuda(
            observed_3d, model_3d, observed_sal, model_sal, config, device
        )

    # Fall back to CPU implementation

    # First, find semantic correspondences
    loss_method = config.get('loss_method', 'kl_divergence')
    temperature = config.get('temperature', 1.0)

    if loss_method == 'kl_divergence':
        print(f"  KL divergence temperature: {temperature}")

    matched_observed, matched_model, match_scores = find_correspondences_3d(
        observed_3d, observed_sal, model_3d, model_sal,
        similarity_threshold=config.get('similarity_threshold', -2.0),
        max_correspondences=config.get('max_correspondences', 500),
        loss_method=loss_method,
        temperature=temperature,
        lambda_reverse=config.get('lambda_reverse', 0.5),
        device=device
    )

    if matched_observed is None or len(matched_observed) < 3:
        return False, None, None, None, None, None

    # Convert to numpy for RANSAC
    obs_np = matched_observed.cpu().numpy() if isinstance(matched_observed, torch.Tensor) else matched_observed
    model_np = matched_model.cpu().numpy() if isinstance(matched_model, torch.Tensor) else matched_model

    # RANSAC parameters
    iterations = config.get('ransac_iterations_3d', 2000)
    inlier_threshold = config.get('ransac_3d_threshold', 0.01)  # 1cm
    estimate_scale = config.get('estimate_scale', False)

    best_inliers = 0
    best_R, best_t, best_s = None, None, None
    best_inlier_mask = None

    for _ in range(iterations):
        # Randomly sample 3 correspondences
        if len(obs_np) < 3:
            continue

        sample_indices = np.random.choice(len(obs_np), 3, replace=False)
        sample_obs = obs_np[sample_indices]
        sample_model = model_np[sample_indices]

        try:
            # Compute alignment from sample
            R, t, s = umeyama_alignment(sample_model, sample_obs, estimate_scale)

            # Transform all model points
            transformed_model = s * (R @ model_np.T).T + t  # (K, 3)

            # Compute distances to observed points
            distances = np.linalg.norm(transformed_model - obs_np, axis=1)

            # Count inliers
            inlier_mask = distances < inlier_threshold
            num_inliers = np.sum(inlier_mask)

            # Update best
            if num_inliers > best_inliers:
                best_inliers = num_inliers
                best_R, best_t, best_s = R, t, s
                best_inlier_mask = inlier_mask

        except (np.linalg.LinAlgError, ValueError):
            continue

    if best_R is None:
        return False, None, None, None, None, None

    # Refine with all inliers
    if best_inliers >= 3:
        inlier_obs = obs_np[best_inlier_mask]
        inlier_model = model_np[best_inlier_mask]

        try:
            R_refined, t_refined, s_refined = umeyama_alignment(
                inlier_model, inlier_obs, estimate_scale
            )
            best_R, best_t, best_s = R_refined, t_refined, s_refined
        except (np.linalg.LinAlgError, ValueError):
            pass

    # Debug: print estimated scale
    print(f"    [RANSAC] estimate_scale={estimate_scale}, final scale s={best_s:.6f}, inliers={best_inliers}")

    return True, best_R, best_t, best_s, best_inlier_mask, (matched_observed, matched_model)


def icp_refinement(observed_3d, model_3d, R_init, t_init, s_init, config):
    """
    Iterative Closest Point (ICP) refinement of 3D pose.

    Args:
        observed_3d: (N, 3) observed 3D points (torch or numpy)
        model_3d: (M, 3) model 3D points (torch or numpy)
        R_init: (3, 3) initial rotation
        t_init: (3,) initial translation
        s_init: initial scale
        config: Configuration dict with ICP parameters

    Returns:
        R: (3, 3) refined rotation
        t: (3,) refined translation
        s: refined scale
        inlier_rmse: RMSE of inliers after final refinement
    """
    # Convert to numpy
    if isinstance(observed_3d, torch.Tensor):
        observed_3d = observed_3d.cpu().numpy()
    if isinstance(model_3d, torch.Tensor):
        model_3d = model_3d.cpu().numpy()

    R, t, s = R_init.copy(), t_init.copy(), s_init

    max_iters = config.get('icp_max_iters', 50)
    convergence_threshold = config.get('icp_convergence', 0.0001)
    distance_threshold = config.get('icp_distance_threshold', 0.02)  # 2cm
    estimate_scale = config.get('estimate_scale', False)

    # Build KD-tree for observed points (for fast nearest neighbor)
    tree = cKDTree(observed_3d)

    final_inlier_rmse = float('inf')

    for iteration in range(max_iters):
        # Transform model points with current estimate
        transformed_model = s * (R @ model_3d.T).T + t  # (M, 3)

        # Find nearest neighbors in observed
        distances, indices = tree.query(transformed_model, k=1)

        # Filter by distance threshold
        valid_mask = distances < distance_threshold

        if np.sum(valid_mask) < 3:
            break  # Too few correspondences

        # Compute RMSE of inliers
        inlier_distances = distances[valid_mask]
        final_inlier_rmse = np.sqrt(np.mean(inlier_distances ** 2))

        # Get matched pairs
        matched_model = model_3d[valid_mask]
        matched_observed = observed_3d[indices[valid_mask]]

        # Compute alignment
        try:
            R_new, t_new, s_new = umeyama_alignment(
                matched_model, matched_observed, estimate_scale
            )
        except (np.linalg.LinAlgError, ValueError):
            break

        # Check convergence
        pose_change = np.linalg.norm(R_new - R) + np.linalg.norm(t_new - t) + abs(s_new - s)

        R, t, s = R_new, t_new, s_new

        if pose_change < convergence_threshold:
            break

    # Debug: print ICP result
    print(f"    [ICP] estimate_scale={estimate_scale}, final scale s={s:.6f}")

    return R, t, s, final_inlier_rmse
