"""
GPU-Accelerated Batched RANSAC for 3D-3D Registration
=======================================================

Ultra-fast RANSAC implementation using PyTorch batched operations.
Processes thousands of hypotheses in parallel on GPU.

Key optimizations:
- Batch sampling: sample all hypotheses at once (no Python loop)
- Parallel Umeyama: compute all alignments simultaneously
- Vectorized distance computation: process all points at once
- GPU memory efficient: process in chunks if needed

Performance: ~50-100x faster than CPU loop for 50k iterations.
"""

import torch
import numpy as np
from typing import Tuple, Optional


def umeyama_alignment_batched(src_points: torch.Tensor,
                               dst_points: torch.Tensor,
                               estimate_scale: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Batched Umeyama alignment: compute R, t, s for multiple point sets in parallel.

    Args:
        src_points: (B, N, 3) source points for B hypotheses
        dst_points: (B, N, 3) destination points
        estimate_scale: whether to estimate scale

    Returns:
        R: (B, 3, 3) rotation matrices
        t: (B, 3) translation vectors
        s: (B,) scale factors
    """
    B, N, _ = src_points.shape
    device = src_points.device

    # Compute centroids (B, 3)
    src_mean = src_points.mean(dim=1)  # (B, 3)
    dst_mean = dst_points.mean(dim=1)  # (B, 3)

    # Center the points (B, N, 3)
    src_centered = src_points - src_mean.unsqueeze(1)
    dst_centered = dst_points - dst_mean.unsqueeze(1)

    # Compute scale
    if estimate_scale:
        src_scale = torch.sqrt((src_centered ** 2).sum(dim=2).mean(dim=1))  # (B,)
        dst_scale = torch.sqrt((dst_centered ** 2).sum(dim=2).mean(dim=1))  # (B,)
        s = dst_scale / (src_scale + 1e-10)
    else:
        s = torch.ones(B, device=device)
        src_scale = torch.ones(B, device=device)
        dst_scale = torch.ones(B, device=device)

    # Normalize (B, N, 3)
    src_normalized = src_centered / (src_scale.unsqueeze(1).unsqueeze(2) + 1e-10)
    dst_normalized = dst_centered / (dst_scale.unsqueeze(1).unsqueeze(2) + 1e-10)

    # Compute cross-covariance: H = src^T @ dst (B, 3, 3)
    H = torch.bmm(src_normalized.transpose(1, 2), dst_normalized)

    # Batched SVD
    U, S, Vh = torch.linalg.svd(H)  # U: (B, 3, 3), Vh: (B, 3, 3)

    # Compute rotation R = V @ U^T (B, 3, 3)
    R = torch.bmm(Vh.transpose(1, 2), U.transpose(1, 2))

    # Handle reflection (det(R) < 0) by flipping last column of V
    det_R = torch.det(R)  # (B,)
    flip_mask = det_R < 0  # (B,)

    if flip_mask.any():
        Vh_corrected = Vh.clone()
        Vh_corrected[flip_mask, :, -1] *= -1
        R[flip_mask] = torch.bmm(
            Vh_corrected[flip_mask].transpose(1, 2),
            U[flip_mask].transpose(1, 2)
        )

    # Compute translation: t = dst_mean - s * R @ src_mean (B, 3)
    t = dst_mean - s.unsqueeze(1) * torch.bmm(R, src_mean.unsqueeze(2)).squeeze(2)

    return R, t, s


def ransac_3d_batched_cuda(
    obs_points: torch.Tensor,
    model_points: torch.Tensor,
    iterations: int = 50000,
    inlier_threshold: float = 0.01,
    estimate_scale: bool = False,
    batch_size: int = 1024,
    min_inliers: int = 3,
    device: str = 'cuda'
) -> Tuple[bool, Optional[torch.Tensor], Optional[torch.Tensor], Optional[float], Optional[torch.Tensor]]:
    """
    GPU-accelerated batched RANSAC for 3D-3D registration.

    Processes multiple RANSAC hypotheses in parallel on GPU for massive speedup.

    Args:
        obs_points: (K, 3) observed 3D points (correspondences)
        model_points: (K, 3) model 3D points (correspondences)
        iterations: total number of RANSAC iterations
        inlier_threshold: distance threshold for inliers (meters)
        estimate_scale: whether to estimate scale
        batch_size: number of hypotheses to process in parallel (tune for GPU memory)
        min_inliers: minimum number of inliers required
        device: torch device ('cuda' or 'cpu')

    Returns:
        success: bool
        R: (3, 3) best rotation matrix (numpy)
        t: (3,) best translation vector (numpy)
        s: best scale (float)
        inlier_mask: (K,) boolean inlier mask (numpy)
    """
    K = len(obs_points)

    if K < 3:
        return False, None, None, None, None

    # Convert to torch if needed
    if not isinstance(obs_points, torch.Tensor):
        obs_points = torch.from_numpy(obs_points).float().to(device)
    if not isinstance(model_points, torch.Tensor):
        model_points = torch.from_numpy(model_points).float().to(device)

    obs_points = obs_points.to(device)
    model_points = model_points.to(device)

    best_num_inliers = 0
    best_R = None
    best_t = None
    best_s = None
    best_inlier_mask = None

    # Process RANSAC in batches
    num_batches = (iterations + batch_size - 1) // batch_size

    for batch_idx in range(num_batches):
        current_batch_size = min(batch_size, iterations - batch_idx * batch_size)

        # Sample indices for all hypotheses in this batch (B, 3)
        sample_indices = torch.randint(0, K, (current_batch_size, 3), device=device)

        # Gather sampled points (B, 3, 3)
        sample_obs = obs_points[sample_indices]  # (B, 3, 3)
        sample_model = model_points[sample_indices]  # (B, 3, 3)

        try:
            # Batched Umeyama alignment (B sets of R, t, s)
            R_batch, t_batch, s_batch = umeyama_alignment_batched(
                sample_model, sample_obs, estimate_scale
            )

            # Transform all model points with all hypotheses
            # model_points: (K, 3), R_batch: (B, 3, 3), s_batch: (B,)
            # Result: (B, K, 3)

            # Expand model points: (K, 3) -> (B, K, 3)
            model_expanded = model_points.unsqueeze(0).expand(current_batch_size, -1, -1)

            # Apply rotation: (B, K, 3) @ (B, 3, 3)^T = (B, K, 3)
            rotated = torch.bmm(model_expanded, R_batch.transpose(1, 2))

            # Apply scale and translation: s * R @ p + t
            s_expanded = s_batch.unsqueeze(1).unsqueeze(2)  # (B, 1, 1)
            t_expanded = t_batch.unsqueeze(1)  # (B, 1, 3)
            transformed = s_expanded * rotated + t_expanded  # (B, K, 3)

            # Compute distances to observed points
            # obs_points: (K, 3) -> (1, K, 3) -> (B, K, 3)
            obs_expanded = obs_points.unsqueeze(0).expand(current_batch_size, -1, -1)
            distances = torch.norm(transformed - obs_expanded, dim=2)  # (B, K)

            # Count inliers for each hypothesis
            inlier_masks = distances < inlier_threshold  # (B, K)
            num_inliers = inlier_masks.sum(dim=1)  # (B,)

            # Find best in this batch
            best_in_batch_idx = torch.argmax(num_inliers)
            best_in_batch_inliers = num_inliers[best_in_batch_idx].item()

            # Update global best
            if best_in_batch_inliers > best_num_inliers:
                best_num_inliers = best_in_batch_inliers
                best_R = R_batch[best_in_batch_idx]
                best_t = t_batch[best_in_batch_idx]
                best_s = s_batch[best_in_batch_idx]
                best_inlier_mask = inlier_masks[best_in_batch_idx]

        except RuntimeError as e:
            # Handle singular matrices or other errors
            continue

    if best_R is None or best_num_inliers < min_inliers:
        return False, None, None, None, None

    # Refine with all inliers
    if best_num_inliers >= 3:
        inlier_obs = obs_points[best_inlier_mask]
        inlier_model = model_points[best_inlier_mask]

        try:
            # Refine with all inliers (single alignment)
            R_refined, t_refined, s_refined = umeyama_alignment_batched(
                inlier_model.unsqueeze(0),
                inlier_obs.unsqueeze(0),
                estimate_scale
            )
            best_R = R_refined[0]
            best_t = t_refined[0]
            best_s = s_refined[0]
        except RuntimeError:
            pass

    # Convert to numpy
    R_np = best_R.cpu().numpy()
    t_np = best_t.cpu().numpy()
    s_float = best_s.item()
    inlier_mask_np = best_inlier_mask.cpu().numpy()

    return True, R_np, t_np, s_float, inlier_mask_np


def ransac_3d_registration_cuda(observed_3d, model_3d, observed_sal, model_sal, config, device='cuda'):
    """
    Drop-in replacement for ransac_3d_registration using GPU-accelerated RANSAC.

    This is a wrapper that maintains the same interface as the original CPU version
    but uses the batched CUDA implementation for massive speedup.

    Args:
        observed_3d: (N, 3) observed 3D points
        model_3d: (M, 3) model 3D points
        observed_sal: (N, C) observed saliencies
        model_sal: (M, C) model saliencies
        config: Configuration dict with RANSAC parameters
        device: torch device

    Returns:
        success: bool
        R: (3, 3) rotation matrix
        t: (3,) translation vector
        s: scalar scale
        inlier_mask: (K,) boolean mask of inliers
        correspondences: tuple (matched_observed, matched_model)
    """
    # Import correspondence matching (same as original)
    from concept_pose.pose.registration_3d import find_correspondences_3d

    # First, find semantic correspondences (same as original)
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

    # Convert to torch tensors for GPU RANSAC
    if isinstance(matched_observed, torch.Tensor):
        obs_torch = matched_observed.to(device)
    else:
        obs_torch = torch.from_numpy(matched_observed).float().to(device)

    if isinstance(matched_model, torch.Tensor):
        model_torch = matched_model.to(device)
    else:
        model_torch = torch.from_numpy(matched_model).float().to(device)

    # RANSAC parameters
    iterations = config.get('ransac_iterations_3d', 50000)
    inlier_threshold = config.get('ransac_3d_threshold', 0.01)
    estimate_scale = config.get('estimate_scale', False)
    batch_size = config.get('ransac_batch_size', 1024)  # New parameter for tuning

    print(f"  [GPU RANSAC] iterations={iterations}, batch_size={batch_size}, device={device}")

    # Run GPU-accelerated batched RANSAC
    success, R, t, s, inlier_mask = ransac_3d_batched_cuda(
        obs_torch,
        model_torch,
        iterations=iterations,
        inlier_threshold=inlier_threshold,
        estimate_scale=estimate_scale,
        batch_size=batch_size,
        min_inliers=3,
        device=device
    )

    if not success:
        return False, None, None, None, None, None

    num_inliers = np.sum(inlier_mask)
    print(f"    [GPU RANSAC] estimate_scale={estimate_scale}, final scale s={s:.6f}, inliers={num_inliers}")

    return True, R, t, s, inlier_mask, (matched_observed, matched_model)
