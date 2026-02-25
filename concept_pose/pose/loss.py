"""
Semantic Saliency Loss Functions for 6D Pose Estimation
=========================================================

This module implements improved loss functions that address the issues with
cosine similarity identified in the meeting:

1. Low-low saliency matches should not contribute (currently they give similarity=1.0)
2. High-low mismatches should be heavily penalized
3. Focus on avoiding bad matches rather than just finding good ones
4. Think in terms of cost minimization rather than similarity maximization

The key insight is that cosine similarity ignores magnitudes, treating
[0.01, 0.01] • [0.01, 0.01] = 1.0 as perfect match, which is incorrect.

Copied from concept_pose/semantic_loss.py with exact functionality preserved.
"""

import torch
import torch.nn.functional as F


def compute_mask_iou(mask1, mask2):
    """
    Compute Intersection over Union (IoU) for binary masks.

    This measures spatial overlap between two binary masks, providing a geometric
    constraint that's particularly sensitive to scale errors (depth misalignment).

    Args:
        mask1, mask2: Binary masks, either torch.Tensor or numpy arrays of shape (H, W)

    Returns:
        iou: Scalar IoU value in [0, 1], where 1 = perfect overlap, 0 = no overlap
    """
    intersection = torch.sum(mask1 & mask2).float()
    union = torch.sum(mask1 | mask2).float()

    # Handle edge case of both masks being empty
    # Use .item() to convert CUDA tensor to Python scalar
    if union.item() == 0:
        return 1.0 if intersection.item() == 0 else 0.0

    return (intersection / union).item()


def soft_rasterize_voxels(voxel_coords_2d, voxel_saliencies, depths, H, W,
                          sigma=1.5, depth_scale=0.1, sparse_radius=3.0, voxel_batch_size=50):
    """
    Differentiable soft rasterization using Gaussian splatting with depth weighting.

    Each voxel projects to a 2D location and contributes to nearby pixels using:
    1. Spatial Gaussian weight: exp(-distance^2 / (2*sigma^2))
    2. Depth weight: exp(-depth / depth_scale) - closer voxels contribute more

    This is fully differentiable for gradient-based pose optimization.

    Args:
        voxel_coords_2d: (N, 2) float tensor - projected 2D coordinates (x, y) in pixels
        voxel_saliencies: (N, C) float tensor - raw saliency vectors
        depths: (N,) float tensor - depth values (z-coordinate in camera frame)
        H, W: int - output image dimensions
        sigma: float - Gaussian spatial spread in pixels (default 1.5)
        depth_scale: float - depth weight decay rate (default 0.1)
        sparse_radius: float - only affect pixels within this many sigmas (default 3.0)
        voxel_batch_size: int - process voxels in batches to reduce memory (default 500)
                         Memory usage: O(voxel_batch_size * H * W) instead of O(N * H * W)

    Returns:
        rendered: (C, H, W) tensor - soft-rasterized saliency map

    Implementation:
        - Batched: Process voxels in batches to keep memory usage bounded
        - Sparse: Only affects pixels within sparse_radius * sigma of each voxel
        - Weighted averaging: Each pixel = weighted sum of nearby voxels
        - Fully differentiable: gradients flow through all computations
    """

    if not isinstance(voxel_coords_2d, torch.Tensor):
        voxel_coords_2d = torch.from_numpy(voxel_coords_2d).float()
    if not isinstance(voxel_saliencies, torch.Tensor):
        voxel_saliencies = torch.from_numpy(voxel_saliencies).float()
    if not isinstance(depths, torch.Tensor):
        depths = torch.from_numpy(depths).float()

    device = voxel_coords_2d.device
    N, C = voxel_saliencies.shape

    # Initialize output: (C, H, W)
    rendered = torch.zeros(C, H, W, device=device, dtype=voxel_saliencies.dtype)
    weights_sum = torch.zeros(H, W, device=device, dtype=voxel_saliencies.dtype)

    # Compute depth weights for all voxels: closer = more weight
    # Shape: (N,)
    depth_weights = torch.exp(-depths / depth_scale)

    # Determine affected pixel range for each voxel (sparse implementation)
    radius_pixels = int(torch.ceil(torch.tensor(sparse_radius * sigma)).item())

    # Create pixel grid coordinates (shared across all batches)
    y_grid, x_grid = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing='ij'
    )
    pixel_coords = torch.stack([x_grid, y_grid], dim=-1)  # (H, W, 2)
    pixel_coords_expanded = pixel_coords.unsqueeze(0)  # (1, H, W, 2)

    # Process voxels in batches to reduce memory usage
    # Memory: O(voxel_batch_size * H * W) instead of O(N * H * W)
    num_batches = (N + voxel_batch_size - 1) // voxel_batch_size

    for batch_idx in range(num_batches):
        start_idx = batch_idx * voxel_batch_size
        end_idx = min(start_idx + voxel_batch_size, N)

        # Get batch of voxels
        batch_coords = voxel_coords_2d[start_idx:end_idx]  # (B, 2)
        batch_saliencies = voxel_saliencies[start_idx:end_idx]  # (B, C)
        batch_depth_weights = depth_weights[start_idx:end_idx]  # (B,)
        B = batch_coords.shape[0]

        # Reshape for broadcasting: batch_coords (B, 1, 1, 2), pixel_coords (1, H, W, 2)
        batch_coords_expanded = batch_coords.unsqueeze(1).unsqueeze(1)  # (B, 1, 1, 2)

        # Compute squared distances: (B, H, W)
        sq_distances = torch.sum((batch_coords_expanded - pixel_coords_expanded) ** 2, dim=-1)

        # Gaussian spatial weights: (B, H, W)
        spatial_weights = torch.exp(-sq_distances / (2 * sigma ** 2))

        # Combined weights: spatial * depth (broadcast from (B,) to (B, H, W))
        combined_weights = spatial_weights * batch_depth_weights.unsqueeze(1).unsqueeze(1)  # (B, H, W)

        # Sparsity: zero out weights beyond sparse_radius
        mask = sq_distances <= (sparse_radius * sigma) ** 2  # (B, H, W)
        combined_weights = combined_weights * mask.float()

        # Accumulate weighted saliencies for this batch
        # batch_saliencies: (B, C), combined_weights: (B, H, W)
        # We want: rendered[c, h, w] += sum_b batch_saliencies[b, c] * combined_weights[b, h, w]

        # Reshape for broadcasting: (B, C, 1, 1) * (B, 1, H, W) = (B, C, H, W), then sum over B
        saliencies_expanded = batch_saliencies.unsqueeze(2).unsqueeze(3)  # (B, C, 1, 1)
        weights_expanded = combined_weights.unsqueeze(1)  # (B, 1, H, W)

        weighted_saliencies = saliencies_expanded * weights_expanded  # (B, C, H, W)
        rendered += torch.sum(weighted_saliencies, dim=0)  # Accumulate to (C, H, W)

        # Accumulate total weights
        weights_sum += torch.sum(combined_weights, dim=0)  # Accumulate to (H, W)

    # Normalize by total weight per pixel
    weights_sum = weights_sum.unsqueeze(0)  # (1, H, W) for broadcasting

    # Avoid division by zero
    rendered = rendered / (weights_sum + 1e-8)

    return rendered


def compute_semantic_cost(observed, predicted, method='asymmetric',
                          low_threshold=0.1, temperature=1.0, pixel_coords=None,
                          saliency_2d_full=None, lambda_iou=1.0, confidence=None,
                          magnitude_threshold=0.15, entropy_threshold=0.7, fast_mode=False,
                          voxel_depths=None):
    """
    Compute semantic matching cost between observed and predicted saliency.

    This replaces cosine similarity with an asymmetric cost that:
    - Ignores regions where both are low (no signal)
    - Heavily penalizes high-low mismatches
    - Returns a cost to minimize (not similarity to maximize)

    Args:
        observed: Observed saliency (C, N)
        predicted: Predicted/rendered saliency, same shape as observed
        method: 'asymmetric' (recommended), 'kl_divergence', or 'cosine' (legacy)
        low_threshold: Threshold below which saliency is considered "low"
        temperature: Temperature for softmax in KL divergence
        pixel_coords: Optional (N, 2) array of pixel coordinates for per-pixel aggregation.
                     If provided, groups voxels by pixel and takes min cost per pixel.
                     This handles overlapping voxels from volumetric models correctly.
        saliency_2d_full: Optional (C, H, W) full 2D saliency map for IoU computation.
                         Used with asymmetric and kl_divergence methods. If provided, IoU
                         constraint is added to penalize scale/depth errors.
        lambda_iou: Weight for IoU cost (default 1.0). Used with asymmetric and kl_divergence
                   methods when saliency_2d_full is provided.
        confidence: Optional (C, N) confidence weights for variance-aware aggregation.
                   High confidence (close to 1.0) = reliable feature, fully weighted.
                   Low confidence (close to 0.0) = unreliable feature, downweighted.
                   Only used with asymmetric method. If None, all voxels weighted equally.
        magnitude_threshold: Relative threshold for observable labels in KL divergence (default 0.15)
        entropy_threshold: Concentration threshold for KL divergence (1/ratio), lower = require more concentration (default 0.7 → ratio > 1.43)
        fast_mode: Skip expensive per-pixel aggregation in asymmetric loss (default False). When True, ~300x faster but less accurate.
        voxel_depths: Optional (N,) tensor of voxel depths. For KL divergence, automatically enables memory-efficient
                     batched soft rasterization when combined with pixel_coords and saliency_2d_full

    Returns:
        cost: Scalar cost value (lower is better)
        valid_mask: Boolean mask of valid pixels/points
    """

    if method == 'cosine':
        # Legacy cosine similarity (for comparison)
        return _compute_cosine_similarity(observed, predicted, pixel_coords)

    elif method == 'asymmetric':
        # Recommended: Asymmetric cost function with optional IoU constraint and confidence weighting
        return _compute_asymmetric_cost(observed, predicted, low_threshold, pixel_coords,
                                       saliency_2d_full, lambda_iou, confidence, fast_mode)

    elif method == 'kl_divergence':
        # KL divergence treating saliencies as distributions
        # Automatically uses batched soft rasterization when voxel_depths, pixel_coords, and saliency_2d_full provided
        return _compute_kl_divergence(observed, predicted, temperature, pixel_coords,
                                     saliency_2d_full, lambda_iou,
                                     magnitude_threshold, entropy_threshold, voxel_depths)

    else:
        raise ValueError(f"Unknown method: {method}")


def _compute_cosine_similarity(observed, predicted, pixel_coords=None):
    """
    Legacy cosine similarity (flawed but kept for comparison).
    Returns negative similarity as cost.

    Args:
        observed: (C, N) PyTorch tensor - saliency
        predicted: (C, N) PyTorch tensor - saliency
        pixel_coords: Optional (N, 2) pixel coordinates for per-pixel aggregation
    """
    # Normalize per voxel/pixel
    # Input: (C, N) → compute norm along C dimension
    obs_norm = observed / (torch.norm(observed, dim=0, keepdim=True) + 1e-8)  # (C, N)
    pred_norm = predicted / (torch.norm(predicted, dim=0, keepdim=True) + 1e-8)  # (C, N)

    # Compute similarity per voxel (dot product along channel dimension)
    similarity = torch.sum(obs_norm * pred_norm, dim=0)  # (N,)

    # Valid where either has signal (any channel non-zero)
    valid_mask = (torch.any(observed != 0, dim=0) | torch.any(predicted != 0, dim=0))  # (N,)

    # Aggregate cost
    if not valid_mask.any():
        cost = torch.tensor(0.0, device=observed.device)
    elif pixel_coords is not None:
        # Per-pixel max aggregation for similarity (min for cost)
        # Note: for similarity, we want max (best match), which is min cost
        valid_similarities = similarity[valid_mask]
        valid_pixel_coords = pixel_coords[valid_mask]

        unique_pixels, inverse_indices = torch.unique(
            valid_pixel_coords, dim=0, return_inverse=True
        )

        # Vectorized max aggregation using scatter_reduce
        pixel_similarities = torch.full(
            (len(unique_pixels),),
            -float('inf'),
            device=observed.device,
            dtype=valid_similarities.dtype
        )

        # Scatter and take max at each pixel
        pixel_similarities.scatter_reduce_(
            0,
            inverse_indices,
            valid_similarities,
            reduce='amax',
            include_self=False
        )

        cost = -torch.mean(pixel_similarities)
    else:
        # Fallback: average over all voxels
        cost = -torch.mean(similarity[valid_mask])

    return cost, valid_mask


def _compute_asymmetric_cost(observed, predicted, low_threshold=0.1, pixel_coords=None,
                            saliency_2d_full=None, lambda_iou=1.0, confidence=None,
                            fast_mode=False):
    """
    Asymmetric cost that penalizes mismatches based on their type, with optional IoU constraint
    and variance-aware confidence weighting.

    **NEW (Magnitude-Weighted)**: Instead of binary threshold (exclude low-low), uses continuous
    magnitude weighting to utilize all VLM information while naturally emphasizing strong signals.

    Semantic cost is normalized to [0, 1] range.
    IoU cost (1 - IoU) is also in [0, 1] range, providing scale sensitivity.
    Total cost = semantic_cost + lambda_iou * (1 - IoU)

    Key behaviors:
    - All non-zero signals contribute (no information loss)
    - Strong signals get full weight → high discrimination
    - Weak signals get reduced weight → noise suppression
    - Mismatches preserve heavy penalties (use max magnitude for weighting)
    - Confidence weighting: Downweights costs from unreliable (high-variance) features

    Cost normalization:
    - Perfect match (identical vectors, perfect IoU): 0.0
    - Worst match (opposite vectors, no IoU): 2.0 (with lambda_iou=1.0)

    Args:
        observed: (C, N) PyTorch tensor - saliency sampled at projected voxel locations
        predicted: (C, N) PyTorch tensor - saliency from 3D voxels
        low_threshold: Signal strength threshold for categorization (default 0.1)
                      Lower values are still included but with reduced weight
        pixel_coords: Optional (N, 2) pixel coordinates. If provided, groups voxels
                     by pixel and takes min cost per pixel (handles occlusion).
        saliency_2d_full: Optional (C, H, W) full 2D saliency map. If provided along
                         with pixel_coords, IoU constraint is computed to penalize
                         scale/depth misalignment.
        lambda_iou: Weight for IoU cost term (default 1.0, equal weight with semantic)
        confidence: Optional (C, N) confidence weights. If provided, costs are weighted
                   by confidence (reliable features contribute more). Average across channels.
    """
    # Compute magnitudes per voxel/pixel
    # Input shape: (C, N) where C=channels, N=voxels/pixels
    # We want magnitude of each N-dimensional point → norm along C dimension
    obs_magnitude = torch.norm(observed, dim=0)  # (N,)
    pred_magnitude = torch.norm(predicted, dim=0)  # (N,)

    # NEW: Use very low noise threshold (filter only numerical artifacts)
    # All actual VLM outputs (> 1e-6) will contribute
    NOISE_THRESHOLD = 1e-6

    # Identify regions with meaningful signal (for categorization)
    # Using low_threshold for backward compatibility, but signals below it still contribute
    obs_strong = obs_magnitude > low_threshold
    pred_strong = pred_magnitude > low_threshold

    # Categorize voxels
    both_strong = obs_strong & pred_strong    # Both have strong signal
    is_mismatch = obs_strong ^ pred_strong    # XOR: only one has strong signal
    both_weak = ~obs_strong & ~pred_strong    # Both weak but non-zero

    # Compute directional alignment cost for ALL voxels
    # Normalize each voxel's channel vector to unit length
    obs_norm = observed / (obs_magnitude.unsqueeze(0) + 1e-8)  # (C, N) / (1, N) → (C, N)
    pred_norm = predicted / (pred_magnitude.unsqueeze(0) + 1e-8)  # (C, N)

    # Vector difference (0 when aligned, sqrt(2) when opposite for normalized vectors)
    # Normalize to [0, 1]: divide by 2 (max possible L2 distance between unit vectors)
    vector_diff = torch.norm(obs_norm - pred_norm, dim=0) / 2.0  # (N,)

    # Build raw cost map (before magnitude weighting)
    raw_cost_map = torch.zeros_like(obs_magnitude)

    # Strong alignment: use directional cost
    raw_cost_map[both_strong] = vector_diff[both_strong]

    # Mismatch (false negative/positive): maximum penalty
    raw_cost_map[is_mismatch] = 1.0

    # Weak signals: use directional cost (will be downweighted by magnitude)
    raw_cost_map[both_weak] = vector_diff[both_weak]

    # NEW: Magnitude-based weighting
    # For mismatches: use MAX magnitude (the confident one matters)
    # For alignment: use AVERAGE magnitude (both need to agree)
    avg_magnitude = (obs_magnitude + pred_magnitude) / 2.0
    max_magnitude = torch.max(obs_magnitude, pred_magnitude)

    # Choose weighting magnitude based on case
    weight_magnitude = torch.where(is_mismatch, max_magnitude, avg_magnitude)

    # Apply smooth weighting function: tanh gives smooth [0,1] transition
    # Signals >> low_threshold get weight ≈ 1.0
    # Signals << low_threshold get weight ≈ 0.0
    magnitude_weight = torch.tanh(weight_magnitude / low_threshold)

    # Apply magnitude weighting to raw cost
    cost_map = raw_cost_map * magnitude_weight

    # NEW: Apply variance-based confidence weighting if provided
    # This is MULTIPLICATIVE with magnitude weighting
    # Both low magnitude AND low confidence reduce the cost contribution
    if confidence is not None:
        # Confidence shape: (C, N)
        # Compute mean confidence across channels for each voxel: (N,)
        confidence_weight = torch.mean(confidence, dim=0)  # (N,)

        # Weight the cost_map: low confidence reduces influence on final cost
        cost_map = cost_map * confidence_weight

    # NEW: Valid pixels are those with non-noise signal
    valid_mask = (obs_magnitude > NOISE_THRESHOLD) | (pred_magnitude > NOISE_THRESHOLD)

    # Aggregate cost
    if not valid_mask.any():
        cost = torch.tensor(0.0, device=observed.device)
    elif pixel_coords is not None and not fast_mode:
        # Per-pixel min aggregation: handles overlapping voxels from volumetric models
        # Only consider voxels with semantic signal
        valid_costs = cost_map[valid_mask]  # (N_valid_signal,)
        valid_pixel_coords = pixel_coords[valid_mask]  # (N_valid_signal, 2)

        # Group by unique pixel coordinates
        unique_pixels, inverse_indices = torch.unique(
            valid_pixel_coords, dim=0, return_inverse=True
        )
        # unique_pixels: (N_unique_pixels, 2)
        # inverse_indices: (N_valid_signal,) - maps each voxel to its pixel group

        # For each unique pixel, find minimum cost among all voxels projecting there
        # This effectively implements "soft depth buffering" - the best matching voxel
        # at each pixel is selected automatically by the loss function

        # Vectorized min aggregation using scatter_reduce (much faster than Python loop)
        # Start with inf and scatter-reduce to min
        pixel_costs = torch.full(
            (len(unique_pixels),),
            float('inf'),
            device=observed.device,
            dtype=valid_costs.dtype
        )

        # Scatter valid_costs into pixel_costs, taking min at each pixel
        pixel_costs.scatter_reduce_(
            0,                      # dimension to scatter along
            inverse_indices,        # which pixel each voxel belongs to
            valid_costs,            # costs to scatter
            reduce='amin',          # min reduction
            include_self=False      # don't include initial inf values
        )

        # Average over pixels (not voxels!) - each pixel gets equal weight
        cost = torch.mean(pixel_costs)
    else:
        # Fast mode: average over all voxels (no per-pixel aggregation)
        # This is much faster but weights pixels by voxel density
        cost = torch.mean(cost_map[valid_mask])

    # Add IoU constraint if full 2D saliency map and pixel coords are provided
    if saliency_2d_full is not None and pixel_coords is not None and lambda_iou > 0:
        # Compute observed mask: any pixel with non-zero saliency
        # Since saliency is bbox-cropped with GT mask, non-zero = within GT object region
        obs_magnitude_2d = torch.norm(saliency_2d_full, dim=0)  # (H, W)
        mask_observed = obs_magnitude_2d > 0

        # Compute predicted mask: pixels where voxels project to
        H, W = saliency_2d_full.shape[1:]
        mask_predicted = torch.zeros((H, W), dtype=torch.bool, device=saliency_2d_full.device)

        # Clamp pixel coordinates to image bounds to avoid out-of-bounds after rounding
        # pixel_coords is (N, 2) with (x, y) format
        pixel_x = torch.clamp(pixel_coords[:, 0], 0, W - 1).long()
        pixel_y = torch.clamp(pixel_coords[:, 1], 0, H - 1).long()

        # Mark pixels where voxels project (image indexing is [y, x])
        mask_predicted[pixel_y, pixel_x] = True

        # Compute IoU and convert to cost
        iou = compute_mask_iou(mask_observed, mask_predicted)
        iou_cost = 1.0 - iou

        # Combine semantic and IoU costs
        cost = cost + lambda_iou * iou_cost

    return cost, valid_mask


def _compute_kl_divergence(observed, predicted, temperature=1.0, pixel_coords=None,
                          saliency_2d_full=None, lambda_iou=1.0,
                          magnitude_threshold=0.15, entropy_threshold=0.7, voxel_depths=None):
    """
    KL divergence for pose evaluation with two distinct modes.

    TWO MODES (automatically selected based on parameters):

    **1. RASTERIZATION MODE** (when voxel_depths provided):
       - Rasterizes voxels to full 2D map using soft Gaussian splatting
       - Treats each label as a spatial probability distribution
       - Conceptual model: "Where does label X appear in the image?"
       - Use case: Gradient-based pose optimization (future)
       - Memory: O(batch_size * H * W) with batched processing

    **2. SAMPLING MODE** (default, when voxel_depths=None):
       - Samples observed saliency at voxel projection points
       - Treats each vector as a label probability distribution
       - Conceptual model: "What is the probability of each label at this point?"
       - Use case: Certifier & Estimator (ALIGNED!)
       - Memory: O(N) where N = number of samples
       - ✓ MATCHES estimator's `_compute_kl_divergence_vectorized` logic

    The sampling mode ensures the certifier validates exactly what the estimator optimizes!

    Args:
        observed: (C, N) sampled saliency vectors
        predicted: (C, N) predicted saliency vectors (from voxel model)
        temperature: Temperature for softmax (higher = more uniform)
        pixel_coords: (N, 2) pixel coordinates where voxels project
        saliency_2d_full: (C, H, W) full observed map (for IoU computation)
        lambda_iou: Weight for IoU constraint (geometric alignment)
        magnitude_threshold: Relative threshold for filtering low-magnitude samples (default 0.15)
        entropy_threshold: Unused in sampling mode (kept for API compatibility)
        voxel_depths: (N,) depth values. When provided, enables rasterization mode.
                     When None (default), uses sampling mode.

    Returns:
        cost: Scalar KL divergence cost (lower is better)
        valid_mask: Boolean mask of valid sample points (sampling) or pixels (rasterization)
    """

    if isinstance(observed, torch.Tensor):
        # TWO MODES: Rasterization (when voxel_depths provided) vs Sampling (default, matches estimator)

        if pixel_coords is not None and saliency_2d_full is not None and voxel_depths is not None:
            # Rasterize predicted voxels to full 2D map for direct comparison
            H, W = saliency_2d_full.shape[1:]

            # predicted is (C, N) - transpose to (N, C) for rasterization
            predicted_rasterized = soft_rasterize_voxels(
                pixel_coords,  # (N, 2)
                predicted.t(),  # (N, C)
                voxel_depths,   # (N,)
                H, W
            )  # Returns (C, H, W)

            # Use full observed map
            obs_flat = saliency_2d_full.reshape(saliency_2d_full.shape[0], -1)  # (C, H*W)
            pred_flat = predicted_rasterized.reshape(predicted_rasterized.shape[0], -1)  # (C, H*W)

            # RASTERIZATION MODE: Per-label spatial distribution
            # For each label, treat its spatial distribution as a probability map
            obs_probs = F.softmax(obs_flat / temperature, dim=-1)  # (C, H*W) - normalize across space
            pred_probs = F.softmax(pred_flat / temperature, dim=-1)  # (C, H*W)

            # Add epsilon
            pred_probs = pred_probs + 1e-10

            # Filter significant labels (observable + concentrated)
            obs_magnitude = torch.norm(obs_flat, dim=1)  # (C,)
            obs_max = torch.max(torch.abs(obs_flat), dim=1)[0]  # (C,)
            obs_mean = torch.mean(torch.abs(obs_flat), dim=1)  # (C,)
            concentration_ratio = obs_max / (obs_mean + 1e-8)
            is_concentrated = concentration_ratio > (1.0 / entropy_threshold)

            adaptive_threshold = magnitude_threshold * torch.max(obs_magnitude)
            significant_mask = (obs_magnitude > adaptive_threshold) & is_concentrated

            # KL divergence per label
            kl_div = torch.sum(obs_probs * torch.log(obs_probs / pred_probs), dim=-1)  # (C,)

            # Average over significant labels
            if torch.any(significant_mask):
                cost = torch.mean(kl_div[significant_mask])
            else:
                cost = torch.tensor(1.0, device=observed.device)

            valid_mask = torch.ones(obs_flat.shape[-1], dtype=torch.bool, device=observed.device)

        else:
            # SAMPLING MODE: Align with estimator's correspondence matching logic
            # Uses per-vector label distribution (NOT per-label spatial distribution)
            # This ensures certifier validates what the estimator optimizes!

            # observed: (C, N), predicted: (C, N) where N = number of sample points
            # Transpose to (N, C) for per-vector processing
            obs_vectors = observed.t()  # (N, C)
            pred_vectors = predicted.t()  # (N, C)

            # For each of N sample points, treat C-dim vector as distribution over labels
            # This matches _compute_kl_divergence_vectorized logic!
            obs_probs = F.softmax(obs_vectors / temperature, dim=1)  # (N, C) - each row is label dist
            pred_probs = F.softmax(pred_vectors / temperature, dim=1)  # (N, C)

            # Add epsilon to avoid log(0)
            pred_probs = pred_probs + 1e-10

            # KL divergence per sample point: KL(obs_i || pred_i) for each i in [0, N)
            kl_per_sample = torch.sum(obs_probs * torch.log(obs_probs / pred_probs), dim=1)  # (N,)

            # Filter out low-magnitude samples (likely noise or background)
            obs_magnitude = torch.norm(obs_vectors, dim=1)  # (N,) - magnitude per sample
            adaptive_threshold = magnitude_threshold * torch.max(obs_magnitude)
            valid_mask = obs_magnitude > adaptive_threshold

            # Average KL over valid samples
            if torch.any(valid_mask):
                cost = torch.mean(kl_per_sample[valid_mask])
            else:
                cost = torch.tensor(1.0, device=observed.device)

        # Common: Add IoU constraint if full 2D saliency map and pixel coords provided
        # IoU penalizes geometric misalignment (spatial footprint mismatch)
        if saliency_2d_full is not None and pixel_coords is not None and lambda_iou > 0:
            # Compute observed mask: any pixel with non-zero saliency
            obs_magnitude_2d = torch.norm(saliency_2d_full, dim=0)  # (H, W)
            mask_observed = obs_magnitude_2d > 0

            # Compute predicted mask: pixels where voxels project to
            H, W = saliency_2d_full.shape[1:]
            mask_predicted = torch.zeros((H, W), dtype=torch.bool, device=saliency_2d_full.device)

            # Clamp pixel coordinates to image bounds
            pixel_x = torch.clamp(pixel_coords[:, 0], 0, W - 1).long()
            pixel_y = torch.clamp(pixel_coords[:, 1], 0, H - 1).long()

            # Mark pixels where voxels project (image indexing is [y, x])
            mask_predicted[pixel_y, pixel_x] = True

            # Compute IoU and convert to cost
            iou = compute_mask_iou(mask_observed, mask_predicted)
            iou_cost = 1.0 - iou

            # Combine semantic and IoU costs
            cost = cost + lambda_iou * iou_cost

    return cost, valid_mask


def compute_correspondence_scores(query_vectors, reference_vectors, method='asymmetric',
                                  low_threshold=0.1, return_costs=False, temperature=1.0,
                                  lambda_reverse=0.5):
    """
    Compute matching scores between query and reference saliency vectors.
    Used in correspondence matching for PnP.

    Args:
        query_vectors: (N, C) query saliency vectors
        reference_vectors: (M, C) reference saliency vectors
        method: Loss method to use ('kl_divergence', 'reverse_kl', 'bidirectional_kl',
                'jensen_shannon', 'weighted_cosine', 'asymmetric', 'cosine')
        low_threshold: Threshold for low saliency (for asymmetric method)
        return_costs: If True, return costs; if False, return similarities
        temperature: Temperature for KL divergence softmax
        lambda_reverse: Weight for reverse KL in bidirectional KL (default 0.5)

    Returns:
        scores: (N, M) matrix of scores (similarities or costs)
    """

    N = len(query_vectors)
    M = len(reference_vectors)

    # Use vectorized versions for all methods (much faster!)
    if method == 'kl_divergence':
        return _compute_kl_divergence_vectorized(query_vectors, reference_vectors,
                                                 temperature, return_costs)

    elif method == 'reverse_kl':
        return _compute_reverse_kl_divergence_vectorized(query_vectors, reference_vectors,
                                                         temperature, return_costs)

    elif method == 'bidirectional_kl':
        return _compute_bidirectional_kl_divergence_vectorized(query_vectors, reference_vectors,
                                                               temperature, lambda_reverse, return_costs)

    elif method == 'jensen_shannon':
        return _compute_jensen_shannon_divergence_vectorized(query_vectors, reference_vectors,
                                                             temperature, return_costs)

    elif method == 'weighted_cosine':
        return _compute_weighted_cosine_similarity_vectorized(query_vectors, reference_vectors,
                                                              return_costs)

    elif method == 'asymmetric':
        return _compute_asymmetric_cost_vectorized(query_vectors, reference_vectors,
                                                   low_threshold, return_costs)

    elif method == 'cosine':
        return _compute_cosine_similarity_vectorized(query_vectors, reference_vectors,
                                                     return_costs)

    # Fall back to loop-based computation for unknown methods
    else:
        if isinstance(query_vectors, torch.Tensor):
            device = query_vectors.device
            scores = torch.zeros(N, M, device=device)

            # Compute pairwise scores
            for i in range(N):
                for j in range(M):
                    cost, _ = compute_semantic_cost(
                        query_vectors[i], reference_vectors[j],
                        method=method, low_threshold=low_threshold
                    )
                    scores[i, j] = cost if return_costs else -cost

        return scores


def _compute_kl_divergence_vectorized(query_vectors, reference_vectors, temperature=1.0, return_costs=False):
    """
    Fast vectorized KL divergence computation for correspondence matching.

    Computes KL divergence between all pairs of query and reference vectors.

    Args:
        query_vectors: (N, C) query saliency vectors
        reference_vectors: (M, C) reference saliency vectors
        temperature: Temperature for softmax
        return_costs: If True, return costs; if False, return similarities (negative costs)

    Returns:
        scores: (N, M) matrix of KL divergences
    """

    if isinstance(query_vectors, torch.Tensor):
        # Reshape for broadcasting: query (N, 1, C), reference (1, M, C)
        query = query_vectors.unsqueeze(1)  # (N, 1, C)
        ref = reference_vectors.unsqueeze(0)  # (1, M, C)

        # Apply temperature and softmax along channel dimension
        # Output: (N, M, C) probability distributions
        query_probs = F.softmax(query / temperature, dim=-1)  # (N, 1, C) -> (N, 1, C)
        ref_probs = F.softmax(ref / temperature, dim=-1)  # (1, M, C) -> (1, M, C)

        # Add epsilon to avoid log(0)
        ref_probs = ref_probs + 1e-10

        # KL divergence: sum over channels
        # KL(P||Q) = sum_c P(c) * log(P(c) / Q(c))
        kl_div = torch.sum(query_probs * torch.log(query_probs / ref_probs), dim=-1)  # (N, M)

        # Return costs or similarities
        scores = kl_div if return_costs else -kl_div

    return scores


def _compute_asymmetric_cost_vectorized(query_vectors, reference_vectors, low_threshold=0.1, return_costs=False):
    """
    Fast vectorized asymmetric cost computation for correspondence matching.

    Computes asymmetric cost between all pairs of query and reference vectors.
    This is ~100-1000× faster than the loop-based version for large sets.

    Args:
        query_vectors: (N, C) query saliency vectors
        reference_vectors: (M, C) reference saliency vectors
        low_threshold: Threshold for low saliency (default 0.1)
        return_costs: If True, return costs; if False, return similarities (negative costs)

    Returns:
        scores: (N, M) matrix of asymmetric costs or similarities
    """

    if isinstance(query_vectors, torch.Tensor):
        # Reshape for broadcasting: query (N, 1, C), reference (1, M, C)
        query = query_vectors.unsqueeze(1)  # (N, 1, C)
        ref = reference_vectors.unsqueeze(0)  # (1, M, C)

        # Compute magnitudes: (N, M)
        obs_magnitude = torch.norm(query, dim=-1)  # (N, 1)
        pred_magnitude = torch.norm(ref, dim=-1)  # (1, M)

        # Identify signal strength categories
        obs_strong = obs_magnitude > low_threshold  # (N, 1)
        pred_strong = pred_magnitude > low_threshold  # (1, M)

        both_strong = obs_strong & pred_strong  # (N, M)
        is_mismatch = obs_strong ^ pred_strong  # (N, M)

        # Normalize vectors for directional comparison
        obs_norm = query / (obs_magnitude.unsqueeze(-1) + 1e-8)  # (N, 1, C)
        pred_norm = ref / (pred_magnitude.unsqueeze(-1) + 1e-8)  # (1, M, C)

        # Vector difference (0 when aligned, sqrt(2) when opposite)
        # Normalize to [0, 1] by dividing by 2
        vector_diff = torch.norm(obs_norm - pred_norm, dim=-1) / 2.0  # (N, M)

        # Build cost map
        cost_map = torch.where(is_mismatch,
                               torch.ones_like(vector_diff),  # Maximum penalty for mismatch
                               vector_diff)  # Directional cost otherwise

        # Magnitude weighting
        avg_magnitude = (obs_magnitude + pred_magnitude) / 2.0  # (N, M)
        max_magnitude = torch.max(obs_magnitude, pred_magnitude)  # (N, M)

        # Use max magnitude for mismatches, avg for alignment
        weight_magnitude = torch.where(is_mismatch, max_magnitude, avg_magnitude)
        magnitude_weight = torch.tanh(weight_magnitude / low_threshold)

        # Apply weighting
        cost_map = cost_map * magnitude_weight  # (N, M)

        # Return costs or similarities
        scores = cost_map if return_costs else -cost_map

    return scores


def _compute_cosine_similarity_vectorized(query_vectors, reference_vectors, return_costs=False):
    """
    Fast vectorized cosine similarity computation for correspondence matching.

    Computes cosine similarity between all pairs of query and reference vectors.

    Args:
        query_vectors: (N, C) query saliency vectors
        reference_vectors: (M, C) reference saliency vectors
        return_costs: If True, return costs (negative similarity); if False, return similarities

    Returns:
        scores: (N, M) matrix of cosine similarities or costs
    """

    if isinstance(query_vectors, torch.Tensor):
        # Normalize all vectors
        query_norm = query_vectors / (torch.norm(query_vectors, dim=-1, keepdim=True) + 1e-8)  # (N, C)
        ref_norm = reference_vectors / (torch.norm(reference_vectors, dim=-1, keepdim=True) + 1e-8)  # (M, C)

        # Cosine similarity via matrix multiplication: (N, C) @ (C, M) = (N, M)
        similarity = torch.matmul(query_norm, ref_norm.t())  # (N, M)

        # Return costs or similarities
        scores = -similarity if return_costs else similarity

    return scores


def _compute_reverse_kl_divergence_vectorized(query_vectors, reference_vectors, temperature=1.0, return_costs=False):
    """
    Fast vectorized reverse KL divergence computation for correspondence matching.

    Computes KL(reference || query) instead of KL(query || reference).
    This penalizes when the model has high probability but observation has low.

    Args:
        query_vectors: (N, C) query saliency vectors
        reference_vectors: (M, C) reference saliency vectors
        temperature: Temperature for softmax
        return_costs: If True, return costs; if False, return similarities

    Returns:
        scores: (N, M) matrix of reverse KL divergences
    """

    if isinstance(query_vectors, torch.Tensor):
        # Reshape for broadcasting
        query = query_vectors.unsqueeze(1)  # (N, 1, C)
        ref = reference_vectors.unsqueeze(0)  # (1, M, C)

        # Apply temperature and softmax
        query_probs = F.softmax(query / temperature, dim=-1)  # (N, 1, C)
        ref_probs = F.softmax(ref / temperature, dim=-1)  # (1, M, C)

        # Add epsilon to avoid log(0)
        query_probs = query_probs + 1e-10

        # Reverse KL: KL(Q||P) = sum_c Q(c) * log(Q(c) / P(c))
        kl_div = torch.sum(ref_probs * torch.log(ref_probs / query_probs), dim=-1)  # (N, M)

        # Return costs or similarities
        scores = kl_div if return_costs else -kl_div

    return scores


def _compute_bidirectional_kl_divergence_vectorized(query_vectors, reference_vectors, temperature=1.0,
                                                    lambda_reverse=0.5, return_costs=False):
    """
    Fast vectorized bidirectional KL divergence computation.

    Computes: KL(query || reference) + lambda * KL(reference || query)
    Symmetric penalty for distribution mismatch.

    Args:
        query_vectors: (N, C) query saliency vectors
        reference_vectors: (M, C) reference saliency vectors
        temperature: Temperature for softmax
        lambda_reverse: Weight for reverse KL term (default 0.5)
        return_costs: If True, return costs; if False, return similarities

    Returns:
        scores: (N, M) matrix of bidirectional KL divergences
    """

    if isinstance(query_vectors, torch.Tensor):
        # Reshape for broadcasting
        query = query_vectors.unsqueeze(1)  # (N, 1, C)
        ref = reference_vectors.unsqueeze(0)  # (1, M, C)

        # Apply temperature and softmax
        query_probs = F.softmax(query / temperature, dim=-1)  # (N, 1, C)
        ref_probs = F.softmax(ref / temperature, dim=-1)  # (1, M, C)

        # Add epsilon to avoid log(0)
        query_probs_safe = query_probs + 1e-10
        ref_probs_safe = ref_probs + 1e-10

        # Forward KL: KL(P||Q)
        forward_kl = torch.sum(query_probs * torch.log(query_probs_safe / ref_probs_safe), dim=-1)

        # Reverse KL: KL(Q||P)
        reverse_kl = torch.sum(ref_probs * torch.log(ref_probs_safe / query_probs_safe), dim=-1)

        # Combine
        bidirectional_kl = forward_kl + lambda_reverse * reverse_kl  # (N, M)

        # Return costs or similarities
        scores = bidirectional_kl if return_costs else -bidirectional_kl

    return scores


def _compute_jensen_shannon_divergence_vectorized(query_vectors, reference_vectors, temperature=1.0, return_costs=False):
    """
    Fast vectorized Jensen-Shannon divergence computation.

    JS divergence is a symmetric, bounded [0, log(2)] version of KL divergence:
    JS(P,Q) = 0.5 * KL(P||M) + 0.5 * KL(Q||M) where M = (P+Q)/2

    Args:
        query_vectors: (N, C) query saliency vectors
        reference_vectors: (M, C) reference saliency vectors
        temperature: Temperature for softmax
        return_costs: If True, return costs; if False, return similarities

    Returns:
        scores: (N, M) matrix of JS divergences
    """

    if isinstance(query_vectors, torch.Tensor):
        # Reshape for broadcasting
        query = query_vectors.unsqueeze(1)  # (N, 1, C)
        ref = reference_vectors.unsqueeze(0)  # (1, M, C)

        # Apply temperature and softmax
        query_probs = F.softmax(query / temperature, dim=-1)  # (N, 1, C)
        ref_probs = F.softmax(ref / temperature, dim=-1)  # (1, M, C)

        # Compute mixture distribution M = (P + Q) / 2
        mixture_probs = (query_probs + ref_probs) / 2.0  # (N, M, C)

        # Add epsilon to avoid log(0)
        query_probs_safe = query_probs + 1e-10
        ref_probs_safe = ref_probs + 1e-10
        mixture_probs_safe = mixture_probs + 1e-10

        # KL(P||M)
        kl_p_m = torch.sum(query_probs * torch.log(query_probs_safe / mixture_probs_safe), dim=-1)

        # KL(Q||M)
        kl_q_m = torch.sum(ref_probs * torch.log(ref_probs_safe / mixture_probs_safe), dim=-1)

        # JS divergence
        js_div = 0.5 * (kl_p_m + kl_q_m)  # (N, M)

        # Return costs or similarities (JS is naturally bounded [0, log(2)])
        scores = js_div if return_costs else -js_div

    return scores


def _compute_weighted_cosine_similarity_vectorized(query_vectors, reference_vectors, return_costs=False):
    """
    Fast vectorized magnitude-weighted cosine similarity.

    Addresses cosine similarity's magnitude blindness by weighting with minimum magnitude.
    Score = cos_sim(P, Q) * min(||P||, ||Q||)

    Args:
        query_vectors: (N, C) query saliency vectors
        reference_vectors: (M, C) reference saliency vectors
        return_costs: If True, return costs; if False, return similarities

    Returns:
        scores: (N, M) matrix of weighted cosine similarities
    """

    if isinstance(query_vectors, torch.Tensor):
        # Compute magnitudes
        query_magnitude = torch.norm(query_vectors, dim=-1, keepdim=True)  # (N, 1)
        ref_magnitude = torch.norm(reference_vectors, dim=-1, keepdim=True)  # (M, 1)

        # Normalize vectors for cosine similarity
        query_norm = query_vectors / (query_magnitude + 1e-8)  # (N, C)
        ref_norm = reference_vectors / (ref_magnitude + 1e-8)  # (M, C)

        # Cosine similarity via matrix multiplication
        cosine_sim = torch.matmul(query_norm, ref_norm.t())  # (N, M)

        # Compute minimum magnitude for each pair
        query_mag_expanded = query_magnitude  # (N, 1)
        ref_mag_expanded = ref_magnitude.t()  # (1, M)
        min_magnitude = torch.minimum(query_mag_expanded, ref_mag_expanded)  # (N, M)

        # Weight cosine similarity by minimum magnitude
        # Normalize by dividing by max possible magnitude to keep in [-1, 1] range
        max_mag = torch.maximum(query_mag_expanded.max(), ref_mag_expanded.max())
        weighted_sim = cosine_sim * (min_magnitude / (max_mag + 1e-8))  # (N, M)

        # Return costs or similarities
        scores = -weighted_sim if return_costs else weighted_sim

    return scores
