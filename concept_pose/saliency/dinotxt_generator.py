"""
DinoTxt-based Semantic Saliency Generator
==========================================

This module provides DinoTxt-based saliency generation using GradCAM and direct
patch similarity methods. DinoTxt is a vision-language model built on DINOv3
with text encoding capabilities.

Key features:
- Text-image similarity at patch level (14x14 native resolution)
- Dual embedding space: 2048-dim split into class (1024) and patch (1024) features
- Direct patch similarity for clean, fast saliency maps
- Bicubic interpolation for smooth high-resolution output
- API-compatible with SigLIP2SaliencyGenerator and CLIPGradCAMGenerator

Implementation follows the approach from dinotxt_gradcam_pytorch.py in the
DinoV3 research codebase.
"""

import sys
import os
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms

# Lazy imports for DinoV3 (will be loaded on first instantiation)
dinov3_module = None
dinotxt_hub = None

DINOV3_PATH = "/media/liming/Data/6DPose/dinov3"
WEIGHTS_DIR = Path(DINOV3_PATH) / "weights"

class DinoTxtGradCAMWrapper(torch.nn.Module):
    """
    Wrapper to make DinoTxt vision model compatible with pytorch_grad_cam.

    DinoTxt produces 2048-dim embeddings split into:
    - First 1024-dim: aligned with class token (global features)
    - Second 1024-dim: aligned with patch tokens (spatial features)

    This wrapper computes spatial similarity using the patch-aligned portion.
    """

    def __init__(self, model, text_patch_features: torch.Tensor, device: str = "cuda"):
        """
        Args:
            model: DinoTxt model
            text_patch_features: Pre-encoded text features (patch-aligned portion only)
                                Shape: [num_labels, 1024]
            device: Device to run on
        """
        super().__init__()
        self.model = model
        self.text_patch_features = text_patch_features  # [N, 1024]
        self.device = device

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass that returns patch-text similarity maps.

        Args:
            x: Input image tensor [B, 3, H, W]

        Returns:
            Spatial similarity maps [B, num_labels, patch_h, patch_w]
        """
        # Get patch tokens from image
        _, image_patch_tokens, _ = self.model.encode_image_with_patch_tokens(x)
        # image_patch_tokens: [B, num_patches, 1024]

        # Normalize
        image_patch_tokens_norm = F.normalize(image_patch_tokens, p=2, dim=-1)
        text_patch_features_norm = F.normalize(self.text_patch_features, p=2, dim=-1)

        # Compute spatial similarity [B, P, N]
        spatial_sim = torch.einsum('bpd,nd->bpn',
                                   image_patch_tokens_norm,
                                   text_patch_features_norm)

        # Reshape to [B, N, H, W] for GradCAM
        B, P, N = spatial_sim.shape
        H = W = int(P ** 0.5)  # sqrt(196) = 14 for 224x224 input
        spatial_sim = spatial_sim.permute(0, 2, 1).view(B, N, H, W)

        return spatial_sim


def reshape_transform_dinotxt(tensor: torch.Tensor, patch_grid_h: int, patch_grid_w: int) -> torch.Tensor:
    """
    Reshape transformer output for GradCAM visualization.

    DinoTxt vision transformer outputs include:
    - 1 CLS token
    - 4 storage/register tokens
    - 196 patch tokens (for 224x224 input with patch_size=16)

    This function removes non-patch tokens and reshapes to spatial grid.

    Args:
        tensor: Transformer layer output [B, num_tokens, D]
        patch_grid_h: Expected patch grid height
        patch_grid_w: Expected patch grid width

    Returns:
        Reshaped tensor [B, D, patch_grid_h, patch_grid_w]
    """
    expected_num_patches = patch_grid_h * patch_grid_w

    # Handle potential transpose (some layers output [N, B, D] instead of [B, N, D])
    if tensor.ndim == 3:
        if tensor.shape[0] == 1 and tensor.shape[1] > 100:
            pass  # (1, N, D) - normal case
        elif tensor.shape[1] == 1 and tensor.shape[0] > 100:
            tensor = tensor.transpose(0, 1)  # Fix (N, 1, D) -> (1, N, D)
        elif tensor.shape[0] > 100 and tensor.shape[1] > 100:
            # Ambiguous, assume (B, N, D)
            pass

    # Remove CLS token (position 0) and storage tokens (positions 1-4)
    # Patch tokens start at position 5
    if tensor.shape[1] == expected_num_patches + 5:  # CLS + 4 storage + patches
        tensor = tensor[:, 5:, :]  # Remove first 5 tokens
    elif tensor.shape[1] == expected_num_patches + 1:  # Only CLS
        tensor = tensor[:, 1:, :]  # Remove CLS
    elif tensor.shape[1] == expected_num_patches:
        pass  # Already correct
    else:
        print(f"Warning: Unexpected number of tokens. Expected {expected_num_patches} patches, "
              f"but got {tensor.shape[1]} total tokens. Attempting to extract patches...")
        # Try to extract the last expected_num_patches tokens
        tensor = tensor[:, -expected_num_patches:, :]

    # Reshape to spatial grid: [B, H, W, D] then permute to [B, D, H, W]
    result = tensor.reshape(tensor.size(0), patch_grid_h, patch_grid_w, tensor.size(2))
    result = result.permute(0, 3, 1, 2)

    return result


class DinoTxtGradCAMGenerator:
    """
    DinoTxt-based semantic saliency generator with GradCAM support.

    API-compatible with SigLIP2SaliencyGenerator and CLIPGradCAMGenerator.
    Uses DinoTxt's vision-language model to generate semantic saliency maps
    for specified text labels.

    Features:
    - Native 14x14 patch resolution (for 224x224 input)
    - Direct patch similarity (primary method, fast and clean)
    - Optional GradCAM support
    - Bicubic upsampling for smooth high-resolution output
    - Dual text embedding (class + patch features)

    Example:
        >>> generator = DinoTxtGradCAMGenerator(['handle', 'spout', 'body'])
        >>> saliency_maps, patch_features = generator.process_frame(image)
    """

    def __init__(self,
                 candidate_labels: List[str],
                 model_name: str = "vitl16",
                 custom_device: Optional[str] = None):
        """
        Initialize DinoTxt saliency generator.

        Args:
            candidate_labels: List of semantic part labels
            model_name: Model variant (currently only 'vitl16' supported)
            custom_device: Device to use (default: auto-detect GPU)
        """
        # Lazy import DinoV3 modules
        self._lazy_load_dinov3()

        # Device setup
        if custom_device:
            self.device = custom_device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"Initializing DinoTxt saliency generator on {self.device}")

        # Model configuration
        self.model_name = model_name
        if model_name != "vitl16":
            raise ValueError(f"Only 'vitl16' model supported, got {model_name}")

        # Load model and tokenizer
        self._load_model()

        # Get model configuration
        self.patch_size = 16  # DinoV3 ViT-L/16
        self.input_resolution = 224  # Standard input size
        self.patch_grid_size = self.input_resolution // self.patch_size  # 14

        # Initialize text embeddings
        self.candidate_labels = candidate_labels
        self.text_features = None
        self.text_class_features = None  # First 1024-dim (class-aligned)
        self.text_patch_features = None  # Second 1024-dim (patch-aligned)

        self._encode_text_labels(candidate_labels)

        # Initialize GradCAM
        self._initialize_gradcam()

    def _lazy_load_dinov3(self):
        """Lazy load DinoV3 modules to avoid import errors if not installed."""
        global dinov3_module, dinotxt_hub

        if dinov3_module is None:
            try:
                # Add DinoV3 path to sys.path
                dinov3_path = DINOV3_PATH
                if str(dinov3_path) not in sys.path:
                    sys.path.insert(0, str(dinov3_path))

                # Import modules
                from dinov3.hub import dinotxt as dinotxt_hub_module
                from dinov3.data import transforms as dinov3_transforms_module

                dinotxt_hub = dinotxt_hub_module
                dinov3_module = dinov3_transforms_module

            except ImportError as e:
                raise ImportError(
                    f"DinoV3 library not found at {dinov3_path}. "
                    f"Error: {e}"
                )

    def _load_model(self):
        """Load DinoTxt model with pretrained weights."""
        # Weights paths 
        weights_dir = WEIGHTS_DIR

        # Find weight files
        dinotxt_weights = list(weights_dir.glob("dinov3_vitl16_dinotxt_vision_head_and_text_encoder-*.pth"))
        backbone_weights = list(weights_dir.glob("dinov3_vitl16_pretrain_lvd1689m-*.pth"))

        if not dinotxt_weights:
            raise FileNotFoundError(f"DinoTxt weights not found in {weights_dir}")
        if not backbone_weights:
            raise FileNotFoundError(f"Backbone weights not found in {weights_dir}")

        dinotxt_weights_path = str(dinotxt_weights[0])
        backbone_weights_path = str(backbone_weights[0])

        print(f"Loading DinoTxt model from:")
        print(f"  DinoTxt weights: {dinotxt_weights_path}")
        print(f"  Backbone weights: {backbone_weights_path}")

        # Load model using hub function
        self.model, self.tokenizer = dinotxt_hub.dinov3_vitl16_dinotxt_tet1280d20h24l(
            pretrained=True,
            dinotxt_weights=dinotxt_weights_path,
            backbone_weights=backbone_weights_path
        )

        self.model = self.model.to(self.device)
        self.model.eval()

        print("DinoTxt model loaded successfully")

    def _encode_text_labels(self, labels: List[str]):
        """
        Encode text labels and cache embeddings.

        DinoTxt produces 2048-dim embeddings:
        - First 1024-dim: aligned with class token (global)
        - Second 1024-dim: aligned with patch tokens (spatial)
        """
        # Format labels (add period for better tokenization)
        formatted_labels = [f"{label.lower()}." for label in labels]

        # Tokenize
        tokens = self.tokenizer.tokenize(formatted_labels).to(self.device)

        # Encode text
        with torch.no_grad():
            text_features = self.model.encode_text(tokens)  # [N, 2048]
            text_features_normalized = F.normalize(text_features, p=2, dim=-1)

            # Split into class and patch portions
            self.text_class_features = text_features_normalized[:, :1024]  # First half
            self.text_patch_features = text_features_normalized[:, 1024:]  # Second half (for saliency!)

            # Store full features as well
            self.text_features = text_features_normalized

        print(f"Encoded {len(labels)} text labels:")
        for label in labels:
            print(f"  - {label}")

    def _initialize_gradcam(self):
        """Initialize GradCAM wrapper (optional, direct patch similarity is primary)."""
        try:
            from pytorch_grad_cam import GradCAM
            from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
            from pytorch_grad_cam.utils.image import show_cam_on_image

            # Create wrapper
            self.gradcam_wrapper = DinoTxtGradCAMWrapper(
                self.model,
                self.text_patch_features,
                self.device
            )

            # Target layer: last transformer block in vision backbone
            target_layers = [self.model.visual_model.backbone.blocks[-1]]

            # Initialize GradCAM
            self.cam = GradCAM(
                model=self.gradcam_wrapper,
                target_layers=target_layers,
                reshape_transform=lambda t: reshape_transform_dinotxt(
                    t, self.patch_grid_size, self.patch_grid_size
                )
            )

            self.gradcam_available = True
            print("GradCAM initialized successfully")

        except ImportError:
            print("Warning: pytorch_grad_cam not available. Using direct patch similarity only.")
            self.gradcam_available = False
            self.cam = None

    def set_labels(self, new_labels: List[str]):
        """
        Update semantic labels and re-encode text embeddings.

        Args:
            new_labels: New list of semantic labels
        """
        self.candidate_labels = new_labels
        self._encode_text_labels(new_labels)

        # Update GradCAM wrapper if available
        if self.gradcam_available:
            self.gradcam_wrapper.text_patch_features = self.text_patch_features

    def _preprocess_image(self, pil_image: Image.Image) -> torch.Tensor:
        """
        Preprocess PIL image for DinoTxt model.

        Uses ImageNet normalization (standard for DinoV3/DinoTxt):
        - Mean: [0.485, 0.456, 0.406]
        - Std: [0.229, 0.224, 0.225]

        Args:
            pil_image: Input PIL image

        Returns:
            Preprocessed tensor [1, 3, 224, 224]
        """
        # Resize to 224x224 with bicubic interpolation
        resized = pil_image.resize(
            (self.input_resolution, self.input_resolution),
            Image.Resampling.BICUBIC
        )

        # ImageNet normalization
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet mean
                std=[0.229, 0.224, 0.225]    # ImageNet std
            )
        ])

        pixel_values = transform(resized).unsqueeze(0)
        return pixel_values

    def _compute_direct_saliency(self,
                                 image_tensor: torch.Tensor,
                                 target_size: Optional[Tuple[int, int]] = None) -> torch.Tensor:
        """
        Compute saliency using direct patch-text similarity (primary method).

        This follows the exact approach from dinotxt_gradcam_pytorch.py.
        Faster and cleaner than GradCAM.

        Args:
            image_tensor: Preprocessed image [1, 3, 224, 224]
            target_size: Optional target size for upsampling (H, W)

        Returns:
            Saliency maps [num_labels, H, W] normalized to [0, 1] as torch.Tensor
        """
        with torch.no_grad():
            # Extract patch tokens
            _, image_patch_tokens, _ = self.model.encode_image_with_patch_tokens(image_tensor)
            # image_patch_tokens: [1, 196, 1024] for 224x224 input

            # Normalize
            image_patch_tokens_norm = F.normalize(image_patch_tokens, p=2, dim=-1)
            text_patch_features_norm = F.normalize(self.text_patch_features, p=2, dim=-1)

            # Compute per-patch similarity for each label
            # [B, P, D] @ [N, D].T -> [B, P, N]
            similarity_maps = torch.einsum('bpd,nd->bpn',
                                          image_patch_tokens_norm,
                                          text_patch_features_norm)

            # Reshape to spatial grid: [B, N, H, W]
            B, P, N = similarity_maps.shape
            H = W = int(P ** 0.5)  # sqrt(196) = 14
            saliency_maps = similarity_maps.permute(0, 2, 1).view(B, N, H, W)

            # Upsample to target size if specified
            if target_size is not None:
                saliency_maps = F.interpolate(
                    saliency_maps,
                    size=target_size,
                    mode='bicubic',
                    align_corners=False
                )

            # Remove batch dimension
            saliency_maps = saliency_maps.squeeze(0)  # [N, H, W]

            # Normalize each saliency map to [0, 1]
            normalized_maps = torch.zeros_like(saliency_maps)
            for i in range(len(self.candidate_labels)):
                smap = saliency_maps[i]
                min_val, max_val = smap.min(), smap.max()
                if max_val > min_val:
                    normalized_maps[i] = (smap - min_val) / (max_val - min_val)
                else:
                    normalized_maps[i] = torch.zeros_like(smap)

            return normalized_maps

    def process_frame(self,
                     raw_image: Image.Image,
                     visualize: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process a single frame to generate semantic saliency maps.

        API-compatible with SigLIP2SaliencyGenerator.

        Args:
            raw_image: Input PIL image
            visualize: Whether to visualize (currently unused, for compatibility)

        Returns:
            Tuple of:
            - saliency_maps: Saliency maps [num_labels, H, W] normalized to [0, 1] as torch.Tensor
            - patch_features: Patch-level features [num_patches, feature_dim] as torch.Tensor
        """
        # Preprocess image
        image_tensor = self._preprocess_image(raw_image).to(self.device)

        # Get target size from original image
        target_size = (raw_image.height, raw_image.width)

        # Compute saliency using direct patch similarity
        saliency_maps = self._compute_direct_saliency(image_tensor, target_size)

        # Extract patch features for compatibility
        with torch.no_grad():
            _, image_patch_tokens, _ = self.model.encode_image_with_patch_tokens(image_tensor)
            patch_features = image_patch_tokens.squeeze(0)  # [196, 1024]

        return saliency_maps, patch_features

    def release_gpu_memory(self):
        """Release GPU memory by moving model to CPU."""
        if hasattr(self, 'model'):
            self.model = self.model.cpu()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("DinoTxt model moved to CPU and GPU memory released")

    def cleanup(self):
        """Cleanup method for compatibility with OneShotPoseEstimator."""
        self.release_gpu_memory()
