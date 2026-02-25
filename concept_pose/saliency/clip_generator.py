"""
CLIP GradCAM Saliency Generator
================================

Drop-in replacement for SigLIP2SaliencyGenerator using CLIP models.
Maintains exact same API and preprocessing consistency for ablation studies.

Requirements:
    pip install git+https://github.com/openai/CLIP.git
"""

import torch
import numpy as np
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import matplotlib.pyplot as plt
import functools
import torchvision.transforms as transforms
from typing import List, Tuple, Optional
import gc

# Lazy import of CLIP (only when CLIPGradCAMGenerator is instantiated)
clip = None


# --- 1. CLIP Model Wrapper for Grad-CAM ---
class CLIPGradCAMWrapper(torch.nn.Module):
    """
    Wrapper around CLIP model for GradCAM compatibility.
    Computes image-text similarity scores that GradCAM can differentiate through.
    """
    def __init__(self, clip_model, text_features, device="cuda"):
        super().__init__()
        self.model = clip_model
        self.text_features = text_features  # Pre-encoded and normalized text features (num_labels, D)
        self.device = device
        self.dtype = clip_model.dtype
        self.patch_features_from_vision_output = None

    def forward(self, image_pixel_values_4d):
        """
        Forward pass for GradCAM.

        Args:
            image_pixel_values_4d: (B, 3, 224, 224) preprocessed images

        Returns:
            logits_per_image: (B, num_labels) similarity scores
        """
        # Encode image
        image_features = self.model.encode_image(image_pixel_values_4d)

        # Store patch features from vision transformer (before final projection)
        # CLIP's vision model structure: visual.transformer.resblocks[...] -> ln_post -> proj
        # We want features before projection
        self.patch_features_from_vision_output = self.model.visual.transformer.resblocks[-1].ln_1.normalized_shape

        # Normalize image features
        image_features_normalized = image_features / image_features.norm(dim=-1, keepdim=True)

        # Compute similarity with text features
        # text_features is already normalized (done in __init__ and set_labels)
        logits_per_image = image_features_normalized @ self.text_features.T

        # Scale by temperature (CLIP uses 100 as learned parameter, but it's absorbed in the model)
        # The model's logit_scale is already applied in the @ operation via normalized features
        logits_per_image = logits_per_image * self.model.logit_scale.exp()

        return logits_per_image  # (B, num_labels)

    def get_patch_features(self):
        """Returns dummy patch features (not used in CLIP, kept for API compatibility)"""
        # CLIP doesn't expose patch features as easily as SigLIP
        # Return None or dummy tensor
        if self.patch_features_from_vision_output is not None:
            return torch.zeros(1, 196, 512)  # Dummy for ViT-B/16
        return None


# --- 2. Reshape Transform for CLIP Vision Transformer Output ---
def reshape_transform_clip(tensor, patch_grid_h, patch_grid_w):
    """
    Reshape CLIP ViT output for GradCAM.

    CLIP ViT structure:
    - Input: 224x224 image
    - Patch size: 16x16 (or 32x32 for ViT-B/32)
    - Patch grid: 14x14 for ViT-B/16 (224/16), 7x7 for ViT-B/32 (224/32)
    - CLS token at position 0, followed by 196 (or 49) patch tokens

    Args:
        tensor: (B, 1 + num_patches, D) output from target layer
        patch_grid_h: Height of patch grid (e.g., 14 for ViT-B/16)
        patch_grid_w: Width of patch grid (e.g., 14 for ViT-B/16)

    Returns:
        (B, D, H_grid, W_grid) conv-like format for GradCAM
    """
    expected_num_patches = patch_grid_h * patch_grid_w

    # Handle different tensor shapes from CLIP's layer norm
    # Sometimes we get (B, N, D), sometimes (N, B, D) depending on the layer
    if tensor.ndim == 3:
        # Check which dimension is the batch dimension
        if tensor.shape[0] == 1 and tensor.shape[1] > 100:
            # (1, N, D) - normal case
            pass
        elif tensor.shape[1] == 1 and tensor.shape[0] > 100:
            # (N, 1, D) - transposed, fix it
            tensor = tensor.transpose(0, 1)  # Now (1, N, D)
        # else: assume (B, N, D) is correct

    # CLIP ViT has CLS token at position 0
    if tensor.shape[1] == expected_num_patches + 1:
        # Remove CLS token (position 0)
        tensor = tensor[:, 1:, :]
    elif tensor.shape[1] != expected_num_patches:
        print(
            f"Warning: Expected {expected_num_patches} patches ({patch_grid_h}x{patch_grid_w}), "
            f"but got {tensor.shape[1]} tokens. "
            f"Full tensor shape: {tensor.shape}"
        )
        # Try to handle by removing first token if close
        if tensor.shape[1] == expected_num_patches + 1:
            tensor = tensor[:, 1:, :]

    # Reshape to (B, H_patch, W_patch, D)
    try:
        result = tensor.reshape(
            tensor.size(0),
            patch_grid_h,
            patch_grid_w,
            tensor.size(2)
        )
    except RuntimeError as e:
        print(f"Error during reshape: {e}.")
        print(f"  Tensor shape: {tensor.shape}")
        print(f"  Target grid: {patch_grid_h}x{patch_grid_w}")
        print(f"  Expected after removing CLS: ({tensor.size(0)}, {expected_num_patches}, {tensor.size(2)})")
        raise e

    # Permute to (B, D, H_patch, W_patch) for GradCAM
    result = result.permute(0, 3, 1, 2)
    return result


# --- 3. CLIP GradCAM Saliency Generator ---
class CLIPGradCAMGenerator:
    """
    CLIP-based saliency generator using GradCAM.

    API-compatible with SigLIP2SaliencyGenerator for drop-in replacement.
    Maintains preprocessing consistency for fair ablation comparisons.
    """

    def __init__(self,
                 candidate_labels: List[str],
                 model_name: str = "ViT-L/14@336px",  # Options: "ViT-B/32", "ViT-B/16", "ViT-L/14"
                 custom_device: Optional[str] = None):
        """
        Initialize CLIP GradCAM generator.

        Args:
            candidate_labels: List of semantic part labels (e.g., ['handle', 'spout', 'body'])
            model_name: CLIP model variant (ViT-B/16, ViT-B/32, ViT-L/14, etc.)
            custom_device: Force specific device (cuda/cpu), or auto-detect
        """
        # Lazy import of CLIP
        global clip
        if clip is None:
            try:
                import clip as clip_module
                clip = clip_module
            except ImportError:
                raise ImportError(
                    "CLIP library not found. Install it with:\n"
                    "  pip install git+https://github.com/openai/CLIP.git"
                )

        # Device setup
        if custom_device:
            self.device = custom_device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"Using device: {self.device}")

        # Load CLIP model
        self.model_name = model_name
        print(f"Loading CLIP model: {self.model_name}")
        try:
            self.model, self.preprocess = clip.load(model_name, device=self.device)
            self.model.eval()
        except Exception as e:
            print(f"Error loading CLIP model '{self.model_name}': {e}")
            raise

        self.candidate_labels = candidate_labels
        print(f"Initializing CLIPGradCAMGenerator for {len(candidate_labels)} labels: {candidate_labels}")

        # Get model configuration
        self.patch_size = self.model.visual.conv1.weight.shape[-1]  # Patch size from conv1 kernel
        self.target_image_height = self.target_image_width = self.model.visual.input_resolution
        self.H_patch_grid = self.target_image_height // self.patch_size
        self.W_patch_grid = self.target_image_width // self.patch_size

        print(f"Generator target image size: {self.target_image_height}x{self.target_image_width}")
        print(f"Generator patch size: {self.patch_size}")
        print(f"Generator patch grid: {self.H_patch_grid}x{self.W_patch_grid}")

        # Encode text labels
        self._encode_text_labels(candidate_labels)

        # Initialize GradCAM components
        self._initialize_gradcam()

        print("CLIPGradCAMGenerator initialized successfully.")

    def _encode_text_labels(self, labels: List[str]):
        """Encode text labels using CLIP text encoder."""
        # Format labels (matching SigLIP's convention of adding period)
        self.texts = [f'{label.lower()}.' for label in labels]

        # Tokenize
        text_tokens = clip.tokenize(self.texts).to(self.device)

        # Encode and normalize
        with torch.no_grad():
            text_features = self.model.encode_text(text_tokens)
            self.text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        print(f"  Encoded {len(labels)} text labels, features shape: {self.text_features.shape}")

    def _initialize_gradcam(self):
        """Initialize GradCAM wrapper and extractor."""
        # Create wrapper
        self.cam_wrapper = CLIPGradCAMWrapper(
            self.model,
            self.text_features,
            self.device
        )

        # Reshape transform
        self.reshape_fn = functools.partial(
            reshape_transform_clip,
            patch_grid_h=self.H_patch_grid,
            patch_grid_w=self.W_patch_grid
        )

        # Target layer: last transformer block's layer norm
        # CLIP structure: visual.transformer.resblocks[11].ln_1 for ViT-B/16
        try:
            num_layers = len(self.model.visual.transformer.resblocks)
            self.target_layers = [self.model.visual.transformer.resblocks[-1].ln_1]
            print(f"Generator using target layer: resblocks[{num_layers-1}].ln_1")
        except AttributeError as e:
            print(f"Error accessing CLIP vision transformer layers: {e}")
            raise

        # Initialize GradCAM
        self.cam_extractor = GradCAM(
            model=self.cam_wrapper,
            target_layers=self.target_layers,
            reshape_transform=self.reshape_fn
        )

    def set_labels(self, new_labels: List[str]):
        """
        Update semantic labels without recreating the model.

        This is much faster than recreating the generator since it only re-encodes
        text (cheap) instead of reloading the CLIP model (expensive).

        Args:
            new_labels: New list of semantic part labels
        """
        # Check if labels actually changed
        if self.candidate_labels == new_labels:
            return

        print(f"Updating CLIP labels: {len(self.candidate_labels)} -> {len(new_labels)} labels")

        # Update labels
        self.candidate_labels = new_labels

        # Re-encode text
        self._encode_text_labels(new_labels)

        # Update wrapper's text features
        self.cam_wrapper.text_features = self.text_features

        print(f"  Labels updated successfully")

    def _preprocess_image(self, pil_image: Image.Image) -> torch.Tensor:
        """
        Preprocess PIL image for CLIP model.

        CRITICAL: Must match CLIP's exact preprocessing:
        - Resize to 224x224 (BICUBIC)
        - Convert to tensor [0, 1]
        - Normalize with CLIP's mean/std

        Args:
            pil_image: PIL Image (any size)

        Returns:
            (1, 3, 224, 224) preprocessed tensor
        """
        # Use CLIP's built-in preprocess, but we need to apply it manually
        # to have full control for consistency with SigLIP preprocessing flow

        # Resize
        resized = pil_image.resize(
            (self.target_image_width, self.target_image_height),
            Image.Resampling.BICUBIC  # CLIP uses BICUBIC
        )

        # Convert to tensor and normalize
        transform = transforms.Compose([
            transforms.ToTensor(),  # Converts to [0, 1] and (C, H, W)
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],  # CLIP-specific
                std=[0.26862954, 0.26130258, 0.27577711]
            )
        ])

        pixel_values = transform(resized).unsqueeze(0)  # (1, 3, H, W)
        return pixel_values

    def process_frame(self, raw_image: Image.Image, visualize: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate saliency maps and patch features for a given image frame.

        Args:
            raw_image: PIL Image (cropped to object bbox, any size)
            visualize: If True, display visualization

        Returns:
            saliency_maps: (num_labels, H_patch, W_patch) torch.Tensor
            patch_features: (num_labels, N_patches, D) torch.Tensor (dummy for CLIP)
        """
        if not hasattr(self, 'cam_extractor') or self.cam_extractor is None:
            print("Error: CAM extractor not initialized. Cannot process frame.")
            return torch.empty(0), torch.empty(0)

        # Preprocess image
        pixel_values_for_cam = self._preprocess_image(raw_image).to(self.device)

        # For visualization
        pil_resized = raw_image.resize(
            (self.target_image_width, self.target_image_height),
            Image.Resampling.BICUBIC
        )
        rgb_img_for_viz = np.array(pil_resized) / 255.0

        saliency_maps_list = []

        # Generate GradCAM for each label sequentially
        for i, label_text_original_case in enumerate(self.candidate_labels):
            targets_for_cam = [ClassifierOutputTarget(i)]

            # Try with OOM retry logic (matching SigLIP)
            max_retries = 2
            grayscale_cam = None

            for retry in range(max_retries):
                try:
                    grayscale_cam = self.cam_extractor(
                        input_tensor=pixel_values_for_cam,
                        targets=targets_for_cam
                    )
                    break  # Success!

                except torch.cuda.OutOfMemoryError as oom_error:
                    print(f"OOM during CAM generation for '{label_text_original_case}' (attempt {retry+1}/{max_retries})")

                    # Clear CUDA cache and retry
                    torch.cuda.empty_cache()
                    gc.collect()

                    if retry == max_retries - 1:
                        print(f"CRITICAL: Failed to generate CAM for '{label_text_original_case}' after {max_retries} attempts")
                        raise RuntimeError(
                            f"OOM during saliency generation for label '{label_text_original_case}'. "
                            f"Cannot generate partial saliency maps as this causes dimension mismatch errors."
                        ) from oom_error
                    else:
                        print(f"  Retrying after cache clear...")
                        continue

                except Exception as e:
                    print(f"Error during CAM generation for '{label_text_original_case}': {e}")
                    import traceback
                    traceback.print_exc()
                    raise RuntimeError(
                        f"Failed to generate CAM for label '{label_text_original_case}': {e}"
                    ) from e

            if grayscale_cam is None:
                raise RuntimeError(f"CAM generation returned None for '{label_text_original_case}'")

            # Extract single CAM (handle batch dimension)
            if grayscale_cam.ndim == 3 and grayscale_cam.shape[0] == 1:
                grayscale_cam_single = grayscale_cam[0, :]
            elif grayscale_cam.ndim == 2:
                grayscale_cam_single = grayscale_cam
            else:
                print(f"Warning: Unexpected CAM output shape {grayscale_cam.shape} for label '{label_text_original_case}'. Skipping.")
                continue

            saliency_maps_list.append(grayscale_cam_single)

            # Visualization
            if visualize:
                try:
                    cam_image = show_cam_on_image(rgb_img_for_viz, grayscale_cam_single, use_rgb=True)
                    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
                    fig.suptitle(f"CLIP Localization for: '{label_text_original_case}' (GradCAM)", fontsize=16)
                    axs[0].imshow(pil_resized)
                    axs[0].set_title("Resized Input Image (for Model)")
                    axs[0].axis('off')
                    axs[1].imshow(cam_image)
                    axs[1].set_title("GradCAM Overlay")
                    axs[1].axis('off')
                    plt.tight_layout(rect=[0, 0, 1, 0.96])
                    plt.show()
                except Exception as e:
                    print(f"Error during visualization for '{label_text_original_case}': {e}")
                    continue

        if not saliency_maps_list:
            print("No saliency maps were generated for this frame.")
            return torch.empty(0), torch.empty(0)

        # Validate we got saliency for ALL labels
        expected_num_labels = len(self.candidate_labels)
        actual_num_labels = len(saliency_maps_list)
        if actual_num_labels != expected_num_labels:
            raise RuntimeError(
                f"Saliency generation incomplete: got {actual_num_labels} labels but expected {expected_num_labels}. "
                f"This would cause dimension mismatch errors during pose estimation."
            )

        # Stack saliency maps
        saliency_maps_tensor = torch.from_numpy(np.stack(saliency_maps_list))

        # Create dummy patch features for API compatibility
        # Shape: (num_labels, N_patches, D)
        num_patches = self.H_patch_grid * self.W_patch_grid
        feature_dim = 512 if 'B' in self.model_name else 768  # ViT-B: 512, ViT-L: 768
        patch_features_tensor = torch.zeros(expected_num_labels, num_patches, feature_dim)

        return saliency_maps_tensor, patch_features_tensor

    def release_gpu_memory(self):
        """Release GPU memory held by this generator."""
        if hasattr(self, 'cam_extractor'):
            del self.cam_extractor
        if hasattr(self, 'cam_wrapper'):
            del self.cam_wrapper
        if hasattr(self, 'model'):
            del self.model
        torch.cuda.empty_cache()
        gc.collect()
        print("CLIP generator GPU memory released")

    def cleanup(self):
        """Cleanup method for compatibility with OneShotPoseEstimator."""
        self.release_gpu_memory()
