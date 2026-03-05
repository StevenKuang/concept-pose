from sklearn.decomposition import PCA
import torch
import requests
from PIL import Image
import numpy as np
from transformers import AutoProcessor, AutoModel
from pytorch_grad_cam import GradCAMPlusPlus, GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import matplotlib.pyplot as plt
import functools
import torchvision.transforms.functional as TF
from typing import List, Tuple, Optional
import gc

# --- 1. SigLIP Model Wrapper for Grad-CAM ---
class SigLIPGradCAMWrapper(torch.nn.Module):
    def __init__(self, siglip_model, text_input_ids, device="cuda"):
        super().__init__()
        self.model = siglip_model 
        self.text_input_ids = text_input_ids
        self.dtype = siglip_model.dtype
        self.patch_features_from_vision_output = None
        self.model_output = None

    def forward(self, image_pixel_values_4d):
        model_device = self.model.device # Get the model's actual device

        # Ensure all inputs are on the same device as the model
        # image_pixel_values_4d is already on model_device from SigLIP2SaliencyGenerator.process_frame
        # self.text_input_ids is already on model_device from SigLIP2SaliencyGenerator.__init__
        model_inputs = {
            "pixel_values": image_pixel_values_4d.to(self.dtype).to(self.model.device), 
            "input_ids": self.text_input_ids,
            "attention_mask":torch.ones_like(self.text_input_ids)
        }

        outputs = self.model(**model_inputs)
        # Store patch features from the vision model's last hidden state
        # These features are captured *before* any projection or pooling specific to logits_per_image
        self.patch_features_from_vision_output = outputs.vision_model_output.last_hidden_state.detach().cpu()
        self.model_output = outputs
        logits = outputs.logits_per_image # These are similarity scores between image and each text
        # probs = torch.sigmoid(logits) # Optional: convert to probabilities if needed elsewhere
        
        # print(f"Logits: {logits}")
        # print(f"Logits shape: {logits.shape}") 
        # print(f"Patch features shape from vision output: {self.patch_features_from_vision_output.shape}") 
        # print(f"Probs are: {[f'{value.item() * 100:.2f}%' for value in probs.flatten()]}")
        return logits # Grad-CAM will use these logits as scores for each class (text label)

    def get_patch_features(self):
        # Returns the raw patch features from the vision transformer's output
        return self.patch_features_from_vision_output

    def get_model_output(self):
        return self.model_output

# --- 2. Reshape Transform for Vision Transformer Output ---
def reshape_transform_siglip(tensor, patch_grid_h, patch_grid_w):
    expected_num_patches = patch_grid_h * patch_grid_w
    # tensor shape is (batch_size, num_patches_from_layer, feature_dim)
    
    if tensor.shape[1] != expected_num_patches:
        print(
            f"Warning: Mismatch in num_patches for reshape. Tensor has {tensor.shape[1]} patches, "
            f"but expected {expected_num_patches} ({patch_grid_h}x{patch_grid_w}). "
        )
        # Common case: Vision Transformers might output a CLS token + patch tokens
        if tensor.shape[1] == expected_num_patches + 1:
            print("    Assuming CLS token present, slicing it off for CAM reshape.")
            tensor = tensor[:, 1:, :] # Remove the CLS token
        elif tensor.shape[1] < expected_num_patches :
             print(f"    ERROR: Not enough patches from target layer ({tensor.shape[1]}) for reshape to {patch_grid_h}x{patch_grid_w}. CAM will likely fail or be incorrect.")
             # Pad if fewer patches, though this might lead to incorrect CAMs
             b, _, c = tensor.shape
             reshaped_tensor = torch.zeros(b, expected_num_patches, c, device=tensor.device, dtype=tensor.dtype)
             if tensor.shape[1] > 0 :
                reshaped_tensor[:, :tensor.shape[1], :] = tensor
             tensor = reshaped_tensor
        # If tensor.shape[1] > expected_num_patches + 1, or other mismatches, the reshape might be problematic.
        # For now, we proceed, but this indicates a potential issue with layer selection or understanding model architecture.

    # Reshape to (batch_size, H_patch, W_patch, feature_dim)
    try:
        result = tensor.reshape(tensor.size(0),
                                patch_grid_h,
                                patch_grid_w,
                                tensor.size(2))
    except RuntimeError as e:
        print(f"Error during reshape: {e}. Tensor shape: {tensor.shape}, Target grid: {patch_grid_h}x{patch_grid_w}")
        # Fallback or re-raise, depending on desired error handling
        # For now, let GradCAM handle potential downstream errors if reshape fails.
        # A possible fallback could be to return a zero tensor of the expected permuted shape.
        # However, it's better to ensure the target layer provides compatible output.
        raise e


    # Permute to (batch_size, feature_dim, H_patch, W_patch) as expected by GradCAM for conv-like features
    result = result.permute(0, 3, 1, 2)
    return result

class SigLIP2SaliencyGenerator:
    def __init__(self, 
                 candidate_labels: List[str], 
                 model_name: str = "google/siglip2-giant-opt-patch16-384",  # MODEL ABLATION: Uncomment to test different models
                 # model_name: str = "google/siglip2-base-patch16-384",      # 86M params, ~4x faster, lower accuracy
                 # model_name: str = "google/siglip2-large-patch16-384",     # 303M params, ~2.5x faster, good trade-off
                 # model_name: str = "google/siglip2-large-patch16-512",     # 303M params, ~2x faster, higher resolution
                 # model_name: str = "google/siglip2-so400m-patch14-384",    # 400M params, ~2x faster, finer patches (14x14)
                 
                 custom_device: Optional[str] = None):
        if custom_device:
            self.device = custom_device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_dtype = torch.float16
        print(f"Using device: {self.device}, Model dtype: {self.model_dtype}")

        # --- Load SigLIP Model and Processor ---
        self.model_name = model_name
        print(f"Loading model: {self.model_name}")
        try:
            self.model = AutoModel.from_pretrained(
                self.model_name,
                torch_dtype=self.model_dtype,
                device_map=self.device,
                attn_implementation="sdpa"
            ).eval().to(self.device)
            self.processor = AutoProcessor.from_pretrained(self.model_name, use_fast=True)
        except Exception as e:
            print(f"Error loading model or processor '{self.model_name}': {e}")
            raise 

        self.candidate_labels = candidate_labels
        print(f"Initializing SigLIP2SaliencyGenerator for labels: {self.candidate_labels}")

        # --- Pre-process Text Inputs  ---
        self.texts = [f'{label.lower()}.' for label in self.candidate_labels]
        self.input_text_tokenized = self.processor(text=self.texts, images=None, padding="max_length", return_tensors="pt", max_length=64)
        self.text_input_ids = self.input_text_tokenized['input_ids'].to(self.device)

        # --- Get Model-specific Configuration  ---
        self.patch_size = self.model.vision_model.config.patch_size
        try:
            target_image_size_config = self.model.vision_model.config.image_size
            if isinstance(target_image_size_config, int):
                self.target_image_height = self.target_image_width = target_image_size_config
            elif isinstance(target_image_size_config, (list, tuple)) and len(target_image_size_config) == 2:
                self.target_image_height, self.target_image_width = target_image_size_config
            else:
                print(f"Unexpected image_size format in config: {target_image_size_config}. Defaulting to 384x384.")
                self.target_image_height = self.target_image_width = 384
        except AttributeError:
            print("model.vision_model.config.image_size not found. Defaulting to 384x384.")
            self.target_image_height = self.target_image_width = 384
        
        print(f"Generator target image size: {self.target_image_height}x{self.target_image_width}")
        self.H_patch_grid = self.target_image_height // self.patch_size
        self.W_patch_grid = self.target_image_width // self.patch_size
        print(f"Generator patch grid: {self.H_patch_grid}x{self.W_patch_grid}")

        # --- Initialize Grad-CAM Components ---
        self.cam_wrapper = SigLIPGradCAMWrapper(self.model, self.text_input_ids, self.device)
        self.reshape_fn = functools.partial(reshape_transform_siglip,
                                            patch_grid_h=self.H_patch_grid,
                                            patch_grid_w=self.W_patch_grid)
        try:
            self.target_layers = [self.model.vision_model.post_layernorm]
            print(f"Generator using target layer for CAM: {self.target_layers[0].__class__.__name__} (Path: model.vision_model.post_layernorm)")
        except AttributeError as e:
            print(f"Error accessing target layer 'model.vision_model.post_layernorm' in generator: {e}. CAM generation will fail.")
            raise 
        
        self.cam_extractor = GradCAM(
            model=self.cam_wrapper,
            target_layers=self.target_layers,
            reshape_transform=self.reshape_fn
        )
        print("SigLIP2SaliencyGenerator initialized successfully.")

    def to(self, device: str):
        """
        Move model and tensors to the given device (e.g. 'cpu' or 'cuda').

        The cam_wrapper.model is the same object as self.model, so moving
        self.model moves it for GradCAM as well.
        """
        self.device = device
        self.model = self.model.to(device)
        self.text_input_ids = self.text_input_ids.to(device)
        # cam_wrapper holds a reference to self.model (already moved)
        # but update its text_input_ids reference too
        self.cam_wrapper.text_input_ids = self.text_input_ids
        return self

    def set_labels(self, new_labels: List[str]):
        """
        Update semantic labels without recreating the model.

        This is much faster than recreating the generator since it only retokenizes
        text (cheap) instead of reloading the 2GB+ SigLIP model (expensive).

        Args:
            new_labels: New list of semantic part labels
        """
        # Check if labels actually changed to avoid unnecessary work
        if self.candidate_labels == new_labels:
            return

        print(f"Updating SigLIP2 labels: {len(self.candidate_labels)} -> {len(new_labels)} labels")

        # Update labels
        self.candidate_labels = new_labels

        # Re-tokenize text (lightweight operation)
        self.texts = [f'{label.lower()}.' for label in self.candidate_labels]
        self.input_text_tokenized = self.processor(
            text=self.texts,
            images=None,
            padding="max_length",
            return_tensors="pt",
            max_length=64
        )
        self.text_input_ids = self.input_text_tokenized['input_ids'].to(self.device)

        # Update wrapper's text_input_ids in-place (no recreation needed!)
        # The wrapper just stores this as an attribute and uses it in forward()
        self.cam_wrapper.text_input_ids = self.text_input_ids

        print(f"  Labels updated successfully to: {self.candidate_labels}")

    def process_frame(self, raw_image: Image.Image, visualize: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generates saliency maps and patch features for a given image frame.
        """
        if not hasattr(self, 'cam_extractor') or self.cam_extractor is None:
            print("Error: CAM extractor not initialized. Cannot process frame.")
            return torch.empty(0), torch.empty(0)
            
        # print(f"\n--- Processing new frame ---")
        # --- Prepare Image Input for the current frame ---
        if raw_image.__class__ == torch.Tensor:
            pil_resized_image = raw_image
        else:
            pil_resized_image = raw_image.resize((self.target_image_width, self.target_image_height), Image.Resampling.BILINEAR)

        image_only_inputs = self.processor(text=None, images=pil_resized_image, return_tensors="pt")
        # Ensure pixel_values are on the correct device and have the model's expected dtype
        pixel_values_for_cam = image_only_inputs['pixel_values'].to(self.device, dtype=self.model_dtype)


        rgb_img_for_viz = np.array(pil_resized_image) / 255.0
        saliency_maps_list = []
        patch_features_list = []

        for i, label_text_original_case in enumerate(self.candidate_labels):
            # print(f"\nGenerating CAM for: '{label_text_original_case}' (label index {i})")
            targets_for_cam = [ClassifierOutputTarget(i)]

            # Try with OOM retry logic
            max_retries = 2
            grayscale_cam = None
            patch_feature_for_current_img = None

            for retry in range(max_retries):
                try:
                    grayscale_cam = self.cam_extractor(
                        input_tensor=pixel_values_for_cam,
                        targets=targets_for_cam
                    )
                    patch_feature_for_current_img = self.cam_wrapper.get_patch_features()
                    # print(f"Shape of patch_features from wrapper for label '{label_text_original_case}': {patch_feature_for_current_img.shape}")
                    break  # Success!

                except torch.cuda.OutOfMemoryError as oom_error:
                    print(f"OOM during CAM generation for '{label_text_original_case}' (attempt {retry+1}/{max_retries})")

                    # Clear CUDA cache and retry
                    torch.cuda.empty_cache()
                    import gc
                    gc.collect()

                    if retry == max_retries - 1:
                        # Final attempt failed - this is critical, raise error
                        print(f"CRITICAL: Failed to generate CAM for '{label_text_original_case}' after {max_retries} attempts")
                        print(f"Cannot proceed with partial saliency maps - dimension mismatch would occur")
                        raise RuntimeError(
                            f"OOM during saliency generation for label '{label_text_original_case}'. "
                            f"Cannot generate partial saliency maps as this causes dimension mismatch errors. "
                            f"Try: (1) Reduce num_semantic_labels, (2) Use fewer labels per category, "
                            f"or (3) Free GPU memory from other processes."
                        ) from oom_error
                    else:
                        print(f"  Retrying after cache clear...")
                        continue

                except Exception as e:
                    print(f"Error during CAM generation for '{label_text_original_case}': {e}")
                    import traceback
                    traceback.print_exc()
                    # For non-OOM errors, also fail critically to avoid dimension mismatch
                    raise RuntimeError(
                        f"Failed to generate CAM for label '{label_text_original_case}': {e}"
                    ) from e

            if grayscale_cam is None:
                # Should not reach here due to raises above, but safety check
                raise RuntimeError(f"CAM generation returned None for '{label_text_original_case}'")

            if grayscale_cam.ndim == 3 and grayscale_cam.shape[0] == 1:
                grayscale_cam_single = grayscale_cam[0, :]
            elif grayscale_cam.ndim == 2:
                grayscale_cam_single = grayscale_cam
            else:
                print(f"Warning: Unexpected CAM output shape {grayscale_cam.shape} for label '{label_text_original_case}'. Skipping.")
                continue
            
            # print(f"Raw CAM min: {grayscale_cam_single.min():.4f}, max: {grayscale_cam_single.max():.4f}, mean: {grayscale_cam_single.mean():.4f}")
            saliency_maps_list.append(grayscale_cam_single)
            patch_features_list.append(patch_feature_for_current_img.squeeze(0)) 

            if visualize:
                try:
                    cam_image = show_cam_on_image(rgb_img_for_viz, grayscale_cam_single, use_rgb=True)
                    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
                    fig.suptitle(f"SigLIP Localization for: '{label_text_original_case}' (GradCAM)", fontsize=16)
                    axs[0].imshow(pil_resized_image)
                    axs[0].set_title("Resized Input Image (for Model)")
                    axs[0].axis('off')
                    axs[1].imshow(cam_image)
                    axs[1].set_title("GradCAM Overlay")
                    axs[1].axis('off')
                    plt.tight_layout(rect=[0, 0, 1, 0.96])
                    plt.show()
                except Exception as e:
                    print(f"Error during show_cam_on_image for '{label_text_original_case}': {e}")
                    continue
        
        if not saliency_maps_list:
            print("No saliency maps were generated for this frame.")
            return torch.empty(0), torch.empty(0)

        # Validate we got saliency for ALL labels (critical to avoid dimension mismatch)
        expected_num_labels = len(self.candidate_labels)
        actual_num_labels = len(saliency_maps_list)
        if actual_num_labels != expected_num_labels:
            raise RuntimeError(
                f"Saliency generation incomplete: got {actual_num_labels} labels but expected {expected_num_labels}. "
                f"This would cause dimension mismatch errors during pose estimation. "
                f"All labels must succeed or the frame should fail."
            )

        saliency_maps_tensor = torch.from_numpy(np.stack(saliency_maps_list))
        patch_features_tensor = torch.stack(patch_features_list)

        # print(f"--- Frame processing finished ---")
        return saliency_maps_tensor, patch_features_tensor

    def process_frames_batch(self, 
                             raw_images: List[Image.Image], 
                             visualize: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generates saliency maps for a BATCH of image frames against all candidate labels efficiently.
        Returns a tensor of shape [B, T, H, W] where B is batch_size, T is num_labels.
        """
        batch_size = len(raw_images)
        print(f"\n--- Processing a batch of {batch_size} frames ---")

        processed_images = self.processor(text=None, images=raw_images, return_tensors="pt")['pixel_values']
        batched_pixel_values = processed_images.to(self.device, dtype=self.model_dtype) # (B, 3, 384, 384)

        all_labels_cams = []

        for j, label_text in enumerate(self.candidate_labels):
            targets_for_batch = [ClassifierOutputTarget(j) for _ in range(batch_size)]
            try:
                grayscale_cams = self.cam_extractor(
                    input_tensor=batched_pixel_values,
                    targets=targets_for_batch
                ) # (B, H, W)
                all_labels_cams.append(grayscale_cams)

            except Exception as e:
                print(f"Error during batched CAM generation for label '{label_text}': {e}")
                all_labels_cams.append(np.zeros((batch_size, self.H_patch_grid, self.W_patch_grid)))
                continue
        
        if not all_labels_cams:
            print("No saliency maps were generated for the batch.")
            return torch.empty(0), torch.empty(0)
            
        saliency_maps_tensor = torch.from_numpy(np.stack(all_labels_cams)).permute(1, 0, 2, 3) # (B, S, H, W)
        
        # grab patch features only once for the batch
        patch_features_for_batch = self.cam_wrapper.get_patch_features() # Shape: (B, num_patches, dim)
        
        if visualize:
            print("Visualizing results for the first frame in the batch...")
            first_frame_viz = np.array(raw_images[0].resize((self.target_image_width, self.target_image_height))) / 255.0
            for j, label_text in enumerate(self.candidate_labels):
                cam_image = show_cam_on_image(first_frame_viz, saliency_maps_tensor[0, j].cpu().numpy(), use_rgb=True)
                plt.imshow(cam_image)
                plt.title(f"Frame 0, Label: '{label_text}'")
                plt.show()

        return saliency_maps_tensor, patch_features_for_batch
    
    def release_gpu_memory(self):
        """
        Attempts to release GPU memory held by this generator instance.
        """
        if hasattr(self, 'cam_extractor'):
            del self.cam_extractor
            self.cam_extractor = None

        if hasattr(self, 'cam_wrapper'):
            if hasattr(self.cam_wrapper, 'model'):
                 # This reference is to self.model, so deleting self.model will handle it.
                 # However, explicit deletion can be clearer.
                del self.cam_wrapper.model
                self.cam_wrapper.model = None
            del self.cam_wrapper
            self.cam_wrapper = None
            
        if hasattr(self, 'model'):
            del self.model # This is the main Hugging Face model
            self.model = None
            
        if hasattr(self, 'text_input_ids') and self.text_input_ids is not None:
            del self.text_input_ids
            self.text_input_ids = None

        # Also good to clean up other potentially large attributes if they exist
        # and are not needed anymore, though processor is usually light.
        if hasattr(self, 'processor'):
            self.processor = None 
        
        self.target_layers = None 

        if torch.cuda.is_available():
            # Perform garbage collection aggressively before emptying cache
            gc.collect() 
            torch.cuda.empty_cache()
        print("GPU memory release attempt complete for SigLIP2SaliencyGenerator.")
    
    def cleanup(self):
        """Convenient alias for release_gpu_memory()"""
        self.release_gpu_memory()

def main():
    try:
        path = "data/bike.png"
        raw_image = Image.open(path).convert("RGB")
        print("Loaded local image" + path)
    except FileNotFoundError:
        print("Error: Local image " + path + " not found. Using fallback URL.")
        image_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
        try:
            raw_image = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")
            print(f"Loaded fallback image from URL: {image_url}")
        except Exception as e_fallback:
            print(f"Error loading fallback image: {e_fallback}. Using a plain gray image.")
            raw_image = Image.new('RGB', (384, 384), color = 'gray') 

    candidate_labels = ["saddle", "front derailleur", "back derailleur" "handlebars", "water bottles", "wheels"]
    # candidate_labels = ["a bike", "bike", "picture of a bicycle"]
    # candidate_labels = ["camera lens", "controls", "camera front", "camera base", "go pro", "camera grip", "price tag", "camera hot shoe"]
    # candidate_labels = ["book on a sofa", "brouchure on a sofa", "camera on a table", "camera on a sofa"]

    # Default model_name is "google/siglip2-giant-opt-patch16-384"
    try:
        generator = SigLIP2SaliencyGenerator(candidate_labels)
    except Exception as e:
        print(f"Failed to initialize SigLIP2SaliencyGenerator: {e}")
        return 
    saliency_maps_tensor, patch_features_tensor = generator.process_frame(
        raw_image=raw_image,
        visualize=True
    )

    # saliency_maps_tensor, patch_features_tensor = generator.process_frames_batch(
    #     raw_images=[raw_image, raw_image, raw_image],
    #     visualize=True
    # )

    # # visualize patch_features_tensor [l, 24*24, N]
    # pca = PCA(n_components=3)
    # patch_features_tensor_i_pca = pca.fit_transform(patch_features_tensor[0])
    # # map the PCA components to the range of 0-255 using min-max scaling
    # patch_features_tensor_i_pca_rgb = patch_features_tensor_i_pca.reshape(24, 24, 3)
    # patch_features_tensor_i_pca_rgb = (patch_features_tensor_i_pca_rgb - patch_features_tensor_i_pca_rgb.min()) / (patch_features_tensor_i_pca_rgb.max() - patch_features_tensor_i_pca_rgb.min() + 1e-6)
    # patch_features_tensor_i_pca_rgb = (patch_features_tensor_i_pca_rgb * 255).astype(np.uint8)
    # plt.imshow(patch_features_tensor_i_pca_rgb)
    # plt.show()
    

    if saliency_maps_tensor.nelement() > 0: 
        print(f"\nReturned Saliency maps tensor shape from main: {saliency_maps_tensor.shape}")
        print(f"Returned Patch features tensor shape from main: {patch_features_tensor.shape}")
    else:
        print("\nNo saliency maps or patch features were generated in main.")
    
if __name__ == '__main__':
    main()
