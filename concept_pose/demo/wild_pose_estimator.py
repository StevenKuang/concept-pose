"""
Wild Pose Estimator - In-the-Wild Relative Pose Estimation
============================================================

Estimate relative 6D pose between two arbitrary images of the same object
using:
- SAM3: Object segmentation (mask extraction via text prompt)
- DepthAnything3: Metric depth prediction + camera intrinsics
- ConceptPose: Semantic part-based 3D-3D registration

This is the core demo entry point for ConceptPose on in-the-wild images.

Usage:
    from concept_pose.demo import estimate_relative_pose

    # Simple API
    result = estimate_relative_pose(
        anchor_image='path/to/anchor.jpg',
        query_image='path/to/query.jpg',
        category='bottle'
    )

    # Access results
    R_relative = result['R']  # (3, 3) rotation
    t_relative = result['t']  # (3,) translation

    # Or with custom concepts
    result = estimate_relative_pose(
        anchor_image='path/to/anchor.jpg',
        query_image='path/to/query.jpg',
        category='bottle',
        concepts=['neck', 'body', 'cap', 'base']
    )

Author: ConceptPose Team
Date: 2025
"""

import random
import numpy as np
import torch
import gc
from pathlib import Path
from PIL import Image
from typing import Dict, List, Optional, Tuple, Union

# Default target size for preprocessing (same as test_oneshot.py)
DEFAULT_TARGET_SIZE = 384


def set_deterministic_mode(seed: int = 42):
    """Set random seed for reproducible RANSAC results (including GPU operations)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"Deterministic mode enabled (seed={seed})")


def _sam3_segment_worker(image_paths: List[str], prompt: str, device: str, result_save_path: str):
    """
    Worker function for SAM3 segmentation that runs in a separate process.
    Uses file-based communication to avoid Queue deadlocks with large data.

    Must be at module level to be picklable for multiprocessing.
    """
    import os
    import torch
    import numpy as np
    import pickle
    from PIL import Image

    print(f"  [SAM3 Worker {os.getpid()}] Starting...")

    try:
        from transformers import Sam3Processor, Sam3Model

        print(f"  [SAM3 Worker {os.getpid()}] Loading SAM3 model...")
        model = Sam3Model.from_pretrained(
            "facebook/sam3",
            torch_dtype=torch.float16,
            device_map=None,
        ).to(device)
        processor = Sam3Processor.from_pretrained("facebook/sam3")
        print(f"  [SAM3 Worker {os.getpid()}] SAM3 loaded")

        masks = []
        for i, img_path in enumerate(image_paths):
            print(f"  [SAM3 Worker {os.getpid()}] Processing image {i+1}/{len(image_paths)}...")
            pil_image = Image.open(img_path).convert('RGB')

            inputs = processor(
                images=pil_image,
                text=prompt,
                return_tensors="pt"
            ).to(device)

            with torch.no_grad():
                outputs = model(**inputs)

            results = processor.post_process_instance_segmentation(
                outputs,
                threshold=0.5,
                mask_threshold=0.5,
                target_sizes=inputs.get("original_sizes").tolist()
            )[0]

            if len(results['masks']) == 0:
                print(f"    Warning: No '{prompt}' found, using full image")
                mask = np.ones((pil_image.height, pil_image.width), dtype=np.float32)
            else:
                best_idx = results['scores'].argmax()
                mask = results['masks'][best_idx].cpu().numpy().astype(np.float32)

            masks.append(mask)

        # Save results to file (avoids Queue deadlock with large data)
        masks_bytes = pickle.dumps(masks)
        np.savez_compressed(
            result_save_path,
            success=True,
            masks_bytes=masks_bytes
        )
        print(f"  [SAM3 Worker {os.getpid()}] Done, saved results to {result_save_path}")

    except Exception as e:
        import traceback
        error_msg = f"{e}\n{traceback.format_exc()}"
        print(f"  [SAM3 Worker {os.getpid()}] Error: {error_msg}")
        np.savez_compressed(result_save_path, success=False, error_message=error_msg)

    print(f"  [SAM3 Worker {os.getpid()}] Exiting (VRAM will be released)")


class WildPoseEstimator:
    """
    End-to-end pose estimator for in-the-wild image pairs.

    Integrates SAM3 (segmentation), DepthAnything3 (depth + intrinsics),
    and ConceptPose (semantic 3D registration) into a single pipeline.

    Attributes:
        device: Torch device ('cuda' or 'cpu')
        depth_model: DepthAnything3 model instance
        sam_model: SAM3 model instance
        sam_processor: SAM3 processor instance
        estimator: OneShotPoseEstimator instance
    """

    def __init__(
        self,
        device: str = 'cuda',
        depth_model_name: str = 'depth-anything/DA3NESTED-GIANT-LARGE',
        lazy_load: bool = True
    ):
        """
        Initialize the wild pose estimator.

        Args:
            device: Torch device
            depth_model_name: HuggingFace model ID for DepthAnything3
            lazy_load: If True, load models on first use (saves memory)
        """
        self.device = device
        self.depth_model_name = depth_model_name

        # Models (lazy loaded)
        self._depth_model = None
        self._sam_model = None
        self._sam_processor = None
        self._estimator = None

        if not lazy_load:
            self._load_depth_model()
            self._load_sam_model()

    def _load_depth_model(self):
        """Load DepthAnything3 model."""
        if self._depth_model is not None:
            return

        print(f"Loading DepthAnything3 model: {self.depth_model_name}")
        try:
            from depth_anything_3.api import DepthAnything3
            self._depth_model = DepthAnything3.from_pretrained(self.depth_model_name)
            self._depth_model = self._depth_model.to(device=self.device)
            print("  DepthAnything3 loaded successfully")
        except ImportError:
            raise ImportError(
                "DepthAnything3 not installed. Please install:\n"
                "  pip install git+https://github.com/ByteDance-Seed/Depth-Anything-3.git"
            )

    def _load_sam_model(self):
        """Load SAM3 model for text-prompted segmentation."""
        if self._sam_model is not None:
            return

        print("Loading SAM3 model...")
        try:
            from transformers import Sam3Processor, Sam3Model
            self._sam_model = Sam3Model.from_pretrained("facebook/sam3").to(self.device)
            self._sam_processor = Sam3Processor.from_pretrained("facebook/sam3")
            print("  SAM3 loaded successfully")
        except ImportError:
            raise ImportError(
                "SAM3 not available. Please install/update transformers:\n"
                "  pip install transformers>=4.40.0"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load SAM3 model: {e}")

    def _unload_sam_model(self):
        """Unload SAM3 to free memory."""
        if self._sam_model is not None:
            print("  Unloading SAM3 model from VRAM...")
            # Move model to CPU first to release VRAM
            self._sam_model.to('cpu')
            del self._sam_model
            del self._sam_processor
            self._sam_model = None
            self._sam_processor = None
            # Force garbage collection and clear CUDA cache
            gc.collect()
            torch.cuda.empty_cache()
            gc.collect()
            print("  SAM3 unloaded")

    @staticmethod
    def _run_sam3_in_subprocess(image_path: str, prompt: str, device: str) -> np.ndarray:
        """
        Run SAM3 segmentation in a separate process to ensure VRAM is fully released.

        This is necessary because PyTorch doesn't always release VRAM even after
        del + empty_cache(). Running in a subprocess guarantees cleanup.
        Uses file-based communication to avoid Queue deadlocks.
        """
        import os
        import multiprocessing as mp
        import tempfile
        import pickle

        result_path = ""
        try:
            with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as tmp_file:
                result_path = tmp_file.name

            ctx = mp.get_context('spawn')
            process = ctx.Process(
                target=_sam3_segment_worker,
                args=([image_path], prompt, device, result_path)
            )
            process.start()
            process.join()

            if os.path.exists(result_path):
                data = np.load(result_path, allow_pickle=True)
                if data.get('success'):
                    masks_bytes = data['masks_bytes']
                    masks = pickle.loads(masks_bytes)
                    return masks[0]  # Return first (and only) mask
                else:
                    error_msg = data.get('error_message', 'Unknown error')
                    raise RuntimeError(f"SAM3 subprocess failed: {error_msg}")
            else:
                raise RuntimeError("SAM3 subprocess crashed - no result file")

        finally:
            if result_path and os.path.exists(result_path):
                try:
                    os.remove(result_path)
                except Exception:
                    pass

    def _unload_depth_model(self):
        """Unload DepthAnything3 to free memory."""
        if self._depth_model is not None:
            print("  Unloading DepthAnything3 from VRAM...")
            # Move to CPU first to release CUDA memory
            self._depth_model.to('cpu')
            del self._depth_model
            self._depth_model = None
            # Ensure CUDA operations complete before clearing cache
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            gc.collect()
            print("  DepthAnything3 unloaded")

    def get_object_mask(
        self,
        pil_image: Image.Image,
        prompt: str
    ) -> np.ndarray:
        """
        Get object mask using SAM3 with text prompt.

        Args:
            pil_image: PIL Image object
            prompt: Text prompt for segmentation (e.g., "bottle")

        Returns:
            mask: (H, W) binary mask as float32
        """
        self._load_sam_model()

        inputs = self._sam_processor(
            images=pil_image,
            text=prompt,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self._sam_model(**inputs)

        # Post-process results
        results = self._sam_processor.post_process_instance_segmentation(
            outputs,
            threshold=0.5,
            mask_threshold=0.5,
            target_sizes=inputs.get("original_sizes").tolist()
        )[0]

        if len(results['masks']) == 0:
            print(f"  Warning: No '{prompt}' found, using full image")
            return np.ones((pil_image.height, pil_image.width), dtype=np.float32)

        # Use the highest confidence mask
        best_idx = results['scores'].argmax()
        mask = results['masks'][best_idx].cpu().numpy().astype(np.float32)

        return mask

    def get_object_masks_subprocess(
        self,
        image_paths: List[str],
        prompt: str
    ) -> List[np.ndarray]:
        """
        Get object masks for multiple images using SAM3 in a subprocess.

        This ensures VRAM is fully released after SAM3 finishes.
        The subprocess loads SAM3, processes all images, then exits (releasing all memory).
        Uses file-based communication to avoid Queue deadlocks with large mask data.

        Args:
            image_paths: List of image file paths
            prompt: Text prompt for segmentation

        Returns:
            List of (H, W) binary masks as float32
        """
        import os
        import multiprocessing as mp
        import tempfile
        import pickle

        print(f"  [Main] Spawning SAM3 subprocess for {len(image_paths)} images...")

        # Create temp file for results (file-based communication avoids Queue deadlocks)
        result_path = ""
        try:
            with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as tmp_file:
                result_path = tmp_file.name

            # Use spawn to ensure clean process with separate CUDA context
            ctx = mp.get_context('spawn')
            process = ctx.Process(
                target=_sam3_segment_worker,
                args=(image_paths, prompt, self.device, result_path)
            )
            process.start()
            process.join()  # Wait for worker to finish

            print(f"  [Main] SAM3 subprocess finished with exit code {process.exitcode}")

            # Load results from file
            if os.path.exists(result_path):
                data = np.load(result_path, allow_pickle=True)
                if data.get('success'):
                    masks_bytes = data['masks_bytes']
                    masks = pickle.loads(masks_bytes)
                    print(f"  [Main] Loaded {len(masks)} masks from subprocess")
                    return masks
                else:
                    error_msg = data.get('error_message', 'Unknown error')
                    raise RuntimeError(f"SAM3 subprocess failed: {error_msg}")
            else:
                raise RuntimeError(f"SAM3 subprocess crashed - no result file found")

        finally:
            # Clean up temp file
            if result_path and os.path.exists(result_path):
                try:
                    os.remove(result_path)
                except Exception:
                    pass

    def get_depth_and_intrinsics(
        self,
        image_paths: List[str],
        original_sizes: Optional[List[Tuple[int, int]]] = None
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Get metric depth maps and camera intrinsics using DepthAnything3.

        IMPORTANT: Images are processed TOGETHER as related views to get
        consistent depth and intrinsics predictions. This is critical for
        multi-view pose estimation.

        To handle different image sizes, we pad all images to the same
        (max) dimensions before DA3, then unpad afterward.

        Args:
            image_paths: List of image file paths
            original_sizes: Optional list of (W, H) tuples for the original
                           image dimensions

        Returns:
            depths: List of (H, W) depth maps in meters (at original sizes)
            intrinsics: List of (3, 3) camera intrinsic matrices (for original sizes)
        """
        self._load_depth_model()
        import cv2
        import tempfile
        import os

        # Load images and find max dimensions
        images = []
        sizes = []  # (W, H) for each image
        for path in image_paths:
            img = cv2.imread(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images.append(img)
            sizes.append((img.shape[1], img.shape[0]))  # (W, H)

        max_w = max(s[0] for s in sizes)
        max_h = max(s[1] for s in sizes)

        # Check if padding is needed
        needs_padding = any(s[0] != max_w or s[1] != max_h for s in sizes)

        if needs_padding:
            print(f"  Padding images to common size ({max_w}x{max_h}) for consistent DA3 processing...")
            padded_paths = []
            pad_offsets = []  # Store (pad_left, pad_top) for each image

            for i, (img, (w, h)) in enumerate(zip(images, sizes)):
                # Center-pad to max dimensions
                pad_left = (max_w - w) // 2
                pad_top = (max_h - h) // 2
                pad_right = max_w - w - pad_left
                pad_bottom = max_h - h - pad_top

                padded = cv2.copyMakeBorder(
                    img, pad_top, pad_bottom, pad_left, pad_right,
                    cv2.BORDER_CONSTANT, value=(0, 0, 0)
                )

                # Save padded image to temp file
                fd, temp_path = tempfile.mkstemp(suffix='.png')
                os.close(fd)
                cv2.imwrite(temp_path, cv2.cvtColor(padded, cv2.COLOR_RGB2BGR))
                padded_paths.append(temp_path)
                pad_offsets.append((pad_left, pad_top))

            # Process padded images together
            print(f"  Running DepthAnything3 on {len(padded_paths)} images together (multi-view mode)...")
            prediction = self._depth_model.inference(padded_paths)

            # Clean up temp files
            for path in padded_paths:
                os.unlink(path)
        else:
            # No padding needed, process directly
            print(f"  Running DepthAnything3 on {len(image_paths)} images together (multi-view mode)...")
            prediction = self._depth_model.inference(image_paths)
            pad_offsets = [(0, 0)] * len(image_paths)

        depths = []
        intrinsics_list = []

        for i in range(len(image_paths)):
            # Extract depth - handle both tensor and numpy outputs
            depth = prediction.depth[i]
            if hasattr(depth, 'cpu'):
                depth = depth.cpu().numpy()
            depth = depth.astype(np.float32)

            # Extract intrinsics - handle both tensor and numpy outputs
            K = prediction.intrinsics[i]
            if hasattr(K, 'cpu'):
                K = K.cpu().numpy()
            K = K.astype(np.float32)

            # Get dimensions
            depth_h, depth_w = depth.shape
            orig_w, orig_h = sizes[i]
            pad_left, pad_top = pad_offsets[i]

            print(f"    Image {i}: DA3 output size=({depth_w}x{depth_h}), original=({orig_w}x{orig_h}), DA3 fx={K[0,0]:.1f}")

            if needs_padding:
                # Calculate where the original image is in the padded depth map
                # DA3 may have resized, so we need to compute the scale
                scale_x = depth_w / max_w
                scale_y = depth_h / max_h

                # Compute crop region in depth map coordinates
                crop_left = int(pad_left * scale_x)
                crop_top = int(pad_top * scale_y)
                crop_w = int(orig_w * scale_x)
                crop_h = int(orig_h * scale_y)

                # Crop out the original image region from depth
                depth = depth[crop_top:crop_top+crop_h, crop_left:crop_left+crop_w]

                # Adjust intrinsics for the crop (shift principal point)
                K[0, 2] -= crop_left  # cx
                K[1, 2] -= crop_top   # cy

            # Resize depth to original size if needed
            depth_h, depth_w = depth.shape
            if (depth_w, depth_h) != (orig_w, orig_h):
                scale_x = orig_w / depth_w
                scale_y = orig_h / depth_h

                print(f"    Image {i}: Resizing depth ({depth_w}x{depth_h}) -> ({orig_w}x{orig_h}), scale=({scale_x:.3f}, {scale_y:.3f})")

                depth = cv2.resize(depth, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

                # Adjust intrinsics for resize
                K[0, 0] *= scale_x  # fx
                K[1, 1] *= scale_y  # fy
                K[0, 2] *= scale_x  # cx
                K[1, 2] *= scale_y  # cy

                print(f"    Image {i}: After resize adjustment, fx={K[0,0]:.1f}")
            else:
                print(f"    Image {i}: No resize needed, depth matches original size")

            depths.append(depth)
            intrinsics_list.append(K)

        # Log intrinsics consistency check
        if len(intrinsics_list) >= 2:
            fx_ratio = intrinsics_list[0][0,0] / intrinsics_list[1][0,0] if intrinsics_list[1][0,0] > 0 else float('inf')
            print(f"  Intrinsics consistency: fx_anchor={intrinsics_list[0][0,0]:.1f}, fx_query={intrinsics_list[1][0,0]:.1f}, ratio={fx_ratio:.2f}")

        return depths, intrinsics_list

    def get_semantic_labels(
        self,
        category: str,
        concepts: Optional[List[str]] = None,
        num_labels: int = 15
    ) -> List[str]:
        """
        Get semantic part labels for the object category.

        Args:
            category: Object category (e.g., 'bottle', 'cup')
            concepts: Optional list of manual concepts (overrides Partonomy)
            num_labels: Number of labels to generate if using Partonomy

        Returns:
            List of semantic part labels
        """
        if concepts is not None:
            print(f"  Using {len(concepts)} user-specified concepts")
            return concepts

        print(f"  Generating semantic labels for '{category}' via Partonomy...")
        from concept_pose.utils.label_utils import get_labels_for_category

        labels = get_labels_for_category(
            category=category,
            num_labels=num_labels
        )

        print(f"  Generated {len(labels)} labels: {labels[:5]}...")
        return labels

    def estimate(
        self,
        anchor_image: Union[str, Path, Image.Image],
        query_image: Union[str, Path, Image.Image],
        category: str,
        concepts: Optional[List[str]] = None,
        num_labels: int = 15,
        mask_prompt: Optional[str] = None,
        voxel_resolution: int = 64,
        ransac_iterations: int = 50000,
        use_icp: bool = True,
        visualize: bool = False,
        output_dir: Optional[str] = None
    ) -> Dict:
        """
        Estimate relative pose between anchor and query images.

        This is the main entry point for in-the-wild pose estimation.

        Args:
            anchor_image: Path or PIL Image of anchor (reference) frame
            query_image: Path or PIL Image of query (target) frame
            category: Object category name (e.g., 'bottle', 'car')
            concepts: Optional list of semantic concepts (overrides Partonomy)
            num_labels: Number of labels to generate if using Partonomy
            mask_prompt: Custom prompt for SAM3 (defaults to category)
            voxel_resolution: Voxel grid resolution
            ransac_iterations: Number of RANSAC iterations
            use_icp: Whether to use ICP refinement
            visualize: Whether to generate visualizations
            output_dir: Directory to save visualizations (if visualize=True)

        Returns:
            Dictionary containing:
                - 'success': bool
                - 'R': (3, 3) relative rotation matrix
                - 't': (3,) relative translation vector
                - 'scale': estimated scale factor
                - 'num_correspondences': number of matched points
                - 'num_inliers': number of RANSAC inliers
                All data below is preprocessed (padded to 384x384 square):
                - 'anchor_rgb': (384, 384, 3) preprocessed anchor RGB
                - 'query_rgb': (384, 384, 3) preprocessed query RGB
                - 'anchor_depth': (384, 384) anchor depth map in meters
                - 'query_depth': (384, 384) query depth map in meters
                - 'anchor_mask': (384, 384) anchor mask in [0, 1]
                - 'query_mask': (384, 384) query mask in [0, 1]
                - 'anchor_intrinsics': (3, 3) adjusted anchor camera matrix
                - 'query_intrinsics': (3, 3) adjusted query camera matrix
                - 'anchor_coords': preprocessing coords for inverse mapping
                - 'query_coords': preprocessing coords for inverse mapping
                - 'target_size': preprocessing target size (384)
                - 'semantic_labels': list of semantic labels used
        """
        # Ensure paths
        anchor_path = self._ensure_path(anchor_image)
        query_path = self._ensure_path(query_image)

        print(f"\n{'='*60}")
        print("Wild Pose Estimation")
        print(f"{'='*60}")
        print(f"Anchor: {anchor_path}")
        print(f"Query: {query_path}")
        print(f"Category: {category}")

        # Load images
        anchor_pil = Image.open(anchor_path).convert('RGB')
        query_pil = Image.open(query_path).convert('RGB')

        # Normalize image sizes - resize larger image to match smaller one's resolution
        # This ensures consistent intrinsics scaling during preprocessing
        anchor_size = anchor_pil.size  # (W, H)
        query_size = query_pil.size
        anchor_pixels = anchor_size[0] * anchor_size[1]
        query_pixels = query_size[0] * query_size[1]

        if abs(anchor_pixels - query_pixels) / max(anchor_pixels, query_pixels) > 0.5:
            # Images differ by more than 50% in pixel count - resize to match
            if anchor_pixels < query_pixels:
                # Resize query to match anchor's resolution
                scale = (anchor_pixels / query_pixels) ** 0.5
                new_w = int(query_size[0] * scale)
                new_h = int(query_size[1] * scale)
                print(f"  Resizing query image ({query_size[0]}x{query_size[1]}) -> ({new_w}x{new_h}) to match anchor resolution")
                query_pil = query_pil.resize((new_w, new_h), Image.LANCZOS)
                # Save resized query to temp file for DA3
                import tempfile
                fd, query_path = tempfile.mkstemp(suffix='.jpg')
                import os
                os.close(fd)
                query_pil.save(query_path)
            else:
                # Resize anchor to match query's resolution
                scale = (query_pixels / anchor_pixels) ** 0.5
                new_w = int(anchor_size[0] * scale)
                new_h = int(anchor_size[1] * scale)
                print(f"  Resizing anchor image ({anchor_size[0]}x{anchor_size[1]}) -> ({new_w}x{new_h}) to match query resolution")
                anchor_pil = anchor_pil.resize((new_w, new_h), Image.LANCZOS)
                # Save resized anchor to temp file for DA3
                import tempfile
                fd, anchor_path = tempfile.mkstemp(suffix='.jpg')
                import os
                os.close(fd)
                anchor_pil.save(anchor_path)

        # Step 1: Get depth and intrinsics (processed together for multi-view consistency)
        print(f"\n[1/5] Predicting depth and intrinsics...")
        original_sizes = [anchor_pil.size, query_pil.size]  # (W, H) tuples
        depths, intrinsics = self.get_depth_and_intrinsics(
            [anchor_path, query_path],
            original_sizes=original_sizes
        )
        anchor_depth, query_depth = depths
        anchor_K, query_K = intrinsics

        print(f"  Anchor depth range: [{anchor_depth.min():.3f}, {anchor_depth.max():.3f}] m")
        print(f"  Query depth range: [{query_depth.min():.3f}, {query_depth.max():.3f}] m")

        # Unload depth model to free memory for SAM
        self._unload_depth_model()

        # Step 2: Get object masks (using subprocess to ensure VRAM is released)
        print(f"\n[2/5] Segmenting objects with SAM3 (subprocess)...")
        seg_prompt = mask_prompt or category

        # Run SAM3 in subprocess - this ensures VRAM is fully released when done
        masks = self.get_object_masks_subprocess([anchor_path, query_path], seg_prompt)
        anchor_mask, query_mask = masks

        print(f"  Anchor mask coverage: {anchor_mask.mean()*100:.1f}%")
        print(f"  Query mask coverage: {query_mask.mean()*100:.1f}%")
        print("  SAM3 subprocess exited, VRAM released")

        # Step 3: Get semantic labels
        print(f"\n[3/5] Loading semantic labels...")
        labels = self.get_semantic_labels(category, concepts, num_labels)

        # Step 4: Preprocess all inputs (resize, pad, adjust intrinsics)
        print(f"\n[4/5] Preprocessing inputs (padding to {DEFAULT_TARGET_SIZE}x{DEFAULT_TARGET_SIZE})...")

        # First ensure depth/mask match original image dimensions
        anchor_rgb_orig = np.array(anchor_pil).astype(np.float32) / 255.0
        query_rgb_orig = np.array(query_pil).astype(np.float32) / 255.0
        anchor_depth = self._resize_to_match(anchor_depth, anchor_rgb_orig.shape[:2])
        query_depth = self._resize_to_match(query_depth, query_rgb_orig.shape[:2])
        anchor_mask = self._resize_to_match(anchor_mask, anchor_rgb_orig.shape[:2])
        query_mask = self._resize_to_match(query_mask, query_rgb_orig.shape[:2])

        # Apply proper preprocessing (same as test_oneshot.py)
        anchor_rgb, anchor_mask, anchor_depth, anchor_K, anchor_coords = \
            self._preprocess_for_estimation(anchor_pil, anchor_mask, anchor_depth, anchor_K)
        query_rgb, query_mask, query_depth, query_K, query_coords = \
            self._preprocess_for_estimation(query_pil, query_mask, query_depth, query_K)

        print(f"  Preprocessed image size: {anchor_rgb.shape[:2]}")
        print(f"  Anchor intrinsics (fx, fy, cx, cy): ({anchor_K[0,0]:.1f}, {anchor_K[1,1]:.1f}, {anchor_K[0,2]:.1f}, {anchor_K[1,2]:.1f})")
        print(f"  Query intrinsics (fx, fy, cx, cy): ({query_K[0,0]:.1f}, {query_K[1,1]:.1f}, {query_K[0,2]:.1f}, {query_K[1,2]:.1f})")

        # Step 5: Run pose estimation
        print(f"\n[5/5] Running pose estimation...")

        # Import and create estimator
        from concept_pose.pose.one_shot_estimator import OneShotPoseEstimator

        estimator = OneShotPoseEstimator(
            voxel_resolution=voxel_resolution,
            ransac_iterations=ransac_iterations,
            use_icp=use_icp,
            device=self.device,
            # Settings for in-the-wild estimation
            max_correspondences=10000,      # Was 500 default, eval uses 10000
            similarity_threshold=-2,        # No filtering
            voxelize_anchor=False,          # Dense point cloud matching
            voxelize_query=False,           # Dense point cloud matching
            estimate_scale=False            # DA3 provides metric depth
        )

        # Create dummy identity pose for anchor (relative mode)
        # In relative mode, we compute the pose of query relative to anchor
        anchor_pose = {'R': np.eye(3), 't': np.zeros(3)}

        # Build reference model
        viz_path = None
        if visualize and output_dir:
            viz_path = str(Path(output_dir) / 'anchor_building.png')

        try:
            estimator.build_reference_model(
                ref_rgb=anchor_rgb,
                ref_mask=anchor_mask,
                ref_depth=anchor_depth,
                ref_pose=anchor_pose,
                K_ref=anchor_K,
                semantic_labels=labels,
                visualize_building=visualize,
                viz_save_path=viz_path,
                anchor_pose_mode='relative'  # Key: no GT pose
            )

            # Estimate pose (always request debug info for visualization)
            query_viz_path = None
            if visualize and output_dir:
                query_viz_path = str(Path(output_dir) / 'query_estimation.png')

            R_est, t_est, info = estimator.estimate_pose(
                est_rgb=query_rgb,
                est_mask=query_mask,
                est_depth=query_depth,
                K_est=query_K,
                visualize_query=visualize,
                viz_save_path=query_viz_path,
                return_debug_info=False  # Don't compute expensive debug KL scores (causes OOM)
            )

            # Cleanup
            estimator.cleanup()

            # Prepare result
            result = {
                'success': info.get('success', False),
                'R': R_est,
                't': t_est,
                'scale': info.get('scale'),
                'num_correspondences': info.get('num_correspondences', 0),
                'num_inliers': info.get('num_inliers', 0),
                # Preprocessed data (padded to square)
                'anchor_rgb': anchor_rgb,
                'query_rgb': query_rgb,
                'anchor_depth': anchor_depth,
                'query_depth': query_depth,
                'anchor_mask': anchor_mask,
                'query_mask': query_mask,
                'anchor_intrinsics': anchor_K,
                'query_intrinsics': query_K,
                # Preprocessing coords for inverse mapping
                'anchor_coords': anchor_coords,
                'query_coords': query_coords,
                'target_size': DEFAULT_TARGET_SIZE,
                'semantic_labels': labels
            }

            if result['success']:
                print(f"\n{'='*60}")
                print("Pose Estimation Successful!")
                print(f"{'='*60}")
                print(f"Translation: {t_est}")
                print(f"Correspondences: {info['num_correspondences']}")
                print(f"Inliers: {info['num_inliers']}")
            else:
                print(f"\n{'='*60}")
                print("Pose Estimation Failed")
                print(f"{'='*60}")
                print(f"Error: {info.get('error', 'Unknown')}")

            # Generate correspondence visualization if output_dir provided
            if output_dir and info.get('debug_info') is not None:
                self._save_correspondence_visualization(
                    anchor_rgb=anchor_rgb,
                    query_rgb=query_rgb,
                    anchor_mask=anchor_mask,
                    query_mask=query_mask,
                    anchor_K=anchor_K,
                    debug_info=info['debug_info'],
                    labels=labels,
                    output_dir=output_dir
                )

            # Generate pose projection visualization (anchor -> query)
            if output_dir and result['success']:
                self._save_pose_projection_visualization(
                    anchor_rgb=anchor_rgb,
                    query_rgb=query_rgb,
                    anchor_depth=anchor_depth,
                    anchor_mask=anchor_mask,
                    anchor_K=anchor_K,
                    query_K=query_K,
                    R=R_est,
                    t=t_est,
                    output_dir=output_dir
                )

            return result

        except Exception as e:
            estimator.cleanup()
            raise

    def _ensure_path(self, image: Union[str, Path, Image.Image]) -> str:
        """Ensure we have a file path (save PIL to temp if needed)."""
        if isinstance(image, (str, Path)):
            return str(image)
        elif isinstance(image, Image.Image):
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
                image.save(f.name)
                return f.name
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")

    def _resize_to_match(
        self,
        array: np.ndarray,
        target_shape: Tuple[int, int]
    ) -> np.ndarray:
        """Resize 2D array to match target shape."""
        if array.shape[:2] == target_shape:
            return array

        from PIL import Image as PILImage

        pil = PILImage.fromarray(array)
        pil_resized = pil.resize((target_shape[1], target_shape[0]), PILImage.Resampling.BILINEAR)
        return np.array(pil_resized)

    def _preprocess_for_estimation(
        self,
        pil_image: Image.Image,
        mask: np.ndarray,
        depth: np.ndarray,
        K: np.ndarray,
        target_size: int = DEFAULT_TARGET_SIZE
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Tuple]:
        """
        Preprocess image, mask, depth, and intrinsics for pose estimation.

        Applies the same padding/cropping mechanism as test_oneshot.py:
        - Resize and pad image to square target_size preserving aspect ratio
        - Apply same transformation to mask and depth
        - Adjust camera intrinsics to account for resize/padding

        Args:
            pil_image: PIL Image (RGB)
            mask: (H, W) binary mask
            depth: (H, W) depth map in meters
            K: (3, 3) camera intrinsics
            target_size: Target square size (default 384)

        Returns:
            rgb: (H, W, 3) preprocessed RGB in [0, 1]
            mask: (H, W) preprocessed mask in [0, 1]
            depth: (H, W) preprocessed depth in meters
            K_adjusted: (3, 3) adjusted intrinsics
            coords: Preprocessing coordinates for inverse mapping
        """
        from concept_pose.data.preprocessing import (
            resize_and_pad_image,
            resize_and_pad_mask,
        )

        # 1. Preprocess RGB image
        img_tensor, coords = resize_and_pad_image(pil_image, target_size)
        rgb = img_tensor.permute(1, 2, 0).numpy()  # (3, H, W) -> (H, W, 3)

        # 2. Preprocess mask
        # Convert mask to uint8 [0, 255] for preprocessing
        mask_uint8 = (mask * 255).astype(np.uint8)
        mask_tensor = resize_and_pad_mask(mask_uint8, target_size, coords)
        mask_preprocessed = mask_tensor.numpy()  # (H, W) in [0, 1]

        # 3. Preprocess depth
        # Note: DA3 outputs depth in meters, so we need depth_scale=1.0
        # We'll manually resize/pad to preserve float values
        paste_x, paste_y, paste_x_end, paste_y_end, orig_w, orig_h = coords
        paste_x, paste_y = int(paste_x), int(paste_y)
        paste_x_end, paste_y_end = int(paste_x_end), int(paste_y_end)
        orig_w, orig_h = int(orig_w), int(orig_h)

        # Calculate resize dimensions (same as image)
        aspect = orig_w / orig_h
        if aspect > 1:
            new_w = target_size
            new_h = int(target_size / aspect)
        else:
            new_h = target_size
            new_w = int(target_size * aspect)

        # Resize depth using nearest neighbor to preserve depth values
        import cv2
        depth_resized = cv2.resize(depth, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

        # Pad depth
        depth_preprocessed = np.zeros((target_size, target_size), dtype=np.float32)
        depth_preprocessed[paste_y:paste_y_end, paste_x:paste_x_end] = depth_resized

        # 4. Adjust camera intrinsics
        # Original: pixel = K @ point / z
        # After resize+pad: pixel_new = pixel_old * scale + offset
        # So: fx_new = fx * scale, cx_new = cx * scale + offset_x
        scale = new_w / orig_w  # Same for both x and y (aspect ratio preserved)

        K_adjusted = K.copy()
        K_adjusted[0, 0] = K[0, 0] * scale  # fx
        K_adjusted[1, 1] = K[1, 1] * scale  # fy
        K_adjusted[0, 2] = K[0, 2] * scale + paste_x  # cx
        K_adjusted[1, 2] = K[1, 2] * scale + paste_y  # cy

        return rgb, mask_preprocessed, depth_preprocessed, K_adjusted, coords

    def _save_correspondence_visualization(
        self,
        anchor_rgb: np.ndarray,
        query_rgb: np.ndarray,
        anchor_mask: np.ndarray,
        query_mask: np.ndarray,
        anchor_K: np.ndarray,
        debug_info: Dict,
        labels: List[str],
        output_dir: str
    ):
        """
        Save correspondence visualization showing before/after RANSAC.

        Creates a side-by-side figure with:
        - Left: All correspondences (before RANSAC)
        - Right: Inliers only (after RANSAC)

        Also saves debug data as .npz for use with visualize_oneshot.py.
        """
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from sklearn.decomposition import PCA

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Extract correspondence data
        correspondences = debug_info.get('correspondences', {})
        if correspondences.get('observed_idx') is None:
            print("  No correspondence data available for visualization")
            return

        pixel_coords_query = debug_info.get('pixel_coords_query')
        inlier_mask = debug_info.get('inlier_mask', np.array([]))

        if pixel_coords_query is None or len(pixel_coords_query) == 0:
            print("  No pixel coordinates available for visualization")
            return

        # Get correspondence indices and pixel coords
        corr_obs_idx = correspondences['observed_idx']
        corr_px_query = pixel_coords_query[corr_obs_idx]  # (K, 2) [x, y]

        # Project model 3D points to reference 2D
        model_3d_points = correspondences.get('model_3d')
        if model_3d_points is None or len(model_3d_points) == 0:
            print("  No model 3D points available for visualization")
            return

        # Project: pixel = K @ point_3d / z
        points_homo = (anchor_K @ model_3d_points.T).T  # (K, 3)
        corr_px_ref = points_homo[:, :2] / points_homo[:, 2:3]  # (K, 2) [x, y]

        # Get saliency for PCA coloring
        saliency_ref = debug_info.get('saliency_ref')
        saliency_query = debug_info.get('saliency_query')

        # Compute PCA colors for correspondences
        if saliency_ref is not None and correspondences.get('model_saliency') is not None:
            model_sal = correspondences['model_saliency']  # (K, C)
            obs_sal = correspondences['observed_saliency']  # (K, C)

            # Apply PCA to get RGB colors (handle cases with < 3 concepts)
            all_sal = np.vstack([model_sal, obs_sal])
            n_concepts = all_sal.shape[1]

            if n_concepts >= 3:
                pca = PCA(n_components=3)
                pca_colors = pca.fit_transform(all_sal)
            elif n_concepts == 2:
                pca_colors = np.zeros((len(all_sal), 3))
                pca_colors[:, 0] = all_sal[:, 0]
                pca_colors[:, 1] = all_sal[:, 1]
            else:
                # 1 concept: grayscale
                pca_colors = np.tile(all_sal, (1, 3))

            # Normalize to [0, 1]
            pca_min = pca_colors.min(axis=0)
            pca_max = pca_colors.max(axis=0)
            pca_colors = (pca_colors - pca_min) / (pca_max - pca_min + 1e-8)

            colors_ref = pca_colors[:len(model_sal)]
            colors_query = pca_colors[len(model_sal):]
        else:
            # Default to red-green based on inlier status
            colors_ref = np.zeros((len(corr_px_ref), 3))
            colors_query = np.zeros((len(corr_px_query), 3))
            colors_ref[:] = [1, 0, 0]  # Red
            colors_query[:] = [1, 0, 0]

        # Create figure with 2x2 layout
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # Top row: All correspondences (before RANSAC)
        # Bottom row: Inliers only (after RANSAC)

        def draw_correspondences(ax_ref, ax_query, px_ref, px_query, colors, title_suffix=""):
            """Draw correspondences on anchor and query images."""
            # Show anchor
            ax_ref.imshow(anchor_rgb)
            ax_ref.scatter(px_ref[:, 0], px_ref[:, 1], c=colors, s=20, alpha=0.7, edgecolors='white', linewidths=0.5)
            ax_ref.set_title(f'Anchor {title_suffix}', fontsize=14, fontweight='bold')
            ax_ref.axis('off')

            # Show query
            ax_query.imshow(query_rgb)
            ax_query.scatter(px_query[:, 0], px_query[:, 1], c=colors, s=20, alpha=0.7, edgecolors='white', linewidths=0.5)
            ax_query.set_title(f'Query {title_suffix}', fontsize=14, fontweight='bold')
            ax_query.axis('off')

        # All correspondences
        n_all = len(corr_px_ref)
        draw_correspondences(
            axes[0, 0], axes[0, 1],
            corr_px_ref, corr_px_query, colors_ref,
            f"- All Correspondences ({n_all})"
        )

        # Inliers only
        if len(inlier_mask) > 0 and inlier_mask.sum() > 0:
            inlier_px_ref = corr_px_ref[inlier_mask]
            inlier_px_query = corr_px_query[inlier_mask]
            inlier_colors = colors_ref[inlier_mask]
            n_inliers = inlier_mask.sum()

            draw_correspondences(
                axes[1, 0], axes[1, 1],
                inlier_px_ref, inlier_px_query, inlier_colors,
                f"- Inliers After RANSAC ({n_inliers}/{n_all})"
            )
        else:
            # No inliers - show empty images with message
            axes[1, 0].imshow(anchor_rgb)
            axes[1, 0].set_title('Anchor - No Inliers', fontsize=14, fontweight='bold')
            axes[1, 0].axis('off')

            axes[1, 1].imshow(query_rgb)
            axes[1, 1].set_title('Query - No Inliers', fontsize=14, fontweight='bold')
            axes[1, 1].axis('off')

        plt.suptitle('Correspondence Visualization: Before vs After RANSAC',
                     fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()

        # Save figure
        viz_path = output_path / 'correspondences.png'
        plt.savefig(viz_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print(f"  Saved correspondence visualization: {viz_path}")

        # Also save debug data as .npz for interactive visualization
        self._save_debug_npz(
            anchor_rgb=anchor_rgb,
            query_rgb=query_rgb,
            anchor_mask=anchor_mask,
            query_mask=query_mask,
            anchor_K=anchor_K,
            debug_info=debug_info,
            corr_px_ref=corr_px_ref,
            labels=labels,
            output_dir=output_dir
        )

    def _save_debug_npz(
        self,
        anchor_rgb: np.ndarray,
        query_rgb: np.ndarray,
        anchor_mask: np.ndarray,
        query_mask: np.ndarray,
        anchor_K: np.ndarray,
        debug_info: Dict,
        corr_px_ref: np.ndarray,
        labels: List[str],
        output_dir: str
    ):
        """Save debug data as .npz for use with visualize_oneshot.py."""
        output_path = Path(output_dir)

        correspondences = debug_info.get('correspondences', {})

        # Compute bounding boxes from masks
        def get_bbox_from_mask(mask):
            if mask is None:
                return None
            y_coords, x_coords = np.where(mask > 0.5)
            if len(x_coords) == 0:
                return None
            return np.array([int(x_coords.min()), int(y_coords.min()),
                           int(x_coords.max()), int(y_coords.max())])

        bbox_ref = get_bbox_from_mask(anchor_mask)
        bbox_query = get_bbox_from_mask(query_mask)

        # Prepare data dict
        save_dict = {
            # Metadata
            'pair_0000_object_name': 'wild_image',
            'pair_0000_category': 'unknown',

            # Images (convert to uint8)
            'pair_0000_rgb_ref': (anchor_rgb * 255).astype(np.uint8),
            'pair_0000_rgb_query': (query_rgb * 255).astype(np.uint8),
            'pair_0000_mask_ref': anchor_mask,
            'pair_0000_mask_query': query_mask,
            'pair_0000_bbox_ref': bbox_ref,
            'pair_0000_bbox_query': bbox_query,

            # Saliency
            'pair_0000_saliency_ref': debug_info.get('saliency_ref'),
            'pair_0000_saliency_query': debug_info.get('saliency_query'),
            'pair_0000_labels': np.array(labels),

            # Pixel coordinates
            'pair_0000_pixel_coords_query': debug_info.get('pixel_coords_query'),

            # Correspondences
            'pair_0000_corr_observed_idx': correspondences.get('observed_idx'),
            'pair_0000_corr_observed_3d': correspondences.get('observed_3d'),
            'pair_0000_corr_model_3d': correspondences.get('model_3d'),
            'pair_0000_corr_kl_divergence': correspondences.get('kl_divergence'),
            'pair_0000_corr_cosine_similarity': correspondences.get('cosine_similarity'),
            'pair_0000_corr_observed_saliency': correspondences.get('observed_saliency'),
            'pair_0000_corr_model_saliency': correspondences.get('model_saliency'),
            'pair_0000_corr_pixel_coords_ref': corr_px_ref,
            'pair_0000_inlier_mask': debug_info.get('inlier_mask'),
        }

        # Filter out None values
        save_dict = {k: v for k, v in save_dict.items() if v is not None}

        # Save
        npz_path = output_path / 'debug_viz.npz'
        np.savez_compressed(npz_path, **save_dict)
        print(f"  Saved debug data: {npz_path}")
        print(f"  (Use 'python scripts/visualize_oneshot.py {output_dir}' for interactive visualization)")

    def _save_pose_projection_visualization(
        self,
        anchor_rgb: np.ndarray,
        query_rgb: np.ndarray,
        anchor_depth: np.ndarray,
        anchor_mask: np.ndarray,
        anchor_K: np.ndarray,
        query_K: np.ndarray,
        R: np.ndarray,
        t: np.ndarray,
        output_dir: str,
        subsample: int = 10
    ):
        """
        Visualize pose estimation by projecting anchor point cloud to query frame.

        This provides a qualitative evaluation of the predicted pose:
        - Back-project anchor depth to 3D points
        - Transform points using predicted R, t
        - Project onto query image using query intrinsics
        - If pose is correct, projected points should align with object in query

        Args:
            anchor_rgb: (H, W, 3) anchor RGB image (preprocessed, e.g., 384x384)
            query_rgb: (H, W, 3) query RGB image (preprocessed, e.g., 384x384)
            anchor_depth: (H, W) anchor depth in meters (preprocessed)
            anchor_mask: (H, W) anchor object mask (preprocessed)
            anchor_K: (3, 3) anchor camera intrinsics (adjusted for preprocessed size)
            query_K: (3, 3) query camera intrinsics (adjusted for preprocessed size)
            R: (3, 3) predicted rotation matrix
            t: (3,) predicted translation vector
            output_dir: Directory to save visualization
            subsample: Subsample factor for points (for cleaner visualization)
        """
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # === VERIFICATION: Check that all sizes are consistent ===
        H_rgb, W_rgb = anchor_rgb.shape[:2]
        H_depth, W_depth = anchor_depth.shape
        H_mask, W_mask = anchor_mask.shape
        H_query, W_query = query_rgb.shape[:2]

        print(f"\n  [Pose Visualization] Verifying data consistency:")
        print(f"    Anchor RGB:   {W_rgb}x{H_rgb}")
        print(f"    Anchor Depth: {W_depth}x{H_depth}")
        print(f"    Anchor Mask:  {W_mask}x{H_mask}")
        print(f"    Query RGB:    {W_query}x{H_query}")
        print(f"    Anchor K: fx={anchor_K[0,0]:.1f}, fy={anchor_K[1,1]:.1f}, cx={anchor_K[0,2]:.1f}, cy={anchor_K[1,2]:.1f}")
        print(f"    Query K:  fx={query_K[0,0]:.1f}, fy={query_K[1,1]:.1f}, cx={query_K[0,2]:.1f}, cy={query_K[1,2]:.1f}")

        # Verify sizes match
        assert H_rgb == H_depth == H_mask, f"Height mismatch: RGB={H_rgb}, Depth={H_depth}, Mask={H_mask}"
        assert W_rgb == W_depth == W_mask, f"Width mismatch: RGB={W_rgb}, Depth={W_depth}, Mask={W_mask}"

        # Verify intrinsics principal point is reasonable (should be near image center)
        cx_expected, cy_expected = W_rgb / 2, H_rgb / 2
        if abs(anchor_K[0, 2] - cx_expected) > W_rgb * 0.3 or abs(anchor_K[1, 2] - cy_expected) > H_rgb * 0.3:
            print(f"    WARNING: Anchor K principal point ({anchor_K[0,2]:.1f}, {anchor_K[1,2]:.1f}) far from image center ({cx_expected:.1f}, {cy_expected:.1f})")

        # 1. Back-project anchor depth to 3D points
        H, W = anchor_depth.shape
        v, u = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')

        # Get masked points
        mask_bool = anchor_mask > 0.5
        valid = mask_bool & (anchor_depth > 0)

        u_valid = u[valid]
        v_valid = v[valid]
        z_valid = anchor_depth[valid]

        # Back-project: X = (u - cx) * Z / fx, Y = (v - cy) * Z / fy
        fx_a, fy_a = anchor_K[0, 0], anchor_K[1, 1]
        cx_a, cy_a = anchor_K[0, 2], anchor_K[1, 2]

        X = (u_valid - cx_a) * z_valid / fx_a
        Y = (v_valid - cy_a) * z_valid / fy_a
        Z = z_valid

        points_anchor = np.stack([X, Y, Z], axis=1)  # (N, 3)

        # Get RGB colors for the points
        anchor_rgb_uint8 = (anchor_rgb * 255).astype(np.uint8) if anchor_rgb.max() <= 1 else anchor_rgb.astype(np.uint8)
        colors = anchor_rgb_uint8[v_valid, u_valid] / 255.0  # (N, 3) normalized

        # Subsample for cleaner visualization
        if subsample > 1:
            indices = np.arange(0, len(points_anchor), subsample)
            points_anchor = points_anchor[indices]
            colors = colors[indices]

        print(f"  Projecting {len(points_anchor)} anchor points to query frame...")

        # 2. Transform points: P_query = R @ P_anchor + t
        points_query = (R @ points_anchor.T).T + t  # (N, 3)

        # 3. Project to query image: pixel = K @ P / Z
        points_query_homo = (query_K @ points_query.T).T  # (N, 3)
        z_proj = points_query_homo[:, 2]

        # Filter points behind camera
        in_front = z_proj > 0.1
        points_query_homo = points_query_homo[in_front]
        colors_valid = colors[in_front]
        z_proj = z_proj[in_front]

        # Project to 2D
        pixels_query = points_query_homo[:, :2] / z_proj[:, np.newaxis]  # (N, 2)

        # Filter points outside image
        H_q, W_q = query_rgb.shape[:2]
        in_bounds = (
            (pixels_query[:, 0] >= 0) & (pixels_query[:, 0] < W_q) &
            (pixels_query[:, 1] >= 0) & (pixels_query[:, 1] < H_q)
        )
        pixels_query = pixels_query[in_bounds]
        colors_valid = colors_valid[in_bounds]
        z_proj = z_proj[in_bounds]

        print(f"  {len(pixels_query)} points visible in query frame")

        # 4. Create visualization
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # Left: Anchor image with mask overlay
        axes[0].imshow(anchor_rgb)
        axes[0].imshow(anchor_mask, alpha=0.3, cmap='Reds')
        axes[0].set_title('Anchor Image (with mask)', fontsize=14, fontweight='bold')
        axes[0].axis('off')

        # Middle: Query image with projected points (colored by depth)
        axes[1].imshow(query_rgb)
        if len(pixels_query) > 0:
            # Color by depth for better visualization
            scatter = axes[1].scatter(
                pixels_query[:, 0], pixels_query[:, 1],
                c=z_proj, cmap='viridis', s=2, alpha=0.6
            )
            plt.colorbar(scatter, ax=axes[1], label='Depth (m)', shrink=0.8)
        axes[1].set_title(f'Query + Projected Anchor Points (n={len(pixels_query)})',
                         fontsize=14, fontweight='bold')
        axes[1].axis('off')

        # Right: Query image with projected points (colored by original RGB)
        axes[2].imshow(query_rgb)
        if len(pixels_query) > 0:
            axes[2].scatter(
                pixels_query[:, 0], pixels_query[:, 1],
                c=colors_valid, s=2, alpha=0.6
            )
        axes[2].set_title('Query + Projected Points (RGB colors)',
                         fontsize=14, fontweight='bold')
        axes[2].axis('off')

        plt.suptitle('Pose Projection Visualization: Anchor → Query',
                     fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()

        # Save
        viz_path = output_path / 'pose_projection.png'
        plt.savefig(viz_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print(f"  Saved pose projection visualization: {viz_path}")

        # Also save a side-by-side comparison showing anchor->anchor and anchor->query projections
        # First, project anchor points back to anchor image (sanity check)
        points_anchor_full = (R @ np.zeros((1,3)).T).T  # dummy, we need original points

        # Re-backproject without subsampling for the overlay (or use the subsampled)
        # Actually, let's pass the original anchor points to the overlay function
        self._save_pose_overlay_visualization(
            anchor_rgb=anchor_rgb,
            query_rgb=query_rgb,
            points_anchor=points_anchor,  # 3D points in anchor frame
            anchor_K=anchor_K,
            query_K=query_K,
            R=R,
            t=t,
            colors=colors,
            output_dir=output_dir
        )

    def _save_pose_overlay_visualization(
        self,
        anchor_rgb: np.ndarray,
        query_rgb: np.ndarray,
        points_anchor: np.ndarray,
        anchor_K: np.ndarray,
        query_K: np.ndarray,
        R: np.ndarray,
        t: np.ndarray,
        colors: np.ndarray,
        output_dir: str
    ):
        """
        Save side-by-side overlay visualization for qualitative evaluation.

        Shows:
        - Left: Anchor image with anchor 3D points projected back using anchor_K (sanity check)
        - Right: Query image with transformed anchor points projected using query_K
        """
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        output_path = Path(output_dir)

        print(f"\n  [Pose Overlay] Creating side-by-side verification:")
        print(f"    Anchor image: {anchor_rgb.shape[1]}x{anchor_rgb.shape[0]}")
        print(f"    Query image:  {query_rgb.shape[1]}x{query_rgb.shape[0]}")
        print(f"    Using anchor_K for anchor projection: fx={anchor_K[0,0]:.1f}, cx={anchor_K[0,2]:.1f}, cy={anchor_K[1,2]:.1f}")
        print(f"    Using query_K for query projection:   fx={query_K[0,0]:.1f}, cx={query_K[0,2]:.1f}, cy={query_K[1,2]:.1f}")

        # === LEFT PANEL: Project anchor 3D points back to anchor image ===
        # This is a SANITY CHECK: if back-projection was correct, re-projection should
        # give us back the original pixel locations
        # Formula: pixel = K @ point_3d, then pixel_2d = pixel[:2] / pixel[2]
        points_anchor_homo = (anchor_K @ points_anchor.T).T  # (N, 3)
        z_anchor = points_anchor_homo[:, 2]
        valid_anchor = z_anchor > 0.1
        pixels_anchor = points_anchor_homo[valid_anchor, :2] / z_anchor[valid_anchor, np.newaxis]
        colors_anchor = colors[valid_anchor]

        # Filter in-bounds for anchor
        H_a, W_a = anchor_rgb.shape[:2]
        in_bounds_a = (
            (pixels_anchor[:, 0] >= 0) & (pixels_anchor[:, 0] < W_a) &
            (pixels_anchor[:, 1] >= 0) & (pixels_anchor[:, 1] < H_a)
        )
        pixels_anchor = pixels_anchor[in_bounds_a]
        colors_anchor = colors_anchor[in_bounds_a]

        print(f"    Anchor->Anchor: {len(pixels_anchor)} points projected")

        # === RIGHT PANEL: Transform and project to query ===
        # Formula: P_query = R @ P_anchor + t, then pixel = query_K @ P_query
        points_query = (R @ points_anchor.T).T + t  # Transform to query frame
        points_query_homo = (query_K @ points_query.T).T  # Project using query_K
        z_query = points_query_homo[:, 2]
        valid_query = z_query > 0.1
        pixels_query = points_query_homo[valid_query, :2] / z_query[valid_query, np.newaxis]
        colors_query = colors[valid_query]

        # Filter in-bounds for query
        H_q, W_q = query_rgb.shape[:2]
        in_bounds_q = (
            (pixels_query[:, 0] >= 0) & (pixels_query[:, 0] < W_q) &
            (pixels_query[:, 1] >= 0) & (pixels_query[:, 1] < H_q)
        )
        pixels_query = pixels_query[in_bounds_q]
        colors_query = colors_query[in_bounds_q]

        print(f"    Anchor->Query:  {len(pixels_query)} points projected (after transform)")

        # 3. Create side-by-side visualization
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))

        # Left: Anchor with projected anchor points (sanity check) - GREEN for visibility
        axes[0].imshow(anchor_rgb)
        if len(pixels_anchor) > 0:
            axes[0].scatter(
                pixels_anchor[:, 0], pixels_anchor[:, 1],
                c='lime', s=3, alpha=0.7, edgecolors='none'
            )
        axes[0].set_title(f'Anchor: 3D Points → Anchor Image (n={len(pixels_anchor)})\n(Sanity check: GREEN points should align with object)',
                         fontsize=12, fontweight='bold')
        axes[0].axis('off')

        # Right: Query with transformed projected points - RED for visibility
        axes[1].imshow(query_rgb)
        if len(pixels_query) > 0:
            axes[1].scatter(
                pixels_query[:, 0], pixels_query[:, 1],
                c='red', s=3, alpha=0.7, edgecolors='none'
            )
        axes[1].set_title(f'Query: Transformed 3D Points → Query Image (n={len(pixels_query)})\n(Good alignment = correct pose)',
                         fontsize=12, fontweight='bold')
        axes[1].axis('off')

        plt.suptitle('Pose Projection Verification',
                     fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()

        viz_path = output_path / 'pose_overlay.png'
        plt.savefig(viz_path, dpi=200, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print(f"  Saved pose overlay visualization: {viz_path}")

    def cleanup(self):
        """Clean up all loaded models."""
        self._unload_sam_model()
        self._unload_depth_model()
        if self._estimator is not None:
            self._estimator.cleanup()
            self._estimator = None
        torch.cuda.empty_cache()
        gc.collect()


def estimate_relative_pose(
    anchor_image: Union[str, Path, Image.Image],
    query_image: Union[str, Path, Image.Image],
    category: str,
    concepts: Optional[List[str]] = None,
    num_labels: int = 15,
    mask_prompt: Optional[str] = None,
    device: str = 'cuda',
    voxel_resolution: int = 64,
    ransac_iterations: int = 50000,
    use_icp: bool = True,
    visualize: bool = False,
    output_dir: Optional[str] = None
) -> Dict:
    """
    Estimate relative pose between two in-the-wild images.

    This is the main entry point for ConceptPose demo. Given two images of
    the same object from different viewpoints, estimates the relative 6D pose
    transformation from anchor to query frame.

    The pipeline:
    1. SAM3: Segment the object in both images
    2. DepthAnything3: Predict metric depth and camera intrinsics
    3. Partonomy/Gemini: Generate semantic part labels (or use provided concepts)
    4. ConceptPose: Build semantic 3D model from anchor, match to query

    Args:
        anchor_image: Reference frame image (path, Path, or PIL Image)
        query_image: Target frame image (path, Path, or PIL Image)
        category: Object category name (e.g., 'bottle', 'mug', 'car')
        concepts: Optional list of semantic concepts. If None, uses Partonomy
                  to auto-generate labels via Gemini API.
        num_labels: Number of labels to generate if using Partonomy (default: 15)
        mask_prompt: Custom prompt for SAM3 segmentation (defaults to category)
        device: Torch device ('cuda' or 'cpu')
        voxel_resolution: Voxel grid resolution (default: 64)
        ransac_iterations: Number of RANSAC iterations (default: 50000)
        use_icp: Whether to use ICP refinement (default: True)
        visualize: Generate debug visualizations (default: False)
        output_dir: Directory for visualizations (required if visualize=True)

    Returns:
        Dictionary containing:
            - 'success': bool - whether pose estimation succeeded
            - 'R': (3, 3) ndarray - relative rotation matrix (anchor → query)
            - 't': (3,) ndarray - relative translation vector (anchor → query)
            - 'scale': float - estimated scale factor
            - 'num_correspondences': int - number of semantic matches
            - 'num_inliers': int - number of RANSAC inliers
            All data below is preprocessed (padded to 384x384 square):
            - 'anchor_rgb': (384, 384, 3) ndarray - preprocessed anchor RGB
            - 'query_rgb': (384, 384, 3) ndarray - preprocessed query RGB
            - 'anchor_depth': (384, 384) ndarray - anchor depth map in meters
            - 'query_depth': (384, 384) ndarray - query depth map in meters
            - 'anchor_mask': (384, 384) ndarray - anchor object mask in [0, 1]
            - 'query_mask': (384, 384) ndarray - query object mask in [0, 1]
            - 'anchor_intrinsics': (3, 3) ndarray - adjusted anchor camera matrix
            - 'query_intrinsics': (3, 3) ndarray - adjusted query camera matrix
            - 'anchor_coords': tuple - preprocessing coords for inverse mapping
            - 'query_coords': tuple - preprocessing coords for inverse mapping
            - 'target_size': int - preprocessing target size (384)
            - 'semantic_labels': list - semantic labels used

    Example:
        >>> # Basic usage with auto-generated concepts
        >>> result = estimate_relative_pose(
        ...     'view1.jpg', 'view2.jpg', 'bottle'
        ... )
        >>> print(f"Translation: {result['t']}")

        >>> # With custom concepts
        >>> result = estimate_relative_pose(
        ...     'view1.jpg', 'view2.jpg', 'bottle',
        ...     concepts=['neck', 'body', 'cap', 'base']
        ... )

        >>> # With visualization (auto-creates output_dir if not specified)
        >>> result = estimate_relative_pose(
        ...     'view1.jpg', 'view2.jpg', 'mug',
        ...     visualize=True
        ... )
    """
    # Auto-enable output_dir if visualize is True
    if visualize and output_dir is None:
        output_dir = './wild_pose_output'

    estimator = WildPoseEstimator(device=device)

    try:
        result = estimator.estimate(
            anchor_image=anchor_image,
            query_image=query_image,
            category=category,
            concepts=concepts,
            num_labels=num_labels,
            mask_prompt=mask_prompt,
            voxel_resolution=voxel_resolution,
            ransac_iterations=ransac_iterations,
            use_icp=use_icp,
            visualize=visualize,
            output_dir=output_dir
        )
        return result
    finally:
        estimator.cleanup()


# CLI entry point
def main():
    """Command-line interface for wild pose estimation."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Estimate relative pose between two in-the-wild images',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage
    python -m concept_pose.demo.wild_pose_estimator \\
        --anchor view1.jpg --query view2.jpg --category bottle

    # With custom concepts
    python -m concept_pose.demo.wild_pose_estimator \\
        --anchor view1.jpg --query view2.jpg --category bottle \\
        --concepts neck body cap base

    # With visualization
    python -m concept_pose.demo.wild_pose_estimator \\
        --anchor view1.jpg --query view2.jpg --category mug \\
        --visualize --output-dir ./debug_output
"""
    )

    parser.add_argument('--anchor', type=str, required=True,
                        help='Path to anchor (reference) image')
    parser.add_argument('--query', type=str, required=True,
                        help='Path to query (target) image')
    parser.add_argument('--category', type=str, required=True,
                        help='Object category (e.g., bottle, mug, car)')
    parser.add_argument('--concepts', type=str, nargs='+', default=None,
                        help='Custom semantic concepts (overrides Partonomy)')
    parser.add_argument('--num-labels', type=int, default=15,
                        help='Number of labels to generate (default: 15)')
    parser.add_argument('--mask-prompt', type=str, default=None,
                        help='Custom prompt for SAM3 (defaults to category)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Torch device (cuda or cpu)')
    parser.add_argument('--voxel-resolution', type=int, default=64,
                        help='Voxel grid resolution (default: 64)')
    parser.add_argument('--ransac-iterations', type=int, default=100000,
                        help='RANSAC iterations (default: 100000)')
    parser.add_argument('--no-icp', action='store_true',
                        help='Disable ICP refinement')
    parser.add_argument('--visualize', action='store_true',
                        help='Generate debug visualizations (auto-enables output)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Directory for visualizations (default: ./wild_pose_output)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducible RANSAC (default: 42)')

    args = parser.parse_args()

    # Set deterministic mode for reproducible results
    set_deterministic_mode(args.seed)

    # Auto-set output directory if visualize is requested
    if args.visualize and args.output_dir is None:
        args.output_dir = './wild_pose_output'
        print(f"Visualization enabled, output directory: {args.output_dir}")

    # Run estimation
    result = estimate_relative_pose(
        anchor_image=args.anchor,
        query_image=args.query,
        category=args.category,
        concepts=args.concepts,
        num_labels=args.num_labels,
        mask_prompt=args.mask_prompt,
        device=args.device,
        voxel_resolution=args.voxel_resolution,
        ransac_iterations=args.ransac_iterations,
        use_icp=not args.no_icp,
        visualize=args.visualize,
        output_dir=args.output_dir
    )

    # Print results
    if result['success']:
        print("\n" + "="*60)
        print("RESULT: Pose Estimation Successful")
        print("="*60)
        print(f"\nRotation matrix:")
        print(result['R'])
        print(f"\nTranslation vector: {result['t']}")
        print(f"Scale: {result['scale']}")
        print(f"Correspondences: {result['num_correspondences']}")
        print(f"Inliers: {result['num_inliers']}")
    else:
        print("\n" + "="*60)
        print("RESULT: Pose Estimation Failed")
        print("="*60)

    return 0 if result['success'] else 1


if __name__ == '__main__':
    import sys
    sys.exit(main())
