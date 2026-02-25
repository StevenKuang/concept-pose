"""
Base Evaluator for Pose Estimation
====================================

Contains common evaluation logic shared across different testing modes
(one-shot, category-level, etc.)
"""

import os
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
from PIL import Image as PILImage


class BaseEvaluator:
    """
    Base class for pose estimation evaluators.

    Provides common functionality:
    - Atomic file saving for crash-resilient incremental results
    - Summary statistics computation
    - Visualization image combining
    """

    def _save_results_atomic(
        self,
        results: List[Dict],
        output_file: str,
        eval_metadata: Optional[Dict] = None
    ):
        """
        Save results to file atomically using temp file + rename.

        This ensures the results file is never corrupted, even if the process
        is interrupted during writing.

        Args:
            results: List of result dicts
            output_file: Target output file path
            eval_metadata: Optional metadata to include
        """
        import json
        import tempfile

        # Prepare output data
        output_data = eval_metadata.copy() if eval_metadata else {}

        # Filter out debug_data from results (contains numpy arrays not JSON serializable)
        # debug_data is saved separately to .npz files
        results_for_json = [
            {k: v for k, v in result.items() if k != 'debug_data'}
            for result in results
        ]
        output_data['results'] = results_for_json

        # Write to temporary file first
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Use a temp file in the same directory to ensure atomic rename works
        temp_fd, temp_path = tempfile.mkstemp(
            dir=output_path.parent,
            prefix=f'.{output_path.name}.',
            suffix='.tmp'
        )

        try:
            # Write to temp file
            with os.fdopen(temp_fd, 'w') as f:
                json.dump(output_data, f, indent=2)

            # Atomic rename (overwrites existing file)
            os.replace(temp_path, str(output_path))

        except Exception as e:
            # Clean up temp file on error
            try:
                os.unlink(temp_path)
            except:
                pass
            raise e

    @staticmethod
    def get_results_summary(results: List[Dict], total_key: str = 'total_items') -> Dict:
        """
        Compute summary statistics from results.

        Args:
            results: List of result dicts
            total_key: Key name for total count ('total_pairs', 'total_frames', etc.)

        Returns:
            Dict with summary statistics including:
            - Success rates (overall, 5deg/2cm, 5deg/5cm, 10deg/5cm, 10deg/10cm)
            - Mean/median errors (rotation, translation, ADD) - computed only on successful poses
            - Threshold-based recall rates (ADD-10, ADD-S-10, IoU-50, IoU-75) - computed over ALL test cases
        """
        successful = [r for r in results if r.get('success', False)]

        if not successful:
            return {
                total_key: len(results),
                'num_success': 0,
                'success_rate': 0.0,
                'success_rate_5deg2cm': 0.0,
                'success_rate_5deg5cm': 0.0,
                'success_rate_10deg5cm': 0.0,
                'success_rate_10deg10cm': 0.0
            }

        # Count success at different thresholds
        success_5deg2cm = sum(1 for r in successful if r.get('success_5deg2cm', False))
        success_5deg5cm = sum(1 for r in successful if r.get('success_5deg5cm', False))
        success_10deg5cm = sum(1 for r in successful if r.get('success_10deg5cm', False))
        success_10deg10cm = sum(1 for r in successful if r.get('success_10deg10cm', False))

        # Threshold-based success metrics
        add_10_success = sum(1 for r in successful if r.get('add_score', False))
        adds_10_success = sum(1 for r in successful if r.get('adds_score', False))
        adds_adaptive_10_success = sum(1 for r in successful if r.get('adds_adaptive_score', False))
        iou_50_success = sum(1 for r in successful if r.get('iou_3d_50', False))
        iou_75_success = sum(1 for r in successful if r.get('iou_3d_75', False))

        # Extract error metrics
        rotation_errors = [r['rotation_error_deg'] for r in successful]
        translation_errors = [r['translation_error_m'] for r in successful]
        add_errors = [r['add_error'] for r in successful]
        adds_errors = [r.get('adds_error', r['add_error']) for r in successful]  # fallback to add_error
        iou_3d_scores = [r['iou_3d'] for r in successful]

        # Compute AUC metrics following One2Any/BOP methodology
        from concept_pose.pose.pose_metrics import compute_auc_from_errors
        add_auc = compute_auc_from_errors(add_errors, max_val=0.1)
        adds_auc = compute_auc_from_errors(adds_errors, max_val=0.1)

        # Compute pose estimation timing statistics (if available)
        pose_times = [r['pose_estimation_time'] for r in results if 'pose_estimation_time' in r]
        avg_pose_time = float(np.mean(pose_times)) if pose_times else None

        return {
            total_key: len(results),
            'num_success': len(successful),
            'success_rate': len(successful) / len(results),
            'success_rate_5deg2cm': success_5deg2cm / len(results),
            'success_rate_5deg5cm': success_5deg5cm / len(results),
            'success_rate_10deg5cm': success_10deg5cm / len(results),
            'success_rate_10deg10cm': success_10deg10cm / len(results),
            'mean_rotation_error_deg': float(np.mean(rotation_errors)),
            'mean_translation_error_m': float(np.mean(translation_errors)),
            'mean_add_error': float(np.mean(add_errors)),
            'mean_iou_3d': float(np.mean(iou_3d_scores)),
            'median_rotation_error_deg': float(np.median(rotation_errors)),
            'median_translation_error_m': float(np.median(translation_errors)),
            # Threshold-based success rates (recall over ALL test cases, not just successful)
            'add_10_success': add_10_success,
            'add_10_rate': add_10_success / len(results),
            'adds_10_success': adds_10_success,
            'adds_10_rate': adds_10_success / len(results),
            'adds_adaptive_10_success': adds_adaptive_10_success,
            'adds_adaptive_10_rate': adds_adaptive_10_success / len(results),
            'iou_50_success': iou_50_success,
            'iou_50_rate': iou_50_success / len(results),
            'iou_75_success': iou_75_success,
            'iou_75_rate': iou_75_success / len(results),
            # AUC metrics (following One2Any/BOP methodology)
            'add_auc': add_auc,
            'adds_auc': adds_auc,
            # Timing statistics
            'avg_pose_estimation_time': avg_pose_time  # Average time for build + estimate (excludes metrics)
        }

    def _combine_images_vertically(
        self,
        image_paths: List[Path],
        output_path: Path,
        dpi: int = 150,
        quality: int = 95
    ):
        """
        Combine multiple images into a single vertical stack.

        Each image is centered horizontally if widths differ.

        Args:
            image_paths: List of paths to images to combine (top to bottom)
            output_path: Path to save combined image
            dpi: DPI for saving (used by some formats)
            quality: JPEG quality (1-100) if saving as JPEG
        """
        if not image_paths:
            raise ValueError("image_paths cannot be empty")

        # Load all images
        images = [PILImage.open(path) for path in image_paths]

        try:
            # Get dimensions
            widths = [img.size[0] for img in images]
            heights = [img.size[1] for img in images]

            # Calculate combined dimensions
            max_width = max(widths)
            total_height = sum(heights)

            # Create white background
            combined = PILImage.new('RGB', (max_width, total_height), (255, 255, 255))

            # Paste images vertically with horizontal centering
            y_offset = 0
            for img in images:
                x_offset = (max_width - img.size[0]) // 2
                combined.paste(img, (x_offset, y_offset))
                y_offset += img.size[1]

            # Save combined image
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Use dpi parameter for formats that support it (PNG, TIFF)
            if output_path.suffix.lower() in ['.png', '.tiff', '.tif']:
                combined.save(output_path, dpi=(dpi, dpi))
            elif output_path.suffix.lower() in ['.jpg', '.jpeg']:
                combined.save(output_path, quality=quality)
            else:
                combined.save(output_path)

            # Close combined image
            combined.close()

        finally:
            # Close all loaded images
            for img in images:
                img.close()
