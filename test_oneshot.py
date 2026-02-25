#!/usr/bin/env python3
"""
Generic One-Shot Pose Estimation Testing
==========================================

Dataset-agnostic one-shot pose estimation evaluation driven by YAML configuration.
Supports multiple datasets (HouseCat6D, NOCS, Toyota, etc.) via the BaseDataset interface.

Deterministic Mode:
    By default, uses seed=42 for reproducible RANSAC results (following Oryon/Any6D).
    Results will be identical across runs.

Usage:
    # Run with config file (deterministic, seed=42)
    python test_oneshot.py --config configs/evaluations/oneshot_housecat_cup.yaml

    # Override specific parameters
    python test_oneshot.py \\
        --config configs/evaluations/oneshot_housecat_cup.yaml \\
        --max_pairs 10 \\
        --output results_debug.json \\
        --device cpu

    # Run with custom seed
    python test_oneshot.py \\
        --config configs/evaluations/oneshot_housecat_cup.yaml \\
        --seed 123
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import argparse
import json
import yaml
import numpy as np
import torch
import random
import time
from pathlib import Path

# Import evaluation framework
from concept_pose.evaluation import OneShotEvaluator, PairSampler
from concept_pose.data import create_dataset, load_dataset_config
from concept_pose.pose.one_shot_estimator import OneShotPoseEstimator
from concept_pose.pose.bop_metrics import BOPEvaluator


def set_deterministic_mode(seed=42):
    """Set random seed for reproducible RANSAC results (including GPU operations)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"Deterministic RANSAC enabled (seed={seed}, CPU+GPU)")


def load_config(config_path):
    """Load YAML evaluation configuration."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_object_config(config_path):
    """Load object semantic labels configuration (JSON)."""
    with open(config_path, 'r') as f:
        obj_config = json.load(f)
    return obj_config


def create_dataset_from_eval_config(eval_config):
    """Create dataset from evaluation config."""
    dataset_config_path = eval_config['dataset']['config']
    dataset_config = load_dataset_config(dataset_config_path)

    # Apply overrides from eval config
    if 'target_size' in eval_config['dataset']:
        dataset_config['params']['target_size'] = eval_config['dataset']['target_size']

    # Create dataset
    dataset = create_dataset(
        dataset_type=eval_config['dataset']['type'],
        config=dataset_config,
        category=eval_config['dataset'].get('category')
    )

    return dataset, dataset_config


def get_test_pairs(eval_config, dataset, sampler):
    """Load or sample test pairs based on config."""
    pairs_config = eval_config['pairs']

    # Option 1: Load from file
    if 'file' in pairs_config:
        pairs, split_config = sampler.load_pairs(pairs_config['file'])
        return pairs, split_config

    # Option 2: Sample on-the-fly
    elif 'sampling' in pairs_config:
        sampling_config = pairs_config['sampling']

        # Check if this is oryon fixed split mode for Real275
        if sampling_config.get('mode') == 'oryon_fixed_real275_split':
            # Load oryon's fixed split
            oryon_root = sampling_config.get(
                'oryon_root',
                'data/oryon_data/datasets/nocs'
            )
            split_name = sampling_config.get('split_name', 'cross_scene_test')
            num_pairs = sampling_config.get('num_pairs')  # Optional: limit to first N pairs

            pairs, split_config = sampler.load_oryon_fixed_split(
                oryon_root=oryon_root,
                split_name=split_name,
                num_pairs=num_pairs
            )
            return pairs, split_config

        # Check if this is oryon fixed split mode for TYOL
        elif sampling_config.get('mode') == 'oryon_fixed_tyol_split':
            # Load oryon's fixed TYOL split
            oryon_root = sampling_config.get(
                'oryon_root',
                'data/oryon_data/datasets/toyl' # oryon got it wrong so keep it toyl here
            )
            split_name = sampling_config.get('split_name', 'cross_scene_test')
            num_pairs = sampling_config.get('num_pairs')  # Optional: limit to first N pairs

            pairs, split_config = sampler.load_oryon_fixed_split_tyol(
                oryon_root=oryon_root,
                split_name=split_name,
                num_pairs=num_pairs
            )
            return pairs, split_config

        # Check if this is One2Any first occurrence mode for YCB-V
        elif sampling_config.get('mode') == 'one2any_first_occurrence_ycbv':
            # Use One2Any's first occurrence sampling strategy
            num_pairs = sampling_config.get('num_pairs')  # Optional: limit to N pairs
            seed = sampling_config.get('seed', 42)  # Random seed for sampling when limiting

            pairs, split_config = sampler.sample_one2any_first_occurrence_ycbv(
                num_pairs=num_pairs,
                seed=seed
            )
            return pairs, split_config

        # Check if this is Oryon-style random sampling for YCB-V
        elif sampling_config.get('mode') == 'oryon_style_ycbv':
            # Use Oryon-style random sampling (like Real275 evaluation)
            num_pairs = sampling_config.get('num_pairs', 2000)
            scene_mode = sampling_config.get('scene_mode', 'cross_scene')
            seed = sampling_config.get('seed', 42)
            use_bop_targets = sampling_config.get('use_bop_targets', True)

            pairs, split_config = sampler.sample_oryon_style_ycbv(
                num_pairs=num_pairs,
                scene_mode=scene_mode,
                seed=seed,
                use_bop_targets=use_bop_targets
            )
            return pairs, split_config

        # Check if this is Oryon-style random sampling for LINEMOD
        elif sampling_config.get('mode') == 'oryon_style_lm':
            # Use Oryon-style random sampling for LINEMOD (same-scene pairs)
            num_pairs = sampling_config.get('num_pairs', 2000)
            scene_mode = sampling_config.get('scene_mode', 'same_scene')
            seed = sampling_config.get('seed', 42)
            use_bop_targets = sampling_config.get('use_bop_targets', True)

            pairs, split_config = sampler.sample_oryon_style_lm(
                num_pairs=num_pairs,
                scene_mode=scene_mode,
                seed=seed,
                use_bop_targets=use_bop_targets
            )
            return pairs, split_config

        # Check if this is Oryon fixed split mode for YCB-V
        elif sampling_config.get('mode') == 'oryon_fixed_ycbv_split':
            # Load Oryon's fixed YCB-V split
            oryon_root = sampling_config.get(
                'oryon_root',
                'data/oryon_data/ycbv'
            )
            split_name = sampling_config.get('split_name', 'cross_scene_test')
            num_pairs = sampling_config.get('num_pairs')  # Optional: limit to first N pairs

            pairs, split_config = sampler.load_oryon_fixed_split_ycbv(
                oryon_root=oryon_root,
                split_name=split_name,
                num_pairs=num_pairs
            )
            return pairs, split_config

        # Check if this is Oryon fixed split mode for LINEMOD
        elif sampling_config.get('mode') == 'oryon_fixed_lm_split':
            # Load Oryon's fixed LINEMOD split
            oryon_root = sampling_config.get(
                'oryon_root',
                'data/oryon_data/lm'
            )
            split_name = sampling_config.get('split_name', 'same_scene_test')
            num_pairs = sampling_config.get('num_pairs')  # Optional: limit to first N pairs

            pairs, split_config = sampler.load_oryon_fixed_split_lm(
                oryon_root=oryon_root,
                split_name=split_name,
                num_pairs=num_pairs
            )
            return pairs, split_config

        # Check if this is One2Any first occurrence mode for LINEMOD
        elif sampling_config.get('mode') == 'one2any_first_occurrence_lm':
            # Use One2Any's first occurrence sampling strategy for LINEMOD
            num_pairs = sampling_config.get('num_pairs')  # Optional: limit to N pairs
            seed = sampling_config.get('seed', 42)  # Random seed for sampling when limiting
            use_bop_targets = sampling_config.get('use_bop_targets', True)

            pairs, split_config = sampler.sample_one2any_first_occurrence_lm(
                num_pairs=num_pairs,
                seed=seed,
                use_bop_targets=use_bop_targets
            )
            return pairs, split_config

        # Check if this is LM-O occlusion study mode
        elif sampling_config.get('mode') == 'lmo_occlusion_study':
            # Special sampling for occlusion vs performance analysis:
            # - Anchor = least occluded frame per object
            # - Queries = all other frames (varying occlusion, uniformly sampled)
            seed = sampling_config.get('seed', 42)
            max_queries_per_object = sampling_config.get('max_queries_per_object')

            pairs, split_config = sampler.sample_lmo_occlusion_study(
                seed=seed,
                max_queries_per_object=max_queries_per_object
            )
            return pairs, split_config

        # Standard on-the-fly sampling
        # Get objects to sample from
        category = eval_config['dataset'].get('category')
        if category:
            objects = dataset.get_all_objects(category=category)
        else:
            objects = dataset.get_all_objects()

        print(f"Found {len(objects)} objects for sampling")

        # Sample pairs
        pairs = sampler.sample_pairs(
            objects=objects,
            num_pairs=sampling_config['num_pairs'],
            scene_mode=sampling_config.get('scene_mode', 'mixed'),
            min_frame_gap=sampling_config.get('min_frame_gap', 10),
            seed=sampling_config.get('seed')
        )

        split_config = {
            'name': f"sampled_{sampling_config['num_pairs']}",
            'total_pairs': len(pairs),
            'scene_mode': sampling_config.get('scene_mode', 'mixed'),
            'category': category or 'all'
        }

        return pairs, split_config

    else:
        raise ValueError("Config must specify either pairs.file or pairs.sampling")


def main():
    parser = argparse.ArgumentParser(
        description='Generic one-shot pose estimation testing',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument('--config', type=str, required=True,
                       help='Path to evaluation YAML config')
    parser.add_argument('--output', type=str, default=None,
                       help='Override output results file')
    parser.add_argument('--max_pairs', type=int, default=None,
                       help='Override max pairs to test (for debugging)')
    parser.add_argument('--device', type=str, default=None,
                       help='Override device (cuda or cpu)')
    parser.add_argument('--sample_pairs', type=int, default=None,
                       help='Override config and sample N pairs on-the-fly')
    parser.add_argument('--visualize', action='store_true',
                       help='Override config and enable visualization')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for RANSAC (default: 42, deterministic)')
    parser.add_argument('--save-debug-viz', type=str, default=None,
                       help='Save debug visualization data to .npz files in this directory')
    parser.add_argument('--gpu-ransac', action='store_true',
                       help='Use GPU-accelerated batched RANSAC (50-100x faster, enabled by default in config)')
    parser.add_argument('--cpu-ransac', action='store_true',
                       help='Force CPU RANSAC (slower, for compatibility testing)')
    parser.add_argument('--ransac-batch-size', type=int, default=None,
                       help='Batch size for GPU RANSAC (default: 1024, tune for GPU memory)')
    parser.add_argument('--mesh-visualize', action='store_true',
                       help='Visualize with downsampled mesh instead of anchor point cloud')
    parser.add_argument('--loss-method', type=str, default=None,
                       help='Correspondence method: kl_divergence, reverse_kl, bidirectional_kl, jensen_shannon, cosine, asymmetric, hellinger (default: from config or kl_divergence)')

    args = parser.parse_args()

    # Start total runtime timer
    start_time_total = time.time()

    # Set deterministic mode (always enabled with default seed=42)
    set_deterministic_mode(args.seed)

    # Load evaluation config
    print(f"\n{'='*60}")
    print(f"Loading evaluation config: {args.config}")
    print(f"{'='*60}\n")

    eval_config = load_config(args.config)

    print(f"Evaluation: {eval_config['name']}")
    print(f"Mode: {eval_config['mode']}")
    print(f"Description: {eval_config.get('description', 'N/A')}")

    # Create dataset
    print(f"\n{'='*60}")
    print("Creating dataset...")
    print(f"{'='*60}\n")

    dataset, dataset_config = create_dataset_from_eval_config(eval_config)
    print(f"Dataset: {dataset}")

    # Create pair sampler
    sampler = PairSampler(dataset)

    # Get test pairs
    print(f"\n{'='*60}")
    print("Loading/sampling test pairs...")
    print(f"{'='*60}\n")

    if args.sample_pairs is not None:
        # Override: sample on-the-fly
        category = eval_config['dataset'].get('category')
        objects = dataset.get_all_objects(category=category) if category else dataset.get_all_objects()
        pairs = sampler.sample_pairs(objects, args.sample_pairs, scene_mode='mixed', seed=42)
        split_config = {'name': f'sampled_{args.sample_pairs}', 'total_pairs': len(pairs), 'scene_mode': 'mixed'}
    else:
        pairs, split_config = get_test_pairs(eval_config, dataset, sampler)

    print(f"\nTest pairs loaded:")
    print(f"  Name: {split_config['name']}")
    print(f"  Total pairs: {split_config['total_pairs']}")
    print(f"  Scene mode: {split_config.get('scene_mode', 'unknown')}")

    # Load object semantic labels (with automatic generation via Partonomy)
    from concept_pose.utils import load_semantic_labels

    # Object config is now optional (only needed for manual labels override)
    object_config = None
    if 'object' in eval_config and 'config' in eval_config['object']:
        object_config = load_object_config(eval_config['object']['config'])

    # Get all unique object names from pairs
    all_objects = sorted(set(pair['object_name'] for pair in pairs))

    # Load semantic labels (auto-generates using Partonomy if not manual)
    num_labels = 20  # Default
    if object_config and 'num_semantic_labels' in object_config:
        num_labels = object_config['num_semantic_labels']
    elif 'object' in eval_config and 'num_semantic_labels' in eval_config['object']:
        num_labels = eval_config['object']['num_semantic_labels']

    # Get parts_json from dataset config (or None for built-in default)
    parts_json = dataset_config.get('parts_json')

    # Dataset-specific: Use custom category extractor for datasets with special naming
    category_extractor = None
    from concept_pose.data.dataset_ycbv import DatasetYCBV
    from concept_pose.data.dataset_lm import DatasetLM
    from concept_pose.data.dataset_lmo import DatasetLMO

    if isinstance(dataset, (DatasetYCBV, DatasetLM, DatasetLMO)):
        # Build object_name -> category mapping from pairs
        object_to_category = {}
        for pair in pairs:
            obj_name = pair['object_name']
            if 'category' in pair:
                object_to_category[obj_name] = pair['category']

        # Create custom category extractor
        def custom_category_extractor(object_name: str) -> str:
            """Extract category from pair metadata (handles special naming for YCBV/LM)."""
            if object_name in object_to_category:
                return object_to_category[object_name]
            # Fallback: use dataset method
            try:
                obj_id = dataset._get_object_id(object_name)
                if obj_id is not None:
                    if isinstance(dataset, DatasetYCBV):
                        return dataset._get_category_name(obj_id)
                    elif isinstance(dataset, (DatasetLM, DatasetLMO)):
                        return dataset._get_bop_name(obj_id)
            except:
                pass
            # Final fallback
            from concept_pose.utils.label_utils import extract_category_auto
            return extract_category_auto(object_name)

        category_extractor = custom_category_extractor

    semantic_labels = load_semantic_labels(
        object_names=all_objects,
        config=object_config,
        num_labels=num_labels,
        parts_json=parts_json,
        category_extractor=category_extractor
    )

    # Print sample labels for first few objects
    print(f"\nSemantic labels loaded for {len(semantic_labels)} objects")
    for obj in list(semantic_labels.keys())[:3]:
        labels = semantic_labels[obj]
        print(f"  {obj}: {labels[:5]}..." if len(labels) > 5 else f"  {obj}: {labels}")

    # Initialize pose estimator
    print(f"\n{'='*60}")
    print("Initializing pose estimator...")
    print(f"{'='*60}\n")

    est_config = eval_config['estimation']
    device = args.device or eval_config.get('device', 'cuda')

    # Determine RANSAC mode (GPU vs CPU)
    use_gpu_ransac = est_config.get('use_gpu_ransac', True)  # Default: GPU
    if args.cpu_ransac:
        use_gpu_ransac = False
        print("  Forcing CPU RANSAC (--cpu-ransac flag)")
    elif args.gpu_ransac:
        use_gpu_ransac = True
        print("  Forcing GPU RANSAC (--gpu-ransac flag)")

    # Get batch size (command line overrides config)
    ransac_batch_size = args.ransac_batch_size or est_config.get('ransac_batch_size', 1024)

    # Get correspondence method parameters (command line overrides config)
    loss_method = args.loss_method or est_config.get('loss_method', 'kl_divergence')
    temperature = est_config.get('temperature', 1.0)
    lambda_reverse = est_config.get('lambda_reverse', 0.5)

    # Get binarization parameters (ablation study)
    binarize_saliency = est_config.get('binarize_saliency', False)
    binarize_threshold = est_config.get('binarize_threshold', 0.5)

    estimator = OneShotPoseEstimator(
        voxel_resolution=est_config['voxel_resolution'],
        ransac_iterations=est_config['ransac_iterations'],
        ransac_threshold=est_config['ransac_threshold'],
        similarity_threshold=est_config['similarity_threshold'],
        max_correspondences=est_config['max_correspondences'],
        use_icp=est_config['use_icp'],
        estimate_scale=est_config.get('estimate_scale', False),
        voxelize_anchor=est_config.get('voxelize_anchor', False),
        voxelize_query=est_config.get('voxelize_query', False),
        saliency_method='siglip',     # 'siglip' or 'clip' or 'dinotxt' (default: siglip)
        loss_method=loss_method,      # Correspondence method for semantic matching
        temperature=temperature,       # Temperature for KL-based methods
        lambda_reverse=lambda_reverse, # Weight for reverse KL in bidirectional
        use_gpu_ransac=use_gpu_ransac,
        ransac_batch_size=ransac_batch_size,
        device=device,
        binarize_saliency=binarize_saliency,
        binarize_threshold=binarize_threshold
    )

    print(f"  Voxel resolution: {est_config['voxel_resolution']}")
    print(f"  RANSAC mode: {'GPU (batched)' if use_gpu_ransac else 'CPU (sequential)'}")
    print(f"  RANSAC iterations: {est_config['ransac_iterations']}")
    if use_gpu_ransac:
        print(f"  RANSAC batch size: {ransac_batch_size}")
    print(f"  Estimate scale: {est_config.get('estimate_scale', False)}")
    print(f"  Correspondence method: {loss_method}")
    if loss_method in ['kl_divergence', 'reverse_kl', 'bidirectional_kl', 'jensen_shannon']:
        print(f"  Temperature: {temperature}")
    if loss_method == 'bidirectional_kl':
        print(f"  Lambda reverse: {lambda_reverse}")
    if binarize_saliency:
        print(f"  Binarize saliency: {binarize_saliency} (threshold={binarize_threshold})")
    print(f"  Device: {device}")

    # Load symmetries for all objects (matching Oryon's approach)
    print(f"\n{'='*60}")
    print("Loading object symmetries...")
    print(f"{'='*60}\n")

    symmetries = {}
    for obj_name in all_objects:
        try:
            # Check if dataset supports symmetry loading
            if hasattr(dataset, 'get_symmetry_transformations'):
                obj_syms = dataset.get_symmetry_transformations(obj_name, max_sym_disc_step=0.05)
                symmetries[obj_name] = obj_syms
                if len(obj_syms) > 1:
                    print(f"  {obj_name}: {len(obj_syms)} symmetry transformations")
            else:
                # Fallback: identity only
                symmetries[obj_name] = [{'R': np.eye(3), 't': np.zeros((3, 1))}]
        except Exception as e:
            print(f"  Warning: Failed to load symmetries for {obj_name}: {e}")
            symmetries[obj_name] = [{'R': np.eye(3), 't': np.zeros((3, 1))}]

    num_with_syms = sum(1 for syms in symmetries.values() if len(syms) > 1)
    print(f"\nLoaded symmetries for {len(symmetries)} objects ({num_with_syms} have symmetries)")

    # Initialize BOP evaluator
    print(f"\n{'='*60}")
    print("Initializing BOP evaluator...")
    print(f"{'='*60}\n")

    # Get mesh directory from dataset config
    mesh_dir = os.path.join(
        dataset_config['paths']['root'],
        dataset_config['paths']['mesh_dir']
    )

    metrics_config = eval_config['metrics']
    mesh_scale = dataset_config.get('mesh_scale', 1.0)  # Default 1.0 = no scaling
    bop_evaluator = BOPEvaluator(
        mesh_dir=mesh_dir,
        symmetries=symmetries,  # Pass loaded symmetries
        renderer=metrics_config.get('renderer', 'pytorch3d'),
        device=metrics_config.get('device', device),
        mesh_scale=mesh_scale
    )

    print(f"  Mesh directory: {mesh_dir}")
    print(f"  Mesh scale: {mesh_scale} ({'mm->m' if mesh_scale == 0.001 else 'no conversion'})")
    print(f"  Renderer: {metrics_config.get('renderer', 'pytorch3d')}")
    print(f"  Symmetries: {len(symmetries)} objects ({num_with_syms} with symmetries)")

    # Create evaluator
    print(f"\n{'='*60}")
    print("Creating OneShotEvaluator...")
    print(f"{'='*60}\n")

    # Get anchor pose mode from config
    anchor_pose_mode = eval_config.get('estimation', {}).get('anchor_pose_mode', 'absolute')

    evaluator = OneShotEvaluator(
        dataset=dataset,
        estimator=estimator,
        bop_evaluator=bop_evaluator,
        semantic_labels=semantic_labels,
        device=device,
        anchor_pose_mode=anchor_pose_mode,
        use_mesh_visualization=args.mesh_visualize
    )

    # Run evaluation
    output_config = eval_config['output']

    # Setup visualization
    viz_dir = None
    if args.visualize or output_config.get('visualize', False):
        viz_dir = Path(output_config.get('viz_dir', 'viz_oneshot'))
        print(f"\nVisualizations will be saved to: {viz_dir}")

    # Determine output file
    output_file = args.output or output_config['results_file']

    # Prepare evaluation metadata (everything except results - will be added incrementally)
    eval_metadata = {
        'eval_config': eval_config['name'],
        'dataset': {
            'type': eval_config['dataset']['type'],
            'category': eval_config['dataset'].get('category'),
        },
        'split_config': split_config,
        'bop_metrics': {
            'ar_vsd': None,  # Will be computed at the end
            'ar_mssd': None,
            'ar_mspd': None,
            'bop_score': 0.0
        }
    }

    # Run evaluation with incremental saving
    max_pairs = args.max_pairs or output_config.get('max_pairs')

    # Setup debug visualization directory if requested
    debug_viz_dir = Path(args.save_debug_viz) if args.save_debug_viz else None

    results = evaluator.evaluate(
        pairs=pairs,
        mask_cache=None,
        viz_dir=viz_dir,
        max_pairs=max_pairs,
        output_file=output_file,
        eval_metadata=eval_metadata,
        debug_viz_dir=debug_viz_dir
    )

    print(f"\n{'='*60}")
    print("Computing final BOP metrics...")
    print(f"{'='*60}\n")

    # Compute BOP metrics
    ar_metrics = bop_evaluator.compute_average_recall()
    bop_score = bop_evaluator.compute_bop_score()

    # Update metadata with final BOP metrics
    eval_metadata['bop_metrics'] = {
        'ar_vsd': ar_metrics.get('ar_vsd'),
        'ar_mssd': ar_metrics.get('ar_mssd'),
        'ar_mspd': ar_metrics.get('ar_mspd'),
        'bop_score': bop_score
    }

    # Add summary statistics
    summary = OneShotEvaluator.get_results_summary(results)
    eval_metadata['summary'] = summary

    # Save final results with BOP metrics and summary
    evaluator._save_results_atomic(results, output_file, eval_metadata)

    # Update debug_viz.npz with final eval_metadata (if debug viz was requested)
    if debug_viz_dir is not None:
        debug_npz_path = debug_viz_dir / "debug_viz.npz"
        if debug_npz_path.exists():
            try:
                import json
                # Load existing debug file
                debug_data = np.load(debug_npz_path)
                # Create new dict with updated metadata
                save_dict = {key: debug_data[key] for key in debug_data.keys()}
                save_dict['eval_metadata_json'] = json.dumps(eval_metadata)
                # Save updated file
                np.savez_compressed(debug_npz_path, **save_dict)
                print(f"   Updated debug_viz.npz with final metrics")
            except Exception as e:
                print(f"   Warning: Failed to update debug_viz.npz with final metrics: {e}")

    print(f"\n{'='*60}")
    print(f"Results saved to: {output_file}")
    print(f"{'='*60}\n")

    # Calculate and print total runtime
    total_runtime = time.time() - start_time_total
    hours = int(total_runtime // 3600)
    minutes = int((total_runtime % 3600) // 60)
    seconds = total_runtime % 60

    print(f"\nTotal Runtime: ", end='')
    if hours > 0:
        print(f"{hours}h {minutes}m {seconds:.1f}s")
    elif minutes > 0:
        print(f"{minutes}m {seconds:.1f}s")
    else:
        print(f"{seconds:.1f}s")

    # Print summary
    print("\nSummary Statistics:")
    print(f"  Total pairs: {summary['total_pairs']}")
    print(f"  Successful: {summary['num_success']}")
    print(f"  Success rate: {summary['success_rate']*100:.1f}%")
    print(f"  Success rate (5deg/2cm): {summary['success_rate_5deg2cm']*100:.1f}%")
    print(f"  Success rate (5deg/5cm): {summary['success_rate_5deg5cm']*100:.1f}%")
    print(f"  Success rate (10deg/5cm): {summary['success_rate_10deg5cm']*100:.1f}%")
    print(f"  Success rate (10deg/10cm): {summary['success_rate_10deg10cm']*100:.1f}%")

    if summary['num_success'] > 0:
        print(f"\n  Mean rotation error: {summary['mean_rotation_error_deg']:.2f} deg")
        print(f"  Mean translation error: {summary['mean_translation_error_m']*1000:.2f} mm")
        print(f"  Mean ADD: {summary['mean_add_error']*1000:.2f} mm")
        print(f"  Mean 3D IoU: {summary['mean_iou_3d']:.3f}")

        # Threshold-based success metrics
        print(f"\n  ADD-10 success: {summary['add_10_success']}/{summary['num_success']} ({summary['add_10_rate']*100:.1f}%)")
        print(f"  ADD-S-10 success: {summary['adds_10_success']}/{summary['num_success']} ({summary['adds_10_rate']*100:.1f}%)")
        print(f"  ADD(S)-10 adaptive: {summary['adds_adaptive_10_success']}/{summary['num_success']} ({summary['adds_adaptive_10_rate']*100:.1f}%)")
        print(f"  3D IoU-50 success: {summary['iou_50_success']}/{summary['num_success']} ({summary['iou_50_rate']*100:.1f}%)")
        print(f"  3D IoU-75 success: {summary['iou_75_success']}/{summary['num_success']} ({summary['iou_75_rate']*100:.1f}%)")

        # AUC metrics
        print(f"\n  ADD-AUC: {summary['add_auc']*100:.1f}%")
        print(f"  ADDS-AUC: {summary['adds_auc']*100:.1f}%")

    # Print BOP metrics
    print(f"\nBOP Metrics:")
    print(f"  AR_VSD: {ar_metrics.get('ar_vsd', 'N/A')}")
    print(f"  AR_MSSD: {ar_metrics.get('ar_mssd', 'N/A')}")
    print(f"  AR_MSPD: {ar_metrics.get('ar_mspd', 'N/A')}")
    print(f"  BOP Score: {bop_score:.4f}")

    print()


if __name__ == '__main__':
    exit(main())
