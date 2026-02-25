"""
Dataset factory for creating dataset instances from configuration.

This factory pattern allows dynamic dataset creation based on JSON/YAML configs,
making it easy to switch between datasets without changing code.
"""

import json
from pathlib import Path
from typing import Dict, Union

from .base_dataset import BaseDataset
from .dataset_real275 import DatasetReal275
from .dataset_tyol import DatasetTyol
from .dataset_ycbv import DatasetYCBV
from .dataset_lm import DatasetLM
from .dataset_lmo import DatasetLMO


def load_dataset_config(config_path: Union[str, Path]) -> Dict:
    """
    Load dataset configuration from JSON file.

    Args:
        config_path: Path to dataset config JSON

    Returns:
        Configuration dictionary

    Raises:
        FileNotFoundError: If config file doesn't exist
        json.JSONDecodeError: If JSON is invalid
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Dataset config not found: {config_path}")

    with open(config_path, 'r') as f:
        config = json.load(f)

    return config


def create_dataset(
    dataset_type: str,
    config: Dict,
    **override_params
) -> BaseDataset:
    """
    Create dataset instance from type and configuration.

    Args:
        dataset_type: Dataset type identifier (e.g., 'real275', 'ycbv', 'lm', 'lmo', 'tyol')
        config: Dataset configuration dictionary
        **override_params: Optional parameters to override config values

    Returns:
        Dataset instance implementing BaseDataset interface

    Raises:
        ValueError: If dataset type is unknown or unsupported

    Example:
        >>> config = load_dataset_config('configs/datasets/ycbv.json')
        >>> dataset = create_dataset('ycbv', config)
    """
    # Merge config params with overrides
    params = {**config.get('params', {}), **override_params}

    # Extract common parameters
    data_dir = config.get('paths', {}).get('root', params.get('data_dir'))
    target_size = config.get('preprocessing', {}).get('target_size', 518)

    # Create dataset based on type
    if dataset_type == 'real275':
        # NOCS Real275 test set
        return DatasetReal275(
            data_dir=data_dir,
            target_size=target_size,
            num_scenes=config.get('num_scenes', 6),
            **{k: v for k, v in params.items()
               if k not in ['data_dir', 'target_size', 'num_scenes']}
        )

    elif dataset_type == 'tyol':
        # TYOL (ToyotaLight) BOP test set
        paths = config.get('paths', {})
        data_root = paths.get('root', params.get('data_dir'))
        oryon_root = paths.get('oryon_root', params.get('oryon_root'))
        preprocessing = config.get('preprocessing', {})
        depth_scale = preprocessing.get('depth_scale', 1000.0)

        if not data_root:
            raise ValueError("TYOL dataset requires 'root' in config paths")
        if not oryon_root:
            raise ValueError("TYOL dataset requires 'oryon_root' in config paths")

        return DatasetTyol(
            bop_root=data_root,
            oryon_root=oryon_root,
            target_size=target_size,
            num_scenes=config.get('num_scenes', 21),
            depth_scale=depth_scale,
            **{k: v for k, v in params.items()
               if k not in ['data_dir', 'target_size', 'num_scenes', 'depth_scale']}
        )

    elif dataset_type == 'ycbv':
        # YCB-Video BOP test set
        paths = config.get('paths', {})
        data_root = paths.get('root', params.get('data_dir'))
        category_mapping_path = paths.get('category_mapping', params.get('category_mapping_path'))
        preprocessing = config.get('preprocessing', {})
        depth_scale = preprocessing.get('depth_scale', 10000.0)  # 0.1mm scale

        if not data_root:
            raise ValueError("YCB-V dataset requires 'root' in config paths")
        if not category_mapping_path:
            raise ValueError("YCB-V dataset requires 'category_mapping' in config paths")

        return DatasetYCBV(
            bop_root=data_root,
            category_mapping_path=category_mapping_path,
            target_size=target_size,
            num_scenes=config.get('num_scenes', 92),
            depth_scale=depth_scale,
            **{k: v for k, v in params.items()
               if k not in ['data_dir', 'target_size', 'num_scenes', 'depth_scale', 'category_mapping_path']}
        )

    elif dataset_type == 'lm' or dataset_type == 'linemod':
        # LINEMOD BOP test set
        paths = config.get('paths', {})
        data_root = paths.get('root', params.get('data_dir'))
        preprocessing = config.get('preprocessing', {})
        depth_scale = preprocessing.get('depth_scale', 1000.0)  # 1000.0 for LM (mm->m)
        use_bop_targets = config.get('test_set', {}).get('use_bop_targets', True)

        if not data_root:
            raise ValueError("LINEMOD dataset requires 'root' in config paths")

        return DatasetLM(
            bop_root=data_root,
            target_size=target_size,
            num_scenes=config.get('num_scenes', 15),
            depth_scale=depth_scale,
            use_bop_targets=use_bop_targets,
            **{k: v for k, v in params.items()
               if k not in ['data_dir', 'target_size', 'num_scenes', 'depth_scale', 'use_bop_targets']}
        )

    elif dataset_type == 'lmo':
        # LINEMOD-Occlusion BOP test set
        paths = config.get('paths', {})
        data_root = paths.get('root', params.get('data_dir'))
        preprocessing = config.get('preprocessing', {})
        depth_scale = preprocessing.get('depth_scale', 1000.0)
        use_bop_targets = config.get('test_set', {}).get('use_bop_targets', True)

        # LM-O uses different model/targets subdirs
        mesh_dir = paths.get('mesh_dir', 'lmo_models/models')
        bop_targets_subdir = paths.get('bop_targets_subdir', 'lmo_base')

        if not data_root:
            raise ValueError("LM-O dataset requires 'root' in config paths")

        return DatasetLMO(
            bop_root=data_root,
            target_size=target_size,
            depth_scale=depth_scale,
            use_bop_targets=use_bop_targets,
            models_subdir=mesh_dir,
            bop_targets_subdir=bop_targets_subdir,
            **{k: v for k, v in params.items()
               if k not in ['data_dir', 'target_size', 'depth_scale', 'use_bop_targets']}
        )

    else:
        raise ValueError(
            f"Unknown dataset type: '{dataset_type}'. "
            f"Supported types: ['real275', 'tyol', 'ycbv', 'lm', 'linemod', 'lmo']"
        )


def create_dataset_from_config_file(
    config_path: Union[str, Path],
    **override_params
) -> BaseDataset:
    """
    Create dataset directly from config file path.

    Convenience function that loads config and creates dataset in one call.

    Args:
        config_path: Path to dataset config JSON
        **override_params: Optional parameters to override config values

    Returns:
        Dataset instance

    Example:
        >>> dataset = create_dataset_from_config_file(
        ...     'configs/datasets/ycbv.json'
        ... )
    """
    config = load_dataset_config(config_path)
    dataset_type = config.get('type')

    if not dataset_type:
        raise ValueError(
            f"Dataset config missing 'type' field: {config_path}"
        )

    return create_dataset(dataset_type, config, **override_params)
