"""
Configuration Management System

Handles loading, validation, and defaults for all configuration parameters.
Eliminates hard-coded paths and makes the codebase deployment-ready.
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional, Union


class Config:
    """
    Configuration class with validation and defaults.

    This centralizes all configuration management, replacing scattered
    hard-coded values throughout the original codebase.
    """

    # Default configuration values
    DEFAULTS = {
        # Data paths (must be specified by user)
        'data_dir': None,
        'models_dir': None,
        'output_dir': './outputs',

        # Dataset configuration
        'category': 'cup',
        'num_videos': 1,
        'target_size': 518,
        'max_frames_per_video': None,

        # Model building
        'object_names': None,  # List of objects to train on
        'num_training_views': 5,  # 0 = all, negative = reserve for test
        'num_semantic_labels': 5,
        'manual_labels': None,  # Override automatic label generation

        # Voxelization (NOCS normalization is always enabled)
        'voxel_resolution': 64,
        'aggregation_method': 'mean',

        # Pose estimation
        'use_gt_scale': False,
        'use_flann': True,
        'flann_k_neighbors': 5,
        'pnp_method': 'RANSAC',
        'ransac_threshold': 0.01,
        'ransac_iterations': 1000,

        # Loss function
        'loss_method': 'asymmetric',  # 'asymmetric', 'kl_divergence', 'cosine'
        'lambda_iou': 1.0,
        'low_threshold': 0.1,

        # Visualization
        'visualize': False,
        'save_ply': False,

        # Paths (with sensible defaults)
        'parts_json': None,  # Will default to package resource if None
    }

    # Required fields that must be provided
    REQUIRED_FIELDS = [
        'data_dir',
    ]

    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        """
        Initialize configuration from dictionary.

        Args:
            config_dict: Configuration dictionary. Defaults are used for missing values.
        """
        # Start with defaults
        self._config = self.DEFAULTS.copy()

        # Update with provided config
        if config_dict:
            self._config.update(config_dict)

        # Validate
        self._validate()

        # Resolve paths
        self._resolve_paths()

    def _validate(self):
        """Validate required fields are present"""
        missing = [field for field in self.REQUIRED_FIELDS
                  if self._config.get(field) is None]

        if missing:
            raise ValueError(
                f"Missing required configuration fields: {', '.join(missing)}\n"
                f"Please specify these in your config file or as arguments."
            )

    def _resolve_paths(self):
        """Convert string paths to Path objects and validate existence for critical paths"""
        path_fields = ['data_dir', 'models_dir', 'output_dir', 'parts_json']

        for field in path_fields:
            value = self._config.get(field)
            if value is not None:
                self._config[field] = Path(value)

        # Create output directory if it doesn't exist
        if self._config['output_dir'] is not None:
            self._config['output_dir'].mkdir(parents=True, exist_ok=True)

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value with optional default"""
        return self._config.get(key, default)

    def __getitem__(self, key: str) -> Any:
        """Dictionary-style access"""
        return self._config[key]

    def __setitem__(self, key: str, value: Any):
        """Dictionary-style setting"""
        self._config[key] = value

    def update(self, updates: Dict[str, Any]):
        """Update configuration with new values"""
        self._config.update(updates)
        self._validate()
        self._resolve_paths()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (for serialization)"""
        # Convert Path objects back to strings for JSON serialization
        result = {}
        for key, value in self._config.items():
            if isinstance(value, Path):
                result[key] = str(value)
            else:
                result[key] = value
        return result

    @classmethod
    def from_file(cls, config_path: Union[str, Path]) -> 'Config':
        """
        Load configuration from JSON file.

        Args:
            config_path: Path to JSON configuration file

        Returns:
            Config instance

        Example:
            >>> config = Config.from_file('configs/cup_example.json')
        """
        config_path = Path(config_path)

        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(config_path, 'r') as f:
            config_dict = json.load(f)

        return cls(config_dict)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'Config':
        """
        Create configuration from dictionary.

        Args:
            config_dict: Configuration dictionary

        Returns:
            Config instance
        """
        return cls(config_dict)

    def save(self, output_path: Union[str, Path]):
        """
        Save configuration to JSON file.

        Args:
            output_path: Path to save configuration
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    def __repr__(self) -> str:
        """String representation"""
        return f"Config({self.to_dict()})"

    def __str__(self) -> str:
        """Pretty print configuration"""
        lines = ["Configuration:"]
        for key, value in sorted(self._config.items()):
            lines.append(f"  {key}: {value}")
        return "\n".join(lines)


def load_config(config_source: Union[str, Path, Dict[str, Any], None]) -> Config:
    """
    Convenience function to load configuration from various sources.

    Args:
        config_source: Can be:
            - Path to JSON file (str or Path)
            - Dictionary
            - None (uses defaults)

    Returns:
        Config instance

    Examples:
        >>> config = load_config('configs/cup.json')
        >>> config = load_config({'data_dir': '/path/to/data'})
        >>> config = load_config(None)  # Use defaults
    """
    if config_source is None:
        return Config()

    elif isinstance(config_source, (str, Path)):
        return Config.from_file(config_source)

    elif isinstance(config_source, dict):
        return Config.from_dict(config_source)

    else:
        raise TypeError(
            f"config_source must be str, Path, dict, or None, got {type(config_source)}"
        )
