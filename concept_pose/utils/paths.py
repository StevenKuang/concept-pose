"""
Path utilities for handling file paths and ensuring cross-platform compatibility.
"""

from pathlib import Path
from typing import Optional, Union


def ensure_path(path: Union[str, Path]) -> Path:
    """
    Convert string to Path object if needed.

    Args:
        path: String or Path object

    Returns:
        Path object
    """
    return Path(path) if isinstance(path, str) else path


def ensure_dir_exists(path: Union[str, Path], parents: bool = True) -> Path:
    """
    Ensure directory exists, creating it if necessary.

    Args:
        path: Directory path
        parents: Create parent directories if needed

    Returns:
        Path object
    """
    path = ensure_path(path)
    path.mkdir(parents=parents, exist_ok=True)
    return path


def resolve_model_path(model_name: str, models_dir: Optional[Union[str, Path]] = None) -> Path:
    """
    Resolve path to a model file.

    Args:
        model_name: Model filename (e.g., 'cup_model.npz')
        models_dir: Base directory for models

    Returns:
        Resolved path to model
    """
    if models_dir is None:
        # Default to outputs directory
        models_dir = Path('./outputs')

    models_dir = ensure_path(models_dir)
    return models_dir / model_name


def resolve_data_path(relative_path: str, data_dir: Union[str, Path]) -> Path:
    """
    Resolve path relative to data directory.

    Args:
        relative_path: Relative path within data directory
        data_dir: Base data directory

    Returns:
        Resolved absolute path
    """
    data_dir = ensure_path(data_dir)
    return data_dir / relative_path


def get_category_model_path(category: str, object_name: str, models_dir: Union[str, Path]) -> Path:
    """
    Get path to object model file following HouseCat6D convention.

    Args:
        category: Object category (e.g., 'cup', 'bottle')
        object_name: Full object name (e.g., 'cup-green_actys')
        models_dir: Base directory for 3D object models

    Returns:
        Path to .obj file
    """
    models_dir = ensure_path(models_dir)
    return models_dir / category / f"{object_name}.obj"
