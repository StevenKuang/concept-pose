"""
GPU memory management utilities.

Provides functions for cleaning up GPU memory, managing model lifecycle,
and preventing memory leaks in PyTorch applications.

Extracted from siglip2_text_localization.py:311-351.
"""

import gc
import torch
from typing import Any, List, Optional


def release_model_memory(
    model: Optional[Any] = None,
    additional_objects: Optional[List[Any]] = None,
    verbose: bool = True
) -> None:
    """
    Release GPU memory held by PyTorch model and related objects.

    This function handles common patterns for cleaning up GPU memory:
    - Delete model references
    - Delete associated objects (processors, extractors, wrappers)
    - Run garbage collection
    - Empty CUDA cache

    Extracted from siglip2_text_localization.py:311-351.

    Args:
        model: PyTorch model to delete. If None, skips model deletion.
        additional_objects: List of additional objects to delete (e.g., processors, wrappers)
        verbose: If True, print cleanup confirmation message

    Example:
        >>> # Release single model
        >>> release_model_memory(model)

        >>> # Release model with additional objects
        >>> release_model_memory(
        ...     model=siglip_model,
        ...     additional_objects=[processor, cam_extractor, cam_wrapper]
        ... )

        >>> # Release only additional objects
        >>> release_model_memory(additional_objects=[tensor1, tensor2, tensor3])
    """
    # Delete main model
    if model is not None:
        del model
        model = None

    # Delete additional objects
    if additional_objects is not None:
        for obj in additional_objects:
            del obj
        additional_objects = None

    # Aggressive garbage collection before emptying cache
    gc.collect()

    # Empty CUDA cache if available
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if verbose:
        print("GPU memory release complete.")


def cleanup_vision_model(
    model_instance: Any,
    model_attr: str = 'model',
    processor_attr: str = 'processor',
    additional_attrs: Optional[List[str]] = None,
    verbose: bool = True
) -> None:
    """
    Clean up vision-language model instance with common attributes.

    This is a higher-level wrapper around release_model_memory that handles
    common patterns in vision-language model classes (SigLIP, CLIP, etc.).

    Extracted from siglip2_text_localization.py:311-351.

    Args:
        model_instance: Instance of the model class to clean up
        model_attr: Name of the main model attribute (default 'model')
        processor_attr: Name of the processor attribute (default 'processor')
        additional_attrs: List of additional attribute names to delete
                         (e.g., ['cam_extractor', 'cam_wrapper', 'text_input_ids'])
        verbose: If True, print cleanup messages

    Example:
        >>> class MySigLIPModel:
        ...     def __init__(self):
        ...         self.model = ...
        ...         self.processor = ...
        ...         self.cam_extractor = ...
        ...
        >>> instance = MySigLIPModel()
        >>> cleanup_vision_model(
        ...     instance,
        ...     additional_attrs=['cam_extractor', 'cam_wrapper']
        ... )
    """
    objects_to_delete = []

    # Collect model
    if hasattr(model_instance, model_attr):
        model = getattr(model_instance, model_attr)
        objects_to_delete.append(model)
        setattr(model_instance, model_attr, None)

    # Collect processor
    if hasattr(model_instance, processor_attr):
        processor = getattr(model_instance, processor_attr)
        objects_to_delete.append(processor)
        setattr(model_instance, processor_attr, None)

    # Collect additional attributes
    if additional_attrs is not None:
        for attr_name in additional_attrs:
            if hasattr(model_instance, attr_name):
                obj = getattr(model_instance, attr_name)
                if obj is not None:
                    objects_to_delete.append(obj)
                setattr(model_instance, attr_name, None)

    # Release all collected objects
    release_model_memory(additional_objects=objects_to_delete, verbose=verbose)


def cleanup_cam_wrapper(wrapper_instance: Any) -> None:
    """
    Clean up CAM (Class Activation Map) wrapper and its model.

    CAM wrappers often hold references to the underlying model,
    requiring special handling to fully release memory.

    Args:
        wrapper_instance: CAM wrapper instance (e.g., from pytorch_grad_cam)

    Example:
        >>> from pytorch_grad_cam import GradCAM
        >>> cam_wrapper = GradCAM(model=model, target_layers=layers)
        >>> # ... use cam_wrapper ...
        >>> cleanup_cam_wrapper(cam_wrapper)
    """
    if hasattr(wrapper_instance, 'model'):
        del wrapper_instance.model
        wrapper_instance.model = None

    del wrapper_instance
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def get_gpu_memory_stats() -> dict:
    """
    Get current GPU memory statistics.

    Returns:
        Dictionary with memory statistics in MB, or empty dict if CUDA unavailable

    Example:
        >>> stats = get_gpu_memory_stats()
        >>> print(f"Allocated: {stats['allocated_mb']:.1f} MB")
        >>> print(f"Reserved: {stats['reserved_mb']:.1f} MB")
    """
    if not torch.cuda.is_available():
        return {}

    return {
        'allocated_mb': torch.cuda.memory_allocated() / 1024**2,
        'reserved_mb': torch.cuda.memory_reserved() / 1024**2,
        'max_allocated_mb': torch.cuda.max_memory_allocated() / 1024**2,
        'device_count': torch.cuda.device_count(),
        'current_device': torch.cuda.current_device()
    }


def print_gpu_memory_stats(prefix: str = "") -> None:
    """
    Print GPU memory statistics in a readable format.

    Args:
        prefix: Optional prefix for the output message

    Example:
        >>> print_gpu_memory_stats("Before model load")
        >>> model = load_large_model()
        >>> print_gpu_memory_stats("After model load")
    """
    stats = get_gpu_memory_stats()

    if not stats:
        print(f"{prefix}CUDA not available")
        return

    message = f"{prefix}GPU Memory: "
    message += f"Allocated={stats['allocated_mb']:.1f}MB, "
    message += f"Reserved={stats['reserved_mb']:.1f}MB, "
    message += f"Peak={stats['max_allocated_mb']:.1f}MB"

    print(message)


def reset_peak_memory_stats() -> None:
    """
    Reset peak memory statistics.

    Useful for benchmarking memory usage of specific code sections.

    Example:
        >>> reset_peak_memory_stats()
        >>> # Run memory-intensive operation
        >>> process_large_batch()
        >>> stats = get_gpu_memory_stats()
        >>> print(f"Peak memory for batch: {stats['max_allocated_mb']:.1f}MB")
    """
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


class GPUMemoryContext:
    """
    Context manager for tracking GPU memory usage.

    Automatically prints memory stats before and after the context,
    and cleans up at the end.

    Example:
        >>> with GPUMemoryContext("Model inference"):
        ...     output = model(input_tensor)
        GPU Memory [Model inference - Start]: Allocated=1024.5MB, Reserved=2048.0MB
        GPU Memory [Model inference - End]: Allocated=1536.2MB, Reserved=2048.0MB
        GPU Memory [Model inference - Delta]: +511.7MB allocated
    """

    def __init__(self, name: str = "Operation", cleanup: bool = True):
        """
        Args:
            name: Name of the operation being tracked
            cleanup: If True, run cleanup on exit
        """
        self.name = name
        self.cleanup = cleanup
        self.start_allocated = 0.0

    def __enter__(self):
        if torch.cuda.is_available():
            self.start_allocated = torch.cuda.memory_allocated() / 1024**2
            print_gpu_memory_stats(f"[{self.name} - Start] ")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if torch.cuda.is_available():
            end_allocated = torch.cuda.memory_allocated() / 1024**2
            print_gpu_memory_stats(f"[{self.name} - End] ")
            delta = end_allocated - self.start_allocated
            print(f"[{self.name} - Delta] {delta:+.1f}MB allocated")

            if self.cleanup:
                gc.collect()
                torch.cuda.empty_cache()
