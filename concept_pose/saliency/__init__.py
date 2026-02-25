"""
Saliency Generation Module
===========================

This module contains vision-language model based saliency generators for semantic
part localization.

Components:
-----------
- siglip2_generator: SigLIP2-based semantic saliency generation using GradCAM
- clip_generator: CLIP-based semantic saliency generation using GradCAM
- dinotxt_generator: DINOtxt-based semantic saliency generation using GradCAM
"""

from .siglip2_generator import (
    SigLIP2SaliencyGenerator,
    SigLIPGradCAMWrapper,
    reshape_transform_siglip
)

from .clip_generator import (
    CLIPGradCAMGenerator,
    CLIPGradCAMWrapper,
    reshape_transform_clip
)

from .dinotxt_generator import (
    DinoTxtGradCAMGenerator,
    DinoTxtGradCAMWrapper,
    reshape_transform_dinotxt
)

__all__ = [
    'SigLIP2SaliencyGenerator',
    'SigLIPGradCAMWrapper',
    'reshape_transform_siglip',
    'CLIPGradCAMGenerator',
    'CLIPGradCAMWrapper',
    'reshape_transform_clip',
    'DinoTxtGradCAMGenerator',
    'DinoTxtGradCAMWrapper',
    'reshape_transform_dinotxt'
]
