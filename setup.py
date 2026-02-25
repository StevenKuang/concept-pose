"""
Setup file for concept-pose package

Semantic Concept-Based 6D Pose Estimation
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text() if readme_file.exists() else ""

setup(
    name="concept-pose",
    version="1.0.0",
    author="Liming Kuang",
    description="Semantic Concept-Based 6D Pose Estimation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/StevenKuang/concept-pose",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Topic :: Scientific/Engineering :: Computer Vision",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Environment :: GPU :: NVIDIA CUDA",
    ],
    python_requires=">=3.8",
    install_requires=[
        # Core dependencies
        "numpy>=1.21.0",
        "scipy>=1.7.0",

        # PyTorch (GPU-accelerated)
        # Note: For CUDA support, install via: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
        "torch>=2.0.0",
        "torchvision>=0.15.0",

        # Computer vision
        "opencv-python>=4.5.0",
        "Pillow>=8.0.0",

        # Machine learning & transformers
        "transformers>=4.30.0",
        "scikit-learn>=1.0.0",
        "timm>=0.9.0",

        # 3D processing
        "trimesh>=3.15.0",

        # LLM API (for semantic label generation)
        "google-genai>=0.2.0",

        # Visualization
        "matplotlib>=3.4.0",

        # Configuration & utilities
        "pyyaml>=6.0",
        "tqdm>=4.60.0",
        "requests>=2.25.0"
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
        ],
    },
    include_package_data=True,
    package_data={
        "concept_pose": [
            "partonomy/*.json",
        ],
    },
    zip_safe=False,
    keywords=[
        "6d-pose-estimation",
        "computer-vision",
        "semantic-saliency",
        "pytorch",
        "cuda",
        "concept-based",
        "robotics",
        "object-pose",
    ],
)
