"""
Canonical Composite Image Renderer
===================================

Generates composite images of object instances at canonical pose for improved
VLM-based part label generation. Emphasizes geometry over appearance.

Usage:
    # Single category
    python -m concept_pose.partonomy.canonical_renderer --dataset HouseCat6D --category bottle

    # All categories in a dataset
    python -m concept_pose.partonomy.canonical_renderer --dataset HouseCat6D --all

    # All datasets
    python -m concept_pose.partonomy.canonical_renderer --all-datasets
"""

import argparse
import json
import os
import glob
from pathlib import Path
import numpy as np
import trimesh
import pyrender
from PIL import Image
import warnings

# Suppress pyrender warnings
warnings.filterwarnings('ignore', category=UserWarning)


class CanonicalRenderer:
    """Renders object meshes at canonical pose and creates composite grids"""

    def __init__(self, resolution=512, distance=2.0, fov=np.pi/6):
        """
        Args:
            resolution: Resolution per instance (e.g., 512x512)
            distance: Camera distance from object center (default: 2.0 for better framing)
            fov: Field of view in radians (default: pi/6 = 30 degrees for less distortion)
        """
        self.resolution = resolution
        self.distance = distance
        self.fov = fov

        # Initialize offscreen renderer
        self.renderer = pyrender.OffscreenRenderer(resolution, resolution)

    def __del__(self):
        """Clean up renderer"""
        if hasattr(self, 'renderer'):
            self.renderer.delete()

    def load_mesh(self, mesh_path, file_format='obj', dataset_name=None):
        """
        Load mesh from file and prepare for rendering

        Args:
            mesh_path: Path to mesh file (.obj or .ply)
            file_format: File format ('obj' or 'ply')
            dataset_name: Dataset name (for dataset-specific transforms)

        Returns:
            pyrender.Mesh object
        """
        # Load mesh using trimesh
        if file_format == 'obj':
            tmesh = trimesh.load(mesh_path, force='mesh')
        elif file_format == 'ply':
            tmesh = trimesh.load(mesh_path, force='mesh')
        else:
            raise ValueError(f"Unsupported format: {file_format}")

        # Remove texture/color to emphasize geometry
        tmesh.visual = trimesh.visual.ColorVisuals()

        # Apply dataset-specific coordinate system transforms
        if dataset_name and dataset_name.lower() == 'tyol':
            # TYOL models are stored with a different orientation (BOP format)
            # Apply -90-degree rotation around X-axis to stand them upright
            rotation_matrix = trimesh.transformations.rotation_matrix(
                angle=-np.pi / 2,  # -90 degrees (negative to flip the right way)
                direction=[1, 0, 0],  # Around X-axis
                point=[0, 0, 0]
            )
            tmesh.apply_transform(rotation_matrix)

        # Normalize mesh to fit well in frame (with some margin)
        bounds = tmesh.bounds
        center = (bounds[0] + bounds[1]) / 2.0
        scale = np.max(bounds[1] - bounds[0])
        # Scale to 0.6 units (leaves 40% margin for better framing)
        tmesh.vertices = (tmesh.vertices - center) / scale * 0.6

        # Convert to pyrender mesh
        mesh = pyrender.Mesh.from_trimesh(tmesh, smooth=True)
        return mesh

    def render_canonical(self, mesh, viewpoint='front'):
        """
        Render mesh at canonical pose with specified viewpoint

        Args:
            mesh: pyrender.Mesh object
            viewpoint: 'front' for horizontal view, 'top' for top-down view

        Returns:
            PIL.Image: Rendered RGB image
        """
        # Create scene
        scene = pyrender.Scene(bg_color=[255, 255, 255])  # White background

        # Add mesh at origin (identity pose)
        scene.add(mesh)

        # Add neutral lighting (emphasize geometry)
        if viewpoint == 'front':
            # Front/3-quarter view for upright objects
            # Main light from front-right
            light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0)
            scene.add(light, pose=self._look_at([self.distance*0.3, self.distance*0.2, self.distance]))

            # Fill light from left
            fill_light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=1.5)
            scene.add(fill_light, pose=self._look_at([-self.distance*0.3, self.distance*0.2, self.distance]))

            # Camera at front-right 3/4 view (slight angle)
            camera = pyrender.PerspectiveCamera(yfov=self.fov)
            camera_pose = self._look_at([self.distance*0.3, self.distance*0.15, self.distance*0.95])
            scene.add(camera, pose=camera_pose)

        else:  # top view
            # Top-down view for flat objects (plates, magazines)
            # Main light from above
            light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0)
            scene.add(light, pose=self._look_at([0, self.distance, 0.3]))

            # Fill light from side
            fill_light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=1.5)
            scene.add(fill_light, pose=self._look_at([self.distance*0.5, self.distance*0.8, 0.3]))

            # Camera looking down at slight angle
            camera = pyrender.PerspectiveCamera(yfov=self.fov)
            camera_pose = self._look_at([0.2, self.distance*0.9, 0.3])
            scene.add(camera, pose=camera_pose)

        # Render
        color, _ = self.renderer.render(scene)

        return Image.fromarray(color)

    def _look_at(self, eye, center=[0, 0, 0], up=[0, 1, 0]):
        """
        Create look-at camera pose matrix

        Args:
            eye: Camera position
            center: Look-at point
            up: Up vector

        Returns:
            4x4 transformation matrix
        """
        eye = np.array(eye)
        center = np.array(center)
        up = np.array(up)

        # Compute forward, right, up vectors
        forward = center - eye
        forward = forward / np.linalg.norm(forward)

        right = np.cross(forward, up)
        right = right / np.linalg.norm(right)

        up = np.cross(right, forward)

        # Build rotation matrix
        R = np.eye(4)
        R[:3, 0] = right
        R[:3, 1] = up
        R[:3, 2] = -forward
        R[:3, 3] = eye

        return R

    def create_composite(self, images, grid_size=(2, 3)):
        """
        Create composite grid image from multiple rendered images

        Args:
            images: List of PIL.Image objects
            grid_size: Tuple (rows, cols)

        Returns:
            PIL.Image: Composite grid image
        """
        rows, cols = grid_size
        num_images = len(images)

        # Adjust grid size if needed
        if num_images < rows * cols:
            # Use smaller grid
            if num_images <= 4:
                rows, cols = 2, 2
            elif num_images <= 6:
                rows, cols = 2, 3

        # Limit to grid capacity
        images = images[:rows * cols]

        # Create composite canvas
        composite_width = cols * self.resolution
        composite_height = rows * self.resolution
        composite = Image.new('RGB', (composite_width, composite_height), (255, 255, 255))

        # Paste images into grid
        for idx, img in enumerate(images):
            row = idx // cols
            col = idx % cols
            x = col * self.resolution
            y = row * self.resolution
            composite.paste(img, (x, y))

        return composite

    def render_category_composite(self, dataset_name, category, num_instances=6, grid_size=(2, 3)):
        """
        Render composite image for a category from a dataset

        Args:
            dataset_name: Name of dataset (HouseCat6D, nocs, tyol)
            category: Category name (e.g., "bottle", "cup")
            num_instances: Number of instances to include
            grid_size: Grid layout (rows, cols)

        Returns:
            PIL.Image: Composite image
        """
        # Determine viewpoint based on category
        # Front view for upright/container objects, top view for flat objects
        front_view_categories = {
            'bottle', 'mug', 'cup', 'can', 'basket', 'milk carton',
            'plastic bottle', 'glass', 'teapot', 'tube', 'box',
            'bowl', 'plastic container', 'cracker box'
        }

        viewpoint = 'front' if category in front_view_categories else 'top'

        # Get mesh files for category
        mesh_files = self._get_category_meshes(dataset_name, category, num_instances)

        if not mesh_files:
            raise ValueError(f"No meshes found for {dataset_name}/{category}")

        print(f"Rendering {len(mesh_files)} instances of {category} from {dataset_name} (viewpoint: {viewpoint})...")

        # Render each mesh
        rendered_images = []
        for mesh_path in mesh_files:
            # Detect file format
            file_format = 'ply' if mesh_path.endswith('.ply') else 'obj'

            # Load and render (pass dataset_name for dataset-specific transforms)
            mesh = self.load_mesh(mesh_path, file_format, dataset_name=dataset_name)
            img = self.render_canonical(mesh, viewpoint=viewpoint)
            rendered_images.append(img)
            print(f"  ✓ Rendered {Path(mesh_path).name}")

        # Create composite
        composite = self.create_composite(rendered_images, grid_size)

        return composite

    def _get_category_meshes(self, dataset_name, category, num_instances):
        """
        Get mesh file paths for a category from a dataset

        Args:
            dataset_name: Dataset name
            category: Category name
            num_instances: Number of instances to retrieve

        Returns:
            List of mesh file paths
        """
        base_path = Path(__file__).parent.parent.parent / "data"

        if dataset_name.lower() == "housecat6d":
            # HouseCat6D: data/HouseCat6D/obj_models_small_size_final/category/*.obj
            mesh_dir = base_path / "HouseCat6D" / "obj_models_small_size_final" / category
            if mesh_dir.exists():
                mesh_files = sorted(glob.glob(str(mesh_dir / "*.obj")))
                # Select diverse instances (evenly spaced)
                step = max(1, len(mesh_files) // num_instances)
                return mesh_files[::step][:num_instances]

        elif dataset_name.lower() == "nocs":
            # NOCS: data/nocs/obj_models/real_test/category/*.obj
            mesh_dir = base_path / "nocs" / "obj_models" / "real_test" / category
            if mesh_dir.exists():
                mesh_files = sorted(glob.glob(str(mesh_dir / "*.obj")))
                return mesh_files[:num_instances]

        elif dataset_name.lower() == "tyol":
            # TYOL: Map category to object IDs via config
            config_path = base_path.parent / "configs" / "datasets" / "tyol.json"
            if not config_path.exists():
                return []

            with open(config_path, 'r') as f:
                config = json.load(f)

            # Get category mapping (list index = object ID - 1)
            categories = config.get('objects', {}).get('categories', [])

            # Find all object IDs that match this category
            matching_ids = []
            for i, cat in enumerate(categories):
                if cat == category:
                    matching_ids.append(i + 1)  # Object IDs are 1-indexed

            if not matching_ids:
                return []

            # Get mesh files for matching object IDs
            mesh_dir = base_path / "tyol" / "tyol_models" / "models"
            mesh_files = []
            for obj_id in matching_ids[:num_instances]:
                mesh_path = mesh_dir / f"obj_{obj_id:06d}.ply"
                if mesh_path.exists():
                    mesh_files.append(str(mesh_path))

            return mesh_files

        return []

    def _get_dataset_categories(self, dataset_name):
        """
        Get list of categories for a dataset from its config

        Args:
            dataset_name: Dataset name

        Returns:
            List of category names
        """
        # Handle dataset name mapping (nocs -> real275)
        config_name_map = {
            'nocs': 'real275',
            'real275': 'real275',
            'housecat6d': 'housecat6d',
            'tyol': 'tyol'
        }

        config_name = config_name_map.get(dataset_name.lower(), dataset_name.lower())
        config_path = Path(__file__).parent.parent.parent / "configs" / "datasets" / f"{config_name}.json"

        if not config_path.exists():
            print(f"Warning: Config not found at {config_path}")
            return []

        with open(config_path, 'r') as f:
            config = json.load(f)

        # Extract categories from config (handle different formats)
        if 'categories' in config:
            return config['categories']
        elif 'category_names' in config:
            return config['category_names']
        elif 'objects' in config and 'categories' in config['objects']:
            # For TYOL format - get unique categories
            return list(set(config['objects']['categories']))

        return []


def render_all_categories(dataset_name, output_dir=None):
    """
    Render canonical composites for all categories in a dataset

    Args:
        dataset_name: Dataset name (HouseCat6D, nocs, tyol)
        output_dir: Output directory (default: data/<dataset>/canonical_composites/)
    """
    # Setup output directory
    if output_dir is None:
        base_path = Path(__file__).parent.parent.parent / "data"
        output_dir = base_path / dataset_name / "canonical_composites"
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize renderer
    renderer = CanonicalRenderer(resolution=512)

    # Get categories
    categories = renderer._get_dataset_categories(dataset_name)

    if not categories:
        print(f"No categories found for {dataset_name}. Check dataset config.")
        return

    print(f"\n{'='*60}")
    print(f"Rendering canonical composites for {dataset_name}")
    print(f"Categories: {', '.join(categories)}")
    print(f"Output: {output_dir}")
    print(f"{'='*60}\n")

    # Render each category
    success_count = 0
    for category in categories:
        try:
            composite = renderer.render_category_composite(dataset_name, category)

            # Save composite
            output_path = output_dir / f"{category}.png"
            composite.save(output_path)
            print(f"✓ Saved composite for '{category}' to {output_path}\n")
            success_count += 1

        except Exception as e:
            print(f"✗ Failed to render '{category}': {e}\n")

    print(f"\n{'='*60}")
    print(f"Completed: {success_count}/{len(categories)} categories rendered successfully")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Render canonical composite images for VLM-based part label generation"
    )
    parser.add_argument(
        '--dataset',
        type=str,
        choices=['HouseCat6D', 'nocs', 'real275', 'tyol'],
        help='Dataset name (use "nocs" or "real275" for NOCS dataset)'
    )
    parser.add_argument(
        '--category',
        type=str,
        help='Category to render (e.g., bottle, cup)'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Render all categories in the dataset'
    )
    parser.add_argument(
        '--all-datasets',
        action='store_true',
        help='Render all categories for all datasets'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        help='Output directory (default: data/<dataset>/canonical_composites/)'
    )
    parser.add_argument(
        '--num-instances',
        type=int,
        default=6,
        help='Number of instances to include in composite (default: 6)'
    )
    parser.add_argument(
        '--resolution',
        type=int,
        default=512,
        help='Resolution per instance in pixels (default: 512)'
    )

    args = parser.parse_args()

    # Validate arguments
    if args.all_datasets:
        # Render all datasets
        for dataset in ['HouseCat6D', 'nocs', 'tyol']:
            render_all_categories(dataset, args.output_dir)
    elif args.dataset:
        if args.all:
            # Render all categories in dataset
            render_all_categories(args.dataset, args.output_dir)
        elif args.category:
            # Render single category
            renderer = CanonicalRenderer(resolution=args.resolution)

            try:
                composite = renderer.render_category_composite(
                    args.dataset,
                    args.category,
                    num_instances=args.num_instances
                )

                # Setup output path
                if args.output_dir:
                    output_dir = Path(args.output_dir)
                else:
                    base_path = Path(__file__).parent.parent.parent / "data"
                    output_dir = base_path / args.dataset / "canonical_composites"

                output_dir.mkdir(parents=True, exist_ok=True)
                output_path = output_dir / f"{args.category}.png"

                composite.save(output_path)
                print(f"✓ Saved composite to {output_path}")

            except Exception as e:
                print(f"✗ Error: {e}")
                return 1
        else:
            print("Error: Must specify --category or --all")
            return 1
    else:
        print("Error: Must specify --dataset or --all-datasets")
        parser.print_help()
        return 1

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
