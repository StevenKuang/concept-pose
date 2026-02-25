"""
Partonomy - Semantic Part Label Management
===========================================

Manages semantic part labels for object categories using:
1. Pre-defined labels stored in parts.json
2. LLM query (Gemini API) to generate new labels if needed

REFACTORED: Removed label_eval dependency for simplified usage.
API key should be set via environment variable for security.

ABLATION: Support for geometry-focused prompts with canonical pose images
"""

import ast
import json
import os
from datetime import datetime

class Partonomy:
    # Geometry-focused prompt template for ablation study
    # Emphasizes shape-based, texture-agnostic part detection
    GEOMETRY_PROMPT_TEMPLATE = """Identify and localize semantically meaningful and geometrically stable regions on {category}-shaped objects that can serve as repeatable reference concepts for category-level 6D pose estimation.

Focus on shape-derived and structure-based cues that are consistent across different {category} designs, materials, and textures (e.g., plastic, glass, metal). Do not rely on color, text, or label appearance.

The goal is to generate concept maps or key regions that are interpretable, distinct, and robust under viewpoint, lighting, and instance variation — allowing the model to describe or anchor poses using language-grounded geometry.

Propose {num_labels} spatial or geometric concepts that can be recognized and localized by vision-language models such as SigLIP or CLIP, described in concise, natural-language terms.

Each concept should be:
– Category-consistent: present or inferable in most {category}s.
– Shape-descriptive: related to form, silhouette, or topology.
– Pose-relevant: useful for alignment or orientation inference.
– Detectable by VLMs: phrased with interpretable language tokens.

Avoid material, texture, or brand-dependent attributes.

Please provide exactly {num_labels} labels as a Python list, no additional explanations."""

    # Affordance-aware geometry prompt template (alternative)
    # Combines geometric features with functional affordances and descriptive adjectives
    AFFORDANCE_GEOMETRY_PROMPT_TEMPLATE = """Identify and localize semantically meaningful and geometrically stable regions on {category}-shaped objects that can serve as repeatable reference concepts for category-level 6D pose estimation.

Focus on shape-derived features combined with their functional affordances and physical properties. Include geometric descriptors, affordance-based phrases, and physically grounded adjectives that describe form, interaction possibilities, and spatial characteristics.

Allowed descriptive terms:
– Geometric adjectives: curved, flat, rounded, angular, tapered, wide, narrow, thick, thin, protruding, recessed, smooth, ridged
– Affordance-based: graspable, sharp, blunt, hollow, stable, openable, pour-able, cuttable, stackable, rollable
– Spatial descriptors: top, bottom, front, back, upper, lower, central, edge, corner, rim, base

The goal is to generate concept maps that combine geometric structure with functional understanding, making them more interpretable and grounded for vision-language models while remaining consistent across different {category} designs, materials (plastic, glass, metal, ceramic), and textures.

Propose {num_labels} spatially, geometrically, and functionally grounded concepts that can be recognized by VLMs such as SigLIP or CLIP, described in natural language.

CRITICAL CONSTRAINTS:
1. CONCISENESS: Each label must be 2-3 words maximum (e.g., "curved handle", "sharp blade", "flat base")
2. ORTHOGONALITY: Labels must be maximally distinct and non-overlapping. Avoid redundant or semantically similar concepts.
   - BAD: "handle", "grip", "graspable part" (too similar)
   - GOOD: "curved handle", "flat base", "narrow neck" (distinct regions)
3. Each label should describe a DIFFERENT spatial region or functional aspect of the {category}

Each concept should be:
– Geometrically descriptive: related to form, silhouette, topology, and measurable physical properties
– Affordance-aware: consider how the shape enables function, manipulation, and interaction
– Category-consistent: present or inferable in most {category} instances
– Pose-relevant: useful for alignment, orientation, and spatial reasoning
– VLM-detectable: phrased with interpretable language including descriptive adjectives
– Maximally distinct: no semantic overlap with other selected labels

Avoid color, brand names, logos, and specific material types when not relevant to geometry or affordance.

Please provide exactly {num_labels} labels as a Python list, no additional explanations. Ensure each label is 2-3 words and semantically orthogonal to all others."""
    def __init__(self, category_label, num_labels=20, query=False, parts_json=None, gemini_api_key=None,
                 use_geometry_prompt=False, use_affordance_prompt=False, canonical_image_path=None, dataset_name=None):
        self.category_label = category_label
        self.num_labels = num_labels
        self.candidate_labels = None
        self.part_labels = None
        self.llm_model = "gemini-2.5-pro"

        # ABLATION: Geometry-focused prompt and canonical image support
        self.use_geometry_prompt = use_geometry_prompt
        self.use_affordance_prompt = use_affordance_prompt
        self.canonical_image_path = canonical_image_path
        self.dataset_name = dataset_name

        # REFACTORED: API key from parameter or environment variable
        self.gemini_api_key = gemini_api_key or os.environ.get('GEMINI_API_KEY')

        # REFACTORED: Default path updated to refactored package location
        if parts_json is None:
            # Try to find parts.json relative to this module
            module_dir = os.path.dirname(os.path.abspath(__file__))
            self.parts_json = os.path.join(module_dir, "parts.json")
        else:
            self.parts_json = parts_json

        if not os.path.exists(self.parts_json):
            with open(self.parts_json, "w") as f:
                json.dump([], f)
                print(f"Created {os.path.abspath(self.parts_json)}")

        if not query:
            try:
                self.candidate_labels, self.part_labels = self.read_from_json()
            except Exception as e:
                # print(e)
                print(f"No candidate labels found for category {self.category_label} in {self.parts_json}, querying LLM...")
                self.candidate_labels = self.query_llm()
                self.write_to_json()
                # Read back to properly set part_labels (None if not curated)
                self.candidate_labels, self.part_labels = self.read_from_json()
        else:
            self.candidate_labels = self.query_llm()
            self.write_to_json()
            # Read back to properly set part_labels (None if not curated)
            self.candidate_labels, self.part_labels = self.read_from_json()

    def response_to_list(self, response):
        text = response.text
        list_str = text[text.find("[") : text.find("]") + 1]
        return ast.literal_eval(list_str)
    
    def read_from_json(self):
        with open(self.parts_json, "r") as f:
            all_categories = json.load(f)

        # Initialize to None before loop
        candidate_labels = None
        part_labels = None

        for category in all_categories:
            if category["category_label"] == self.category_label:
                candidate_labels = category["candidate_labels"][0]["response"]
                if "part_labels" in category["candidate_labels"][0]:
                    part_labels = category["candidate_labels"][0]["part_labels"]
                else:
                    part_labels = None
                break

        if candidate_labels is None:
            raise ValueError(f"No candidate labels found for category {self.category_label}")
        return candidate_labels, part_labels

    def write_to_json(self):
        with open(self.parts_json, "r") as f:
            all_categories = json.load(f)

        # Build metadata entry
        entry = {
            "llm_model": self.llm_model,
            "response": self.candidate_labels,
            "timestamp": datetime.now().isoformat(),
        }

        # Add ablation metadata if using special prompts
        if self.use_affordance_prompt:
            entry["use_affordance_prompt"] = True
            if self.canonical_image_path:
                entry["canonical_image_path"] = self.canonical_image_path
            if self.dataset_name:
                entry["dataset_name"] = self.dataset_name
        elif self.use_geometry_prompt:
            entry["use_geometry_prompt"] = True
            if self.canonical_image_path:
                entry["canonical_image_path"] = self.canonical_image_path
            if self.dataset_name:
                entry["dataset_name"] = self.dataset_name

        found = False
        for category in all_categories:
            if category["category_label"] == self.category_label:
                found = True
                category["candidate_labels"].insert(0, entry)  # add to the front
                break
        if not found:
            all_categories.append({
                "category_label": self.category_label,
                "candidate_labels": [entry]
            })

        with open(self.parts_json, "w") as f:
            json.dump(all_categories, f, indent=2)
        return True

    def query_llm(self, contents=None, image_path=None):
        """
        Query Gemini LLM API to generate semantic part labels

        Args:
            contents: Custom text prompt (optional)
            image_path: Path to canonical composite image (optional, overrides self.canonical_image_path)

        Returns:
            List of part label strings
        """
        if not self.gemini_api_key:
            raise ValueError(
                "Gemini API key not provided. "
                "Set GEMINI_API_KEY environment variable or pass gemini_api_key parameter."
            )

        from google import genai
        from google.genai import types

        client = genai.Client(api_key=self.gemini_api_key)

        # Determine which image to use (parameter overrides instance variable)
        img_path = image_path or self.canonical_image_path

        # Build content for multimodal input
        content_parts = []

        # Add image if available
        if img_path and os.path.exists(img_path):
            try:
                from PIL import Image
                image = Image.open(img_path)
                # Directly add PIL Image object to content_parts
                content_parts.append(image)
                print(f"Using canonical image: {img_path}")
            except Exception as e:
                print(f"Warning: Failed to load image {img_path}: {e}")
                print("Falling back to text-only query")

        # Add text prompt
        if not contents:
            if self.use_affordance_prompt:
                # Use affordance-aware geometry prompt template
                contents = self.AFFORDANCE_GEOMETRY_PROMPT_TEMPLATE.format(
                    category=self.category_label,
                    num_labels=self.num_labels
                )
            elif self.use_geometry_prompt:
                # Use geometry-focused prompt template
                contents = self.GEOMETRY_PROMPT_TEMPLATE.format(
                    category=self.category_label,
                    num_labels=self.num_labels
                )
            else:
                # Use original prompt
                contents = f"Give me {self.num_labels} labels that describes different semantic parts of a {self.category_label}. I will be using these labels for part segmentation, please make sure they are generalizable to different instances within the same category {self.category_label}, semantically orthogonal to each other, and must be visible from at least one external viewpoint. Also, please use the most common names for the parts and do not use positional descriptions. Please just give me all the labels as a python list, no additional explanations please. "

        content_parts.append(contents)

        # Query LLM with multimodal content
        response = client.models.generate_content(
            model=self.llm_model,
            contents=content_parts
        )
        return self.response_to_list(response)

    def print_evaluation_results(self, part_labels, scores, conflict_groups_detected):
        print("\n--- Evaluation Results ---")
        print(f"\nFinal Selected Labels: {part_labels}")
        
        print("\nConflict Groups Detected:")
        for group in conflict_groups_detected:
            print(f"- {group}")
            
        print("\nDetailed Scores:")
        # Pretty print the scores, sorted by the new quality_score
        for label, label_scores in sorted(scores.items(), key=lambda item: item[1]['quality_score'], reverse=True):
            print(f"Label: {label}")
            for score_name, value in label_scores.items():
                print(f"  - {score_name:<15}: {value:.4f}")
            print("-" * 20)
    
    def select_labels(self, scores, part_labels, top_k=None, min_quality_score=0.85):
        # select the labels that have a quality score greater than min_quality_score
        # and sort them by the quality score
        part_labels = sorted(part_labels, key=lambda x: scores[x]['quality_score'], reverse=True)
        if top_k is None:
            return part_labels
        else:
            return part_labels[:top_k]
    
    def write_part_labels_to_json(self, part_labels):
        with open(self.parts_json, "r") as f:
            all_categories = json.load(f)
        for category in all_categories:
            if category["category_label"] == self.category_label:
                if "part_labels" in category["candidate_labels"][0]:
                    # if exist part_labels, copy the current dict, insert to the front and overwrite the part_labels
                    new_dict = category["candidate_labels"][0].copy()
                    new_dict["part_labels"] = part_labels
                    category["candidate_labels"].insert(0, new_dict)
                else:
                    # if not exist, simply add the part_labels to the existing dict
                    category["candidate_labels"][0]["part_labels"] = part_labels
                break
        with open(self.parts_json, "w") as f:
            json.dump(all_categories, f, indent=2)
        return True

    @classmethod
    def with_geometry_prompt(cls, category_label, dataset_name, num_labels=15, query=True, **kwargs):
        """
        Factory method to create Partonomy instance with geometry-focused prompt + canonical image

        This is a convenience method for ablation testing. It automatically locates
        the canonical composite image for the given category and dataset.

        Args:
            category_label: Object category (e.g., "bottle", "cup")
            dataset_name: Dataset name (e.g., "HouseCat6D", "nocs", "tyol")
            num_labels: Number of labels to generate (default: 15)
            query: Whether to force LLM query (default: True for ablation)
            **kwargs: Additional parameters for Partonomy.__init__

        Returns:
            Partonomy instance configured for geometry-focused ablation

        Example:
            >>> # Baseline (text-only, original prompt)
            >>> p_baseline = Partonomy("bottle", num_labels=15)
            >>>
            >>> # Ablation (geometry prompt + canonical image)
            >>> p_ablation = Partonomy.with_geometry_prompt("bottle", "HouseCat6D", num_labels=15)
        """
        # Map dataset name to actual directory name (handle case sensitivity)
        dataset_dir_map = {
            'housecat6d': 'HouseCat6D',
            'tyol': 'tyol',
            'nocs': 'nocs',
            'real275': 'nocs'
        }

        # Construct path to canonical composite image
        module_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(module_dir, "..", "..", "data")
        actual_dataset_dir = dataset_dir_map.get(dataset_name.lower(), dataset_name)
        image_path = os.path.join(data_dir, actual_dataset_dir, "canonical_composites", f"{category_label}.png")

        # Check if image exists
        if not os.path.exists(image_path):
            print(f"Warning: Canonical image not found at {image_path}")
            print(f"Run: python -m concept_pose.partonomy.canonical_renderer --dataset {dataset_name} --category {category_label}")
            image_path = None

        return cls(
            category_label=category_label,
            num_labels=num_labels,
            query=query,
            use_geometry_prompt=True,
            canonical_image_path=image_path,
            dataset_name=dataset_name,
            **kwargs
        )

    @classmethod
    def with_affordance_prompt(cls, category_label, dataset_name, num_labels=15, query=True, **kwargs):
        """
        Factory method to create Partonomy instance with affordance-aware prompt + canonical image

        This is a convenience method for ablation testing. It automatically locates
        the canonical composite image for the given category and dataset, and uses
        the affordance-aware prompt that combines geometric features with functional
        affordances and descriptive adjectives.

        Args:
            category_label: Object category (e.g., "knife", "cup", "bottle")
            dataset_name: Dataset name (e.g., "HouseCat6D", "nocs", "tyol")
            num_labels: Number of labels to generate (default: 15)
            query: Whether to force LLM query (default: True for ablation)
            **kwargs: Additional parameters for Partonomy.__init__

        Returns:
            Partonomy instance configured for affordance-aware ablation

        Example:
            >>> # Baseline (text-only, original prompt)
            >>> p_baseline = Partonomy("knife", num_labels=20)
            >>>
            >>> # Geometry ablation (shape-focused)
            >>> p_geometry = Partonomy.with_geometry_prompt("knife", "HouseCat6D", num_labels=20)
            >>>
            >>> # Affordance ablation (shape + function + adjectives)
            >>> p_affordance = Partonomy.with_affordance_prompt("knife", "HouseCat6D", num_labels=20)
        """
        # Map dataset name to actual directory name (handle case sensitivity)
        dataset_dir_map = {
            'housecat6d': 'HouseCat6D',
            'tyol': 'tyol',
            'nocs': 'nocs',
            'real275': 'nocs'
        }

        # Construct path to canonical composite image
        module_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(module_dir, "..", "..", "data")
        actual_dataset_dir = dataset_dir_map.get(dataset_name.lower(), dataset_name)
        image_path = os.path.join(data_dir, actual_dataset_dir, "canonical_composites", f"{category_label}.png")

        # Check if image exists
        if not os.path.exists(image_path):
            print(f"Warning: Canonical image not found at {image_path}")
            print(f"Run: python -m concept_pose.partonomy.canonical_renderer --dataset {dataset_name} --category {category_label}")
            image_path = None

        return cls(
            category_label=category_label,
            num_labels=num_labels,
            query=query,
            use_affordance_prompt=True,
            canonical_image_path=image_path,
            dataset_name=dataset_name,
            **kwargs
        )


if __name__ == "__main__":
    # Example usage
    import sys

    if len(sys.argv) < 2:
        print("Usage: python partonomy.py <category_label> [--query]")
        print("\nExample:")
        print("  python partonomy.py cup")
        print("  python partonomy.py shoe --query  # Force LLM query")
        sys.exit(1)

    category = sys.argv[1]
    query = "--query" in sys.argv

    # Read from JSON (or query if not found)
    p = Partonomy(category_label=category, query=query)

    print(f"\nCategory: {p.category_label}")
    print(f"Candidate labels: {p.candidate_labels}")
    if p.part_labels:
        print(f"Selected part labels: {p.part_labels}")
    else:
        print("No curated part labels yet (use label evaluation to select)")

    # NOTE: Label evaluation requires additional dependencies (Objectron dataset, Grounded-SAM2, etc.)
    # For advanced usage, use the label_eval module from the main vlm_pose codebase
