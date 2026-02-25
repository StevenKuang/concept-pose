"""
Semantic Label Loading Utilities
=================================

Provides flexible label loading with automatic generation via Partonomy.

Features:
- Backward compatible with manual labels from configs
- Auto-generates category-level part labels using LLM (Partonomy)
- Caches generated labels in parts.json
- Supports both HouseCat6D and Real275 naming conventions
"""

import os
from typing import List, Dict, Optional, Callable
from pathlib import Path


def extract_category_housecat(object_name: str) -> str:
    """
    Extract category from HouseCat6D object name.

    Examples:
        'bottle-with-label-4' → 'bottle'
        'shoe-aqua_cyan_right' → 'shoe'
        'cup-red-plastic' → 'cup'
    """
    return object_name.split('-')[0]


def extract_category_real275(object_name: str) -> str:
    """
    Extract category from Real275/NOCS object name.

    Examples:
        'bottle_red_stanford_norm' → 'bottle'
        'bowl_white_small_norm' → 'bowl'
        'camera_canon_len_norm' → 'camera'
    """
    return object_name.split('_')[0]


def extract_category_auto(object_name: str) -> str:
    """
    Auto-detect naming convention and extract category.

    Tries both HouseCat6D (hyphen) and Real275 (underscore) formats.
    """
    if '-' in object_name:
        return extract_category_housecat(object_name)
    elif '_' in object_name:
        return extract_category_real275(object_name)
    else:
        # Single-word object name is the category itself
        return object_name


def load_semantic_labels(
    object_names: List[str],
    config: Optional[Dict] = None,
    num_labels: int = 20,
    parts_json: Optional[str] = None,
    category_extractor: Optional[Callable] = None,
    gemini_api_key: Optional[str] = None
) -> Dict[str, List[str]]:
    """
    Load semantic part labels for objects with automatic generation fallback.

    Priority:
    1. If config has 'manual_labels' dict → use those (backward compatibility)
    2. Otherwise → auto-generate using Partonomy per category

    Args:
        object_names: List of object identifiers (e.g., ['bottle_red_stanford_norm', ...])
        config: Optional config dict that may contain 'manual_labels'
        num_labels: Number of part labels to generate per category (default: 20)
        parts_json: Optional path to parts.json cache file
        category_extractor: Optional function to extract category from object name
        gemini_api_key: Optional Gemini API key (or set GEMINI_API_KEY env var)

    Returns:
        Dictionary mapping object_name → list of part labels

    Examples:
        >>> # With manual labels
        >>> config = {'manual_labels': {'bottle-1': ['neck', 'body']}}
        >>> labels = load_semantic_labels(['bottle-1'], config)
        >>> labels['bottle-1']
        ['neck', 'body']

        >>> # Auto-generate from Partonomy
        >>> labels = load_semantic_labels(['bottle_red_norm', 'bottle_blue_norm'])
        >>> labels['bottle_red_norm']  # Same labels for all bottles
        ['neck', 'body', 'cap', 'base', 'label', ...]
    """
    # Priority 1: Use manual labels if provided in config
    if config and 'manual_labels' in config:
        manual_labels = config['manual_labels']

        # Check if it's a dict (per-object labels) or list (single label set)
        if isinstance(manual_labels, dict):
            return manual_labels
        elif isinstance(manual_labels, list):
            # Single label set for all objects
            return {obj: manual_labels for obj in object_names}
        else:
            raise ValueError(
                f"Invalid manual_labels format: {type(manual_labels)}. "
                f"Expected dict or list."
            )

    # Priority 2: Auto-generate using Partonomy
    print("\n" + "="*60)
    print("Auto-generating semantic labels using Partonomy")
    print("="*60)

    # Use provided extractor or auto-detect
    extractor = category_extractor or extract_category_auto

    # Extract unique categories from object names
    categories = {}
    for obj_name in object_names:
        category = extractor(obj_name)
        if category not in categories:
            categories[category] = []
        categories[category].append(obj_name)

    print(f"\nDetected {len(categories)} categories:")
    for cat, objs in categories.items():
        print(f"  {cat}: {len(objs)} objects")

    # Determine parts.json path
    if parts_json is None:
        # Default to partonomy module location
        module_dir = Path(__file__).parent.parent / "partonomy"
        parts_json = str(module_dir / "parts.json")

    # Generate labels for each category using Partonomy
    from concept_pose.partonomy import Partonomy

    category_labels = {}
    for category in categories.keys():
        print(f"\n{'='*50}")
        print(f"Loading labels for category: '{category}'")
        print(f"{'='*50}")

        # Query=False will use cache if available, otherwise query LLM
        print(f"  Calling Partonomy(category_label='{category}', num_labels={num_labels}, query=False)")

        try:
            partonomy = Partonomy(
                category_label=category,
                num_labels=num_labels,
                query=False,
                parts_json=parts_json,
                gemini_api_key=gemini_api_key
            )
        except Exception as e:
            # Handle Gemini API errors (503 overload, missing key, etc.)
            error_msg = str(e)

            if "503" in error_msg or "overloaded" in error_msg.lower():
                print(f"  ⚠️  Gemini API is overloaded (503). Cannot auto-generate labels.")
            elif "api key" in error_msg.lower() or "GEMINI_API_KEY" in error_msg:
                print(f"  ⚠️  Gemini API key not configured. Cannot auto-generate labels.")
            elif "No candidate labels found" in error_msg:
                print(f"  ⚠️  Category '{category}' not found in cache. Cannot auto-generate labels.")
            else:
                print(f"  ⚠️  Partonomy error: {e}")

            print(f"  → Will prompt for manual input...")

            # Set empty labels to trigger interactive prompt
            partonomy = type('obj', (object,), {
                'part_labels': None,
                'candidate_labels': None
            })()

        # Use curated part_labels if available, otherwise use all candidate_labels
        if partonomy.part_labels:
            labels = partonomy.part_labels
            print(f"  ✓ Found {len(labels)} curated part labels from cache")
        elif partonomy.candidate_labels:
            labels = partonomy.candidate_labels
            print(f"  ✓ Found {len(labels)} candidate labels from cache")
        else:
            labels = None

        # Check if we need to re-query LLM due to insufficient labels
        if labels is not None and len(labels) < num_labels:
            print(f"  ⚠️  Cached labels ({len(labels)}) < requested ({num_labels})")
            print(f"  → Re-querying LLM for {num_labels} labels...")
            try:
                partonomy = Partonomy(
                    category_label=category,
                    num_labels=num_labels,
                    query=True,  # Force re-query
                    parts_json=parts_json,
                    gemini_api_key=gemini_api_key
                )
                labels = partonomy.part_labels or partonomy.candidate_labels
                print(f"  ✓ Re-queried successfully: {len(labels)} labels")
            except Exception as e:
                print(f"  ❌ Re-query failed: {e}")
                print(f"  → Will prompt for manual input...")
                labels = None

        if labels is None:
            # No labels available - prompt user interactively
            print(f"\n{'='*70}")
            print(f"❌ No semantic labels found for category: '{category}'")
            print(f"{'='*70}\n")
            print(f"Semantic labels are required for part-based pose estimation.")
            print(f"\nYou can either:")
            print(f"  1. Paste a Python list of semantic part labels (e.g., ['neck', 'body', 'cap'])")
            print(f"  2. Press ENTER to quit and set up Gemini API key instead")
            print(f"\nExample labels for '{category}':")

            # Provide helpful examples for common categories
            examples = {
                'bottle': "['neck', 'body', 'cap', 'base', 'label']",
                'mug': "['handle', 'body', 'rim', 'base']",
                'cup': "['handle', 'body', 'rim', 'base']",
                'magazine': "['cover', 'spine', 'pages', 'binding', 'title']",
                'book': "['cover', 'spine', 'pages', 'binding']",
                'remote': "['buttons', 'body', 'antenna', 'battery_cover']",
                'camera': "['lens', 'body', 'viewfinder', 'flash', 'buttons']",
                'laptop': "['screen', 'keyboard', 'touchpad', 'base', 'hinge']",
                'shoe': "['sole', 'upper', 'laces', 'tongue', 'heel']",
                'teapot': "['spout', 'body', 'handle', 'lid', 'base']",
                'can': "['body', 'top', 'bottom', 'label']",
                'bowl': "['rim', 'body', 'base', 'interior']",
                'glass': "['rim', 'body', 'base', 'stem']",
            }

            if category in examples:
                print(f"  Suggestion: {examples[category]}")
            else:
                print(f"  Example format: ['part1', 'part2', 'part3', ...]")

            print(f"\nPaste your labels as a Python list (or press ENTER to quit):")
            user_input = input("> ").strip()

            if not user_input:
                print("\n❌ No labels provided. Exiting.")
                print("\nTo fix this, you can:")
                print(f"  1. Set GEMINI_API_KEY environment variable to auto-generate labels")
                print(f"  2. Copy parts.json from another machine")
                print(f"  3. Re-run and paste labels when prompted")
                sys.exit(1)

            # Try to parse the input as a Python list
            try:
                import ast
                labels = ast.literal_eval(user_input)

                if not isinstance(labels, list):
                    raise ValueError("Input must be a list")
                if not all(isinstance(x, str) for x in labels):
                    raise ValueError("All labels must be strings")
                if len(labels) == 0:
                    raise ValueError("Label list cannot be empty")

                print(f"\n✓ Parsed {len(labels)} labels: {labels[:5]}...")

                # Save to parts.json for future use
                print(f"\n💾 Saving labels to cache: {parts_json}")

                # Load existing cache
                import json
                try:
                    with open(parts_json, 'r') as f:
                        cache = json.load(f)
                        if not isinstance(cache, list):
                            cache = []
                except (FileNotFoundError, json.JSONDecodeError):
                    cache = []

                # Add new entry in the same format as LLM-generated entries
                new_entry = {
                    'category_label': category,  # Use "category_label" to match Partonomy format
                    'candidate_labels': [
                        {
                            'llm_model': 'manual',  # Mark as manually entered
                            'response': labels,
                            'part_labels': labels  # Use same labels for both
                        }
                    ]
                }

                # Remove old entry for this category if exists
                cache = [entry for entry in cache if entry.get('category_label') != category]
                cache.append(new_entry)

                # Save updated cache
                with open(parts_json, 'w') as f:
                    json.dump(cache, f, indent=2)

                print(f"✓ Labels saved to cache and will be reused in future runs")

            except Exception as e:
                print(f"\n❌ Invalid input: {e}")
                print(f"Expected format: ['part1', 'part2', 'part3', ...]")
                print(f"Example: ['neck', 'body', 'cap', 'base', 'label']")
                sys.exit(1)

        # Truncate if we have more labels than requested
        if len(labels) > num_labels:
            print(f"  ⚠️  Cached labels ({len(labels)}) > requested ({num_labels})")
            print(f"  → Truncating to first {num_labels} labels")
            labels = labels[:num_labels]
            print(f"  ✓ Using: {labels[:5]}..." if len(labels) > 5 else f"  ✓ Using: {labels}")

        category_labels[category] = labels

    # Map each object to its category's labels
    object_label_map = {}
    for obj_name in object_names:
        category = extractor(obj_name)
        object_label_map[obj_name] = category_labels[category]

    print("\n" + "="*60)
    print(f"Loaded semantic labels for {len(object_names)} objects")
    print("="*60 + "\n")

    return object_label_map


def get_labels_for_category(
    category: str,
    num_labels: int = 20,
    parts_json: Optional[str] = None,
    gemini_api_key: Optional[str] = None
) -> List[str]:
    """
    Get semantic part labels for a single category.

    Convenience function for single-category use cases.

    Args:
        category: Category name (e.g., 'bottle', 'cup', 'shoe')
        num_labels: Number of labels to generate
        parts_json: Optional path to parts.json
        gemini_api_key: Optional Gemini API key

    Returns:
        List of semantic part labels
    """
    from concept_pose.partonomy import Partonomy

    if parts_json is None:
        module_dir = Path(__file__).parent.parent / "partonomy"
        parts_json = str(module_dir / "parts.json")

    # First try to load from cache
    partonomy = Partonomy(
        category_label=category,
        num_labels=num_labels,
        query=False,
        parts_json=parts_json,
        gemini_api_key=gemini_api_key
    )

    labels = partonomy.part_labels or partonomy.candidate_labels

    # Re-query if cached labels are insufficient
    if labels is not None and len(labels) < num_labels:
        print(f"  Cached labels ({len(labels)}) < requested ({num_labels}), re-querying LLM...")
        try:
            partonomy = Partonomy(
                category_label=category,
                num_labels=num_labels,
                query=True,  # Force re-query
                parts_json=parts_json,
                gemini_api_key=gemini_api_key
            )
            labels = partonomy.part_labels or partonomy.candidate_labels
            print(f"  Re-queried successfully: {len(labels)} labels")
        except Exception as e:
            print(f"  Re-query failed: {e}, using cached {len(labels)} labels")

    return labels
