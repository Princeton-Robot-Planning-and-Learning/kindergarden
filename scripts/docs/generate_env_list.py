"""Generate a markdown-compatible list of all KinDER environments.

Usage:
  python generate_env_list.py
"""

from __future__ import annotations

import kinder


def generate_env_list_markdown() -> str:
    """Generate a markdown table of all environments with categories and examples.

    Returns:
        Markdown-formatted table as a string
    """
    kinder.register_all_environments()

    env_classes = kinder.get_env_classes()
    env_categories = kinder.get_env_categories()

    # Build a list of (class_name, category, example_id) tuples
    # Sort categories in specific order
    category_order = ["Kinematic2D", "Kinematic3D", "Dynamic2D", "Dynamic3D"]
    env_list = []
    for category in category_order:
        if category not in env_categories:
            continue
        for class_name in sorted(env_categories[category]):
            variants = env_classes[class_name]["variants"]
            # Pick a representative variant (middle one if multiple, or the only one)
            example_variant = variants[len(variants) // 2]
            env_list.append((class_name, category, example_variant))

    # Generate markdown table
    md = "| Environment | Category | Example Environment ID |\n"
    md += "|---|---|---|\n"

    for class_name, category, example_id in env_list:
        md += f"| {class_name} | {category} | `{example_id}` |\n"

    return md


def main() -> None:
    """Print the markdown table to stdout."""
    markdown = generate_env_list_markdown()
    print(markdown)


if __name__ == "__main__":
    main()
