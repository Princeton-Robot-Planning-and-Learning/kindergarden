"""Automatically create markdown documents for every registered environment.

Usage:
  python generate_env_docs.py                    # Generate docs for changed environments
  python generate_env_docs.py --force            # Force regenerate all environments
  python generate_env_docs.py --env Motion2D     # Generate docs for specific environment
  python generate_env_docs.py --env_category Kinematic2D
    # Generate docs for specific category
"""

from __future__ import annotations

import argparse
import inspect
import json
import re
import subprocess
from pathlib import Path

import gymnasium
import imageio.v2 as iio

import kinder
from kinder.gif_utils import optimize_gif

REPO_ROOT = Path(__file__).parents[2]
OUTPUT_DIR = REPO_ROOT / "docs" / "envs"

# Dynamically read all folder names from the tasks directory
DYNAMIC3D_TASKS_DIR = REPO_ROOT / "src" / "kinder" / "envs" / "dynamic3d" / "tasks"
DYNAMIC3D_ENVS = sorted([d.name for d in DYNAMIC3D_TASKS_DIR.iterdir() if d.is_dir()])


FINE_TIMESTEP_ENVS: dict[str, tuple[int, int]] = {
    "LimbRepositioning3D": (1500, 30),
}

NATURALLY_SORTED_VARIANT_ENVS = {"ConstrainedCupboard3D"}


def natural_sort_key(value: str) -> tuple[tuple[int, int | str], ...]:
    """Return a key that sorts embedded numbers by their numeric value."""
    return tuple(
        (1, int(part)) if part.isdigit() else (0, part.casefold())
        for part in re.split(r"(\d+)", value)
    )


def get_documented_variants(class_name: str, variants: list[str]) -> list[str]:
    """Return variants in the order used by the generated documentation."""
    if class_name in NATURALLY_SORTED_VARIANT_ENVS:
        return sorted(variants, key=natural_sort_key)
    return variants.copy()


def get_random_action_gif_params(class_name: str) -> tuple[int, int]:
    """The number of random actions to take and the render stride, for a class."""
    return FINE_TIMESTEP_ENVS.get(class_name, (25, 1))


def get_changed_files() -> set[Path]:
    """Get the set of files that have changed compared to origin/main."""
    # Get the list of changed files from git diff.
    result = subprocess.run(
        ["git", "diff", "origin/main", "--name-only"],
        capture_output=True,
        text=True,
        check=True,
    )

    if not result.stdout.strip():
        return set()

    # Convert to Path objects.
    changed_files = set()
    for line in result.stdout.strip().split("\n"):
        if line.strip():
            if not line.startswith("src/kinder/"):
                continue
            changed_files.add((REPO_ROOT / line.strip()).resolve())

    return changed_files


def is_env_changed(env: gymnasium.Env, changed_files: set[Path]) -> bool:
    """Check if the environment has changed based on git diff."""
    module_path = Path(inspect.getfile(env.unwrapped.__class__)).resolve()
    return module_path in changed_files


def sanitize_env_id(env_id: str) -> str:
    """Remove unnecessary stuff from the env ID."""
    assert env_id.startswith("kinder/")
    env_id = env_id[len("kinder/") :]
    env_id = env_id.replace("/", "_")
    assert env_id[-3:-1] == "-v"
    return env_id[:-3]


def sanitize_class_name(class_name: str) -> str:
    """Sanitize class name for use in filenames."""
    return class_name.replace("/", "_")


def create_random_action_gif(
    class_name: str,
    env: gymnasium.Env,
    num_actions: int = 25,
    seed: int = 0,
    default_fps: int = 10,
    render_every: int = 1,
) -> tuple[bool, dict[str, float | bool]]:
    """Create a GIF of taking random actions in the environment.

    Args:
        class_name: The environment class name (e.g., "ClutteredStorage2D")
        env: The environment instance to use for generating the GIF
        num_actions: Number of random actions to take
        seed: Random seed
        default_fps: Default FPS if not specified in metadata

    Returns:
        Tuple of (success, stats_dict) where stats_dict contains:
            - total_reward: cumulative reward
            - terminated_successfully: whether episode terminated successfully
            - num_steps: number of steps taken
    """
    try:
        imgs: list = []
        total_reward = 0.0
        terminated_successfully = False
        num_steps = 0

        env.reset(seed=seed)
        env.action_space.seed(seed)
        imgs.append(env.render())

        for step in range(num_actions):
            action = env.action_space.sample()
            _, reward, terminated, truncated, _ = env.step(action)
            total_reward += float(reward)
            num_steps += 1
            if (step + 1) % render_every == 0 or terminated or truncated:
                imgs.append(env.render())

            if terminated or truncated:
                terminated_successfully = terminated
                break

        class_filename = sanitize_class_name(class_name)
        outfile = OUTPUT_DIR / "assets" / "random_action_gifs" / f"{class_filename}.gif"
        fps = env.metadata.get("render_fps", default_fps)
        iio.mimsave(outfile, imgs, fps=fps, loop=0)
        optimize_gif(outfile)

        stats = {
            "total_reward": float(total_reward),
            "terminated_successfully": bool(terminated_successfully),
            "num_steps": int(num_steps),
        }
        return True, stats
    except Exception as e:  # pylint: disable=broad-except
        print(f"    Warning: Failed to create random action GIF for {class_name}: {e}")
        return False, {}


def create_initial_state_gif(
    class_name: str,
    variant_ids: list[str],
    num_resets_per_variant: int = 5,
    seed: int = 0,
    fps: int = 10,
) -> bool:
    """Create a GIF showing initial states across all variants.

    Args:
        class_name: The environment class name (e.g., "ClutteredStorage2D")
        variant_ids: List of all variant IDs for this class
        num_resets_per_variant: Number of resets to show for each variant
        seed: Random seed
        fps: Frames per second for the GIF

    Returns:
        bool: True if successful, False if rendering failed.
    """
    try:
        imgs: list = []
        for variant_id in variant_ids:
            kwargs: dict[str, str | bool] = {"render_mode": "rgb_array"}
            if class_name in DYNAMIC3D_ENVS:
                kwargs["scene_bg"] = True
            env = kinder.make(variant_id, **kwargs)
            for i in range(num_resets_per_variant):
                env.reset(seed=seed + i)
                imgs.append(env.render())
        class_filename = sanitize_class_name(class_name)
        outfile = OUTPUT_DIR / "assets" / "initial_state_gifs" / f"{class_filename}.gif"
        iio.mimsave(outfile, imgs, fps=fps, loop=0)
        optimize_gif(outfile)
        return True
    except Exception as e:  # pylint: disable=broad-except
        print(f"    Warning: Failed to create initial state GIF for {class_name}: {e}")
        return False


def create_variant_initial_state_gif(
    variant_name: str,
    env: gymnasium.Env,
    num_resets: int = 5,
    seed: int = 0,
    fps: int = 10,
) -> bool:
    """Create a GIF showing initial states for a single variant.

    Args:
        variant_name: The sanitized variant name (e.g., "ClutteredStorage2D-b1")
        env: The environment instance for this variant
        num_resets: Number of resets to show
        seed: Random seed
        fps: Frames per second for the GIF

    Returns:
        bool: True if successful, False if rendering failed.
    """
    try:
        imgs: list = []
        for i in range(num_resets):
            env.reset(seed=seed + i)
            imgs.append(env.render())
        outfile = (
            OUTPUT_DIR
            / "assets"
            / "initial_state_gifs"
            / "variants"
            / f"{variant_name}.gif"
        )
        iio.mimsave(outfile, imgs, fps=fps, loop=0)
        optimize_gif(outfile)
        return True
    except Exception as e:  # pylint: disable=broad-except
        print(
            f"    Warning: Failed to create initial state GIF for {variant_name}: {e}"
        )
        return False


def create_variant_random_action_gif(
    variant_name: str,
    env: gymnasium.Env,
    num_actions: int = 25,
    render_every: int = 1,
    seed: int = 0,
    default_fps: int = 10,
) -> tuple[bool, dict[str, float | bool]]:
    """Create a GIF of taking random actions for a single variant.

    Args:
        variant_name: The sanitized variant name (e.g., "ClutteredStorage2D-b1")
        env: The environment instance to use for generating the GIF
        num_actions: Number of random actions to take
        seed: Random seed
        default_fps: Default FPS if not specified in metadata

    Returns:
        Tuple of (success, stats_dict) where stats_dict contains:
            - total_reward: cumulative reward
            - terminated_successfully: whether episode terminated successfully
            - num_steps: number of steps taken
    """
    try:
        imgs: list = []
        total_reward = 0.0
        terminated_successfully = False
        num_steps = 0

        env.reset(seed=seed)
        env.action_space.seed(seed)
        imgs.append(env.render())

        for step in range(num_actions):
            action = env.action_space.sample()
            _, reward, terminated, truncated, _ = env.step(action)
            total_reward += float(reward)
            num_steps += 1
            if (step + 1) % render_every == 0 or terminated or truncated:
                imgs.append(env.render())

            if terminated or truncated:
                terminated_successfully = terminated
                break

        outfile = (
            OUTPUT_DIR
            / "assets"
            / "random_action_gifs"
            / "variants"
            / f"{variant_name}.gif"
        )
        fps = env.metadata.get("render_fps", default_fps)
        iio.mimsave(outfile, imgs, fps=fps, loop=0)
        optimize_gif(outfile)

        stats = {
            "total_reward": float(total_reward),
            "terminated_successfully": bool(terminated_successfully),
            "num_steps": int(num_steps),
        }
        return True, stats
    except Exception as e:  # pylint: disable=broad-except
        print(
            f"    Warning: Failed to create random action GIF for {variant_name}: {e}"
        )
        return False, {}


def generate_variant_markdown(
    variant_id: str,
    env: gymnasium.Env,
    has_initial_gif: bool = True,
    has_random_gif: bool = True,
    random_action_stats: dict[str, float | bool] | None = None,
) -> str:
    """Generate markdown for a single environment variant.

    Args:
        variant_id: The full variant ID (e.g., "kinder/ClutteredStorage2D-b1-v0")
        env: The environment instance for this specific variant
        has_initial_gif: Whether the initial state GIF was successfully generated
        has_random_gif: Whether the random action GIF was successfully generated
        random_action_stats: Stats from random action GIF (reward, success, num_steps)

    Returns:
        The markdown content as a string
    """
    # Extract the variant name without kinder/ prefix and version suffix
    variant_name = sanitize_env_id(variant_id)

    md = f"# {variant_name}\n\n"

    md += "## Usage\n"
    md += "```python\n"
    md += "import kinder\n"
    md += "kinder.register_all_environments()\n"
    md += f'env = kinder.make("{variant_id}")\n'
    md += "```\n\n"

    md += "## Description\n"
    variant_specific = env.metadata.get("variant_specific_description", "")
    if (
        variant_specific
        and variant_specific != "No variant-specific description available."
    ):
        md += f"{variant_specific}\n\n"
    else:
        md += "No variant-specific description available.\n\n"

    md += "## Initial State Distribution\n"
    if has_initial_gif:
        gif_path = f"../../assets/initial_state_gifs/variants/{variant_name}.gif"
        md += f"![initial state GIF]({gif_path})\n\n"
    else:
        md += "*(Initial state GIF not available)*\n\n"

    md += "## Random Action Behavior\n"
    if has_random_gif:
        gif_path = f"../../assets/random_action_gifs/variants/{variant_name}.gif"
        md += f"![random action GIF]({gif_path})\n\n"
        if random_action_stats:
            total_reward = random_action_stats.get("total_reward", 0.0)
            success = random_action_stats.get("terminated_successfully", False)
            num_steps = random_action_stats.get("num_steps", 0)
            success_text = "Yes" if success else "No"
            md += (
                f"**Random Action Stats**: Total Reward: {total_reward:.2f}, "
                f"Success: {success_text}, Steps: {num_steps}\n\n"
            )
    else:
        md += "*(Random action GIF not available)*\n\n"

    md += "## Example Demonstration\n"
    demo_subdir = OUTPUT_DIR / "assets" / "demo_gifs" / variant_name
    gif_to_use = None
    if demo_subdir.exists():
        canonical_gif = demo_subdir / f"{variant_name}.gif"
        if canonical_gif.exists():
            gif_to_use = canonical_gif
        else:
            gif_files = sorted(
                [f for f in demo_subdir.iterdir() if f.suffix.lower() == ".gif"]
            )
            if gif_files:
                gif_to_use = gif_files[0]

    if gif_to_use:
        md += (
            f"![demo GIF](../../assets/demo_gifs/{variant_name}/{gif_to_use.name})\n\n"
        )
        json_file = gif_to_use.with_suffix(".json")
        if json_file.exists():
            with open(json_file, "r", encoding="utf-8") as f:
                demo_stats = json.load(f)
            total_reward = demo_stats.get("total_reward", 0.0)
            success = demo_stats.get("terminated_successfully", False)
            num_steps = demo_stats.get("num_steps", 0)
            success_text = "Yes" if success else "No"
            md += (
                f"**Demo Stats**: Total Reward: {total_reward:.2f}, "
                f"Success: {success_text}, Steps: {num_steps}\n\n"
            )
    else:
        md += "*(No demonstration GIFs available)*\n\n"

    md += "## Observation Space\n"
    md += env.metadata["observation_space_description"] + "\n\n"

    return md.rstrip() + "\n"


def generate_markdown(
    class_name: str,
    env: gymnasium.Env,
    variants: list[str],
    has_random_gif: bool = True,
    has_initial_gif: bool = True,
    random_action_stats: dict[str, float | bool] | None = None,
) -> str:
    """Generate markdown for a given environment class.

    Args:
        class_name: The environment class name (e.g., "ClutteredStorage2D")
        env: A representative environment instance for extracting metadata
        variants: List of all variant IDs for this class
        has_random_gif: Whether the random action GIF was successfully generated
        has_initial_gif: Whether the initial state GIF was successfully generated
        random_action_stats: Stats from random action GIF (reward, success, num_steps)

    Returns:
        The markdown content as a string
    """
    md = f"# {class_name}\n\n"
    class_filename = sanitize_class_name(class_name)

    if has_random_gif:
        md += (
            f"![random action GIF](assets/random_action_gifs/{class_filename}.gif)\n\n"
        )
        # Add stats if available
        if random_action_stats:
            total_reward = random_action_stats.get("total_reward", 0.0)
            success = random_action_stats.get("terminated_successfully", False)
            num_steps = random_action_stats.get("num_steps", 0)
            success_text = "Yes" if success else "No"
            stats_line = (
                f"**Random Action Stats**: Total Reward: {total_reward:.2f}, "
                f"Success: {success_text}, Steps: {num_steps}\n\n"
            )
            md += stats_line
    else:
        md += "*(Random action GIF could not be generated due to rendering issues)*\n\n"

    description = env.metadata.get("description", "No description defined.")
    md += f"## Description\n{description}\n\n"

    # List all available variants with links to their individual docs
    md += "## Available Variants\n"
    variant_description = env.metadata.get("variant_description", "")
    if variant_description and variant_description != "Variant description not defined":
        md += f"{variant_description}\n\n"
    for variant_id in variants:
        # Extract just the suffix (e.g., "b1" from "kinder/ClutteredStorage2D-b1-v0")
        variant_suffix = (
            variant_id.replace("kinder/", "")
            .replace(f"{class_name}-", "")
            .replace("-v0", "")
        )
        variant_filename = sanitize_env_id(variant_id)
        link = f"variants/{class_name}/{variant_filename}.md"
        md += f"- [`{variant_id}`]({link}) ({variant_suffix})\n"
    md += "\n"

    md += "## Initial State Distribution\n"

    if has_initial_gif:
        md += (
            f"![initial state GIF](assets/initial_state_gifs/{class_filename}.gif)\n\n"
        )
    else:
        md += "*(Initial state GIF could not be generated due to rendering issues)*\n\n"

    md += "## Example Demonstration\n"

    demo_gif = OUTPUT_DIR / "assets" / "group_gifs" / f"{class_filename}.gif"
    if demo_gif.exists():
        md += f"![demo GIF](assets/group_gifs/{class_filename}.gif)\n\n"
    else:
        md += "*(No demonstration GIFs available)*\n\n"

    md += "## Observation Space\n"
    md += "*(Differs per variant, see individual variant pages)*\n\n"
    md += "## Action Space\n"
    md += env.metadata["action_space_description"] + "\n\n"
    md += "## Rewards\n"
    md += env.metadata["reward_description"] + "\n\n"
    if "references" in env.metadata:
        md += "## References\n"
        md += env.metadata["references"] + "\n\n"
    return md.rstrip() + "\n"


def _main() -> None:
    parser = argparse.ArgumentParser(description="Generate environment documentation")
    parser.add_argument(
        "--force", action="store_true", help="Force regeneration of all environments"
    )
    parser.add_argument(
        "--env",
        type=str,
        nargs="+",
        help=(
            "Generate docs for specific environment classes "
            "(e.g., Motion2D Manipulation3D)"
        ),
    )
    parser.add_argument(
        "--env_category",
        type=str,
        nargs="+",
        help=(
            "Generate docs for specific environment categories "
            "(e.g., Kinematic2D Dynamic2D Kinematic3D Dynamic3D)"
        ),
    )
    args = parser.parse_args()

    print("Regenerating environment docs...")
    if args.force:
        print("Force flag detected - regenerating all environment classes")
    elif args.env:
        print(f"Generating docs for environments: {', '.join(args.env)}")
    elif args.env_category:
        env_cats = ", ".join(args.env_category)
        print(f"Generating docs for environment categories: {env_cats}")
    else:
        print("Checking for changes using git diff origin/main...")

    OUTPUT_DIR.mkdir(exist_ok=True)
    random_gifs_dir = OUTPUT_DIR / "assets" / "random_action_gifs"
    initial_gifs_dir = OUTPUT_DIR / "assets" / "initial_state_gifs"
    random_gifs_dir.mkdir(parents=True, exist_ok=True)
    (random_gifs_dir / "variants").mkdir(parents=True, exist_ok=True)
    initial_gifs_dir.mkdir(parents=True, exist_ok=True)
    (initial_gifs_dir / "variants").mkdir(parents=True, exist_ok=True)

    kinder.register_all_environments()

    changed_files = get_changed_files()

    total_classes = 0
    regenerated_classes = 0

    env_classes = kinder.get_env_classes()

    # Filter to specific environments if requested
    if args.env:
        not_found = [env for env in args.env if env not in env_classes]
        if not_found:
            print(f"Error: Environment class(es) not found: {', '.join(not_found)}")
            print(f"Available classes: {', '.join(sorted(env_classes.keys()))}")
            return
        env_classes = {env: env_classes[env] for env in args.env}

    # Filter to specific environment categories if requested
    if args.env_category:
        env_categories = kinder.get_env_categories()
        not_found = [cat for cat in args.env_category if cat not in env_categories]
        if not_found:
            print(f"Error: Category(ies) not found: {', '.join(not_found)}")
            print(f"Available categories: {', '.join(sorted(env_categories.keys()))}")
            return
        # Collect all class names from the requested categories
        filtered_classes = set()
        for category in args.env_category:
            filtered_classes.update(env_categories[category])
        env_classes = {env: env_classes[env] for env in filtered_classes}

    for class_name, class_info in env_classes.items():
        total_classes += 1
        registered_variants = class_info["variants"]

        # Use a middle variant as representative (or first if only one variant)
        representative_variant = registered_variants[len(registered_variants) // 2]
        variants = get_documented_variants(class_name, registered_variants)
        kwargs: dict[str, str | bool] = {"render_mode": "rgb_array"}
        if class_name in DYNAMIC3D_ENVS:
            kwargs["scene_bg"] = True
        env = kinder.make(representative_variant, **kwargs)

        # Check if any variant of this class has changed
        def make_env(variant_id: str, class_name: str = class_name):
            kwargs: dict[str, str | bool] = {"render_mode": "rgb_array"}
            if class_name in DYNAMIC3D_ENVS:
                kwargs["scene_bg"] = True
            return kinder.make(variant_id, **kwargs)

        class_changed = any(
            is_env_changed(make_env(v), changed_files) for v in variants
        )

        if args.force or args.env or args.env_category or class_changed:
            print(f"  Regenerating {class_name}...")
            num_actions, render_every = get_random_action_gif_params(class_name)
            has_random_gif, random_action_stats = create_random_action_gif(
                class_name, env, num_actions=num_actions, render_every=render_every
            )
            has_initial_gif = create_initial_state_gif(class_name, variants)
            md = generate_markdown(
                class_name,
                env,
                variants,
                has_random_gif,
                has_initial_gif,
                random_action_stats,
            )
            class_filename = sanitize_class_name(class_name)
            filename = OUTPUT_DIR / f"{class_filename}.md"
            with open(filename, "w", encoding="utf-8") as f:
                f.write(md)

            # Generate individual variant markdown files
            variant_dir = OUTPUT_DIR / "variants" / class_name
            variant_dir.mkdir(parents=True, exist_ok=True)
            for variant_id in variants:
                variant_kwargs: dict[str, str | bool] = {"render_mode": "rgb_array"}
                if class_name in DYNAMIC3D_ENVS:
                    variant_kwargs["scene_bg"] = True
                variant_env = kinder.make(variant_id, **variant_kwargs)
                variant_name = sanitize_env_id(variant_id)
                variant_has_initial_gif = create_variant_initial_state_gif(
                    variant_name, variant_env
                )
                (
                    variant_has_random_gif,
                    variant_random_stats,
                ) = create_variant_random_action_gif(
                    variant_name,
                    variant_env,
                    num_actions=num_actions,
                    render_every=render_every,
                )
                variant_md = generate_variant_markdown(
                    variant_id,
                    variant_env,
                    variant_has_initial_gif,
                    variant_has_random_gif,
                    variant_random_stats,
                )
                variant_file = variant_dir / f"{variant_name}.md"
                with open(variant_file, "w", encoding="utf-8") as f:
                    f.write(variant_md)

            regenerated_classes += 1
        else:
            print(f"  Skipping {class_name} (no changes detected)")

    print(
        "Finished generating environment docs. "
        f"Regenerated {regenerated_classes}/{total_classes} classes."
    )

    # Add the results.
    subprocess.run(["git", "add", OUTPUT_DIR], check=True)


if __name__ == "__main__":
    _main()
