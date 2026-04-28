"""Generate videos from pickled demonstrations.

This script can generate GIFs from demonstration files in several modes:

1. Single demo: python generate_demo_video.py path/to/demo.p
2. Latest demo: python generate_demo_video.py --latest
3. All demos: python generate_demo_video.py --all [--max-demos N]
4. Environment demos: python generate_demo_video.py --env kinder/Motion2D-p1
5. One per variant: python generate_demo_video.py --one-per-variant [--force]

The script automatically discovers demonstration files in the demos/ directory
and generates appropriately named output files in docs/envs/assets/demo_gifs/.

The --one-per-variant mode is recommended for generating documentation. It creates
exactly one representative GIF for each registered environment variant, using a
consistent naming scheme (variant_name.gif instead of variant_name_seedX_timestampY.gif).

NOTE: this currently assumes that environments are deterministic. If that is
not the case, we will need to be able to render observations (which are being
saving in the demonstrations also).
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import imageio.v2 as iio
from kindergarden.scripts.docs.generate_env_docs import sanitize_env_id

import kinder
from kinder.gif_utils import optimize_gif
from kinder.utils import load_demo


def discover_all_demos(demos_dir: Path = Path("demos")) -> List[Path]:
    """Discover all demo files in the demos directory.

    Returns:
        List of demo file paths sorted by modification time (newest first).
    """
    if not demos_dir.exists():
        print(f"Error: Demos directory {demos_dir} does not exist")
        return []

    demo_files = []
    for demo_file in demos_dir.rglob("*.p"):
        if demo_file.is_file():
            demo_files.append(demo_file)

    # Sort by modification time (newest first)
    demo_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return demo_files


def discover_demos_by_env(env_id: str, demos_dir: Path = Path("demos")) -> List[Path]:
    """Discover all demo files for a specific environment.

    Args:
        env_id: Environment ID (with or without 'kinder/' prefix)
        demos_dir: Directory containing demos

    Returns:
        List of demo file paths for the environment sorted by modification time
        (newest first).
    """
    # Remove kinder/ prefix if present for directory matching
    sanitized_env_name = sanitize_env_id(env_id)

    env_dir = demos_dir / sanitized_env_name
    if not env_dir.exists():
        print(f"Error: No demos found for environment {env_id} in {env_dir}")
        return []

    demo_files = []
    for demo_file in env_dir.rglob("*.p"):
        if demo_file.is_file():
            demo_files.append(demo_file)

    # Sort by modification time (newest first)
    demo_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return demo_files


def find_best_demo_for_variant(
    env_id: str, demos_dir: Path = Path("demos")
) -> Optional[Path]:
    """Find the best demo for a variant (most recent).

    Args:
        env_id: Environment ID (e.g., 'kinder/ClutteredStorage2D-b1-v0')
        demos_dir: Directory containing demos

    Returns:
        Path to the best demo file, or None if no demos found.
    """
    demos = discover_demos_by_env(env_id, demos_dir)
    return demos[0] if demos else None


def find_latest_demo(demos_dir: Path = Path("demos")) -> Optional[Path]:
    """Find the most recently modified demo file.

    Args:
        demos_dir: Directory containing demos

    Returns:
        Path to the latest demo file, or None if no demos found.
    """
    all_demos = discover_all_demos(demos_dir)
    return all_demos[0] if all_demos else None


def get_demo_info(demo_path: Path) -> Tuple[str, int, int]:
    """Extract basic info from demo path without loading the full file.

    Returns:
        Tuple of (env_id_from_path, seed, timestamp)
    """
    # Parse path: demos/ENV_ID/SEED/TIMESTAMP.p
    parts = demo_path.parts
    if len(parts) >= 3:
        env_id_from_path = parts[-3]  # Environment directory name
        seed = int(parts[-2])  # Seed directory name
        timestamp = int(demo_path.stem)  # Filename without extension
        return env_id_from_path, seed, timestamp

    # Fallback - try to parse from filename and parent dirs
    return "unknown", 0, int(demo_path.stem)


def generate_demo_video(
    demo_path: Path,
    output_path: Path | None = None,
    fps: int | None = None,
    loop: int = 0,
) -> None:
    """Generate a video from a pickled demonstration."""
    # Load the demonstration.
    demo_data = load_demo(demo_path)

    # Extract demo information.
    env_id = demo_data["env_id"]
    actions = demo_data["actions"]
    seed = demo_data["seed"]

    print(f"Loaded demo for environment: {env_id}")
    print(f"Demo length: {len(actions)} actions")
    print(f"Demo seed: {seed}")

    # Create the environment.
    kinder.register_all_environments()
    if "TidyBot" in env_id:
        env = kinder.make(
            env_id,
            render_mode="rgb_array",
            scene_bg=True,
        )
    elif "3D" in env_id and "TidyBot" not in env_id:
        env = kinder.make(
            env_id,
            render_mode="rgb_array",
            realistic_bg=True,
        )
    else:
        env = kinder.make(env_id, render_mode="rgb_array")

    # Get FPS from environment metadata if not specified.
    if fps is None:
        fps = env.metadata.get("render_fps", 10)

    # Generate output path if not specified.
    if output_path is None:
        env_filename = sanitize_env_id(env_id)
        output_dir = Path("./docs/envs/assets/demo_gifs")
        env_subdir = output_dir / env_filename
        env_subdir.mkdir(parents=True, exist_ok=True)
        # Include timestamp in filename to avoid conflicts
        timestamp = int(demo_path.stem) if demo_path.stem.isdigit() else "unknown"
        output_path = env_subdir / f"{env_filename}_{timestamp}.gif"

    # Ensure output directory exists.
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Reset environment to initial state with the correct seed.
    env.reset(seed=seed)

    # Collect frames by replaying the demonstration.
    frames = []
    total_reward = 0.0
    terminated_successfully = False

    if "TidyBot" in env_id:
        env.unwrapped._object_centric_env.set_render_camera("agentview_1")  # type: ignore # pylint: disable=protected-access
    # Add initial frame.
    initial_frame = env.render()  # type: ignore
    frames.append(initial_frame)

    # Replay each action and capture frames.
    for i, action in enumerate(actions):
        try:
            _, reward, terminated, truncated, _ = env.step(action)
            total_reward += float(reward)

            if "TidyBot" in env_id:
                env.unwrapped._object_centric_env.set_render_camera("agentview_1")  # type: ignore # pylint: disable=protected-access
            frame = env.render()  # type: ignore
            frames.append(frame)

            if terminated or truncated:
                terminated_successfully = terminated
                print(f"Episode ended after {i+1} actions")
                print(f"Success: {terminated_successfully}")
                break
        except Exception as e:
            print(f"Error during action {i}: {e}")
            print(f"Continuing with {len(frames)} frames collected so far")
            break

    # Check if we have enough frames.
    if len(frames) < 2:
        raise ValueError("Not enough frames collected to create a video")

    # Save the video.
    print(f"Saving video to {output_path}")
    print(f"Video specs: {len(frames)} frames, {fps} fps")
    print(f"Total reward: {total_reward:.2f}, Success: {terminated_successfully}")

    try:
        iio.mimsave(output_path, frames, fps=fps, loop=loop)  # type: ignore
        print("Video saved successfully!")
        # Optimize the GIF to reduce file size.
        optimize_gif(output_path)
    except Exception as e:
        raise ValueError(f"Error saving video: {e}") from e

    # Save stats to JSON file alongside the GIF.
    stats_path = output_path.with_suffix(".json")
    stats = {
        "total_reward": float(total_reward),
        "terminated_successfully": bool(terminated_successfully),
        "num_steps": len(actions),
    }
    try:
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)
        print(f"Stats saved to {stats_path}")
    except Exception as e:
        print(f"Warning: Failed to save stats to {stats_path}: {e}")


def generate_latest_demo_video(
    demos_dir: Path = Path("demos"),
    output_dir: Optional[Path] = None,
    fps: Optional[int] = None,
    loop: int = 0,
) -> None:
    """Generate a video for the most recently collected demo."""
    latest_demo = find_latest_demo(demos_dir)
    if latest_demo is None:
        print(f"Error: No demos found in {demos_dir}")
        sys.exit(1)

    # Generate descriptive output path
    if output_dir is None:
        output_dir = Path("./docs/envs/assets/demo_gifs")

    env_id_from_path, seed, timestamp = get_demo_info(latest_demo)
    env_subdir = output_dir / env_id_from_path
    env_subdir.mkdir(parents=True, exist_ok=True)
    output_filename = f"latest_{env_id_from_path}_seed{seed}_{timestamp}.gif"
    output_path = env_subdir / output_filename

    print(f"Generating GIF for latest demo: {latest_demo}")

    # Check if output already exists
    if output_path.exists():
        print(f"GIF already exists: {output_path}")
        print("Skipping generation.")
        return

    try:
        generate_demo_video(latest_demo, output_path, fps, loop)
    except Exception as e:
        print(f"Error processing latest demo {latest_demo}: {e}")
        raise


def generate_all_demo_videos(
    demos_dir: Path = Path("demos"),
    output_dir: Optional[Path] = None,
    fps: Optional[int] = None,
    loop: int = 0,
    max_demos: Optional[int] = None,
) -> None:
    """Generate videos for all collected demos."""
    all_demos = discover_all_demos(demos_dir)
    if not all_demos:
        print(f"Error: No demos found in {demos_dir}")
        sys.exit(1)

    if max_demos:
        all_demos = all_demos[:max_demos]

    if output_dir is None:
        output_dir = Path("./docs/envs/assets/demo_gifs")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Found {len(all_demos)} demos. Generating GIFs...")

    successful = 0
    failed = 0
    skipped = 0
    failed_demos = []

    for i, demo_path in enumerate(all_demos, 1):
        try:
            env_id_from_path, seed, timestamp = get_demo_info(demo_path)
            env_subdir = output_dir / env_id_from_path
            env_subdir.mkdir(parents=True, exist_ok=True)
            output_filename = f"{env_id_from_path}_seed{seed}_{timestamp}.gif"
            output_path = env_subdir / output_filename

            # Check if output already exists
            if output_path.exists():
                print(f"[{i}/{len(all_demos)}] Skipping {demo_path} (GIF exists)")
                skipped += 1
                continue

            print(f"[{i}/{len(all_demos)}] Processing {demo_path}")
            generate_demo_video(demo_path, output_path, fps, loop)
            successful += 1

        except Exception as e:
            error_msg = f"{demo_path}: {e}"
            failed_demos.append(error_msg)
            print(f"[{i}/{len(all_demos)}] Error processing {demo_path}: {e}")
            failed += 1
            continue

    print(f"\nCompleted: {successful} successful, {failed} failed, {skipped} skipped")

    # Report all failed demos at the end
    if failed_demos:
        print(f"\n=== Failed Demos ({len(failed_demos)}) ===")
        for error_msg in failed_demos:
            print(f"  • {error_msg}")
        print()


def generate_env_demo_videos(
    env_id: str,
    demos_dir: Path = Path("demos"),
    output_dir: Optional[Path] = None,
    fps: Optional[int] = None,
    loop: int = 0,
    max_demos: Optional[int] = None,
) -> None:
    """Generate videos for all demos of a specific environment."""
    env_demos = discover_demos_by_env(env_id, demos_dir)
    if not env_demos:
        print(f"Error: No demos found for environment {env_id}")
        sys.exit(1)

    if max_demos:
        env_demos = env_demos[:max_demos]

    if output_dir is None:
        output_dir = Path("./docs/envs/assets/demo_gifs")
    output_dir.mkdir(parents=True, exist_ok=True)

    sanitized_env = sanitize_env_id(env_id)
    print(f"Found {len(env_demos)} demos for {env_id}. Generating GIFs...")

    successful = 0
    failed = 0
    skipped = 0
    failed_demos = []

    for i, demo_path in enumerate(env_demos, 1):
        try:
            _, seed, timestamp = get_demo_info(demo_path)
            env_subdir = output_dir / sanitized_env
            env_subdir.mkdir(parents=True, exist_ok=True)
            output_filename = f"{sanitized_env}_seed{seed}_{timestamp}.gif"
            output_path = env_subdir / output_filename

            # Check if output already exists
            if output_path.exists():
                print(f"[{i}/{len(env_demos)}] Skipping {demo_path} (GIF exists)")
                skipped += 1
                continue

            print(f"[{i}/{len(env_demos)}] Processing {demo_path}")
            generate_demo_video(demo_path, output_path, fps, loop)
            successful += 1

        except Exception as e:
            error_msg = f"{demo_path}: {e}"
            failed_demos.append(error_msg)
            print(f"[{i}/{len(env_demos)}] Error processing {demo_path}: {e}")
            failed += 1
            continue

    print(f"\nCompleted: {successful} successful, {failed} failed, {skipped} skipped")

    # Report all failed demos at the end
    if failed_demos:
        print(f"\n=== Failed Demos ({len(failed_demos)}) ===")
        for error_msg in failed_demos:
            print(f"  • {error_msg}")
        print()


def generate_one_per_variant(
    demos_dir: Path = Path("demos"),
    output_dir: Optional[Path] = None,
    fps: Optional[int] = None,
    loop: int = 0,
    force: bool = False,
) -> None:
    """Generate exactly one representative GIF for each environment variant.

    Args:
        demos_dir: Directory containing demo files
        output_dir: Output directory for GIFs
        fps: Frames per second for the video
        loop: Number of loops for GIF (0 = infinite)
        force: If True, regenerate even if GIF already exists
    """
    if output_dir is None:
        output_dir = Path("./docs/envs/assets/demo_gifs")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get all registered environment variants
    kinder.register_all_environments()
    env_classes = kinder.get_env_classes()

    # Collect all variant IDs
    all_variants = []
    for _, class_info in env_classes.items():
        all_variants.extend(class_info["variants"])

    print(f"Found {len(all_variants)} registered variants")
    print("Generating one representative GIF per variant...\n")

    successful = 0
    skipped_no_demo = 0
    skipped_exists = 0
    failed = 0
    failed_variants = []

    for i, variant_id in enumerate(all_variants, 1):
        try:
            sanitized_variant = sanitize_env_id(variant_id)
            variant_subdir = output_dir / sanitized_variant
            variant_subdir.mkdir(parents=True, exist_ok=True)

            # Use consistent naming without timestamp
            output_filename = f"{sanitized_variant}.gif"
            output_path = variant_subdir / output_filename

            # Check if output already exists
            if output_path.exists() and not force:
                print(f"[{i}/{len(all_variants)}] Skipping {variant_id} (GIF exists)")
                skipped_exists += 1
                continue

            # Find the best demo for this variant
            demo_path = find_best_demo_for_variant(variant_id, demos_dir)
            if demo_path is None:
                msg = f"[{i}/{len(all_variants)}] Skipping {variant_id}"
                print(f"{msg} (no demos available)")
                skipped_no_demo += 1
                continue

            msg = f"[{i}/{len(all_variants)}] Generating {variant_id}"
            print(f"{msg} from {demo_path.name}")
            generate_demo_video(demo_path, output_path, fps, loop)
            successful += 1

        except Exception as e:
            error_msg = f"{variant_id}: {e}"
            failed_variants.append(error_msg)
            print(f"[{i}/{len(all_variants)}] Error processing {variant_id}: {e}")
            failed += 1
            continue

    print(f"\n{'='*60}")
    print("Summary:")
    print(f"  Successful: {successful}")
    print(f"  Skipped (no demo): {skipped_no_demo}")
    print(f"  Skipped (exists): {skipped_exists}")
    print(f"  Failed: {failed}")
    print(f"{'='*60}")

    # Report all failed variants at the end
    if failed_variants:
        print(f"\n=== Failed Variants ({len(failed_variants)}) ===")
        for error_msg in failed_variants:
            print(f"  • {error_msg}")
        print()


def _main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate videos from pickled demonstrations"
    )

    # Create mutually exclusive group for input methods
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "demo_path", nargs="?", type=Path, help="Path to the pickled demonstration file"
    )
    input_group.add_argument(
        "--latest",
        action="store_true",
        help="Generate GIF for the most recently collected demo",
    )
    input_group.add_argument(
        "--all", action="store_true", help="Generate GIFs for all collected demos"
    )
    input_group.add_argument(
        "--env",
        type=str,
        help="Generate GIFs for all demos of the specified environment",
    )
    input_group.add_argument(
        "--one-per-variant",
        action="store_true",
        help="Generate one representative GIF per registered environment variant",
    )

    # Common options
    parser.add_argument(
        "--demos-dir",
        type=Path,
        default=Path("demos"),
        help="Directory containing demo files (default: demos)",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        help="Output directory (default: docs/envs/assets/demo_gifs/)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        help="Frames per second for the video (default: from environment metadata)",
    )
    parser.add_argument(
        "--loop",
        type=int,
        default=0,
        help="Number of loops for GIF (0 = infinite, default: 0)",
    )
    parser.add_argument(
        "--max-demos",
        type=int,
        help="Maximum number of demos to process (for --all and --env modes)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force regeneration even if GIF exists (--one-per-variant)",
    )

    args = parser.parse_args()

    # Handle different input modes
    if args.demo_path:
        # Original single demo mode
        if not args.demo_path.exists():
            print(f"Error: Demo file {args.demo_path} does not exist")
            sys.exit(1)

        # For single demo mode, also organize by environment if output_dir is specified
        if args.output_dir:
            env_id_from_path, _, _ = get_demo_info(args.demo_path)
            env_subdir = args.output_dir / env_id_from_path
            env_subdir.mkdir(parents=True, exist_ok=True)
            custom_output_path = env_subdir / f"custom_{args.demo_path.stem}.gif"
        else:
            custom_output_path = None

        # Check if output already exists for single demo mode
        if custom_output_path and custom_output_path.exists():
            print(f"GIF already exists: {custom_output_path}")
            print("Skipping generation.")
        else:
            try:
                generate_demo_video(
                    demo_path=args.demo_path,
                    output_path=custom_output_path,
                    fps=args.fps,
                    loop=args.loop,
                )
            except Exception as e:
                print(f"Error processing demo {args.demo_path}: {e}")
                sys.exit(1)

    elif args.latest:
        # Generate GIF for latest demo
        generate_latest_demo_video(
            demos_dir=args.demos_dir,
            output_dir=args.output_dir,
            fps=args.fps,
            loop=args.loop,
        )

    elif args.all:
        # Generate GIFs for all demos
        generate_all_demo_videos(
            demos_dir=args.demos_dir,
            output_dir=args.output_dir,
            fps=args.fps,
            loop=args.loop,
            max_demos=args.max_demos,
        )

    elif args.env:
        # Generate GIFs for specific environment
        generate_env_demo_videos(
            env_id=args.env,
            demos_dir=args.demos_dir,
            output_dir=args.output_dir,
            fps=args.fps,
            loop=args.loop,
            max_demos=args.max_demos,
        )

    elif args.one_per_variant:
        # Generate one GIF per variant
        generate_one_per_variant(
            demos_dir=args.demos_dir,
            output_dir=args.output_dir,
            fps=args.fps,
            loop=args.loop,
            force=args.force,
        )


if __name__ == "__main__":
    _main()
