#!/usr/bin/env python3
"""Generate a 5x5 grid video from TidyBot3D sweep demos."""

import subprocess
import sys
import tempfile
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

# pylint: disable=wrong-import-position
from generate_demo_video import discover_demos_by_env, generate_demo_video


def get_gif_info(gif_path: Path) -> tuple[int, int, float]:
    """Get width, height, and duration of a GIF using ffprobe."""
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height",
            "-show_entries",
            "format=duration",
            "-of",
            "csv=p=0",
            str(gif_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    lines = result.stdout.strip().split("\n")
    w, h = lines[0].split(",")
    duration = float(lines[1])
    return int(w), int(h), duration


def main() -> None:
    """Generate a grid video from multiple TidyBot3D demos."""
    env_id = (
        "kinder/TidyBot3D-tool_use-lab2_kitchen"
        "-o5-sweep_the_blocks_into_the_top_drawer_of_the_kitchen_island-v0"
    )
    demos_dir = Path(__file__).parent.parent / "demos"
    output_path = (
        Path(__file__).parent.parent / "docs/envs/assets/tidybot_sweep_grid.mp4"
    )

    all_demos = discover_demos_by_env(env_id, demos_dir)
    if len(all_demos) < 25:
        print(f"Error: Need at least 25 demos, found {len(all_demos)}")
        sys.exit(1)

    # Get one demo per seed (different seeds = different initial conditions)
    demos_by_seed: dict[int, Path] = {}
    for demo in all_demos:
        seed = int(demo.parent.name)
        if seed not in demos_by_seed:
            demos_by_seed[seed] = demo

    # Start at seed 10 and go up numerically
    seeds = [s for s in sorted(demos_by_seed.keys()) if s >= 10][:25]
    selected_demos = [demos_by_seed[s] for s in seeds]
    print(f"Selected 25 demos from seeds: {seeds}")

    # Generate GIFs in a temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        gif_paths = []
        for i, demo_path in enumerate(selected_demos):
            gif_path = Path(tmpdir) / f"demo_{i:02d}.gif"
            print(f"\n=== Generating GIF {i+1}/25 from {demo_path} ===")
            generate_demo_video(demo_path, gif_path)
            gif_paths.append(gif_path)

        # Gather info for all GIFs
        print("\n=== Combining into grid video ===")
        gif_info = []
        for gif_path in gif_paths:
            w, h, duration = get_gif_info(gif_path)
            gif_info.append((gif_path, w, h, duration))
            print(f"  {gif_path.name}: {w}x{h}, {duration:.2f}s")

        # Grid parameters
        rows, cols = 5, 5
        cell_size = 240
        speed_multiplier = 4
        max_duration = max(info[3] for info in gif_info)
        target_duration = max_duration / speed_multiplier
        print(f"Target duration: {target_duration:.2f}s (4x speed)")

        # Build ffmpeg filter complex
        inputs = []
        filter_parts = []

        for i, (gif_path, w, h, duration) in enumerate(gif_info):
            inputs.extend(["-i", str(gif_path)])
            speed_factor = target_duration / duration
            sz = cell_size

            if w == h:
                filt = f"[{i}:v]setpts={speed_factor}*PTS,scale={sz}:{sz}[v{i}]"
            elif w > h:
                # Wider than tall - crop horizontally to make square
                filt = (
                    f"[{i}:v]setpts={speed_factor}*PTS,"
                    f"crop=ih:ih:(iw-ih)/2:0,scale={sz}:{sz}[v{i}]"
                )
            else:
                # Taller than wide - crop vertically to make square
                filt = (
                    f"[{i}:v]setpts={speed_factor}*PTS,"
                    f"crop=iw:iw:0:(ih-iw)/2,scale={sz}:{sz}[v{i}]"
                )
            filter_parts.append(filt)

        # Build xstack layout
        layout_parts = []
        for i in range(rows * cols):
            row = i // cols
            col = i % cols
            x = col * cell_size
            y = row * cell_size
            layout_parts.append(f"{x}_{y}")

        stream_refs = "".join(f"[v{i}]" for i in range(25))
        layout = "|".join(layout_parts)
        filter_parts.append(f"{stream_refs}xstack=inputs=25:layout={layout}[out]")

        filter_complex = ";".join(filter_parts)

        cmd = [
            "ffmpeg",
            "-y",
            *inputs,
            "-filter_complex",
            filter_complex,
            "-map",
            "[out]",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-t",
            str(target_duration),
            str(output_path),
        ]

        print(f"Running ffmpeg to create {output_path}")
        subprocess.run(cmd, check=True)
        print(f"\nCreated {output_path}")


if __name__ == "__main__":
    main()
