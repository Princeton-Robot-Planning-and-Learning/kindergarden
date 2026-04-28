#!/usr/bin/env python3
"""Combine GIFs from group_gifs into a grid MP4 video."""

import subprocess
import sys
from pathlib import Path

# Kinematic3D environments (from kinder registration)
KINEMATIC3D_ENVS = {
    "Motion3D",
    "BaseMotion3D",
    "Ground3D",
    "Table3D",
    "Transport3D",
    "Shelf3D",
    "Obstruction3D",
    "Packing3D",
}

# Kinematic2D environments
KINEMATIC2D_ENVS = {
    "Obstruction2D",
    "ClutteredRetrieval2D",
    "ClutteredStorage2D",
    "Motion2D",
    "StickButton2D",
    "PushPullHook2D",
}

# Dynamic2D environments
DYNAMIC2D_ENVS = {
    "DynObstruction2D",
    "DynPushPullHook2D",
    "DynPushT2D",
    "DynScoopPour2D",
}


def get_category(name: str) -> int:
    """
    Return sort key for category order: Dynamic3D, Kinematic3D,
    Dynamic2D, Kinematic2D.
    """
    if name in KINEMATIC3D_ENVS:
        return 1  # Kinematic3D
    if name in KINEMATIC2D_ENVS:
        return 3  # Kinematic2D
    if name in DYNAMIC2D_ENVS:
        return 2  # Dynamic2D
    if name.endswith("3D"):
        return 0  # Dynamic3D (anything else ending in 3D)
    return 4  # Unknown


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
    """Combine GIFs into a grid video."""
    group_gifs_dir = Path(__file__).parent.parent / "docs/envs/assets/group_gifs"
    output_path = Path(__file__).parent.parent / "docs/envs/assets/group_grid.mp4"

    gif_files = sorted(
        group_gifs_dir.glob("*.gif"),
        key=lambda p: (get_category(p.stem), p.stem),
    )
    if not gif_files:
        print("No GIFs found in group_gifs directory")
        sys.exit(1)

    print(f"Found {len(gif_files)} GIFs")

    # Gather info for all GIFs
    gif_info = []
    for gif_path in gif_files:
        w, h, duration = get_gif_info(gif_path)
        gif_info.append((gif_path, w, h, duration))
        print(f"  {gif_path.name}: {w}x{h}, {duration:.2f}s")

    # Determine grid dimensions
    n = len(gif_files)
    cols = 5
    rows = 5

    # Target cell size and duration
    cell_size = 240
    speed_multiplier = 4
    max_duration = max(info[3] for info in gif_info)
    target_duration = max_duration / speed_multiplier
    print(f"Target duration: {target_duration:.2f}s")

    # Build ffmpeg filter complex
    inputs = []
    filter_parts = []
    sz = cell_size

    for i, (gif_path, w, h, duration) in enumerate(gif_info):
        # Loop short GIFs twice (if less than 1/3 of max duration)
        if duration < max_duration / 3:
            inputs.extend(["-stream_loop", "1", "-i", str(gif_path)])
            effective_duration = duration * 2
            print(f"  Looping {gif_path.name} (2x)")
        else:
            inputs.extend(["-i", str(gif_path)])
            effective_duration = duration

        # Calculate speed factor to normalize duration
        speed_factor = target_duration / effective_duration

        is_dynamic3d = get_category(gif_path.stem) == 0

        if w == h:
            # Already square, just scale and adjust duration
            filt = f"[{i}:v]setpts={speed_factor}*PTS,scale={sz}:{sz}[v{i}]"
        elif w > h:
            if is_dynamic3d:
                # Crop horizontally to make square, then scale
                filt = (
                    f"[{i}:v]setpts={speed_factor}*PTS,"
                    f"crop=ih:ih:(iw-ih)/2:0,scale={sz}:{sz}[v{i}]"
                )
            else:
                # Add vertical padding
                filt = (
                    f"[{i}:v]setpts={speed_factor}*PTS,scale={sz}:-1,"
                    f"pad={sz}:{sz}:(ow-iw)/2:(oh-ih)/2:color=white[v{i}]"
                )
        else:
            if is_dynamic3d:
                # Crop vertically to make square, then scale
                filt = (
                    f"[{i}:v]setpts={speed_factor}*PTS,"
                    f"crop=iw:iw:0:(ih-iw)/2,scale={sz}:{sz}[v{i}]"
                )
            else:
                # Add horizontal padding
                filt = (
                    f"[{i}:v]setpts={speed_factor}*PTS,scale=-1:{sz},"
                    f"pad={sz}:{sz}:(ow-iw)/2:(oh-ih)/2:color=white[v{i}]"
                )
        filter_parts.append(filt)

    # Add white frames for empty cells if needed
    empty_cells = rows * cols - n
    for i in range(empty_cells):
        idx = n + i
        filter_parts.append(f"color=white:s={sz}x{sz}:d={target_duration}[v{idx}]")

    # Build xstack layout
    total_cells = rows * cols
    layout_parts = []
    for i in range(total_cells):
        row = i // cols
        col = i % cols
        x = col * cell_size
        y = row * cell_size
        layout_parts.append(f"{x}_{y}")

    stream_refs = "".join(f"[v{i}]" for i in range(total_cells))
    layout = "|".join(layout_parts)
    filter_parts.append(
        f"{stream_refs}xstack=inputs={total_cells}:layout={layout}[out]"
    )

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
    print(f"Created {output_path}")


if __name__ == "__main__":
    main()
