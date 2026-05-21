#!/usr/bin/env python3
"""Render depth from a MuJoCo camera in a KinDER Dynamic3D env and build a world-frame point cloud (N×3).

================================================================================
Installation
================================================================================
1. Python 3.10–3.12
2. Install KinDER from the repo root (includes Dynamic3D / MuJoCo deps)::

       cd kindergarden
       pip install -e ".[dynamic3d]"
       # Or full dev environment: pip install -e ".[develop]"

3. The first Dynamic3D run may auto-download MimicLabs scene assets (network required).
   On headless Linux, if EGL fails, try::

       export MUJOCO_GL=osmesa

================================================================================
Usage
================================================================================
From the repo root::

    python scripts/render_pointcloud.py

Common options::

    python scripts/render_pointcloud.py --seed 42
    python scripts/render_pointcloud.py --output pointcloud.npy
    python scripts/render_pointcloud.py --show
    python scripts/render_pointcloud.py --save-rgb camera_view.png
    python scripts/visualize_pointcloud.py --input pointcloud.npy
    python scripts/render_pointcloud.py --camera tidybot_base

Note: Rearrange3D task JSON often names the robot instance ``robot``, so the MuJoCo
base camera is ``robot_base``. If ``tidybot_base`` is missing, the library falls back
to ``{robot_name}_base`` and prints a note.

================================================================================
Expected output on success
================================================================================
- Terminal prints: env ID, camera name, image size, valid point count, XYZ min/max/std
- With ``--output``, saves a float32 .npy file with shape (N, 3)
- With ``--show``, opens a window with the camera RGB view (what the robot camera sees)
- With ``--save-rgb PATH``, writes that RGB frame to a PNG
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import kinder  # noqa: E402  # pylint: disable=wrong-import-position
from kinder.envs.dynamic3d.pointcloud import (  # noqa: E402
    capture_pointcloud_from_gym_env,
    show_camera_rgb,
)

DEFAULT_ENV_ID = (
    "kinder/Rearrange3D-o1-put_the_boxed_drink_next_to_the_bowl-v0"
)
DEFAULT_CAMERA = "tidybot_base"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render a world-frame point cloud from a KinDER Dynamic3D camera.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="See script docstring for install and run instructions.",
    )
    parser.add_argument(
        "--env-id",
        default=DEFAULT_ENV_ID,
        help=f"Gymnasium environment ID (default: {DEFAULT_ENV_ID})",
    )
    parser.add_argument(
        "--camera",
        default=DEFAULT_CAMERA,
        help=f"MuJoCo camera name in the merged model (default: {DEFAULT_CAMERA})",
    )
    parser.add_argument("--seed", type=int, default=0, help="Reset seed")
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Optional path to save point cloud as .npy (float32, shape Nx3)",
    )
    parser.add_argument(
        "--min-depth",
        type=float,
        default=0.02,
        help="Ignore points closer than this distance in meters",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Open a window showing the camera RGB view (requires a display)",
    )
    parser.add_argument(
        "--save-rgb",
        type=str,
        default="",
        help="Save the camera RGB frame to this PNG path (e.g. camera_view.png)",
    )
    args = parser.parse_args()

    kinder.register_all_environments()
    env = kinder.make(args.env_id)
    try:
        env.reset(seed=args.seed)
        points_world, rgb, camera_name, camera_id = capture_pointcloud_from_gym_env(
            env,
            args.camera,
            min_depth=args.min_depth,
        )
        robot_env = env.unwrapped._object_centric_env._robot_env  # pylint: disable=protected-access
        width = robot_env.camera_width
        height = robot_env.camera_height

        if args.show or args.save_rgb:
            show_camera_rgb(
                rgb,
                title=f"{camera_name} — {args.env_id}",
                show=args.show,
                save_path=args.save_rgb or None,
            )

        print(f"Environment: {args.env_id}")
        print(f"Camera:      {camera_name} (id={camera_id})")
        print(f"Image size:  {width} x {height}")
        print(
            f"Valid points: {points_world.shape[0]} / {width * height} pixels"
        )
        if points_world.shape[0] == 0:
            print("WARNING: No valid points; check camera name and depth filters.")
            sys.exit(1)
        print(f"World XYZ min: {points_world.min(axis=0)}")
        print(f"World XYZ max: {points_world.max(axis=0)}")
        print(f"World XYZ std: {points_world.std(axis=0)}")

        if args.output:
            out_path = Path(args.output)
            np.save(out_path, points_world)
            print(
                f"Saved point cloud to {out_path.resolve()} "
                f"with shape {points_world.shape}"
            )
    finally:
        env.close()


if __name__ == "__main__":
    main()
