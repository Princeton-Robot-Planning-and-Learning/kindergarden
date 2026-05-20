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
base camera is ``robot_base``. If ``tidybot_base`` is missing, the script falls back
to ``{robot_name}_base`` and prints a note.

================================================================================
Expected output on success
================================================================================
- Terminal prints: env ID, camera name, image size, valid point count, XYZ min/max/std
- With ``--output``, saves a float32 .npy file with shape (N, 3)
- With ``--show``, opens a window with the camera RGB view (what the robot camera sees)
- With ``--save-rgb PATH``, writes that RGB frame to a PNG
- Valid point count is often tens of thousands (depends on depth filtering and visibility)

Example::

    Environment: kinder/Rearrange3D-o1-put_the_boxed_drink_next_to_the_bowl-v0
    Camera:      robot_base (id=4)
    Image size:  640 x 480
    Note: camera 'tidybot_base' not in model; using 'robot_base' ...
    Valid points: 307200 / 307200 pixels
    World XYZ min / max / std: (base camera close range; X spread may be small)
    Saved point cloud to pointcloud.npy with shape (307200, 3)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Allow importing kinder when run from repo root without pip install -e .
_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import kinder  # noqa: E402  # pylint: disable=wrong-import-position
import mujoco  # noqa: E402  # pylint: disable=wrong-import-position


DEFAULT_ENV_ID = (
    "kinder/Rearrange3D-o1-put_the_boxed_drink_next_to_the_bowl-v0"
)
# Users often say "tidybot base camera"; the MuJoCo prefix comes from the robot
# instance name in task JSON (e.g. Rearrange3D uses "robot" -> camera "robot_base").
DEFAULT_CAMERA = "tidybot_base"


def resolve_camera_name(oc_env: object, requested: str) -> str:
    """Resolve camera name: use requested if present, else {robot_name}_base / _wrist."""
    robot_env = oc_env._robot_env  # type: ignore[attr-defined]  # pylint: disable=protected-access
    assert robot_env is not None and robot_env.sim is not None
    available = [n for n in robot_env.sim.model.camera_names if n is not None]
    if requested in available:
        return requested
    robot_name = oc_env.robot_name  # type: ignore[attr-defined]
    for suffix in ("_base", "_wrist"):
        candidate = f"{robot_name}{suffix}"
        if requested.endswith(suffix) and candidate in available:
            print(
                f"Note: camera '{requested}' not in model; using '{candidate}' "
                f"(robot instance name is '{robot_name}')."
            )
            return candidate
    raise ValueError(
        f"Camera '{requested}' not found. Available cameras: {available}"
    )


def depth_buffer_to_metric(
    depth_buffer: np.ndarray,
    model: mujoco.MjModel,  # type: ignore[name-defined]  # pylint: disable=no-member
) -> np.ndarray:
    """Convert mjr_readPixels [0,1] depth buffer to metric distance along the optical axis.

    Uses the same near/far formula as common MuJoCo / dm_control implementations
    (model.vis.map znear/zfar scaled by model.stat.extent).
    """
    extent = float(model.stat.extent)
    near = float(model.vis.map.znear) * extent
    far = float(model.vis.map.zfar) * extent
    # Avoid divide-by-zero; invalid pixels (0 or 1) are filtered later
    denom = 1.0 - depth_buffer * (1.0 - near / far)
    with np.errstate(divide="ignore", invalid="ignore"):
        metric = near / denom
    return metric.astype(np.float64)


def metric_depth_to_pointcloud_world(
    metric_depth: np.ndarray,
    depth_buffer: np.ndarray,
    model: mujoco.MjModel,  # type: ignore[name-defined]  # pylint: disable=no-member
    data: mujoco.MjData,  # type: ignore[name-defined]  # pylint: disable=no-member
    camera_id: int,
    *,
    min_depth: float = 1e-4,
    max_depth: float | None = None,
    depth_buffer_far: float = 0.999,
) -> np.ndarray:
    """Build a world-frame point cloud (N, 3) from a metric depth image.

    MuJoCo camera frame: +X right, +Y up, view direction along -Z.
    Pixel (u, v): u increases right, v increases down (matches flipud RGB/depth).
    """
    height, width = metric_depth.shape
    fovy_rad = np.deg2rad(float(model.cam_fovy[camera_id]))
    fy = height / (2.0 * np.tan(fovy_rad / 2.0))
    fx = fy
    cx = width / 2.0
    cy = height / 2.0

    u_grid, v_grid = np.meshgrid(np.arange(width), np.arange(height))
    z_forward = metric_depth
    x_cam = (u_grid - cx) * z_forward / fx
    y_cam = (cy - v_grid) * z_forward / fy
    z_cam = -z_forward

    valid = np.isfinite(metric_depth) & (metric_depth > min_depth)
    valid &= depth_buffer < depth_buffer_far
    if max_depth is not None:
        valid &= metric_depth < max_depth

    points_cam = np.stack([x_cam[valid], y_cam[valid], z_cam[valid]], axis=-1)
    rot = data.cam_xmat[camera_id].reshape(3, 3)
    pos = data.cam_xpos[camera_id]
    points_world = points_cam @ rot.T + pos
    return points_world.astype(np.float32)


def render_rgbd(
    sim: object,
    camera_name: str,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Call MjSim.render to obtain RGB and depth buffer."""
    result = sim.render(  # type: ignore[union-attr]
        width=width,
        height=height,
        camera_name=camera_name,
        depth=True,
        mode="offscreen",
    )
    if not isinstance(result, tuple):
        raise RuntimeError("Expected (rgb, depth) tuple when depth=True")
    rgb, depth = result
    return rgb, np.asarray(depth, dtype=np.float64)


def show_camera_rgb(
    rgb: np.ndarray,
    *,
    title: str,
    show: bool,
    save_path: Path | None,
) -> None:
    """Display or save the rendered RGB camera frame."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 7.5))
    ax.imshow(rgb)
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved camera view to {save_path.resolve()}")
    if show:
        plt.show()
    else:
        plt.close(fig)


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
        # gymnasium.make() wraps with OrderEnforcing; real env is on unwrapped
        base_env = env.unwrapped
        oc_env = base_env._object_centric_env  # pylint: disable=protected-access
        robot_env = oc_env._robot_env  # pylint: disable=protected-access
        if robot_env is None or robot_env.sim is None:
            raise RuntimeError("Robot simulation is not initialized after reset().")

        sim = robot_env.sim
        mj_model = sim.model.mj_model
        mj_data = sim.data.mj_data
        sim.forward()

        camera_name = resolve_camera_name(oc_env, args.camera)
        camera_id = sim.model.camera_name2id(camera_name)

        width = robot_env.camera_width
        height = robot_env.camera_height
        rgb, depth_buffer = render_rgbd(sim, camera_name, width, height)
        if args.show or args.save_rgb:
            show_camera_rgb(
                rgb,
                title=f"{camera_name} — {args.env_id}",
                show=args.show,
                save_path=Path(args.save_rgb) if args.save_rgb else None,
            )
        metric_depth = depth_buffer_to_metric(depth_buffer, mj_model)
        points_world = metric_depth_to_pointcloud_world(
            metric_depth,
            depth_buffer,
            mj_model,
            mj_data,
            camera_id,
            min_depth=args.min_depth,
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
            print(f"Saved point cloud to {out_path.resolve()} with shape {points_world.shape}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
