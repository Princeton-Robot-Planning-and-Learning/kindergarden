#!/usr/bin/env python3
"""Replay a point cloud trajectory (.npz) from generate_demo_video.py --pointcloud in 3D.

================================================================================
Dependencies
================================================================================
Open3D (install yourself)::

    pip install open3d

================================================================================
Usage
================================================================================
From the repo root::

    python scripts/replay_pointcloud_trajectory.py \\
        docs/envs/assets/demo_gifs/.../demo_pointcloud.npz

    python scripts/replay_pointcloud_trajectory.py --input traj.npz --fps 10
    python scripts/replay_pointcloud_trajectory.py --input traj.npz --accumulate
    python scripts/replay_pointcloud_trajectory.py --input traj.npz --frame 0 --save frame0.png

================================================================================
Controls (interactive window)
================================================================================
- Space: pause / resume autoplay
- N or Right: next frame
- P or Left: previous frame
- R: restart from frame 0
- Q or Esc: quit

Mouse: drag rotate, scroll zoom, Shift+drag pan.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def _import_open3d():
    try:
        import open3d as o3d
    except ImportError as exc:
        raise SystemExit(
            "Open3D is required for this script. Install it yourself, e.g.:\n"
            "  pip install open3d"
        ) from exc
    return o3d


def z_to_colors(z: np.ndarray) -> np.ndarray:
    """Map Z values to RGB in [0, 1] (blue low, green mid, yellow high)."""
    span = float(z.max() - z.min())
    t = (z - z.min()) / span if span > 0 else np.zeros_like(z)
    red = np.clip(1.5 * t - 0.5, 0.0, 1.0)
    green = np.clip(1.5 - np.abs(2.0 * t - 1.0), 0.0, 1.0)
    blue = np.clip(1.5 - 1.5 * t, 0.0, 1.0)
    return np.stack([red, green, blue], axis=1)


@dataclass
class TrajectoryData:
    """One frame of world-frame points per demo timestep."""

    frames: list[np.ndarray]
    counts: np.ndarray
    camera: str
    env_id: str
    demo_seed: int | None
    max_points_per_frame: int
    has_rgb: bool
    rgb: np.ndarray | None


def load_trajectory_npz(path: Path) -> TrajectoryData:
    """Load trajectory written by ``PointCloudRecorder.save_npz``."""
    data = np.load(path, allow_pickle=False)
    if "points" not in data or "counts" not in data:
        raise ValueError(
            f"Expected 'points' and 'counts' in {path}; keys: {list(data.files)}"
        )

    stacked = np.asarray(data["points"], dtype=np.float64)
    counts = np.asarray(data["counts"], dtype=np.int32)
    if stacked.ndim != 3 or stacked.shape[2] != 3:
        raise ValueError(f"Expected points shape (T, K, 3), got {stacked.shape}")

    frames: list[np.ndarray] = []
    for t in range(stacked.shape[0]):
        n = int(counts[t])
        if n <= 0:
            frames.append(np.empty((0, 3), dtype=np.float64))
            continue
        chunk = stacked[t, :n]
        valid = np.isfinite(chunk).all(axis=1)
        frames.append(chunk[valid])

    camera = str(data["camera"]) if "camera" in data else ""
    env_id = str(data["env_id"]) if "env_id" in data else ""
    demo_seed = int(data["demo_seed"]) if "demo_seed" in data else None
    max_k = int(data["max_points_per_frame"]) if "max_points_per_frame" in data else (
        stacked.shape[1]
    )
    rgb = np.asarray(data["rgb"]) if "rgb" in data else None

    return TrajectoryData(
        frames=frames,
        counts=counts,
        camera=camera,
        env_id=env_id,
        demo_seed=demo_seed,
        max_points_per_frame=max_k,
        has_rgb=rgb is not None,
        rgb=rgb,
    )


def subsample_frame(points: np.ndarray, max_points: int, rng: np.random.Generator) -> np.ndarray:
    n = points.shape[0]
    if n <= max_points:
        return points
    idx = rng.choice(n, size=max_points, replace=False)
    return points[idx]


def merge_frames(frames: list[np.ndarray], end_index: int) -> np.ndarray:
    """Concatenate frames [0, end_index] inclusive."""
    parts = [f for f in frames[: end_index + 1] if f.shape[0] > 0]
    if not parts:
        return np.empty((0, 3), dtype=np.float64)
    return np.concatenate(parts, axis=0)


def _set_point_cloud(
    pcd: object,
    points: np.ndarray,
    o3d: object,
) -> None:
    if points.shape[0] == 0:
        pcd.points = o3d.utility.Vector3dVector(np.zeros((0, 3)))
        pcd.colors = o3d.utility.Vector3dVector(np.zeros((0, 3)))
        return
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(z_to_colors(points[:, 2]))


def _fit_view_to_points(vis: object, points: np.ndarray, *, reset: bool) -> None:
    """Frame the camera on ``points``; ``reset`` uses Open3D auto-fit."""
    if points.shape[0] == 0:
        return
    if reset:
        vis.reset_view_point(True)
        return
    center = points.mean(axis=0)
    extent = np.ptp(points, axis=0)
    max_extent = float(max(extent.max(), 1e-3))
    view_control = vis.get_view_control()
    view_control.set_lookat(center)
    view_control.set_front([0.4, -0.6, -0.7])
    view_control.set_up([0.0, 0.0, 1.0])
    # Lower zoom pulls the camera back (Open3D zoom is inverse distance).
    view_control.set_zoom(0.35 * max_extent)


def replay_interactive(
    trajectory: TrajectoryData,
    *,
    start_frame: int,
    fps: float,
    max_points: int,
    accumulate: bool,
    point_size: float,
    seed: int,
    save_path: Path | None,
) -> None:
    """Open3D player for the full trajectory."""
    o3d = _import_open3d()
    num_frames = len(trajectory.frames)
    if num_frames == 0:
        raise SystemExit("Trajectory has no frames.")

    rng = np.random.default_rng(seed)
    state = {
        "index": max(0, min(start_frame, num_frames - 1)),
        "playing": save_path is None,
        "done": False,
        "view_initialized": False,
        "last_rendered_index": None,
    }

    vis = o3d.visualization.VisualizerWithKeyCallback()
    title = trajectory.env_id or "point cloud trajectory"
    vis.create_window(window_name=title, width=1280, height=720, visible=save_path is None)
    pcd = o3d.geometry.PointCloud()
    render_opt = vis.get_render_option()
    render_opt.point_size = float(point_size)
    render_opt.background_color = np.asarray([0.08, 0.08, 0.08])

    def points_for_index(idx: int) -> np.ndarray:
        if accumulate:
            pts = merge_frames(trajectory.frames, idx)
        else:
            pts = trajectory.frames[idx]
        return subsample_frame(pts, max_points, rng)

    initial_points = points_for_index(state["index"])
    _set_point_cloud(pcd, initial_points, o3d)
    vis.add_geometry(pcd)

    def refresh(*, refit_view: bool = False) -> None:
        idx = state["index"]
        pts = points_for_index(idx)
        _set_point_cloud(pcd, pts, o3d)
        vis.update_geometry(pcd)
        vis.poll_events()
        vis.update_renderer()
        n_pts = int(pts.shape[0])
        if n_pts > 0 and (refit_view or not state["view_initialized"]):
            _fit_view_to_points(vis, pts, reset=True)
            state["view_initialized"] = True
            vis.poll_events()
            vis.update_renderer()
        state["last_rendered_index"] = idx
        print(f"Frame {idx + 1}/{num_frames}  ({n_pts:,} points displayed)")

    def clamp_and_refresh(delta: int) -> None:
        state["index"] = int(np.clip(state["index"] + delta, 0, num_frames - 1))
        refresh()

    def toggle_play(vis_obj: object) -> bool:  # pylint: disable=unused-argument
        state["playing"] = not state["playing"]
        print("Playing" if state["playing"] else "Paused")
        return False

    def next_frame(vis_obj: object) -> bool:  # pylint: disable=unused-argument
        state["playing"] = False
        clamp_and_refresh(1)
        return False

    def prev_frame(vis_obj: object) -> bool:  # pylint: disable=unused-argument
        state["playing"] = False
        clamp_and_refresh(-1)
        return False

    def restart(vis_obj: object) -> bool:  # pylint: disable=unused-argument
        state["index"] = 0
        state["playing"] = True
        state["view_initialized"] = False
        refresh(refit_view=True)
        return False

    def quit_vis(vis_obj: object) -> bool:  # pylint: disable=unused-argument
        state["done"] = True
        return False

    for key in (ord(" "),):
        vis.register_key_callback(key, toggle_play)
    for key in (ord("N"), ord("n"), 262):  # GLFW_KEY_RIGHT
        vis.register_key_callback(key, next_frame)
    for key in (ord("P"), ord("p"), 263):  # GLFW_KEY_LEFT
        vis.register_key_callback(key, prev_frame)
    for key in (ord("R"), ord("r")):
        vis.register_key_callback(key, restart)
    for key in (ord("Q"), ord("q"), 256):  # GLFW_KEY_ESCAPE
        vis.register_key_callback(key, quit_vis)

    refresh(refit_view=True)
    if save_path is not None:
        for _ in range(30):
            vis.poll_events()
            vis.update_renderer()
        vis.capture_screen_image(str(save_path), do_render=True)
        print(f"Saved screenshot to {save_path.resolve()}")
        vis.destroy_window()
        return

    print(
        "Controls: Space=pause/play, N/Right=next, P/Left=prev, R=restart, Q/Esc=quit\n"
        "Mouse: drag=rotate, scroll=zoom, Shift+drag=pan\n"
        "Tip: depth-camera clouds are thin from one angle; drag to rotate the view."
    )
    interval = 1.0 / max(fps, 0.1)
    while not state["done"]:
        if state["index"] != state["last_rendered_index"]:
            refresh()
        vis.poll_events()
        vis.update_renderer()
        if state["playing"]:
            time.sleep(interval)
            state["index"] = (state["index"] + 1) % num_frames
        else:
            time.sleep(0.03)

    vis.destroy_window()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay a .npz point cloud trajectory in 3D (Open3D).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="See script docstring. Requires: pip install open3d",
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        required=True,
        help="Path to *_pointcloud.npz from generate_demo_video.py --pointcloud",
    )
    parser.add_argument(
        "--frame",
        type=int,
        default=0,
        help="Starting frame index (default: 0)",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=10.0,
        help="Autoplay frames per second (default: 10)",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=50_000,
        help="Max points drawn per frame after subsampling (default: 50000)",
    )
    parser.add_argument(
        "--accumulate",
        action="store_true",
        help="Show merged points from frame 0 through current (builds up over time)",
    )
    parser.add_argument(
        "--point-size",
        type=float,
        default=4.0,
        help="Open3D point size (default: 4.0)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="RNG seed for display subsampling",
    )
    parser.add_argument(
        "--save",
        type=str,
        default="",
        help="Save a PNG of --frame and exit (no interactive loop)",
    )
    args = parser.parse_args()

    in_path = Path(args.input)
    if not in_path.is_file():
        raise SystemExit(f"Input not found: {in_path.resolve()}")

    trajectory = load_trajectory_npz(in_path)
    print(f"Loaded {len(trajectory.frames)} frames from {in_path.resolve()}")
    if trajectory.camera:
        print(f"Camera: {trajectory.camera}")
    if trajectory.env_id:
        print(f"Environment: {trajectory.env_id}")
    if trajectory.demo_seed is not None:
        print(f"Demo seed: {trajectory.demo_seed}")
    print(f"Points per frame (valid): min={int(trajectory.counts.min())}, "
          f"max={int(trajectory.counts.max())}, mean={trajectory.counts.mean():.0f}")

    save_path = Path(args.save) if args.save else None
    replay_interactive(
        trajectory,
        start_frame=args.frame if save_path is None else args.frame,
        fps=args.fps,
        max_points=args.max_points,
        accumulate=args.accumulate,
        point_size=args.point_size,
        seed=args.seed,
        save_path=save_path,
    )


if __name__ == "__main__":
    main()
