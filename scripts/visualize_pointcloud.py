#!/usr/bin/env python3
"""Visualize a world-frame point cloud saved by render_pointcloud.py (Open3D).

================================================================================
Dependencies
================================================================================
Open3D is not part of the KinDER package extras; install it in your environment::

    pip install open3d

================================================================================
Usage
================================================================================
From the repo root::

    python scripts/visualize_pointcloud.py
    python scripts/visualize_pointcloud.py --input pointcloud.npy
    python scripts/visualize_pointcloud.py --input pointcloud.npy --max-points 50000

Save a screenshot without keeping the window open (headless-friendly)::

    python scripts/visualize_pointcloud.py --input pointcloud.npy --save preview.png

================================================================================
What you will see
================================================================================
An interactive Open3D window: colored 3D points in MuJoCo world coordinates
(meters), colored by height (Z). Mouse: rotate, scroll zoom, Shift+drag pan, Q
to close. This is a single-view depth unprojection, not a mesh.
"""

from __future__ import annotations

import argparse
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


def load_pointcloud(path: Path) -> np.ndarray:
    """Load (N, 3) float point cloud; accept (N, 3+) and keep first three columns."""
    points = np.load(path)
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError(
            f"Expected array shape (N, 3) or wider, got {points.shape} from {path}"
        )
    return np.asarray(points[:, :3], dtype=np.float64)


def subsample(points: np.ndarray, max_points: int, seed: int) -> np.ndarray:
    """Uniformly subsample when N exceeds max_points."""
    n = points.shape[0]
    if n <= max_points:
        return points
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=max_points, replace=False)
    return points[idx]


def z_to_colors(z: np.ndarray) -> np.ndarray:
    """Map Z values to RGB in [0, 1] (blue low, green mid, yellow high)."""
    span = float(z.max() - z.min())
    t = (z - z.min()) / span if span > 0 else np.zeros_like(z)
    red = np.clip(1.5 * t - 0.5, 0.0, 1.0)
    green = np.clip(1.5 - np.abs(2.0 * t - 1.0), 0.0, 1.0)
    blue = np.clip(1.5 - 1.5 * t, 0.0, 1.0)
    return np.stack([red, green, blue], axis=1)


def show_pointcloud_open3d(
    points: np.ndarray,
    *,
    window_name: str,
    point_size: float,
    save_path: Path | None,
) -> None:
    """Display or screenshot a point cloud with Open3D."""
    o3d = _import_open3d()

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(z_to_colors(points[:, 2]))

    vis = o3d.visualization.Visualizer()
    vis.create_window(
        window_name=window_name,
        width=1280,
        height=720,
        visible=save_path is None,
    )
    vis.add_geometry(pcd)
    render_opt = vis.get_render_option()
    render_opt.point_size = float(point_size)
    render_opt.background_color = np.asarray([0.08, 0.08, 0.08])

    if save_path is not None:
        for _ in range(30):
            vis.poll_events()
            vis.update_renderer()
        vis.capture_screen_image(str(save_path), do_render=True)
        print(f"Saved screenshot to {save_path.resolve()}")
        vis.destroy_window()
        return

    print(
        "Open3D viewer: drag = rotate, scroll = zoom, "
        "Shift+drag = pan, Q or close window = exit"
    )
    vis.run()
    vis.destroy_window()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize a .npy point cloud from render_pointcloud.py (Open3D).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="See script docstring for usage. Requires: pip install open3d",
    )
    parser.add_argument(
        "--input",
        type=str,
        default="pointcloud.npy",
        help="Path to .npy file with shape (N, 3) (default: pointcloud.npy)",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=50_000,
        help="Max points to draw (uniform subsample if larger; default: 50000)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="RNG seed for subsampling",
    )
    parser.add_argument(
        "--point-size",
        type=float,
        default=2.0,
        help="Open3D point size in the render options (default: 2.0)",
    )
    parser.add_argument(
        "--save",
        type=str,
        default="",
        help="Save a PNG screenshot instead of opening an interactive window",
    )
    args = parser.parse_args()

    in_path = Path(args.input)
    if not in_path.is_file():
        raise SystemExit(f"Input not found: {in_path.resolve()}")

    points = load_pointcloud(in_path)
    if points.shape[0] == 0:
        raise SystemExit(f"No points in {in_path.resolve()}")

    drawn = subsample(points, args.max_points, args.seed)
    window_name = f"{in_path.name} ({drawn.shape[0]:,} / {points.shape[0]:,} points)"
    print(f"Loaded {points.shape[0]:,} points from {in_path.resolve()}")
    print(f"XYZ min: {points.min(axis=0)}")
    print(f"XYZ max: {points.max(axis=0)}")

    save_path = Path(args.save) if args.save else None
    show_pointcloud_open3d(
        drawn,
        window_name=window_name,
        point_size=args.point_size,
        save_path=save_path,
    )


if __name__ == "__main__":
    main()
