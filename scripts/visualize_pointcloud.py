#!/usr/bin/env python3
"""Visualize a world-frame point cloud saved by render_pointcloud.py.

================================================================================
Usage
================================================================================
From the repo root::

    python scripts/visualize_pointcloud.py
    python scripts/visualize_pointcloud.py --input pointcloud.npy
    python scripts/visualize_pointcloud.py --input pointcloud.npy --max-points 50000

On a headless machine, save a PNG instead of opening a window::

    python scripts/visualize_pointcloud.py --input pointcloud.npy --save preview.png

================================================================================
What you will see
================================================================================
A 3D scatter of XYZ points in MuJoCo world coordinates (meters). Points are
colored by height (Z). This is a single-view depth unprojection, not a mesh.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


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


def plot_pointcloud(
    points: np.ndarray,
    *,
    title: str,
    point_size: float,
    save_path: Path | None,
) -> None:
    """Scatter plot with equal axis scaling."""
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    z = points[:, 2]
    ax.scatter(
        points[:, 0],
        points[:, 1],
        points[:, 2],
        s=point_size,
        c=z,
        cmap="viridis",
        linewidths=0,
        alpha=0.85,
    )
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title(title)

    # Equal aspect so the scene is not stretched.
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = 0.5 * (mins + maxs)
    radius = 0.5 * np.max(maxs - mins)
    if radius <= 0:
        radius = 1e-3
    for axis, c in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim), center):
        axis(c - radius, c + radius)

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150)
        print(f"Saved figure to {save_path.resolve()}")
    else:
        plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize a .npy point cloud from render_pointcloud.py.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="See script docstring for usage.",
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
        default=0.3,
        help="Matplotlib scatter marker size",
    )
    parser.add_argument(
        "--save",
        type=str,
        default="",
        help="Save PNG to this path instead of opening an interactive window",
    )
    args = parser.parse_args()

    in_path = Path(args.input)
    if not in_path.is_file():
        raise SystemExit(f"Input not found: {in_path.resolve()}")

    points = load_pointcloud(in_path)
    if points.shape[0] == 0:
        raise SystemExit(f"No points in {in_path.resolve()}")

    drawn = subsample(points, args.max_points, args.seed)
    title = f"{in_path.name} ({drawn.shape[0]:,} / {points.shape[0]:,} points)"
    print(f"Loaded {points.shape[0]:,} points from {in_path.resolve()}")
    print(f"XYZ min: {points.min(axis=0)}")
    print(f"XYZ max: {points.max(axis=0)}")

    save_path = Path(args.save) if args.save else None
    plot_pointcloud(
        drawn,
        title=title,
        point_size=args.point_size,
        save_path=save_path,
    )


if __name__ == "__main__":
    main()
