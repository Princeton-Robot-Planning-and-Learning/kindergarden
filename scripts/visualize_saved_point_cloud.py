#!/usr/bin/env python3
"""Visualise a saved point cloud .npz file.

Loads a point cloud previously saved by the ``--save-point-cloud`` pytest
option and displays RGB renders, depth projections, and a 3-D scatter plot.

Usage
-----
::

    # Basic visualisation
    python scripts/visualize_saved_point_cloud.py /tmp/scene_pc.npz

    # Limit number of points rendered (for speed)
    python scripts/visualize_saved_point_cloud.py /tmp/scene_pc.npz --max-pts 20000

    # Save figures instead of showing interactively
    python scripts/visualize_saved_point_cloud.py /tmp/scene_pc.npz \\
        --save-figs /tmp/pc --no-show

    # Colour points by camera source instead of RGB
    python scripts/visualize_saved_point_cloud.py /tmp/scene_pc.npz --by-camera
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Matplotlib backend must be chosen before pyplot is imported.
# select_backend lives in the same scripts/ directory.
sys.path.insert(0, str(Path(__file__).parent))
from _viz_backend import select_backend  # pylint: disable=wrong-import-position

_no_show_early = "--no-show" in sys.argv
select_backend(_no_show_early)

import matplotlib.pyplot as plt  # pylint: disable=wrong-import-position,ungrouped-imports
from mpl_toolkits.mplot3d import (  # type: ignore[import-untyped]  # noqa: F401  # pylint: disable=wrong-import-position,ungrouped-imports,unused-import
    Axes3D,
)

_CAMERA_PALETTE = [
    [0.90, 0.30, 0.30],
    [0.30, 0.70, 0.30],
    [0.30, 0.50, 0.90],
    [0.90, 0.80, 0.20],
    [0.80, 0.40, 0.90],
    [0.20, 0.80, 0.80],
    [0.90, 0.60, 0.20],
    [0.60, 0.90, 0.40],
]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(
        description="Visualise a saved point cloud .npz file",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("npz_path", type=Path, help="Path to the .npz point cloud file")
    p.add_argument(
        "--max-pts",
        type=int,
        default=50_000,
        metavar="N",
        help="Maximum number of points to render (sub-sampled for speed)",
    )
    p.add_argument(
        "--by-camera",
        action="store_true",
        help="Colour points by camera source instead of RGB texture",
    )
    p.add_argument(
        "--save-figs",
        type=str,
        default=None,
        metavar="PREFIX",
        help="Save figures as PNGs with this path prefix (e.g. /tmp/pc)",
    )
    p.add_argument(
        "--no-show",
        action="store_true",
        help="Do not call plt.show() (useful for headless / scripted runs)",
    )
    return p.parse_args()


def load_point_cloud(
    path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Load xyz, rgb, camera_indices, and camera_names from a .npz file."""
    data = np.load(path)
    xyz: np.ndarray = data["xyz"]
    rgb: np.ndarray = data["rgb"]
    camera_indices: np.ndarray = data["camera_indices"]
    camera_names: list[str] = (
        json.loads(str(data["camera_names"])) if "camera_names" in data else []
    )
    return xyz, rgb, camera_indices, camera_names


def subsample(
    xyz: np.ndarray,
    colours: np.ndarray,
    max_pts: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Randomly sub-sample arrays to at most *max_pts* rows."""
    n = len(xyz)
    if n <= max_pts:
        return xyz, colours
    idx = np.random.default_rng(0).choice(n, max_pts, replace=False)
    return xyz[idx], colours[idx]


def plot_3d_rgb(
    xyz: np.ndarray,
    rgb: np.ndarray,
    max_pts: int,
    title: str,
) -> plt.Figure:
    """3-D scatter coloured by RGB texture."""
    sub_xyz, sub_rgb = subsample(xyz, rgb, max_pts)
    colours = sub_rgb.astype(np.float32) / 255.0

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(  # type: ignore[misc]
        sub_xyz[:, 0],
        sub_xyz[:, 1],
        sub_xyz[:, 2],
        c=colours,
        s=0.5,
        marker=".",
        linewidths=0,
    )
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")  # type: ignore[attr-defined]
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_3d_by_camera(
    xyz: np.ndarray,
    camera_indices: np.ndarray,
    camera_names: list[str],
    max_pts: int,
) -> plt.Figure:
    """3-D scatter with each camera source in a distinct colour."""
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    per_cam = max_pts // max(len(camera_names), 1)
    for cam_idx, cname in enumerate(camera_names):
        mask = camera_indices == cam_idx
        pts = xyz[mask]
        if len(pts) == 0:
            continue
        if len(pts) > per_cam:
            keep = np.random.default_rng(cam_idx).choice(
                len(pts), per_cam, replace=False
            )
            pts = pts[keep]
        colour = _CAMERA_PALETTE[cam_idx % len(_CAMERA_PALETTE)]
        ax.scatter(  # type: ignore[misc]
            pts[:, 0],
            pts[:, 1],
            pts[:, 2],
            c=[colour],
            s=0.5,
            marker=".",
            linewidths=0,
            label=cname,
        )

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")  # type: ignore[attr-defined]
    ax.set_title("Point cloud — per-camera colour")
    ax.legend(loc="upper right", markerscale=8, fontsize="small")
    fig.tight_layout()
    return fig


def plot_projections(
    xyz: np.ndarray,
    rgb: np.ndarray,
    max_pts: int,
    title: str,
) -> plt.Figure:
    """XY, XZ, YZ 2-D projections plus a 3-D view in a 1×4 panel."""
    sub_xyz, sub_rgb = subsample(xyz, rgb, max_pts)
    colours = sub_rgb.astype(np.float32) / 255.0

    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    fig.suptitle(title, fontsize=10)

    for ax, (xl, yl, xi, yi) in zip(
        axes[:3],
        [("X", "Y", 0, 1), ("X", "Z", 0, 2), ("Y", "Z", 1, 2)],
    ):
        ax.scatter(sub_xyz[:, xi], sub_xyz[:, yi], c=colours, s=0.3, linewidths=0)
        ax.set_xlabel(f"{xl} (m)")
        ax.set_ylabel(f"{yl} (m)")
        ax.set_title(f"{xl}–{yl} projection")
        ax.set_aspect("equal", "datalim")

    axes[3].remove()
    ax3d = fig.add_subplot(1, 4, 4, projection="3d")
    ax3d.scatter(  # type: ignore[misc]
        sub_xyz[:, 0],
        sub_xyz[:, 1],
        sub_xyz[:, 2],
        c=colours,
        s=0.3,
        linewidths=0,
    )
    ax3d.set_xlabel("X")
    ax3d.set_ylabel("Y")
    ax3d.set_zlabel("Z")  # type: ignore[attr-defined]
    ax3d.set_title("3-D view")

    fig.tight_layout()
    return fig


def main() -> None:
    """Load and visualise a saved point cloud file."""
    args = parse_args()

    if not args.npz_path.exists():
        print(f"Error: file not found: {args.npz_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading: {args.npz_path}")
    xyz, rgb, camera_indices, camera_names = load_point_cloud(args.npz_path)
    n = len(xyz)
    print(f"  Total points : {n:,}")
    print(f"  X range      : [{xyz[:, 0].min():.3f}, {xyz[:, 0].max():.3f}] m")
    print(f"  Y range      : [{xyz[:, 1].min():.3f}, {xyz[:, 1].max():.3f}] m")
    print(f"  Z range      : [{xyz[:, 2].min():.3f}, {xyz[:, 2].max():.3f}] m")
    print(f"  Cameras      : {camera_names}")
    for i, cname in enumerate(camera_names):
        count = int((camera_indices == i).sum())
        print(f"    {cname}: {count:,} points")

    if n == 0:
        print("Point cloud is empty — nothing to plot.")
        return

    stem = args.npz_path.stem

    # --- 3-D scatter (RGB or by-camera) ---
    if args.by_camera and camera_names:
        fig_3d = plot_3d_by_camera(xyz, camera_indices, camera_names, args.max_pts)
    else:
        fig_3d = plot_3d_rgb(xyz, rgb, args.max_pts, f"{stem} — RGB colour")
    if args.save_figs:
        path = f"{args.save_figs}_3d.png"
        fig_3d.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved: {path}")

    # --- 2-D projections panel ---
    fig_panel = plot_projections(xyz, rgb, args.max_pts, f"{stem} — projections")
    if args.save_figs:
        path = f"{args.save_figs}_projections.png"
        fig_panel.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved: {path}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
