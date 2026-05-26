#!/usr/bin/env python3
"""Visualise RGB frames and point clouds stored in an HDF5 demo file.

Loads data written by ``demos_to_hdf5_with_pointclouds.py`` and renders a
combined view: per-camera RGB images in a top row and a merged 3-D point
cloud (cameras colour-coded) below.

Usage
-----
::

    # Show step 0 of demo 0
    python scripts/visualize_hdf5.py data.hdf5

    # Show step 10 of demo 2
    python scripts/visualize_hdf5.py data.hdf5 --demo 2 --step 10

    # Animate through all steps
    python scripts/visualize_hdf5.py data.hdf5 --animate

    # Limit cameras shown
    python scripts/visualize_hdf5.py data.hdf5 --cameras agentview_1 robot_wrist

    # Save a figure for every step (headless)
    python scripts/visualize_hdf5.py data.hdf5 --animate \\
        --save-figs /tmp/frames/step --no-show

    # List available demos and cameras without displaying
    python scripts/visualize_hdf5.py data.hdf5 --info
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import h5py  # type: ignore[import-untyped]  # pylint: disable=import-error
import matplotlib
import numpy as np
from numpy.typing import NDArray


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


def _select_backend(no_show: bool) -> None:
    """Switch to an interactive matplotlib backend unless --no-show is set."""
    if no_show:
        matplotlib.use("Agg")
        return
    for _b in ["Qt5Agg", "Qt6Agg", "TkAgg", "GTK3Agg", "WXAgg", "WebAgg"]:
        try:
            matplotlib.use(_b)
            import matplotlib.pyplot as _plt  # pylint: disable=import-outside-toplevel,reimported
            _plt.figure()
            _plt.close("all")
            return
        except Exception:  # pylint: disable=broad-except
            continue
    matplotlib.use("Agg")
    print(
        "Warning: no interactive matplotlib backend found — "
        "install PyQt5 for interactive display:  pip install PyQt5",
        file=sys.stderr,
    )


_no_show_early = "--no-show" in sys.argv
_select_backend(_no_show_early)

import matplotlib.pyplot as plt  # pylint: disable=wrong-import-position,ungrouped-imports
from matplotlib import animation  # pylint: disable=wrong-import-position,ungrouped-imports
from mpl_toolkits.mplot3d import Axes3D  # type: ignore[import-untyped]  # noqa: F401  # pylint: disable=wrong-import-position,ungrouped-imports,unused-import


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Visualise RGB frames and point clouds from an HDF5 demo file",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("hdf5_path", type=Path, help="Path to the HDF5 file")
    p.add_argument(
        "--demo",
        type=int,
        default=0,
        metavar="N",
        help="Demo index to visualise",
    )
    p.add_argument(
        "--step",
        type=int,
        default=0,
        metavar="N",
        help="Step index to show (ignored when --animate is set)",
    )
    p.add_argument(
        "--cameras",
        nargs="+",
        default=None,
        metavar="CAM",
        help="Camera names to show (default: all cameras in the file)",
    )
    p.add_argument(
        "--animate",
        action="store_true",
        help="Cycle through all steps of the demo",
    )
    p.add_argument(
        "--fps",
        type=float,
        default=10.0,
        metavar="N",
        help="Playback speed in frames per second (--animate only)",
    )
    p.add_argument(
        "--max-pts",
        type=int,
        default=20_000,
        metavar="N",
        help="Maximum points to render in the 3-D scatter (sub-sampled for speed)",
    )
    p.add_argument(
        "--save-figs",
        type=str,
        default=None,
        metavar="PREFIX",
        help=(
            "Save each figure as a PNG.  For a single step the output is "
            "PREFIX.png; when animating, PREFIX_{step:04d}.png per frame."
        ),
    )
    p.add_argument(
        "--no-show",
        action="store_true",
        help="Do not call plt.show() (useful for headless / scripted runs)",
    )
    p.add_argument(
        "--info",
        action="store_true",
        help="Print available demos and cameras then exit",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# HDF5 helpers
# ---------------------------------------------------------------------------


def _list_cameras(grp: h5py.Group) -> list[str]:
    """Return camera names found in an obs group (strip the _rgb suffix)."""
    cameras: list[str] = []
    for key in grp.keys():
        if key.endswith("_rgb") and not key.endswith("_pc_rgb"):
            cameras.append(key[: -len("_rgb")])
    return cameras


def _print_info(hf: h5py.File) -> None:
    """Print a summary of the HDF5 file contents."""
    data = hf["data"]
    print(f"env_id        : {data.attrs.get('env_id', 'N/A')}")
    print(f"total_episodes: {data.attrs.get('total_episodes', 'N/A')}")
    print(f"total_frames  : {data.attrs.get('total_frames', 'N/A')}")
    for key in sorted(data.keys()):
        grp = data[key]
        nf = grp.attrs.get("num_frames", "?")
        seed = grp.attrs.get("seed", "?")
        cams = _list_cameras(grp["obs"])
        print(f"  {key}: {nf} frames, seed={seed}, cameras={cams}")


def _load_step(
    obs_grp: h5py.Group,
    step: int,
    cameras: list[str],
) -> tuple[
    dict[str, NDArray[np.uint8]],
    dict[str, NDArray[np.float32]],
    dict[str, NDArray[np.uint8]],
]:
    """Load RGB images and point clouds for one step.

    Returns:
        rgb_imgs  : {cam: (H, W, 3) uint8}
        pc_xyz    : {cam: (N, 3) float32}  — NaN rows already stripped
        pc_rgb    : {cam: (N, 3) uint8}
    """
    rgb_imgs: dict[str, NDArray[np.uint8]] = {}
    pc_xyz: dict[str, NDArray[np.float32]] = {}
    pc_rgb: dict[str, NDArray[np.uint8]] = {}
    for cam in cameras:
        rgb_imgs[cam] = obs_grp[f"{cam}_rgb"][step]
        raw_xyz: NDArray[np.float32] = obs_grp[f"{cam}_pc_xyz"][step]
        raw_rgb: NDArray[np.uint8] = obs_grp[f"{cam}_pc_rgb"][step]
        # Strip NaN-padded rows inserted when the live point count < max_pts
        valid = np.isfinite(raw_xyz).all(axis=1)
        pc_xyz[cam] = raw_xyz[valid]
        pc_rgb[cam] = raw_rgb[valid]
    return rgb_imgs, pc_xyz, pc_rgb


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------


def _subsample(
    xyz: NDArray[np.float32],
    rgb: NDArray[np.uint8],
    max_pts: int,
    rng: np.random.Generator,
) -> tuple[NDArray[np.float32], NDArray[np.uint8]]:
    n = len(xyz)
    if n <= max_pts:
        return xyz, rgb
    idx = rng.choice(n, max_pts, replace=False)
    return xyz[idx], rgb[idx]


def _build_figure(
    cameras: list[str],
) -> tuple[plt.Figure, list[plt.Axes], plt.Axes]:
    """Create the combined RGB + point-cloud figure.

    Returns:
        fig     : the Figure
        rgb_axes: one Axes per camera (top row, imshow)
        pc_ax   : one 3-D Axes for the merged point cloud (bottom)
    """
    n_cams = len(cameras)
    # Top row: RGB images; bottom: full-width 3-D scatter
    fig = plt.figure(figsize=(4 * n_cams, 8))
    rgb_axes: list[plt.Axes] = []
    for col, cam in enumerate(cameras):
        ax = fig.add_subplot(2, n_cams, col + 1)
        ax.set_title(cam, fontsize=8)
        ax.axis("off")
        rgb_axes.append(ax)
    pc_ax = fig.add_subplot(2, 1, 2, projection="3d")
    pc_ax.set_xlabel("X (m)")
    pc_ax.set_ylabel("Y (m)")
    pc_ax.set_zlabel("Z (m)")  # type: ignore[attr-defined]
    return fig, rgb_axes, pc_ax


def _draw_step(
    fig: plt.Figure,
    rgb_axes: list[plt.Axes],
    pc_ax: plt.Axes,
    cameras: list[str],
    rgb_imgs: dict[str, NDArray[np.uint8]],
    pc_xyz: dict[str, NDArray[np.float32]],
    pc_rgb: dict[str, NDArray[np.uint8]],
    step: int,
    num_frames: int,
    max_pts: int,
    rng: np.random.Generator,
    title_prefix: str = "",
) -> None:
    """Update the figure in-place with data from *step*."""
    # RGB row
    for ax, cam in zip(rgb_axes, cameras):
        ax.cla()
        ax.imshow(rgb_imgs[cam])
        ax.set_title(cam, fontsize=8)
        ax.axis("off")

    # 3-D point cloud
    pc_ax.cla()
    pc_ax.set_xlabel("X (m)")
    pc_ax.set_ylabel("Y (m)")
    pc_ax.set_zlabel("Z (m)")  # type: ignore[attr-defined]

    pts_per_cam = max(max_pts // max(len(cameras), 1), 1)
    for cam in cameras:
        xyz = pc_xyz[cam]
        rgb = pc_rgb[cam]
        if len(xyz) == 0:
            continue
        sub_xyz, sub_rgb = _subsample(xyz, rgb, pts_per_cam, rng)
        colours = sub_rgb.astype(np.float32) / 255.0
        pc_ax.scatter(  # type: ignore[misc]
            sub_xyz[:, 0], sub_xyz[:, 1], sub_xyz[:, 2],
            c=colours, s=0.5, marker=".", linewidths=0,
            label=cam,
        )

    if any(len(pc_xyz[c]) > 0 for c in cameras):
        pc_ax.legend(loc="upper right", markerscale=8, fontsize="small")

    title = f"{title_prefix}Step {step} / {num_frames - 1}"
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Entry point."""
    args = _parse_args()

    if not args.hdf5_path.exists():
        print(f"Error: file not found: {args.hdf5_path}", file=sys.stderr)
        sys.exit(1)

    rng = np.random.default_rng(0)

    with h5py.File(args.hdf5_path, "r") as hf:
        if args.info:
            _print_info(hf)
            return

        data_grp = hf["data"]
        demo_key = f"demo_{args.demo}"
        if demo_key not in data_grp:
            available = sorted(data_grp.keys())
            print(
                f"Error: '{demo_key}' not found. Available: {available}",
                file=sys.stderr,
            )
            sys.exit(1)

        demo_grp = data_grp[demo_key]
        obs_grp: h5py.Group = demo_grp["obs"]
        num_frames: int = int(demo_grp.attrs["num_frames"])
        seed: Any = demo_grp.attrs.get("seed", "?")

        # Resolve cameras
        available_cams = _list_cameras(obs_grp)
        if args.cameras:
            missing = [c for c in args.cameras if c not in available_cams]
            if missing:
                print(
                    f"Error: cameras {missing} not in file. "
                    f"Available: {available_cams}",
                    file=sys.stderr,
                )
                sys.exit(1)
            cameras = args.cameras
        else:
            cameras = available_cams

        if not cameras:
            print("Error: no camera data found in the HDF5 file.", file=sys.stderr)
            sys.exit(1)

        title_prefix = (
            f"Demo {args.demo} (seed={seed})  |  "
            f"Cameras: {', '.join(cameras)}  |  "
        )
        print(
            f"Demo {args.demo}: {num_frames} frames, seed={seed}, "
            f"cameras={cameras}"
        )

        fig, rgb_axes, pc_ax = _build_figure(cameras)

        if args.animate:
            # --- Animation mode: update figure for each step ---
            def _update(step: int) -> list[Any]:
                rgb_imgs, pc_xyz, pc_rgb = _load_step(obs_grp, step, cameras)
                _draw_step(
                    fig, rgb_axes, pc_ax, cameras,
                    rgb_imgs, pc_xyz, pc_rgb,
                    step, num_frames, args.max_pts, rng,
                    title_prefix=title_prefix,
                )
                if args.save_figs:
                    path = f"{args.save_figs}_{step:04d}.png"
                    fig.savefig(path, dpi=120, bbox_inches="tight")
                fig.canvas.draw_idle()  # type: ignore[attr-defined]
                return []

            interval_ms = int(1000.0 / max(args.fps, 0.1))
            ani = animation.FuncAnimation(  # pylint: disable=unused-variable
                fig,
                _update,
                frames=num_frames,
                interval=interval_ms,
                repeat=False,
            )
            if not args.no_show:
                plt.show()

        else:
            # --- Single-step mode ---
            step = min(max(args.step, 0), num_frames - 1)
            rgb_imgs, pc_xyz, pc_rgb = _load_step(obs_grp, step, cameras)
            _draw_step(
                fig, rgb_axes, pc_ax, cameras,
                rgb_imgs, pc_xyz, pc_rgb,
                step, num_frames, args.max_pts, rng,
                title_prefix=title_prefix,
            )
            if args.save_figs:
                path = f"{args.save_figs}.png"
                fig.savefig(path, dpi=150, bbox_inches="tight")
                print(f"Saved: {path}")
            if not args.no_show:
                plt.show()


if __name__ == "__main__":
    main()
