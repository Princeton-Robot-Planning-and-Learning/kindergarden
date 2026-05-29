#!/usr/bin/env python3
"""Visualise HDF5 demo data using Open3D for point clouds.

Opens two windows simultaneously:
  - **Open3D** — interactive 3-D point cloud viewer with full orbit/pan/zoom.
  - **Matplotlib** — per-camera RGB image grid (updated alongside Open3D).

Navigation inside the Open3D window
-------------------------------------
  Right arrow / ``N``  — next step
  Left arrow  / ``P``  — previous step
  ``R``                — jump back to step 0
  ``Q`` / close window — quit

Requirements
------------
::

    pip install open3d   # or: uv add open3d

Usage
-----
::

    # Interactive viewer, step 0 of demo 0
    python scripts/visualize_hdf5_open3d.py data.hdf5

    # Start at a specific step
    python scripts/visualize_hdf5_open3d.py data.hdf5 --step 20

    # Choose demo and cameras
    python scripts/visualize_hdf5_open3d.py data.hdf5 \\
        --demo 1 --cameras agentview_1 robot_wrist

    # Auto-play at 5 fps (Open3D window updates continuously)
    python scripts/visualize_hdf5_open3d.py data.hdf5 --animate --fps 5

    # Save per-step PNG screenshots (Open3D + matplotlib), headless
    python scripts/visualize_hdf5_open3d.py data.hdf5 --animate \\
        --save-figs /tmp/frames/step --no-show

    # List available demos and cameras
    python scripts/visualize_hdf5_open3d.py data.hdf5 --info
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from numpy.typing import NDArray

try:
    import open3d as o3d  # type: ignore[import-untyped]
except ImportError:
    print(
        "Error: open3d is not installed.\n  Install it with:  pip install open3d",
        file=sys.stderr,
    )
    sys.exit(1)

# Matplotlib backend must be chosen before pyplot is imported.
# We need a non-blocking interactive backend to update the RGB window while
# Open3D runs its own event loop. select_backend lives in the same scripts/ dir.
sys.path.insert(0, str(Path(__file__).parent))
from _viz_backend import select_backend  # pylint: disable=wrong-import-position

_no_show_early = "--no-show" in sys.argv
select_backend(_no_show_early)

import matplotlib.pyplot as plt  # pylint: disable=wrong-import-position,ungrouped-imports

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=("Visualise HDF5 demo data: Open3D point clouds + matplotlib RGB"),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("hdf5_path", type=Path, help="Path to the HDF5 file")
    p.add_argument(
        "--demo", type=int, default=0, metavar="N", help="Demo index to visualise"
    )
    p.add_argument(
        "--step", type=int, default=0, metavar="N", help="Initial step index"
    )
    p.add_argument(
        "--cameras",
        nargs="+",
        default=None,
        metavar="CAM",
        help="Cameras to show (default: all cameras in the file)",
    )
    p.add_argument(
        "--animate",
        action="store_true",
        help="Auto-play through all steps at --fps speed",
    )
    p.add_argument(
        "--fps",
        type=float,
        default=5.0,
        metavar="N",
        help="Playback speed for --animate",
    )
    p.add_argument(
        "--max-pts",
        type=int,
        default=50_000,
        metavar="N",
        help="Max points to render per step (sub-sampled for speed)",
    )
    p.add_argument(
        "--point-size",
        type=float,
        default=2.0,
        metavar="R",
        help="Point radius in the Open3D viewer",
    )
    p.add_argument(
        "--background",
        nargs=3,
        type=float,
        default=[0.1, 0.1, 0.1],
        metavar=("R", "G", "B"),
        help="Open3D background colour (0-1 per channel)",
    )
    p.add_argument(
        "--save-figs",
        type=str,
        default=None,
        metavar="PREFIX",
        help=(
            "Save screenshots for each step.  "
            "Writes PREFIX_{step:04d}_rgb.png and "
            "PREFIX_{step:04d}_pcd.png"
        ),
    )
    p.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open interactive windows (useful headless)",
    )
    p.add_argument(
        "--info",
        action="store_true",
        help="Print available demos and cameras then exit",
    )
    p.add_argument(
        "--completepointcloud",
        action="store_true",
        help=(
            "Visualise the mesh-based point cloud stored in obs/mesh_pc_xyz "
            "and obs/mesh_pc_geom_indices (written by --completepointcloud or "
            "--allpointcloud in demos_to_hdf5_with_pointclouds.py). "
            "Points are coloured by geom index."
        ),
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# HDF5 helpers
# ---------------------------------------------------------------------------


def _list_cameras(obs_grp: h5py.Group) -> list[str]:
    """Cameras in an obs group, derived from *_rgb dataset names."""
    return [
        k[: -len("_rgb")]
        for k in obs_grp.keys()
        if k.endswith("_rgb") and not k.endswith("_pc_rgb")
    ]


def _print_info(hf: h5py.File) -> None:
    data = hf["data"]
    print(f"env_id        : {data.attrs.get('env_id', 'N/A')}")
    print(f"total_episodes: {data.attrs.get('total_episodes', 'N/A')}")
    print(f"total_frames  : {data.attrs.get('total_frames', 'N/A')}")
    for key in sorted(data.keys()):
        grp = data[key]
        cams = _list_cameras(grp["obs"])
        print(
            f"  {key}: {grp.attrs.get('num_frames', '?')} frames, "
            f"seed={grp.attrs.get('seed', '?')}, cameras={cams}"
        )


def _load_step(
    obs_grp: h5py.Group,
    step: int,
    cameras: list[str],
    max_pts: int,
    rng: np.random.Generator,
) -> tuple[
    dict[str, NDArray[np.uint8]],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """Load and merge camera data for *step*.

    Returns:
        rgb_imgs   : {cam: (H, W, 3) uint8}
        merged_xyz : (N, 3) float64 — all cameras concatenated, NaN rows removed
        merged_rgb : (N, 3) float64 — colours normalised to [0, 1]
    """
    rgb_imgs: dict[str, NDArray[np.uint8]] = {}
    all_xyz: list[NDArray[np.float32]] = []
    all_rgb: list[NDArray[np.float32]] = []

    for cam in cameras:
        rgb_imgs[cam] = obs_grp[f"{cam}_rgb"][step]

        raw_xyz: NDArray[np.float32] = obs_grp[f"{cam}_pc_xyz"][step]
        raw_rgb: NDArray[np.uint8] = obs_grp[f"{cam}_pc_rgb"][step]

        # Strip NaN padding
        valid = np.isfinite(raw_xyz).all(axis=1)
        xyz = raw_xyz[valid].astype(np.float32)
        rgb_f = raw_rgb[valid].astype(np.float32) / 255.0
        all_xyz.append(xyz)
        all_rgb.append(rgb_f)

    if all_xyz:
        merged_xyz = np.concatenate(all_xyz, axis=0).astype(np.float64)
        merged_rgb = np.concatenate(all_rgb, axis=0).astype(np.float64)
    else:
        merged_xyz = np.empty((0, 3), dtype=np.float64)
        merged_rgb = np.empty((0, 3), dtype=np.float64)

    # Sub-sample for render speed
    n = len(merged_xyz)
    if n > max_pts:
        idx = rng.choice(n, max_pts, replace=False)
        merged_xyz = merged_xyz[idx]
        merged_rgb = merged_rgb[idx]

    return rgb_imgs, merged_xyz, merged_rgb


def _load_mesh_step(
    obs_grp: h5py.Group,
    step: int,
    max_pts: int,
    rng: np.random.Generator,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Load the mesh-based point cloud for *step*, colouring by geom index.

    Returns:
        xyz    : (N, 3) float64 world-frame positions (NaN rows stripped)
        colors : (N, 3) float64 RGB in [0, 1], one colour per unique geom
    """
    raw_xyz: NDArray[np.float32] = obs_grp["mesh_pc_xyz"][step]
    raw_idx: NDArray[np.int32] = obs_grp["mesh_pc_geom_indices"][step]

    valid = np.isfinite(raw_xyz).all(axis=1) & (raw_idx >= 0)
    xyz = raw_xyz[valid].astype(np.float64)
    geom_idx = raw_idx[valid]

    n = len(xyz)
    if n > max_pts:
        sel = rng.choice(n, max_pts, replace=False)
        xyz = xyz[sel]
        geom_idx = geom_idx[sel]

    # Colour each geom with a distinct hue using tab20 (cycles every 20 geoms)
    cmap = plt.get_cmap("tab20")
    colors: NDArray[np.float64] = cmap(geom_idx % 20)[:, :3]
    return xyz, colors


# ---------------------------------------------------------------------------
# Open3D helpers
# ---------------------------------------------------------------------------


def _make_pcd(
    xyz: NDArray[np.float64],
    rgb: NDArray[np.float64],
) -> "o3d.geometry.PointCloud":
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(rgb)
    return pcd


def _update_pcd(
    pcd: "o3d.geometry.PointCloud",
    xyz: NDArray[np.float64],
    rgb: NDArray[np.float64],
) -> None:
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(rgb)


# ---------------------------------------------------------------------------
# Matplotlib RGB panel helpers
# ---------------------------------------------------------------------------


def _build_rgb_figure(
    cameras: list[str],
) -> tuple[plt.Figure, list[plt.Axes]]:
    n = len(cameras)
    fig, axes = plt.subplots(
        1,
        n,
        figsize=(4 * n, 3.5),
        squeeze=False,
    )
    ax_list = list(axes[0])
    for ax, cam in zip(ax_list, cameras):
        ax.set_title(cam, fontsize=8)
        ax.axis("off")
    fig.tight_layout()
    return fig, ax_list


def _update_rgb_figure(
    fig: plt.Figure,
    axes: list[plt.Axes],
    cameras: list[str],
    rgb_imgs: dict[str, NDArray[np.uint8]],
    step: int,
    num_frames: int,
    title_prefix: str,
) -> None:
    for ax, cam in zip(axes, cameras):
        ax.cla()
        ax.imshow(rgb_imgs[cam])
        ax.set_title(cam, fontsize=8)
        ax.axis("off")
    fig.suptitle(f"{title_prefix}Step {step} / {num_frames - 1}", fontsize=9)
    fig.canvas.draw_idle()  # type: ignore[attr-defined]
    fig.canvas.flush_events()  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:  # pylint: disable=too-many-locals,too-many-statements
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
            print(
                f"Error: '{demo_key}' not found. "
                f"Available: {sorted(data_grp.keys())}",
                file=sys.stderr,
            )
            sys.exit(1)

        demo_grp = data_grp[demo_key]
        obs_grp: h5py.Group = demo_grp["obs"]
        num_frames: int = int(demo_grp.attrs["num_frames"])
        seed: Any = demo_grp.attrs.get("seed", "?")

        # ---- Validate mesh dataset when requested -------------------------
        if args.completepointcloud and "mesh_pc_xyz" not in obs_grp:
            print(
                "Error: mesh_pc_xyz not found in this demo. "
                "Re-generate the HDF5 with --completepointcloud or --allpointcloud.",
                file=sys.stderr,
            )
            sys.exit(1)

        # ---- Resolve cameras (still used for the RGB panel) ---------------
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

        if not cameras and not args.completepointcloud:
            print("Error: no camera data found.", file=sys.stderr)
            sys.exit(1)

        mode_label = "mesh PC" if args.completepointcloud else "camera PC"
        title_prefix = f"Demo {args.demo} (seed={seed})  |  [{mode_label}]  |  " + (
            ", ".join(cameras) + "  |  " if cameras else ""
        )
        print(
            f"Demo {args.demo}: {num_frames} frames, seed={seed}, "
            f"cameras={cameras}, mode={mode_label}"
        )
        print("  Keys: Right/N = next  Left/P = prev  R = reset  Q = quit")

        start_step = min(max(args.step, 0), num_frames - 1)

        def _load_pc(step: int) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
            """Return (xyz, colors) for the active point-cloud mode."""
            if args.completepointcloud:
                return _load_mesh_step(obs_grp, step, args.max_pts, rng)
            _, xyz, rgb_f = _load_step(obs_grp, step, cameras, args.max_pts, rng)
            return xyz, rgb_f

        # ---- Matplotlib RGB window (only when camera data is available) ----
        plt.ion()
        rgb_fig: plt.Figure | None = None
        rgb_axes: list[plt.Axes] = []
        if cameras:
            rgb_fig, rgb_axes = _build_rgb_figure(cameras)

        def _refresh_rgb(step: int) -> None:
            if rgb_fig is None:
                return
            rgb_imgs, _, _ = _load_step(obs_grp, step, cameras, args.max_pts, rng)
            _update_rgb_figure(
                rgb_fig,
                rgb_axes,
                cameras,
                rgb_imgs,
                step,
                num_frames,
                title_prefix,
            )

        # ---- Open3D window ------------------------------------------------
        xyz0, rgb0 = _load_pc(start_step)
        pcd = _make_pcd(xyz0, rgb0)

        if args.no_show:
            # Headless path: use OffscreenRenderer for Open3D screenshots
            renderer = o3d.visualization.rendering.OffscreenRenderer(  # pylint: disable=no-member
                1280, 720
            )
            mat = (
                o3d.visualization.rendering.MaterialRecord()  # pylint: disable=no-member
            )
            mat.shader = "defaultLit"
            mat.point_size = args.point_size
            renderer.scene.set_background(args.background + [1.0])

            steps = range(num_frames) if args.animate else [start_step]
            for step in steps:
                xyz, rgb_f = _load_pc(step)
                _update_pcd(pcd, xyz, rgb_f)
                renderer.scene.clear_geometry()
                renderer.scene.add_geometry("pcd", pcd, mat)
                _refresh_rgb(step)

                if args.save_figs:
                    Path(args.save_figs).parent.mkdir(parents=True, exist_ok=True)
                    pcd_path = f"{args.save_figs}_{step:04d}_pcd.png"
                    img = renderer.render_to_image()
                    o3d.io.write_image(pcd_path, img)  # pylint: disable=no-member
                    if rgb_fig is not None:
                        rgb_path = f"{args.save_figs}_{step:04d}_rgb.png"
                        rgb_fig.savefig(rgb_path, dpi=120, bbox_inches="tight")
                        print(f"  Saved step {step}: {pcd_path}, {rgb_path}")
                    else:
                        print(f"  Saved step {step}: {pcd_path}")
            return

        # Interactive Open3D visualizer
        vis = o3d.visualization.VisualizerWithKeyCallback()  # pylint: disable=no-member
        vis.create_window(
            window_name=(f"Point Cloud — {args.hdf5_path.name}  demo={args.demo}"),
            width=1024,
            height=768,
        )
        vis.add_geometry(pcd)

        opt = vis.get_render_option()
        opt.point_size = args.point_size
        opt.background_color = np.asarray(args.background)

        # Shared mutable state accessed inside key callbacks
        state: dict[str, Any] = {
            "step": start_step,
            "playing": args.animate,
            "last_time": time.monotonic(),
        }

        _refresh_rgb(start_step)

        def _go_to(step: int) -> None:
            step = max(0, min(step, num_frames - 1))
            state["step"] = step
            xyz, rgb_f = _load_pc(step)
            _update_pcd(pcd, xyz, rgb_f)
            vis.update_geometry(pcd)
            _refresh_rgb(step)
            if args.save_figs:
                Path(args.save_figs).parent.mkdir(parents=True, exist_ok=True)
                pcd_path = f"{args.save_figs}_{step:04d}_pcd.png"
                vis.capture_screen_image(pcd_path, do_render=True)
                if rgb_fig is not None:
                    rgb_path = f"{args.save_figs}_{step:04d}_rgb.png"
                    rgb_fig.savefig(rgb_path, dpi=120, bbox_inches="tight")

        def _cb_next(vis_: Any) -> bool:  # pylint: disable=unused-argument
            _go_to(state["step"] + 1)
            return False

        def _cb_prev(vis_: Any) -> bool:  # pylint: disable=unused-argument
            _go_to(state["step"] - 1)
            return False

        def _cb_reset(vis_: Any) -> bool:  # pylint: disable=unused-argument
            _go_to(0)
            return False

        def _cb_toggle_play(vis_: Any) -> bool:  # pylint: disable=unused-argument
            state["playing"] = not state["playing"]
            return False

        # Key codes: Right=262, Left=263, N=78, P=80, R=82, Space=32, Q=81
        vis.register_key_callback(262, _cb_next)  # Right arrow
        vis.register_key_callback(78, _cb_next)  # N
        vis.register_key_callback(263, _cb_prev)  # Left arrow
        vis.register_key_callback(80, _cb_prev)  # P
        vis.register_key_callback(82, _cb_reset)  # R
        vis.register_key_callback(32, _cb_toggle_play)  # Space

        interval = 1.0 / max(args.fps, 0.1)

        while vis.poll_events():
            if state["playing"]:
                now = time.monotonic()
                if now - state["last_time"] >= interval:
                    state["last_time"] = now
                    next_step = state["step"] + 1
                    if next_step >= num_frames:
                        state["playing"] = False
                    else:
                        _go_to(next_step)
            vis.update_renderer()

        vis.destroy_window()
        plt.close("all")


if __name__ == "__main__":
    main()
