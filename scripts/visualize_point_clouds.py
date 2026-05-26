#!/usr/bin/env python3
"""Visualise Dynamic3D scene point clouds generated from MuJoCo depth rendering.

Renders RGB + depth images from every named camera in the scene, back-projects
the valid pixels into world-frame 3-D positions, and displays the merged point
cloud with matplotlib.  Optionally saves the point cloud to a .npz file.

Usage
-----
::

    # visualise the default BalanceBeam3D environment
    python scripts/visualize_point_clouds.py

    # choose a specific env and seed
    python scripts/visualize_point_clouds.py \\
        --env-id kinder/Dynamo3D-o3-v0 --seed 42

    # restrict to a subset of cameras
    python scripts/visualize_point_clouds.py \\
        --env-id kinder/BalanceBeam3D-o3-v0 \\
        --cameras task_view robot_base robot_wrist

    # save the point cloud to disk (numpy .npz) without showing a plot
    python scripts/visualize_point_clouds.py \\
        --save-npz /tmp/scene_pc.npz --no-show

    # render at higher resolution
    python scripts/visualize_point_clouds.py --width 1280 --height 960

"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import mujoco  # type: ignore[import-untyped]
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # type: ignore[import-untyped]  # noqa: F401  # pylint: disable=unused-import

# ── project root on path (must precede kinder imports) ──────────────────────
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

import kinder  # pylint: disable=wrong-import-position
from kinder.envs.dynamic3d.point_cloud import (  # pylint: disable=wrong-import-position
    PointCloud,
    generate_scene_point_cloud,
    get_sim_from_env,
)
from kinder.envs.dynamic3d.point_cloud import (  # pylint: disable=wrong-import-position
    depth_buffer_to_metric,
)


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

_CAMERA_PALETTE = [
    [0.90, 0.30, 0.30],  # red
    [0.30, 0.70, 0.30],  # green
    [0.30, 0.50, 0.90],  # blue
    [0.90, 0.80, 0.20],  # yellow
    [0.80, 0.40, 0.90],  # purple
    [0.20, 0.80, 0.80],  # cyan
    [0.90, 0.60, 0.20],  # orange
    [0.60, 0.90, 0.40],  # lime
]


def _subsample(
    xyz: np.ndarray,
    rgb: np.ndarray,
    max_pts: int = 50_000,
) -> tuple[np.ndarray, np.ndarray]:
    """Randomly sub-sample a point cloud for faster rendering."""
    n = len(xyz)
    if n <= max_pts:
        return xyz, rgb
    idx = np.random.default_rng(0).choice(n, max_pts, replace=False)
    return xyz[idx], rgb[idx]


def plot_point_cloud_rgb(
    pc: PointCloud,
    ax: "plt.Axes",
    *,
    max_pts: int = 50_000,
    point_size: float = 0.5,
) -> None:
    """Scatter-plot the point cloud coloured by the RGB texture."""
    xyz, rgb = _subsample(pc.xyz, pc.rgb, max_pts)
    colours = rgb.astype(np.float32) / 255.0
    ax.scatter(  # type: ignore[misc]
        xyz[:, 0], xyz[:, 1], xyz[:, 2],
        c=colours,
        s=point_size,
        marker=".",
        linewidths=0,
    )
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")  # type: ignore[attr-defined]
    ax.set_title("Point cloud — RGB colour")


def plot_point_cloud_by_camera(
    pc: PointCloud,
    ax: "plt.Axes",
    *,
    max_pts: int = 50_000,
    point_size: float = 0.5,
) -> None:
    """Scatter-plot the point cloud with each camera in a distinct colour."""
    for cam_idx, cname in enumerate(pc.camera_names):
        mask = pc.camera_indices == cam_idx
        if not np.any(mask):
            continue
        xyz_cam = pc.xyz[mask]
        n = len(xyz_cam)
        if n > max_pts // max(len(pc.camera_names), 1):
            rng = np.random.default_rng(cam_idx)
            keep = rng.choice(n, max_pts // max(len(pc.camera_names), 1), replace=False)
            xyz_cam = xyz_cam[keep]
        colour = _CAMERA_PALETTE[cam_idx % len(_CAMERA_PALETTE)]
        ax.scatter(  # type: ignore[misc]
            xyz_cam[:, 0], xyz_cam[:, 1], xyz_cam[:, 2],
            c=[colour],
            s=point_size,
            marker=".",
            linewidths=0,
            label=cname,
        )
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")  # type: ignore[attr-defined]
    ax.set_title("Point cloud — per-camera colour")
    ax.legend(loc="upper right", markerscale=8, fontsize="small")


def _build_name2id(mj_model: Any) -> dict[str, int]:
    """Build a camera-name → camera-id mapping from a MuJoCo model."""
    name2id: dict[str, int] = {}
    for i in range(mj_model.ncam):
        name = mujoco.mj_id2name(  # pylint: disable=no-member
            mj_model, mujoco.mjtObj.mjOBJ_CAMERA, i  # pylint: disable=no-member
        )
        if name is not None:
            name2id[name] = i
    return name2id


def plot_rgb_views(
    env_id: str,
    sim: Any,
    camera_names: list[str],
    width: int,
    height: int,
) -> "plt.Figure":
    """Show side-by-side RGB renders from each camera."""
    mj_model = sim.model.mj_model
    ctx = sim._render_context_offscreen  # pylint: disable=protected-access
    name2id = _build_name2id(mj_model)

    n_cams = len(camera_names)
    fig, axes = plt.subplots(
        1, n_cams, figsize=(4 * n_cams, 3.5), squeeze=False
    )
    fig.suptitle(f"RGB renders — {env_id}", fontsize=10)

    for col, cname in enumerate(camera_names):
        cam_id = name2id.get(cname)
        ax = axes[0][col]
        if cam_id is None:
            ax.set_title(f"{cname}\n(not found)")
            ax.axis("off")
            continue
        ctx.render(width=width, height=height, camera_id=cam_id)
        rgb_depth = ctx.read_pixels(width, height, depth=False)
        if isinstance(rgb_depth, tuple):
            rgb_img = rgb_depth[0]
        else:
            rgb_img = rgb_depth
        ax.imshow(rgb_img)
        ax.set_title(cname, fontsize=9)
        ax.axis("off")

    fig.tight_layout()
    return fig


def plot_depth_views(
    env_id: str,
    sim: Any,
    camera_names: list[str],
    width: int,
    height: int,
) -> "plt.Figure":
    """Show side-by-side linearised metric depth renders from each camera."""
    mj_model = sim.model.mj_model
    ctx = sim._render_context_offscreen  # pylint: disable=protected-access
    name2id = _build_name2id(mj_model)

    n_cams = len(camera_names)
    fig, axes = plt.subplots(
        1, n_cams, figsize=(4 * n_cams, 3.5), squeeze=False
    )
    fig.suptitle(f"Metric depth (m) — {env_id}", fontsize=10)

    for col, cname in enumerate(camera_names):
        cam_id = name2id.get(cname)
        ax = axes[0][col]
        if cam_id is None:
            ax.set_title(f"{cname}\n(not found)")
            ax.axis("off")
            continue
        ctx.render(width=width, height=height, camera_id=cam_id)
        rgb_depth = ctx.read_pixels(width, height, depth=True)
        assert isinstance(rgb_depth, tuple)
        _, depth_raw = rgb_depth
        assert depth_raw is not None
        z_m = depth_buffer_to_metric(depth_raw, mj_model)
        z_vis = np.where(np.isfinite(z_m), z_m, np.nan)
        im = ax.imshow(z_vis, cmap="turbo")
        ax.set_title(cname, fontsize=9)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="m")

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Statistics printer
# ---------------------------------------------------------------------------


def print_summary(pc: PointCloud, env_id: str) -> None:
    """Print a brief point-cloud summary to stdout."""
    print(f"\n{'='*60}")
    print(f"Environment : {env_id}")
    print(f"Total points: {len(pc):,}")
    if len(pc) > 0:
        print(f"X  range    : [{pc.xyz[:,0].min():.3f}, {pc.xyz[:,0].max():.3f}] m")
        print(f"Y  range    : [{pc.xyz[:,1].min():.3f}, {pc.xyz[:,1].max():.3f}] m")
        print(f"Z  range    : [{pc.xyz[:,2].min():.3f}, {pc.xyz[:,2].max():.3f}] m")
    print(f"Cameras     : {pc.camera_names}")
    for cam_idx, cname in enumerate(pc.camera_names):
        n = int((pc.camera_indices == cam_idx).sum())
        print(f"  {cname}: {n:,} points")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(
        description="Visualise Dynamic3D scene point clouds from MuJoCo depth rendering",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--env-id",
        default="kinder/BalanceBeam3D-o3-v0",
        help="kinder environment ID",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="RNG seed for env.reset()",
    )
    p.add_argument(
        "--cameras",
        nargs="*",
        default=None,
        metavar="CAM",
        help=(
            "Camera names to render (default: all named cameras in the model). "
            "Example: --cameras task_view robot_base"
        ),
    )
    p.add_argument(
        "--width",
        type=int,
        default=640,
        help="Render width in pixels",
    )
    p.add_argument(
        "--height",
        type=int,
        default=480,
        help="Render height in pixels",
    )
    p.add_argument(
        "--max-depth",
        type=float,
        default=10.0,
        help="Discard points further than this distance (metres)",
    )
    p.add_argument(
        "--min-depth",
        type=float,
        default=0.01,
        help="Discard points closer than this distance (metres)",
    )
    p.add_argument(
        "--max-pts",
        type=int,
        default=80_000,
        help="Maximum points shown per 3-D plot (sub-sampled for speed)",
    )
    p.add_argument(
        "--save-npz",
        type=str,
        default=None,
        metavar="PATH",
        help="Save the point cloud as a .npz file (xyz, rgb, camera_indices)",
    )
    p.add_argument(
        "--save-figs",
        type=str,
        default=None,
        metavar="PREFIX",
        help=(
            "Save figures as PNG files with this prefix "
            "(e.g. /tmp/pc → /tmp/pc_rgb.png, /tmp/pc_by_cam.png, …)"
        ),
    )
    p.add_argument(
        "--no-show",
        action="store_true",
        help="Do not call plt.show() (useful in headless / scripted runs)",
    )
    p.add_argument(
        "--no-3d",
        action="store_true",
        help="Skip the 3-D scatter plots (only show RGB / depth views)",
    )
    return p.parse_args()


def main() -> None:
    """Generate and visualise a scene point cloud for a Dynamic3D environment."""
    args = parse_args()

    # ── register environments ────────────────────────────────────────────────
    kinder.register_all_environments()

    # ── create and reset env ────────────────────────────────────────────────
    print(f"Loading environment: {args.env_id}")
    env = kinder.make(args.env_id)
    env.reset(seed=args.seed)
    print(f"Environment reset (seed={args.seed})")

    # ── get sim ─────────────────────────────────────────────────────────────
    sim = get_sim_from_env(env)

    # ── discover available cameras ──────────────────────────────────────────
    mj_model = sim.model.mj_model
    all_cam_names = list(_build_name2id(mj_model).keys())

    print(f"Available cameras in model: {all_cam_names}")

    camera_names: list[str] | None = args.cameras
    if camera_names is not None:
        # Validate
        unknown = [c for c in camera_names if c not in all_cam_names]
        if unknown:
            print(
                f"\nWarning: cameras not found in model and will be skipped: {unknown}"
            )
        camera_names = [c for c in camera_names if c in all_cam_names]
        if not camera_names:
            print("No valid cameras to render.  Exiting.")
            env.close()
            return
    else:
        camera_names = all_cam_names

    # ── generate point cloud ─────────────────────────────────────────────────
    print(
        f"\nGenerating point cloud from {len(camera_names)} camera(s): {camera_names}"
    )
    pc = generate_scene_point_cloud(
        sim,
        camera_names=camera_names,
        width=args.width,
        height=args.height,
        max_depth=args.max_depth,
        min_depth=args.min_depth,
    )
    print_summary(pc, args.env_id)

    # ── optionally save ──────────────────────────────────────────────────────
    if args.save_npz and len(pc) > 0:
        save_path = Path(args.save_npz)
        np.savez_compressed(save_path, **pc.to_dict())
        print(f"\nPoint cloud saved to: {save_path}")

    # ── RGB and depth views ──────────────────────────────────────────────────
    fig_rgb = plot_rgb_views(args.env_id, sim, camera_names, args.width, args.height)
    if args.save_figs:
        fig_rgb.savefig(f"{args.save_figs}_rgb_views.png", dpi=150, bbox_inches="tight")
        print(f"Saved: {args.save_figs}_rgb_views.png")

    fig_depth = plot_depth_views(
        args.env_id, sim, camera_names, args.width, args.height
    )
    if args.save_figs:
        fig_depth.savefig(
            f"{args.save_figs}_depth_views.png", dpi=150, bbox_inches="tight"
        )
        print(f"Saved: {args.save_figs}_depth_views.png")

    # ── 3-D point cloud plots ─────────────────────────────────────────────────
    if not args.no_3d and len(pc) > 0:
        # Figure 1: RGB colour
        fig_3d_rgb = plt.figure(figsize=(9, 7))
        ax_rgb = fig_3d_rgb.add_subplot(111, projection="3d")
        plot_point_cloud_rgb(pc, ax_rgb, max_pts=args.max_pts)
        fig_3d_rgb.tight_layout()
        if args.save_figs:
            fig_3d_rgb.savefig(
                f"{args.save_figs}_3d_rgb.png", dpi=150, bbox_inches="tight"
            )
            print(f"Saved: {args.save_figs}_3d_rgb.png")

        # Figure 2: per-camera colour
        fig_3d_cam = plt.figure(figsize=(9, 7))
        ax_cam = fig_3d_cam.add_subplot(111, projection="3d")
        plot_point_cloud_by_camera(pc, ax_cam, max_pts=args.max_pts)
        fig_3d_cam.tight_layout()
        if args.save_figs:
            fig_3d_cam.savefig(
                f"{args.save_figs}_3d_by_camera.png", dpi=150, bbox_inches="tight"
            )
            print(f"Saved: {args.save_figs}_3d_by_camera.png")

        # Figure 3: combined panel — 2D projections (XY, XZ, YZ) + 3D
        fig_panel, axes_panel = plt.subplots(1, 4, figsize=(18, 4))
        fig_panel.suptitle(
            f"Point cloud projections — {args.env_id} (seed={args.seed})", fontsize=10
        )

        sub_xyz, sub_rgb = _subsample(pc.xyz, pc.rgb, args.max_pts)
        colours = sub_rgb.astype(np.float32) / 255.0

        proj_labels = [("X", "Y", 0, 1), ("X", "Z", 0, 2), ("Y", "Z", 1, 2)]
        for ax, (xl, yl, xi, yi) in zip(axes_panel[:3], proj_labels):
            ax.scatter(sub_xyz[:, xi], sub_xyz[:, yi], c=colours, s=0.3, linewidths=0)
            ax.set_xlabel(f"{xl} (m)")
            ax.set_ylabel(f"{yl} (m)")
            ax.set_title(f"{xl}–{yl} projection")
            ax.set_aspect("equal", "datalim")

        # 3-D in last panel slot (recreate as 3-D axes)
        axes_panel[3].remove()
        ax_3d = fig_panel.add_subplot(1, 4, 4, projection="3d")
        ax_3d.scatter(
            sub_xyz[:, 0], sub_xyz[:, 1], sub_xyz[:, 2],
            c=colours, s=0.3, linewidths=0
        )
        ax_3d.set_xlabel("X")
        ax_3d.set_ylabel("Y")
        ax_3d.set_zlabel("Z")  # type: ignore[attr-defined]
        ax_3d.set_title("3-D view")

        fig_panel.tight_layout()
        if args.save_figs:
            fig_panel.savefig(
                f"{args.save_figs}_panel.png", dpi=150, bbox_inches="tight"
            )
            print(f"Saved: {args.save_figs}_panel.png")

    elif not args.no_3d and len(pc) == 0:
        print("\nWarning: point cloud is empty — nothing to plot in 3-D.")

    # ── show ─────────────────────────────────────────────────────────────────
    if not args.no_show:
        plt.show()

    env.close()
    print("\nDone.")


if __name__ == "__main__":
    main()
