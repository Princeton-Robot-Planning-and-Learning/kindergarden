#!/usr/bin/env python3
"""Replay demonstration pickle files and save RGB images + point clouds to HDF5.

For each demo in a given directory, the script:
  1. Creates the environment from the stored ``env_id``.
  2. Resets with the stored seed and steps through the stored actions.
  3. At every step renders each requested camera and generates a per-camera
     point cloud using depth back-projection.
  4. Appends the results to an HDF5 file with the structure::

       data/
         attrs: env_id, total_episodes, total_frames
         demo_0/
           attrs: num_frames, seed
           obs/
             state           (N, obs_dim)   float32
             {cam}_rgb       (N, H, W, 3)   uint8
             {cam}_pc_xyz    (N, max_pts, 3) float32   (NaN-padded if fewer pts)
             {cam}_pc_rgb    (N, max_pts, 3) uint8     (zero-padded if fewer pts)
           actions           (N, act_dim)   float32
         demo_1/
           ...

Usage
-----
::

    # Convert all demos in a directory (default cameras, 640×480 renders)
    python scripts/demos_to_hdf5_with_pointclouds.py \\
        --demo-dir demos/SweepIntoDrawer3D-o5 \\
        --output /tmp/sweep_pcs.hdf5

    # Choose specific cameras and render size
    python scripts/demos_to_hdf5_with_pointclouds.py \\
        --demo-dir demos/SweepIntoDrawer3D-o5 \\
        --output /tmp/sweep_pcs.hdf5 \\
        --cameras agentview_1 robot_wrist \\
        --render-width 320 --render-height 240

    # Fix maximum number of points per camera per step (padded with NaN/0)
    python scripts/demos_to_hdf5_with_pointclouds.py \\
        --demo-dir demos/SweepIntoDrawer3D-o5 \\
        --output /tmp/sweep_pcs.hdf5 \\
        --max-pts 8192

    # Limit the number of demos processed (useful for quick tests)
    python scripts/demos_to_hdf5_with_pointclouds.py \\
        --demo-dir demos/SweepIntoDrawer3D-o5 \\
        --output /tmp/sweep_pcs.hdf5 \\
        --max-demos 5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import gymnasium
import h5py  # pylint: disable=import-error
import numpy as np
from numpy.typing import NDArray

import kinder
from kinder.envs.dynamic3d.point_cloud import (
    PointCloud,
    generate_scene_point_cloud,
    get_sim_from_env,
)
from kinder.utils import load_demo

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Replay demos and save RGB + point clouds to HDF5",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--demo-dir",
        type=Path,
        required=True,
        metavar="DIR",
        help="Directory containing demo .p files (searched recursively)",
    )
    p.add_argument(
        "--output",
        type=Path,
        required=True,
        metavar="FILE",
        help="Output HDF5 file path (created or overwritten)",
    )
    p.add_argument(
        "--cameras",
        nargs="+",
        default=None,
        metavar="CAM",
        help=(
            "Camera names to render.  Pass None to use every named camera "
            "in the model (default: all cameras)"
        ),
    )
    p.add_argument(
        "--render-width",
        type=int,
        default=640,
        metavar="W",
        help="Render width in pixels",
    )
    p.add_argument(
        "--render-height",
        type=int,
        default=480,
        metavar="H",
        help="Render height in pixels",
    )
    p.add_argument(
        "--max-pts",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Fixed number of points per camera per step.  "
            "Point clouds with fewer points are NaN/zero-padded; "
            "those with more are randomly sub-sampled.  "
            "Pass None to store variable-length arrays (default: None)"
        ),
    )
    p.add_argument(
        "--max-demos",
        type=int,
        default=None,
        metavar="N",
        help="Stop after processing this many demos (default: all)",
    )
    p.add_argument(
        "--max-depth",
        type=float,
        default=10.0,
        metavar="M",
        help="Discard point cloud points farther than this distance in metres",
    )
    p.add_argument(
        "--min-depth",
        type=float,
        default=0.01,
        metavar="M",
        help="Discard point cloud points closer than this distance in metres",
    )
    return p.parse_args()


def _find_demo_files(demo_dir: Path) -> list[Path]:
    """Return sorted list of .p demo files under *demo_dir*."""
    files = sorted(demo_dir.glob("**/*.p"))
    if not files:
        print(f"Error: no .p demo files found in {demo_dir}", file=sys.stderr)
        sys.exit(1)
    return files


def _make_env(env_id: str) -> gymnasium.Env:
    """Create a kinder environment suitable for replay."""
    kinder.register_all_environments()
    make_kwargs: dict[str, Any] = {"render_mode": "rgb_array"}
    entrypoint = gymnasium.registry[env_id].entry_point
    assert isinstance(entrypoint, str)
    if "kinematic3d" in entrypoint:
        make_kwargs["realistic_bg"] = False
    return kinder.make(env_id, **make_kwargs)


def _pad_or_subsample(
    xyz: NDArray[np.float32],
    rgb: NDArray[np.uint8],
    max_pts: int,
    rng: np.random.Generator,
) -> tuple[NDArray[np.float32], NDArray[np.uint8]]:
    """Return arrays with exactly *max_pts* rows.

    If *n < max_pts*:  pad xyz with NaN, rgb with 0.
    If *n > max_pts*:  randomly sub-sample without replacement.
    """
    n = len(xyz)
    if n == max_pts:
        return xyz, rgb
    if n > max_pts:
        idx = rng.choice(n, max_pts, replace=False)
        return xyz[idx], rgb[idx]
    # Pad
    pad = max_pts - n
    xyz_pad = np.full((pad, 3), np.nan, dtype=np.float32)
    rgb_pad = np.zeros((pad, 3), dtype=np.uint8)
    return (
        np.concatenate([xyz, xyz_pad], axis=0),
        np.concatenate([rgb, rgb_pad], axis=0),
    )


def _render_rgb(
    env: gymnasium.Env,
    cam_name: str,
    width: int,
    height: int,
) -> NDArray[np.uint8]:
    """Render a single camera's RGB image via the environment render API."""
    # kinder environments expose render() for the primary camera; for
    # per-camera access we use the underlying MjSim render context directly.
    sim = get_sim_from_env(env)
    ctx = sim._render_context_offscreen  # pylint: disable=protected-access
    import mujoco  # pylint: disable=import-outside-toplevel  # type: ignore[import]

    mj_model = sim.model.mj_model  # type: ignore[attr-defined]
    cam_id = mujoco.mj_name2id(  # pylint: disable=no-member
        mj_model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name  # pylint: disable=no-member
    )
    if cam_id < 0:
        raise ValueError(f"Camera '{cam_name}' not found in model.")
    ctx.render(width=width, height=height, camera_id=cam_id)
    rgb_img: NDArray[np.uint8] = ctx.read_pixels(width, height, depth=False)
    return rgb_img


# ---------------------------------------------------------------------------
# Per-step capture
# ---------------------------------------------------------------------------


def _capture_step(
    env: gymnasium.Env,
    camera_names: list[str],
    width: int,
    height: int,
    max_depth: float,
    min_depth: float,
    max_pts: int | None,
    rng: np.random.Generator,
) -> dict[str, Any]:
    """Capture RGB images and point clouds for all cameras at the current state.

    Returns a dict with keys:
    - ``"{cam}_rgb"``    → (H, W, 3) uint8
    - ``"{cam}_pc_xyz"`` → (N, 3) or (max_pts, 3) float32
    - ``"{cam}_pc_rgb"`` → (N, 3) or (max_pts, 3) uint8
    """
    sim = get_sim_from_env(env)

    # Generate merged point cloud (all cameras in one call for efficiency)
    full_pc: PointCloud = generate_scene_point_cloud(
        sim,
        camera_names=camera_names,
        width=width,
        height=height,
        max_depth=max_depth,
        min_depth=min_depth,
    )

    result: dict[str, Any] = {}
    for cam_name in camera_names:
        # RGB image
        rgb_img = _render_rgb(env, cam_name, width, height)
        result[f"{cam_name}_rgb"] = rgb_img

        # Per-camera point cloud
        if cam_name in full_pc.camera_names:
            cam_pc = full_pc.filter_by_camera(cam_name)
            xyz, rgb = cam_pc.xyz, cam_pc.rgb
        else:
            xyz = np.empty((0, 3), dtype=np.float32)
            rgb = np.empty((0, 3), dtype=np.uint8)

        if max_pts is not None:
            xyz, rgb = _pad_or_subsample(xyz, rgb, max_pts, rng)

        result[f"{cam_name}_pc_xyz"] = xyz
        result[f"{cam_name}_pc_rgb"] = rgb

    return result


# ---------------------------------------------------------------------------
# HDF5 writers
# ---------------------------------------------------------------------------


def _create_datasets(
    grp: h5py.Group,
    obs_grp: h5py.Group,
    first_step: dict[str, Any],
    obs0: NDArray,
    actions: list[Any],
    camera_names: list[str],
    num_frames: int,
) -> None:
    """Pre-allocate HDF5 datasets for one demo group."""
    act_arr = np.array(actions, dtype=np.float32)
    obs_dim = obs0.shape[0] if obs0.ndim == 1 else obs0.shape[-1]
    act_dim = act_arr.shape[-1]

    obs_grp.create_dataset(
        "state",
        shape=(num_frames, obs_dim),
        dtype=np.float32,
        compression="gzip",
    )
    grp.create_dataset(
        "actions",
        shape=(num_frames, act_dim),
        dtype=np.float32,
        compression="gzip",
    )

    for cam_name in camera_names:
        rgb = first_step[f"{cam_name}_rgb"]
        obs_grp.create_dataset(
            f"{cam_name}_rgb",
            shape=(num_frames, *rgb.shape),
            dtype=np.uint8,
            compression="gzip",
        )

        xyz = first_step[f"{cam_name}_pc_xyz"]
        obs_grp.create_dataset(
            f"{cam_name}_pc_xyz",
            shape=(num_frames, *xyz.shape),
            dtype=np.float32,
            compression="gzip",
        )

        pc_rgb = first_step[f"{cam_name}_pc_rgb"]
        obs_grp.create_dataset(
            f"{cam_name}_pc_rgb",
            shape=(num_frames, *pc_rgb.shape),
            dtype=np.uint8,
            compression="gzip",
        )


def _write_step(
    grp: h5py.Group,
    obs_grp: h5py.Group,
    step_idx: int,
    obs: NDArray,
    action: Any,
    capture: dict[str, Any],
    camera_names: list[str],
    cam_sizes: dict[str, int],
    rng: np.random.Generator,
) -> None:
    """Write one step's data into pre-allocated HDF5 datasets.

    Point cloud arrays are padded or sub-sampled to the fixed per-camera sizes
    established at step 0, so every row has a consistent shape.
    """
    obs_grp["state"][step_idx] = obs.astype(np.float32)
    grp["actions"][step_idx] = np.asarray(action, dtype=np.float32)

    for cam_name in camera_names:
        obs_grp[f"{cam_name}_rgb"][step_idx] = capture[f"{cam_name}_rgb"]

        xyz = capture[f"{cam_name}_pc_xyz"]
        rgb = capture[f"{cam_name}_pc_rgb"]
        size = cam_sizes[cam_name]
        if len(xyz) != size:
            xyz, rgb = _pad_or_subsample(xyz, rgb, size, rng)
        obs_grp[f"{cam_name}_pc_xyz"][step_idx] = xyz
        obs_grp[f"{cam_name}_pc_rgb"][step_idx] = rgb


# ---------------------------------------------------------------------------
# Main replay loop
# ---------------------------------------------------------------------------


def _process_demo(
    demo_path: Path,
    env: gymnasium.Env | None,
    hdf5_file: h5py.File,
    demo_idx: int,
    camera_names: list[str] | None,
    width: int,
    height: int,
    max_depth: float,
    min_depth: float,
    max_pts: int | None,
    rng: np.random.Generator,
) -> tuple[gymnasium.Env, list[str], int]:
    """Replay one demo and write it to HDF5.

    Returns:
        Tuple of (env, resolved_cameras, num_frames).
    """
    demo_data = load_demo(demo_path)
    env_id: str = demo_data["env_id"]
    actions: list[Any] = demo_data["actions"]
    seed: int = demo_data["seed"]

    if not actions:
        print(f"  Skipping {demo_path.name}: no actions")
        return env, camera_names or [], 0  # type: ignore[return-value]

    # Create / reuse environment
    if env is None:
        print(f"  Creating environment: {env_id}")
        env = _make_env(env_id)

    # Reset
    obs_np, _ = env.reset(seed=seed)

    # Resolve camera names on first use
    if camera_names is None:
        sim = get_sim_from_env(env)
        import mujoco  # pylint: disable=import-outside-toplevel  # type: ignore[import]

        mj_model = sim.model.mj_model  # type: ignore[attr-defined]
        camera_names = []
        for i in range(mj_model.ncam):
            cam_name = mujoco.mj_id2name(  # pylint: disable=no-member
                mj_model, mujoco.mjtObj.mjOBJ_CAMERA, i  # pylint: disable=no-member
            )
            if cam_name is not None:
                camera_names.append(cam_name)
        print(f"  Auto-detected cameras: {camera_names}")

    num_frames = len(actions)
    grp_name = f"demo_{demo_idx}"
    grp = hdf5_file.require_group(f"data/{grp_name}")
    grp.attrs["num_frames"] = num_frames
    grp.attrs["seed"] = seed
    obs_grp = grp.require_group("obs")

    # Capture step 0 to establish per-camera point-cloud sizes for the whole
    # demo.  The number of valid depth pixels fluctuates slightly each step,
    # so we pin each camera to the count from step 0 (or --max-pts when set)
    # and pad / sub-sample every subsequent step to match.
    capture0 = _capture_step(
        env, camera_names, width, height, max_depth, min_depth, max_pts, rng
    )
    cam_sizes: dict[str, int] = {
        cam: len(capture0[f"{cam}_pc_xyz"]) for cam in camera_names
    }
    _create_datasets(grp, obs_grp, capture0, obs_np, actions, camera_names, num_frames)
    _write_step(grp, obs_grp, 0, obs_np, actions[0], capture0, camera_names,
                cam_sizes, rng)

    # Step through remaining actions
    for step_idx in range(1, num_frames):
        obs_np, _reward, _term, _trunc, _ = env.step(actions[step_idx - 1])

        # Capture AFTER the step that produced this obs
        capture = _capture_step(
            env, camera_names, width, height, max_depth, min_depth, max_pts, rng
        )
        _write_step(
            grp, obs_grp, step_idx, obs_np, actions[step_idx], capture,
            camera_names, cam_sizes, rng,
        )

    # Final step: execute last action to get terminal observation
    obs_np, _reward, _term, _trunc, _ = env.step(actions[-1])
    # (terminal obs is not stored; dataset length == len(actions))

    print(f"  demo_{demo_idx}: {num_frames} frames, seed={seed}")
    return env, camera_names, num_frames


def main() -> None:
    """Entry point."""
    args = _parse_args()

    demo_files = _find_demo_files(args.demo_dir)
    if args.max_demos is not None:
        demo_files = demo_files[: args.max_demos]

    print(f"Found {len(demo_files)} demo(s) in {args.demo_dir}")
    print(f"Output: {args.output}")

    rng = np.random.default_rng(0)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.output, "w") as hf:
        data_grp = hf.require_group("data")

        env: gymnasium.Env | None = None
        resolved_cameras: list[str] | None = args.cameras
        total_frames = 0
        env_id_recorded: str | None = None

        for demo_idx, demo_path in enumerate(demo_files):
            print(f"[{demo_idx + 1}/{len(demo_files)}] {demo_path.name}")
            try:
                demo_data = load_demo(demo_path)
                env_id_recorded = demo_data["env_id"]
                env, resolved_cameras, nf = _process_demo(
                    demo_path=demo_path,
                    env=env,
                    hdf5_file=hf,
                    demo_idx=demo_idx,
                    camera_names=resolved_cameras,
                    width=args.render_width,
                    height=args.render_height,
                    max_depth=args.max_depth,
                    min_depth=args.min_depth,
                    max_pts=args.max_pts,
                    rng=rng,
                )
                total_frames += nf
            except Exception as exc:  # pylint: disable=broad-except
                print(f"  ERROR: {exc}", file=sys.stderr)
                continue

        # Write file-level attributes
        data_grp.attrs["env_id"] = env_id_recorded or ""
        data_grp.attrs["total_episodes"] = len(demo_files)
        data_grp.attrs["total_frames"] = total_frames
        data_grp.attrs["cameras"] = resolved_cameras or []

        if env is not None:
            env.close()

    print(f"\nDone. Wrote {total_frames} total frames to {args.output}")


if __name__ == "__main__":
    main()
