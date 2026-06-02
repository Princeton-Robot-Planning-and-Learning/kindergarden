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
from typing import Any, Literal

import gymnasium
import h5py
import numpy as np
from numpy.typing import NDArray

import kinder
from kinder.envs.dynamic3d.point_cloud import (
    MeshPointCloud,
    PointCloud,
    find_ee_frame,
    generate_full_point_cloud,
    generate_scene_point_cloud,
    generate_track_point_cloud,
    get_ee_pose,
    get_geom_world_transforms,
    get_sim_from_env,
    sample_canonical_points,
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

    mode = p.add_mutually_exclusive_group()
    mode.add_argument(
        "--completepointcloud",
        action="store_true",
        help=(
            "Store only mesh-based point clouds (obs/mesh_pc_xyz and "
            "obs/mesh_pc_geom_indices). No camera point clouds are written."
        ),
    )
    mode.add_argument(
        "--allpointcloud",
        action="store_true",
        help=(
            "Store both camera point clouds (existing behaviour) and mesh "
            "point clouds (obs/mesh_pc_xyz and obs/mesh_pc_geom_indices)."
        ),
    )
    p.add_argument(
        "--trackpointcloud",
        action="store_true",
        help=(
            "Store tracked point clouds with fixed point identity across "
            "timesteps (obs/track_pc_xyz and obs/track_pc_geom_indices). "
            "Points are sampled once after env.reset() in local coordinates "
            "and re-transformed each step. Combinable with "
            "--completepointcloud and --allpointcloud."
        ),
    )
    p.add_argument(
        "--canonicalpointcloud",
        action="store_true",
        help=(
            "Store canonical per-geom surface points (sampled once after reset) "
            "and per-timestep rigid-body transforms instead of full per-timestep "
            "point clouds.  Writes canonical_pointcloud/<geom>/xyz, "
            "geom_transforms/<geom> (T,4,4), obs/ee_pose (T,4,4), and "
            "actions_delta_ee_transform (T,4,4).  Combinable with "
            "--completepointcloud, --allpointcloud, and --trackpointcloud."
        ),
    )
    p.add_argument(
        "--ee-site",
        type=str,
        default=None,
        metavar="NAME",
        help=(
            "MuJoCo site or body name to use as the end-effector frame for "
            "obs/ee_pose and actions_delta_ee_transform.  Sites are tried "
            "before bodies.  When omitted, common names are auto-detected "
            "(robot_pinch_site, eef_site, …; then EE_BODY_R, end_effector, …)."
        ),
    )
    p.add_argument(
        "--num-points-per-geom",
        type=int,
        default=500,
        metavar="N",
        help="Surface points sampled per geom for mesh-based point clouds",
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


def _hdf5_key(name: str) -> str:
    """Sanitise a geom name for use as an HDF5 dataset/group key.

    HDF5 treats ``/`` as a path separator, so forward slashes in geom names
    (e.g. ``gen3/base_link_geom144``) must be escaped.
    """
    return name.replace("/", "__")


def _se3_inv(T: NDArray[np.float32]) -> NDArray[np.float32]:
    """Compute the inverse of a 4×4 SE(3) matrix without calling np.linalg.inv."""
    R = T[:3, :3]
    p = T[:3, 3]
    T_inv = np.eye(4, dtype=np.float32)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -(R.T @ p)
    return T_inv


def _create_canonical_datasets(
    grp: h5py.Group,
    obs_grp: h5py.Group,
    canonical_pts: dict[str, tuple[int, NDArray[np.float32]]],
    num_frames: int,
    ee_frame_name: str,
    ee_frame_type: Literal["site", "body"],
) -> None:
    """Pre-allocate HDF5 datasets for canonical point cloud mode."""
    str_dt = h5py.string_dtype()

    # canonical_pointcloud/<hdf5_key>/xyz  +  geom_name attr on every group
    canon_grp = grp.require_group("canonical_pointcloud")
    for name, (_, pts) in canonical_pts.items():
        key = _hdf5_key(name)
        geom_grp = canon_grp.require_group(key)
        geom_grp.attrs["geom_name"] = name
        geom_grp.create_dataset("xyz", data=pts, dtype=np.float32, compression="gzip")

    # Demo-level ordered mapping: original names ↔ HDF5 keys
    orig_names = list(canonical_pts.keys())
    hdf5_keys = [_hdf5_key(n) for n in orig_names]
    canon_grp.create_dataset("geom_names", data=orig_names, dtype=str_dt)
    canon_grp.create_dataset("geom_hdf5_keys", data=hdf5_keys, dtype=str_dt)

    # geom_transforms/<hdf5_key>  +  geom_name attr on every dataset
    gt_grp = grp.require_group("geom_transforms")
    for name in canonical_pts:
        key = _hdf5_key(name)
        ds = gt_grp.create_dataset(
            key,
            shape=(num_frames, 4, 4),
            dtype=np.float32,
            compression="gzip",
        )
        ds.attrs["geom_name"] = name

    obs_grp.create_dataset(
        "ee_pose",
        shape=(num_frames, 4, 4),
        dtype=np.float32,
        compression="gzip",
    )
    grp.create_dataset(
        "actions_delta_ee_transform",
        shape=(num_frames, 4, 4),
        dtype=np.float32,
        compression="gzip",
    )

    # Schema / convention attrs so the file is self-documenting
    grp.attrs["pointcloud_schema"] = "canonical_pointcloud_v1"
    grp.attrs["delta_ee_convention"] = (
        "actions_delta_ee_transform[t] = "
        "inv(ee_pose_before_action[t]) @ ee_pose_after_action[t]"
    )
    grp.attrs["ee_frame_name"] = ee_frame_name
    grp.attrs["ee_frame_type"] = ee_frame_type


def _pad_or_subsample_mesh(
    xyz: NDArray[np.float32],
    geom_indices: NDArray[np.int32],
    max_pts: int,
    rng: np.random.Generator,
) -> tuple[NDArray[np.float32], NDArray[np.int32]]:
    """Return mesh arrays with exactly *max_pts* rows.

    If *n < max_pts*:  pad xyz with NaN, geom_indices with -1.
    If *n > max_pts*:  randomly sub-sample without replacement.
    """
    n = len(xyz)
    if n == max_pts:
        return xyz, geom_indices
    if n > max_pts:
        idx = rng.choice(n, max_pts, replace=False)
        return xyz[idx], geom_indices[idx]
    pad = max_pts - n
    xyz_pad = np.full((pad, 3), np.nan, dtype=np.float32)
    geom_pad = np.full((pad,), -1, dtype=np.int32)
    return (
        np.concatenate([xyz, xyz_pad], axis=0),
        np.concatenate([geom_indices, geom_pad], axis=0),
    )


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
    result = ctx.read_pixels(width, height, depth=False)
    assert isinstance(result, np.ndarray)
    return result


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
    *,
    store_camera: bool = True,
    store_mesh: bool = False,
    store_track: bool = False,
    num_points_per_geom: int = 500,
    local_points_dict: dict[str, NDArray[np.float32]] | None = None,
) -> tuple[dict[str, Any], dict[str, NDArray[np.float32]] | None]:
    """Capture RGB images and/or point clouds at the current env state.

    Returns ``(result, local_points_dict)`` where ``local_points_dict`` must
    be passed back on the next call when ``store_track=True``.

    Camera mode (``store_camera=True``):
    - ``"{cam}_rgb"``    → (H, W, 3) uint8
    - ``"{cam}_pc_xyz"`` → (N, 3) or (max_pts, 3) float32
    - ``"{cam}_pc_rgb"`` → (N, 3) or (max_pts, 3) uint8

    Mesh mode (``store_mesh=True``):
    - ``"mesh_pc_xyz"``          → (M, 3) or (max_pts, 3) float32
    - ``"mesh_pc_geom_indices"`` → (M,)  or (max_pts,)  int32

    Track mode (``store_track=True``):
    - ``"track_pc_xyz"``          → (M, 3) or (max_pts, 3) float32
    - ``"track_pc_geom_indices"`` → (M,)  or (max_pts,)  int32
    """
    sim = get_sim_from_env(env)
    result: dict[str, Any] = {}

    if store_camera:
        scene_pc: PointCloud = generate_scene_point_cloud(
            sim,
            camera_names=camera_names,
            width=width,
            height=height,
            max_depth=max_depth,
            min_depth=min_depth,
        )
        for cam_name in camera_names:
            result[f"{cam_name}_rgb"] = _render_rgb(env, cam_name, width, height)
            if cam_name in scene_pc.camera_names:
                cam_pc = scene_pc.filter_by_camera(cam_name)
                xyz, rgb = cam_pc.xyz, cam_pc.rgb
            else:
                xyz = np.empty((0, 3), dtype=np.float32)
                rgb = np.empty((0, 3), dtype=np.uint8)
            if max_pts is not None:
                xyz, rgb = _pad_or_subsample(xyz, rgb, max_pts, rng)
            result[f"{cam_name}_pc_xyz"] = xyz
            result[f"{cam_name}_pc_rgb"] = rgb

    if store_mesh:
        mesh_pc: MeshPointCloud = generate_full_point_cloud(
            sim, num_points_per_geom=num_points_per_geom
        )
        xyz_m = mesh_pc.xyz
        idx_m = mesh_pc.geom_indices
        if max_pts is not None:
            xyz_m, idx_m = _pad_or_subsample_mesh(xyz_m, idx_m, max_pts, rng)
        result["mesh_pc_xyz"] = xyz_m
        result["mesh_pc_geom_indices"] = idx_m

    if store_track:
        track_pc, local_points_dict = generate_track_point_cloud(
            sim, local_points_dict, num_points_per_geom=num_points_per_geom
        )
        xyz_t = track_pc.xyz
        idx_t = track_pc.geom_indices
        if max_pts is not None:
            xyz_t, idx_t = _pad_or_subsample_mesh(xyz_t, idx_t, max_pts, rng)
        result["track_pc_xyz"] = xyz_t
        result["track_pc_geom_indices"] = idx_t

    return result, local_points_dict


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
    *,
    store_camera: bool = True,
    store_mesh: bool = False,
    store_track: bool = False,
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

    if store_camera:
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

    if store_mesh:
        mesh_xyz = first_step["mesh_pc_xyz"]
        obs_grp.create_dataset(
            "mesh_pc_xyz",
            shape=(num_frames, *mesh_xyz.shape),
            dtype=np.float32,
            compression="gzip",
        )
        mesh_idx = first_step["mesh_pc_geom_indices"]
        obs_grp.create_dataset(
            "mesh_pc_geom_indices",
            shape=(num_frames, *mesh_idx.shape),
            dtype=np.int32,
            compression="gzip",
        )

    if store_track:
        track_xyz = first_step["track_pc_xyz"]
        obs_grp.create_dataset(
            "track_pc_xyz",
            shape=(num_frames, *track_xyz.shape),
            dtype=np.float32,
            compression="gzip",
        )
        track_idx = first_step["track_pc_geom_indices"]
        obs_grp.create_dataset(
            "track_pc_geom_indices",
            shape=(num_frames, *track_idx.shape),
            dtype=np.int32,
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
    *,
    store_camera: bool = True,
    store_mesh: bool = False,
    mesh_pc_size: int = 0,
    store_track: bool = False,
    track_pc_size: int = 0,
) -> None:
    """Write one step's data into pre-allocated HDF5 datasets.

    Point cloud arrays are padded or sub-sampled to the fixed sizes established
    at step 0, so every row has a consistent shape.
    """
    obs_grp["state"][step_idx] = obs.astype(np.float32)
    grp["actions"][step_idx] = np.asarray(action, dtype=np.float32)

    if store_camera:
        for cam_name in camera_names:
            obs_grp[f"{cam_name}_rgb"][step_idx] = capture[f"{cam_name}_rgb"]
            xyz = capture[f"{cam_name}_pc_xyz"]
            rgb = capture[f"{cam_name}_pc_rgb"]
            size = cam_sizes[cam_name]
            if len(xyz) != size:
                xyz, rgb = _pad_or_subsample(xyz, rgb, size, rng)
            obs_grp[f"{cam_name}_pc_xyz"][step_idx] = xyz
            obs_grp[f"{cam_name}_pc_rgb"][step_idx] = rgb

    if store_mesh:
        xyz_m = capture["mesh_pc_xyz"]
        idx_m = capture["mesh_pc_geom_indices"]
        if len(xyz_m) != mesh_pc_size:
            xyz_m, idx_m = _pad_or_subsample_mesh(xyz_m, idx_m, mesh_pc_size, rng)
        obs_grp["mesh_pc_xyz"][step_idx] = xyz_m
        obs_grp["mesh_pc_geom_indices"][step_idx] = idx_m

    if store_track:
        xyz_t = capture["track_pc_xyz"]
        idx_t = capture["track_pc_geom_indices"]
        if len(xyz_t) != track_pc_size:
            xyz_t, idx_t = _pad_or_subsample_mesh(xyz_t, idx_t, track_pc_size, rng)
        obs_grp["track_pc_xyz"][step_idx] = xyz_t
        obs_grp["track_pc_geom_indices"][step_idx] = idx_t


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
    *,
    store_camera: bool = True,
    store_mesh: bool = False,
    store_track: bool = False,
    store_canonical: bool = False,
    ee_site: str | None = None,
    num_points_per_geom: int = 500,
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

    # Resolve camera names (skipped when camera-only data is not needed)
    if store_camera and camera_names is None:
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
    elif not store_camera:
        camera_names = camera_names or []
    assert camera_names is not None

    # Canonical setup: sample fixed local-frame points and locate the EE frame
    # immediately after reset so geometry is in its initial pose.
    canonical_pts: dict[str, tuple[int, NDArray[np.float32]]] = {}
    ee_frame_name: str | None = None
    ee_frame_type: Literal["site", "body"] = "site"
    if store_canonical:
        sim = get_sim_from_env(env)
        ee_frame_name, ee_frame_type = find_ee_frame(sim, ee_site)
        canonical_pts = sample_canonical_points(
            sim, num_points_per_geom=num_points_per_geom
        )
        print(
            f"  Canonical: {len(canonical_pts)} geoms, "
            f"EE {ee_frame_type}='{ee_frame_name}'"
        )

    num_frames = len(actions)
    grp_name = f"demo_{demo_idx}"
    grp = hdf5_file.require_group(f"data/{grp_name}")
    grp.attrs["num_frames"] = num_frames
    grp.attrs["seed"] = seed
    obs_grp = grp.require_group("obs")

    # Capture step 0 to pin array sizes for the rest of the demo.
    # local_points_dict starts as None; generate_track_point_cloud populates it
    # on the first call so the same surface points are re-used every step.
    local_points_dict: dict[str, NDArray[np.float32]] | None = None
    capture0, local_points_dict = _capture_step(
        env,
        camera_names,
        width,
        height,
        max_depth,
        min_depth,
        max_pts,
        rng,
        store_camera=store_camera,
        store_mesh=store_mesh,
        store_track=store_track,
        num_points_per_geom=num_points_per_geom,
        local_points_dict=local_points_dict,
    )
    cam_sizes: dict[str, int] = (
        {cam: len(capture0[f"{cam}_pc_xyz"]) for cam in camera_names}
        if store_camera
        else {}
    )
    mesh_pc_size: int = len(capture0["mesh_pc_xyz"]) if store_mesh else 0
    track_pc_size: int = len(capture0["track_pc_xyz"]) if store_track else 0

    _create_datasets(
        grp,
        obs_grp,
        capture0,
        obs_np,
        actions,
        camera_names,
        num_frames,
        store_camera=store_camera,
        store_mesh=store_mesh,
        store_track=store_track,
    )
    if store_canonical:
        assert ee_frame_name is not None
        _create_canonical_datasets(
            grp, obs_grp, canonical_pts, num_frames, ee_frame_name, ee_frame_type
        )

    _write_step(
        grp,
        obs_grp,
        0,
        obs_np,
        actions[0],
        capture0,
        camera_names,
        cam_sizes,
        rng,
        store_camera=store_camera,
        store_mesh=store_mesh,
        mesh_pc_size=mesh_pc_size,
        store_track=store_track,
        track_pc_size=track_pc_size,
    )

    ee_poses_all: list[NDArray[np.float32]] = []
    geom_transforms_all: dict[str, list[NDArray[np.float32]]] = (
        {name: [] for name in canonical_pts} if store_canonical else {}
    )
    if store_canonical:
        assert ee_frame_name is not None
        ee_poses_all.append(get_ee_pose(sim, ee_frame_name, ee_frame_type))
        for name, xform in get_geom_world_transforms(
            sim, {n: gid for n, (gid, _) in canonical_pts.items()}
        ).items():
            geom_transforms_all[name].append(xform)

    # Step through remaining actions
    for step_idx in range(1, num_frames):
        obs_np, _reward, _term, _trunc, _ = env.step(actions[step_idx - 1])

        if store_canonical:
            assert ee_frame_name is not None
            ee_poses_all.append(get_ee_pose(sim, ee_frame_name, ee_frame_type))
            for name, xform in get_geom_world_transforms(
                sim, {n: gid for n, (gid, _) in canonical_pts.items()}
            ).items():
                geom_transforms_all[name].append(xform)

        capture, local_points_dict = _capture_step(
            env,
            camera_names,
            width,
            height,
            max_depth,
            min_depth,
            max_pts,
            rng,
            store_camera=store_camera,
            store_mesh=store_mesh,
            store_track=store_track,
            num_points_per_geom=num_points_per_geom,
            local_points_dict=local_points_dict,
        )
        _write_step(
            grp,
            obs_grp,
            step_idx,
            obs_np,
            actions[step_idx],
            capture,
            camera_names,
            cam_sizes,
            rng,
            store_camera=store_camera,
            store_mesh=store_mesh,
            mesh_pc_size=mesh_pc_size,
            store_track=store_track,
            track_pc_size=track_pc_size,
        )

    # Final step: execute last action to get terminal observation and EE pose.
    obs_np, _reward, _term, _trunc, _ = env.step(actions[-1])
    # terminal obs is not stored; dataset length == len(actions)

    # Capture terminal EE pose (after actions[-1]) for the last delta.
    if store_canonical:
        assert ee_frame_name is not None
        ee_poses_all.append(get_ee_pose(sim, ee_frame_name, ee_frame_type))

    # Write canonical data in one pass after all steps are collected.
    if store_canonical and ee_poses_all:
        ee_arr = np.stack(ee_poses_all)  # (num_frames + 1, 4, 4)
        # delta[t] = inv(ee_before_action[t]) @ ee_after_action[t]
        #           = inv(ee_arr[t]) @ ee_arr[t+1]
        delta_arr = np.stack(
            [_se3_inv(ee_arr[t]) @ ee_arr[t + 1] for t in range(num_frames)]
        ).astype(np.float32)
        obs_grp["ee_pose"][:] = ee_arr[:num_frames]
        grp["actions_delta_ee_transform"][:] = delta_arr
        gt_grp = grp["geom_transforms"]
        for name, xforms in geom_transforms_all.items():
            gt_grp[_hdf5_key(name)][:] = np.stack(xforms).astype(np.float32)

    print(f"  demo_{demo_idx}: {num_frames} frames, seed={seed}")
    return env, camera_names, num_frames


def main() -> None:
    """Entry point."""
    args = _parse_args()

    demo_files = _find_demo_files(args.demo_dir)
    if args.max_demos is not None:
        demo_files = demo_files[: args.max_demos]

    store_canonical = args.canonicalpointcloud
    store_camera = not args.completepointcloud
    store_mesh = args.completepointcloud or args.allpointcloud
    store_track = args.trackpointcloud

    mode_label = (
        "camera + mesh"
        if args.allpointcloud
        else "mesh only" if args.completepointcloud else "camera only"
    )
    if store_track:
        mode_label += " + track"
    if store_canonical:
        mode_label += " + canonical"
    print(f"Found {len(demo_files)} demo(s) in {args.demo_dir}")
    print(f"Output: {args.output}  |  mode: {mode_label}")

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
                    store_camera=store_camera,
                    store_mesh=store_mesh,
                    store_track=store_track,
                    store_canonical=store_canonical,
                    ee_site=args.ee_site,
                    num_points_per_geom=args.num_points_per_geom,
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
