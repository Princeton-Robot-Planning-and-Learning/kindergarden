"""Point cloud generation for Dynamic3D MuJoCo environments.

Generates dense, colored 3D point clouds from depth + RGB rendering across
one or more named cameras. The per-pixel back-projection uses each camera's
intrinsics (from cam_fovy) and extrinsics (cam_xpos / cam_xmat) live from
the simulation data so point clouds track the current physics state.

Typical usage
-------------
::

    env = kinder.make("kinder/BalanceBeam3D-o3-v0")
    env.reset(seed=0)

    sim = get_sim_from_env(env)
    pc = generate_scene_point_cloud(sim)   # uses all env cameras

    # pc.xyz  -> (N, 3) float32 world-frame positions
    # pc.rgb  -> (N, 3) uint8  colours  [0, 255]

"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Sequence

import mujoco  # type: ignore
import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# Public data type
# ---------------------------------------------------------------------------


@dataclass
class PointCloud:
    """Merged, world-frame coloured point cloud from one or more cameras.

    Attributes:
        xyz: (N, 3) float32 array of world-frame XYZ positions.
        rgb: (N, 3) uint8 array of RGB colours in [0, 255].
        camera_indices: (N,) int array mapping each point to its source
            camera index in ``camera_names``.
        camera_names: Ordered list of camera names that contributed points.
    """

    xyz: NDArray[np.float32]
    rgb: NDArray[np.uint8]
    camera_indices: NDArray[np.int32]
    camera_names: list[str] = field(default_factory=list)

    def __len__(self) -> int:
        return int(self.xyz.shape[0])

    def filter_by_camera(self, name: str) -> "PointCloud":
        """Return a new PointCloud with only the points from *name*."""
        if name not in self.camera_names:
            raise ValueError(
                f"Camera '{name}' not in point cloud. "
                f"Available cameras: {self.camera_names}"
            )
        idx = self.camera_names.index(name)
        mask = self.camera_indices == idx
        return PointCloud(
            xyz=self.xyz[mask],
            rgb=self.rgb[mask],
            camera_indices=self.camera_indices[mask],
            camera_names=[name],
        )

    def to_dict(self) -> dict[str, NDArray]:
        """Serialise to a plain dict (e.g. for np.savez)."""
        return {
            "xyz": self.xyz,
            "rgb": self.rgb,
            "camera_indices": self.camera_indices,
        }


# ---------------------------------------------------------------------------
# Intrinsics / extrinsics helpers
# ---------------------------------------------------------------------------


def get_camera_intrinsics(
    model: mujoco.MjModel,  # type: ignore[name-defined]
    cam_id: int,
    width: int,
    height: int,
) -> NDArray[np.float64]:
    """Return the 3×3 intrinsic matrix K for a MuJoCo fixed camera.

    MuJoCo uses a single vertical field-of-view (``cam_fovy``, in degrees)
    and assumes square pixels, so ``fx == fy``.

    Args:
        model: The MuJoCo model.
        cam_id: Zero-based camera index.
        width: Render width in pixels.
        height: Render height in pixels.

    Returns:
        K: 3×3 float64 intrinsics matrix::

            [[fx,  0, cx],
             [ 0, fy, cy],
             [ 0,  0,  1]]
    """
    fovy_deg: float = float(model.cam_fovy[cam_id])
    fovy_rad = math.radians(fovy_deg)
    fy = height / (2.0 * math.tan(fovy_rad / 2.0))
    fx = fy  # MuJoCo cameras have square pixels
    cx = width / 2.0
    cy = height / 2.0
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)


def get_camera_extrinsics(
    data: mujoco.MjData,  # type: ignore[name-defined]
    cam_id: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return the camera-to-world rotation and translation.

    MuJoCo's ``cam_xmat`` stores the camera frame orientation as a 9-element
    row-major rotation matrix whose *columns* are the camera X, Y, Z axes
    expressed in world coordinates (camera-to-world, same convention as
    ``xmat`` for bodies).  The camera Z axis points *backward* (away from
    the scene).

    Args:
        data: MuJoCo simulation data (after ``mj_forward``).
        cam_id: Zero-based camera index.

    Returns:
        R_c2w: (3, 3) rotation matrix, camera → world.
        t_w: (3,) camera position in world frame.
    """
    R_c2w = data.cam_xmat[cam_id].reshape(3, 3).copy().astype(np.float64)
    t_w = data.cam_xpos[cam_id].copy().astype(np.float64)
    return R_c2w, t_w


def depth_buffer_to_metric(
    depth_raw: NDArray[np.float32],
    model: mujoco.MjModel,  # type: ignore[name-defined]
) -> NDArray[np.float64]:
    """Convert a raw MuJoCo depth buffer to metric distances (metres).

    MuJoCo uses the standard OpenGL non-linear depth buffer where the stored
    value ``d ∈ [0, 1]`` relates to metric depth ``z`` via::

        z = znear * zfar / (zfar − d * (zfar − znear))

    where ``znear = vis.map.znear * stat.extent`` and similarly for
    ``zfar``.

    Args:
        depth_raw: (H, W) float32 array of raw depth buffer values in [0, 1].
        model: The MuJoCo model (needed for clipping-plane parameters).

    Returns:
        z_metric: (H, W) float64 array of metric distances along the optical
            axis.  Pixels at or beyond the far plane are mapped to ``inf``.
    """
    extent: float = float(model.stat.extent)
    znear: float = float(model.vis.map.znear) * extent
    zfar: float = float(model.vis.map.zfar) * extent
    d = depth_raw.astype(np.float64)
    denom = zfar - d * (zfar - znear)
    # Guard against divide-by-zero (shouldn't happen within [0,1] range)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(denom > 0.0, znear * zfar / denom, np.inf)
    return z


# ---------------------------------------------------------------------------
# Per-camera back-projection
# ---------------------------------------------------------------------------


def rgbd_to_point_cloud(
    rgb: NDArray[np.uint8],
    depth_raw: NDArray[np.float32],
    cam_id: int,
    model: mujoco.MjModel,  # type: ignore[name-defined]
    data: mujoco.MjData,  # type: ignore[name-defined]
    *,
    max_depth: float = 10.0,
    min_depth: float = 0.01,
) -> tuple[NDArray[np.float32], NDArray[np.uint8]]:
    """Back-project an RGBD image pair into a world-frame point cloud.

    Pixels whose metric depth falls outside [``min_depth``, ``max_depth``]
    are discarded.

    Args:
        rgb: (H, W, 3) uint8 RGB image.
        depth_raw: (H, W) float32 raw MuJoCo depth buffer (values in [0, 1]).
            The image must already be flipped vertically (as returned by
            ``MjSim.render``).
        cam_id: Zero-based camera index in the model.
        model: MuJoCo model (for intrinsics and depth conversion).
        data: MuJoCo data (for extrinsics).
        max_depth: Discard points farther than this distance (metres).
        min_depth: Discard points closer than this distance (metres).

    Returns:
        xyz_world: (N, 3) float32 positions in world frame.
        rgb_points: (N, 3) uint8 colours.
    """
    height, width = depth_raw.shape
    K = get_camera_intrinsics(model, cam_id, width, height)
    R_c2w, t_w = get_camera_extrinsics(data, cam_id)

    z_metric = depth_buffer_to_metric(depth_raw, model)  # (H, W)

    # Build pixel grid
    u = np.arange(width, dtype=np.float64)   # (W,)
    v = np.arange(height, dtype=np.float64)  # (H,)
    uu, vv = np.meshgrid(u, v)               # (H, W) each

    # Valid mask
    mask = (z_metric >= min_depth) & (z_metric <= max_depth) & np.isfinite(z_metric)
    if not np.any(mask):
        empty = np.empty((0, 3), dtype=np.float32)
        return empty, np.empty((0, 3), dtype=np.uint8)

    z = z_metric[mask]        # (N,)
    u_m = uu[mask]            # (N,)
    v_m = vv[mask]            # (N,)

    # Back-project to camera frame.
    # MuJoCo camera convention: X = right, Y = up, Z = backward.
    # The optical axis points in the -Z direction.
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    x_cam = (u_m - cx) / fx * z
    y_cam = -(v_m - cy) / fy * z  # flip: image v increases down, camera Y up
    z_cam = -z                     # camera looks in -Z

    pts_cam = np.stack([x_cam, y_cam, z_cam], axis=-1)  # (N, 3)

    # Transform to world frame: p_w = R_c2w @ p_c + t_w
    xyz_world = (pts_cam @ R_c2w.T + t_w).astype(np.float32)

    rgb_points = rgb[mask].astype(np.uint8)
    return xyz_world, rgb_points


# ---------------------------------------------------------------------------
# MjSim-level API
# ---------------------------------------------------------------------------


def generate_scene_point_cloud(
    sim: "MjSim",  # type: ignore[name-defined]  # kinder MjSim
    camera_names: Sequence[str] | None = None,
    *,
    width: int = 640,
    height: int = 480,
    max_depth: float = 10.0,
    min_depth: float = 0.01,
) -> PointCloud:
    """Generate a merged world-frame point cloud from one or more cameras.

    Renders each requested camera's RGB + depth image, back-projects the
    valid pixels into world-frame 3-D positions, and concatenates the
    results into a single :class:`PointCloud`.

    Args:
        sim: A ``kinder`` :class:`MjSim` instance (already reset and
            forwarded).
        camera_names: Camera names to use.  Pass ``None`` to use every
            named camera in the model.
        width: Render width in pixels.
        height: Render height in pixels.
        max_depth: Maximum metric depth to keep (metres).
        min_depth: Minimum metric depth to keep (metres).

    Returns:
        A :class:`PointCloud` containing the merged scene geometry.

    Raises:
        ValueError: If a requested camera name is not found in the model.
    """
    mj_model: mujoco.MjModel = sim.model.mj_model  # type: ignore[name-defined]
    mj_data: mujoco.MjData = sim.data.mj_data  # type: ignore[name-defined]

    # Resolve camera names → camera IDs
    name2id: dict[str, int] = {}
    for i in range(mj_model.ncam):
        name = mujoco.mj_id2name(  # pylint: disable=no-member
            mj_model, mujoco.mjtObj.mjOBJ_CAMERA, i  # pylint: disable=no-member
        )
        if name is not None:
            name2id[name] = i

    if camera_names is None:
        # Use all named cameras
        cam_items = list(name2id.items())
    else:
        cam_items = []
        for cname in camera_names:
            if cname not in name2id:
                raise ValueError(
                    f"Camera '{cname}' not found in model. "
                    f"Available cameras: {list(name2id.keys())}"
                )
            cam_items.append((cname, name2id[cname]))

    if not cam_items:
        empty_f = np.empty((0, 3), dtype=np.float32)
        empty_u = np.empty((0, 3), dtype=np.uint8)
        empty_i = np.empty((0,), dtype=np.int32)
        return PointCloud(xyz=empty_f, rgb=empty_u, camera_indices=empty_i)

    all_xyz: list[NDArray[np.float32]] = []
    all_rgb: list[NDArray[np.uint8]] = []
    all_idx: list[NDArray[np.int32]] = []
    resolved_names: list[str] = []

    ctx = sim._render_context_offscreen  # pylint: disable=protected-access

    for local_idx, (cname, cam_id) in enumerate(cam_items):
        ctx.render(width=width, height=height, camera_id=cam_id)
        rgb_depth = ctx.read_pixels(width, height, depth=True)
        assert isinstance(rgb_depth, tuple)
        rgb_img, depth_img = rgb_depth
        assert depth_img is not None

        xyz, rgb = rgbd_to_point_cloud(
            rgb_img,
            depth_img,
            cam_id,
            mj_model,
            mj_data,
            max_depth=max_depth,
            min_depth=min_depth,
        )

        if len(xyz) > 0:
            all_xyz.append(xyz)
            all_rgb.append(rgb)
            all_idx.append(
                np.full(len(xyz), local_idx, dtype=np.int32)
            )
        resolved_names.append(cname)

    if not all_xyz:
        empty_f = np.empty((0, 3), dtype=np.float32)
        empty_u = np.empty((0, 3), dtype=np.uint8)
        empty_i = np.empty((0,), dtype=np.int32)
        return PointCloud(xyz=empty_f, rgb=empty_u, camera_indices=empty_i,
                          camera_names=resolved_names)

    return PointCloud(
        xyz=np.concatenate(all_xyz, axis=0),
        rgb=np.concatenate(all_rgb, axis=0),
        camera_indices=np.concatenate(all_idx, axis=0),
        camera_names=resolved_names,
    )


# ---------------------------------------------------------------------------
# Convenience: extract sim from a wrapped kinder env
# ---------------------------------------------------------------------------


def get_sim_from_env(env: Any) -> "MjSim":  # type: ignore[name-defined]
    """Extract the underlying :class:`MjSim` from a wrapped kinder env.

    Works with the standard ``kinder.make(...)`` wrapper stack::

        kinder_wrapper
          └─ ObjectCentricRobotEnv (_object_centric_env)
               └─ RobotEnv         (_robot_env)
                    └─ MujocoEnv   (sim)

    Args:
        env: Any gymnasium-wrapped kinder environment.

    Returns:
        The ``MjSim`` instance (guaranteed non-None after a reset).

    Raises:
        RuntimeError: If the expected attributes are not found, or if the
            sim has not been initialised (i.e. no reset has been called yet).
    """
    unwrapped = env.unwrapped

    # Path: object_centric_env → _robot_env → sim
    if hasattr(unwrapped, "_object_centric_env"):
        robot_env = unwrapped._object_centric_env._robot_env  # pylint: disable=protected-access
    elif hasattr(unwrapped, "_robot_env"):
        robot_env = unwrapped._robot_env  # pylint: disable=protected-access
    else:
        raise RuntimeError(
            "Cannot locate a robot env inside the provided environment. "
            "Expected either '_object_centric_env._robot_env' or '_robot_env' "
            f"on {type(unwrapped).__name__}."
        )

    sim = robot_env.sim
    if sim is None:
        raise RuntimeError(
            "sim is None — call env.reset() before generating a point cloud."
        )
    return sim
