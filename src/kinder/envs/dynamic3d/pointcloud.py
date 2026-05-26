"""Point cloud capture from MuJoCo depth cameras in KinDER Dynamic3D environments."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from numpy.typing import NDArray

from kinder.envs.dynamic3d.mujoco_utils import MjSim


def resolve_camera_name(
    available: list[str],
    robot_name: str,
    requested: str,
    *,
    on_alias: Callable[[str], None] | None = None,
) -> str:
    """Resolve a MuJoCo camera name for a robot instance.

    If ``requested`` is in ``available``, return it. Otherwise, if ``requested`` ends
    with ``_base`` or ``_wrist``, try ``{robot_name}{suffix}`` (e.g. tidybot_base ->
    robot_base when the instance is named ``robot``).
    """
    if requested in available:
        return requested
    for suffix in ("_base", "_wrist"):
        candidate = f"{robot_name}{suffix}"
        if requested.endswith(suffix) and candidate in available:
            message = (
                f"Note: camera '{requested}' not in model; using '{candidate}' "
                f"(robot instance name is '{robot_name}')."
            )
            if on_alias is not None:
                on_alias(message)
            return candidate
    raise ValueError(
        f"Camera '{requested}' not found. Available cameras: {available}"
    )


def resolve_camera_name_for_oc_env(
    oc_env: Any,
    requested: str,
    *,
    on_alias: Callable[[str], None] | None = print,
) -> str:
    """Resolve camera name from an object-centric Dynamic3D env wrapper."""
    robot_env = oc_env._robot_env
    assert robot_env is not None and robot_env.sim is not None
    available = [n for n in robot_env.sim.model.camera_names if n is not None]
    return resolve_camera_name(
        available, oc_env.robot_name, requested, on_alias=on_alias
    )


def depth_buffer_to_metric(
    depth_buffer: NDArray[np.floating],
    model: mujoco.MjModel,  # type: ignore[name-defined]  # pylint: disable=no-member
) -> NDArray[np.float64]:
    """Convert mjr_readPixels [0,1] depth buffer to metric distance along the optical axis.

    Uses the same near/far formula as common MuJoCo / dm_control implementations
    (model.vis.map znear/zfar scaled by model.stat.extent).
    """
    extent = float(model.stat.extent)
    near = float(model.vis.map.znear) * extent
    far = float(model.vis.map.zfar) * extent
    denom = 1.0 - depth_buffer * (1.0 - near / far)
    with np.errstate(divide="ignore", invalid="ignore"):
        metric = near / denom
    return metric.astype(np.float64)


def metric_depth_to_pointcloud_world(
    metric_depth: NDArray[np.floating],
    depth_buffer: NDArray[np.floating],
    model: mujoco.MjModel,  # type: ignore[name-defined]  # pylint: disable=no-member
    data: mujoco.MjData,  # type: ignore[name-defined]  # pylint: disable=no-member
    camera_id: int,
    *,
    min_depth: float = 1e-4,
    max_depth: float | None = None,
    depth_buffer_far: float = 0.999,
) -> NDArray[np.float32]:
    """Build a world-frame point cloud (N, 3) from a metric depth image.

    MuJoCo camera frame: +X right, +Y up, view direction along -Z.
    Pixel (u, v): u increases right, v increases down (matches flipud RGB/depth).
    """
    height, width = metric_depth.shape
    fovy_rad = np.deg2rad(float(model.cam_fovy[camera_id]))
    fy = height / (2.0 * np.tan(fovy_rad / 2.0))
    fx = fy
    cx = width / 2.0
    cy = height / 2.0

    u_grid, v_grid = np.meshgrid(np.arange(width), np.arange(height))
    z_forward = metric_depth
    x_cam = (u_grid - cx) * z_forward / fx
    y_cam = (cy - v_grid) * z_forward / fy
    z_cam = -z_forward

    valid = np.isfinite(metric_depth) & (metric_depth > min_depth)
    valid &= depth_buffer < depth_buffer_far
    if max_depth is not None:
        valid &= metric_depth < max_depth

    points_cam = np.stack([x_cam[valid], y_cam[valid], z_cam[valid]], axis=-1)
    rot = data.cam_xmat[camera_id].reshape(3, 3)
    pos = data.cam_xpos[camera_id]
    points_world = points_cam @ rot.T + pos
    return points_world.astype(np.float32)


def render_rgbd(
    sim: MjSim,
    camera_name: str,
    width: int,
    height: int,
) -> tuple[NDArray[np.uint8], NDArray[np.float64]]:
    """Render RGB and depth buffer from a named MuJoCo camera."""
    result = sim.render(
        width=width,
        height=height,
        camera_name=camera_name,
        depth=True,
        mode="offscreen",
    )
    if not isinstance(result, tuple):
        raise RuntimeError("Expected (rgb, depth) tuple when depth=True")
    rgb, depth = result
    return rgb, np.asarray(depth, dtype=np.float64)


def depth_buffer_to_pointcloud_world(
    depth_buffer: NDArray[np.floating],
    model: mujoco.MjModel,  # type: ignore[name-defined]  # pylint: disable=no-member
    data: mujoco.MjData,  # type: ignore[name-defined]  # pylint: disable=no-member
    camera_id: int,
    **kwargs: Any,
) -> NDArray[np.float32]:
    """Convert a depth buffer to a world-frame point cloud in one step."""
    metric_depth = depth_buffer_to_metric(depth_buffer, model)
    return metric_depth_to_pointcloud_world(
        metric_depth, depth_buffer, model, data, camera_id, **kwargs
    )


def get_object_centric_env(gym_env: Any) -> Any:
    """Return the unwrapped KinDER object-centric Dynamic3D env."""
    return gym_env.unwrapped._object_centric_env


def capture_pointcloud_from_robot_env(
    robot_env: Any,
    camera_name: str,
    *,
    robot_name: str | None = None,
    min_depth: float = 0.02,
    max_depth: float | None = None,
    on_alias: Callable[[str], None] | None = print,
    forward: bool = True,
) -> tuple[NDArray[np.float32], NDArray[np.uint8], str, int]:
    """Capture a world-frame point cloud from a Dynamic3D robot env.

    Returns:
        (points_world, rgb, resolved_camera_name, camera_id)
    """
    if robot_env is None or robot_env.sim is None:
        raise RuntimeError("Robot simulation is not initialized.")

    sim: MjSim = robot_env.sim
    if forward:
        sim.forward()

    name = robot_name if robot_name is not None else robot_env.name
    available = [n for n in sim.model.camera_names if n is not None]
    resolved = resolve_camera_name(
        available,
        name,
        camera_name,
        on_alias=on_alias,
    )
    camera_id = sim.model.camera_name2id(resolved)
    width = robot_env.camera_width
    height = robot_env.camera_height
    rgb, depth_buffer = render_rgbd(sim, resolved, width, height)
    mj_model = sim.model.mj_model
    mj_data = sim.data.mj_data
    points = depth_buffer_to_pointcloud_world(
        depth_buffer,
        mj_model,
        mj_data,
        camera_id,
        min_depth=min_depth,
        max_depth=max_depth,
    )
    return points, rgb, resolved, camera_id


def capture_pointcloud_from_gym_env(
    gym_env: Any,
    camera_name: str,
    *,
    min_depth: float = 0.02,
    max_depth: float | None = None,
    on_alias: Callable[[str], None] | None = print,
) -> tuple[NDArray[np.float32], NDArray[np.uint8], str, int]:
    """Capture a world-frame point cloud from a wrapped KinDER Dynamic3D gym env."""
    oc_env = get_object_centric_env(gym_env)
    return capture_pointcloud_from_robot_env(
        oc_env._robot_env,
        camera_name,
        robot_name=oc_env.robot_name,
        min_depth=min_depth,
        max_depth=max_depth,
        on_alias=on_alias,
    )


def is_dynamic3d_gym_env(gym_env: Any) -> bool:
    """Return True if ``gym_env`` is a KinDER Dynamic3D env with an active MuJoCo sim."""
    try:
        oc_env = get_object_centric_env(gym_env)
    except AttributeError:
        return False
    robot_env = oc_env._robot_env
    return robot_env is not None and robot_env.sim is not None


def subsample_point_cloud(
    points: NDArray[np.float32],
    max_points: int,
    rng: np.random.Generator,
) -> NDArray[np.float32]:
    """Uniformly subsample a point cloud to at most ``max_points`` rows."""
    n = int(points.shape[0])
    if n <= max_points:
        return points
    indices = rng.choice(n, size=max_points, replace=False)
    return points[indices]


def stack_point_cloud_frames(
    frames: list[NDArray[np.float32]],
    max_points: int,
) -> tuple[NDArray[np.float32], NDArray[np.int32]]:
    """Stack per-frame clouds into (T, max_points, 3) with NaN padding.

    Returns:
        stacked: float32 array of shape (T, max_points, 3)
        counts: int32 array of shape (T,) with the number of valid points per frame
    """
    if not frames:
        raise ValueError("No point cloud frames to stack.")

    num_frames = len(frames)
    stacked = np.full((num_frames, max_points, 3), np.nan, dtype=np.float32)
    counts = np.zeros(num_frames, dtype=np.int32)
    for index, points in enumerate(frames):
        num_valid = min(int(points.shape[0]), max_points)
        counts[index] = num_valid
        stacked[index, :num_valid] = points[:num_valid]
    return stacked, counts


@dataclass
class PointCloudRecordingConfig:
    """Options for recording point clouds alongside demo replay."""

    camera: str = "tidybot_base"
    max_points_per_frame: int = 50_000
    min_depth: float = 0.02
    subsample_seed: int = 0
    output_path: Path | None = None


def default_pointcloud_output_path(video_path: Path) -> Path:
    """Default ``.npz`` path next to a demo GIF (``foo.gif`` -> ``foo_pointcloud.npz``)."""
    return video_path.with_name(f"{video_path.stem}_pointcloud.npz")


@dataclass
class PointCloudRecorder:
    """Record world-frame point clouds over an episode (reset + each step).

    Usage::

        recorder = PointCloudRecorder(camera_name="tidybot_base")
        env.reset(seed=seed)
        recorder.on_reset(env)
        for action in actions:
            env.step(action)
            recorder.on_step(env)
        recorder.save_npz("trajectory_pointcloud.npz", env_id=env_id)

    Aligns with demo GIF replay: call ``on_reset`` after ``env.reset``, then
    ``on_step`` after every ``env.step`` (same cadence as ``env.render()`` for video).
    """

    camera_name: str = "tidybot_base"
    max_points_per_frame: int = 50_000
    min_depth: float = 0.02
    max_depth: float | None = None
    seed: int = 0
    on_alias: Callable[[str], None] | None = print
    store_rgb: bool = False
    _frames: list[NDArray[np.float32]] = field(default_factory=list, init=False)
    _rgb_frames: list[NDArray[np.uint8]] = field(default_factory=list, init=False)
    _rng: np.random.Generator = field(init=False)
    _resolved_camera: str | None = field(default=None, init=False)
    _camera_id: int | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)

    def clear(self) -> None:
        """Drop all buffered frames."""
        self._frames.clear()
        self._rgb_frames.clear()
        self._resolved_camera = None
        self._camera_id = None

    def num_frames(self) -> int:
        """Number of captured frames (reset frame + one per step so far)."""
        return len(self._frames)

    def _capture(self, gym_env: Any) -> None:
        if not is_dynamic3d_gym_env(gym_env):
            raise TypeError(
                "PointCloudRecorder only supports KinDER Dynamic3D environments."
            )
        alias_cb = self.on_alias if len(self._frames) == 0 else None
        points, rgb, resolved, camera_id = capture_pointcloud_from_gym_env(
            gym_env,
            self.camera_name,
            min_depth=self.min_depth,
            max_depth=self.max_depth,
            on_alias=alias_cb,
        )
        if self._resolved_camera is None:
            self._resolved_camera = resolved
            self._camera_id = camera_id
        subsampled = subsample_point_cloud(
            points, self.max_points_per_frame, self._rng
        )
        self._frames.append(subsampled)
        if self.store_rgb:
            self._rgb_frames.append(np.asarray(rgb, dtype=np.uint8))

    def on_reset(self, gym_env: Any) -> None:
        """Capture a point cloud immediately after ``env.reset()``."""
        self._capture(gym_env)

    def on_step(self, gym_env: Any) -> None:
        """Capture a point cloud immediately after ``env.step()``."""
        self._capture(gym_env)

    def save_npz(
        self,
        path: str | Path,
        *,
        env_id: str | None = None,
        demo_seed: int | None = None,
    ) -> Path:
        """Save buffered frames to a compressed ``.npz`` file.

        Arrays:
            - ``points``: (T, max_points_per_frame, 3) float32, NaN-padded
            - ``counts``: (T,) int32, valid points per frame
            - ``camera``: resolved MuJoCo camera name (string)
            - ``camera_id``: MuJoCo camera index
            - ``num_frames``, ``max_points_per_frame``, ``min_depth``
            - ``env_id``, ``demo_seed`` when provided
            - ``rgb``: (T, H, W, 3) uint8 if ``store_rgb=True``
        """
        if not self._frames:
            raise ValueError("No frames recorded; call on_reset/on_step before save_npz.")

        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        points, counts = stack_point_cloud_frames(
            self._frames, self.max_points_per_frame
        )
        payload: dict[str, Any] = {
            "points": points,
            "counts": counts,
            "camera": np.asarray(self._resolved_camera or ""),
            "camera_id": np.int32(self._camera_id if self._camera_id is not None else -1),
            "num_frames": np.int32(len(self._frames)),
            "max_points_per_frame": np.int32(self.max_points_per_frame),
            "min_depth": np.float32(self.min_depth),
        }
        if env_id is not None:
            payload["env_id"] = np.asarray(env_id)
        if demo_seed is not None:
            payload["demo_seed"] = np.int32(demo_seed)
        if self.store_rgb and self._rgb_frames:
            payload["rgb"] = np.stack(self._rgb_frames, axis=0)

        np.savez_compressed(out_path, **payload)
        return out_path


def show_camera_rgb(
    rgb: NDArray[np.uint8],
    *,
    title: str,
    show: bool,
    save_path: str | None = None,
) -> None:
    """Display or save a rendered RGB camera frame."""
    import matplotlib.pyplot as plt
    from pathlib import Path

    fig, ax = plt.subplots(figsize=(10, 7.5))
    ax.imshow(rgb)
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    if save_path is not None:
        path = Path(save_path)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved camera view to {path.resolve()}")
    if show:
        plt.show()
    else:
        plt.close(fig)
