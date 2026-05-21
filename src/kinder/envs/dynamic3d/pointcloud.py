"""Point cloud capture from MuJoCo depth cameras in KinDER Dynamic3D environments."""

from __future__ import annotations

from collections.abc import Callable
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
