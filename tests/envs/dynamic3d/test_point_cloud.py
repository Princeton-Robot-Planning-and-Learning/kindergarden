"""Unit tests for Dynamic3D point cloud generation.

Pass ``--visualize`` to pytest to open matplotlib windows showing the RGB
renders, depth maps, and 3-D point cloud for a sample environment::

    pytest tests/envs/dynamic3d/test_point_cloud.py --visualize -v -s
"""

from pathlib import Path

import numpy as np
import pytest

from kinder.envs.dynamic3d.envs import ObjectCentricTidyBot3DEnv
from kinder.envs.dynamic3d.point_cloud import (
    PointCloud,
    depth_buffer_to_metric,
    generate_scene_point_cloud,
    get_camera_extrinsics,
    get_camera_intrinsics,
    get_sim_from_env,
    rgbd_to_point_cloud,
)

_TEST_TASKS = Path(__file__).parent / "test_tasks"
_TASK_CONFIG = str(_TEST_TASKS / "tidybot-ground-o3.json")


@pytest.fixture(name="env")
def _env():
    env = ObjectCentricTidyBot3DEnv(
        num_objects=3,
        task_config_path=_TASK_CONFIG,
    )
    env.reset(seed=0)
    yield env
    env.close()


@pytest.fixture(name="sim")
def _sim(env):
    return get_sim_from_env(env)


# ---------------------------------------------------------------------------
# get_sim_from_env
# ---------------------------------------------------------------------------


def test_get_sim_from_env_returns_sim(env):
    """get_sim_from_env extracts a non-None sim from a wrapped kinder env."""
    sim = get_sim_from_env(env)
    assert sim is not None


def test_get_sim_from_env_raises_before_reset():
    """get_sim_from_env raises RuntimeError when env has not been reset."""
    env = ObjectCentricTidyBot3DEnv(
        num_objects=3,
        task_config_path=_TASK_CONFIG,
    )
    with pytest.raises(RuntimeError, match="sim is None"):
        get_sim_from_env(env)
    env.close()


# ---------------------------------------------------------------------------
# Camera intrinsics
# ---------------------------------------------------------------------------


def test_camera_intrinsics_shape(sim):
    """get_camera_intrinsics returns a 3×3 matrix."""
    model = sim.model.mj_model
    K = get_camera_intrinsics(model, cam_id=0, width=640, height=480)
    assert K.shape == (3, 3)


def test_camera_intrinsics_structure(sim):
    """Intrinsics matrix has the expected pinhole structure."""
    model = sim.model.mj_model
    K = get_camera_intrinsics(model, cam_id=0, width=640, height=480)
    # Off-diagonal entries in the first two rows must be zero (no skew)
    assert K[0, 1] == pytest.approx(0.0)
    assert K[1, 0] == pytest.approx(0.0)
    # Principal point at image centre
    assert K[0, 2] == pytest.approx(320.0)
    assert K[1, 2] == pytest.approx(240.0)
    # Homogeneous row
    assert K[2, 0] == pytest.approx(0.0)
    assert K[2, 1] == pytest.approx(0.0)
    assert K[2, 2] == pytest.approx(1.0)
    # Focal lengths must be positive
    assert K[0, 0] > 0
    assert K[1, 1] > 0


def test_camera_intrinsics_square_pixels(sim):
    """MuJoCo cameras have square pixels (fx == fy)."""
    model = sim.model.mj_model
    K = get_camera_intrinsics(model, cam_id=0, width=640, height=480)
    assert K[0, 0] == pytest.approx(K[1, 1])


# ---------------------------------------------------------------------------
# Camera extrinsics
# ---------------------------------------------------------------------------


def test_camera_extrinsics_shapes(sim):
    """get_camera_extrinsics returns R (3×3) and t (3,)."""
    data = sim.data.mj_data
    R, t = get_camera_extrinsics(data, cam_id=0)
    assert R.shape == (3, 3)
    assert t.shape == (3,)


def test_camera_extrinsics_rotation_orthogonal(sim):
    """The rotation matrix R must be orthogonal (R @ R.T ≈ I)."""
    data = sim.data.mj_data
    R, _ = get_camera_extrinsics(data, cam_id=0)
    np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-6)


# ---------------------------------------------------------------------------
# depth_buffer_to_metric
# ---------------------------------------------------------------------------


def test_depth_buffer_to_metric_zero_is_near(sim):
    """A raw depth of 0.0 maps to the near-clipping distance."""
    model = sim.model.mj_model
    d = np.zeros((4, 4), dtype=np.float32)
    z = depth_buffer_to_metric(d, model)
    extent = float(model.stat.extent)
    znear = float(model.vis.map.znear) * extent
    np.testing.assert_allclose(z, znear, rtol=1e-5)


def test_depth_buffer_to_metric_positive(sim):
    """All finite metric depths for a mid-range buffer must be positive."""
    model = sim.model.mj_model
    d = np.full((8, 8), 0.5, dtype=np.float32)
    z = depth_buffer_to_metric(d, model)
    assert np.all(z[np.isfinite(z)] > 0)


def test_depth_buffer_to_metric_output_shape(sim):
    """Output shape matches input shape."""
    model = sim.model.mj_model
    d = np.random.default_rng(0).random((12, 16)).astype(np.float32)
    z = depth_buffer_to_metric(d, model)
    assert z.shape == d.shape


# ---------------------------------------------------------------------------
# rgbd_to_point_cloud
# ---------------------------------------------------------------------------


def test_rgbd_to_point_cloud_shapes(sim):
    """Back-projected arrays have matching leading dimensions."""
    model = sim.model.mj_model
    data = sim.data.mj_data
    h, w = 32, 32
    rgb = np.random.default_rng(0).integers(0, 255, (h, w, 3), dtype=np.uint8)
    depth = np.full((h, w), 0.3, dtype=np.float32)
    xyz, rgb_pts = rgbd_to_point_cloud(rgb, depth, cam_id=0, model=model, data=data)
    assert xyz.shape[1] == 3
    assert rgb_pts.shape[1] == 3
    assert xyz.shape[0] == rgb_pts.shape[0]


def test_rgbd_to_point_cloud_dtypes(sim):
    """XYZ is float32 and RGB is uint8."""
    model = sim.model.mj_model
    data = sim.data.mj_data
    h, w = 16, 16
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    depth = np.full((h, w), 0.5, dtype=np.float32)
    xyz, rgb_pts = rgbd_to_point_cloud(rgb, depth, cam_id=0, model=model, data=data)
    assert xyz.dtype == np.float32
    assert rgb_pts.dtype == np.uint8


def test_rgbd_to_point_cloud_empty_when_no_valid_pixels(sim):
    """Returns empty arrays when all pixels are outside the depth range."""
    model = sim.model.mj_model
    data = sim.data.mj_data
    h, w = 8, 8
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    depth = np.zeros((h, w), dtype=np.float32)  # all at near plane → clipped
    xyz, rgb_pts = rgbd_to_point_cloud(
        rgb, depth, cam_id=0, model=model, data=data, min_depth=1.0, max_depth=10.0
    )
    assert len(xyz) == 0
    assert len(rgb_pts) == 0


# ---------------------------------------------------------------------------
# generate_scene_point_cloud
# ---------------------------------------------------------------------------


def test_generate_scene_point_cloud_returns_point_cloud(sim):
    """generate_scene_point_cloud returns a PointCloud instance."""
    pc = generate_scene_point_cloud(sim, width=64, height=48)
    assert isinstance(pc, PointCloud)


def test_generate_scene_point_cloud_nonempty(sim):
    """Point cloud generated from a reset env has at least one point."""
    pc = generate_scene_point_cloud(sim, width=64, height=48)
    assert len(pc) > 0


def test_generate_scene_point_cloud_array_shapes(sim):
    """Xyz, rgb, and camera_indices have consistent leading dimensions."""
    pc = generate_scene_point_cloud(sim, width=64, height=48)
    n = len(pc)
    assert pc.xyz.shape == (n, 3)
    assert pc.rgb.shape == (n, 3)
    assert pc.camera_indices.shape == (n,)


def test_generate_scene_point_cloud_dtypes(sim):
    """Xyz is float32, rgb is uint8, camera_indices is int32."""
    pc = generate_scene_point_cloud(sim, width=64, height=48)
    assert pc.xyz.dtype == np.float32
    assert pc.rgb.dtype == np.uint8
    assert pc.camera_indices.dtype == np.int32


def test_generate_scene_point_cloud_rgb_range(sim):
    """RGB values are in [0, 255]."""
    pc = generate_scene_point_cloud(sim, width=64, height=48)
    assert pc.rgb.min() >= 0
    assert pc.rgb.max() <= 255


def test_generate_scene_point_cloud_camera_indices_valid(sim):
    """Camera indices reference valid positions in camera_names."""
    pc = generate_scene_point_cloud(sim, width=64, height=48)
    assert pc.camera_indices.min() >= 0
    assert pc.camera_indices.max() < len(pc.camera_names)


def test_generate_scene_point_cloud_camera_names_populated(sim):
    """camera_names is non-empty after generation."""
    pc = generate_scene_point_cloud(sim, width=64, height=48)
    assert len(pc.camera_names) > 0


def test_generate_scene_point_cloud_single_camera(sim):
    """Restricting to one camera returns only points from that camera."""
    all_names = sim.model.mj_model.ncam
    if all_names == 0:
        pytest.skip("No named cameras in this model")
    import mujoco  # type: ignore[import-untyped]  # pylint: disable=import-outside-toplevel

    name = mujoco.mj_id2name(  # pylint: disable=no-member
        sim.model.mj_model, mujoco.mjtObj.mjOBJ_CAMERA, 0  # pylint: disable=no-member
    )
    pc = generate_scene_point_cloud(sim, camera_names=[name], width=64, height=48)
    assert pc.camera_names == [name]
    assert set(pc.camera_indices.tolist()).issubset({0})


def test_generate_scene_point_cloud_invalid_camera_raises(sim):
    """Requesting an unknown camera name raises ValueError."""
    with pytest.raises(ValueError, match="not found"):
        generate_scene_point_cloud(sim, camera_names=["nonexistent_cam"])


# ---------------------------------------------------------------------------
# PointCloud methods
# ---------------------------------------------------------------------------


def test_point_cloud_filter_by_camera(sim):
    """filter_by_camera returns only points from the requested camera."""
    pc = generate_scene_point_cloud(sim, width=64, height=48)
    if len(pc.camera_names) == 0:
        pytest.skip("No cameras in point cloud")
    cam = pc.camera_names[0]
    sub = pc.filter_by_camera(cam)
    assert sub.camera_names == [cam]
    assert len(sub) > 0
    assert np.all(sub.camera_indices == 0)


def test_point_cloud_filter_by_camera_unknown_raises(sim):
    """filter_by_camera raises ValueError for an unknown camera name."""
    pc = generate_scene_point_cloud(sim, width=64, height=48)
    with pytest.raises(ValueError, match="not in point cloud"):
        pc.filter_by_camera("no_such_camera")


def test_point_cloud_to_dict(sim):
    """to_dict contains xyz, rgb, and camera_indices keys."""
    pc = generate_scene_point_cloud(sim, width=64, height=48)
    d = pc.to_dict()
    assert set(d.keys()) == {"xyz", "rgb", "camera_indices"}
    np.testing.assert_array_equal(d["xyz"], pc.xyz)
    np.testing.assert_array_equal(d["rgb"], pc.rgb)
    np.testing.assert_array_equal(d["camera_indices"], pc.camera_indices)


def test_point_cloud_len(sim):
    """Len(pc) matches the first dimension of xyz."""
    pc = generate_scene_point_cloud(sim, width=64, height=48)
    assert len(pc) == pc.xyz.shape[0]


# ---------------------------------------------------------------------------
# Optional visualization (requires --visualize flag)
# ---------------------------------------------------------------------------


def test_visualize_point_cloud(sim, request):
    """Show RGB renders, depth maps, and a 3-D point cloud.

    Skipped unless pytest is invoked with ``--visualize``.
    """
    if not request.config.getoption("--visualize", default=False):
        pytest.skip("Pass --visualize to enable this test")

    import matplotlib.pyplot as plt  # pylint: disable=import-outside-toplevel
    import mujoco  # type: ignore[import-untyped]  # pylint: disable=import-outside-toplevel
    from mpl_toolkits.mplot3d import (  # type: ignore[import-untyped]  # noqa: F401  # pylint: disable=import-outside-toplevel,unused-import
        Axes3D,)

    mj_model = sim.model.mj_model
    ctx = sim._render_context_offscreen  # pylint: disable=protected-access
    width, height = 320, 240

    # Collect named cameras
    camera_names = [
        mujoco.mj_id2name(
            mj_model, mujoco.mjtObj.mjOBJ_CAMERA, i
        )  # pylint: disable=no-member
        for i in range(mj_model.ncam)
    ]
    camera_names = [n for n in camera_names if n is not None]
    name2id = {
        mujoco.mj_id2name(
            mj_model, mujoco.mjtObj.mjOBJ_CAMERA, i
        ): i  # pylint: disable=no-member
        for i in range(mj_model.ncam)
    }

    n_cams = len(camera_names)
    assert n_cams > 0, "No named cameras found in model"

    # --- RGB renders ---
    fig_rgb, axes_rgb = plt.subplots(
        1, n_cams, figsize=(4 * n_cams, 3.5), squeeze=False
    )
    fig_rgb.suptitle("RGB renders", fontsize=10)
    for col, cname in enumerate(camera_names):
        ctx.render(width=width, height=height, camera_id=name2id[cname])
        result = ctx.read_pixels(width, height, depth=False)
        rgb_img = result[0] if isinstance(result, tuple) else result
        axes_rgb[0][col].imshow(rgb_img)
        axes_rgb[0][col].set_title(cname, fontsize=9)
        axes_rgb[0][col].axis("off")
    fig_rgb.tight_layout()

    # --- Depth renders ---
    fig_depth, axes_depth = plt.subplots(
        1, n_cams, figsize=(4 * n_cams, 3.5), squeeze=False
    )
    fig_depth.suptitle("Metric depth (m)", fontsize=10)
    for col, cname in enumerate(camera_names):
        ctx.render(width=width, height=height, camera_id=name2id[cname])
        _, depth_raw = ctx.read_pixels(width, height, depth=True)
        z_m = depth_buffer_to_metric(depth_raw, mj_model)
        z_vis = np.where(np.isfinite(z_m), z_m, np.nan)
        im = axes_depth[0][col].imshow(z_vis, cmap="turbo")
        axes_depth[0][col].set_title(cname, fontsize=9)
        axes_depth[0][col].axis("off")
        fig_depth.colorbar(  # pylint: disable=line-too-long
            im, ax=axes_depth[0][col], fraction=0.046, pad=0.04, label="m"
        )
    fig_depth.tight_layout()

    # --- 3-D point cloud ---
    pc = generate_scene_point_cloud(
        sim, camera_names=camera_names, width=width, height=height
    )
    assert len(pc) > 0, "Point cloud is empty — nothing to visualize"

    rng = np.random.default_rng(0)
    max_pts = 30_000
    idx = rng.choice(len(pc), min(max_pts, len(pc)), replace=False)
    colours = pc.rgb[idx].astype(np.float32) / 255.0

    fig_3d = plt.figure(figsize=(9, 7))
    ax_3d = fig_3d.add_subplot(111, projection="3d")
    ax_3d.scatter(
        pc.xyz[idx, 0],
        pc.xyz[idx, 1],
        pc.xyz[idx, 2],
        c=colours,
        s=0.5,
        marker=".",
        linewidths=0,
    )
    ax_3d.set_xlabel("X (m)")
    ax_3d.set_ylabel("Y (m)")
    ax_3d.set_zlabel("Z (m)")  # type: ignore[attr-defined]
    ax_3d.set_title(f"Point cloud — {len(pc):,} pts from {camera_names}")
    fig_3d.tight_layout()

    plt.show()


def test_save_point_cloud(sim, request):
    """Save the scene point cloud to a .npz file for offline visualization.

    Skipped unless pytest is invoked with ``--save-point-cloud <path>``.

    Example::

        pytest tests/envs/dynamic3d/test_point_cloud.py \\
            --save-point-cloud /tmp/scene_pc.npz -v -s

    The saved file contains three arrays that can be loaded with
    ``np.load("/tmp/scene_pc.npz")``:

    * ``xyz``             — (N, 3) float32 world-frame positions
    * ``rgb``             — (N, 3) uint8 colours [0, 255]
    * ``camera_indices``  — (N,) int32 per-point camera index
    * ``camera_names``    — stored as a JSON string in the ``allow_pickle=False``
                            compatible format (use ``str(arr)`` to read back)
    """
    save_path = request.config.getoption("--save-point-cloud", default=None)
    if save_path is None:
        pytest.skip("Pass --save-point-cloud <path> to enable this test")

    pc = generate_scene_point_cloud(sim, width=320, height=240)
    assert len(pc) > 0, "Point cloud is empty — nothing to save"

    import json  # pylint: disable=import-outside-toplevel

    out = Path(save_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        xyz=pc.xyz,
        rgb=pc.rgb,
        camera_indices=pc.camera_indices,
        camera_names=np.array(json.dumps(pc.camera_names)),
    )
    print(f"\nPoint cloud saved to: {out}  ({len(pc):,} points)")
    print(f"  Load with: data = np.load('{out}')")
    print("  Arrays:    data['xyz'], data['rgb'], data['camera_indices']")
    print(f"  Cameras:   {pc.camera_names}")
