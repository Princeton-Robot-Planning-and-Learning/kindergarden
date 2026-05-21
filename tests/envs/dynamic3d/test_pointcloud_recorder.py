"""Tests for Dynamic3D point cloud recording."""

from pathlib import Path

import numpy as np
import pytest

from kinder.envs.dynamic3d.pointcloud import (
    PointCloudRecorder,
    stack_point_cloud_frames,
    subsample_point_cloud,
)


def test_subsample_point_cloud() -> None:
    rng = np.random.default_rng(0)
    points = np.random.randn(1000, 3).astype(np.float32)
    out = subsample_point_cloud(points, 100, rng)
    assert out.shape == (100, 3)


def test_stack_point_cloud_frames() -> None:
    frames = [
        np.zeros((10, 3), dtype=np.float32),
        np.ones((5, 3), dtype=np.float32),
    ]
    stacked, counts = stack_point_cloud_frames(frames, max_points=20)
    assert stacked.shape == (2, 20, 3)
    assert counts.tolist() == [10, 5]
    assert np.isnan(stacked[0, 10:, 0]).all()


def test_point_cloud_recorder_short_trajectory(tmp_path: Path) -> None:
    pytest.importorskip("mujoco")
    import kinder  # pylint: disable=import-outside-toplevel
    kinder.register_all_environments()
    env_id = "kinder/Rearrange3D-o1-put_the_boxed_drink_next_to_the_bowl-v0"
    env = kinder.make(env_id)
    recorder = PointCloudRecorder(
        camera_name="tidybot_base",
        max_points_per_frame=1000,
        on_alias=None,
        seed=0,
    )
    try:
        env.reset(seed=0)
        recorder.on_reset(env)
        for _ in range(3):
            env.step(env.action_space.sample())
            recorder.on_step(env)
        assert recorder.num_frames() == 4
        out = recorder.save_npz(tmp_path / "traj.npz", env_id=env_id, demo_seed=0)
        data = np.load(out)
        assert data["points"].shape[0] == 4
        assert data["counts"].shape[0] == 4
        assert str(data["camera"]) in ("robot_base", "tidybot_base")
    finally:
        env.close()
