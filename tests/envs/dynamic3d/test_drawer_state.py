"""Observed drawer positions survive state restoration."""

import numpy as np
import pytest

import kinder
from kinder.envs.dynamic3d.object_types import (
    MujocoDrawerObjectType,
    MujocoFixtureObjectType,
)


def test_restore_drawer_positions() -> None:
    """Restore all drawer slides and their attached goal without moving fixtures."""
    kinder.register_all_environments()
    env = kinder.make("kinder/SweepIntoDrawer3D-o5-v0", allow_state_access=True)
    inner = getattr(env.unwrapped, "_object_centric_env")
    try:
        env.reset(seed=0)
        robot_env = getattr(inner, "_robot_env")
        state = inner.get_state()
        drawers = state.get_objects(MujocoDrawerObjectType)
        for drawer in drawers:
            robot_env.set_joint_pos_quat(
                f"{drawer.name}_joint", np.array([0.3], dtype=np.float32)
            )
        # A step refreshes the public state after directly opening simulator joints.
        env.step(np.zeros(env.action_space.shape, dtype=np.float32))
        state = inner.get_state()
        for index in range(5):
            cube = state.get_object_from_name(f"cube_{index}")
            for feature, value in zip(
                ["x", "y", "z"], [0.99, (index - 2) * 0.04, 0.2373], strict=True
            ):
                state.set(cube, feature, value)
        inner.set_state(state)
        assert getattr(inner, "_check_goals")()
        saved = inner.get_state()

        for drawer in drawers:
            robot_env.set_joint_pos_quat(
                f"{drawer.name}_joint", np.array([0.0], dtype=np.float32)
            )
        env.step(np.zeros(env.action_space.shape, dtype=np.float32))
        assert not getattr(inner, "_check_goals")()
        inner.set_state(saved)
        restored = inner.get_state()
        for drawer in drawers:
            assert restored.get(drawer, "pos") == pytest.approx(
                saved.get(drawer, "pos")
            )
        assert getattr(inner, "_check_goals")()
        for fixture in state.get_objects(MujocoFixtureObjectType):
            for feature in ["x", "y", "z", "qw", "qx", "qy", "qz"]:
                assert restored.get(fixture, feature) == saved.get(fixture, feature)
    finally:
        env.close()
        inner.close()
