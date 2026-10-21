"""Tests that static collision boxes are observed only by tasks that opt in."""

import pytest

import kinder
from kinder.envs.dynamic3d.object_types import MujocoStaticColliderType


def _make(env_id: str):
    kinder.register_all_environments()
    return kinder.make(
        env_id, render_mode="rgb_array", scene_bg=False, allow_state_access=True
    )


def test_shelf3d_does_not_observe_static_colliders():
    """Shelf3D-o1 does not opt in, so it keeps its original 45-D observation."""
    env = _make("kinder/Shelf3D-o1-v0")
    inner = getattr(env.unwrapped, "_object_centric_env")
    try:
        observation, _ = env.reset(seed=0)
        assert observation.shape == (45,)
        assert env.observation_space.shape == (45,)
        state = inner.get_state()
        assert not state.get_objects(MujocoStaticColliderType)
        assert not any(
            name.startswith("collider:") for name in state.get_object_names()
        )
    finally:
        env.close()


@pytest.mark.parametrize("env_id", ["kinder/Tossing3D-o1-v0", "kinder/Tossing3D-o2-v0"])
def test_tossing3d_observes_static_colliders(env_id: str):
    """Tossing3D opts in, so its room walls and barrier stay in the state."""
    env = _make(env_id)
    inner = getattr(env.unwrapped, "_object_centric_env")
    try:
        env.reset(seed=0)
        names = [
            obj.name for obj in inner.get_state().get_objects(MujocoStaticColliderType)
        ]
        assert len(names) >= 7
        assert any(name.startswith("collider:cuboid_barrier:") for name in names)
    finally:
        env.close()
