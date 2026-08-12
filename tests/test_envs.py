"""Common tests for all environments."""

import gymnasium
import pytest
from gymnasium.utils.env_checker import check_env

import kinder
from kinder import _BACKEND_DEPS


def test_env_classes_declare_a_known_backend():
    """Tests that every registered environment class declares a known backend."""
    kinder.register_all_environments()
    env_classes = kinder.get_env_classes()
    assert len(env_classes) > 0
    for metadata in env_classes.values():
        assert metadata["backend"] in _BACKEND_DEPS


@pytest.mark.skip(reason="Memory issue to investigate - test causes OOM in CI")
def test_env_make_and_check_env():
    """Tests that all registered environments can be created with make.

    Also calls gymnasium.utils.env_checker.check_env() to test API functions.
    """
    kinder.register_all_environments()
    env_ids = kinder.get_all_env_ids()
    assert len(env_ids) > 0
    for env_id in env_ids:
        # TidyBot mujoco_env is currently unstable, so we skip it.
        if "TidyBot" in env_id or "RBY1A" in env_id:
            continue
        # We currently require all environments to have RGB rendering.
        make_kwargs = {"render_mode": "rgb_array"}
        entrypoint = gymnasium.registry[env_id].entry_point
        assert isinstance(entrypoint, str)
        if "kinematic3d." in entrypoint:
            make_kwargs["realistic_bg"] = False
        env = kinder.make(env_id, **make_kwargs)
        assert env.render_mode == "rgb_array"
        assert isinstance(env, gymnasium.Env)
        check_env(env.unwrapped)
        env.close()
