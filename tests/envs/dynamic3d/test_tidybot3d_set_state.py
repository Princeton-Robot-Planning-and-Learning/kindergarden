"""Tests that set_state fully restores the TidyBot's gripper."""

from pathlib import Path

import numpy as np

from kinder.envs.dynamic3d.envs import ObjectCentricTidyBot3DEnv

_TEST_TASKS = Path(__file__).parent / "test_tasks"

# The Robotiq 2F-85 linkage: two driver joints plus the coupler, spring-link and
# follower joints that trail them. Only the drivers are named in qpos["gripper"].
_FINGER_JOINT_NAMES = [
    f"robot_{side}_{link}_joint"
    for side in ("left", "right")
    for link in ("driver", "coupler", "follower", "spring_link")
]

_CLOSE_STEPS = 40


def _make_env() -> ObjectCentricTidyBot3DEnv:
    env = ObjectCentricTidyBot3DEnv(
        num_objects=3,
        task_config_path=str(_TEST_TASKS / "tidybot-ground-o3.json"),
        allow_state_access=True,
    )
    env.reset(seed=0)
    return env


def _finger_joint_positions(env: ObjectCentricTidyBot3DEnv) -> np.ndarray:
    sim = env._robot_env.sim  # pylint: disable=protected-access
    return np.array(
        [
            sim.data.mj_data.qpos[sim.model.get_joint_qpos_addr(name)]
            for name in _FINGER_JOINT_NAMES
        ]
    )


def _close_gripper_on_nothing(env: ObjectCentricTidyBot3DEnv) -> None:
    # Base and arm are commanded as deltas (act_delta), the gripper absolutely, so
    # zeros here hold the pose while the fingers shut.
    action = np.zeros(11)
    action[10] = 1.0
    for _ in range(_CLOSE_STEPS):
        env.step(action)


def test_set_state_restores_the_gripper_finger_joints():
    """Restoring a state must put the fingers back, not just the gripper command."""
    untouched = _make_env()
    expected = _finger_joint_positions(untouched)
    untouched.close()

    env = _make_env()
    initial_state = env.get_state()
    _close_gripper_on_nothing(env)
    env.set_state(initial_state)
    restored = _finger_joint_positions(env)
    env.close()

    assert np.allclose(restored, expected, atol=1e-2), (
        f"gripper fingers not restored: max |diff| = "
        f"{np.abs(restored - expected).max():.4f} rad"
    )


def test_set_state_is_deterministic_regardless_of_gripper_history():
    """The same state must yield the same simulator whatever the gripper just did."""
    env = _make_env()
    initial_state = env.get_state()
    env.set_state(initial_state)
    without_history = _finger_joint_positions(env)
    env.close()

    env = _make_env()
    initial_state = env.get_state()
    _close_gripper_on_nothing(env)
    env.set_state(initial_state)
    with_history = _finger_joint_positions(env)
    env.close()

    assert np.allclose(without_history, with_history, atol=1e-6)
