"""Tests that set_state fully restores the Franka FR3's gripper.

The FR3 carries the same Robotiq 2F-85 as the TidyBot and restores its robot state
through the same shared helper, so it had the same defect and is fixed by the same
change; this pins that the shared path really does reach it.
"""

from pathlib import Path

import numpy as np

import kinder
from kinder.envs.dynamic3d.envs import ObjectCentricFranka3DEnv

_TASK_CONFIG_PATH = (
    Path(kinder.__path__[0])
    / "envs"
    / "dynamic3d"
    / "tasks"
    / "FrankaPickPlace3D"
    / "FrankaPickPlace3D-o1.json"
)

_FINGER_JOINT_NAMES = [
    f"robot_{side}_{link}_joint"
    for side in ("left", "right")
    for link in ("driver", "coupler", "follower", "spring_link")
]

_CLOSE_STEPS = 40


def _make_env() -> ObjectCentricFranka3DEnv:
    env = ObjectCentricFranka3DEnv(
        num_objects=1,
        task_config_path=str(_TASK_CONFIG_PATH),
        scene_bg=False,
        allow_state_access=True,
    )
    env.reset(seed=0)
    return env


def _finger_joint_positions(env: ObjectCentricFranka3DEnv) -> np.ndarray:
    sim = env._robot_env.sim  # pylint: disable=protected-access
    return np.array(
        [
            sim.data.mj_data.qpos[sim.model.get_joint_qpos_addr(name)]
            for name in _FINGER_JOINT_NAMES
        ]
    )


def _close_gripper_on_nothing(env: ObjectCentricFranka3DEnv) -> None:
    # Arm is commanded as deltas (act_delta), the gripper absolutely, so zeros here
    # hold the pose while the fingers shut.
    action = np.zeros(8, dtype=np.float32)
    action[7] = 1.0
    for _ in range(_CLOSE_STEPS):
        env.step(action)


def test_fr3_set_state_restores_the_gripper_finger_joints():
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
