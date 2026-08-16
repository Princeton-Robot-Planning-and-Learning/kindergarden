"""Tests that set_state restores the FR3's gripper joints, not only its command."""

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

# Every joint of the Robotiq 2F-85, in model order. The FR3 carries the same gripper as
# the TidyBot; spelled out here rather than imported from the code under test.
_GRIPPER_JOINT_SUFFIXES = (
    "right_driver_joint",
    "right_coupler_joint",
    "right_spring_link_joint",
    "right_follower_joint",
    "left_driver_joint",
    "left_coupler_joint",
    "left_spring_link_joint",
    "left_follower_joint",
)


def test_franka3d_set_state_restores_every_gripper_joint():
    """The FR3 reads the gripper through the same two helpers as the TidyBot.

    Both robots carry the Robotiq 2F-85 and go through ObjectCentricRobotEnv's
    `_get_arm_and_gripper_pos_data` / `_set_arm_and_gripper_state`, so the features and
    the restore have to work for both or the shared helper is wrong for one of them.
    """
    env = ObjectCentricFranka3DEnv(
        num_objects=1,
        task_config_path=str(_TASK_CONFIG_PATH),
        scene_bg=False,
        allow_state_access=True,
    )
    env.reset(seed=0)

    robot_env = env._robot_env  # pylint: disable=protected-access
    assert robot_env is not None
    model = robot_env.sim.model
    joint_names = [f"{robot_env.name}_{s}" for s in _GRIPPER_JOINT_SUFFIXES]

    def gripper_qpos() -> np.ndarray:
        qpos = robot_env.sim.data.mj_data.qpos
        return np.array(
            [qpos[model.get_joint_qpos_addr(n)] for n in joint_names], dtype=float
        )

    def drive_gripper(gripper: float) -> None:
        action = np.zeros(8, dtype=np.float32)
        action[7] = gripper
        for _ in range(50):
            env.step(action)

    drive_gripper(1.0)
    closed_qpos = gripper_qpos()
    closed_state = env._get_current_state()  # pylint: disable=protected-access

    drive_gripper(0.0)
    # Every joint has to move by far more than the restore is checked at, or a fix that
    # restored only the two driver joints would still pass. The couplers move least,
    # by ~3e-4 rad.
    assert np.all(np.abs(closed_qpos - gripper_qpos()) > 1e-4)

    env.set_state(closed_state)
    assert np.allclose(gripper_qpos(), closed_qpos, atol=1e-9)
    env.close()
