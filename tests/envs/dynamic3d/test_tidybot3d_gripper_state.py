"""Tests that set_state restores the gripper's joints, not only its command."""

from pathlib import Path

import numpy as np

import kinder
from kinder.envs.dynamic3d.envs import ObjectCentricTidyBot3DEnv

_TASK_CONFIG_PATH = (
    Path(kinder.__path__[0])
    / "envs"
    / "dynamic3d"
    / "tasks"
    / "Tossing3D"
    / "Tossing3D-o1.json"
)

# Control steps for the fingers to travel their whole range and settle.
_SETTLE_STEPS = 50

# Every joint of the Robotiq 2F-85, in model order. Spelled out here rather than
# imported from the code under test, so a wrong list there fails this test.
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


def _make_env() -> ObjectCentricTidyBot3DEnv:
    env = ObjectCentricTidyBot3DEnv(
        num_objects=1,
        task_config_path=str(_TASK_CONFIG_PATH),
        scene_bg=False,
        allow_state_access=True,
    )
    env.reset(seed=0)
    return env


def _gripper_qpos_qvel(
    env: ObjectCentricTidyBot3DEnv,
) -> tuple[np.ndarray, np.ndarray]:
    """All eight gripper joint positions and velocities, read straight from MuJoCo.

    Resolved by joint name rather than through the robot env's "gripper" view, so this
    stays an independent oracle for the tests that check what that view covers.
    """
    robot_env = env._robot_env  # pylint: disable=protected-access
    assert robot_env is not None
    model = robot_env.sim.model
    names = [f"{robot_env.name}_{s}" for s in _GRIPPER_JOINT_SUFFIXES]
    qpos = robot_env.sim.data.mj_data.qpos
    qvel = robot_env.sim.data.mj_data.qvel
    return (
        np.array([qpos[model.get_joint_qpos_addr(n)] for n in names], dtype=float),
        np.array([qvel[model.get_joint_qvel_addr(n)] for n in names], dtype=float),
    )


def _drive_gripper(
    env: ObjectCentricTidyBot3DEnv, gripper: float, steps: int = _SETTLE_STEPS
) -> None:
    """Step with zero base and arm deltas, so only the gripper moves."""
    action = np.zeros(11, dtype=np.float32)
    action[10] = gripper
    for _ in range(steps):
        env.step(action)


def _grasp_the_cube(env: ObjectCentricTidyBot3DEnv) -> None:
    """Open the fingers, teleport cube_0 between them, and close on it."""
    _drive_gripper(env, 0.0)
    sim = env._robot_env.sim  # pylint: disable=protected-access
    pinch = np.array(sim.data.get_site_xpos("robot_pinch_site"), dtype=float)
    state = env._get_current_state()  # pylint: disable=protected-access
    cube = state.get_object_from_name("cube_0")
    state.set(cube, "x", pinch[0])
    state.set(cube, "y", pinch[1])
    state.set(cube, "z", pinch[2])
    env.set_state(state)
    _drive_gripper(env, 1.0)


def test_tidybot3d_the_gripper_views_cover_every_robotiq_joint():
    """The robot env's "gripper" qpos/qvel views must expose all eight joints.

    Only the two drivers are actuated; the six passive coupler/spring-link/follower
    joints carry the grasp geometry. The views are where this robot says which joints
    are its gripper, so anything that reads or restores the hand goes through them the
    same way the base and arm go through their own views.
    """
    env = _make_env()
    robot_env = env._robot_env  # pylint: disable=protected-access
    assert robot_env is not None
    model = robot_env.sim.model
    names = [f"{robot_env.name}_{s}" for s in _GRIPPER_JOINT_SUFFIXES]

    assert list(robot_env.qpos["gripper"].indices) == [
        model.get_joint_qpos_addr(n) for n in names
    ]
    assert list(robot_env.qvel["gripper"].indices) == [
        model.get_joint_qvel_addr(n) for n in names
    ]
    env.close()


def test_tidybot3d_restoring_a_grasp_after_a_release_restores_the_grasp():
    """A state captured mid-grasp restores a grasp, even after a release.

    This is the retry-after-release case a trajectory sampler hits. `pos_gripper` is the
    commanded ctrl value, so a state carrying only the command leaves the fingers
    wherever the release left them -- open, around a cube the state says is held.

    What this does not claim is that the replay is bit-faithful: MuJoCo's solver
    warm-start is not part of the object-centric state.
    """
    env = _make_env()

    _grasp_the_cube(env)
    holding_state = env._get_current_state()  # pylint: disable=protected-access
    holding_qpos, holding_qvel = _gripper_qpos_qvel(env)
    cube = holding_state.get_object_from_name("cube_0")
    held_z = holding_state.get(cube, "z")
    assert np.max(holding_qpos) < 0.5, (
        f"the gripper closed to {holding_qpos}, so it closed on air: the cube is not "
        f"between the fingers and the rest of this test proves nothing"
    )

    # Release, as a failed rollout would end: the hand opens and the cube falls out.
    _drive_gripper(env, 0.0)
    released_qpos, _ = _gripper_qpos_qvel(env)
    released_state = env._get_current_state()  # pylint: disable=protected-access
    assert released_state.get(cube, "z") < held_z - 0.05
    # Every joint has to move by far more than the restore is checked at, or a fix
    # that restored only the two driver joints would still pass.
    assert np.all(np.abs(holding_qpos - released_qpos) > 1e-4)

    # Retry: restore the same start state, as a sampler would before resampling.
    env.set_state(holding_state)
    restored_qpos, restored_qvel = _gripper_qpos_qvel(env)
    assert np.allclose(restored_qpos, holding_qpos, atol=1e-9)
    assert np.allclose(restored_qvel, holding_qvel, atol=1e-9)

    # And the restored grasp holds when the retry steps on from there.
    _drive_gripper(env, 1.0, steps=10)
    retried_state = env._get_current_state()  # pylint: disable=protected-access
    assert retried_state.get(cube, "z") > held_z - 0.02
    env.close()
