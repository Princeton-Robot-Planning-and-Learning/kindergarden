"""Tests for masstransport3d.py and kinematic_masstransport3d.py."""

# pylint: disable=protected-access

import numpy as np
import pytest
from pybullet_helpers.geometry import Pose, SE2Pose, multiply_poses, set_pose
from pybullet_helpers.inverse_kinematics import check_body_collisions

import kinder
from kinder.envs.dynamic3d.kinematic_masstransport3d import (
    MassTransport3DEnvConfig,
    ObjectCentricKinematicMassTransport3DEnv,
    holding_torques_at,
)
from kinder.envs.dynamic3d.masstransport3d import ObjectCentricMassTransport3DEnv

ENV_CLASSES = [
    ObjectCentricKinematicMassTransport3DEnv,
    ObjectCentricMassTransport3DEnv,
]
_NO_OP = np.zeros(11, dtype=np.float32)
_CLOSE = np.array([0.0] * 10 + [-1.0], dtype=np.float32)


def _make(env_cls, num_items=2, **config_kwargs):
    return env_cls(
        num_items=num_items,
        config=MassTransport3DEnvConfig(**config_kwargs),
        realistic_bg=False,
        allow_state_access=True,
    )


def _grasp_item_in_place(env, name):
    """Move item `name` into the gripper and close on it; return the step info."""
    ee_pose = env.robot.arm.get_end_effector_pose()
    item_id = env._object_name_to_pybullet_id(name)  # pylint: disable=protected-access
    set_pose(
        item_id, multiply_poses(ee_pose, Pose((0.0, 0.0, 0.02))), env.physics_client_id
    )
    _, _, _, _, info = env.step(_CLOSE)
    return info


@pytest.mark.parametrize(
    "env_id", ["kinder/MassTransport3D-o4-v0", "kinder/KinematicMassTransport3D-o4-v0"]
)
def test_registered_variants_reset_and_step(env_id):
    """Both variants are registered, reset, and report torque measurements."""
    kinder.register_all_environments()
    env = kinder.make(env_id, realistic_bg=False)
    obs, _ = env.reset(seed=0)
    assert env.observation_space.contains(obs)
    obs, _, terminated, truncated, info = env.step(_NO_OP)
    assert env.observation_space.contains(obs)
    assert not terminated and not truncated
    assert {"arm_torques", "payload_torques", "torque_violation", "num_trips"} <= set(
        info
    )
    assert not info["torque_violation"]
    env.close()


@pytest.mark.parametrize("env_cls", ENV_CLASSES)
def test_masses_are_hidden_and_settable(env_cls):
    """Masses are drawn from the configured range unless reset fixes them."""
    env = _make(env_cls, num_items=4)
    env.reset(seed=0)
    low, high = env.config.item_mass_range
    assert all(low <= m <= high for m in env.item_masses.values())
    fixed = {name: 0.5 + i for i, name in enumerate(env.item_names)}
    env.reset(seed=0, options={"item_masses": fixed})
    assert env.item_masses == fixed


@pytest.mark.parametrize("env_cls", ENV_CLASSES)
def test_items_start_in_distinct_grid_cells_off_the_tote(env_cls):
    """Items start one per grid cell, spaced for the open fingers, clear of the tote."""
    env = _make(env_cls, num_items=12)
    for seed in range(3):
        state, _ = env.reset(seed=seed)
        positions = [
            np.asarray(state.get_object_pose(n).position) for n in env.item_names
        ]
        for i, a in enumerate(positions):
            for b in positions[i + 1 :]:
                assert np.max(np.abs(a[:2] - b[:2])) >= env.config.item_grid_step - 1e-6
        tote_id = env.tote_id
        for name in env.item_names:
            item_id = env._object_name_to_pybullet_id(name)
            assert not check_body_collisions(
                item_id, tote_id, env.physics_client_id, distance_threshold=0.03
            )
        assert not env.goal_reached()


def test_too_many_items_for_the_grid_is_an_error():
    """More items than free cells cannot be placed."""
    env = _make(ObjectCentricKinematicMassTransport3DEnv, num_items=13)
    with pytest.raises(ValueError):
        env.reset(seed=0)


def test_payload_torques_agree_between_kinematic_and_physics():
    """The static torque model matches what the torque-capped motors exert."""
    measured = {}
    for env_cls in ENV_CLASSES:
        env = _make(env_cls)
        env.reset(seed=0, options={"item_masses": {"item0": 1.5, "item1": 1.0}})
        info = _grasp_item_in_place(env, "item0")
        assert env._grasped_object == "item0"  # pylint: disable=protected-access
        for _ in range(5):
            _, _, _, _, info = env.step(_NO_OP)
        measured[env_cls] = info["payload_torques"]
    kinematic = measured[ObjectCentricKinematicMassTransport3DEnv]
    physics = measured[ObjectCentricMassTransport3DEnv]
    assert np.max(np.abs(kinematic)) > 1.0, "a 1.5 kg item should load the arm"
    assert np.allclose(kinematic, physics, atol=0.5)


def test_overloaded_grasp_slips_in_the_kinematic_env():
    """An item the arm cannot hold slips out of the kinematic grasp."""
    env = _make(ObjectCentricKinematicMassTransport3DEnv)
    env.reset(seed=0, options={"item_masses": {"item0": 20.0, "item1": 1.0}})
    info = _grasp_item_in_place(env, "item0")
    assert info["torque_violation"]
    assert env._grasped_object is None  # pylint: disable=protected-access


def test_overloaded_arm_sags_under_physics():
    """Under physics the arm holds on but saturates and cannot keep its pose."""
    env = _make(ObjectCentricMassTransport3DEnv)
    env.reset(seed=0, options={"item_masses": {"item0": 20.0, "item1": 1.0}})
    _grasp_item_in_place(env, "item0")
    _, _, _, _, info = env.step(_NO_OP)
    assert info["torque_violation"]
    assert info["tracking_error"] > 0.01


def test_physics_env_is_still_at_rest():
    """With nothing held, the motors hold the arm still and items stay put."""
    env = _make(ObjectCentricMassTransport3DEnv)
    state, _ = env.reset(seed=0)
    for _ in range(5):
        new_state, _, _, _, info = env.step(_NO_OP)
    assert not info["torque_violation"]
    assert info["tracking_error"] < 1e-3
    for name in env.item_names:
        assert new_state.get_object_pose(name).allclose(
            state.get_object_pose(name), atol=1e-3
        )


@pytest.mark.parametrize("env_cls", ENV_CLASSES)
def test_crossing_to_the_destination_counts_a_trip(env_cls):
    """A base move from y < 0 to y >= 0 is one trip; max_trips truncates after."""
    env = _make(env_cls, max_trips=0)
    env.reset(seed=0)
    env.robot.set_base(SE2Pose(-0.5, -0.1, 0.0))
    _, _, _, truncated, info = env.step(
        np.array([0.0, 0.2] + [0.0] * 9, dtype=np.float32)
    )
    assert info["num_trips"] == 1
    assert truncated


@pytest.mark.parametrize("env_cls", ENV_CLASSES)
def test_goal_needs_every_item_directly_on_the_destination(env_cls):
    """Items on the destination table reach the goal; one in the tote does not."""
    env = _make(env_cls)
    env.reset(seed=0)
    config = env.config
    x, y, _ = config.dest_table_pose.position
    z = config.table_top_z + config.item_size / 2 + 1e-3
    for i, name in enumerate(env.item_names):
        item_id = env._object_name_to_pybullet_id(
            name
        )  # pylint: disable=protected-access
        set_pose(item_id, Pose((x, y - 0.2 + 0.2 * i, z)), env.physics_client_id)
    assert env.goal_reached()
    tote_pose = Pose((x, y + 0.1, config.tote_initial_pose.position[2]))
    set_pose(env.tote_id, tote_pose, env.physics_client_id)
    in_tote = multiply_poses(tote_pose, Pose((0.0, 0.0, -0.02)))
    item_id = env._object_name_to_pybullet_id(
        "item0"
    )  # pylint: disable=protected-access
    set_pose(item_id, in_tote, env.physics_client_id)
    assert not env.goal_reached()


def test_holding_torques_scale_with_the_payload():
    """The static model adds the payload's torques to the arm's own."""
    env = _make(ObjectCentricKinematicMassTransport3DEnv)
    env.reset(seed=0)
    joints = list(env.config.initial_joints)
    offset = Pose((0.0, 0.0, 0.1))
    arm = env.robot.arm
    client = env.physics_client_id
    empty = holding_torques_at(arm, client, joints, 0.0, None)
    one = holding_torques_at(arm, client, joints, 1.0, offset)
    two = holding_torques_at(arm, client, joints, 2.0, offset)
    assert np.allclose(two - empty, 2 * (one - empty), atol=1e-6)
