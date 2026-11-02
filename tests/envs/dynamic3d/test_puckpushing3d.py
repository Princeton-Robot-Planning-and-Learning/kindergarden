"""Tests for puckpushing3d.py."""

import math

import numpy as np
from gymnasium.spaces import Box

import kinder
from kinder.envs.dynamic3d.puckpushing3d import (
    ObjectCentricPuckPushing3DEnv,
    PuckPushing3DEnvConfig,
)
from kinder.envs.puckpushing_utils import PuckPhysics

NOISELESS = PuckPhysics(
    speed_noise=0.0, angle_noise=0.0, speed_obs_noise=0.0, position_obs_noise=0.0
)
FRICTIONS = (0.03, 0.07, 0.15)


def test_puckpushing3d_spaces():
    """Observations are fixed-size vectors and actions are (angle, speed)."""
    kinder.register_all_environments()
    env = kinder.make("kinder/PuckPushing3D-threeroutes-v0")
    assert isinstance(env.observation_space, Box)
    obs, _ = env.reset(seed=0)
    assert env.observation_space.contains(obs)
    obs, reward, terminated, _, info = env.step(np.array([math.pi / 2, 0.5]))
    assert env.observation_space.contains(obs)
    assert reward == -1.0 and info["push_feasible"] and not terminated


def test_puckpushing3d_matches_the_analytic_slide():
    """Without noise, the simulated stop matches the analytic model, bounces
    included."""
    env = ObjectCentricPuckPushing3DEnv(PuckPushing3DEnvConfig(physics=NOISELESS))
    world = env.world
    # The second push bounces off the top wall.
    for seed, (angle, speed) in enumerate(
        [(math.pi / 2, 0.6), (1.2, 1.0), (-2.0, 0.8)]
    ):
        env.reset(seed=seed, options={"frictions": FRICTIONS})
        puck = np.array(world.puck_map.puck_start)
        _, _, _, _, info = env.step(np.array([angle, speed]))
        predicted = world.slide(
            world.release_point(puck, angle), info["release_speed"], angle, FRICTIONS
        )
        assert np.linalg.norm(predicted.stop - info["stop_position"]) < 0.03


def test_puckpushing3d_robot_follows_and_render():
    """The robot ends up behind where it let go, and the scene renders."""
    env = ObjectCentricPuckPushing3DEnv(PuckPushing3DEnvConfig(physics=NOISELESS))
    env.reset(seed=0, options={"frictions": FRICTIONS})
    puck = np.array(env.world.puck_map.puck_start)
    env.step(np.array([math.pi / 2, 0.6]))
    base = env.robot.get_base()
    station = env.world.push_station(puck, math.pi / 2)
    assert math.isclose(base.rot, math.pi / 2, abs_tol=1e-6)
    assert math.isclose(
        base.y - station[1], env.config.physics.accel_distance, abs_tol=0.02
    )
    assert env.render().ndim == 3


def test_puckpushing3d_paddle_clears_the_arm_and_base():
    """Held for a strike at any stance, or carried for driving, the paddle touches
    no part of the robot except the gripper holding it."""
    env = ObjectCentricPuckPushing3DEnv()
    env.reset(seed=0)
    base = env.robot.get_base()
    heading = np.array([math.cos(base.rot), math.sin(base.rot)])
    face = np.array([base.x, base.y]) + env.config.physics.paddle_offset * heading
    for stance in np.radians([-60, -30, 0, 30, 60]):
        env._place_paddle(face, base.rot - stance)  # pylint: disable=protected-access
        assert env.paddle_robot_gap() > 0.01
    env._stow_paddle()  # pylint: disable=protected-access
    assert env.paddle_robot_gap() > 0.05
    env.close()
