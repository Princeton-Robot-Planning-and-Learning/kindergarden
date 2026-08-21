"""Tests for MujocoEnv's per-substep control schedule."""

import numpy as np
import pytest
from gymnasium.wrappers import RenderCollection

from kinder.envs.dynamic3d.mujoco_utils import (
    CONTROL_SCHEDULE_TIMESTEP,
    SIMULATION_TIMESTEP,
    MjObs,
    MjSim,
    MujocoEnv,
)

CONTROL_FREQUENCY = 10.0
TICKS_PER_STEP = int((1.0 / CONTROL_FREQUENCY) / SIMULATION_TIMESTEP)
TICKS_PER_ROW = int(round(CONTROL_SCHEDULE_TIMESTEP / SIMULATION_TIMESTEP))
SCHEDULE_ROWS = TICKS_PER_STEP // TICKS_PER_ROW

_XML = """
<mujoco>
  <option gravity="0 0 0"/>
  <worldbody>
    <body name="slider" pos="0 0 0">
      <joint name="slide" type="slide" axis="1 0 0"/>
      <geom type="box" size="0.1 0.1 0.1" mass="1"/>
    </body>
  </worldbody>
  <actuator>
    <motor joint="slide" name="push" gear="1"/>
  </actuator>
</mujoco>
"""


class _SliderEnv(MujocoEnv):
    """A one-actuator MujocoEnv, so a schedule's effect is a single number."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 10}

    def __init__(self) -> None:
        super().__init__(control_frequency=CONTROL_FREQUENCY)
        self.render_mode = "rgb_array"

    def reward(self, obs: MjObs) -> float:
        return 0.0

    def render(self):
        return np.zeros((2, 2, 3), dtype=np.uint8)


def make_env() -> _SliderEnv:
    """A reset slider env, ready to step."""
    env = _SliderEnv()
    env.reset(options={"xml": _XML})
    return env


def replay_ticks(env: _SliderEnv, rows: np.ndarray) -> None:
    """Drive the substep loop directly, from one explicit row per physics tick."""
    assert env.timestep is not None
    assert env.sim is not None
    env.timestep += 1
    for tick in range(TICKS_PER_STEP):
        env._update_ctrl(rows[tick])  # pylint: disable=protected-access
        env.sim.forward()
        env.sim.step()


def test_a_1d_action_is_held_for_every_tick_of_the_control_period():
    """The 1-D path is a broadcast view, so it must match an explicit replay exactly."""
    stepped, replayed = make_env(), make_env()
    action = np.array([0.7])

    stepped.step(action)
    replay_ticks(replayed, np.repeat(action[None], TICKS_PER_STEP, axis=0))

    assert np.array_equal(stepped.get_obs()["qpos"], replayed.get_obs()["qpos"])
    assert np.array_equal(stepped.get_obs()["qvel"], replayed.get_obs()["qvel"])
    assert stepped.timestep == replayed.timestep


def test_a_schedule_holds_each_row_for_one_millisecond():
    """Row j drives the j-th millisecond, checked against a tick-by-tick replay."""
    scheduled, replayed, plain = make_env(), make_env(), make_env()
    schedule = np.linspace(-1.0, 1.0, SCHEDULE_ROWS)[:, None]

    scheduled.step(schedule)
    replay_ticks(replayed, np.repeat(schedule, TICKS_PER_ROW, axis=0))
    plain.step(schedule[0])

    assert np.array_equal(scheduled.get_obs()["qpos"], replayed.get_obs()["qpos"])
    assert np.array_equal(scheduled.get_obs()["qvel"], replayed.get_obs()["qvel"])
    # Otherwise the two would agree by both ignoring every row after the first.
    assert not np.allclose(scheduled.get_obs()["qpos"], plain.get_obs()["qpos"])


@pytest.mark.parametrize("rows", [1, SCHEDULE_ROWS - 1, SCHEDULE_ROWS + 1])
def test_a_schedule_shorter_or_longer_than_the_control_period_is_rejected(rows):
    """A schedule covers the whole period, so a partial one is a caller bug.

    Accepting one would mean inventing a rule for the ticks it does not cover, and
    every such rule silently reinterprets the caller's timing.
    """
    env = make_env()
    with pytest.raises(AssertionError, match="control schedule"):
        env.step(np.zeros((rows, 1)))


def test_a_schedule_survives_a_gymnasium_wrapper():
    """Why the schedule rides on the action: a wrapper forwards only that.

    RenderCollection takes one positional argument and passes it by value, so a
    keyword or a second parameter would not reach the env it wraps -- and the
    recording path wraps every env it films.
    """
    unwrapped, wrapped = make_env(), RenderCollection(make_env())
    schedule = np.linspace(-1.0, 1.0, SCHEDULE_ROWS)[:, None]

    unwrapped.step(schedule)
    wrapped.step(schedule)

    assert np.array_equal(
        unwrapped.get_obs()["qpos"], wrapped.unwrapped.get_obs()["qpos"]
    )
    assert len(wrapped.render()) == 1


def test_a_control_period_does_not_call_forward_dynamics_twice_per_tick(monkeypatch):
    """mj_step already runs the forward pass, so the loop must not run it as well.

    Contact solving dominates a control period, and calling forward() before step()
    pays for it twice. This counts rather than times, so it fails on a slow machine
    for the right reason instead of flaking. Patched on the class via monkeypatch:
    binding a counter onto the instance would close over the bound method and leave
    a reference cycle, which defers the env's teardown into a later test -- and a
    render context freed at that point takes the process down on macOS.
    """
    calls = 0
    real_forward = MjSim.forward

    def counting_forward(self) -> None:
        nonlocal calls
        calls += 1
        real_forward(self)

    # After make_env, so the forward() that reset legitimately runs is not counted.
    env = make_env()
    monkeypatch.setattr(MjSim, "forward", counting_forward)
    env.step(np.array([0.7]))

    assert calls == 0, f"step() ran mj_forward {calls} times; mj_step already does"


def test_dropping_the_per_tick_forward_leaves_the_trajectory_untouched():
    """The removed call was dead work: nothing reads mjData between it and mj_step.

    Pinned as an equality rather than a tolerance because the claim is that the two
    loops issue the same MuJoCo calls in the same order, not that they merely agree
    to within integration error.
    """
    stepped, with_forward = make_env(), make_env()
    action = np.array([0.7])

    stepped.step(action)
    # replay_ticks still calls forward() before every step(), i.e. the old loop.
    replay_ticks(with_forward, np.repeat(action[None], TICKS_PER_STEP, axis=0))

    assert np.array_equal(stepped.get_obs()["qpos"], with_forward.get_obs()["qpos"])
    assert np.array_equal(stepped.get_obs()["qvel"], with_forward.get_obs()["qvel"])
