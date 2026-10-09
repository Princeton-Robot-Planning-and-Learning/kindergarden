"""Fragile-object damage uses physics contacts, not sampled pose heuristics."""

import math

import mujoco
import pytest

# MuJoCo exposes these functions through its native extension.
# pylint: disable=no-member

from kinder.envs.dynamic3d.fragile_tossing import DamageTracker, inside_mat


def test_mat_moves_rotates_and_includes_boundary():
    """The mat uses bin-local coordinates and includes its boundary."""
    assert inside_mat((3, 2), (1, 2), 0, 4)
    assert not inside_mat((3.01, 2), (1, 2), 0, 4)
    assert inside_mat((3.1, 2), (1, 2), math.pi / 4, 4)
    assert not inside_mat((3, 2), (-2, 2), 0, 4)


def make_sim():
    """Build a minimal physical cube, bin, and ground fixture."""
    model = mujoco.MjModel.from_xml_string("""<mujoco><worldbody>
    <geom name="floor" type="plane" size="10 10 .1"/>
    <body name="bin_0" pos="0 0 .1"><geom type="box" size=".15 .15 .1"/></body>
    <body name="cube_0" pos="3 0 1"><freejoint/><geom type="box" size=".025 .025 .025" mass=".1"/></body>
    </worldbody></mujoco>""")
    data = mujoco.MjData(model)
    return model, data


def test_real_impact_once_then_lift_and_impact_again():
    """Resting contacts do not repeatedly charge; later impacts do."""
    model, data = make_sim()
    tracker = DamageTracker(model, mat_size=4, damage_cost=10)
    events = []
    for _ in range(600):
        mujoco.mj_step(model, data)
        events.extend(tracker.update(data))
    # Count contact onsets, including real bounces, but never resting ticks.
    assert len(events) >= 1
    assert all(e["cost"] == 10 and e["cube"] == "cube_0" for e in events)
    for _ in range(100):
        mujoco.mj_step(model, data)
        assert not tracker.update(data)
    data.qpos[2] = 1
    data.qvel[:] = 0
    extra = []
    for _ in range(600):
        mujoco.mj_step(model, data)
        extra.extend(tracker.update(data))
    assert extra


def test_mat_and_placement_exemptions():
    """Protect mat landings and exempt human placement settling."""
    model, data = make_sim()
    tracker = DamageTracker(model, mat_size=4, damage_cost=10)
    data.qpos[:3] = [1, 0, 1]
    for _ in range(600):
        mujoco.mj_step(model, data)
        assert not tracker.update(data)
    data.qpos[:3] = [3, 0, 0.025]
    data.qvel[:] = 0
    tracker.exempt_placement()
    for _ in range(600):
        mujoco.mj_step(model, data)
        assert not tracker.update(data)


@pytest.mark.parametrize("size,cost", [(0, 10), (-1, 10), (4, -1), (4, float("nan"))])
def test_invalid_settings(size, cost):
    """Reject invalid mat sizes and damage prices."""
    model, _ = make_sim()
    with pytest.raises(ValueError):
        DamageTracker(model, mat_size=size, damage_cost=cost)


def test_small_robot_lift_ends_placement_exemption():
    """Even a short robot lift makes a later bare-ground drop chargeable."""
    model, data = make_sim()
    tracker = DamageTracker(model, mat_size=4, damage_cost=10)
    tracker.exempt_placement()
    data.qpos[:3] = [3, 0, 0.06]
    events = []
    for _ in range(300):
        mujoco.mj_step(model, data)
        events.extend(tracker.update(data))
    assert events


def test_heavy_bin_resists_push_and_remains_resettable(monkeypatch):
    """Only variant bins gain inertia; resets can still reposition free bodies."""
    import xml.etree.ElementTree as ET

    import numpy as np

    from kinder.envs.dynamic3d.fragile_tossing import (
        ObjectCentricFragileTossing3DEnv,
        ObjectCentricTidyBot3DEnv,
    )

    xml = """<mujoco><worldbody>
    <geom name="floor" type="plane" size="10 10 .1"/>
    <body name="bin_0" pos="0 0 .1"><freejoint/>
      <geom type="box" size=".15 .15 .1" mass=".1"/>
    </body>
    <body name="cube_0" pos="2 0 1"><freejoint/>
      <geom type="box" size=".025 .025 .025" mass=".1"/>
    </body></worldbody></mujoco>"""
    monkeypatch.setattr(ObjectCentricTidyBot3DEnv, "_create_scene_xml", lambda _: xml)
    env = object.__new__(ObjectCentricFragileTossing3DEnv)
    env.mat_size = 4
    env.bin_mass = 100
    modified = env._create_scene_xml()
    model = mujoco.MjModel.from_xml_string(modified)
    baseline = mujoco.MjModel.from_xml_string(xml)
    body = model.body("bin_0").id
    assert model.body_mass[body] == pytest.approx(100)
    assert model.body_mass[model.body("cube_0").id] == pytest.approx(0.1)
    assert model.body_inertia[body] == pytest.approx(baseline.body_inertia[body] * 1000)
    assert ET.fromstring(modified).find(".//body[@name='bin_0']/freejoint") is not None
    mat = model.geom("protective_mat_visual_bin_0").id
    assert model.geom_contype[mat] == model.geom_conaffinity[mat] == 0
    displacements = []
    for current in (baseline, model):
        data = mujoco.MjData(current)
        for _ in range(500):
            mujoco.mj_step(current, data)
        start = data.qpos[:2].copy()
        for _ in range(100):
            data.xfrc_applied[body, 0] = 50
            mujoco.mj_step(current, data)
        displacements.append(np.linalg.norm(data.qpos[:2] - start))
        data.qpos[:2] = [2, -1]
        data.qvel[:] = 0
        mujoco.mj_forward(current, data)
        assert data.xpos[body, :2] == pytest.approx([2, -1])
    assert displacements[0] > 0.1
    assert displacements[1] < 0.001
