"""Native dynamics, initialization, goal, and rendering contracts for pouring."""

# pylint: disable=protected-access,no-member

import os
from pathlib import Path

import mujoco
import numpy as np

import kinder
import kinder.envs.dynamic3d.pitcher_pour3d as module
from kinder.envs.dynamic3d.pitcher_pour3d import PitcherPour3DEnv


def test_native_pitcher_task_initializes_five_dynamic_cubes():
    """Ensure reset uses native motors, free objects, and contact dynamics."""
    env = PitcherPour3DEnv(scene_bg=False, render_mode="rgb_array")
    try:
        env.reset(seed=0)
        scene = env._object_centric_env
        model = scene._robot_env.sim.model.mj_model
        assert scene.cube_locations() == ["pitcher"] * 5
        assert scene.cubes_settled()
        assert not scene._check_goals()
        for name in ["pitcher", "tray", *scene.cube_names]:
            body = model.body(name).id
            assert model.body_mocapid[body] == -1
            assert model.body_jntnum[body] == 1
        assert model.nu == 11
        assert sum(model.geom_contype) > 10
        assert not any(
            "weld" in (model.equality(i).name or "") for i in range(model.neq)
        )
        env.step(np.zeros(11, dtype=np.float32))
        assert scene.cube_locations() == ["pitcher"] * 5
    finally:
        env.close()


def test_native_registration_and_render():
    """Construct the task through the public factory and use its renderer."""
    backend = {key: os.environ.get(key) for key in ["MUJOCO_GL", "PYOPENGL_PLATFORM"]}
    kinder.register_all_environments()
    for key, value in backend.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value
    assert "kinder/PitcherPour3D-o5-v0" in kinder.get_all_env_ids()
    env = kinder.make(
        "kinder/PitcherPour3D-o5-v0", scene_bg=False, render_mode="rgb_array"
    )
    try:
        assert isinstance(env.unwrapped, PitcherPour3DEnv)
        env.reset(seed=0)
        frame = env.render()
        assert frame.shape == (480, 640, 3)
        assert frame.dtype == np.uint8
    finally:
        env.close()


def test_accupour_scale_mass_and_round_containment():
    """Use product-scale free geometry and reject corners outside the round cup."""
    env = PitcherPour3DEnv(scene_bg=False, render_mode="rgb_array")
    try:
        env.reset(seed=0)
        scene = env._object_centric_env
        model = scene._robot_env.sim.model.mj_model
        pitcher = scene.get_object("pitcher")
        assert np.allclose(pitcher.get_bounding_box_dimensions(), [0.15, 0.112, 0.178])
        assert np.isclose(model.body_mass[model.body("pitcher").id], 0.12)
        assert np.allclose(
            model.geom_size[model.body_geomadr[model.body("cube_0").id]], 0.01
        )
        assert scene.cube_locations() == ["pitcher"] * 5
        origin = pitcher.get_position().astype(float)
        cube = scene.get_object("cube_0")
        cube.set_pose(origin + [0, 0, 0.013], np.array([1.0, 0.0, 0.0, 0.0]))
        mujoco.mj_forward(
            model, scene._robot_env.sim.data.mj_data
        )  # pylint: disable=no-member
        assert scene.cube_in_vessel("cube_0", "pitcher")
        # This is inside a 112 mm bounding square but outside the circular cavity.
        cube.set_pose(origin + [0.038, 0.038, 0.013], np.array([1.0, 0.0, 0.0, 0.0]))
        mujoco.mj_forward(
            model, scene._robot_env.sim.data.mj_data
        )  # pylint: disable=no-member
        assert not scene.cube_in_vessel("cube_0", "pitcher")
    finally:
        env.close()


def test_explicit_human_respawn_preserves_vessels_and_robot():
    """A privileged human reset changes cube state only, with no tray animation."""
    env = PitcherPour3DEnv(scene_bg=False, render_mode="rgb_array")
    try:
        env.reset(seed=0)
        scene = env._object_centric_env
        model, data = scene._model_and_data()
        cube = scene.get_object("cube_0")
        cube.set_pose([0.65, 0.2, 0.03], [1.0, 0.0, 0.0, 0.0])
        cube.set_velocity([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
        mujoco.mj_forward(model, data)
        before = data.qpos.copy()
        preserved = np.ones(model.nq, dtype=bool)
        for name in scene.cube_names:
            joint = model.body_jntadr[model.body(name).id]
            address = model.jnt_qposadr[joint]
            preserved[address : address + 7] = False
        scene.respawn_payload_in_pitcher()
        np.testing.assert_array_equal(data.qpos[preserved], before[preserved])
        assert scene.cube_locations() == ["pitcher"] * 5
        assert scene.cubes_settled()
    finally:
        env.close()


def test_requested_count_requires_exact_split_and_settling():
    """Goal fixtures are privileged setup, never a robot pour implementation."""
    env = PitcherPour3DEnv(requested_count=3, scene_bg=False)
    try:
        env.reset(seed=0)
        scene = env._object_centric_env
        model, data = scene._model_and_data()
        assert scene.requested_count == 3
        assert not scene._check_goals()
        for i in range(4):
            cube = scene.get_object(f"cube_{i}")
            cube.set_pose([0.56 + i * 0.04, 0.2, 0.02], [1, 0, 0, 0])
            cube.set_velocity([0, 0, 0], [0, 0, 0])
            mujoco.mj_forward(model, data)
            assert scene._check_goals() == (i == 2)
            assert scene.physical_snapshot()["requested_count"] == 3
            if i == 2:
                cube.set_velocity([0.1, 0, 0], [0, 0, 0])
                assert not scene._check_goals()
                cube.set_velocity([0, 0, 0], [0, 0, 0])
                retained = scene.get_object("cube_4")
                pose = retained.get_position().copy()
                retained.set_pose([0.9, 0.5, 0.01], [1, 0, 0, 0])
                mujoco.mj_forward(model, data)
                assert not scene._check_goals()
                retained.set_pose(pose, [1, 0, 0, 0])
                mujoco.mj_forward(model, data)
                assert scene._check_goals()
        scene.respawn_payload_in_pitcher()
        assert not scene._check_goals()
    finally:
        env.close()


def test_requested_count_rejects_invalid_targets():
    """Do not silently round requests or accept impossible counts."""
    for count in [0, 6, 2.5, True]:
        try:
            PitcherPour3DEnv(requested_count=count, scene_bg=False)
        except ValueError:
            continue
        raise AssertionError(f"Accepted invalid requested_count={count}")


def test_winco_material_profiles_set_pair_contacts_and_density():
    """Read the effective compiled pairs, not just per-geom friction inputs."""
    task = (
        Path(module.__file__).parent / "tasks/PitcherPour3D/PitcherPour3D-o5-winco.json"
    )
    for material, mass, surface_mu, self_mu in [
        ("wood", 0.0048, 0.3, 0.4),
        ("foam", 0.00064, 0.6, 0.8),
    ]:
        env = PitcherPour3DEnv(
            task_config_path=str(task), cube_material=material, scene_bg=False
        )
        try:
            env.reset(seed=0)
            scene = env._object_centric_env
            model, data = scene._model_and_data()
            assert np.allclose(
                scene.get_object("pitcher").get_bounding_box_dimensions(),
                [0.14, 0.114, 0.14],
            )
            assert np.isclose(model.body_mass[model.body("pitcher").id], 0.17)
            assert np.isclose(model.body_mass[model.body("cube_0").id], mass)
            assert np.isclose(scene.get_object("cube_0").mass, mass)
            pair = model.pair("cube_0_material_surface_cube_1_material_surface").id
            assert np.allclose(model.pair_friction[pair][:2], self_mu)
            assert model.pair_dim[pair] == 3
            cube_bodies = {model.body(name).id for name in scene.cube_names}
            contacts = [
                c
                for c in data.contact
                if model.geom_bodyid[c.geom1] in cube_bodies
                or model.geom_bodyid[c.geom2] in cube_bodies
            ]
            assert contacts
            assert all(np.isclose(c.friction[0], surface_mu) for c in contacts)
            assert scene.cube_locations() == ["pitcher"] * 5
            assert scene.cubes_settled()
        finally:
            env.close()
