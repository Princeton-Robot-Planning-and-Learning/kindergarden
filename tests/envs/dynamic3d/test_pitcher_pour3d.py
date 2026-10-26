"""Native dynamics, initialization, goal, and rendering contracts for pouring."""

# pylint: disable=protected-access,no-member

import os

import mujoco
import numpy as np

import kinder
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
