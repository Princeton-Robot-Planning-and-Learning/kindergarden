"""Native dynamics, initialization, goal, and rendering contracts for pouring."""

import numpy as np


def test_native_pitcher_task_initializes_five_dynamic_cubes():
    from kinder.envs.dynamic3d.pitcher_pour3d import PitcherPour3DEnv

    env = PitcherPour3DEnv(scene_bg=False, render_mode="rgb_array")
    try:
        env.reset(seed=0)
        scene = env._object_centric_env
        model = scene._robot_env.sim.model.mj_model
        assert scene.cube_locations() == ["pitcher"] * 5
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
    import os

    import kinder
    from kinder.envs.dynamic3d.pitcher_pour3d import PitcherPour3DEnv

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
