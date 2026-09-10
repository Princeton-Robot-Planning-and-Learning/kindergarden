"""Focused checks for the local Tossing room preview."""

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import kinder
from kinder.envs.dynamic3d.envs import TidyBot3DEnv
from kinder.envs.dynamic3d.objects.fixtures import FixedCuboid
from kinder.envs.dynamic3d.scene_loader import SceneLoader


def test_room_geometry_is_shared_by_backgrounds() -> None:
    """The fancy scene replaces its old walls instead of adding hidden copies."""
    models = Path(kinder.__file__).parent / "envs/dynamic3d/models/stanford_tidybot"
    signatures = []
    for config in ({"type": "simple"}, {"type": "mimiclabs", "lab": 2}):
        original = ET.fromstring(SceneLoader.load_scene(config, models))
        updated = ET.fromstring(
            SceneLoader.load_scene(config, models, "tossing_room.xml")
        )
        room = updated.find("worldbody/body[@name='tossing_room']")
        assert room is not None
        assert "floor_visual_size" not in room.attrib
        signature = {}
        for geom in room.findall("geom"):
            name = geom.get("name")
            assert len(updated.findall(f".//geom[@name='{name}']")) == 1
            assert geom.get("contype") == geom.get("conaffinity") == "1"
            signature[name] = {
                key: value
                for key, value in geom.attrib.items()
                if key not in ("material", "rgba")
            }
            if config["type"] == "simple":
                assert geom.get("rgba") == "1 1 1 1"
            else:
                assert geom.get("material") == "walls_mat"
        assert len(signature) == 6
        assert signature["wall_front_visual"]["pos"] == "4.2 0 1.5"
        signatures.append(signature)
        before_floor = original.find(".//geom[@type='plane']")
        after_floor = updated.find(".//geom[@type='plane']")
        assert before_floor is not None and after_floor is not None
        assert after_floor.get("size", "").split()[:2] == ["4.5", "3.5"]
        assert (
            before_floor.get("size", "").split()[2]
            == after_floor.get("size", "").split()[2]
        )
        assert {k: v for k, v in before_floor.attrib.items() if k != "size"} == {
            k: v for k, v in after_floor.attrib.items() if k != "size"
        }
    assert signatures[0] == signatures[1]


def test_fixed_barrier_preserves_original_box_geometry() -> None:
    """Anchoring removes mobility without resizing or moving the barrier."""
    config: dict[str, Any] = {"size": [0.03, 5.0, 0.10], "mass": 10.0}
    fixture = FixedCuboid(
        "cuboid_barrier",
        config,
        [1.3, 0.0, 0.1],
        0.0,
    )
    assert fixture.xml_element.find("freejoint") is None
    assert fixture.xml_element.get("pos") == "1.3 0.0 0.1"
    assert np.allclose(fixture.primitive.get_bounding_box_dimensions(), [0.06, 10, 0.2])
    assert set(fixture.get_object_centric_state()[fixture.symbolic_object]) == {
        "x",
        "y",
        "z",
        "qw",
        "qx",
        "qy",
        "qz",
    }


@pytest.mark.parametrize("num_objects", [1, 2])
def test_state_restoration_across_resets(num_objects: int) -> None:
    """The anchored barrier must agree between independently reset models."""
    env = TidyBot3DEnv(
        num_objects=num_objects,
        task_config_path=f"tasks/Tossing3D/Tossing3D-o{num_objects}.json",
        scene_bg=False,
        allow_state_access=True,
    )
    try:
        env.reset(seed=0)
        saved = env.get_state()
        env.reset(seed=1)
        env.set_state(saved)
        np.testing.assert_array_equal(env.get_state(), saved)
        inner = env._object_centric_env  # pylint: disable=protected-access
        sim = inner._robot_env.sim  # pylint: disable=protected-access
        assert sim is not None
        barrier_id = sim.model.mj_model.body("cuboid_barrier").id
        np.testing.assert_allclose(
            sim.data.mj_data.xpos[barrier_id], [1.3, 0.0, 0.1], atol=1e-7
        )
    finally:
        env.close()
