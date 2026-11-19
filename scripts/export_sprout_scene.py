"""Export a Sprout task as MuJoCo XML files for other simulators.

A simulator with its own Sprout model (for example mjlab, where the Sprout policies
train) can build the scene of a Kinder task from the output folder:

    fixtures.xml          all fixtures in one body, with their joints (drawers)
    objects/<name>.xml    one file for each movable object, with its free joint,
                          at the pose of the first start state
    task.json             start states, goal regions, and fixture shapes

The XML files keep the meshes and the contact settings of the Kinder scene. They have
no robot, no floor, and no lights.

``task.json`` has ``--num-starts`` start states for the same fixture layout: the start
pose of the robot and of each object, sampled from the regions of the task and
settled, as after a reset of the Kinder environment.

Example:
    python scripts/export_sprout_scene.py kinder/SproutShelf3D-o1-v0 out/SproutShelf3D-o1
"""

# pylint cannot see the members of the mujoco module, and the script reads the
# internals of the environment.
# pylint: disable=no-member,protected-access

import argparse
import copy
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

import kinder

_SHAPES = {
    mujoco.mjtGeom.mjGEOM_BOX: "box",
    mujoco.mjtGeom.mjGEOM_CYLINDER: "cylinder",
    mujoco.mjtGeom.mjGEOM_SPHERE: "sphere",
    mujoco.mjtGeom.mjGEOM_CAPSULE: "capsule",
}
FIXTURES_BODY = "kinder_fixtures"


def _floats(values: Any) -> list[float]:
    return [float(v) for v in values]


def _quat(matrix: np.ndarray) -> np.ndarray:
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, np.asarray(matrix, dtype=np.float64).reshape(9))
    return quat


def _relative_pose(
    frame_pos: np.ndarray, frame_mat: np.ndarray, pos: np.ndarray, mat: np.ndarray
) -> tuple[list[float], list[float]]:
    """Return the pose (position, quaternion) of a frame in a parent frame."""
    frame_mat, mat = frame_mat.reshape(3, 3), mat.reshape(3, 3)
    return _floats(frame_mat.T @ (pos - frame_pos)), _floats(_quat(frame_mat.T @ mat))


def _write_xml(
    path: Path, model_name: str, assets: ET.Element, body: ET.Element
) -> None:
    """Write one body with the assets that it uses as a MuJoCo XML file."""
    used = {
        element.get(key)
        for element in body.iter()
        for key in ("mesh", "material", "hfield")
        if element.get(key)
    }
    materials = [a for a in assets if a.tag == "material" and a.get("name") in used]
    used |= {m.get("texture") for m in materials if m.get("texture")}
    root = ET.Element("mujoco", model=model_name)
    # The Kinder scene uses radians: the robot XML sets this for the full model.
    ET.SubElement(root, "compiler", angle="radian")
    asset_section = ET.SubElement(root, "asset")
    for asset in assets:
        if asset.get("name") in used:
            file_name = asset.get("file")
            assert file_name is None or Path(file_name).is_absolute(), file_name
            asset_section.append(copy.deepcopy(asset))
    ET.SubElement(root, "worldbody").append(body)
    ET.indent(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(ET.tostring(root, encoding="unicode") + "\n", encoding="utf-8")


def export_scene(env_id: str, output_dir: Path, seed: int, num_starts: int) -> str:
    """Reset the environment and write its scene.

    Returns a summary line.
    """
    kinder.register_all_environments()
    env = kinder.make(env_id)
    object_centric_env = env.unwrapped._object_centric_env  # type: ignore[attr-defined]

    # Keep the scene XML (without the robot) that the reset creates.
    captured: dict[str, str] = {}
    create_scene_xml = object_centric_env._create_scene_xml

    def capture_scene_xml() -> str:
        captured["xml"] = create_scene_xml()
        return captured["xml"]

    object_centric_env._create_scene_xml = capture_scene_xml
    env.reset(seed=seed)
    object_centric_env._create_scene_xml = create_scene_xml

    robot_env = object_centric_env._robot_env
    model, data = robot_env.sim.model.mj_model, robot_env.sim.data.mj_data
    objects_dict = object_centric_env._objects_dict
    robot_prefix = f"{object_centric_env.robot_name}_"

    # Contact and constraint buffer sizes for a parallel simulator, from the settled
    # scene with the same rule as Sprout3DBatchEnv.
    buffers = {
        "nconmax": max(256, 2 * int(data.ncon)),
        "njmax": max(800, 2 * int(data.nefc)),
    }

    scene_root = ET.fromstring(captured["xml"])
    assets = scene_root.find("asset")
    worldbody = scene_root.find("worldbody")
    assert assets is not None and worldbody is not None

    # The XML files: one for each object, and one for all fixtures.
    fixtures_body = ET.Element("body", name=FIXTURES_BODY)
    for element in worldbody:
        if element.tag == "body" and element.get("name") in objects_dict:
            # The body has the pose of the first start state, so that the objects are
            # not inside each other in the default state of a model with all of them.
            name = str(element.get("name"))
            object_body = copy.deepcopy(element)
            object_body.attrib.pop("euler", None)
            position, quat = robot_env.get_joint_pos_quat(objects_dict[name].joint_name)
            object_body.set("pos", " ".join(str(float(v)) for v in position))
            object_body.set("quat", " ".join(str(float(v)) for v in quat))
            _write_xml(
                output_dir / "objects" / f"{name}.xml", name, assets, object_body
            )
        elif element.tag in ("body", "site") or (
            element.tag == "geom" and element.get("type") != "plane"
        ):
            fixtures_body.append(copy.deepcopy(element))
    _write_xml(output_dir / "fixtures.xml", FIXTURES_BODY, assets, fixtures_body)

    # The static fixture shapes in the world frame (for example for a planner).
    fixture_shapes = []
    fixture_joints = {}
    for geom in range(model.ngeom):
        body = model.body(model.geom_bodyid[geom])
        root_name = model.body(model.body_rootid[body.id]).name
        collides = model.geom_contype[geom] != 0 or model.geom_conaffinity[geom] != 0
        if (
            root_name.startswith(robot_prefix)
            or root_name in objects_dict
            or model.body_weldid[body.id] != 0
            or int(model.geom_type[geom]) not in _SHAPES
            or not collides
        ):
            continue
        fixture_shapes.append(
            {
                "name": model.geom(geom).name,
                "type": _SHAPES[int(model.geom_type[geom])],
                "size": _floats(model.geom_size[geom]),
                "pos": _floats(data.geom_xpos[geom]),
                "quat": _floats(_quat(data.geom_xmat[geom])),
            }
        )
    for joint in range(model.njnt):
        root_name = model.body(model.body_rootid[model.jnt_bodyid[joint]]).name
        if model.jnt_type[joint] != mujoco.mjtJoint.mjJNT_FREE and not (
            root_name.startswith(robot_prefix) or root_name in objects_dict
        ):
            fixture_joints[model.joint(joint).name] = {
                "start": float(data.qpos[model.jnt_qposadr[joint]]),
                "range": _floats(model.jnt_range[joint]),
            }

    # The goal regions, each in the frame that it moves with.
    goals = []
    for predicate in object_centric_env.task_config.get("goal_state", []):
        if not isinstance(predicate, list) or predicate[0] not in ("on", "in"):
            goals.append({"predicate": predicate})
            continue
        target = object_centric_env.task_config["regions"][predicate[2]]["target"]
        if target == "ground":
            entity = object_centric_env._ground_fixture
        else:
            entity = (
                object_centric_env._fixtures_dict.get(target) or objects_dict[target]
            )
        boxes = []
        for region in entity.region_objects[predicate[2]]:
            site = model.site(region.site_element.get("name"))
            body = model.body(site.bodyid[0])
            root_name = model.body(model.body_rootid[body.id]).name
            if root_name in objects_dict:
                frame = {"kind": "object", "name": root_name}
                frame_id = model.body(root_name).id
            elif model.body_weldid[body.id] != 0:
                frame = {"kind": "fixture_body", "name": body.name}
                frame_id = body.id
            else:
                frame = {"kind": "world"}
                frame_id = 0
            pos, quat = _relative_pose(
                data.xpos[frame_id],
                data.xmat[frame_id],
                data.site_xpos[site.id],
                data.site_xmat[site.id],
            )
            half_size = _floats(region.bbox_at_origin[3:])
            boxes.append(
                {"frame": frame, "pos": pos, "quat": quat, "half_size": half_size}
            )
        goals.append({"object": predicate[1], "target": target, "boxes": boxes})

    # The start states: the state after the reset, and more samples for the same
    # fixtures, each settled as in a reset.
    robot_starts: list[list[float]] = []
    object_starts: dict[str, list[list[float]]] = {name: [] for name in objects_dict}
    for index in range(num_starts):
        if index > 0:
            object_centric_env.sample_initial_qpos()
            object_centric_env._settle_scene()
        base = robot_env.qpos["base"]
        yaw = 2.0 * np.arctan2(base[6], base[3])
        robot_starts.append([float(base[0]), float(base[1]), float(yaw)])
        for name, mujoco_object in objects_dict.items():
            position, quat = robot_env.get_joint_pos_quat(mujoco_object.joint_name)
            object_starts[name].append(_floats(position) + _floats(quat))

    task: dict[str, Any] = {
        "env_id": env_id,
        "seed": seed,
        "robot": {"name": object_centric_env.robot_name, "starts": robot_starts},
        "fixtures": {
            "file": "fixtures.xml",
            "body": FIXTURES_BODY,
            "joints": fixture_joints,
            "shapes": fixture_shapes,
        },
        "objects": [
            {
                "name": name,
                "file": f"objects/{name}.xml",
                "mass": float(model.body(name).subtreemass[0]),
                "half_size": [
                    float(v) / 2 for v in mujoco_object.get_bounding_box_dimensions()
                ],
                "starts": object_starts[name],
            }
            for name, mujoco_object in objects_dict.items()
        ],
        "goals": goals,
        "buffers": buffers,
    }
    (output_dir / "task.json").write_text(
        json.dumps(task, indent=1) + "\n", encoding="utf-8"
    )
    env.close()
    return (
        f"{env_id}: {len(task['objects'])} objects, {len(fixture_shapes)} fixture "
        f"shapes, {len(fixture_joints)} fixture joints, {len(goals)} goals, "
        f"{num_starts} start states -> {output_dir}"
    )


def main() -> None:
    """Write the scene of one Sprout environment."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("env_id")
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-starts", type=int, default=64)
    args = parser.parse_args()
    print(export_scene(args.env_id, args.output_dir, args.seed, args.num_starts))


if __name__ == "__main__":
    main()
