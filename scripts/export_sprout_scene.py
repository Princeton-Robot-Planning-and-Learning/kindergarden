"""Export the start state of a Sprout task as a plain description for other simulators.

The output is one JSON file with the fixtures as primitive shapes in the world frame,
the movable objects with their start poses, the start pose of the robot, and the goal
regions. It has no robot model. A simulator with its own Sprout model (for example
mjlab, where the Sprout policies train) can build the same scene from this file.

Example:
    python scripts/export_sprout_scene.py kinder/SproutShelf3D-o1-v0 scene.json
"""

# pylint cannot see the members of the mujoco module, and the script reads the
# internals of the environment.
# pylint: disable=no-member,protected-access

import argparse
import json
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


def _quat(matrix: np.ndarray) -> list[float]:
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, np.asarray(matrix, dtype=np.float64).reshape(9))
    return [float(v) for v in quat]


def export_scene(env_id: str, seed: int) -> dict[str, Any]:
    """Reset the environment and describe its scene."""
    kinder.register_all_environments()
    env = kinder.make(env_id)
    env.reset(seed=seed)
    object_centric_env = env.unwrapped._object_centric_env  # type: ignore[attr-defined]
    robot_env = object_centric_env._robot_env
    model, data = robot_env.sim.model.mj_model, robot_env.sim.data.mj_data
    objects_dict = object_centric_env._objects_dict
    robot_prefix = f"{object_centric_env.robot_name}_"

    def shape(geom: int, position: np.ndarray, matrix: np.ndarray) -> dict[str, Any]:
        return {
            "name": model.geom(geom).name,
            "type": _SHAPES[int(model.geom_type[geom])],
            "size": [float(v) for v in model.geom_size[geom]],
            "pos": [float(v) for v in position],
            "quat": _quat(matrix),
            "rgba": [float(v) for v in model.geom_rgba[geom]],
        }

    fixtures, skipped = [], []
    object_geoms: dict[str, list[dict[str, Any]]] = {name: [] for name in objects_dict}
    for geom in range(model.ngeom):
        body = model.body(model.geom_bodyid[geom])
        root = model.body(model.body_rootid[body.id])
        if root.name.startswith(robot_prefix):
            continue
        if model.geom_contype[geom] == 0 and model.geom_conaffinity[geom] == 0:
            continue  # visual only
        if int(model.geom_type[geom]) not in _SHAPES:
            if model.geom_type[geom] != mujoco.mjtGeom.mjGEOM_PLANE:
                skipped.append(model.geom(geom).name)
            continue
        if root.name in objects_dict:
            # In the frame of the object body.
            object_geoms[root.name].append(
                shape(geom, model.geom_pos[geom], np.eye(3))
                | {"quat": [float(v) for v in model.geom_quat[geom]]}
            )
        elif model.body_weldid[body.id] == 0:
            fixtures.append(shape(geom, data.geom_xpos[geom], data.geom_xmat[geom]))
        else:
            skipped.append(model.geom(geom).name)  # a moving fixture part (a drawer)

    objects = []
    for name, mujoco_object in objects_dict.items():
        position, quat = robot_env.get_joint_pos_quat(mujoco_object.joint_name)
        objects.append(
            {
                "name": name,
                "pos": [float(v) for v in position],
                "quat": [float(v) for v in quat],
                "mass": float(model.body(name).subtreemass[0]),
                "geoms": object_geoms[name],
            }
        )

    goals = []
    for predicate in object_centric_env.task_config.get("goal_state", []):
        if not isinstance(predicate, list) or predicate[0] not in ("on", "in"):
            goals.append({"predicate": predicate})
            continue
        target = object_centric_env.task_config["regions"][predicate[2]]["target"]
        if target == "ground":
            entity = object_centric_env._ground_fixture
        else:
            fixtures_dict = object_centric_env._fixtures_dict
            entity = fixtures_dict.get(target) or objects_dict[target]
        boxes = []
        for region in entity.region_objects[predicate[2]]:
            site = region.site_element.get("name")
            boxes.append(
                {
                    "pos": [float(v) for v in robot_env.sim.data.get_site_xpos(site)],
                    "quat": _quat(robot_env.sim.data.get_site_xmat(site)),
                    "half_size": [float(v) for v in region.bbox_at_origin[3:]],
                }
            )
        goals.append({"object": predicate[1], "target": target, "boxes": boxes})

    base = robot_env.qpos["base"]
    scene = {
        "env_id": env_id,
        "seed": seed,
        "robot": {
            "pos": [float(v) for v in base[:3]],
            "yaw": float(2.0 * np.arctan2(base[6], base[3])),
        },
        "fixtures": fixtures,
        "objects": objects,
        "goals": goals,
        "skipped_geoms": skipped,
    }
    env.close()
    return scene


def main() -> None:
    """Write the scene description of one Sprout environment."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("env_id")
    parser.add_argument("output", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    scene = export_scene(args.env_id, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(scene, indent=1) + "\n", encoding="utf-8")
    print(
        f"{args.env_id}: {len(scene['fixtures'])} fixture shapes, "
        f"{len(scene['objects'])} objects, {len(scene['goals'])} goals, "
        f"{len(scene['skipped_geoms'])} geoms not exported -> {args.output}"
    )


if __name__ == "__main__":
    main()
