"""Create the Sprout task files from the TidyBot task files.

Each ``Sprout<Family>`` folder in ``src/kinder/envs/dynamic3d/tasks`` is a copy of the
TidyBot family, with the layout changes that the Sprout humanoid needs. The Sprout is
about 1.1 m tall, and its hand reaches down to 0.48 m when it stands. So objects start
on tables and not on the floor, and the work surfaces are higher.

This script is the source of the Sprout task files. To change a layout, change the
function of its family here, and run the script again:

python scripts/make_sprout_tasks.py

Use ``--check`` to compare the task files with the output of this script, with no change
of the files.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable, Iterator

TASKS_DIR = (
    Path(__file__).parents[1] / "src" / "kinder" / "envs" / "dynamic3d" / "tasks"
)

Task = dict[str, Any]

# The height of the tables where objects start. The Sprout reaches it when it stands.
TABLE_HEIGHT = 0.65
# A kitchen island with its top at 0.68 m (the heights of its two shelf levels).
HIGH_ISLAND = {
    "fixtures": {"cupboard": {"kitchen_island": {"shelf_heights": [0.25, 0.37]}}}
}


def on_ground(x: float, y: float, yaw: float, z: float = 0.0) -> Task:
    """A region that puts a fixture at one pose.

    The ranges have 6 numbers. With 4 numbers, the fixture gets a random height.
    """
    return {
        "target": "ground",
        "ranges": [[x, y, z, x, y, z]],
        "yaw_ranges": [[yaw, yaw]],
    }


def table(length: float, width: float, height: float = TABLE_HEIGHT) -> Task:
    """A rectangular table fixture."""
    return {
        "shape": "rectangle",
        "height": height,
        "thickness": 0.04,
        "length": length,
        "width": width,
    }


def set_camera(task: Task, position: list[float], lookat: list[float]) -> None:
    """Point the render camera at the task area."""
    task["cameras"]["task_view"].update(position=position, lookat=lookat)


def add_fixture(
    task: Task, kind: str, name: str, config: Task, region_name: str, region: Task
) -> None:
    """Add a fixture and the region that gives its pose."""
    task.setdefault("fixtures", {}).setdefault(kind, {})[name] = config
    task["regions"][region_name] = region
    task["initial_state"].insert(0, ["on", name, region_name])


def rename_region(task: Task, old: str, new: str) -> None:
    """Rename a region in the regions and in the predicates."""
    task["regions"][new] = task["regions"].pop(old)
    goals = [goal for goal in task["goal_state"] if isinstance(goal, list)]
    for predicate in task["initial_state"] + goals:
        if len(predicate) > 2 and predicate[2] == old:
            predicate[2] = new


def pick_place(task: Task) -> None:
    """The cube and its goal are on a low table in front of the robot."""
    task["description"] = (
        "A 3D tabletop task where the Sprout humanoid stands at a low table and must "
        "pick up objects and place them at target locations on the table."
    )
    task["variant_specific_description"] = (
        "Pick up one cube from the table and place it in the goal region on the table."
    )
    task["fixtures"] = {"table": {"table_1": table(0.6, 0.8)}}
    task["regions"] = {
        "ground_table_init_region": on_ground(0.55, 0.0, 0),
        "robot_init_region": {
            "target": "ground",
            "ranges": [[-0.02, -0.02, 0.02, 0.02]],
            "yaw_ranges": [[0, 0]],
        },
        "cube_init_region": {
            **task["regions"]["cube_init_region"],
            "target": "table_1",
            "ranges": [[-0.25, -0.2, -0.15, -0.05]],
        },
        "goal_region": {
            **task["regions"]["goal_region"],
            "target": "table_1",
            "ranges": [[-0.25, 0.05, -0.1, 0.2]],
            "rgba": [0.0, 0.0, 1.0, 0.3],
        },
    }
    task["objects"]["cube"]["cube1"].update(size=0.025, mass=0.05)
    task["initial_state"] = [
        ["on", "table_1", "ground_table_init_region"],
        ["on", "robot", "robot_init_region"],
        ["on", "cube1", "cube_init_region"],
    ]
    set_camera(task, [1.9, -1.2, 1.7], [0.5, 0.1, 0.75])


def dynamo(task: Task) -> None:
    """The chair has a mass that the Sprout can push."""
    task["description"] = task["description"].replace(
        "must reach a goal", "must walk to a goal"
    )
    task["variant_specific_description"] = (
        "Walk to a goal region on the floor. One chair (5 kg) stands in the goal "
        "region."
    )
    for chair in task["objects"]["robocasafrlapartmentchair01"].values():
        chair["mass"] = 5.0


def shelf(task: Task) -> None:
    """The cubes start in two rows on a low table, and the shelves are higher."""
    task["description"] = task["description"].replace(
        "from the ground", "from a low table"
    )
    task["variant_specific_description"] = task["variant_specific_description"].replace(
        "from the floor", "from the table"
    )
    task["fixtures"]["cupboard"]["cupboard_1"]["shelf_heights"] = [0.33, 0.33, 0.254]
    task["fixtures"]["table"] = {"table_1": table(1.0, 0.5)}
    regions = task["regions"]
    regions["ground_cupboard_init_region"]["ranges"] = [[1.5, 0.0, 0.0, 1.5, 0.0, 0.0]]
    regions["ground_table_init_region"] = on_ground(0.3, 0.85, 0)

    # Each cube has its own small region: 4 cubes in a row, 0.25 m apart.
    cube_predicates = []
    for predicate in list(task["initial_state"]):
        if predicate[1].startswith("cube"):
            task["initial_state"].remove(predicate)
            del regions[predicate[2]]
            index = int(predicate[1].removeprefix("cube")) - 1
            x = -0.375 + 0.25 * (index % 4)
            y = -0.17 + 0.12 * (index // 4)
            region_name = f"table_1_{predicate[1]}_init_region"
            regions[region_name] = {
                "target": "table_1",
                "ranges": [
                    [
                        round(x - 0.04, 3),
                        round(y - 0.02, 3),
                        round(x + 0.04, 3),
                        round(y + 0.02, 3),
                    ]
                ],
                "yaw_ranges": [[0, 0]],
                "rgba": [1.0, 1.0, 0.0, 0.0],
            }
            cube_predicates.append(["on", predicate[1], region_name])
    task["initial_state"].insert(1, ["on", "table_1", "ground_table_init_region"])
    task["initial_state"].extend(cube_predicates)

    # Without a start region, the robot can start inside the table.
    if "robot_0_task_init_region" not in regions:
        regions["robot_0_task_init_region"] = {
            "target": "ground",
            "ranges": [[-0.1, -0.1, 0.1, 0.1]],
            "yaw_ranges": [[0, 0]],
        }
        task["initial_state"].append(["on", "robot", "robot_0_task_init_region"])
    set_camera(task, [-1.3, -1.4, 1.9], [0.8, 0.3, 0.5])


def rearrange(task: Task) -> None:
    """A higher island, with the objects near the edge where the robot stands."""
    task["scene_overrides"] = HIGH_ISLAND
    regions = task["regions"]
    regions["robot_task_init_region"]["ranges"] = [[-0.12, -0.03, -0.08, 0.03]]
    for name, region in regions.items():
        if name.startswith("kitchen_island_shelf_2_"):
            # The start height is 6 to 8 cm above the island. With 4 numbers, an
            # object starts up to 0.37 m above it (the height of the shelf level),
            # and the can then falls on its side in about 9% of the resets.
            region["ranges"] = [
                [ranges[0], -0.3, 0.06, ranges[2], -0.16, 0.08]
                for ranges in region["ranges"]
            ]
    # Some variants use the "in" predicate for these regions. "in" needs a region that
    # is taller than the object, so they get "on", as the other variants.
    for predicate in task["initial_state"]:
        if predicate[2].startswith("kitchen_island_shelf_2_"):
            predicate[0] = "on"
    set_camera(task, [-1.0, -0.6, 2.4], [0.3, 0.0, 0.7])


def constrained_cupboard(task: Task) -> None:
    """The rods start on a table, and the cupboards stand on a low stand."""
    regions = task["regions"]
    ys = [ranges[1] for ranges in regions["cupboard_init_region"]["ranges"]]
    for ranges in regions["cupboard_init_region"]["ranges"]:
        ranges[0] = ranges[3] = 1.5
        ranges[2] = ranges[5] = 0.5
    add_fixture(
        task,
        "table",
        "stand_1",
        table(0.4, max(ys) - min(ys) + 0.3, 0.5),
        "ground_stand_init_region",
        on_ground(1.5, (max(ys) + min(ys)) / 2, 0),
    )
    add_fixture(
        task,
        "table",
        "table_1",
        table(1.2, 0.7),
        "ground_table_init_region",
        on_ground(0.6, 0.85, 0),
    )
    rename_region(task, "ground_object_init_region", "table_1_object_init_region")
    regions["table_1_object_init_region"] = {
        "target": "table_1",
        "ranges": [[-0.5, -0.25, 0.5, 0.25]],
        "yaw_ranges": [[60, 120]],
    }
    task["cameras"]["task_view"] = {
        "position": [-1.2, -1.4, 2.0],
        "lookat": [0.9, 0.3, 0.6],
        "fovy": 42,
        "resolution": [640, 480],
    }


def sort_cluttered_blocks(task: Task) -> None:
    """A higher and narrower table, or two tables and a cupboard on a stand."""
    regions = task["regions"]
    if "table" in task.get("fixtures", {}):
        task["fixtures"]["table"]["table_1"].update(height=TABLE_HEIGHT, width=0.5)
        regions["robot_task_init_region"]["ranges"] = [[0.48, -0.1, 0.52, 0.1]]
        camera = {"position": [1.8, -1.4, 1.9], "lookat": [0.0, 0.0, 0.6]}
    else:
        regions["cupboard_init_region"] = on_ground(1.7, 0.0, 90, z=0.5)
        add_fixture(
            task,
            "table",
            "stand_1",
            table(0.4, 0.75, 0.5),
            "ground_stand_init_region",
            on_ground(1.7, 0.0, 0),
        )
        names = []
        for side, y in (("left", 0.9), ("right", -0.9)):
            add_fixture(
                task,
                "table",
                f"table_{side}",
                table(1.2, 0.6),
                f"ground_table_{side}_init_region",
                on_ground(0.8, y, 0),
            )
            names.append(f"table_{side}_object_init_region")
            # The rods are 0.7 m long, so they lie along the table.
            regions[names[-1]] = {
                "target": f"table_{side}",
                "ranges": [[-0.5, -0.2, 0.5, 0.2]],
                "yaw_ranges": [[80, 100]],
            }
        del regions["ground_object_init_region"]
        count = 0
        for predicate in task["initial_state"]:
            if predicate[2] == "ground_object_init_region":
                predicate[2] = names[count % 2]
                count += 1
        camera = {"position": [-1.6, -1.8, 2.3], "lookat": [1.0, 0.0, 0.5]}
    task["cameras"]["task_view"] = {**camera, "fovy": 42, "resolution": [640, 480]}


def scoop_pour(task: Task) -> None:
    """A higher island, the bins near its edge, and a scoop handle for the hand."""
    task["scene_overrides"] = HIGH_ISLAND
    regions = task["regions"]
    regions["robot_task_init_region"]["ranges"] = [[-0.12, -0.03, -0.08, 0.03]]
    regions["bin_yellow_init_region"]["ranges"] = [
        [0.2, -0.2, 0.0, 0.201, -0.199, 0.001]
    ]
    regions["bin_green_init_region"]["ranges"] = [
        [-0.201, -0.2, 0.0, -0.2, -0.199, 0.001]
    ]
    for ranges in regions["blocks_init_region"]["ranges"]:
        ranges[1], ranges[4] = ranges[1] - 0.2, ranges[4] - 0.2
    regions["scoop_init_region"]["ranges"] = [[0.42, -0.37, 0.72, -0.21]]
    regions["scoop_init_region"]["yaw_ranges"] = [[-20, 20]]
    for scoop in task["objects"]["scoop"].values():
        scoop.update(handle_length=0.08, handle_width=0.025, handle_height=0.025)
    task["cameras"]["task_view"] = {
        "position": [-1.0, -0.9, 2.4],
        "lookat": [0.3, -0.15, 0.7],
        "fovy": 42,
        "resolution": [640, 480],
    }


def sweep_into_drawer(task: Task) -> None:
    """A higher island, with the drawer handles in the standing reach."""
    task["scene_overrides"] = {
        "fixtures": {"cupboard": {"kitchen_island": {"shelf_heights": [0.33, 0.29]}}}
    }
    regions = task["regions"]
    regions["robot_kitchen_init_region"]["ranges"] = [[1.15, -0.3, 1.25, -0.1]]
    regions["blocks_init_region"]["ranges"] = [[0.0, 0.2, 0.2, 0.33]]
    task["cameras"]["task_view"] = {
        "position": [2.3, -1.3, 2.0],
        "lookat": [0.7, -0.1, 0.6],
        "fovy": 42,
        "resolution": [640, 480],
    }


def sweep_simple(task: Task) -> None:
    """The blocks stay on the floor, and the wiper has a handle like a broom."""
    for wiper in task["objects"]["wiper"].values():
        wiper["handle_height"] = max(wiper["handle_height"], 0.7)


def tossing(task: Task) -> None:
    """The block starts on a table, and the barrier and the bin are nearer."""
    add_fixture(
        task,
        "table",
        "table_1",
        table(0.5, 0.8),
        "ground_table_init_region",
        on_ground(0.5, 0.0, 0),
    )
    regions = task["regions"]
    regions["blocks_init_region"] = {
        "target": "table_1",
        "ranges": [[-0.2, -0.2, -0.1, 0.2]],
        "yaw_ranges": [[-45, 45]],
    }
    regions["barrier_init_region"]["ranges"] = [[1.1, 0.0, 0.1, 1.1, 0.0, 0.1]]
    regions["bin_init_region"]["ranges"] = [[1.5, -0.8, 1.9, 0.8]]
    task["cameras"]["task_view"] = {
        "position": [-1.5, -1.6, 2.0],
        "lookat": [1.0, 0.0, 0.4],
        "fovy": 42,
        "resolution": [640, 480],
    }


def balance_beam(task: Task) -> None:
    """The seesaw and the blocks start on a table."""
    add_fixture(
        task,
        "table",
        "table_1",
        table(0.7, 1.0),
        "ground_table_init_region",
        on_ground(0.6, 0.0, 0),
    )
    regions = task["regions"]
    regions["seesaw_init_region"] = {
        "target": "table_1",
        "ranges": [[-0.14, 0.2, -0.10, 0.3]],
        "yaw_ranges": [[90, 90]],
    }
    regions["blocks_init_region"] = {
        "target": "table_1",
        "ranges": [[-0.28, -0.4, -0.18, -0.1]],
        "yaw_ranges": [[-45, 45]],
    }
    task["cameras"]["task_view"] = {
        "position": [-0.9, -1.2, 1.8],
        "lookat": [0.5, 0.0, 0.65],
        "fovy": 42,
        "resolution": [640, 480],
    }


# The layout function of each family, and the variants that have a Sprout version
# (an empty list: all variants of the family).
FAMILIES: dict[str, tuple[Callable[[Task], None], list[str]]] = {
    "PickPlace3D": (pick_place, []),
    "Dynamo3D": (dynamo, ["o1"]),
    "Shelf3D": (shelf, []),
    "Rearrange3D": (rearrange, []),
    "ConstrainedCupboard3D": (constrained_cupboard, []),
    "SortClutteredBlocks3D": (sort_cluttered_blocks, []),
    "ScoopPour3D": (scoop_pour, []),
    "SweepIntoDrawer3D": (sweep_into_drawer, []),
    "SweepSimple3D": (sweep_simple, []),
    "Tossing3D": (tossing, []),
    "BalanceBeam3D": (balance_beam, []),
}


def make_sprout_tasks() -> Iterator[tuple[Path, Task]]:
    """Make all Sprout tasks.

    Yields the path of each task file and its content.
    """
    for family, (change_layout, variants) in FAMILIES.items():
        for source in sorted((TASKS_DIR / family).glob(f"{family}-o*.json")):
            variant = source.stem.removeprefix(f"{family}-").split("-")[0]
            if variants and variant not in variants:
                continue
            task = json.loads(source.read_text(encoding="utf-8"))
            task["description"] = task["description"].replace(
                "the robot", "the Sprout humanoid"
            )
            task["robots"] = {"sprout": {"robot": {}}}
            change_layout(task)
            name = source.name.replace(family, f"Sprout{family}")
            yield TASKS_DIR / f"Sprout{family}" / name, task


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="compare the task files with the output, with no change of the files",
    )
    args = parser.parse_args()
    different = []
    count = 0
    for path, task in make_sprout_tasks():
        count += 1
        if args.check:
            if not path.exists() or json.loads(path.read_text("utf-8")) != task:
                different.append(path.name)
        else:
            path.parent.mkdir(exist_ok=True)
            path.write_text(json.dumps(task, indent=4) + "\n", encoding="utf-8")
    if args.check:
        print(f"{count - len(different)} of {count} task files agree with the script")
        for name in different:
            print(f"  differs: {name}")
        raise SystemExit(1 if different else 0)
    print(f"Wrote {count} task files")


if __name__ == "__main__":
    _main()
