"""Tests for the count-parameterized dynamic3D task family envs."""

import json

import pytest

from kinder.envs.dynamic3d import envs as dynamic3d_envs
from kinder.envs.dynamic3d.task_families import (
    _TASKS_DIR,
    TASK_FAMILY_ENVS,
    ConstrainedCupboard3DEnv,
    Shelf3DEnv,
    SortClutteredBlocks3DEnv,
    SweepSimple3DEnv,
    TaskFamilyEnvMixin,
)

# One reset per count is enough to show the count is a constructor argument, so
# these are the two cheapest counts per family; the largest variants (100 scooped
# cubes) only make the suite slower.
_RESET_CASES = [
    pytest.param(ConstrainedCupboard3DEnv, "cuboid_", [1, 2], id="cupboard"),
    pytest.param(Shelf3DEnv, "cube", [1, 2], id="shelf"),
    pytest.param(SweepSimple3DEnv, "cube_", [1, 5], id="sweepsimple"),
    pytest.param(SortClutteredBlocks3DEnv, "cube", [4, 20], id="sortclutteredblocks"),
]

# The robot key a task JSON names, mapped to the env class implementing it. Kept
# here rather than shared with the gym registration in :mod:`kinder`, which holds
# its own copy: this is the expectation the test checks task files against, so it
# is worth stating independently.
_ROBOT_ENV_CLASSES = {
    "tidybot": "TidyBot3D",
    "fr3": "Franka3D",
    "rby1a": "RBY1A3D",
}


def test_every_family_registers_its_declared_counts() -> None:
    """Each declared count resolves to a task file that exists."""
    for env_cls in TASK_FAMILY_ENVS:
        assert len(env_cls.supported_counts) > 1, env_cls
        for count in sorted(env_cls.supported_counts):
            assert env_cls.task_path(count).exists(), (env_cls, count)


def test_every_family_subclasses_the_env_its_tasks_declare() -> None:
    """A family's base class matches the robot named in its task JSONs.

    The robot decides the env class, so a family whose tasks switch robots would
    otherwise be built on the wrong base and fail only at reset.
    """
    for env_cls in TASK_FAMILY_ENVS:
        robots = set()
        for count in sorted(env_cls.supported_counts):
            with open(env_cls.task_path(count), encoding="utf-8") as task_file:
                robots.add(next(iter(json.load(task_file)["robots"])))
        assert len(robots) == 1, f"{env_cls.family} mixes robots: {sorted(robots)}"
        expected = getattr(dynamic3d_envs, f"{_ROBOT_ENV_CLASSES[robots.pop()]}Env")
        assert issubclass(env_cls, expected), (env_cls, expected)


def test_default_instruction_spans_every_count() -> None:
    """A family's default instruction is registered for all of its counts.

    Otherwise the goal would silently change with the count, which is the one thing
    these classes exist to hold fixed.
    """
    for env_cls in TASK_FAMILY_ENVS:
        if not env_cls.default_instruction:
            continue
        for count in sorted(env_cls.supported_counts):
            assert env_cls.default_instruction in env_cls.available_instructions(count)


def test_available_instructions_lists_alternatives() -> None:
    """Instruction discovery finds the extra goals a count registers."""
    # 50 cubes is the only SweepSimple3D count with right-side instructions.
    assert len(SweepSimple3DEnv.available_instructions(50)) > 1
    assert SweepSimple3DEnv.available_instructions(1) == [
        SweepSimple3DEnv.default_instruction
    ]
    # Count-only families report no instructions at all.
    assert not ConstrainedCupboard3DEnv.available_instructions(1)


def test_rejects_unregistered_count_and_explicit_task_path() -> None:
    """Bad arguments fail at construction rather than as a missing-file error."""
    # Shelf3D registers 1, 2 and 8; 3 lies inside that range but has no task.
    with pytest.raises(ValueError, match="registers counts"):
        Shelf3DEnv(num_objects=3)

    with pytest.raises(ValueError, match="builds task_config_path"):
        Shelf3DEnv(num_objects=1, task_config_path="ignored.json")


def test_rejects_unregistered_instruction() -> None:
    """An unknown instruction reports the ones the count does register."""
    with pytest.raises(ValueError, match="has no task"):
        SweepSimple3DEnv(num_objects=1, instruction="sweep_the_blocks_into_orbit")


@pytest.mark.parametrize("env_cls,prefix,counts", _RESET_CASES)
def test_family_resets_at_each_count(
    env_cls: type[TaskFamilyEnvMixin], prefix: str, counts: list[int]
) -> None:
    """num_objects selects the task, and the instance holds that many objects."""
    for count in counts:
        env = env_cls(num_objects=count, scene_bg=False)
        inner = getattr(env, "_object_centric_env")
        try:
            state, _ = inner.reset(seed=0)
            names = state.get_object_names()
            assert sum(name.startswith(prefix) for name in names) == count
        finally:
            env.close()
            inner.close()


def test_every_task_declares_a_goal() -> None:
    """Every shipped task JSON has a non-empty goal_state.

    The goal check treats a missing goal as never satisfied, so a task without one runs
    to its step limit on every episode and can never be solved.
    """
    task_files = sorted(_TASKS_DIR.rglob("*.json"))
    assert task_files
    for path in task_files:
        with open(path, encoding="utf-8") as task_file:
            assert json.load(task_file).get("goal_state"), path
