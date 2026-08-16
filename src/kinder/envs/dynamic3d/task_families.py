"""Count-parameterized access to the registered dynamic3D task families.

Every dynamic3D task ships as its own JSON under ``tasks/<Family>/``, named
``<Family>-o<count>[-<instruction>].json``, and the robot envs in
:mod:`kinder.envs.dynamic3d.envs` take one such file through ``task_config_path``.
An object count is therefore baked into the file, unlike every other kinder
family (``Obstruction2D``, ``DynScoopPour2D``, ``Table3D``, ...), which takes its
count as a constructor argument.

The classes here restore that constructor form for dynamic3D: each one names a
family and the counts it registers, and turns ``num_objects`` back into a normal
argument. That lets a caller sweep an object count over one family without
knowing how the task files are laid out -- for generalization experiments, for
instance, where a single policy must run across instance sizes.

Families whose variants also differ by instruction ("sweep the blocks to the left
side of the kitchen island") carry a ``default_instruction`` that holds the goal
fixed as the count varies. Pass ``instruction=`` to choose another one;
:meth:`TaskFamilyEnvMixin.available_instructions` lists what a count registers.

Only families registering more than one count appear here. BalanceBeam3D and
SweepIntoDrawer3D ship a single variant each, and Rearrange3D's two variants swap
in a different object rather than adding more of one, so for those a plain
``gymnasium.make("kinder/<Family>-<variant>-v0")`` is the whole story.
"""

from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

from kinder.envs.dynamic3d.envs import TidyBot3DEnv

_TASKS_DIR = Path(__file__).parent / "tasks"

if TYPE_CHECKING:
    # The mixin is only ever combined with a robot env class, and it forwards to
    # that class's __init__. Declaring the base for type checkers alone gives
    # callers the full env API (close, reset, ...) on a family class without
    # putting ConstantObjectKinDEREnv into the runtime MRO twice.
    from kinder.core import ConstantObjectKinDEREnv

    _MixinBase = ConstantObjectKinDEREnv
else:
    _MixinBase = object


class TaskFamilyEnvMixin(_MixinBase):
    """Select one task JSON of a dynamic3D family by its object count.

    Mixed in ahead of the robot env class that the family's task JSONs declare, so
    ``ConstrainedCupboard3DEnv(num_objects=4)`` builds the same environment as
    ``TidyBot3DEnv(task_config_path=".../ConstrainedCupboard3D-o4.json")``.
    """

    # Directory under tasks/, and the leading part of each task filename.
    family: ClassVar[str]
    # Counts with a registered task JSON. Rarely contiguous, and not always equal
    # to the number of count-defining objects -- the ``o`` label names the variant.
    supported_counts: ClassVar[frozenset[int]]
    # Instruction used when the caller does not choose one. Empty for families
    # whose variants are distinguished by count alone.
    default_instruction: ClassVar[str] = ""

    @classmethod
    def task_path(cls, num_objects: int, instruction: str | None = None) -> Path:
        """Return the task JSON for *num_objects* under *instruction*."""
        chosen = cls.default_instruction if instruction is None else instruction
        suffix = f"-{chosen}" if chosen else ""
        return _TASKS_DIR / cls.family / f"{cls.family}-o{num_objects}{suffix}.json"

    @classmethod
    def available_instructions(cls, num_objects: int) -> list[str]:
        """Return the instructions *num_objects* registers, empty if count-only."""
        prefix = f"{cls.family}-o{num_objects}"
        instructions = []
        for path in sorted((_TASKS_DIR / cls.family).glob(f"{prefix}*.json")):
            remainder = path.stem[len(prefix) :]
            if remainder.startswith("-"):
                instructions.append(remainder[1:])
        return instructions

    def __init__(
        self,
        num_objects: int | None = None,
        instruction: str | None = None,
        **kwargs: Any,
    ) -> None:
        if num_objects is None:
            num_objects = min(self.supported_counts)
        if num_objects not in self.supported_counts:
            raise ValueError(
                f"{self.family} registers counts "
                f"{sorted(self.supported_counts)}, got {num_objects}"
            )
        if "task_config_path" in kwargs:
            raise ValueError(
                f"{self.family} builds task_config_path from num_objects; pass "
                f"num_objects (and instruction) instead"
            )
        task_path = self.task_path(num_objects, instruction)
        if not task_path.exists():
            available = self.available_instructions(num_objects)
            raise ValueError(
                f"{self.family} has no task {task_path.name}; "
                f"instructions registered for o{num_objects}: {available}"
            )
        # Every dynamic3D task JSON defines this camera, and it frames the task
        # rather than the room.
        kwargs.setdefault("scene_render_camera", "task_view")
        super().__init__(  # type: ignore[call-arg]
            num_objects=num_objects,
            task_config_path=str(task_path),
            **kwargs,
        )


class ConstrainedCupboard3DEnv(TaskFamilyEnvMixin, TidyBot3DEnv):
    """Select a ConstrainedCupboard3D task by its number of rods."""

    family = "ConstrainedCupboard3D"
    supported_counts = frozenset(range(1, 7))


class Dynamo3DEnv(TaskFamilyEnvMixin, TidyBot3DEnv):
    """Select a Dynamo3D task by its number of obstacle chairs.

    The variants draw their chairs from three chair models, and ``o12`` places
    eleven of them, so the label names the variant rather than counting objects.
    """

    family = "Dynamo3D"
    supported_counts = frozenset({1, 3, 12})


class Shelf3DEnv(TaskFamilyEnvMixin, TidyBot3DEnv):
    """Select a dynamic Shelf3D task by its number of cubes.

    Unrelated to :class:`kinder.envs.kinematic3d.shelf3d.Shelf3DEnv`, which shares
    the name but is kinematic.
    """

    family = "Shelf3D"
    supported_counts = frozenset({1, 2, 8})


class Tossing3DEnv(TaskFamilyEnvMixin, TidyBot3DEnv):
    """Select a Tossing3D task by its number of cubes to toss."""

    family = "Tossing3D"
    supported_counts = frozenset({1, 2})


class ScoopPour3DEnv(TaskFamilyEnvMixin, TidyBot3DEnv):
    """Select a ScoopPour3D task by its number of cubes to scoop."""

    family = "ScoopPour3D"
    supported_counts = frozenset({10, 100})


class SweepSimple3DEnv(TaskFamilyEnvMixin, TidyBot3DEnv):
    """Select a SweepSimple3D task by its number of cubes to sweep.

    Defaults to sweeping left on the island, the one instruction every count
    registers; the right-side instructions exist at 50 cubes only.
    """

    family = "SweepSimple3D"
    supported_counts = frozenset({1, 5, 10, 50})
    default_instruction = "sweep_the_blocks_to_the_left_side_of_the_kitchen_island"


class SortClutteredBlocks3DEnv(TaskFamilyEnvMixin, TidyBot3DEnv):
    """Select a SortClutteredBlocks3D task by its number of cubes to sort.

    Defaults to sorting into bins, which keeps four bins fixed while the cubes
    scale. The cupboard and bowl instructions each register one count only.
    """

    family = "SortClutteredBlocks3D"
    supported_counts = frozenset({4, 20})
    default_instruction = "sort_the_cluttered_blocks_into_bins"


#: Every family exposed here, for tests and for callers enumerating the set.
TASK_FAMILY_ENVS: tuple[type[TaskFamilyEnvMixin], ...] = (
    ConstrainedCupboard3DEnv,
    Dynamo3DEnv,
    Shelf3DEnv,
    Tossing3DEnv,
    ScoopPour3DEnv,
    SweepSimple3DEnv,
    SortClutteredBlocks3DEnv,
)

__all__ = [
    "TASK_FAMILY_ENVS",
    "ConstrainedCupboard3DEnv",
    "Dynamo3DEnv",
    "ScoopPour3DEnv",
    "Shelf3DEnv",
    "SortClutteredBlocks3DEnv",
    "SweepSimple3DEnv",
    "TaskFamilyEnvMixin",
    "Tossing3DEnv",
]
