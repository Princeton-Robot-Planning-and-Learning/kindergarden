"""Object definitions for TidyBot environments."""

# Import these modules to trigger auto-registration of all object classes
from kinder.envs.dynamic3d.objects import generated_objects, other_objects, robocasa_objects
from kinder.envs.dynamic3d.objects.base import (
    MujocoFixture,
    MujocoGround,
    MujocoObject,
    get_fixture_class,
    get_object_class,
    register_fixture,
    register_object,
)
from kinder.envs.dynamic3d.objects.fixtures import Cupboard, Table
from kinder.envs.dynamic3d.objects.generated_objects import (
    GeneratedBowl,
    GeneratedSeesaw,
)
from kinder.envs.dynamic3d.objects.primitive_objects import Cube, Cuboid

__all__ = [
    "MujocoObject",
    "MujocoFixture",
    "MujocoGround",
    "register_object",
    "register_fixture",
    "get_object_class",
    "get_fixture_class",
    "Cuboid",
    "Cube",
    "Table",
    "Cupboard",
    "GeneratedBowl",
    "GeneratedSeesaw",
    "generated_objects",
    "other_objects",
    "robocasa_objects",
]
