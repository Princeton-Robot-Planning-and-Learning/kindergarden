"""Register environments and expose them through make()."""

import os
import sys
import warnings
from pathlib import Path
from typing import Any

# Silence warnings from third-party packages
warnings.filterwarnings("ignore", category=DeprecationWarning, module="kortex_api")
warnings.filterwarnings("ignore", category=UserWarning, module="phoenix6")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="pkg_resources")

# Need to import after silencing warnings
import gymnasium  # pylint: disable=wrong-import-position
from gymnasium.envs.registration import (  # pylint: disable=wrong-import-position
    register,
)

# Registry of environment classes with their metadata
ENV_CLASSES: dict[str, dict[str, Any]] = {}


def _check_deps(*modules: str) -> bool:
    """Return True if all named modules are importable.

    Catches any exception (not just ImportError) because some packages
    may import successfully but fail during initialization — e.g. mujoco
    raising AttributeError when OpenGL is unavailable in headless CI.
    """
    for mod in modules:
        try:
            __import__(mod)
        except Exception:  # pylint: disable=broad-except
            return False
    return True


# Map from category name to the modules required for that category.
_CATEGORY_DEPS: dict[str, tuple[str, ...]] = {
    "Kinematic2D": ("tomsgeoms2d",),
    "Dynamic2D": ("pymunk", "tomsgeoms2d"),
    "Kinematic3D": ("pybullet", "pybullet_helpers"),
    "Dynamic3D": ("mujoco",),
}


def register_all_environments() -> None:
    """Add all benchmark environments to the gymnasium registry.

    Categories whose optional dependencies are not installed are silently
    skipped.  Install the corresponding extras to enable them, e.g.
    ``pip install kindergarden[dynamic2d]``.
    """
    # NOTE: ids must start with "kinder/" to be properly registered.

    # Detect headless mode (no DISPLAY) and set OSMesa if needed
    if not os.environ.get("DISPLAY"):
        if sys.platform == "darwin":
            os.environ["MUJOCO_GL"] = "glfw"
            os.environ["PYOPENGL_PLATFORM"] = "glfw"
        else:
            os.environ["MUJOCO_GL"] = "osmesa"
            os.environ["PYOPENGL_PLATFORM"] = "osmesa"

    if _check_deps(*_CATEGORY_DEPS["Kinematic2D"]):
        _register_kinematic2d()

    if _check_deps(*_CATEGORY_DEPS["Dynamic2D"]):
        _register_dynamic2d()

    if _check_deps(*_CATEGORY_DEPS["Kinematic3D"]):
        _register_kinematic3d()

    if _check_deps(*_CATEGORY_DEPS["Dynamic3D"]):
        _register_dynamic3d()


def _register_kinematic2d() -> None:
    # Obstructions2D environment with different numbers of obstructions.
    num_obstructions = [0, 1, 2, 3, 4]
    variant_ids = []
    for num_obstruction in num_obstructions:
        variant_id = f"kinder/Obstruction2D-o{num_obstruction}-v0"
        _register(
            id=variant_id,
            entry_point="kinder.envs.kinematic2d.obstruction2d:Obstruction2DEnv",
            kwargs={"num_obstructions": num_obstruction},
        )
        variant_ids.append(variant_id)
    _register_env_class(
        class_name="Obstruction2D",
        entry_point="kinder.envs.kinematic2d.obstruction2d:Obstruction2DEnv",
        category="Kinematic2D",
        variant_ids=variant_ids,
    )

    # ClutteredRetrieval2D environment with different numbers of obstructions.
    num_obstructions = [1, 10, 25]
    variant_ids = []
    for num_obstruction in num_obstructions:
        variant_id = f"kinder/ClutteredRetrieval2D-o{num_obstruction}-v0"
        _register(
            id=variant_id,
            entry_point="kinder.envs.kinematic2d.clutteredretrieval2d:ClutteredRetrieval2DEnv",  # pylint: disable=line-too-long
            kwargs={"num_obstructions": num_obstruction},
        )
        variant_ids.append(variant_id)
    _register_env_class(
        class_name="ClutteredRetrieval2D",
        entry_point="kinder.envs.kinematic2d.clutteredretrieval2d:ClutteredRetrieval2DEnv",  # pylint: disable=line-too-long
        category="Kinematic2D",
        variant_ids=variant_ids,
    )

    # ClutteredStorage2D environment with different numbers of blocks.
    num_blocks = [1, 3, 7, 15]
    variant_ids = []
    for num_block in num_blocks:
        variant_id = f"kinder/ClutteredStorage2D-b{num_block}-v0"
        _register(
            id=variant_id,
            entry_point="kinder.envs.kinematic2d.clutteredstorage2d:ClutteredStorage2DEnv",  # pylint: disable=line-too-long
            kwargs={"num_blocks": num_block},
        )
        variant_ids.append(variant_id)
    _register_env_class(
        class_name="ClutteredStorage2D",
        entry_point="kinder.envs.kinematic2d.clutteredstorage2d:ClutteredStorage2DEnv",
        category="Kinematic2D",
        variant_ids=variant_ids,
    )

    # Motion2D environment with different numbers of passages.
    num_passages = [0, 1, 2, 3, 4, 5]
    variant_ids = []
    for num_passage in num_passages:
        variant_id = f"kinder/Motion2D-p{num_passage}-v0"
        _register(
            id=variant_id,
            entry_point="kinder.envs.kinematic2d.motion2d:Motion2DEnv",
            kwargs={"num_passages": num_passage},
        )
        variant_ids.append(variant_id)
    _register_env_class(
        class_name="Motion2D",
        entry_point="kinder.envs.kinematic2d.motion2d:Motion2DEnv",
        category="Kinematic2D",
        variant_ids=variant_ids,
    )

    # StickButton2D environment with different numbers of buttons.
    num_buttons = [1, 2, 3, 5, 10]
    variant_ids = []
    for num_button in num_buttons:
        variant_id = f"kinder/StickButton2D-b{num_button}-v0"
        _register(
            id=variant_id,
            entry_point="kinder.envs.kinematic2d.stickbutton2d:StickButton2DEnv",
            kwargs={"num_buttons": num_button},
        )
        variant_ids.append(variant_id)
    _register_env_class(
        class_name="StickButton2D",
        entry_point="kinder.envs.kinematic2d.stickbutton2d:StickButton2DEnv",
        category="Kinematic2D",
        variant_ids=variant_ids,
    )

    # PushPullHook2D environment
    variant_id = "kinder/PushPullHook2D-v0"
    _register(
        id=variant_id,
        entry_point="kinder.envs.kinematic2d.pushpullhook2d:PushPullHook2DEnv",
    )
    _register_env_class(
        class_name="PushPullHook2D",
        entry_point="kinder.envs.kinematic2d.pushpullhook2d:PushPullHook2DEnv",
        category="Kinematic2D",
        variant_ids=[variant_id],
    )


def _register_dynamic2d() -> None:
    # DynObstruction2D environment with different numbers of obstructions.
    num_obstructions = [0, 1, 2, 3]
    variant_ids = []
    for num_obstruction in num_obstructions:
        variant_id = f"kinder/DynObstruction2D-o{num_obstruction}-v0"
        _register(
            id=variant_id,
            entry_point="kinder.envs.dynamic2d.dyn_obstruction2d:DynObstruction2DEnv",
            kwargs={"num_obstructions": num_obstruction},
        )
        variant_ids.append(variant_id)
    _register_env_class(
        class_name="DynObstruction2D",
        entry_point="kinder.envs.dynamic2d.dyn_obstruction2d:DynObstruction2DEnv",
        category="Dynamic2D",
        variant_ids=variant_ids,
    )

    # DynPushPullStick2D environment with different numbers of obstructions.
    num_obstructions = [0, 1, 5]
    variant_ids = []
    for num_obstruction in num_obstructions:
        variant_id = f"kinder/DynPushPullHook2D-o{num_obstruction}-v0"
        _register(
            id=variant_id,
            entry_point="kinder.envs.dynamic2d.dyn_pushpullhook2d:DynPushPullHook2DEnv",
            kwargs={"num_obstructions": num_obstruction},
        )
        variant_ids.append(variant_id)
    _register_env_class(
        class_name="DynPushPullHook2D",
        entry_point="kinder.envs.dynamic2d.dyn_pushpullhook2d:DynPushPullHook2DEnv",
        category="Dynamic2D",
        variant_ids=variant_ids,
    )

    # DynPushT2D environment
    variant_id = "kinder/DynPushT2D-t1-v0"
    _register(
        id=variant_id,
        entry_point="kinder.envs.dynamic2d.dyn_pusht2d:DynPushT2DEnv",
        kwargs={"num_tee": 1},
    )
    _register_env_class(
        class_name="DynPushT2D",
        entry_point="kinder.envs.dynamic2d.dyn_pusht2d:DynPushT2DEnv",
        category="Dynamic2D",
        variant_ids=[variant_id],
    )

    # DynScoopPour2D environment with different numbers of small objects
    num_objects = [10, 20, 30, 50]
    variant_ids = []
    for num_object in num_objects:
        num_circles = num_object // 2
        num_squares = num_object - num_circles
        variant_id = f"kinder/DynScoopPour2D-o{num_object}-v0"
        _register(
            id=variant_id,
            entry_point="kinder.envs.dynamic2d.dyn_scooppour2d:DynScoopPour2DEnv",
            kwargs={"num_small_circles": num_circles, "num_small_squares": num_squares},
        )
        variant_ids.append(variant_id)
    _register_env_class(
        class_name="DynScoopPour2D",
        entry_point="kinder.envs.dynamic2d.dyn_scooppour2d:DynScoopPour2DEnv",
        category="Dynamic2D",
        variant_ids=variant_ids,
    )


def _register_kinematic3d() -> None:
    # Motion3D environment.
    variant_id = "kinder/Motion3D-v0"
    _register(
        id=variant_id,
        entry_point="kinder.envs.kinematic3d.motion3d:Motion3DEnv",
    )
    _register_env_class(
        class_name="Motion3D",
        entry_point="kinder.envs.kinematic3d.motion3d:Motion3DEnv",
        category="Kinematic3D",
        variant_ids=[variant_id],
    )

    # BaseMotion3D environment.
    variant_id = "kinder/BaseMotion3D-v0"
    _register(
        id=variant_id,
        entry_point="kinder.envs.kinematic3d.base_motion3d:BaseMotion3DEnv",
    )
    _register_env_class(
        class_name="BaseMotion3D",
        entry_point="kinder.envs.kinematic3d.base_motion3d:BaseMotion3DEnv",
        category="Kinematic3D",
        variant_ids=[variant_id],
    )

    # Ground3D environment.
    num_cubes = [1, 2, 3]
    variant_ids = []
    for num_cube in num_cubes:
        variant_id = f"kinder/Ground3D-o{num_cube}-v0"
        _register(
            id=variant_id,
            entry_point="kinder.envs.kinematic3d.ground3d:Ground3DEnv",
            kwargs={"num_cubes": num_cube},
        )
        variant_ids.append(variant_id)
    _register_env_class(
        class_name="Ground3D",
        entry_point="kinder.envs.kinematic3d.ground3d:Ground3DEnv",
        category="Kinematic3D",
        variant_ids=variant_ids,
    )

    # Table3D environment.
    num_cubes = [1, 2, 3]
    variant_ids = []
    for num_cube in num_cubes:
        variant_id = f"kinder/Table3D-o{num_cube}-v0"
        _register(
            id=variant_id,
            entry_point="kinder.envs.kinematic3d.table3d:Table3DEnv",
            kwargs={"num_cubes": num_cube},
        )
        variant_ids.append(variant_id)
    _register_env_class(
        class_name="Table3D",
        entry_point="kinder.envs.kinematic3d.table3d:Table3DEnv",
        category="Kinematic3D",
        variant_ids=variant_ids,
    )

    # Transport3D environment.
    num_cubes = [1, 2]
    num_boxes = 1
    variant_ids = []
    for num_cube in num_cubes:
        variant_id = f"kinder/Transport3D-o{num_cube}-v0"
        _register(
            id=variant_id,
            entry_point="kinder.envs.kinematic3d.transport3d:Transport3DEnv",
            kwargs={"num_cubes": num_cube, "num_boxes": num_boxes},
        )
        variant_ids.append(variant_id)
    _register_env_class(
        class_name="Transport3D",
        entry_point="kinder.envs.kinematic3d.transport3d:Transport3DEnv",
        category="Kinematic3D",
        variant_ids=variant_ids,
    )

    # KinematicShelf3D environment.
    num_cubes = [1, 2, 3, 5, 10]
    variant_ids = []
    for num_cube in num_cubes:
        variant_id = f"kinder/KinematicShelf3D-o{num_cube}-v0"
        _register(
            id=variant_id,
            entry_point="kinder.envs.kinematic3d.shelf3d:Shelf3DEnv",
            kwargs={"num_cubes": num_cube},
        )
        variant_ids.append(variant_id)
    _register_env_class(
        class_name="KinematicShelf3D",
        entry_point="kinder.envs.kinematic3d.shelf3d:Shelf3DEnv",
        category="Kinematic3D",
        variant_ids=variant_ids,
    )

    # Obstructions3D environment with different numbers of obstructions.
    num_obstructions = [0, 1, 2, 3, 4]
    variant_ids = []
    for num_obstruction in num_obstructions:
        variant_id = f"kinder/Obstruction3D-o{num_obstruction}-v0"
        _register(
            id=variant_id,
            entry_point="kinder.envs.kinematic3d.obstruction3d:Obstruction3DEnv",
            kwargs={"num_obstructions": num_obstruction},
        )
        variant_ids.append(variant_id)
    _register_env_class(
        class_name="Obstruction3D",
        entry_point="kinder.envs.kinematic3d.obstruction3d:Obstruction3DEnv",
        category="Kinematic3D",
        variant_ids=variant_ids,
    )

    # Packing3D environment with different numbers of parts to be packed.
    num_parts = [1, 2, 3]
    variant_ids = []
    for num_part in num_parts:
        variant_id = f"kinder/Packing3D-p{num_part}-v0"
        _register(
            id=variant_id,
            entry_point="kinder.envs.kinematic3d.packing3d:Packing3DEnv",
            kwargs={"num_parts": num_part},
        )
        variant_ids.append(variant_id)
    _register_env_class(
        class_name="Packing3D",
        entry_point="kinder.envs.kinematic3d.packing3d:Packing3DEnv",
        category="Kinematic3D",
        variant_ids=variant_ids,
    )


def _register_dynamic3d() -> None:
    # Tasks with different scenes and object counts
    tasks_root = Path(__file__).parent / "envs" / "dynamic3d" / "tasks"

    env_class_variants: dict[str, dict[str, list[str]]] = {}
    for task_item in tasks_root.iterdir():
        if task_item.is_file() and task_item.suffix == ".json":
            # Handle single config file directly in tasks_root
            config_name = task_item.stem
            robot = {"tidybot": "TidyBot3D", "rby1a": "RBY1A3D"}[
                config_name.split("-")[0]
            ]
            scene_type = config_name.split("-")[1]
            num_task_objects = int(config_name.split("-")[2][1:])
            task_cfg = "-".join(config_name.split("-")[1:])
            variant_id = f"kinder/{robot}-{task_cfg}-v0"
            _register(
                id=variant_id,
                entry_point=f"kinder.envs.dynamic3d.tidybot3d:{robot}Env",
                kwargs={
                    "scene_type": scene_type,
                    "num_objects": num_task_objects,
                    "task_config_path": str(task_item),
                },
            )
            if robot not in env_class_variants:
                env_class_variants[robot] = {}
            if robot not in env_class_variants[robot]:
                env_class_variants[robot][robot] = []
            env_class_variants[robot][robot].append(variant_id)
        elif task_item.is_dir():
            # Handle folders and register each config file within
            # Each folder corresponds to a task type
            folder_name = task_item.name
            for task_config in task_item.iterdir():
                # Go through variants for this task
                if task_config.is_file():
                    config_name = task_config.stem
                    robot = "TidyBot3D"
                    # Note: we only support one robot at the moment
                    # In the future, get robot from config.
                    scene_type = config_name.split("-")[1]
                    task_cfg = "-".join(config_name.split("-")[1:])
                    variant_id = f"kinder/{folder_name}-{task_cfg}-v0"
                    _register(
                        id=variant_id,
                        entry_point=f"kinder.envs.dynamic3d.tidybot3d:{robot}Env",
                        kwargs={
                            "task_config_path": str(task_config),
                            "scene_render_camera": "task_view",
                        },
                    )
                    if folder_name not in env_class_variants:
                        env_class_variants[folder_name] = {}
                    if robot not in env_class_variants[folder_name]:
                        env_class_variants[folder_name][robot] = []
                    env_class_variants[folder_name][robot].append(variant_id)

    for class_name, robot_variant_ids in env_class_variants.items():
        for robot, variant_ids in robot_variant_ids.items():
            _register_env_class(
                class_name=class_name,
                entry_point=f"kinder.envs.dynamic3d.tidybot3d:{robot}Env",
                category="Dynamic3D",
                variant_ids=variant_ids,
            )


def _register(id: str, *args, **kwargs) -> None:  # pylint: disable=redefined-builtin
    """Call register(), but only if the environment id is not already registered.

    This is to avoid noisy logging.warnings in register(). We are assuming that envs
    with the same id are equivalent, so this is safe.
    """
    if id not in gymnasium.registry:
        register(id, *args, **kwargs)


def _register_env_class(
    class_name: str,
    entry_point: str,
    category: str,
    variant_ids: list[str],
) -> None:
    """Register an environment class with its metadata.

    Args:
        class_name: Base name of the environment class (e.g., "ClutteredStorage2D")
        entry_point: Python import path to the environment class
        category: Category of the environment (e.g., "Kinematic2D", "Dynamic2D")
        variant_ids: List of registered variant IDs for this class
    """
    ENV_CLASSES[class_name] = {
        "entry_point": entry_point,
        "category": category,
        "variants": variant_ids,
    }


def make(*args, **kwargs) -> gymnasium.Env:
    """Create a registered environment from its name."""
    return gymnasium.make(*args, **kwargs)


def get_all_env_ids() -> set[str]:
    """Get all known benchmark environments."""
    return {env for env in gymnasium.registry if env.startswith("kinder/")}


def get_env_classes() -> dict[str, dict[str, Any]]:
    """Get all registered environment classes with their metadata.

    Returns:
        Dictionary mapping class names to their metadata including:
        - entry_point: Python import path to the class
        - category: Environment category (e.g., "Kinematic2D", "Dynamic2D")
        - variants: List of registered variant IDs
    """
    return ENV_CLASSES.copy()


def get_env_variants(class_name: str) -> list[str]:
    """Get all variant IDs for a given environment class.

    Args:
        class_name: Base name of the environment class (e.g., "ClutteredStorage2D")

    Returns:
        List of variant IDs (e.g., ["kinder/ClutteredStorage2D-b1-v0", ...])

    Raises:
        KeyError: If the environment class is not registered
    """
    return ENV_CLASSES[class_name]["variants"]


def get_env_categories() -> dict[str, list[str]]:
    """Get environment classes organized by category.

    Returns:
        Dictionary mapping categories to lists of environment class names
    """
    categories: dict[str, list[str]] = {}
    for class_name, metadata in ENV_CLASSES.items():
        category = metadata["category"]
        if category not in categories:
            categories[category] = []
        categories[category].append(class_name)
    return categories
