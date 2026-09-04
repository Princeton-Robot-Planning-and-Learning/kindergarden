"""Register environments and expose them through make()."""

import json
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

from kinder.macros import (  # pylint: disable=wrong-import-position
    DISABLE_AUTO_DYNAMIC3D_SCENES_DOWNLOAD,
    is_truthy_env_var,
)
from kinder.wrappers import (  # pylint: disable=wrong-import-position,unused-import
    NoisyAction,
    NoisyObservation,
)

# Registry of environment classes with their metadata
ENV_CLASSES: dict[str, dict[str, Any]] = {}


def _check_deps(*modules: str) -> bool:
    """Return True if all named modules are importable.

    Catches any exception (not just ImportError) because some packages may import
    successfully but fail during initialization — e.g. mujoco raising AttributeError
    when OpenGL is unavailable in headless CI.
    """
    for mod in modules:
        try:
            __import__(mod)
        except Exception:  # pylint: disable=broad-except
            return False
    return True


# Map from physics backend name to the modules required for that backend.
_BACKEND_DEPS: dict[str, tuple[str, ...]] = {
    "tomsgeoms2d": ("tomsgeoms2d",),
    "pymunk": ("pymunk", "tomsgeoms2d"),
    "pybullet": ("pybullet", "pybullet_helpers"),
    "prpl_kinematics": ("pybullet", "prpl_kinematics"),
    "mujoco": ("mujoco",),
}


def register_all_environments() -> None:
    """Add all benchmark environments to the gymnasium registry.

    Environments whose backend dependencies are not installed are silently skipped. The
    default ``pip install kindergarden`` includes every backend; to install a single
    one, see the requirements files described in the README.
    """
    # NOTE: ids must start with "kinder/" to be properly registered.

    # Detect headless mode (no DISPLAY) and pick that platform's offscreen backend.
    if not os.environ.get("DISPLAY"):
        if sys.platform == "darwin":
            # cgl is the macOS counterpart of osmesa below. glfw would open an
            # NSWindow, which is both pointless with no display and fatal when the
            # render is driven from a worker thread, since AppKit allows NSWindow
            # on the main thread only.
            os.environ["MUJOCO_GL"] = "cgl"
            os.environ["PYOPENGL_PLATFORM"] = "darwin"
        else:
            os.environ["MUJOCO_GL"] = "osmesa"
            os.environ["PYOPENGL_PLATFORM"] = "osmesa"

    if _check_deps(*_BACKEND_DEPS["tomsgeoms2d"]):
        _register_kinematic2d()

    if _check_deps(*_BACKEND_DEPS["pymunk"]):
        _register_dynamic2d()

    if _check_deps(*_BACKEND_DEPS["pybullet"]):
        _register_kinematic3d()
        # Registration follows the backend, not the category.
        _register_dynamic3d_pybullet()

    if _check_deps(*_BACKEND_DEPS["prpl_kinematics"]):
        _register_kinematic3d_v2()

    if _check_deps(*_BACKEND_DEPS["mujoco"]):
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
        backend="tomsgeoms2d",
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
        backend="tomsgeoms2d",
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
        backend="tomsgeoms2d",
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
        backend="tomsgeoms2d",
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
        backend="tomsgeoms2d",
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
        backend="tomsgeoms2d",
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
        backend="pymunk",
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
        backend="pymunk",
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
        backend="pymunk",
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
        backend="pymunk",
        variant_ids=variant_ids,
    )


def _register_kinematic3d() -> None:
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
        backend="pybullet",
        variant_ids=[variant_id],
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
        backend="pybullet",
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
        backend="pybullet",
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
        backend="pybullet",
        variant_ids=variant_ids,
    )

    # KinematicCylinderShelf3D environment.
    num_cylinders = [1, 2, 3]
    variant_ids = []
    for num_cylinder in num_cylinders:
        variant_id = f"kinder/KinematicCylinderShelf3D-o{num_cylinder}-v0"
        _register(
            id=variant_id,
            entry_point="kinder.envs.kinematic3d.cylinder_shelf3d:CylinderShelf3DEnv",
            kwargs={"num_cylinders": num_cylinder},
        )
        variant_ids.append(variant_id)
    _register_env_class(
        class_name="KinematicCylinderShelf3D",
        entry_point="kinder.envs.kinematic3d.cylinder_shelf3d:CylinderShelf3DEnv",
        category="Kinematic3D",
        backend="pybullet",
        variant_ids=variant_ids,
    )

    # PrplLab3D environment.
    num_cubes = [1, 2]
    variant_ids = []
    for num_cube in num_cubes:
        variant_id = f"kinder/PrplLab3D-o{num_cube}-v0"
        _register(
            id=variant_id,
            entry_point="kinder.envs.kinematic3d.prpl3d:PrplLab3DEnv",
            kwargs={"num_cubes": num_cube},
        )
        variant_ids.append(variant_id)
    _register_env_class(
        class_name="PrplLab3D",
        entry_point="kinder.envs.kinematic3d.prpl3d:PrplLab3DEnv",
        category="Kinematic3D",
        backend="pybullet",
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
        backend="pybullet",
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
        backend="pybullet",
        variant_ids=variant_ids,
    )


def _register_kinematic3d_v2() -> None:
    # VegaMotion3D environment.
    variant_id = "kinder/VegaMotion3D-v0"
    _register(
        id=variant_id,
        entry_point="kinder.envs.kinematic3d_v2.vega_motion3d:VegaMotion3DEnv",
    )
    _register_env_class(
        class_name="VegaMotion3D",
        entry_point="kinder.envs.kinematic3d_v2.vega_motion3d:VegaMotion3DEnv",
        category="Kinematic3Dv2",
        backend="prpl_kinematics",
        variant_ids=[variant_id],
    )

    # VegaPickPlace3D environment.
    variant_id = "kinder/VegaPickPlace3D-v0"
    _register(
        id=variant_id,
        entry_point="kinder.envs.kinematic3d_v2.vega_pickplace3d:VegaPickPlace3DEnv",
    )
    _register_env_class(
        class_name="VegaPickPlace3D",
        entry_point="kinder.envs.kinematic3d_v2.vega_pickplace3d:VegaPickPlace3DEnv",
        category="Kinematic3Dv2",
        backend="prpl_kinematics",
        variant_ids=[variant_id],
    )


def _register_dynamic3d_pybullet() -> None:
    """Register the Dynamic3D environments that run on PyBullet."""
    # LimbRepositioning3D environment, one variant per (scene, limb) pair.
    scene_types = ["isolated", "human", "wheelchair", "bed"]
    limb_names = ["left-arm", "right-arm", "left-leg", "right-leg"]
    entry_point = "kinder.envs.dynamic3d.limbrepositioning3d:LimbRepositioning3DEnv"
    variant_ids = []
    for scene_type in scene_types:
        for limb_name in limb_names:
            variant = f"{scene_type}-{limb_name}"
            variant_id = f"kinder/LimbRepositioning3D-{variant}-v0"
            _register(
                id=variant_id,
                entry_point=entry_point,
                kwargs={"variant": variant},
            )
            variant_ids.append(variant_id)
    _register_env_class(
        class_name="LimbRepositioning3D",
        entry_point=entry_point,
        category="Dynamic3D",
        backend="pybullet",
        variant_ids=variant_ids,
    )


def _register_dynamic3d() -> None:
    # Tasks with different scenes and object counts
    tasks_root = Path(__file__).parent / "envs" / "dynamic3d" / "tasks"

    env_class_variants: dict[str, dict[str, list[str]]] = {}
    for task_item in tasks_root.iterdir():
        if task_item.is_dir():
            # Handle folders and register each config file within
            # Each folder corresponds to a task type
            folder_name = task_item.name
            for task_config in task_item.iterdir():
                # Go through variants for this task
                if task_config.is_file():
                    config_name = task_config.stem
                    # Map the robot type declared in the task config to the
                    # environment class implementing it.
                    robot_env_classes = {
                        "tidybot": "TidyBot3D",
                        "fr3": "Franka3D",
                        "rby1a": "RBY1A3D",
                    }
                    try:
                        with open(task_config, "r", encoding="utf-8") as f:
                            robot_key = next(iter(json.load(f)["robots"]))
                        robot = robot_env_classes[robot_key]
                    except (
                        json.JSONDecodeError,
                        KeyError,
                        StopIteration,
                    ) as e:
                        raise RuntimeError(
                            f"Cannot determine the robot for task config "
                            f"{task_config}: it must be valid JSON with a "
                            f"non-empty 'robots' dict whose first key is one "
                            f"of {sorted(robot_env_classes)}."
                        ) from e
                    task_cfg = "-".join(config_name.split("-")[1:])
                    variant_id = f"kinder/{folder_name}-{task_cfg}-v0"
                    _register(
                        id=variant_id,
                        entry_point=f"kinder.envs.dynamic3d.envs:{robot}Env",
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
                entry_point=f"kinder.envs.dynamic3d.envs:{robot}Env",
                category="Dynamic3D",
                backend="mujoco",
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
    backend: str,
    variant_ids: list[str],
) -> None:
    """Register an environment class with its metadata.

    Args:
        class_name: Base name of the environment class (e.g., "ClutteredStorage2D")
        entry_point: Python import path to the environment class
        category: Category of the environment (e.g., "Kinematic2D", "Dynamic2D")
        backend: Physics backend running the environment, a key of _BACKEND_DEPS
        variant_ids: List of registered variant IDs for this class
    """
    assert backend in _BACKEND_DEPS, f"Unknown backend {backend}"
    ENV_CLASSES[class_name] = {
        "entry_point": entry_point,
        "category": category,
        "backend": backend,
        "variants": variant_ids,
    }


def _ensure_assets_for_env(env_id: str) -> None:
    """Auto-download MimicLabs assets once, only for MuJoCo environments."""
    mujoco_env_ids = {
        vid
        for metadata in ENV_CLASSES.values()
        if metadata["backend"] == "mujoco"
        for vid in metadata["variants"]
    }
    if env_id not in mujoco_env_ids:
        return

    if is_truthy_env_var(DISABLE_AUTO_DYNAMIC3D_SCENES_DOWNLOAD):
        print(
            f"Auto-download of MimicLabs assets is disabled via "
            f"{DISABLE_AUTO_DYNAMIC3D_SCENES_DOWNLOAD} environment variable. "
            "If you want to enable it, unset this variable or set it to a falsy value."
        )
        return

    package_root = Path(__file__).parent
    mimiclabs_scenes_dir = (
        package_root
        / "envs"
        / "dynamic3d"
        / "models"
        / "assets"
        / "mimiclabs_scenes"
        / "meshes"
    )

    # If assets are already present, skip download.
    if mimiclabs_scenes_dir.exists():
        return

    from kinder.utils import (  # pylint: disable=import-outside-toplevel
        download_mimiclabs_assets,
    )

    try:
        print(
            "Auto-downloading MimicLabs assets for MuJoCo environments "
            "(this may take a few minutes)..."
        )
        download_mimiclabs_assets(non_interactive=True)
    except Exception as err:  # pylint: disable=broad-except
        raise RuntimeError(
            "Auto-download of MimicLabs assets failed. "
            "Call kinder.utils.download_mimiclabs_assets() manually. "
            f"Error: {err}"
        ) from err


def make(*args, **kwargs) -> gymnasium.Env:
    """Create a registered environment from its name."""
    env_id = kwargs.get("id")
    if env_id is None and args:
        env_id = args[0]
    if isinstance(env_id, str):
        _ensure_assets_for_env(env_id)
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
        - backend: Physics backend (e.g., "pybullet", "mujoco")
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
