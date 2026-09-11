"""Tests for the TidyBot3D NAMO (Navigation Among Movable Objects) environment."""

from pathlib import Path

import mujoco
import numpy as np
import pytest
from gymnasium.wrappers import RecordVideo

import kinder
from kinder.envs.dynamic3d.envs import ObjectCentricTidyBot3DEnv
from tests.conftest import MAKE_VIDEOS

# Path to tasks directory
TASKS_DIR = (
    Path(__file__).parent.parent.parent.parent
    / "src"
    / "kinder"
    / "envs"
    / "dynamic3d"
    / "tasks"
)

# Path to mimiclabs scenes for skip condition
MIMICLABS_SCENES_DIR = (
    Path(__file__).parent.parent.parent.parent
    / "src"
    / "kinder"
    / "envs"
    / "dynamic3d"
    / "models"
    / "assets"
    / "mimiclabs_scenes"
    / "meshes"
)


def test_namo_env_loads():
    """Test that the NAMO environment loads correctly."""
    env = ObjectCentricTidyBot3DEnv(
        scene_type="namo",
        num_objects=1,
        task_config_path=str(TASKS_DIR / "Dynamo3D" / "Dynamo3D-o1.json"),
    )

    obs, info = env.reset(seed=42)
    assert env.observation_space.contains(obs)
    assert isinstance(info, dict)

    # Verify we have the obstacle block
    obstacle = obs.get_object_from_name("obstacle_chair")
    assert obstacle is not None

    env.close()


def test_namo_goal_not_satisfied_initially():
    """Test that goal is not satisfied after reset (block not in goal region)."""
    env = ObjectCentricTidyBot3DEnv(
        scene_type="namo",
        num_objects=1,
        task_config_path=str(TASKS_DIR / "Dynamo3D" / "Dynamo3D-o1.json"),
    )

    env.reset(seed=42)

    # Goal should not be satisfied initially
    assert not env._check_goals(), (  # pylint: disable=protected-access
        "Goal should not be satisfied after reset - "
        "obstacle_chair should not be in goal region initially"
    )

    env.close()


def test_namo_goal_satisfied_when_robot_in_region():
    """Test that goal is satisfied when robot reaches the goal region.

    In this NAMO task, the goal is for the robot (tidybot) to navigate to the goal
    region, potentially by pushing the obstacle out of the way.
    """
    kinder.register_all_environments()
    env = kinder.make(
        "kinder/Dynamo3D-o1-v0",
        render_mode="rgb_array",
        allow_state_access=True,
    )

    if MAKE_VIDEOS:
        env = RecordVideo(env, "unit_test_videos_namo_goal_satisfied")

    env.reset(seed=42)

    # Access the underlying object-centric environment
    oc_env = env.unwrapped._object_centric_env  # pylint: disable=protected-access

    # Get current state
    current_state = oc_env._get_current_state()  # pylint: disable=protected-access

    # Get the robot object
    robot = current_state.get_object_from_name(
        oc_env.robot_name
    )  # type: ignore[attr-defined]

    # Move robot to the goal region (center of goal region is at x=1.0, y=0.0)
    modified_state = current_state.copy()
    modified_state.set(robot, "pos_base_x", 1.0)
    modified_state.set(robot, "pos_base_y", 0.0)

    # Set the modified state
    oc_env.set_state(modified_state)

    # Now goal should be satisfied (robot is in the goal region)
    assert (
        oc_env._check_goals()  # pylint: disable=protected-access
    ), "Goal should be satisfied after moving robot to goal region"

    env.close()


def test_namo_goal_achieved_after_teleporting_chair_and_robot():
    """Test that goal is achieved by teleporting chair away and robot to goal region.

    This test verifies the goal checking logic by:
    1. Teleporting the obstacle chair away from the goal region
    2. Teleporting the robot into the goal region
    3. Checking that the goal is now satisfied
    """
    kinder.register_all_environments()
    env = kinder.make(
        "kinder/Dynamo3D-o1-v0",
        render_mode="rgb_array",
        allow_state_access=True,
    )

    if MAKE_VIDEOS:
        env = RecordVideo(env, "unit_test_videos_namo_teleport_goal")

    env.reset(seed=42)

    # Access the underlying object-centric environment
    oc_env = env.unwrapped._object_centric_env  # pylint: disable=protected-access

    # Goal should not be satisfied initially
    assert (
        not oc_env._check_goals()  # pylint: disable=protected-access
    ), "Goal should not be satisfied after reset"

    # Get current state
    current_state = oc_env._get_current_state()  # pylint: disable=protected-access

    # Get the robot and obstacle chair objects
    robot = current_state.get_object_from_name(
        oc_env.robot_name
    )  # type: ignore[attr-defined]
    obstacle_chair = current_state.get_object_from_name("obstacle_chair")

    # Create modified state
    modified_state = current_state.copy()

    # Teleport the obstacle chair away from the goal region
    # Goal region is at x=[0.8, 1.2], y=[-0.2, 0.2]
    # Move chair to x=-1.0, y=0.0 (far from goal region)
    modified_state.set(obstacle_chair, "x", -1.0)
    modified_state.set(obstacle_chair, "y", 0.0)
    modified_state.set(obstacle_chair, "z", 0.55)  # Keep it at its original height

    # Teleport the robot to the center of the goal region
    # Goal region center is at x=1.0, y=0.0
    modified_state.set(robot, "pos_base_x", 1.0)
    modified_state.set(robot, "pos_base_y", 0.0)

    # Set the modified state
    oc_env.set_state(modified_state)

    # Verify the chair was moved
    updated_state = oc_env._get_current_state()  # pylint: disable=protected-access
    chair_x = updated_state.get(obstacle_chair, "x")
    assert chair_x < 0, f"Chair should be moved to negative x, got {chair_x}"

    # Verify the robot was moved to goal region
    robot_x = updated_state.get(robot, "pos_base_x")
    robot_y = updated_state.get(robot, "pos_base_y")
    assert 0.8 <= robot_x <= 1.2, f"Robot x should be in [0.8, 1.2], got {robot_x}"
    assert -0.2 <= robot_y <= 0.2, f"Robot y should be in [-0.2, 0.2], got {robot_y}"

    # Now goal should be satisfied (robot is in the goal region)
    goal_satisfied = oc_env._check_goals()  # pylint: disable=protected-access
    assert (
        goal_satisfied
    ), "Goal should be satisfied after teleporting chair away and robot to goal region"

    env.close()


def test_namo_robot_can_navigate_to_goal():
    """Test that robot can navigate to the goal region by moving forward.

    This test verifies that:
    1. The chair is moved out of the way (simulating a successful push)
    2. The robot can navigate forward to the goal region
    3. The goal is achieved when the robot reaches the goal region
    """
    kinder.register_all_environments()
    if MIMICLABS_SCENES_DIR.exists():
        env = kinder.make(
            "kinder/Dynamo3D-o1-v0",
            render_mode="rgb_array",
            scene_bg=True,
            scene_render_camera="agentview_1",
        )
    else:
        env = kinder.make("kinder/Dynamo3D-o1-v0", render_mode="rgb_array")

    if MAKE_VIDEOS:
        env = RecordVideo(env, "unit_test_videos_namo_navigate")

    obs, _ = env.reset(seed=123)

    # Access the underlying object-centric environment
    oc_env = env.unwrapped._object_centric_env  # pylint: disable=protected-access

    # Get initial state
    state = env.observation_space.devectorize(obs)
    robot_name = oc_env.robot_name  # type: ignore[attr-defined]
    robot = state.get_object_from_name(robot_name)
    robot_x = state.get(robot, "pos_base_x")
    robot_y = state.get(robot, "pos_base_y")

    # Goal region is at x=0.8 to x=1.2, y=-0.2 to 0.2
    # Move robot towards the goal region center
    goal_x = 1.0
    goal_y = 0.0

    # Calculate delta per step (constant delta actions)
    max_magnitude = 1e-2
    dx = goal_x - robot_x
    dy = goal_y - robot_y
    distance = (dx**2 + dy**2) ** 0.5
    steps = int(distance / max_magnitude) + 1

    # Normalize to get unit direction, then scale by max_magnitude
    unit_dx = dx / distance * max_magnitude
    unit_dy = dy / distance * max_magnitude

    # Execute constant delta steps to reach the goal
    for _ in range(steps):
        action = np.array([unit_dx, unit_dy, 0.0] + [0.0] * 8)
        env.step(action)

    # Verify the goal is achieved
    goal_achieved = oc_env._check_goals()  # pylint: disable=protected-access
    assert goal_achieved, "Goal should be achieved after robot navigates to goal region"

    env.close()


def test_namo_action_space():
    """Test that action space is valid."""
    env = ObjectCentricTidyBot3DEnv(
        scene_type="namo",
        num_objects=1,
        task_config_path=str(TASKS_DIR / "Dynamo3D" / "Dynamo3D-o1.json"),
    )

    env.reset(seed=42)
    action = env.action_space.sample()
    assert env.action_space.contains(action)

    env.close()


def test_namo_step():
    """Test that step returns valid outputs."""
    env = ObjectCentricTidyBot3DEnv(
        scene_type="namo",
        num_objects=1,
        task_config_path=str(TASKS_DIR / "Dynamo3D" / "Dynamo3D-o1.json"),
    )

    env.reset(seed=42)
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

    assert env.observation_space.contains(obs)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)

    env.close()


@pytest.mark.skipif(
    not MIMICLABS_SCENES_DIR.exists(),
    reason="MimicLabs scenes not downloaded. "
    "Run: python scripts/download_mimiclabs_assets.py",
)
def test_namo_with_mimiclabs_scene():
    """Test NAMO environment with MimicLabs background scene."""
    env = ObjectCentricTidyBot3DEnv(
        scene_type="namo",
        num_objects=1,
        task_config_path=str(TASKS_DIR / "Dynamo3D" / "Dynamo3D-o1.json"),
        scene_bg=True,
    )

    obs, _ = env.reset(seed=42)
    assert env.observation_space.contains(obs)

    # Verify scene configuration
    active_scene = env.task_config.get("_active_scene", {})
    assert active_scene.get("type") == "mimiclabs"
    assert active_scene.get("lab") == 2

    # Take a few steps
    for _ in range(5):
        action = env.action_space.sample()
        obs, _, terminated, truncated, _ = env.step(action)
        assert env.observation_space.contains(obs)
        if terminated or truncated:
            break

    env.close()


# The Dynamo3D room: every variant must be solvable with or without the background.
DYNAMO_VARIANTS = ("o1", "o3", "o12")
ROBOT_HALF_DIAGONAL = 0.35  # TidyBot base, generous
SCENE_BG_OPTIONS = (False, True) if MIMICLABS_SCENES_DIR.exists() else (False,)


def _make_dynamo_env(variant: str, scene_bg: bool) -> ObjectCentricTidyBot3DEnv:
    return ObjectCentricTidyBot3DEnv(
        scene_type="namo",
        num_objects=int(variant[1:]),
        task_config_path=str(TASKS_DIR / "Dynamo3D" / f"Dynamo3D-{variant}.json"),
        scene_bg=scene_bg,
    )


def _colliding_walls(env: ObjectCentricTidyBot3DEnv) -> dict[str, tuple]:
    """Name -> (pos, size, quat) of every colliding wall box in the compiled model."""
    # pylint: disable=no-member
    assert env._robot_env is not None  # pylint: disable=protected-access
    model = env._robot_env.sim.model.mj_model  # pylint: disable=protected-access
    walls = {}
    for geom_id in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
        if (
            not name.startswith("wall_")
            or model.geom_type[geom_id] != mujoco.mjtGeom.mjGEOM_BOX
        ):
            continue
        if model.geom_contype[geom_id] == 0 and model.geom_conaffinity[geom_id] == 0:
            continue
        walls[name] = (
            tuple(np.round(model.geom_pos[geom_id], 4)),
            tuple(np.round(model.geom_size[geom_id], 4)),
            tuple(np.round(model.geom_quat[geom_id], 4)),
        )
    return walls


def _wall_segments(walls: dict[str, tuple]) -> list[tuple[np.ndarray, np.ndarray]]:
    """Each wall box as the xy segment along its long axis."""
    segments = []
    for pos, size, quat in walls.values():
        rotation = np.zeros(9)
        mujoco.mju_quat2Mat(rotation, np.array(quat))  # pylint: disable=no-member
        along = rotation.reshape(3, 3)[:2, 0] * size[0]
        center = np.array(pos[:2])
        segments.append((center - along, center + along))
    return segments


def _wall_clearance(walls: dict[str, tuple], x: float, y: float) -> float:
    """Distance from (x, y) to the nearest wall, negative when outside the room."""
    point = np.array([x, y])
    clearance = np.inf
    crossings = 0
    ray = np.array([1.0, 0.37])  # off-axis so no wall end lies on the ray
    for start, end in _wall_segments(walls):
        direction = end - start
        t = np.clip((point - start) @ direction / (direction @ direction), 0.0, 1.0)
        clearance = min(
            clearance, float(np.linalg.norm(point - (start + t * direction)))
        )
        # Ray-crossing parity decides whether the point is enclosed by the walls.
        denominator = np.cross(ray, direction)
        if abs(denominator) < 1e-9:
            continue
        offset = start - point
        ray_t = np.cross(offset, direction) / denominator
        wall_t = np.cross(offset, ray) / denominator
        if ray_t > 0 and 0 <= wall_t <= 1:
            crossings += 1
    return clearance if crossings % 2 == 1 else -clearance


def _footprint_half_diagonal(task_config: dict, object_name: str) -> float:
    for objects in task_config["objects"].values():
        if object_name in objects:
            length, width = objects[object_name]["footprint"][:2]
            return float(np.hypot(length, width) / 2)
    raise KeyError(object_name)


@pytest.mark.parametrize("variant", DYNAMO_VARIANTS)
def test_dynamo_room_walls_are_identical_across_backgrounds(variant: str) -> None:
    """The plain scene and the lab background collide with the same room."""
    signatures = []
    for scene_bg in SCENE_BG_OPTIONS:
        env = _make_dynamo_env(variant, scene_bg)
        env.reset(seed=0)
        walls = _colliding_walls(env)
        env.close()
        assert len(walls) == 7, sorted(walls)
        signatures.append(walls)
    assert all(walls == signatures[0] for walls in signatures)


@pytest.mark.parametrize("variant", DYNAMO_VARIANTS)
def test_dynamo_regions_lie_inside_the_room(variant: str) -> None:
    """Goal box, chair spawn regions and the robot spawn box clear every wall."""
    env = _make_dynamo_env(variant, scene_bg=False)
    env.reset(seed=0)
    walls = _colliding_walls(env)
    regions = env.task_config["regions"]
    margins = {"ground_goal_region": ROBOT_HALF_DIAGONAL}
    for predicate in env.task_config["initial_state"]:
        if predicate[0] == "on" and predicate[1] != env.robot_name:
            margins[predicate[2]] = _footprint_half_diagonal(
                env.task_config, predicate[1]
            )
    assert set(margins) == set(regions)
    boxes = {name: regions[name]["ranges"][0] for name in regions}
    boxes["robot spawn box"] = [-1.0, -1.0, 1.0, 1.0]
    margins["robot spawn box"] = ROBOT_HALF_DIAGONAL
    for name, (x_min, y_min, x_max, y_max) in boxes.items():
        for x, y in ((x_min, y_min), (x_min, y_max), (x_max, y_min), (x_max, y_max)):
            clearance = _wall_clearance(walls, x, y)
            assert (
                clearance >= margins[name]
            ), f"{name} corner ({x}, {y}) clears walls by {clearance:.2f} m"
    env.close()


@pytest.mark.parametrize("scene_bg", SCENE_BG_OPTIONS)
@pytest.mark.parametrize("variant", DYNAMO_VARIANTS)
def test_dynamo_goal_is_reachable(variant: str, scene_bg: bool) -> None:
    """Driving the base straight at the goal box satisfies the goal in every variant.

    Chairs in the way are shoved aside by the base, so a straight line is enough; what
    this checks is that no wall stands between the spawn box and the goal.
    """
    env = _make_dynamo_env(variant, scene_bg)
    env.reset(seed=0)
    x_min, y_min, x_max, y_max = env.task_config["regions"]["ground_goal_region"][
        "ranges"
    ][0]
    goal_x, goal_y = (x_min + x_max) / 2, (y_min + y_max) / 2
    robot_env = env._robot_env  # pylint: disable=protected-access
    assert robot_env is not None
    goal_achieved = False
    for _ in range(400):
        base_x, base_y = robot_env.qpos["base"][:2]
        step = np.clip([goal_x - base_x, goal_y - base_y], -0.05, 0.05)
        action = np.array([step[0], step[1], 0.0] + [0.0] * 8)
        _, _, goal_achieved, _, _ = env.step(action)
        if goal_achieved:
            break
    env.close()
    assert goal_achieved, f"{variant} goal not reached with scene_bg={scene_bg}"
