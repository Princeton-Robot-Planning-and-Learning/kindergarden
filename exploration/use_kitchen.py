"""Kitchen environment demo: picks cube1 and places it on the kitchen counter.

Usage:
    python exploration/use_kitchen.py
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from typing import Type as TypingType

import numpy as np
import pybullet as p
from prpl_utils.utils import wrap_angle
from pybullet_helpers.geometry import Pose, SE2Pose, matrix_from_quat, set_pose
from pybullet_helpers.motion_planning import (
    create_joint_distance_fn,
    remap_joint_position_plan_to_constant_distance,
    run_motion_planning,
    run_single_arm_mobile_base_motion_planning,
    run_smooth_motion_planning_to_pose,
    smoothly_follow_end_effector_path,
)
from pybullet_helpers.utils import create_pybullet_block
from relational_structs import Object, ObjectCentricState
from relational_structs.utils import create_state_from_dict

from kinder.core import ConstantObjectKinDEREnv, FinalConfigMeta
from kinder.envs.kinematic3d.base_env import (
    Kinematic3DEnvConfig,
    ObjectCentricKinematic3DRobotEnv,
)
from kinder.envs.kinematic3d.object_types import (
    Kinematic3DCuboidType,
    Kinematic3DEnvTypeFeatures,
    Kinematic3DFixtureType,
    Kinematic3DRobotType,
)
from kinder.envs.kinematic3d.utils import (
    Kinematic3DObjectCentricState,
    extend_joints_to_include_fingers,
    sample_collision_free_object_poses,
)

KITCHEN_URDF = Path(__file__).parent / "PRPL_lab" / "urdf" / "PRPL_lab_collision.urdf"
STEP_DELAY = 0.05

# Countertop1_link sits at z=0.729 from the kitchen root (world z=0); add ~3 cm
# for the mesh thickness to get the actual resting surface.
COUNTER_SURFACE_Z = 0.76


@dataclass(frozen=True)
class KitchenEnvConfig(Kinematic3DEnvConfig, metaclass=FinalConfigMeta):
    """Config for the kitchen environment."""

    # Kitchen placed at y=1.5; its fronts (doors) face -y, so the robot at y=0 faces
    # the cabinets from the front.
    kitchen_pose: Pose = Pose((0.0, 1.5, 0.0))
    block_half_extents: tuple[float, float, float] = (0.05, 0.025, 0.025)
    floor_included_as_object: bool = True

    # Robot centred in x on the cabinet run (~1.3 m), facing +y toward the kitchen.
    robot_base_home_pose: SE2Pose = SE2Pose(1.3, 0.0, np.pi / 2)

    def get_camera_kwargs(self) -> dict[str, Any]:
        # Camera on the -y side of the scene (same side the cabinet fronts face),
        # looking toward +y so the robot and cabinet faces are visible.
        return {
            "camera_target": (1, -0.9, 0.8),
            "camera_yaw": -70,
            "camera_distance": 4.0,
            "camera_pitch": -20,
        }


class KitchenObjectCentricState(Kinematic3DObjectCentricState):
    """State for the kitchen environment."""


class ObjectCentricKitchenEnv(
    ObjectCentricKinematic3DRobotEnv[KitchenObjectCentricState, KitchenEnvConfig]
):
    """Inner kitchen environment: loads the kitchen URDF and cubes."""

    def __init__(
        self,
        num_cubes: int = 2,
        config: KitchenEnvConfig = KitchenEnvConfig(),
        **kwargs,
    ) -> None:
        super().__init__(config=config, **kwargs)
        self._num_cubes = num_cubes

        # Load kitchen URDF.
        self._kitchen_id = p.loadURDF(
            str(KITCHEN_URDF),
            basePosition=list(config.kitchen_pose.position),
            baseOrientation=list(config.kitchen_pose.orientation),
            physicsClientId=self.physics_client_id,
            useFixedBase=True,
        )

        # Invisible flat box to act as the counter collision surface.
        # The URDF has only visual geometry (mesh collision is too slow for planning),
        # so we approximate the counter with a thin box.
        self._counter_id = create_pybullet_block(
            (0.0, 0.0, 0.0, 0.0),  # fully transparent
            (1.0, 0.35, 0.01),      # ~2m wide, 0.7m deep, 2cm tall
            physics_client_id=self.physics_client_id,
        )
        set_pose(
            self._counter_id,
            Pose((1.3, 1.1, COUNTER_SURFACE_Z - 0.01)),
            self.physics_client_id,
        )

        # Create cubes.
        self._cubes: dict[str, int] = {}
        for idx in range(self._num_cubes):
            cube_id = create_pybullet_block(
                (1.0, 0.5, 0.0, 1.0),
                config.block_half_extents,
                physics_client_id=self.physics_client_id,
            )
            self._cubes[f"cube{idx}"] = cube_id

    @property
    def state_cls(self) -> TypingType[KitchenObjectCentricState]:
        return KitchenObjectCentricState

    def _create_constant_initial_state_dict(self) -> dict[Object, dict[str, float]]:
        return self._create_state_dict([("kitchen", Kinematic3DFixtureType)])

    def _reset_objects(self) -> None:
        sample_collision_free_object_poses(
            object_ids=set(self._cubes.values()),
            lb=(self.config.x_lb, self.config.y_lb, self.config.block_half_extents[2]),
            ub=(self.config.x_ub, self.config.y_ub, self.config.block_half_extents[2]),
            physics_client_id=self.physics_client_id,
            rng=self.np_random,
            other_collision_ids={self.robot.base.robot_id, self._kitchen_id},
        )

    def _set_object_states(self, obs: KitchenObjectCentricState) -> None:
        for cube_name, cube_id in self._cubes.items():
            set_pose(cube_id, obs.get_object_pose(cube_name), self.physics_client_id)

    def _object_name_to_pybullet_id(self, object_name: str) -> int:
        if object_name == "kitchen":
            return self._kitchen_id
        if object_name.startswith("cube"):
            return self._cubes[object_name]
        raise ValueError(f"Unrecognized object name: {object_name}")

    def _get_collision_object_ids(self) -> set[int]:
        return {self._kitchen_id}

    def _get_movable_object_names(self) -> set[str]:
        return set(self._cubes.keys())

    def _get_surface_object_names(self) -> set[str]:
        return {"kitchen"}

    def _get_half_extents(self, object_name: str) -> tuple[float, float, float]:
        if object_name.startswith("cube"):
            return self.config.block_half_extents
        raise ValueError(f"Unrecognized object name: {object_name}")

    def _get_obs(self) -> KitchenObjectCentricState:
        state_dict = self._create_state_dict(
            [("robot", Kinematic3DRobotType)]
            + [("kitchen", Kinematic3DFixtureType)]
            + [(f"cube{i}", Kinematic3DCuboidType) for i in range(self._num_cubes)]
        )
        state = create_state_from_dict(
            state_dict, Kinematic3DEnvTypeFeatures, state_cls=KitchenObjectCentricState
        )
        assert isinstance(state, KitchenObjectCentricState)
        return state

    def goal_reached(self) -> bool:
        return False


class KitchenEnv(ConstantObjectKinDEREnv):
    """Gym wrapper for ObjectCentricKitchenEnv."""

    def __init__(self, num_cubes: int = 2, **kwargs) -> None:
        self._num_cubes = num_cubes
        super().__init__(num_cubes=num_cubes, **kwargs)

    def _create_object_centric_env(self, *args, **kwargs) -> ObjectCentricKitchenEnv:
        return ObjectCentricKitchenEnv(*args, **kwargs)

    def _get_constant_object_names(self, exemplar_state: ObjectCentricState) -> list[str]:
        constant_objects = ["robot", "kitchen"]
        for obj in exemplar_state:
            if obj.name.startswith("cube"):
                constant_objects.append(obj.name)
        return constant_objects

    def _create_env_markdown_description(self) -> str:
        return "A 3D kitchen environment loaded from a URDF."

    def _create_reward_markdown_description(self) -> str:
        return "No reward defined for this demo."

    def _create_references_markdown_description(self) -> str:
        return ""


# ── Helpers ───────────────────────────────────────────────────────────────────

def compute_grasp_pose(
    robot_arm,
    object_position: tuple[float, float, float],
    approach_height: float = 0.05,
    contact_offset: float = 0.005,
) -> tuple[Pose, Pose]:
    """Compute pre-grasp and grasp poses from the robot arm's end-effector frame.

    The approach direction is the EE's local +Z axis (per the Kinova URDF and the
    pybullet_helpers retract convention). We rotate the current EE orientation so
    that local +Z aligns with world -Z (straight-down, top-down grasp).
    """
    from scipy.spatial.transform import Rotation

    R_ee = np.array(matrix_from_quat(robot_arm.get_end_effector_pose().orientation))
    approach_axis = R_ee[:, 2]  # EE local +Z in world frame

    # Minimal rotation that maps approach_axis → world -Z
    rot, _ = Rotation.align_vectors([[0.0, 0.0, -1.0]], [approach_axis])
    grasp_quat = tuple(Rotation.from_matrix(rot.as_matrix() @ R_ee).as_quat())

    x, y, z = object_position
    return (
        Pose((x, y, z + approach_height), grasp_quat),
        Pose((x, y, z + contact_offset),  grasp_quat),
    )


def _execute_base_plan(environment, base_plan, obs):
    for target_base_pose in base_plan[1:]:
        current_base_pose = obs.base_pose
        delta = target_base_pose - current_base_pose
        action = np.array([delta.x, delta.y, delta.rot] + [0.0] * 7 + [0.0], dtype=np.float32)
        vec_obs, _, _, _, _ = environment.step(action)
        oc_obs = environment.observation_space.devectorize(vec_obs)
        obs = KitchenObjectCentricState(oc_obs.data, oc_obs.type_features)
        time.sleep(STEP_DELAY)
    return obs


def _execute_joint_plan(environment, joint_plan, obs):
    for target_joints in joint_plan[1:]:
        delta = [wrap_angle(a) for a in np.subtract(target_joints[:7], obs.joint_positions)]
        action = np.array([0.0] * 3 + delta + [0.0], dtype=np.float32)
        vec_obs, _, _, _, _ = environment.step(action)
        oc_obs = environment.observation_space.devectorize(vec_obs)
        obs = KitchenObjectCentricState(oc_obs.data, oc_obs.type_features)
        time.sleep(STEP_DELAY)
    return obs


# ── Main ──────────────────────────────────────────────────────────────────────

env = KitchenEnv(num_cubes=2, use_gui=True, render_mode="rgb_array", realistic_bg=False)
vec_obs, _ = env.reset(seed=123)
oc_obs = env.observation_space.devectorize(vec_obs)
obs = KitchenObjectCentricState(oc_obs.data, oc_obs.type_features)

config = env.unwrapped._object_centric_env.config
sim = ObjectCentricKitchenEnv(
    num_cubes=2,
    config=config,
    use_gui=False,
    realistic_bg=False,
    allow_state_access=True,
)
sim.set_state(obs)

kitchen_id = sim._kitchen_id
base_id = sim.robot.base.robot_id
oc_env = env.unwrapped._object_centric_env

# Disable kitchen collision in both physics clients — the headless sim for
# fast planning, and the GUI client so arm execution isn't blocked by mesh geometry.
_num_kitchen_links = p.getNumJoints(kitchen_id, physicsClientId=sim.physics_client_id)
for _link_idx in range(-1, _num_kitchen_links):
    p.setCollisionFilterGroupMask(
        kitchen_id, _link_idx, 0, 0, physicsClientId=sim.physics_client_id
    )
_num_kitchen_links_gui = p.getNumJoints(oc_env._kitchen_id, physicsClientId=oc_env.physics_client_id)
for _link_idx in range(-1, _num_kitchen_links_gui):
    p.setCollisionFilterGroupMask(
        oc_env._kitchen_id, _link_idx, 0, 0, physicsClientId=oc_env.physics_client_id
    )

# Enable gravity for all cubes from the start.  When a cube is grasped its mass
# is set to 0 so the kinematic transport doesn't fight the physics; it is restored
# to 0.1 on release so the cube settles naturally onto whatever surface it lands on.
p.setGravity(0, 0, -9.81, physicsClientId=oc_env.physics_client_id)
for cube_id in oc_env._cubes.values():
    p.changeDynamics(cube_id, -1, mass=0.1, physicsClientId=oc_env.physics_client_id)

# ── Camera controls ────────────────────────────────────────────────────────────
# Start early so controls work even during planning.
from pynput import keyboard, mouse as pynput_mouse  # noqa: E402

CAMERA_SPEED  = 0.05  # m per tick for arrow keys
ROT_SCALE_DEG = 0.3   # degrees per pixel for right-click drag

pressed: set = set()
done = False

# Right-click rotation state (updated by mouse listener, consumed by camera thread).
_rot_lock = threading.Lock()
_pending_rot_px = [0.0, 0.0]  # [dx, dy] accumulated since last camera tick
_rmb_held = False
_last_mouse_xy: tuple[int, int] | None = None


def on_press(key):
    pressed.add(key)
    if key == keyboard.Key.enter:
        global done
        done = True
        return False


def on_release(key):
    pressed.discard(key)


def on_mouse_click(x, y, button, is_pressed):
    global _rmb_held, _last_mouse_xy
    if button == pynput_mouse.Button.right:
        _rmb_held = is_pressed
        _last_mouse_xy = (x, y) if is_pressed else None


def on_mouse_move(x, y):
    global _last_mouse_xy
    if not _rmb_held or _last_mouse_xy is None:
        return
    dx = x - _last_mouse_xy[0]
    dy = y - _last_mouse_xy[1]
    _last_mouse_xy = (x, y)
    with _rot_lock:
        _pending_rot_px[0] += dx
        _pending_rot_px[1] += dy



def _camera_thread():
    """Continuously update the camera based on arrow-key panning and right-click rotation."""
    while not done:
        cam = p.getDebugVisualizerCamera(physicsClientId=oc_env.physics_client_id)
        distance, yaw, pitch, target = cam[10], cam[8], cam[9], list(cam[11])

        yaw_rad = np.deg2rad(yaw)
        forward = np.array([-np.cos(yaw_rad), -np.sin(yaw_rad), 0.0])
        right   = np.array([-np.sin(yaw_rad),  np.cos(yaw_rad), 0.0])

        # Arrow-key pan.
        move = np.zeros(3)
        for k in list(pressed):
            if k == keyboard.Key.up:
                move += right
            elif k == keyboard.Key.down:
                move -= right
            elif k == keyboard.Key.right:
                move -= forward
            elif k == keyboard.Key.left:
                move += forward
        if np.any(move):
            target = [target[i] + CAMERA_SPEED * move[i] / np.linalg.norm(move) for i in range(3)]

        # Right-click drag rotation: rotate around the camera's own position.
        with _rot_lock:
            pdx, pdy = _pending_rot_px
            _pending_rot_px[0] = 0.0
            _pending_rot_px[1] = 0.0

        if pdx or pdy:
            # Current camera position in world space.
            pr = np.deg2rad(pitch)
            yr = np.deg2rad(yaw)
            look = np.array([np.cos(pr) * np.cos(yr),
                             np.cos(pr) * np.sin(yr),
                             -np.sin(pr)])
            cam_pos = np.array(target) + distance * look

            # Apply rotation deltas.
            yaw   += pdx * ROT_SCALE_DEG
            pitch  = float(np.clip(pitch + pdy * ROT_SCALE_DEG, -89, 89))

            # Recompute target so camera position stays fixed.
            new_pr = np.deg2rad(pitch)
            new_yr = np.deg2rad(yaw)
            new_look = np.array([np.cos(new_pr) * np.cos(new_yr),
                                 np.cos(new_pr) * np.sin(new_yr),
                                 -np.sin(new_pr)])
            target = (cam_pos - distance * new_look).tolist()

        if np.any(move) or pdx or pdy:
            p.resetDebugVisualizerCamera(
                distance, yaw, pitch, target,
                physicsClientId=oc_env.physics_client_id,
            )

        time.sleep(1 / 60.0)


keyboard.Listener(on_press=on_press, on_release=on_release).start()
pynput_mouse.Listener(on_click=on_mouse_click, on_move=on_mouse_move).start()
threading.Thread(target=_camera_thread, daemon=True).start()
print("Arrow keys = pan  |  Right-click drag = rotate  |  Enter = close")

# Counter placement constants.
COUNTER_X = 1.3
COUNTER_Y = 1.0
PLACE_Z = COUNTER_SURFACE_Z + config.block_half_extents[2]
# Offset the two cubes side-by-side in x; spacing = 2 * half_extent_x + small gap.
CUBE_SPACING = 2 * config.block_half_extents[0] + 0.04
PLACE_POSITIONS = {
    "cube1": (COUNTER_X - CUBE_SPACING / 2, COUNTER_Y),
    "cube0": (COUNTER_X + CUBE_SPACING / 2, COUNTER_Y),
}


def pick_and_place(cube_name: str, place_x: float, place_y: float, obs, step_offset: int):
    """Pick `cube_name` from the floor and place it at (place_x, place_y) on the counter."""
    joint_distance_fn = create_joint_distance_fn(sim.robot.arm)

    # Move base in front of the cube.
    target_cube_temp = obs.get_object_pose(cube_name).to_se2()
    target_base = SE2Pose(target_cube_temp.x - 0.5, target_cube_temp.y, target_cube_temp.rot)
    base_plan = run_single_arm_mobile_base_motion_planning(
        sim.robot,
        sim.robot.base.get_pose(),
        target_base,
        collision_bodies={kitchen_id},
        seed=123,
    )
    assert base_plan is not None
    obs = _execute_base_plan(env, base_plan, obs)
    print(f"Step {step_offset + 1} done (base → {cube_name}). Base: {obs.base_pose}")

    # Move arm to pre-grasp pose.
    sim.set_state(obs)
    cx, cy, cz = obs.get_object_pose(cube_name).position
    pre_grasp_pose, grasp_pose = compute_grasp_pose(sim.robot.arm, (cx, cy, cz))

    joint_plan = run_smooth_motion_planning_to_pose(
        pre_grasp_pose,
        sim.robot.arm,
        collision_ids={base_id, kitchen_id},
        end_effector_frame_to_plan_frame=Pose.identity(),
        seed=123,
        max_candidate_plans=1,
    )
    joint_plan = remap_joint_position_plan_to_constant_distance(
        joint_plan, sim.robot.arm, max_distance=config.max_action_mag / 2
    )
    obs = _execute_joint_plan(env, joint_plan, obs)
    print(f"Step {step_offset + 2} done (arm → pre-grasp).")

    # Move straight down to grasp.
    sim.set_state(obs)
    joint_plan = smoothly_follow_end_effector_path(
        sim.robot.arm,
        [sim.robot.arm.get_end_effector_pose(), grasp_pose],
        sim.robot.arm.get_joint_positions(),
        collision_ids={kitchen_id, base_id},
        joint_distance_fn=joint_distance_fn,
        max_smoothing_iters_per_step=1,
    )
    assert joint_plan is not None
    joint_plan = remap_joint_position_plan_to_constant_distance(
        joint_plan, sim.robot.arm, max_distance=config.max_action_mag / 2
    )
    obs = _execute_joint_plan(env, joint_plan, obs)
    print(f"Step {step_offset + 3} done (arm → grasp).")

    # Close gripper.
    for _ in range(5):
        action = np.array([0.0] * 3 + [0.0] * 7 + [-1.0], dtype=np.float32)
        vec_obs, _, _, _, _ = env.step(action)
        oc_obs = env.observation_space.devectorize(vec_obs)
        obs = KitchenObjectCentricState(oc_obs.data, oc_obs.type_features)
        time.sleep(STEP_DELAY)
    assert obs.grasped_object == cube_name
    print(f"Step {step_offset + 4} done (grasped {obs.grasped_object}).")

    # Suspend gravity on the held cube so kinematic transport is clean.
    p.changeDynamics(oc_env._cubes[cube_name], -1, mass=0, physicsClientId=oc_env.physics_client_id)

    # Retract arm to home joints.
    sim.set_state(obs)
    joint_plan = run_motion_planning(
        sim.robot.arm,
        sim.robot.arm.get_joint_positions(),
        extend_joints_to_include_fingers(sim.config.initial_joints),
        collision_bodies={kitchen_id, base_id},
        seed=123,
        physics_client_id=sim.physics_client_id,
        held_object=sim._grasped_object_id,
        base_link_to_held_obj=sim._grasped_object_transform,
    )
    joint_plan = remap_joint_position_plan_to_constant_distance(
        joint_plan, sim.robot.arm, max_distance=config.max_action_mag / 2
    )
    obs = _execute_joint_plan(env, joint_plan, obs)
    print(f"Step {step_offset + 5} done (arm retracted).")

    # Move base in front of counter.
    sim.set_state(obs)
    target_counter_base = SE2Pose(COUNTER_X, 0.2, np.pi / 2)
    base_plan = run_single_arm_mobile_base_motion_planning(
        sim.robot,
        sim.robot.base.get_pose(),
        target_counter_base,
        collision_bodies={kitchen_id},
        seed=456,
    )
    assert base_plan is not None
    obs = _execute_base_plan(env, base_plan, obs)
    print(f"Step {step_offset + 6} done (base → counter). Base: {obs.base_pose}")

    # Move arm to pre-place pose.
    sim.set_state(obs)
    pre_place_pose = Pose.from_rpy((place_x, place_y - 0.1, PLACE_Z + 0.1), (-np.pi / 2, np.pi, 0))
    place_pose     = Pose.from_rpy((place_x, place_y,       PLACE_Z + 0.1), (-np.pi / 2, np.pi, 0))

    joint_plan = run_smooth_motion_planning_to_pose(
        pre_place_pose,
        sim.robot.arm,
        collision_ids={kitchen_id, base_id},
        end_effector_frame_to_plan_frame=Pose.identity(),
        seed=123,
        max_time=10,
        max_candidate_plans=10,
        held_object=sim._grasped_object_id,
        base_link_to_held_obj=sim._grasped_object_transform,
    )
    assert joint_plan is not None
    joint_plan = remap_joint_position_plan_to_constant_distance(
        joint_plan, sim.robot.arm, max_distance=config.max_action_mag / 2
    )
    obs = _execute_joint_plan(env, joint_plan, obs)
    print(f"Step {step_offset + 7} done (arm → pre-place).")

    # Lower arm to place pose.
    sim.set_state(obs)
    joint_plan = smoothly_follow_end_effector_path(
        sim.robot.arm,
        [sim.robot.arm.get_end_effector_pose(), place_pose],
        sim.robot.arm.get_joint_positions(),
        collision_ids={kitchen_id, base_id},
        joint_distance_fn=joint_distance_fn,
        max_smoothing_iters_per_step=1,
        held_object=sim._grasped_object_id,
        base_link_to_held_obj=sim._grasped_object_transform,
    )
    assert joint_plan is not None
    joint_plan = remap_joint_position_plan_to_constant_distance(
        joint_plan, sim.robot.arm, max_distance=config.max_action_mag / 2
    )
    obs = _execute_joint_plan(env, joint_plan, obs)
    print(f"Step {step_offset + 8} done (arm → place).")

    # Restore mass so the cube settles under gravity.
    p.changeDynamics(oc_env._cubes[cube_name], -1, mass=0.1, physicsClientId=oc_env.physics_client_id)

    # Release.
    oc_env._grasped_object = None
    oc_env._grasped_object_transform = None
    oc_env.robot.arm.open_fingers()
    print(f"{cube_name} placed on counter at ({place_x:.2f}, {place_y:.2f})!")

    # Let the cube fall and settle before moving on.
    for _ in range(480):  # 2 s at 240 Hz
        p.stepSimulation(physicsClientId=oc_env.physics_client_id)
        time.sleep(1 / 240.0)

    return obs


# obs = pick_and_place("cube1", *PLACE_POSITIONS["cube1"], obs, step_offset=0)
# obs = pick_and_place("cube0", *PLACE_POSITIONS["cube0"], obs, step_offset=9)

# # Retract arm and return base home.
# sim.set_state(obs)
# joint_distance_fn = create_joint_distance_fn(sim.robot.arm)
# joint_plan = run_motion_planning(
#     sim.robot.arm,
#     sim.robot.arm.get_joint_positions(),
#     extend_joints_to_include_fingers(sim.config.initial_joints),
#     collision_bodies={kitchen_id, base_id},
#     seed=123,
#     physics_client_id=sim.physics_client_id,
# )
# joint_plan = remap_joint_position_plan_to_constant_distance(
#     joint_plan, sim.robot.arm, max_distance=config.max_action_mag / 2
# )
# obs = _execute_joint_plan(env, joint_plan, obs)
# print("Arm retracted.")

# sim.set_state(obs)
# home_base = config.robot_base_home_pose
# base_plan = run_single_arm_mobile_base_motion_planning(
#     sim.robot,
#     sim.robot.base.get_pose(),
#     home_base,
#     collision_bodies={kitchen_id},
#     seed=789,
# )
# assert base_plan is not None
# obs = _execute_base_plan(env, base_plan, obs)
# print(f"Home. Base: {obs.base_pose}")

# ── Cabinet-door opening helper ──────────────────────────────────────────────

def open_lower_cabinet_door(door_joint_name: str, handle_joint_name: str, obs,
                            yaw_sign: float = 1.0):
    """Open a lower-cabinet door by name.

    Parameters
    ----------
    door_joint_name  : URDF joint that rotates the door   (e.g. "cab_6_d_2")
    handle_joint_name: URDF joint/link for the handle     (e.g. "cab_6_h_2")
    obs              : current observation (updated through base planning)
    yaw_sign         : +1 or -1; sign of gripper yaw rotation relative to door
                       angle.  Doors with hinge on the right use +1; hinge on
                       the left use -1.

    Returns
    -------
    obs after the base has been driven in front of the cabinet.
    (Subsequent steps use direct PyBullet calls and do not update obs.)
    """
    _pc_id = oc_env.physics_client_id
    _kg_id = oc_env._kitchen_id
    _CAB_OPEN_ANGLE = 1.5   # radians — joint upper limit
    _GRASP_QUAT = Pose.from_rpy((0, 0, 0), (np.pi, 0, 0)).orientation
    _N_OPEN = 80

    # ── Locate joints by name ────────────────────────────────────────────────
    _num_j = p.getNumJoints(_kg_id, physicsClientId=_pc_id)
    _cab_joint_idx = next(
        i for i in range(_num_j)
        if p.getJointInfo(_kg_id, i, physicsClientId=_pc_id)[1].decode()
        == door_joint_name
    )
    _handle_link_idx = next(
        i for i in range(_num_j)
        if p.getJointInfo(_kg_id, i, physicsClientId=_pc_id)[1].decode()
        == handle_joint_name
    )

    # ── Closed-door handle position ──────────────────────────────────────────
    p.resetJointState(_kg_id, _cab_joint_idx, 0.0, 0.0, physicsClientId=_pc_id)
    _ls_closed = p.getLinkState(_kg_id, _handle_link_idx,
                                computeForwardKinematics=True,
                                physicsClientId=_pc_id)
    _handle_closed = tuple(_ls_closed[4])
    print(f"[{door_joint_name}] handle (closed): {_handle_closed}")

    # ── Step a: drive base in front of the cabinet ───────────────────────────
    _base_target_x = _handle_closed[0]
    sim.set_state(obs)
    _base_plan = run_single_arm_mobile_base_motion_planning(
        sim.robot, sim.robot.base.get_pose(),
        SE2Pose(_base_target_x, 0.4, np.pi / 2),
        collision_bodies={kitchen_id}, seed=999,
    )
    assert _base_plan is not None, f"Base plan failed for {door_joint_name}"
    obs = _execute_base_plan(env, _base_plan, obs)
    print(f"[{door_joint_name}] step a done (base → cabinet).")

    # ── Step b: arm to pre-grasp (15 cm above handle) ────────────────────────
    _pre_grasp_pos = np.array(_handle_closed) + np.array([0.0, 0.0, 0.15])
    _pre_grasp_pose = Pose(tuple(_pre_grasp_pos), _GRASP_QUAT)

    sim.set_state(obs)
    _joint_plan = None
    for _seed in range(10):
        _joint_plan = run_smooth_motion_planning_to_pose(
            _pre_grasp_pose, sim.robot.arm,
            collision_ids=set(),
            end_effector_frame_to_plan_frame=Pose.identity(),
            seed=_seed, max_time=10, max_candidate_plans=5,
        )
        if _joint_plan is not None:
            print(f"  pre-grasp found (seed={_seed})")
            break
    assert _joint_plan is not None, \
        f"Motion planning failed for pre-grasp above {handle_joint_name}"
    _joint_plan = remap_joint_position_plan_to_constant_distance(
        _joint_plan, sim.robot.arm, max_distance=config.max_action_mag / 2
    )
    obs = _execute_joint_plan(env, _joint_plan, obs)
    print(f"[{door_joint_name}] step b done (arm above handle).")

    # ── Step c: arm descends to handle (15 cm straight down from pre-grasp) ──
    # Plan from pre-grasp to the handle — a trivial short motion with no
    # obstacles (collision_ids=set()), so seed=0 almost always works.
    sim.set_state(obs)
    _joint_plan_c = None
    for _seed in range(10):
        _joint_plan_c = run_smooth_motion_planning_to_pose(
            Pose(_handle_closed, _GRASP_QUAT), sim.robot.arm,
            collision_ids=set(),
            end_effector_frame_to_plan_frame=Pose.identity(),
            seed=_seed, max_time=10, max_candidate_plans=5,
        )
        if _joint_plan_c is not None:
            print(f"  descent found (seed={_seed})")
            break
    assert _joint_plan_c is not None, \
        f"Motion planning failed for descent to {handle_joint_name}"
    _joint_plan_c = remap_joint_position_plan_to_constant_distance(
        _joint_plan_c, sim.robot.arm, max_distance=config.max_action_mag / 2
    )
    obs = _execute_joint_plan(env, _joint_plan_c, obs)
    print(f"[{door_joint_name}] step c done (EE at handle).")

    # ── Gradually close fingers ───────────────────────────────────────────────
    _arm_gui = oc_env.robot.arm
    _closed_fingers = _arm_gui.finger_state_to_joints(_arm_gui.closed_fingers_state)
    _N_CLOSE = 20
    for _fi in range(_N_CLOSE):
        _frac = (_fi + 1) / _N_CLOSE
        for _fname, _fpos in zip(_arm_gui.finger_joint_names, _closed_fingers):
            p.resetJointState(_arm_gui.robot_id, _arm_gui.joint_from_name(_fname),
                              _fpos * _frac, physicsClientId=_pc_id)
        time.sleep(STEP_DELAY)
    print(f"[{door_joint_name}] gripper closed.")

    # ── Compute base end position from fully-open handle world pos ───────────
    p.resetJointState(_kg_id, _cab_joint_idx, _CAB_OPEN_ANGLE, 0.0,
                      physicsClientId=_pc_id)
    _ls_end = p.getLinkState(_kg_id, _handle_link_idx,
                             computeForwardKinematics=True, physicsClientId=_pc_id)
    _base_end_x = _ls_end[4][0]
    _base_end_y = 0.1
    p.resetJointState(_kg_id, _cab_joint_idx, 0.0, 0.0, physicsClientId=_pc_id)

    # ── Door-opening loop ─────────────────────────────────────────────────────
    for _i in range(_N_OPEN):
        _t = (_i + 1) / _N_OPEN
        _door_angle = _CAB_OPEN_ANGLE * _t
        _bx = _base_target_x + (_base_end_x - _base_target_x) * _t
        _by = 0.4 + (_base_end_y - 0.4) * _t

        p.resetJointState(_kg_id, _cab_joint_idx, _door_angle, 0.0,
                          physicsClientId=_pc_id)
        oc_env.robot.set_base(SE2Pose(_bx, _by, np.pi / 2))

        _ls = p.getLinkState(_kg_id, _handle_link_idx,
                             computeForwardKinematics=True, physicsClientId=_pc_id)
        _handle_now = _ls[4]

        # Rotate gripper yaw with the door so it stays parallel to the surface.
        _tracking_quat = Pose.from_rpy(
            (0, 0, 0), (np.pi, 0, yaw_sign * _door_angle)
        ).orientation
        _ik = p.calculateInverseKinematics(
            _arm_gui.robot_id, _arm_gui.end_effector_id,
            _handle_now, _tracking_quat,
            physicsClientId=_pc_id,
        )
        _arm_gui.set_joints(list(_ik[: len(_arm_gui.arm_joints)]))

        for _fn, _fp in zip(_arm_gui.finger_joint_names, _closed_fingers):
            p.resetJointState(_arm_gui.robot_id, _arm_gui.joint_from_name(_fn),
                              _fp, physicsClientId=_pc_id)
        time.sleep(STEP_DELAY)

    print(f"[{door_joint_name}] door fully open!")
    return obs


# ── Step 12: Open both lower cabinet doors ───────────────────────────────────
obs = open_lower_cabinet_door("cab_6_d_2", "cab_6_h_2", obs, yaw_sign=+1.0)
obs = open_lower_cabinet_door("cab_6_d_1", "cab_6_h_1", obs, yaw_sign=-1.0)

while not done:
    p.stepSimulation(physicsClientId=oc_env.physics_client_id)
    time.sleep(1 / 240.0)

env.close()
sim.close()
