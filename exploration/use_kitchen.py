"""Kitchen environment demo: picks cube1 and places it on the kitchen counter.

Usage:
    python exploration/use_kitchen.py
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from typing import Type as TypingType

import numpy as np
import pybullet as p
from prpl_utils.utils import wrap_angle
from pybullet_helpers.geometry import Pose, SE2Pose, set_pose
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

KITCHEN_URDF = Path(__file__).parent / "unnamed" / "urdf" / "unnamed.urdf"
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
counter_id = sim._counter_id
base_id = sim.robot.base.robot_id

# Teleport cube0 to the target landing spot so it's visible for debugging.
oc_env = env.unwrapped._object_centric_env
# target_x = 1.3
# target_y = 1.0  # counter_place_y
# target_z = COUNTER_SURFACE_Z + config.block_half_extents[2]  # resting on surface
# p.resetBasePositionAndOrientation(
#     oc_env._cubes["cube0"],
#     [target_x, target_y, target_z],
#     [0, 0, 0, 1],
#     physicsClientId=oc_env.physics_client_id,
# )

# Step 1: Move base in front of cube1.
target_cube_temp = obs.get_object_pose("cube1").to_se2()
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
print(f"Step 1 done. Base: {obs.base_pose}")

# Step 2: Move arm to pre-grasp pose above cube1.
sim.set_state(obs)
x, y, z = obs.get_object_pose("cube1").position
pre_grasp_pose = Pose.from_rpy((x, y, z + 0.05), (np.pi, 0, np.pi / 2))
grasp_pose     = Pose.from_rpy((x, y, z + 0.005), (np.pi, 0, np.pi / 2))

joint_distance_fn = create_joint_distance_fn(sim.robot.arm)
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
print("Step 2 done.")

# Step 3: Move arm straight down to grasp cube1.
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
print("Step 3 done.")

# Step 4: Close gripper.
for _ in range(5):
    action = np.array([0.0] * 3 + [0.0] * 7 + [-1.0], dtype=np.float32)
    vec_obs, _, _, _, _ = env.step(action)
    oc_obs = env.observation_space.devectorize(vec_obs)
    obs = KitchenObjectCentricState(oc_obs.data, oc_obs.type_features)
    time.sleep(STEP_DELAY)

assert obs.grasped_object == "cube1"
print(f"Grasped: {obs.grasped_object}")

# Step 5: Retract arm to home.
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
print("Step 5 done.")

# Step 6: Move base to in front of the counter.
sim.set_state(obs)
counter_x = 1.3
counter_place_y = 1.0
target_counter_base = SE2Pose(counter_x, 0.2, np.pi / 2)
base_plan = run_single_arm_mobile_base_motion_planning(
    sim.robot,
    sim.robot.base.get_pose(),
    target_counter_base,
    collision_bodies={kitchen_id},
    seed=456,
)
assert base_plan is not None
obs = _execute_base_plan(env, base_plan, obs)
print(f"Step 6 done. Base: {obs.base_pose}")

# Step 7: Move arm to pre-place pose above the counter surface.
sim.set_state(obs)
place_z = COUNTER_SURFACE_Z + config.block_half_extents[2]
pre_place_pose = Pose.from_rpy((counter_x, counter_place_y - 0.1, place_z), (-np.pi / 2, np.pi, 0))
place_pose     = Pose.from_rpy((counter_x, counter_place_y,       place_z), (-np.pi / 2, np.pi, 0))

joint_plan = run_smooth_motion_planning_to_pose(
    pre_place_pose,
    sim.robot.arm,
    collision_ids={kitchen_id, base_id},
    end_effector_frame_to_plan_frame=Pose.identity(),
    seed=123,
    max_time=10,
    max_candidate_plans=10,
)
print("7...")
assert joint_plan is not None
joint_plan = remap_joint_position_plan_to_constant_distance(
    joint_plan, sim.robot.arm, max_distance=config.max_action_mag / 2
)
obs = _execute_joint_plan(env, joint_plan, obs)
print("Step 7 done.")

# Step 8: Lower arm to place pose.
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
print("8...")
assert joint_plan is not None
joint_plan = remap_joint_position_plan_to_constant_distance(
    joint_plan, sim.robot.arm, max_distance=config.max_action_mag / 2
)
obs = _execute_joint_plan(env, joint_plan, obs)
print("Step 8 done.")

# Enable gravity and give cube1 mass so it settles onto the counter.
p.setGravity(0, 0, -9.81, physicsClientId=oc_env.physics_client_id)
p.changeDynamics(oc_env._cubes["cube1"], -1, mass=0.1, physicsClientId=oc_env.physics_client_id)

# Step 9: Release cube1.
oc_env._grasped_object = None
oc_env._grasped_object_transform = None
oc_env.robot.arm.open_fingers()
print("cube1 placed on the kitchen counter!")

# Step 10: Retract arm to home joints.
sim.set_state(obs)
joint_plan = run_motion_planning(
    sim.robot.arm,
    sim.robot.arm.get_joint_positions(),
    extend_joints_to_include_fingers(sim.config.initial_joints),
    collision_bodies={kitchen_id, base_id},
    seed=123,
    physics_client_id=sim.physics_client_id,
)
joint_plan = remap_joint_position_plan_to_constant_distance(
    joint_plan, sim.robot.arm, max_distance=config.max_action_mag / 2
)
obs = _execute_joint_plan(env, joint_plan, obs)
print("Step 10 done.")

# Step 11: Drive base back to home pose.
sim.set_state(obs)
home_base = config.robot_base_home_pose
base_plan = run_single_arm_mobile_base_motion_planning(
    sim.robot,
    sim.robot.base.get_pose(),
    home_base,
    collision_bodies={kitchen_id},
    seed=789,
)
assert base_plan is not None
obs = _execute_base_plan(env, base_plan, obs)
print(f"Step 11 done. Base: {obs.base_pose}")

from pynput import keyboard

done = False


def on_press(key):
    global done
    if key == keyboard.Key.enter:
        done = True
        return False  # stop listener


print("Press Enter to close...")
listener = keyboard.Listener(on_press=on_press)
listener.start()

while not done:
    p.stepSimulation(physicsClientId=oc_env.physics_client_id)
    time.sleep(1 / 240.0)

listener.stop()
env.close()
sim.close()
