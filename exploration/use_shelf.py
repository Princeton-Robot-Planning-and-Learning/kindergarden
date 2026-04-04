"""Pick-and-place demo: picks cube0 and places it on shelf layer 1.

Adapted directly from tests/envs/kinematic3d/test_shelf3d.py::test_pick_place.

Usage:
    python exploration/use_shelf.py [--delay SECS] [--layer N]
"""
import argparse
import time

import numpy as np
from prpl_utils.utils import wrap_angle
from pybullet_helpers.geometry import Pose, SE2Pose
from pybullet_helpers.motion_planning import (
    create_joint_distance_fn,
    remap_joint_position_plan_to_constant_distance,
    run_motion_planning,
    run_single_arm_mobile_base_motion_planning,
    run_smooth_motion_planning_to_pose,
    smoothly_follow_end_effector_path,
)

from kinder.envs.kinematic3d.shelf3d import (
    ObjectCentricShelf3DEnv,
    Shelf3DEnv,
    Shelf3DObjectCentricState,
)
from kinder.envs.kinematic3d.utils import extend_joints_to_include_fingers

parser = argparse.ArgumentParser()
parser.add_argument("--delay", type=float, default=0.05, help="Seconds to sleep between steps")
parser.add_argument("--layer", type=int, default=1, help="Shelf layer to place block on (1 = lowest)")
args = parser.parse_args()

STEP_DELAY = args.delay


def _execute_base_plan(environment, base_plan, obs):
    for target_base_pose in base_plan[1:]:
        current_base_pose = obs.base_pose
        delta = target_base_pose - current_base_pose
        delta_lst = [delta.x, delta.y, delta.rot]
        action_lst = delta_lst + [0.0] * 7 + [0.0]
        action = np.array(action_lst, dtype=np.float32)
        vec_obs, _, _, _, _ = environment.step(action)
        oc_obs = environment.observation_space.devectorize(vec_obs)
        obs = Shelf3DObjectCentricState(oc_obs.data, oc_obs.type_features)
        time.sleep(STEP_DELAY)
    return obs


def _execute_joint_plan(environment, joint_plan, obs):
    for target_joints in joint_plan[1:]:
        delta = np.subtract(target_joints[:7], obs.joint_positions)
        delta_lst = [wrap_angle(a) for a in delta]
        action_lst = [0.0] * 3 + delta_lst + [0.0]
        action = np.array(action_lst, dtype=np.float32)
        vec_obs, _, _, _, _ = environment.step(action)
        oc_obs = environment.observation_space.devectorize(vec_obs)
        obs = Shelf3DObjectCentricState(oc_obs.data, oc_obs.type_features)
        time.sleep(STEP_DELAY)
    return obs


env = Shelf3DEnv(num_cubes=2, use_gui=True, render_mode="rgb_array", realistic_bg=False)
vec_obs, _ = env.reset(seed=123)
oc_obs = env.observation_space.devectorize(vec_obs)
obs = Shelf3DObjectCentricState(oc_obs.data, oc_obs.type_features)

config = env.unwrapped._object_centric_env.config
sim = ObjectCentricShelf3DEnv(
    num_cubes=2,
    config=config,
    use_gui=False,
    realistic_bg=False,
    allow_state_access=True,
)
sim.set_state(obs)

shelf_id = sim._shelf_id
base_id = sim.robot.base.robot_id

# Step 1: Move the base in front of cube0
target_object_pose_temp = obs.get_object_pose("cube0").to_se2()
target_object_pose = SE2Pose(
    target_object_pose_temp.x - 0.5,
    target_object_pose_temp.y,
    target_object_pose_temp.rot,
)
base_plan = run_single_arm_mobile_base_motion_planning(
    sim.robot,
    sim.robot.base.get_pose(),
    target_object_pose,
    collision_bodies={shelf_id},
    seed=123,
)
assert base_plan is not None
obs = _execute_base_plan(env, base_plan, obs)

# Step 2: Move arm to pre-grasp pose and then to grasp pose
sim.set_state(obs)
x, y, z = obs.get_object_pose("cube0").position
dz = 0.05
pre_grasp_pose = Pose.from_rpy((x, y, z + dz), (np.pi, 0, np.pi / 2))
grasp_pose = Pose.from_rpy((x, y, z + 0.005), (np.pi, 0, np.pi / 2))

joint_distance_fn = create_joint_distance_fn(sim.robot.arm)
joint_plan = run_smooth_motion_planning_to_pose(
    pre_grasp_pose,
    sim.robot.arm,
    collision_ids={base_id, shelf_id},
    end_effector_frame_to_plan_frame=Pose.identity(),
    seed=123,
    max_candidate_plans=1,
)
joint_plan = remap_joint_position_plan_to_constant_distance(
    joint_plan, sim.robot.arm, max_distance=config.max_action_mag / 2
)
obs = _execute_joint_plan(env, joint_plan, obs)

# Step 3: Move down to grasp cube0
sim.set_state(obs)
joint_plan = smoothly_follow_end_effector_path(
    sim.robot.arm,
    [sim.robot.arm.get_end_effector_pose(), grasp_pose],
    sim.robot.arm.get_joint_positions(),
    collision_ids={shelf_id, base_id},
    joint_distance_fn=joint_distance_fn,
    max_smoothing_iters_per_step=1,
)
assert joint_plan is not None
joint_plan = remap_joint_position_plan_to_constant_distance(
    joint_plan, sim.robot.arm, max_distance=config.max_action_mag / 2
)
obs = _execute_joint_plan(env, joint_plan, obs)

# Step 4: Close the gripper to grasp cube0
for _ in range(5):
    action = np.array([0.0] * 3 + [0.0] * 7 + [-1.0], dtype=np.float32)
    vec_obs, _, _, _, _ = env.step(action)
    oc_obs = env.observation_space.devectorize(vec_obs)
    obs = Shelf3DObjectCentricState(oc_obs.data, oc_obs.type_features)
    time.sleep(STEP_DELAY)

assert obs.grasped_object == "cube0"
print(f"Grasped: {obs.grasped_object}")

# Step 5: Retract the arm
sim.set_state(obs)
joint_plan = run_motion_planning(
    sim.robot.arm,
    sim.robot.arm.get_joint_positions(),
    extend_joints_to_include_fingers(sim.config.initial_joints),
    collision_bodies={shelf_id, base_id},
    seed=123,
    physics_client_id=sim.physics_client_id,
    held_object=sim._grasped_object_id,
    base_link_to_held_obj=sim._grasped_object_transform,
)
joint_plan = remap_joint_position_plan_to_constant_distance(
    joint_plan, sim.robot.arm, max_distance=config.max_action_mag / 2
)
obs = _execute_joint_plan(env, joint_plan, obs)

# Step 6: Move the base in front of the shelf
sim.set_state(obs)
shelf_pose = obs.get_object_pose("shelf")
target_shelf_base_pose = SE2Pose(
    shelf_pose.position[0],
    shelf_pose.position[1] - 0.8,
    np.pi / 2,
)
base_plan = run_single_arm_mobile_base_motion_planning(
    sim.robot,
    sim.robot.base.get_pose(),
    target_shelf_base_pose,
    collision_bodies={shelf_id},
    seed=456,
)
assert base_plan is not None
obs = _execute_base_plan(env, base_plan, obs)

# Step 7: Move arm to pre-place pose at the chosen shelf layer
sim.set_state(obs)
place_x = shelf_pose.position[0]
place_y = shelf_pose.position[1] - 0.05
place_z = (
    args.layer * (config.shelf_spacing + 0.035)
    + config.shelf_height / 2
    + config.block_half_extents[0]
)
pre_place_pose = Pose.from_rpy((place_x, place_y - 0.1, place_z), (-np.pi / 2, np.pi, 0))
place_pose = Pose.from_rpy((place_x, place_y, place_z), (-np.pi / 2, np.pi, 0))

joint_plan = run_smooth_motion_planning_to_pose(
    pre_place_pose,
    sim.robot.arm,
    collision_ids={base_id, shelf_id},
    end_effector_frame_to_plan_frame=Pose.identity(),
    seed=123,
    max_candidate_plans=1,
    held_object=sim._grasped_object_id,
    base_link_to_held_obj=sim._grasped_object_transform,
)
assert joint_plan is not None
joint_plan = remap_joint_position_plan_to_constant_distance(
    joint_plan, sim.robot.arm, max_distance=config.max_action_mag / 2
)
obs = _execute_joint_plan(env, joint_plan, obs)

# Step 8: Move to place pose
sim.set_state(obs)
joint_plan = smoothly_follow_end_effector_path(
    sim.robot.arm,
    [sim.robot.arm.get_end_effector_pose(), place_pose],
    sim.robot.arm.get_joint_positions(),
    collision_ids={shelf_id, base_id},
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

# Enable gravity and give the cube mass so it falls naturally after release
import pybullet as p
oc_env = env.unwrapped._object_centric_env
p.setGravity(0, 0, -9.81, physicsClientId=oc_env.physics_client_id)
p.changeDynamics(oc_env._cubes["cube0"], -1, mass=0.1, physicsClientId=oc_env.physics_client_id)

# Step 9: Force-release cube0 by directly clearing the grasp state
oc_env._grasped_object = None
oc_env._grasped_object_transform = None
oc_env.robot.arm.open_fingers()
print(f"Block placed on shelf layer {args.layer}!")

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
    time.sleep(1 / 240.0)  # PyBullet default timestep

listener.stop()
env.close()
sim.close()
