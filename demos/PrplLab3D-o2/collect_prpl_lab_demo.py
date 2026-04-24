"""Collect a scripted demonstration for PrplLab3D.

Runs a headless pick-and-place demo (floor → counter) for kinder/PrplLab3D-o2-v0
and saves the result in the standard demos/ format so generate_demo_video.py can
replay it into a GIF.

Usage (from repo root):
    python scripts/collect_prpl_lab_demo.py
    python scripts/collect_prpl_lab_demo.py --seed 0 --demo-dir demos
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Allow imports from scripts/ directory (for generate_env_docs / sanitize_env_id).
sys.path.insert(0, str(Path(__file__).parent))

import dill as pkl
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

import kinder
from kinder.envs.kinematic3d.prpl3d import (
    ObjectCentricPrplLab3DEnv,
    PrplLab3DEnv,
    PrplLab3DObjectCentricState,
)
from kinder.envs.kinematic3d.utils import extend_joints_to_include_fingers

ENV_ID = "kinder/PrplLab3D-o2-v0"


# ── Execution helpers ──────────────────────────────────────────────────────────

def _execute_base_plan(environment, base_plan, obs, actions, rewards, observations):
    for target_base_pose in base_plan[1:]:
        delta = target_base_pose - obs.base_pose
        action = np.array(
            [delta.x, delta.y, delta.rot] + [0.0] * 7 + [0.0], dtype=np.float32
        )
        vec_obs, reward, _, _, _ = environment.step(action)
        oc_obs = environment.observation_space.devectorize(vec_obs)
        obs = PrplLab3DObjectCentricState(oc_obs.data, oc_obs.type_features)
        actions.append(action)
        rewards.append(float(reward))
        observations.append(vec_obs)
    return obs


def _execute_joint_plan(environment, joint_plan, obs, actions, rewards, observations):
    for target_joints in joint_plan[1:]:
        delta = [
            wrap_angle(a)
            for a in np.subtract(target_joints[:7], obs.joint_positions)
        ]
        action = np.array([0.0] * 3 + delta + [0.0], dtype=np.float32)
        vec_obs, reward, _, _, _ = environment.step(action)
        oc_obs = environment.observation_space.devectorize(vec_obs)
        obs = PrplLab3DObjectCentricState(oc_obs.data, oc_obs.type_features)
        actions.append(action)
        rewards.append(float(reward))
        observations.append(vec_obs)
    return obs


# ── Main demo logic ────────────────────────────────────────────────────────────

def collect_demo(seed: int = 0, demo_dir: Path = Path("demos")) -> Path:
    kinder.register_all_environments()

    env = PrplLab3DEnv(
        num_cubes=2, use_gui=False, render_mode="rgb_array", realistic_bg=False
    )
    vec_obs, _ = env.reset(seed=seed)
    oc_obs = env.observation_space.devectorize(vec_obs)
    obs = PrplLab3DObjectCentricState(oc_obs.data, oc_obs.type_features)

    config = env.unwrapped._object_centric_env.config
    sim = ObjectCentricPrplLab3DEnv(
        num_cubes=2,
        config=config,
        use_gui=False,
        realistic_bg=False,
        allow_state_access=True,
    )
    sim.set_state(obs)

    observations = [vec_obs]
    actions: list = []
    rewards: list[float] = []

    joint_distance_fn = create_joint_distance_fn(sim.robot.arm)

    # ── Step 1: base → cube1 ──────────────────────────────────────────────────
    print("Step 1: base → cube1")
    cube_se2 = obs.get_object_pose("cube1").to_se2()
    target_base = SE2Pose(cube_se2.x, cube_se2.y - 0.5, np.pi / 2)
    base_plan = None
    for s in range(20):
        base_plan = run_single_arm_mobile_base_motion_planning(
            sim.robot, sim.robot.base.get_pose(), target_base,
            collision_bodies={sim._lab_id}, seed=s,
        )
        if base_plan is not None:
            break
    assert base_plan is not None, "Base plan to cube1 failed"
    obs = _execute_base_plan(env, base_plan, obs, actions, rewards, observations)

    # ── Step 2: arm → pre-grasp ───────────────────────────────────────────────
    print("Step 2: arm → pre-grasp")
    sim.set_state(obs)
    x, y, z = obs.get_object_pose("cube1").position
    pre_grasp_pose = Pose.from_rpy((x, y, z + 0.05), (np.pi, 0, np.pi / 2))
    grasp_pose = Pose.from_rpy((x, y, z + 0.005), (np.pi, 0, np.pi / 2))

    joint_plan = run_smooth_motion_planning_to_pose(
        pre_grasp_pose, sim.robot.arm,
        collision_ids={sim._lab_id},
        end_effector_frame_to_plan_frame=Pose.identity(),
        seed=123, max_candidate_plans=5,
    )
    joint_plan = remap_joint_position_plan_to_constant_distance(
        joint_plan, sim.robot.arm, max_distance=config.max_action_mag / 2
    )
    obs = _execute_joint_plan(env, joint_plan, obs, actions, rewards, observations)

    # ── Step 3: arm → grasp ───────────────────────────────────────────────────
    print("Step 3: arm → grasp")
    sim.set_state(obs)
    joint_plan = smoothly_follow_end_effector_path(
        sim.robot.arm,
        [sim.robot.arm.get_end_effector_pose(), grasp_pose],
        sim.robot.arm.get_joint_positions(),
        collision_ids={sim._lab_id},
        joint_distance_fn=joint_distance_fn,
        max_smoothing_iters_per_step=1,
    )
    assert joint_plan is not None
    joint_plan = remap_joint_position_plan_to_constant_distance(
        joint_plan, sim.robot.arm, max_distance=config.max_action_mag / 2
    )
    obs = _execute_joint_plan(env, joint_plan, obs, actions, rewards, observations)

    # ── Step 4: close gripper ─────────────────────────────────────────────────
    print("Step 4: close gripper")
    for _ in range(5):
        action = np.array([0.0] * 3 + [0.0] * 7 + [-1.0], dtype=np.float32)
        vec_obs, reward, _, _, _ = env.step(action)
        oc_obs = env.observation_space.devectorize(vec_obs)
        obs = PrplLab3DObjectCentricState(oc_obs.data, oc_obs.type_features)
        actions.append(action)
        rewards.append(float(reward))
        observations.append(vec_obs)
    assert obs.grasped_object == "cube1", f"Grasp failed: {obs.grasped_object}"

    # ── Step 5: retract arm ───────────────────────────────────────────────────
    print("Step 5: retract arm")
    sim.set_state(obs)
    joint_plan = run_motion_planning(
        sim.robot.arm,
        sim.robot.arm.get_joint_positions(),
        extend_joints_to_include_fingers(sim.config.initial_joints),
        collision_bodies={sim._lab_id},
        seed=123,
        physics_client_id=sim.physics_client_id,
        held_object=sim._grasped_object_id,
        base_link_to_held_obj=sim._grasped_object_transform,
    )
    joint_plan = remap_joint_position_plan_to_constant_distance(
        joint_plan, sim.robot.arm, max_distance=config.max_action_mag / 2
    )
    obs = _execute_joint_plan(env, joint_plan, obs, actions, rewards, observations)

    # ── Step 6: base → counter ────────────────────────────────────────────────
    print("Step 6: base → counter")
    sim.set_state(obs)
    counter_x = 0.3
    target_counter_base = SE2Pose(counter_x, 0.7, np.pi / 2)
    base_plan = None
    for s in range(10):
        base_plan = run_single_arm_mobile_base_motion_planning(
            sim.robot, sim.robot.base.get_pose(), target_counter_base,
            collision_bodies={sim._lab_id}, seed=456 + s,
        )
        if base_plan is not None:
            break
    assert base_plan is not None, "Base plan to counter failed"
    obs = _execute_base_plan(env, base_plan, obs, actions, rewards, observations)

    # ── Step 7: arm → place on counter ───────────────────────────────────────
    print("Step 7: arm → place")
    place_rpy = (-np.pi / 2, np.pi, 0)
    sim.set_state(obs)

    # Release above the counter — goal_reached checks z > 0.5, not surface contact.
    place_pose = Pose.from_rpy((counter_x, 1.6, 0.85), place_rpy)
    joint_plan = None
    for s in range(20):
        joint_plan = run_smooth_motion_planning_to_pose(
            place_pose, sim.robot.arm,
            collision_ids={sim._lab_id},
            end_effector_frame_to_plan_frame=Pose.identity(),
            seed=s, max_time=5, max_candidate_plans=5,
            held_object=sim._grasped_object_id,
            base_link_to_held_obj=sim._grasped_object_transform,
        )
        if joint_plan is not None:
            break
    assert joint_plan is not None, "Place plan failed"
    joint_plan = remap_joint_position_plan_to_constant_distance(
        joint_plan, sim.robot.arm, max_distance=config.max_action_mag / 2
    )
    obs = _execute_joint_plan(env, joint_plan, obs, actions, rewards, observations)

    # ── Step 8: open gripper to release cube1 ─────────────────────────────────
    print("Step 8: open gripper")
    for _ in range(5):
        action = np.array([0.0] * 3 + [0.0] * 7 + [1.0], dtype=np.float32)
        vec_obs, reward, _, _, _ = env.step(action)
        oc_obs = env.observation_space.devectorize(vec_obs)
        obs = PrplLab3DObjectCentricState(oc_obs.data, oc_obs.type_features)
        actions.append(action)
        rewards.append(float(reward))
        observations.append(vec_obs)
    assert obs.grasped_object is None, f"cube1 release failed: {obs.grasped_object}"

    # ── Steps 9-16: pick cube0 and place next to cube1 ────────────────────────

    # ── Step 9: base → cube0 ─────────────────────────────────────────────────
    print("Step 9: base → cube0")
    sim.set_state(obs)
    cube_se2 = obs.get_object_pose("cube0").to_se2()
    target_base = SE2Pose(cube_se2.x, cube_se2.y - 0.5, np.pi / 2)
    base_plan = None
    for s in range(20):
        base_plan = run_single_arm_mobile_base_motion_planning(
            sim.robot, sim.robot.base.get_pose(), target_base,
            collision_bodies={sim._lab_id}, seed=s,
        )
        if base_plan is not None:
            break
    assert base_plan is not None, "Base plan to cube0 failed"
    obs = _execute_base_plan(env, base_plan, obs, actions, rewards, observations)

    # ── Step 10: arm → pre-grasp ─────────────────────────────────────────────
    print("Step 10: arm → pre-grasp")
    sim.set_state(obs)
    x, y, z = obs.get_object_pose("cube0").position
    pre_grasp_pose = Pose.from_rpy((x, y, z + 0.05), (np.pi, 0, np.pi / 2))
    grasp_pose = Pose.from_rpy((x, y, z + 0.005), (np.pi, 0, np.pi / 2))

    joint_plan = run_smooth_motion_planning_to_pose(
        pre_grasp_pose, sim.robot.arm,
        collision_ids={sim._lab_id},
        end_effector_frame_to_plan_frame=Pose.identity(),
        seed=123, max_candidate_plans=5,
    )
    joint_plan = remap_joint_position_plan_to_constant_distance(
        joint_plan, sim.robot.arm, max_distance=config.max_action_mag / 2
    )
    obs = _execute_joint_plan(env, joint_plan, obs, actions, rewards, observations)

    # ── Step 11: arm → grasp ─────────────────────────────────────────────────
    print("Step 11: arm → grasp")
    sim.set_state(obs)
    joint_plan = smoothly_follow_end_effector_path(
        sim.robot.arm,
        [sim.robot.arm.get_end_effector_pose(), grasp_pose],
        sim.robot.arm.get_joint_positions(),
        collision_ids={sim._lab_id},
        joint_distance_fn=joint_distance_fn,
        max_smoothing_iters_per_step=1,
    )
    assert joint_plan is not None
    joint_plan = remap_joint_position_plan_to_constant_distance(
        joint_plan, sim.robot.arm, max_distance=config.max_action_mag / 2
    )
    obs = _execute_joint_plan(env, joint_plan, obs, actions, rewards, observations)

    # ── Step 12: close gripper ────────────────────────────────────────────────
    print("Step 12: close gripper")
    for _ in range(5):
        action = np.array([0.0] * 3 + [0.0] * 7 + [-1.0], dtype=np.float32)
        vec_obs, reward, _, _, _ = env.step(action)
        oc_obs = env.observation_space.devectorize(vec_obs)
        obs = PrplLab3DObjectCentricState(oc_obs.data, oc_obs.type_features)
        actions.append(action)
        rewards.append(float(reward))
        observations.append(vec_obs)
    assert obs.grasped_object == "cube0", f"Grasp failed: {obs.grasped_object}"

    # ── Step 13: retract arm ──────────────────────────────────────────────────
    print("Step 13: retract arm")
    sim.set_state(obs)
    joint_plan = run_motion_planning(
        sim.robot.arm,
        sim.robot.arm.get_joint_positions(),
        extend_joints_to_include_fingers(sim.config.initial_joints),
        collision_bodies={sim._lab_id},
        seed=123,
        physics_client_id=sim.physics_client_id,
        held_object=sim._grasped_object_id,
        base_link_to_held_obj=sim._grasped_object_transform,
    )
    joint_plan = remap_joint_position_plan_to_constant_distance(
        joint_plan, sim.robot.arm, max_distance=config.max_action_mag / 2
    )
    obs = _execute_joint_plan(env, joint_plan, obs, actions, rewards, observations)

    # ── Step 14: base → counter ───────────────────────────────────────────────
    print("Step 14: base → counter")
    sim.set_state(obs)
    counter_x2 = 0.0  # place cube0 slightly to the left of cube1
    target_counter_base2 = SE2Pose(counter_x2, 0.7, np.pi / 2)
    base_plan = None
    for s in range(10):
        base_plan = run_single_arm_mobile_base_motion_planning(
            sim.robot, sim.robot.base.get_pose(), target_counter_base2,
            collision_bodies={sim._lab_id}, seed=456 + s,
        )
        if base_plan is not None:
            break
    assert base_plan is not None, "Base plan to counter (cube0) failed"
    obs = _execute_base_plan(env, base_plan, obs, actions, rewards, observations)

    # ── Step 15: arm → place on counter ──────────────────────────────────────
    print("Step 15: arm → place")
    sim.set_state(obs)

    place_pose2 = Pose.from_rpy((counter_x2, 1.6, 1.0), place_rpy)
    joint_plan = None
    for s in range(20):
        joint_plan = run_smooth_motion_planning_to_pose(
            place_pose2, sim.robot.arm,
            collision_ids={sim._lab_id},
            end_effector_frame_to_plan_frame=Pose.identity(),
            seed=s, max_time=5, max_candidate_plans=5,
            held_object=sim._grasped_object_id,
            base_link_to_held_obj=sim._grasped_object_transform,
        )
        if joint_plan is not None:
            break
    assert joint_plan is not None, "Place plan (cube0) failed"
    joint_plan = remap_joint_position_plan_to_constant_distance(
        joint_plan, sim.robot.arm, max_distance=config.max_action_mag / 2
    )
    obs = _execute_joint_plan(env, joint_plan, obs, actions, rewards, observations)

    # ── Step 16: open gripper to release cube0 ────────────────────────────────
    print("Step 16: open gripper")
    terminated = False
    for _ in range(5):
        action = np.array([0.0] * 3 + [0.0] * 7 + [1.0], dtype=np.float32)
        vec_obs, reward, terminated, _, _ = env.step(action)
    oc_obs = env.observation_space.devectorize(vec_obs)
    obs = PrplLab3DObjectCentricState(oc_obs.data, oc_obs.type_features)
    actions.append(action)
    rewards.append(float(reward))
    observations.append(vec_obs)

    # ── Save demo ─────────────────────────────────────────────────────────────
    timestamp = int(time.time())
    demo_subdir = Path(__file__).parent / str(seed)
    demo_subdir.mkdir(parents=True, exist_ok=True)
    demo_path = demo_subdir / f"{timestamp}.p"

    demo_data = {
        "env_id": ENV_ID,
        "timestamp": timestamp,
        "seed": seed,
        "observations": observations,
        "actions": actions,
        "rewards": rewards,
        "terminated": terminated,
        "truncated": False,
    }
    with open(demo_path, "wb") as f:
        pkl.dump(demo_data, f)

    print(f"\nDemo saved: {demo_path}")
    print(f"  {len(actions)} steps, total reward {sum(rewards):.1f}")

    env.close()
    sim.close()
    return demo_path


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--demo-dir", type=Path, default=Path("demos"))
    args = parser.parse_args()
    collect_demo(seed=args.seed, demo_dir=args.demo_dir)


if __name__ == "__main__":
    _main()
