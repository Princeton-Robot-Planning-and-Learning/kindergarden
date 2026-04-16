"""PRPL lab demo: opens lower cabinet doors, then picks and places cubes inside.

Usage:
    python exploration/use_kitchen.py
"""
from __future__ import annotations

import threading
import time

import numpy as np
import pybullet as p
from prpl_utils.utils import wrap_angle
from pybullet_helpers.geometry import Pose, SE2Pose, matrix_from_quat
from pybullet_helpers.motion_planning import (
    create_joint_distance_fn,
    remap_joint_position_plan_to_constant_distance,
    run_motion_planning,
    run_single_arm_mobile_base_motion_planning,
    run_smooth_motion_planning_to_pose,
    smoothly_follow_end_effector_path,
)

from kinder.envs.kinematic3d.prpl3d import (
    ObjectCentricPrplLab3DEnv,
    PrplLab3DEnv,
    PrplLab3DObjectCentricState,
)
from kinder.envs.kinematic3d.utils import extend_joints_to_include_fingers

STEP_DELAY = 0.01
COUNTER_SURFACE_Z = 0.76  # z of the countertop resting surface


# ── Helpers ───────────────────────────────────────────────────────────────────

def compute_grasp_pose(
    robot_arm,
    object_position: tuple[float, float, float],
    approach_height: float = 0.05,
    contact_offset: float = 0.005,
) -> tuple[Pose, Pose]:
    """Return (pre_grasp_pose, grasp_pose) for a top-down grasp."""
    from scipy.spatial.transform import Rotation

    R_ee = np.array(matrix_from_quat(robot_arm.get_end_effector_pose().orientation))
    approach_axis = R_ee[:, 2]
    rot, _ = Rotation.align_vectors([[0.0, 0.0, -1.0]], [approach_axis])
    grasp_quat = tuple(Rotation.from_matrix(rot.as_matrix() @ R_ee).as_quat())

    x, y, z = object_position
    return (
        Pose((x, y, z + approach_height), grasp_quat),
        Pose((x, y, z + contact_offset),  grasp_quat),
    )


def _execute_base_plan(environment, base_plan, obs):
    for target_base_pose in base_plan[1:]:
        delta = target_base_pose - obs.base_pose
        action = np.array([delta.x, delta.y, delta.rot] + [0.0] * 7 + [0.0], dtype=np.float32)
        vec_obs, _, _, _, _ = environment.step(action)
        oc_obs = environment.observation_space.devectorize(vec_obs)
        obs = PrplLab3DObjectCentricState(oc_obs.data, oc_obs.type_features)
        time.sleep(STEP_DELAY)
    return obs


def _execute_joint_plan(environment, joint_plan, obs):
    for target_joints in joint_plan[1:]:
        delta = [wrap_angle(a) for a in np.subtract(target_joints[:7], obs.joint_positions)]
        action = np.array([0.0] * 3 + delta + [0.0], dtype=np.float32)
        vec_obs, _, _, _, _ = environment.step(action)
        oc_obs = environment.observation_space.devectorize(vec_obs)
        obs = PrplLab3DObjectCentricState(oc_obs.data, oc_obs.type_features)
        time.sleep(STEP_DELAY)
    return obs


def _sync_obs(environment):
    """Re-read current PyBullet state into obs without moving anything.

    Use this after direct resetJointState/set_joints manipulation to keep obs
    consistent with what PyBullet actually holds.
    """
    action = np.zeros(11, dtype=np.float32)
    vec_obs, _, _, _, _ = environment.step(action)
    oc_obs = environment.observation_space.devectorize(vec_obs)
    return PrplLab3DObjectCentricState(oc_obs.data, oc_obs.type_features)


# ── Environment setup ─────────────────────────────────────────────────────────

env = PrplLab3DEnv(num_cubes=2, use_gui=True, render_mode="rgb_array", realistic_bg=False)
vec_obs, _ = env.reset(seed=123)
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

lab_id  = sim._lab_id
base_id = sim.robot.base.robot_id
oc_env  = env.unwrapped._object_centric_env

# Initial GUI camera.  Adjust cameraTargetPosition x to taste.
p.resetDebugVisualizerCamera(
    cameraDistance=1.0,
    cameraYaw=-45,
    cameraPitch=-20,
    cameraTargetPosition=[0.0, -2.0, 0.8],
    physicsClientId=oc_env.physics_client_id,
)

# Suppress the step-level collision-revert in the GUI env.  The env's
# _robot_or_held_object_collision_exists() calls check_body_collisions via
# getClosestPoints, which ignores setCollisionFilterPair and would revert every
# arm step that brings the arm near the lab geometry.  Setting the flag makes
# _get_collision_object_ids() return set(), so no revert ever fires.
oc_env.disable_collision_checking = True

# Disable lab collision in the headless sim so planning calls with
# collision_ids=set() don't report spurious contacts.
_num_lab_links = p.getNumJoints(lab_id, physicsClientId=sim.physics_client_id)
for _link_idx in range(-1, _num_lab_links):
    p.setCollisionFilterGroupMask(
        lab_id, _link_idx, 0, 0, physicsClientId=sim.physics_client_id
    )

# ── Camera controls ───────────────────────────────────────────────────────────

from pynput import keyboard, mouse as pynput_mouse  # noqa: E402

CAMERA_SPEED  = 0.05
ROT_SCALE_DEG = 0.3

pressed: set = set()
done = False

_rot_lock = threading.Lock()
_pending_rot_px = [0.0, 0.0]
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
    while not done:
        cam = p.getDebugVisualizerCamera(physicsClientId=oc_env.physics_client_id)
        distance, yaw, pitch, target = cam[10], cam[8], cam[9], list(cam[11])

        yaw_rad = np.deg2rad(yaw)
        forward = np.array([-np.cos(yaw_rad), -np.sin(yaw_rad), 0.0])
        right   = np.array([-np.sin(yaw_rad),  np.cos(yaw_rad), 0.0])

        move = np.zeros(3)
        for k in list(pressed):
            if k == keyboard.Key.up:    move += right
            elif k == keyboard.Key.down:  move -= right
            elif k == keyboard.Key.right: move -= forward
            elif k == keyboard.Key.left:  move += forward
        if np.any(move):
            target = [target[i] + CAMERA_SPEED * move[i] / np.linalg.norm(move) for i in range(3)]

        with _rot_lock:
            pdx, pdy = _pending_rot_px
            _pending_rot_px[0] = 0.0
            _pending_rot_px[1] = 0.0

        if pdx or pdy:
            pr = np.deg2rad(pitch)
            yr = np.deg2rad(yaw)
            look    = np.array([np.cos(pr) * np.cos(yr), np.cos(pr) * np.sin(yr), -np.sin(pr)])
            cam_pos = np.array(target) + distance * look
            yaw    += pdx * ROT_SCALE_DEG
            pitch   = float(np.clip(pitch + pdy * ROT_SCALE_DEG, -89, 89))
            new_pr  = np.deg2rad(pitch)
            new_yr  = np.deg2rad(yaw)
            new_look = np.array([np.cos(new_pr) * np.cos(new_yr),
                                  np.cos(new_pr) * np.sin(new_yr),
                                  -np.sin(new_pr)])
            target = (cam_pos - distance * new_look).tolist()

        if np.any(move) or pdx or pdy:
            p.resetDebugVisualizerCamera(distance, yaw, pitch, target,
                                         physicsClientId=oc_env.physics_client_id)
        time.sleep(1 / 60.0)


keyboard.Listener(on_press=on_press, on_release=on_release).start()
pynput_mouse.Listener(on_click=on_mouse_click, on_move=on_mouse_move).start()
threading.Thread(target=_camera_thread, daemon=True).start()
print("Arrow keys = pan  |  Right-click drag = rotate  |  Enter = close")

# ── Placement constants ───────────────────────────────────────────────────────

COUNTER_X   = 0.3
COUNTER_Y   = 1.6
PLACE_Z     = COUNTER_SURFACE_Z + config.block_half_extents[2]
CUBE_SPACING = 2 * config.block_half_extents[0] + 0.04
PLACE_POSITIONS = {
    "cube1": (COUNTER_X - CUBE_SPACING / 2, COUNTER_Y),
    "cube0": (COUNTER_X + CUBE_SPACING / 2, COUNTER_Y),
}


# ── pick_and_place ────────────────────────────────────────────────────────────

def pick_and_place(
    cube_name: str,
    place_x: float,
    place_y: float,
    obs,
    step_offset: int,
    place_base: SE2Pose | None = None,
    place_z_ee: float | None = None,
    place_rpy: tuple | None = None,
    place_collision_ids: set | None = None,
    place_direct: bool = False,
):
    """Pick `cube_name` from the floor and place it at (place_x, place_y).

    Optional overrides for cabinet placement:
        place_base           : SE2Pose for the base at placement
        place_z_ee           : EE z at placement
        place_rpy            : EE orientation RPY
        place_collision_ids  : collision IDs for arm planning
        place_direct         : skip pre-place + EE-path; plan directly to place_pose
    """
    joint_distance_fn = create_joint_distance_fn(sim.robot.arm)

    # ── Step 1: base → cube ───────────────────────────────────────────────────
    target_cube_temp = obs.get_object_pose(cube_name).to_se2()
    target_base = SE2Pose(target_cube_temp.x, target_cube_temp.y - 0.5, np.pi / 2)
    print(f"  {cube_name} at ({target_cube_temp.x:.2f}, {target_cube_temp.y:.2f}); "
          f"base target: ({target_base.x:.2f}, {target_base.y:.2f})")
    base_plan = None
    for _seed in range(20):
        base_plan = run_single_arm_mobile_base_motion_planning(
            sim.robot, sim.robot.base.get_pose(), target_base,
            collision_bodies=set(), seed=_seed,
        )
        if base_plan is not None:
            break
    assert base_plan is not None, f"Base plan to {cube_name} failed"
    obs = _execute_base_plan(env, base_plan, obs)
    print(f"Step {step_offset + 1} done (base → {cube_name}). Base: {obs.base_pose}")

    # ── Step 2: arm → pre-grasp ───────────────────────────────────────────────
    sim.set_state(obs)
    cx, cy, cz = obs.get_object_pose(cube_name).position
    pre_grasp_pose, grasp_pose = compute_grasp_pose(sim.robot.arm, (cx, cy, cz))

    joint_plan = run_smooth_motion_planning_to_pose(
        pre_grasp_pose, sim.robot.arm,
        collision_ids=set(),
        end_effector_frame_to_plan_frame=Pose.identity(),
        seed=123, max_candidate_plans=1,
    )
    joint_plan = remap_joint_position_plan_to_constant_distance(
        joint_plan, sim.robot.arm, max_distance=config.max_action_mag / 2
    )
    obs = _execute_joint_plan(env, joint_plan, obs)
    print(f"Step {step_offset + 2} done (arm → pre-grasp).")

    # ── Step 3: arm → grasp ───────────────────────────────────────────────────
    sim.set_state(obs)
    joint_plan = smoothly_follow_end_effector_path(
        sim.robot.arm,
        [sim.robot.arm.get_end_effector_pose(), grasp_pose],
        sim.robot.arm.get_joint_positions(),
        collision_ids=set(),
        joint_distance_fn=joint_distance_fn,
        max_smoothing_iters_per_step=1,
    )
    assert joint_plan is not None
    joint_plan = remap_joint_position_plan_to_constant_distance(
        joint_plan, sim.robot.arm, max_distance=config.max_action_mag / 2
    )
    obs = _execute_joint_plan(env, joint_plan, obs)
    print(f"Step {step_offset + 3} done (arm → grasp).")

    # ── Step 4: close gripper ─────────────────────────────────────────────────
    for _ in range(5):
        action = np.array([0.0] * 3 + [0.0] * 7 + [-1.0], dtype=np.float32)
        vec_obs, _, _, _, _ = env.step(action)
        oc_obs = env.observation_space.devectorize(vec_obs)
        obs = PrplLab3DObjectCentricState(oc_obs.data, oc_obs.type_features)
        time.sleep(STEP_DELAY)
    assert obs.grasped_object == cube_name
    print(f"Step {step_offset + 4} done (grasped {obs.grasped_object}).")

    # ── Step 5: retract arm ───────────────────────────────────────────────────
    sim.set_state(obs)
    joint_plan = run_motion_planning(
        sim.robot.arm,
        sim.robot.arm.get_joint_positions(),
        extend_joints_to_include_fingers(sim.config.initial_joints),
        collision_bodies=set(),
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

    # ── Step 6: base → place target ───────────────────────────────────────────
    sim.set_state(obs)
    _place_base = place_base if place_base is not None else SE2Pose(COUNTER_X, 0.7, np.pi / 2)
    _place_z    = place_z_ee if place_z_ee is not None else PLACE_Z + 0.1
    _place_rpy  = place_rpy if place_rpy is not None else (-np.pi / 2, np.pi, 0)
    _place_col  = place_collision_ids if place_collision_ids is not None else set()
    base_plan = None
    for _seed in range(10):
        base_plan = run_single_arm_mobile_base_motion_planning(
            sim.robot, sim.robot.base.get_pose(), _place_base,
            collision_bodies=set(), seed=456 + _seed,
        )
        if base_plan is not None:
            break
    assert base_plan is not None, "Base plan to placement target failed"
    obs = _execute_base_plan(env, base_plan, obs)
    print(f"Step {step_offset + 6} done (base → place target). Base: {obs.base_pose}")

    # ── Steps 7–8: arm → place ────────────────────────────────────────────────
    sim.set_state(obs)
    place_pose = Pose.from_rpy((place_x, place_y, _place_z), _place_rpy)

    if place_direct:
        # Plan directly to final pose — more robust for deep-inside-cabinet targets
        # where a forced Cartesian approach line fails IK mid-path.
        joint_plan = None
        for _seed in range(20):
            joint_plan = run_smooth_motion_planning_to_pose(
                place_pose, sim.robot.arm,
                collision_ids=_place_col,
                end_effector_frame_to_plan_frame=Pose.identity(),
                seed=_seed, max_time=5, max_candidate_plans=5,
                held_object=sim._grasped_object_id,
                base_link_to_held_obj=sim._grasped_object_transform,
            )
            if joint_plan is not None:
                break
        assert joint_plan is not None, "Direct place plan failed"
        joint_plan = remap_joint_position_plan_to_constant_distance(
            joint_plan, sim.robot.arm, max_distance=config.max_action_mag / 2
        )
        obs = _execute_joint_plan(env, joint_plan, obs)
        print(f"Step {step_offset + 7} done (arm → place, direct).")
    else:
        pre_place_pose = Pose.from_rpy((place_x, place_y - 0.1, _place_z), _place_rpy)
        joint_plan = run_smooth_motion_planning_to_pose(
            pre_place_pose, sim.robot.arm,
            collision_ids=_place_col,
            end_effector_frame_to_plan_frame=Pose.identity(),
            seed=123, max_time=10, max_candidate_plans=10,
            held_object=sim._grasped_object_id,
            base_link_to_held_obj=sim._grasped_object_transform,
        )
        assert joint_plan is not None
        joint_plan = remap_joint_position_plan_to_constant_distance(
            joint_plan, sim.robot.arm, max_distance=config.max_action_mag / 2
        )
        obs = _execute_joint_plan(env, joint_plan, obs)
        print(f"Step {step_offset + 7} done (arm → pre-place).")

        sim.set_state(obs)
        joint_plan = smoothly_follow_end_effector_path(
            sim.robot.arm,
            [sim.robot.arm.get_end_effector_pose(), place_pose],
            sim.robot.arm.get_joint_positions(),
            collision_ids=_place_col,
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

    # ── Release ───────────────────────────────────────────────────────────────
    oc_env._grasped_object = None
    oc_env._grasped_object_transform = None
    oc_env.robot.arm.open_fingers()
    print(f"{cube_name} placed at ({place_x:.2f}, {place_y:.2f})!")

    for _ in range(480):  # 2 s at 240 Hz — let cube settle
        p.stepSimulation(physicsClientId=oc_env.physics_client_id)
        time.sleep(1 / 240.0)

    # ── Step 9: retract arm after placement ───────────────────────────────────
    sim.set_state(obs)
    _pp_retract = None
    for _seed in range(10):
        _pp_retract = run_motion_planning(
            sim.robot.arm,
            sim.robot.arm.get_joint_positions(),
            extend_joints_to_include_fingers(sim.config.initial_joints),
            collision_bodies=set(),
            seed=_seed,
            physics_client_id=sim.physics_client_id,
        )
        if _pp_retract is not None:
            break
    assert _pp_retract is not None, f"Post-place arm retract failed for {cube_name}"
    _pp_retract = remap_joint_position_plan_to_constant_distance(
        _pp_retract, sim.robot.arm, max_distance=config.max_action_mag / 2
    )
    obs = _execute_joint_plan(env, _pp_retract, obs)
    print(f"Step {step_offset + 9} done (arm retracted after place).")

    return obs


# ── open_lower_cabinet_door ───────────────────────────────────────────────────

def open_lower_cabinet_door(door_joint_name: str, handle_joint_name: str, obs,
                            yaw_sign: float = 1.0,
                            approach_from_front: bool = False):
    """Open a lower-cabinet door by URDF joint name.

    Parameters
    ----------
    door_joint_name    : revolute joint that rotates the door (e.g. "cab_6_d_2")
    handle_joint_name  : fixed joint/link for the handle     (e.g. "cab_6_h_2")
    obs                : current observation
    yaw_sign           : +1 hinge-on-right, -1 hinge-on-left
    approach_from_front: True → pre-grasp 15 cm in front; False → 15 cm above
    """
    _pc_id   = oc_env.physics_client_id
    _kg_id   = oc_env._lab_id
    _arm_gui = oc_env.robot.arm
    _CAB_OPEN_ANGLE = 1.5
    _GRASP_QUAT = Pose.from_rpy((0, 0, 0), (np.pi, 0, 0)).orientation
    _N_OPEN = 80

    # ── Pre-rotate wrist to yaw 0 ─────────────────────────────────────────────
    _wrist_idx = _arm_gui.arm_joints.index(_arm_gui.joint_from_name("joint_7"))
    _cur_joints = list(_arm_gui.get_joint_positions())
    _target_wrist = list(_cur_joints)
    _target_wrist[_wrist_idx] = 0.0
    for _i in range(20):
        _frac = (_i + 1) / 20
        _arm_gui.set_joints([(1 - _frac) * c + _frac * t
                             for c, t in zip(_cur_joints, _target_wrist)])
        time.sleep(STEP_DELAY)

    # ── Locate URDF joints ────────────────────────────────────────────────────
    _num_j = p.getNumJoints(_kg_id, physicsClientId=_pc_id)
    _cab_joint_idx = next(
        i for i in range(_num_j)
        if p.getJointInfo(_kg_id, i, physicsClientId=_pc_id)[1].decode() == door_joint_name
    )
    _handle_link_idx = next(
        i for i in range(_num_j)
        if p.getJointInfo(_kg_id, i, physicsClientId=_pc_id)[1].decode() == handle_joint_name
    )

    # ── Closed-door handle world position ────────────────────────────────────
    p.resetJointState(_kg_id, _cab_joint_idx, 0.0, 0.0, physicsClientId=_pc_id)
    _handle_closed = tuple(
        p.getLinkState(_kg_id, _handle_link_idx,
                       computeForwardKinematics=True, physicsClientId=_pc_id)[4]
    )
    print(f"[{door_joint_name}] handle (closed): {_handle_closed}")

    # ── Step a: base → cabinet ────────────────────────────────────────────────
    sim.set_state(obs)
    _base_plan = run_single_arm_mobile_base_motion_planning(
        sim.robot, sim.robot.base.get_pose(),
        SE2Pose(_handle_closed[0], 0.9, np.pi / 2),
        collision_bodies={lab_id}, seed=999,
    )
    assert _base_plan is not None, f"Base plan failed for {door_joint_name}"
    obs = _execute_base_plan(env, _base_plan, obs)
    print(f"[{door_joint_name}] step a done (base → cabinet).")

    # ── Step b: arm → pre-grasp ───────────────────────────────────────────────
    _pre_grasp_offset = np.array([0.0, -0.15, 0.0]) if approach_from_front \
        else np.array([0.0, 0.0, 0.15])
    _pre_grasp_pose = Pose(
        tuple(np.array(_handle_closed) + _pre_grasp_offset), _GRASP_QUAT
    )
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
    assert _joint_plan is not None, f"Pre-grasp planning failed for {handle_joint_name}"
    _joint_plan = remap_joint_position_plan_to_constant_distance(
        _joint_plan, sim.robot.arm, max_distance=config.max_action_mag / 2
    )
    obs = _execute_joint_plan(env, _joint_plan, obs)
    print(f"[{door_joint_name}] step b done (arm → pre-grasp).")

    # ── Step c: arm → handle ──────────────────────────────────────────────────
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
    assert _joint_plan_c is not None, f"Descent planning failed for {handle_joint_name}"
    _joint_plan_c = remap_joint_position_plan_to_constant_distance(
        _joint_plan_c, sim.robot.arm, max_distance=config.max_action_mag / 2
    )
    for _wp in _joint_plan_c:
        _arm_gui.set_joints(list(np.asarray(_wp)[: len(_arm_gui.arm_joints)]))
        time.sleep(STEP_DELAY)

    # Fine-tune with IK if EE fell short.
    _ee_pos = p.getLinkState(_arm_gui.robot_id, _arm_gui.end_effector_id,
                             physicsClientId=_pc_id)[0]
    _ee_dist = float(np.linalg.norm(np.array(_ee_pos) - np.array(_handle_closed)))
    print(f"[{door_joint_name}] step c done — EE gap: {_ee_dist:.3f} m")
    if _ee_dist < 0.08:
        _ik = p.calculateInverseKinematics(
            _arm_gui.robot_id, _arm_gui.end_effector_id,
            _handle_closed, _GRASP_QUAT, physicsClientId=_pc_id,
        )
        _ft = list(_ik[: len(_arm_gui.arm_joints)])
        if _arm_gui.check_joint_limits(_ft):
            _arm_gui.set_joints(_ft)
            print(f"  fine-tuned {_ee_dist:.3f} m gap.")
    else:
        print(f"  WARNING: gap {_ee_dist:.3f} m too large — skipping fine-tune.")

    # ── Close gripper ─────────────────────────────────────────────────────────
    _closed_fingers = _arm_gui.finger_state_to_joints(_arm_gui.closed_fingers_state)
    for _fi in range(20):
        _frac = (_fi + 1) / 20
        for _fname, _fpos in zip(_arm_gui.finger_joint_names, _closed_fingers):
            p.resetJointState(_arm_gui.robot_id, _arm_gui.joint_from_name(_fname),
                              _fpos * _frac, physicsClientId=_pc_id)
        time.sleep(STEP_DELAY)
    print(f"[{door_joint_name}] gripper closed.")

    # ── Door-opening loop ─────────────────────────────────────────────────────
    p.resetJointState(_kg_id, _cab_joint_idx, _CAB_OPEN_ANGLE, 0.0, physicsClientId=_pc_id)
    _base_end_x = p.getLinkState(_kg_id, _handle_link_idx,
                                  computeForwardKinematics=True, physicsClientId=_pc_id)[4][0]
    _base_end_y = 0.1
    p.resetJointState(_kg_id, _cab_joint_idx, 0.0, 0.0, physicsClientId=_pc_id)

    _base_target_x = _handle_closed[0]
    for _i in range(_N_OPEN):
        _t = (_i + 1) / _N_OPEN
        _door_angle = _CAB_OPEN_ANGLE * _t
        p.resetJointState(_kg_id, _cab_joint_idx, _door_angle, 0.0, physicsClientId=_pc_id)
        oc_env.robot.set_base(SE2Pose(
            _base_target_x + (_base_end_x - _base_target_x) * _t,
            0.4 + (_base_end_y - 0.4) * _t,
            np.pi / 2,
        ))
        _handle_now = p.getLinkState(_kg_id, _handle_link_idx,
                                      computeForwardKinematics=True, physicsClientId=_pc_id)[4]
        _tracking_quat = Pose.from_rpy((0, 0, 0), (np.pi, 0, yaw_sign * _door_angle)).orientation
        _ik = p.calculateInverseKinematics(
            _arm_gui.robot_id, _arm_gui.end_effector_id,
            _handle_now, _tracking_quat, physicsClientId=_pc_id,
        )
        _arm_gui.set_joints(list(_ik[: len(_arm_gui.arm_joints)]))
        for _fn, _fp in zip(_arm_gui.finger_joint_names, _closed_fingers):
            p.resetJointState(_arm_gui.robot_id, _arm_gui.joint_from_name(_fn),
                              _fp, physicsClientId=_pc_id)
        time.sleep(STEP_DELAY)
    print(f"[{door_joint_name}] door fully open!")

    # ── Open gripper ──────────────────────────────────────────────────────────
    for _fn in _arm_gui.finger_joint_names:
        p.resetJointState(_arm_gui.robot_id, _arm_gui.joint_from_name(_fn),
                          0.0, physicsClientId=_pc_id)
    time.sleep(STEP_DELAY)

    # Sync obs with the actual post-door-open arm/base state so that retract
    # planning starts from the correct configuration.
    obs = _sync_obs(env)

    # ── Retract arm ───────────────────────────────────────────────────────────
    sim.set_state(obs)
    _retract_plan = None
    for _seed in range(10):
        _retract_plan = run_motion_planning(
            sim.robot.arm,
            sim.robot.arm.get_joint_positions(),
            extend_joints_to_include_fingers(sim.config.initial_joints),
            collision_bodies=set(),
            seed=_seed,
            physics_client_id=sim.physics_client_id,
        )
        if _retract_plan is not None:
            break
    assert _retract_plan is not None, f"[{door_joint_name}] arm retract plan failed"
    _retract_plan = remap_joint_position_plan_to_constant_distance(
        _retract_plan, sim.robot.arm, max_distance=config.max_action_mag / 2
    )
    obs = _execute_joint_plan(env, _retract_plan, obs)
    print(f"[{door_joint_name}] arm retracted.")

    # ── Base → home ───────────────────────────────────────────────────────────
    sim.set_state(obs)
    _home_base_plan = None
    for _seed in range(10):
        _home_base_plan = run_single_arm_mobile_base_motion_planning(
            sim.robot, sim.robot.base.get_pose(),
            config.robot_base_home_pose,
            collision_bodies=set(), seed=_seed,
        )
        if _home_base_plan is not None:
            break
    assert _home_base_plan is not None, f"[{door_joint_name}] base home plan failed"
    obs = _execute_base_plan(env, _home_base_plan, obs)
    print(f"[{door_joint_name}] base home: {obs.base_pose}")

    return obs


# ── Query closed-door handle positions BEFORE opening ────────────────────────

def _handle_world_pos(door_joint_name: str, handle_joint_name: str) -> tuple:
    _kg = oc_env._lab_id
    _pc = oc_env.physics_client_id
    _nj = p.getNumJoints(_kg, physicsClientId=_pc)
    _dj = next(i for i in range(_nj)
               if p.getJointInfo(_kg, i, physicsClientId=_pc)[1].decode() == door_joint_name)
    _hj = next(i for i in range(_nj)
               if p.getJointInfo(_kg, i, physicsClientId=_pc)[1].decode() == handle_joint_name)
    p.resetJointState(_kg, _dj, 0.0, 0.0, physicsClientId=_pc)
    return tuple(p.getLinkState(_kg, _hj, computeForwardKinematics=True,
                                physicsClientId=_pc)[4])


_h_r = _handle_world_pos("cab_6_d_2", "cab_6_h_2")
_h_l = _handle_world_pos("cab_6_d_1", "cab_6_h_1")
print(f"Cabinet handle positions — right: {_h_r}, left: {_h_l}")

# ── Open both lower cabinet doors ─────────────────────────────────────────────

obs = open_lower_cabinet_door("cab_6_d_2", "cab_6_h_2", obs, yaw_sign=+1.0)
obs = open_lower_cabinet_door("cab_6_d_1", "cab_6_h_1", obs, yaw_sign=-1.0,
                              approach_from_front=True)

# ── Place one cube inside each open cabinet ───────────────────────────────────

CAB_R_X = _h_r[0] + 0.1
CAB_L_X = _h_l[0] - 0.1
CAB_Y   = _h_r[1] + 0.3     # 30 cm past the handle face, into the interior
CAB_Z   = 0.5               # EE z above the cabinet floor

obs = pick_and_place(
    "cube1", CAB_R_X, CAB_Y, obs, step_offset=0,
    place_base=SE2Pose(CAB_R_X, 0.6, np.pi / 2),
    place_z_ee=CAB_Z,
    place_collision_ids=set(),
    place_direct=True,
)

obs = pick_and_place(
    "cube0", CAB_L_X, CAB_Y, obs, step_offset=9,
    place_base=SE2Pose(CAB_L_X, 0.6, np.pi / 2),
    place_z_ee=CAB_Z,
    place_collision_ids=set(),
    place_direct=True,
)

# ── Retract arm and drive home ────────────────────────────────────────────────

sim.set_state(obs)
_retract_plan = None
for _seed in range(10):
    _retract_plan = run_motion_planning(
        sim.robot.arm,
        sim.robot.arm.get_joint_positions(),
        extend_joints_to_include_fingers(sim.config.initial_joints),
        collision_bodies={lab_id, base_id},
        seed=_seed,
        physics_client_id=sim.physics_client_id,
    )
    if _retract_plan is not None:
        break
assert _retract_plan is not None, "Final arm retract plan failed"
_retract_plan = remap_joint_position_plan_to_constant_distance(
    _retract_plan, sim.robot.arm, max_distance=config.max_action_mag / 2
)
obs = _execute_joint_plan(env, _retract_plan, obs)

sim.set_state(obs)
_home_base_plan = None
for _seed in range(10):
    _home_base_plan = run_single_arm_mobile_base_motion_planning(
        sim.robot, sim.robot.base.get_pose(),
        config.robot_base_home_pose,
        collision_bodies={lab_id}, seed=789 + _seed,
    )
    if _home_base_plan is not None:
        break
assert _home_base_plan is not None, "Final base home plan failed"
obs = _execute_base_plan(env, _home_base_plan, obs)
print(f"Done. Base: {obs.base_pose}")

while not done:
    p.stepSimulation(physicsClientId=oc_env.physics_client_id)
    time.sleep(1 / 240.0)

env.close()
sim.close()
