"""Run the vendor walk policy of the Sprout in the parallel Sprout environments.

This is an example of a policy as an external layer over ``Sprout3DBatchEnv``: the
environment takes joint position targets, and the policy computes them from the robot
state and a velocity command. The policy (an ONNX file of the Fauna SDK) is not a part
of this repository; give its folder with ``--policy-dir``.

The script is also a check of the scenarios. In each task, the robot stands, walks
backward, walks forward again, and stands. The report gives the number of worlds
where the robot fell, the distance that it walked, and the motion of the objects.

Example:
    python scripts/run_sprout_walk_policy.py --policy-dir <sdk>/onnx_motor_control/model
"""

# Warp functions have no type annotations.
# mypy: disable-error-code="no-untyped-call,attr-defined"

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import warp as wp

import kinder
from kinder.envs.dynamic3d.robots.sprout_robot_env import load_sprout_model
from kinder.envs.dynamic3d.sprout_batch_env import Sprout3DBatchEnv

# Joint order of the policy inputs and outputs (the hardware order of the robot).
CONTROL_JOINTS = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_knee_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_knee_joint",
    "left_ankle_joint",
    "right_ankle_joint",
    "waist_yaw_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_gripper_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_gripper_joint",
    "neck_pitch_joint",
    "neck_yaw_joint",
    "left_wrist_pitch_joint",
    "right_wrist_pitch_joint",
]
# Velocity commands (m/s, m/s, rad/s) that the joystick axes -1 and 1 give.
COMMAND_LIMITS = {"vx": (-0.5, 1.0), "vy": (-0.5, 0.5), "wz": (-1.25, 1.25)}


class WalkPolicy:
    """The ONNX walk policy, with its recurrent state for each world."""

    def __init__(self, policy_dir: Path, num_worlds: int) -> None:
        options = ort.SessionOptions()
        options.log_severity_level = 3
        options.intra_op_num_threads = 1
        self.session = ort.InferenceSession(
            str(policy_dir / "policy.onnx"), options, providers=["CPUExecutionProvider"]
        )
        with open(policy_dir / "settings.json", "r", encoding="utf-8") as f:
            carry = json.load(f)["carry"]
        self.carry = {
            key: np.zeros((num_worlds, *shape[1:]), dtype=np.float32)
            for key, shape in carry.items()
        }
        self.outputs = ["motor_targets_control", "stiffness_control"]
        self.outputs += [f"updated_{key}" for key in self.carry]
        model_joints = list(load_sprout_model().joint_names)
        # control_of_model[i] is the model index of control joint i.
        self.control_of_model = np.array(
            [model_joints.index(n) for n in CONTROL_JOINTS]
        )
        self.stiffness = np.zeros(len(CONTROL_JOINTS))

    def reset(self) -> None:
        """Clear the recurrent state of all worlds."""
        for value in self.carry.values():
            value[:] = 0.0

    def __call__(
        self,
        base_quat: np.ndarray,
        base_gyro: np.ndarray,
        joint_pos: np.ndarray,
        joint_vel: np.ndarray,
        command: tuple[float, float, float],
    ) -> np.ndarray:
        """Return joint position targets in the joint order of the robot model."""
        axes = np.array(
            [
                value / high if value >= 0 else value / -low
                for value, (low, high) in zip(command, COMMAND_LIMITS.values())
            ],
            dtype=np.float32,
        )
        # The policy uses the IMU frame, which is the torso frame turned 180 degrees
        # about the vertical axis.
        w, x, y, z = base_quat.T
        imu_quat = np.stack((-z, y, -x, w), axis=1)
        imu_gyro = base_gyro * np.array([-1.0, -1.0, 1.0])
        targets = np.zeros_like(joint_pos)
        for world in range(len(joint_pos)):  # the ONNX graph has a batch size of 1
            inputs = {
                "joint_positions_control": joint_pos[world, self.control_of_model][
                    None
                ],
                "joint_velocities_control": joint_vel[world, self.control_of_model][
                    None
                ],
                "imu_quat": imu_quat[world, None],
                "base_ang_vel": imu_gyro[world, None],
                "joystick_axes": axes[None],
            }
            inputs = {k: v.astype(np.float32) for k, v in inputs.items()}
            inputs |= {key: value[world, None] for key, value in self.carry.items()}
            output = self.session.run(self.outputs, inputs)
            targets[world, self.control_of_model] = output[0][0]
            self.stiffness = output[1][0]
            for index, key in enumerate(self.carry):
                self.carry[key][world] = output[2 + index][0]
        return targets


def robot_columns(env: Sprout3DBatchEnv) -> dict[str, list[int]]:
    """Find the observation columns of the robot features."""
    space = env.observation_space
    names = [
        f"{obj.name}.{feature}"
        for obj in space.constant_objects
        for feature in space.type_features[obj.type]
    ]
    robot = env._object_centric_env.robot_name  # pylint: disable=protected-access

    def find(features: list[str]) -> list[int]:
        return [names.index(f"{robot}.{feature}") for feature in features]

    num_joints = env.action_dim
    return {
        "xy": find(["pos_base_x", "pos_base_y"]),
        "quat": find([f"pos_base_q{c}" for c in "wxyz"]),
        "gyro": find([f"vel_base_w{c}" for c in "xyz"]),
        "joint_pos": find([f"pos_joint{i + 1}" for i in range(num_joints)]),
        "joint_vel": find([f"vel_joint{i + 1}" for i in range(num_joints)]),
        "object_xyz": [
            i for i, name in enumerate(names) if name.split(".")[1] in ("x", "y", "z")
        ],
    }


def check_task(task_path: Path, policy_dir: Path, args: argparse.Namespace) -> str:
    """Run the stand and walk sequence in one task and return one report line."""
    env = Sprout3DBatchEnv(str(task_path), num_worlds=args.num_worlds, seed=args.seed)
    policy = WalkPolicy(policy_dir, args.num_worlds)
    columns = robot_columns(env)
    obs = env.reset().numpy()
    start_xy = obs[:, columns["xy"]].copy()
    start_objects = obs[:, columns["object_xyz"]].copy()
    fallen_after: dict[str, int] = {}
    back_distance = np.zeros(args.num_worlds)
    # (phase, velocity command, seconds)
    phases = [
        ("stand", (0.0, 0.0, 0.0), args.stand_seconds),
        ("back", (-args.speed, 0.0, 0.0), args.walk_seconds),
        ("forward", (args.speed, 0.0, 0.0), args.walk_seconds),
        ("stop", (0.0, 0.0, 0.0), 2.0),
    ]
    fallen = np.zeros(args.num_worlds, dtype=bool)
    for phase, command, seconds in phases:
        for _ in range(int(round(seconds / env.step_dt))):
            action = policy(
                obs[:, columns["quat"]],
                obs[:, columns["gyro"]],
                obs[:, columns["joint_pos"]],
                obs[:, columns["joint_vel"]],
                command,
            )
            obs_wp, _, _, _, info = env.step(action)
            obs = obs_wp.numpy()
            fallen |= info["fallen"].numpy()
        fallen_after[phase] = int(fallen.sum())
        if phase == "back":
            back_distance = np.linalg.norm(obs[:, columns["xy"]] - start_xy, axis=1)
    end_distance = np.linalg.norm(obs[:, columns["xy"]] - start_xy, axis=1)
    object_motion = np.abs(obs[:, columns["object_xyz"]] - start_objects)
    object_motion = object_motion.max() if object_motion.size else 0.0
    gain_error = np.abs(
        policy.stiffness - load_sprout_model().stiffness[policy.control_of_model]
    ).max()
    usage = env.buffer_use()
    env.close()
    upright = ~fallen
    return (
        f"{task_path.stem.removeprefix('Sprout')[:44]:44} "
        f"fell: stand {fallen_after['stand']:2d} back {fallen_after['back']:2d} "
        f"forward {fallen_after['forward']:2d} stop {fallen_after['stop']:2d} "
        f"of {args.num_worlds} | walked back "
        f"{back_distance[upright].mean() if upright.any() else float('nan'):.2f} m, "
        f"end {end_distance[upright].mean() if upright.any() else float('nan'):.2f} m "
        f"from start | objects moved {object_motion:.3f} m | gain difference "
        f"{gain_error:.1f} | constraints {usage['constraints']}/"
        f"{usage['constraints_limit']}"
    )


def main() -> None:
    """Run the walk policy check in the Sprout tasks."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-dir", type=Path, required=True)
    parser.add_argument(
        "--tasks", default="", help="substrings of task names, with commas"
    )
    parser.add_argument("--num-worlds", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--speed", type=float, default=0.3, help="walk speed (m/s)")
    parser.add_argument("--stand-seconds", type=float, default=4.0)
    parser.add_argument("--walk-seconds", type=float, default=3.0)
    args = parser.parse_args()
    wp.config.quiet = True

    tasks_dir = Path(kinder.__file__).parent / "envs" / "dynamic3d" / "tasks"
    keys = [k for k in args.tasks.split(",") if k]
    for task_path in sorted(tasks_dir.glob("Sprout*/*.json")):
        if keys and not any(key in task_path.stem for key in keys):
            continue
        print(check_task(task_path, args.policy_dir, args), flush=True)


if __name__ == "__main__":
    main()
