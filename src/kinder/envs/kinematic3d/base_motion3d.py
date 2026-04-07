"""Environment where only base motion is required to reach some goal."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Type as TypingType

import numpy as np
import pybullet as p
from pybullet_helpers.geometry import Pose, SE2Pose, get_pose, set_pose
from relational_structs import Object, ObjectCentricState
from relational_structs.utils import create_state_from_dict

from kinder.core import ConstantObjectKinDEREnv, FinalConfigMeta
from kinder.envs.kinematic3d.base_env import (
    Kinematic3DEnvConfig,
    ObjectCentricKinematic3DRobotEnv,
)
from kinder.envs.kinematic3d.object_types import (
    Kinematic3DEnvTypeFeatures,
    Kinematic3DPointType,
    Kinematic3DRobotType,
)
from kinder.envs.kinematic3d.utils import Kinematic3DObjectCentricState


@dataclass(frozen=True)
class BaseMotion3DEnvConfig(Kinematic3DEnvConfig, metaclass=FinalConfigMeta):
    """Config for BaseMotion3DEnv()."""

    # Robot.
    robot_name: str = "tidybot-kinova"
    check_base_collisions: bool = True

    # Target.
    target_radius: float = 0.05
    target_z: float = 0.2
    target_color: tuple[float, float, float, float] = (1.0, 0.2, 0.2, 0.5)
    target_lower_bound: SE2Pose = SE2Pose(-2, -2, -np.pi)
    target_upper_bound: SE2Pose = SE2Pose(2, 2, np.pi)


class BaseMotion3DObjectCentricState(Kinematic3DObjectCentricState):
    """A state in the BaseMotion3DEnv().

    Adds convenience methods on top of Kinematic3DObjectCentricState().
    """

    @property
    def target_base_pose(self) -> SE2Pose:
        """The pose of the base target, assuming the name "target"."""
        target = self.get_object_from_name("target")
        pose = Pose(
            (self.get(target, "x"), self.get(target, "y"), self.get(target, "z"))
        )
        se2_pose = pose.to_se2()
        return se2_pose


class ObjectCentricBaseMotion3DEnv(
    ObjectCentricKinematic3DRobotEnv[
        BaseMotion3DObjectCentricState, BaseMotion3DEnvConfig
    ]
):
    """Environment where only base motion planning is needed to reach a goal."""

    def __init__(
        self, config: BaseMotion3DEnvConfig = BaseMotion3DEnvConfig(), **kwargs
    ) -> None:
        super().__init__(config=config, **kwargs)

        # Create target.
        visual_id = p.createVisualShape(
            p.GEOM_SPHERE,
            radius=self.config.target_radius,
            rgbaColor=self.config.target_color,
            physicsClientId=self.physics_client_id,
        )

        # Create the body.
        self.target_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=visual_id,
            basePosition=(0, 0, 0),  # set in reset()
            baseOrientation=(0, 0, 0, 1),
            physicsClientId=self.physics_client_id,
        )

    @property
    def state_cls(self) -> TypingType[Kinematic3DObjectCentricState]:
        return BaseMotion3DObjectCentricState

    def _create_constant_initial_state_dict(self) -> dict[Object, dict[str, float]]:
        # Neither the target nor the robot are constant in this env.
        return {}

    def _reset_objects(self) -> None:
        # Reset the target. Sample and check that the robot has not already reached it.
        target_pose: SE2Pose | None = None
        lb = self.config.target_lower_bound
        ub = self.config.target_upper_bound
        robot_base_pose = self.robot.base.get_pose()
        for _ in range(100_000):
            x, y, rot = self.np_random.uniform(
                (lb.x, lb.y, lb.rot), (ub.x, ub.y, ub.rot)
            )
            target_pose = SE2Pose(x, y, rot)
            # If the goal is already reached, keep sampling.
            if not self._robot_at_target_pose(robot_base_pose, target_pose):
                break
        else:
            raise RuntimeError("Failed to find reachable target position")
        target_se3_pose = target_pose.to_se3(self.config.target_z)
        set_pose(self.target_id, target_se3_pose, self.physics_client_id)

    def _set_object_states(self, obs: BaseMotion3DObjectCentricState) -> None:
        assert self.target_id is not None
        target_se3_pose = obs.target_base_pose.to_se3(0.0)
        set_pose(self.target_id, target_se3_pose, self.physics_client_id)

    def _object_name_to_pybullet_id(self, object_name: str) -> int:
        if object_name == "target":
            return self.target_id
        raise ValueError(f"Unrecognized object name: {object_name}")

    def _get_collision_object_ids(self) -> set[int]:
        return set()

    def _get_movable_object_names(self) -> set[str]:
        return set()

    def _get_surface_object_names(self) -> set[str]:
        return set()

    def _get_half_extents(self, object_name: str) -> tuple[float, float, float]:
        raise NotImplementedError("No objects have half extents")

    def _get_obs(self) -> BaseMotion3DObjectCentricState:
        state_dict = self._create_state_dict(
            [("robot", Kinematic3DRobotType), ("target", Kinematic3DPointType)]
        )
        state = create_state_from_dict(
            state_dict,
            Kinematic3DEnvTypeFeatures,
            state_cls=BaseMotion3DObjectCentricState,
        )
        assert isinstance(state, BaseMotion3DObjectCentricState)
        return state

    def _robot_at_target_pose(
        self, robot_base_pose: SE2Pose, target_pose: SE2Pose
    ) -> bool:
        dist = float(
            np.linalg.norm(
                np.array(
                    [
                        target_pose.x - robot_base_pose.x,
                        target_pose.y - robot_base_pose.y,
                    ]
                )
            )
        )
        return dist < self.config.target_radius

    def goal_reached(self) -> bool:
        robot_base_pose = self.robot.base.get_pose()
        target_se3_pose = get_pose(self.target_id, self.physics_client_id)
        target_pose = target_se3_pose.to_se2()
        return self._robot_at_target_pose(robot_base_pose, target_pose)


class BaseMotion3DEnv(ConstantObjectKinDEREnv):
    """Base motion 3D env with a constant number of objects."""

    def _create_object_centric_env(
        self, *args, **kwargs
    ) -> ObjectCentricKinematic3DRobotEnv:
        return ObjectCentricBaseMotion3DEnv(*args, **kwargs)

    def _get_constant_object_names(
        self, exemplar_state: ObjectCentricState
    ) -> list[str]:
        return ["robot", "target"]

    def _create_env_markdown_description(self) -> str:
        """Create environment description."""
        # pylint: disable=line-too-long
        return """A very simple environment where only base motion planning is needed to reach a goal."""

    def _create_in_context_examples(self) -> str:
        """Create concrete examples showing strategies for solving the BaseMotion3D
        task.

        Returns formatted examples with actual object positions and action sequences.
        """
        # pylint: disable=line-too-long
        return """
**Example 1: Direct Path to Target**

Initial State:
- Robot: base at (0.0, 0.0, 0.0), orientation 0°
- Target: position (1.0, 0.5, 0.2), radius 0.05m

Goal: Move robot base to within 0.05m of target position (1.0, 0.5)

Strategy: Navigate directly to target using collision-free motion planning

High-Level Plan:
1. move_base_to_target(robot, target, params=[])
   - Navigate robot base from (0.0, 0.0, 0.0) to (1.0, 0.5, 0.2)
   - No parameters needed (params=[])
   - Uses PyBullet motion planning internally to compute collision-free path
   - Plans sequence of base poses (x, y, theta) as waypoints
   - Controller executes waypoint trajectory:
     - Computes velocities [dx_base, dy_base, dtheta_base] for base motion
     - Maintains arm joints [7 values] and gripper [1 value] at fixed configuration
     - Action space: [dx_base, dy_base, dtheta_base, joint1, ..., joint7, gripper] (11D)
   - Simple case: path to (1.0, 0.5) with no obstacles

Goal Reached:
- Distance between robot base and target < 0.05m ✓
- Efficient navigation with automatic path planning ✓

Key Insights:
- move_base_to_target handles full motion planning pipeline internally
- No parameters required (empty tuple params=[])
- Controllers track planned waypoints via low-level actions
- Base controller maintains arm/gripper configuration during navigation

**Example 2: Target Behind Robot**

Initial State:
- Robot: base at (1.0, 1.0, 0.0), orientation 0° (facing right)
- Target: position (-0.5, 0.5, 0.2), radius 0.05m

Goal: Move robot base to within 0.05m of target position (-0.5, 0.5)

Strategy: Use motion planner to compute path (may include rotation/reversal)

High-Level Plan:
1. move_base_to_target(robot, target, params=[])
   - Navigate from (1.0, 1.0, 0.0) to (-0.5, 0.5, 0.2)
   - Motion planner computes efficient path:
     - Option A: Move backwards along trajectory
     - Option B: Rotate then drive forward
     - Option C: Combined motion optimizing smoothness
   - Controller executes planned waypoints
   - Base actions [dx_base, dy_base, dtheta_base] implement trajectory
   - Distance: √((1.0-(-0.5))² + (1.0-0.5)²) ≈ 1.58m

Goal Reached:
- Distance between robot base and target < 0.05m ✓
- Backward/rotational navigation handled automatically ✓

Key Insights:
- Motion planner handles complex scenarios (backward motion, rotation)
- Single skill call abstracts trajectory details
- Controller ensures smooth base velocity profiles
- No manual waypoint specification needed

**Example 3: Diagonal Movement**

Initial State:
- Robot: base at (0.0, 0.0, 0.0), orientation 0°
- Target: position (0.8, 0.6, 0.2), radius 0.05m

Goal: Move robot base to within 0.05m of target position (0.8, 0.6)

Strategy: Use motion planning for efficient diagonal path

High-Level Plan:
1. move_base_to_target(robot, target, params=[])
   - Navigate from (0.0, 0.0, 0.0) to (0.8, 0.6, 0.2)
   - Distance: √(0.8² + 0.6²) = 1.0m at angle ≈ 36.87°
   - Motion planner computes diagonal trajectory:
     - May rotate towards target direction initially
     - Drive along optimal path
     - Adjust orientation as needed for smooth motion
   - Controller executes waypoint sequence:
     - [dx_base, dy_base] follow planned trajectory
     - [dtheta_base] adjusts orientation
   - PyBullet motion planner ensures kinematic feasibility

Goal Reached:
- Distance between robot base and target < 0.05m ✓
- Diagonal path executed smoothly ✓
- Combined translation handled automatically ✓

Key Insights:
- Motion planning automatically computes optimal paths
- Base controller can execute diagonal trajectories
- No explicit waypoint calculation required by user
- Efficient motion in 2D workspace

**Example 4: Precise Positioning**

Initial State:
- Robot: base at (0.5, 0.5, 0.0), orientation 0° (facing right)
- Target: position (0.5, 1.5, 0.2), radius 0.05m

Goal: Move robot base to within 0.05m of target position (0.5, 1.5)

Note: Robot orientation does not affect goal condition, only base position matters

Strategy: Motion planning with position constraint

High-Level Plan:
1. move_base_to_target(robot, target, params=[])
   - Navigate from (0.5, 0.5, 0°) to (0.5, 1.5, 0.2)
   - Straight-line motion in y-direction (1.0m)
   - Motion planner computes direct path:
     - Robot may maintain current orientation (0°) 
     - Or adjust orientation for smoother dynamics
   - Controller executes waypoints to reach target
   - Final position: (0.5, 1.5) within tolerance

Goal Reached:
- Distance between robot base and target < 0.05m ✓
- Direct path utilized ✓

Key Insights:
- Goal only checks base position distance, not orientation
- Motion planner optimizes for reaching target position
- May or may not adjust orientation depending on dynamics
- Single skill handles complete navigation

**General Parameterized Skill Interpretation:**

Skill:
- move_base_to_target(robot, target, params=[]):
  - Navigate robot base from current position to target position
  - params=[] (empty tuple): no additional parameters needed
  - Target specifies desired position (x, y, z)
  - Uses PyBullet motion planning for collision-free path computation

Motion Planning:
- Computes sequence of waypoints in SE(2): [(x₁, y₁, θ₁), (x₂, y₂, θ₂), ..., (xₙ, yₙ, θₙ)]
- Considers robot kinematics (differential drive or holonomic base)
- Ensures collision-free trajectory in workspace (though no obstacles in this environment)
- Optimizes for smoothness and efficiency

Controller Execution:
- Tracks planned waypoints using low-level base actions
- Action space: [dx_base, dy_base, dtheta_base, joint1, ..., joint7, gripper]
  - Indices 0-2: Base velocities [dx_base, dy_base, dtheta_base] for translation and rotation
  - Indices 3-9: Joint angles [joint1, ..., joint7] for 7-DOF arm (held fixed during base motion)
  - Index 10: Gripper state (held fixed during base motion)
- Base controller modulates [dx_base, dy_base, dtheta_base] to follow waypoints
- Arm joints and gripper remain at home/safe configuration during navigation

Goal Conditions:
- Position error: ||base_position - target_position|| < 0.05m (tolerance)
- Orientation not constrained (goal only checks position distance)
- Robot base must be within threshold distance of target

Key Differences from Motion2D:
- 3D environment with full robot (base + arm + gripper)
- Uses PyBullet physics simulation for motion planning
- Base is circular (not point mass with radius in 2D)
- Action space includes arm joints and gripper (though not actuated for base motion task)
- No obstacles in BaseMotion3D environment (unlike Motion2D)

Implementation Notes:
- Parameterized skill abstracts away low-level motion planning details
- Users only specify target position, planner handles trajectory generation
- Suitable for mobile manipulation tasks where base positioning is critical step
- Foundation for more complex tasks involving arm manipulation after base motion
"""

    def _create_variant_markdown_description(self) -> str:
        # pylint: disable=line-too-long
        return "This environment has only one variant."

    def _create_reward_markdown_description(self) -> str:
        """Create reward description."""
        # pylint: disable=line-too-long
        return (
            """The reward is -1 per timestep to encourage reaching the goal quickly."""
        )

    def _create_references_markdown_description(self) -> str:
        """Create references description."""
        # pylint: disable=line-too-long
        return """This is a very common kind of environment. The background is adapted from the [Replica dataset](https://arxiv.org/abs/1906.05797) (Straub et al., 2019)."""
