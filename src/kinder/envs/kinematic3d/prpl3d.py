"""PyBullet environment loaded from the PRPL lab URDF.

The robot picks cubes from the floor and places them on the countertop.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
from typing import Type as TypingType

import numpy as np
import pybullet as p
from pybullet_helpers.geometry import Pose, SE2Pose, set_pose
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
    sample_collision_free_object_poses,
)

# Path to the PRPL lab URDF.
_PRPL_LAB_URDF = (
    Path(__file__).parent.parent
    / "dynamic3d"
    / "models"
    / "assets"
    / "prpl_lab"
    / "urdf"
    / "PRPL_lab_full_collision.urdf"
)


@dataclass(frozen=True)
class PrplLab3DEnvConfig(Kinematic3DEnvConfig, metaclass=FinalConfigMeta):
    """Config for PrplLab3DEnv."""

    # URDF root pose.  The cabinet fronts face -y, so a robot at y≈0 looks
    # directly into the open doors.  x=-0.5 shifts the lab left so the stove
    # (at the +x end of the counter run) clears the room's right wall.
    lab_pose: Pose = Pose((-1.0, 2.0, 0.0))

    # Block geometry and colour.
    block_half_extents: tuple[float, float, float] = (0.05, 0.025, 0.025)
    block_rgba: tuple[float, float, float, float] = (1.0, 0.5, 0.0, 1.0)

    # World bounds for cube spawn (floor area in front of the lab).
    x_lb: float = -1.0
    x_ub: float = 1.6
    y_lb: float = -1.5
    y_ub: float = -0.2

    # Robot home: centred in x on the cabinet run, facing +y toward the lab.
    robot_base_home_pose: SE2Pose = SE2Pose(0.3, 0.0, np.pi / 2)

    def get_camera_kwargs(self) -> dict[str, Any]:
        """Get kwargs to pass to PyBullet camera."""
        return {
            "camera_target": (0.0, -0.9, 0.8),
            "camera_yaw": -70,
            "camera_distance": 4.0,
            "camera_pitch": -20,
        }


class PrplLab3DObjectCentricState(Kinematic3DObjectCentricState):
    """State for PrplLab3DEnv."""


class ObjectCentricPrplLab3DEnv(
    ObjectCentricKinematic3DRobotEnv[PrplLab3DObjectCentricState, PrplLab3DEnvConfig]
):
    """Inner PyBullet environment: loads the PRPL lab URDF and spawns cubes."""

    def __init__(
        self,
        num_cubes: int = 2,
        config: PrplLab3DEnvConfig = PrplLab3DEnvConfig(),
        **kwargs,
    ) -> None:
        super().__init__(config=config, **kwargs)
        self._num_cubes = num_cubes

        # Load PRPL lab URDF.
        self._lab_id = p.loadURDF(
            str(_PRPL_LAB_URDF),
            basePosition=list(config.lab_pose.position),
            baseOrientation=list(config.lab_pose.orientation),
            physicsClientId=self.physics_client_id,
            useFixedBase=True,
        )

        # Cubes (poses randomised in _reset_objects).
        self._cubes: dict[str, int] = {}
        for idx in range(self._num_cubes):
            cube_id = create_pybullet_block(
                config.block_rgba,
                config.block_half_extents,
                physics_client_id=self.physics_client_id,
            )
            self._cubes[f"cube{idx}"] = cube_id

    # ── Abstract-method implementations ──────────────────────────────────────

    @property
    def state_cls(self) -> TypingType[PrplLab3DObjectCentricState]:
        return PrplLab3DObjectCentricState

    def _create_constant_initial_state_dict(self) -> dict[Object, dict[str, float]]:
        return self._create_state_dict([("prpl_lab", Kinematic3DFixtureType)])

    def _reset_objects(self) -> None:
        sample_collision_free_object_poses(
            object_ids=set(self._cubes.values()),
            lb=(
                self.config.x_lb,
                self.config.y_lb,
                self.config.block_half_extents[2],
            ),
            ub=(
                self.config.x_ub,
                self.config.y_ub,
                self.config.block_half_extents[2],
            ),
            physics_client_id=self.physics_client_id,
            rng=self.np_random,
            other_collision_ids={self.robot.base.robot_id, self._lab_id},
        )

    def _set_object_states(self, obs: PrplLab3DObjectCentricState) -> None:
        assert isinstance(obs, PrplLab3DObjectCentricState)
        for cube_name, cube_id in self._cubes.items():
            set_pose(cube_id, obs.get_object_pose(cube_name), self.physics_client_id)

    def _object_name_to_pybullet_id(self, object_name: str) -> int:
        if object_name == "prpl_lab":
            return self._lab_id
        if object_name.startswith("cube"):
            return self._cubes[object_name]
        raise ValueError(f"Unrecognized object name: {object_name}")

    def _get_collision_object_ids(self) -> set[int]:
        return {self._lab_id} | set(self._cubes.values())

    def _get_movable_object_names(self) -> set[str]:
        return set(self._cubes.keys())

    def _get_surface_object_names(self) -> set[str]:
        return {"prpl_lab"}

    def _get_half_extents(self, object_name: str) -> tuple[float, float, float]:
        if object_name.startswith("cube"):
            return self.config.block_half_extents
        raise ValueError(f"Unrecognized object name: {object_name}")

    def _get_obs(self) -> PrplLab3DObjectCentricState:
        state_dict = self._create_state_dict(
            [("robot", Kinematic3DRobotType)]
            + [("prpl_lab", Kinematic3DFixtureType)]
            + [(f"cube{i}", Kinematic3DCuboidType) for i in range(self._num_cubes)]
        )
        state = create_state_from_dict(
            state_dict,
            Kinematic3DEnvTypeFeatures,
            state_cls=PrplLab3DObjectCentricState,
        )
        assert isinstance(state, PrplLab3DObjectCentricState)
        return state

    def _get_surfaces_supporting_object(self, _object_id: int) -> set[int]:
        # Always allow release — no surface-contact requirement for this env.
        return {self._lab_id}

    def goal_reached(self) -> bool:
        # All cubes must be above the counter (z > 0.5 distinguishes counter from floor).
        for cube_id in self._cubes.values():
            pos, _ = p.getBasePositionAndOrientation(
                cube_id, physicsClientId=self.physics_client_id
            )
            if pos[2] < 0.5:
                return False
        return True


class PrplLab3DEnv(ConstantObjectKinDEREnv):
    """Gym wrapper for ObjectCentricPrplLab3DEnv."""

    def __init__(self, num_cubes: int = 2, **kwargs) -> None:
        self._num_cubes = num_cubes
        super().__init__(num_cubes=num_cubes, **kwargs)

    def _create_object_centric_env(
        self, *args, **kwargs
    ) -> ObjectCentricKinematic3DRobotEnv:
        return ObjectCentricPrplLab3DEnv(*args, **kwargs)

    def _get_constant_object_names(
        self, exemplar_state: ObjectCentricState
    ) -> list[str]:
        constant_objects = ["robot", "prpl_lab"]
        for obj in exemplar_state:
            if obj.name.startswith("cube"):
                constant_objects.append(obj.name)
        return constant_objects

    def _create_env_markdown_description(self) -> str:
        """Create environment description."""
        return (
            "A 3D environment loaded from the PRPL lab URDF. "
            "The robot picks cubes from the floor and places them on the countertop."
        )

    def _create_variant_markdown_description(self) -> str:
        return (
            "The number of cubes differs between variants. "
            "For example, PrplLab3D-o1 has 1 cube and PrplLab3D-o2 has 2 cubes."
        )

    def _create_variant_specific_description(self) -> str:
        if self._num_cubes == 1:
            return "This variant has 1 cube to place on the counter."
        return f"This variant has {self._num_cubes} cubes to place on the counter."

    def _create_reward_markdown_description(self) -> str:
        """Create reward description."""
        return "The reward is -1 per timestep. The episode terminates successfully" \
            "when all cubes are on the countertop."

    def _create_references_markdown_description(self) -> str:
        """Create references description."""
        return ""
