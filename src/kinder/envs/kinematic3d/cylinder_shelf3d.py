"""PyBullet environment where tall cylinders must be picked from the ground and placed
on a shelf.

The cylinders are tall enough that a side grasp (horizontal approach) is required rather
than a top grasp. Because cylinders are rotationally symmetric, any horizontal approach
angle is a valid grasp direction.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
from typing import Type as TypingType

import pybullet as p
from pybullet_helpers.geometry import Pose, get_pose, set_pose
from pybullet_helpers.utils import create_pybullet_cylinder, create_pybullet_shelf
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

# The "object_type" feature value for cylinders. Cuboids use -1.0 (see
# ObjectCentricKinematic3DRobotEnv._create_state_dict).
CYLINDER_OBJECT_TYPE = 1.0


@dataclass(frozen=True)
class CylinderShelf3DEnvConfig(Kinematic3DEnvConfig, metaclass=FinalConfigMeta):
    """Config for CylinderShelf3DEnv()."""

    max_action_mag: float = 0.2

    # Shelf.
    shelf_pose: Pose = Pose((2.0, 2.4, 0.02))
    shelf_rgba: tuple[float, float, float, float] = (0.5, 0.5, 0.5, 1.0)
    shelf_width: float = 0.60198
    shelf_depth: float = 0.254
    shelf_height: float = 0.0127
    shelf_spacing: float = 0.254
    shelf_support_width: float = 0.0127
    shelf_num_layers: int = 4
    shelf_texture: Path = Path(__file__).parent / "assets" / "dark-wood-texture.png"

    # World bounds.
    x_lb: float = -1.0
    x_ub: float = 1.0
    y_lb: float = -1.0
    y_ub: float = 1.0

    # Cylinders. Heights are cycled by cylinder index, so multi-object
    # variants get cylinders of different heights. All heights should stay
    # comfortably below shelf_spacing so a placed cylinder fits between
    # shelf layers.
    cylinder_radius: float = 0.03
    cylinder_heights: tuple[float, ...] = (0.20, 0.15, 0.175)
    cylinder_rgbas: tuple[tuple[float, float, float, float], ...] = (
        (0.8, 0.2, 0.2, 1.0),
        (0.2, 0.5, 0.8, 1.0),
        (0.2, 0.7, 0.3, 1.0),
        (0.8, 0.6, 0.2, 1.0),
        (0.6, 0.3, 0.7, 1.0),
    )

    # Gripper.
    gripper_open_threshold: float = 0.01

    # Goal checking: tolerance below the first shelf layer for determining if
    # a cylinder is "on the shelf". Cylinders must be above (shelf_pose.z +
    # shelf_spacing - on_shelf_z_tolerance) to count as placed.
    on_shelf_z_tolerance: float = 0.05

    def get_camera_kwargs(self) -> dict[str, Any]:
        """Get kwargs to pass to PyBullet camera."""
        return {
            "camera_target": (0, 0, 0),
            "camera_yaw": 0,
            "camera_distance": 2.0,
            "camera_pitch": -20,
        }

    def get_cylinder_height(self, idx: int) -> float:
        """Get the height of the cylinder with the given index."""
        return self.cylinder_heights[idx % len(self.cylinder_heights)]

    def get_cylinder_rgba(self, idx: int) -> tuple[float, float, float, float]:
        """Get the color of the cylinder with the given index."""
        return self.cylinder_rgbas[idx % len(self.cylinder_rgbas)]


class CylinderShelf3DObjectCentricState(Kinematic3DObjectCentricState):
    """A state in the CylinderShelf3DEnv().

    Adds convenience methods on top of Kinematic3DObjectCentricState().
    """


class ObjectCentricCylinderShelf3DEnv(
    ObjectCentricKinematic3DRobotEnv[
        Kinematic3DObjectCentricState, CylinderShelf3DEnvConfig
    ]
):
    """PyBullet environment where tall cylinders must be picked from the ground and
    placed on a shelf."""

    def __init__(
        self,
        num_cylinders: int = 1,
        config: CylinderShelf3DEnvConfig = CylinderShelf3DEnvConfig(),
        **kwargs,
    ) -> None:
        super().__init__(config=config, **kwargs)
        self._num_cylinders = num_cylinders

        # Create the cylinders, but their poses will be reset (with collision
        # checking) in the reset() method.
        self._cylinders: dict[str, int] = {}
        for idx in range(self._num_cylinders):
            cylinder_id = create_pybullet_cylinder(
                self.config.get_cylinder_rgba(idx),
                self.config.cylinder_radius,
                self.config.get_cylinder_height(idx),
                physics_client_id=self.physics_client_id,
            )
            self._cylinders[f"cylinder{idx}"] = cylinder_id

        # Create shelf.
        self._shelf_id, self._shelf_surface_ids = create_pybullet_shelf(
            color=self.config.shelf_rgba,
            shelf_width=self.config.shelf_width,
            shelf_depth=self.config.shelf_depth,
            shelf_height=self.config.shelf_height,
            spacing=self.config.shelf_spacing,
            support_width=self.config.shelf_support_width,
            num_layers=self.config.shelf_num_layers,
            physics_client_id=self.physics_client_id,
        )
        set_pose(self._shelf_id, self.config.shelf_pose, self.physics_client_id)

        shelf_texture_id = p.loadTexture(
            str(self.config.shelf_texture), self.physics_client_id
        )
        for shelf_link_id in range(
            p.getNumJoints(self._shelf_id, physicsClientId=self.physics_client_id)
        ):
            p.changeVisualShape(
                self._shelf_id,
                shelf_link_id,
                textureUniqueId=shelf_texture_id,
                physicsClientId=self.physics_client_id,
            )

    @property
    def state_cls(self) -> TypingType[Kinematic3DObjectCentricState]:
        return CylinderShelf3DObjectCentricState

    def _create_constant_initial_state_dict(self) -> dict[Object, dict[str, float]]:
        return self._create_state_dict([("shelf", Kinematic3DFixtureType)])

    def _reset_objects(self) -> None:
        # Sample one cylinder at a time because each has its own height, and
        # therefore its own resting z. The sampler adds each placed cylinder
        # to the collision set for the next one.
        placed_ids: set[int] = {self.robot.base.robot_id}
        for idx in range(self._num_cylinders):
            cylinder_id = self._cylinders[f"cylinder{idx}"]
            half_height = self.config.get_cylinder_height(idx) / 2
            sample_collision_free_object_poses(
                object_ids={cylinder_id},
                lb=(self.config.x_lb, self.config.y_lb, half_height),
                ub=(self.config.x_ub, self.config.y_ub, half_height),
                physics_client_id=self.physics_client_id,
                rng=self.np_random,
                other_collision_ids=placed_ids,
            )
            placed_ids.add(cylinder_id)

    def _set_object_states(self, obs: Kinematic3DObjectCentricState) -> None:
        assert isinstance(obs, CylinderShelf3DObjectCentricState)
        for cylinder_name, cylinder_id in self._cylinders.items():
            assert cylinder_id is not None
            set_pose(
                cylinder_id,
                obs.get_object_pose(cylinder_name),
                self.physics_client_id,
            )

    def _object_name_to_pybullet_id(self, object_name: str) -> int:
        if object_name == "shelf":
            return self._shelf_id
        if object_name.startswith("cylinder"):
            return self._cylinders[object_name]
        raise ValueError(f"Unrecognized object name: {object_name}")

    def _get_collision_object_ids(self) -> set[int]:
        collision_ids = {self._shelf_id} | set(self._cylinders.values())
        return collision_ids

    def _get_movable_object_names(self) -> set[str]:
        return set(self._cylinders.keys())

    def _get_surface_object_names(self) -> set[str]:
        return {"shelf"}

    def _get_half_extents(self, object_name: str) -> tuple[float, float, float]:
        if object_name.startswith("cylinder"):
            idx = int(object_name[len("cylinder") :])
            return (
                self.config.cylinder_radius,
                self.config.cylinder_radius,
                self.config.get_cylinder_height(idx) / 2,
            )
        raise ValueError(f"Unrecognized object name: {object_name}")

    def _get_obs(self) -> CylinderShelf3DObjectCentricState:
        state_dict = self._create_state_dict(
            [("robot", Kinematic3DRobotType)]
            + [("shelf", Kinematic3DFixtureType)]
            + [
                ("cylinder" + str(i), Kinematic3DCuboidType)
                for i in range(self._num_cylinders)
            ]
        )
        state = create_state_from_dict(
            state_dict,
            Kinematic3DEnvTypeFeatures,
            state_cls=CylinderShelf3DObjectCentricState,
        )
        assert isinstance(state, CylinderShelf3DObjectCentricState)
        # The generic cuboid serializer tags every Kinematic3DCuboidType
        # object as a cuboid; re-tag the cylinders.
        for cylinder_name in self._cylinders:
            obj = state.get_object_from_name(cylinder_name)
            state.set(obj, "object_type", CYLINDER_OBJECT_TYPE)
        return state

    def goal_reached(self) -> bool:
        robot_gripper_pose = self._robot_arm.get_finger_state()
        if robot_gripper_pose > self.config.gripper_open_threshold:
            return False
        # Check that all cylinders are above the first shelf layer (with
        # tolerance).
        min_on_shelf_z = (
            self.config.shelf_pose.position[2]
            + self.config.shelf_spacing
            - self.config.on_shelf_z_tolerance
        )
        for _, cylinder_id in self._cylinders.items():
            cylinder_pose = get_pose(cylinder_id, self.physics_client_id)
            if cylinder_pose.position[2] < min_on_shelf_z:
                return False

        return True


class CylinderShelf3DEnv(ConstantObjectKinDEREnv):
    """Cylinder shelf 3D env with a constant number of objects."""

    def __init__(self, num_cylinders: int = 1, **kwargs) -> None:
        self._num_cylinders = num_cylinders
        super().__init__(num_cylinders=num_cylinders, **kwargs)

    def _create_object_centric_env(
        self, *args, **kwargs
    ) -> ObjectCentricKinematic3DRobotEnv:
        return ObjectCentricCylinderShelf3DEnv(*args, **kwargs)

    def _get_constant_object_names(
        self, exemplar_state: ObjectCentricState
    ) -> list[str]:
        constant_objects = ["robot", "shelf"]
        for obj in exemplar_state:
            if obj.name.startswith("cylinder"):
                constant_objects.append(obj.name)
        return constant_objects

    def _create_env_markdown_description(self) -> str:
        """Create environment description."""
        # pylint: disable=line-too-long
        return """A 3D environment where the goal is to pick up tall cylinders from the ground and place them onto a shelf. The cylinders are tall enough that they must be grasped from the side rather than from the top."""

    def _create_variant_markdown_description(self) -> str:
        # pylint: disable=line-too-long
        return "The number of cylinders differs between environment variants. For example, CylinderShelf3D-o1 has 1 cylinder, while CylinderShelf3D-o3 has 3 cylinders of different heights."

    def _create_variant_specific_description(self) -> str:
        if self._num_cylinders == 1:
            return "This variant has 1 cylinder to place on the shelf."
        return (
            f"This variant has {self._num_cylinders} cylinders of different "
            "heights to place on the shelf."
        )

    def _create_reward_markdown_description(self) -> str:
        """Create reward description."""
        # pylint: disable=line-too-long
        return """The reward is -1 per timestep to encourage efficient task completion. The episode terminates successfully when all cylinders are placed on the shelf (i.e., above the first shelf layer) and the gripper is closed. The gripper must be closed to prevent accidental "success" while a cylinder is still being held above the shelf."""

    def _create_references_markdown_description(self) -> str:
        """Create references description."""
        # pylint: disable=line-too-long
        return """This is a very common kind of environment. The background is adapted from the [Replica dataset](https://arxiv.org/abs/1906.05797) (Straub et al., 2019)."""
