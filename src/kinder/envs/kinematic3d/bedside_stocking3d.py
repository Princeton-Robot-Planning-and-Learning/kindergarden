"""PyBullet environment where a mobile manipulator stocks the bedside of a seated user.

The user sits up in bed. Requested items start on far furniture (a book cart and a
dresser) and must end up where the user can reach them. The user's reach is an
ellipsoid centered at their right shoulder with semi-axes `reach_theta`; an item counts
as reachable if `reach_margin >= 0` and the segment from the shoulder to the item is not
blocked by another movable object. Clutter may go anywhere.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from typing import Type as TypingType

import numpy as np
import pybullet as p
from numpy.typing import NDArray
from pybullet_helpers.geometry import Pose, SE2Pose, get_pose, set_pose
from pybullet_helpers.robots.human import Human
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
from kinder.envs.kinematic3d.utils import Kinematic3DObjectCentricState

Region = tuple[tuple[float, float], tuple[float, float], float]

WOOD = (0.62, 0.45, 0.30, 1.0)
WOOD_DARK = (0.45, 0.32, 0.22, 1.0)
LINEN = (0.92, 0.93, 0.96, 1.0)
BOOK_COLORS = [
    (0.85, 0.15, 0.15, 1.0),
    (0.15, 0.35, 0.85, 1.0),
    (0.10, 0.65, 0.25, 1.0),
    (0.95, 0.55, 0.05, 1.0),
]
SMALL_ITEMS = [
    ("cup", (0.035, 0.035, 0.05), (0.95, 0.85, 0.10, 1.0)),
    ("pills", (0.025, 0.025, 0.045), (0.95, 0.40, 0.10, 1.0)),
    ("remote", (0.025, 0.06, 0.04), (0.20, 0.20, 0.25, 1.0)),
    ("glasses", (0.06, 0.03, 0.04), (0.55, 0.15, 0.65, 1.0)),
]
BOOK_HALF_EXTENTS = (0.04, 0.09, 0.02)
CLUTTER_SHAPES = [
    (0.05, 0.035, 0.04),
    (0.04, 0.04, 0.035),
    (0.035, 0.035, 0.04),
    (0.05, 0.04, 0.03),
]
CLUTTER_COLORS = [
    (0.55, 0.55, 0.58, 1.0),
    (0.70, 0.66, 0.60, 1.0),
    (0.48, 0.50, 0.54, 1.0),
    (0.66, 0.62, 0.66, 1.0),
]


def reach_margin(
    points: NDArray, theta: NDArray, shoulder: NDArray
) -> NDArray[np.float64]:
    """m(x, theta) = 1 - ||(x - shoulder) / theta||; broadcasts over leading dims."""
    d = (np.asarray(points, dtype=float) - shoulder) / np.asarray(theta, dtype=float)
    return 1.0 - np.linalg.norm(d, axis=-1)


@dataclass(frozen=True)
class BedsideStocking3DEnvConfig(Kinematic3DEnvConfig, metaclass=FinalConfigMeta):
    """Config for BedsideStocking3DEnv()."""

    realistic_bg: bool = False
    x_lb: float = -1.0
    x_ub: float = 3.0
    y_lb: float = -1.5
    y_ub: float = 2.0
    robot_base_home_pose: SE2Pose = SE2Pose(1.5, 0.3, np.pi)
    max_action_mag: float = 0.2
    gripper_open_threshold: float = 0.01

    # The user sits facing +y with the pelvis at this pose; the right arm faces +x.
    human_base_pose: Pose = Pose.from_rpy((0.0, 0.0, 0.56), (0.0, 0.0, np.pi))
    reach_theta: tuple[float, float, float] = (0.80, 0.62, 0.72)
    reach_theta_lb: tuple[float, float, float] = (0.45, 0.35, 0.40)
    reach_theta_ub: tuple[float, float, float] = (0.95, 0.80, 0.90)
    sample_theta_on_reset: bool = False
    arm_radius: float = 0.04

    # Open-top surfaces near the bed: (x range, y range, top z).
    bedside_regions: dict[str, Region] = field(
        default_factory=lambda: {
            "nightstand": ((0.56, 0.96), (-0.40, 0.08), 0.72),
            "side_table": ((0.56, 0.90), (0.16, 0.50), 0.60),
            "foot_chest": ((0.56, 0.96), (0.75, 1.25), 0.50),
        }
    )
    # Far furniture where requested items start.
    book_cart: Region = ((2.10, 2.50), (0.70, 1.50), 0.75)
    dresser: Region = ((2.05, 2.40), (-0.70, 0.30), 0.60)

    slot_spacing: float = 0.11
    book_spacing: float = 0.13

    def get_camera_kwargs(self) -> dict[str, Any]:
        """Get kwargs to pass to PyBullet camera."""
        return {
            "camera_target": (1.0, 0.2, 0.5),
            "camera_yaw": 120,
            "camera_distance": 3.2,
            "camera_pitch": -40,
        }

    @property
    def surface_regions(self) -> dict[str, Region]:
        """All placement surfaces by name."""
        return {
            **self.bedside_regions,
            "book_cart": self.book_cart,
            "dresser": self.dresser,
        }

    def region_slots(self, name: str) -> NDArray[np.float64]:
        """Grid of (x, y, top z) slot centers on a surface."""
        (x0, x1), (y0, y1), z = self.surface_regions[name]
        s = self.slot_spacing
        xs = np.arange(x0 + s / 2, x1 - s / 2 + 1e-9, s)
        ys = np.arange(y0 + s / 2, y1 - s / 2 + 1e-9, s)
        return np.array([[x, y, z] for x in xs for y in ys])


class BedsideStocking3DObjectCentricState(Kinematic3DObjectCentricState):
    """A state in the BedsideStocking3DEnv()."""

    @property
    def requested_objects(self) -> list[str]:
        """Names of the requested items."""
        return sorted(o.name for o in self if o.name.startswith("req"))

    @property
    def clutter_objects(self) -> list[str]:
        """Names of the clutter items."""
        return sorted(o.name for o in self if o.name.startswith("clutter"))


class ObjectCentricBedsideStocking3DEnv(
    ObjectCentricKinematic3DRobotEnv[
        Kinematic3DObjectCentricState, BedsideStocking3DEnvConfig
    ]
):
    """Stock the bedside of a seated user so every requested item is in reach."""

    def __init__(
        self,
        num_requested: int = 4,
        num_clutter: int = 2,
        config: BedsideStocking3DEnvConfig = BedsideStocking3DEnvConfig(),
        **kwargs,
    ) -> None:
        super().__init__(config=config, **kwargs)
        self._num_requested = num_requested
        self._num_clutter = num_clutter
        self.reach_theta = np.array(self.config.reach_theta, dtype=float)

        self._surfaces: dict[str, int] = {}
        self._static_ids: set[int] = set()
        self._build_furniture()

        legs = {"home_joint_positions": [-np.pi / 2, 0.0, 0.0, 0.0, 0.0, 0.0]}
        self.human = Human(
            self.physics_client_id,
            base_pose=self.config.human_base_pose,
            right_leg_kwargs=legs,
            left_leg_kwargs=legs,
        )
        self.human_ids = {
            self.human.torso,
            self.human.right_arm.robot_id,
            self.human.left_arm.robot_id,
            self.human.right_leg.robot_id,
            self.human.left_leg.robot_id,
        }
        self.shoulder = np.array(self.human.right_arm.get_base_pose().position)

        self._items: dict[str, int] = {}
        self._half_extents: dict[str, tuple[float, float, float]] = {}
        for i in range(num_requested):
            if i < 4 or i >= 4 + len(SMALL_ITEMS):
                he, color = BOOK_HALF_EXTENTS, BOOK_COLORS[i % len(BOOK_COLORS)]
            else:
                _, he, color = SMALL_ITEMS[i - 4]
            self._add_item(f"req{i}", he, color)
        for i in range(num_clutter):
            self._add_item(
                f"clutter{i}",
                CLUTTER_SHAPES[i % len(CLUTTER_SHAPES)],
                CLUTTER_COLORS[i % len(CLUTTER_COLORS)],
            )

    def _block(self, lo, hi, color) -> int:
        lo, hi = np.asarray(lo, dtype=float), np.asarray(hi, dtype=float)
        body = create_pybullet_block(
            color, tuple((hi - lo) / 2), physics_client_id=self.physics_client_id
        )
        set_pose(body, Pose(tuple((hi + lo) / 2)), self.physics_client_id)
        return body

    def _build_furniture(self) -> None:
        cfg = self.config
        bed = [
            self._block((-0.52, -0.45, 0.0), (0.52, 1.55, 0.40), WOOD_DARK),
            self._block((-0.50, -0.43, 0.40), (0.50, 1.53, 0.56), LINEN),
            self._block((-0.52, -0.50, 0.0), (0.52, -0.45, 1.00), WOOD_DARK),
            self._block((-0.46, -0.42, 0.56), (0.46, -0.30, 1.15), LINEN),
        ]
        self._static_ids |= set(bed)
        self._surfaces["bed"] = bed[1]
        for name, ((x0, x1), (y0, y1), z) in cfg.surface_regions.items():
            pad = 0.02
            color = WOOD_DARK if name in ("foot_chest", "book_cart") else WOOD
            top = self._block((x0 - pad, y0 - pad, 0.0), (x1 + pad, y1 + pad, z), color)
            self._static_ids.add(top)
            self._surfaces[name] = top

    def _add_item(self, name: str, he, color) -> None:
        body = create_pybullet_block(color, he, physics_client_id=self.physics_client_id)
        self._items[name] = body
        self._half_extents[name] = tuple(he)

    @property
    def state_cls(self) -> TypingType[Kinematic3DObjectCentricState]:
        return BedsideStocking3DObjectCentricState

    @property
    def requested_objects(self) -> list[str]:
        """Names of the requested items."""
        return sorted(n for n in self._items if n.startswith("req"))

    @property
    def clutter_objects(self) -> list[str]:
        """Names of the clutter items."""
        return sorted(n for n in self._items if n.startswith("clutter"))

    @property
    def static_collision_ids(self) -> set[int]:
        """Furniture and the human."""
        return set(self._static_ids) | set(self.human_ids)

    def surface_id(self, name: str) -> int:
        """PyBullet id of a named surface."""
        return self._surfaces[name]

    def resting_pose(self, name: str, xy: tuple[float, float], surface: str) -> Pose:
        """Pose of item `name` resting on `surface` at `xy`."""
        z = self.config.surface_regions[surface][2]
        gap = self.config.min_placement_dist / 2
        return Pose((xy[0], xy[1], z + self._half_extents[name][2] + gap))

    def _create_constant_initial_state_dict(self) -> dict[Object, dict[str, float]]:
        return self._create_state_dict(
            [(s, Kinematic3DFixtureType) for s in sorted(self._surfaces)]
        )

    def _reset_objects(self) -> None:
        cfg = self.config
        rng = self.np_random
        if cfg.sample_theta_on_reset:
            self.reach_theta = rng.uniform(cfg.reach_theta_lb, cfg.reach_theta_ub)
        else:
            self.reach_theta = np.array(cfg.reach_theta, dtype=float)
        for body in self._items.values():
            set_pose(body, Pose((10.0, 10.0, -5.0)), self.physics_client_id)

        books = [n for n in self.requested_objects if self._half_extents[n] == BOOK_HALF_EXTENTS]
        small = [n for n in self.requested_objects if n not in books]
        (bx0, bx1), (by0, _), _ = cfg.book_cart
        for i, name in enumerate(books):
            xy = ((bx0 + bx1) / 2, by0 + 0.10 + i * cfg.book_spacing)
            self._place(name, self.resting_pose(name, xy, "book_cart"))
        dresser_slots = cfg.region_slots("dresser")
        for name, slot in zip(small, rng.permutation(dresser_slots)[: len(small)]):
            self._place(name, self.resting_pose(name, tuple(slot[:2]), "dresser"))

        bedside = [(r, s) for r in cfg.bedside_regions for s in cfg.region_slots(r)]
        reach = np.array([reach_margin(s, self.reach_theta, self.shoulder) >= 0 for _, s in bedside])
        order = list(rng.permutation(np.flatnonzero(reach))) + list(rng.permutation(np.flatnonzero(~reach)))
        for name, k in zip(self.clutter_objects, order):
            region, slot = bedside[k]
            self._place(name, self.resting_pose(name, tuple(slot[:2]), region))

    def _place(self, name: str, pose: Pose) -> None:
        set_pose(self._items[name], pose, self.physics_client_id)

    def _set_object_states(self, obs: Kinematic3DObjectCentricState) -> None:
        for name, body in self._items.items():
            set_pose(body, obs.get_object_pose(name), self.physics_client_id)

    def _object_name_to_pybullet_id(self, object_name: str) -> int:
        if object_name in self._items:
            return self._items[object_name]
        if object_name in self._surfaces:
            return self._surfaces[object_name]
        raise ValueError(f"Unrecognized object name: {object_name}")

    def _get_collision_object_ids(self) -> set[int]:
        return self.static_collision_ids | set(self._items.values())

    def _get_movable_object_names(self) -> set[str]:
        return set(self._items)

    def _get_surface_object_names(self) -> set[str]:
        return set(self._surfaces) - {"bed"}

    def _get_half_extents(self, object_name: str) -> tuple[float, float, float]:
        return self._half_extents[object_name]

    def _get_obs(self) -> BedsideStocking3DObjectCentricState:
        state_dict = self._create_state_dict(
            [("robot", Kinematic3DRobotType)]
            + [(s, Kinematic3DFixtureType) for s in sorted(self._surfaces)]
            + [(n, Kinematic3DCuboidType) for n in sorted(self._items)]
        )
        state = create_state_from_dict(
            state_dict,
            Kinematic3DEnvTypeFeatures,
            state_cls=BedsideStocking3DObjectCentricState,
        )
        assert isinstance(state, BedsideStocking3DObjectCentricState)
        return state

    def grasp_point(self, name: str, pose: Pose | None = None) -> NDArray[np.float64]:
        """Top center of an item, where the user would grab it."""
        if pose is None:
            pose = get_pose(self._items[name], self.physics_client_id)
        return np.array(pose.position) + [0.0, 0.0, self._half_extents[name][2]]

    def is_unobstructed(self, name: str) -> bool:
        """Whether rays from the shoulder to the item's grasp point miss other items."""
        target = self.grasp_point(name)
        return not segment_blocked(
            self.shoulder,
            target,
            {b for n, b in self._items.items() if n != name},
            self.config.arm_radius,
            self.physics_client_id,
        )

    def item_reachable(self, name: str) -> bool:
        """Whether the user can reach item `name` where it is now."""
        m = reach_margin(self.grasp_point(name), self.reach_theta, self.shoulder)
        return bool(m >= 0) and self.is_unobstructed(name)

    def goal_reached(self) -> bool:
        if self._grasped_object is not None:
            return False
        for name in self.requested_objects:
            body = self._items[name]
            if not self._get_surfaces_supporting_object(body):
                return False
            if not self.item_reachable(name):
                return False
        return True


def segment_blocked(
    start: NDArray,
    end: NDArray,
    body_ids: set[int],
    radius: float,
    physics_client_id: int,
) -> bool:
    """Whether a thick segment (center ray plus four offset rays) hits any of
    `body_ids`."""
    if not body_ids:
        return False
    start, end = np.asarray(start, dtype=float), np.asarray(end, dtype=float)
    d = end - start
    d /= np.linalg.norm(d)
    u = np.cross(d, [0.0, 0.0, 1.0])
    if np.linalg.norm(u) < 1e-6:
        u = np.array([1.0, 0.0, 0.0])
    u /= np.linalg.norm(u)
    v = np.cross(d, u)
    offsets = [np.zeros(3), radius * u, -radius * u, radius * v, -radius * v]
    starts = [start + o for o in offsets]
    ends = [end - 0.01 * d + o for o in offsets]
    hits = p.rayTestBatch(starts, ends, physicsClientId=physics_client_id)
    return any(h[0] in body_ids for h in hits)


class BedsideStocking3DEnv(ConstantObjectKinDEREnv):
    """Bedside stocking env with a constant number of objects."""

    def __init__(self, num_requested: int = 4, num_clutter: int = 2, **kwargs) -> None:
        self._num_requested = num_requested
        self._num_clutter = num_clutter
        super().__init__(num_requested=num_requested, num_clutter=num_clutter, **kwargs)

    def _create_object_centric_env(
        self, *args, **kwargs
    ) -> ObjectCentricKinematic3DRobotEnv:
        return ObjectCentricBedsideStocking3DEnv(*args, **kwargs)

    def _get_constant_object_names(
        self, exemplar_state: ObjectCentricState
    ) -> list[str]:
        names = ["robot"]
        names += sorted(o.name for o in exemplar_state if o.type == Kinematic3DFixtureType)
        names += sorted(o.name for o in exemplar_state if o.type == Kinematic3DCuboidType)
        return names

    def _create_env_markdown_description(self) -> str:
        # pylint: disable=line-too-long
        return """A 3D environment where a mobile manipulator stocks the bedside of a user sitting up in bed. Requested items (books, cup, pills, remote, glasses) start on a far book cart and dresser and must be placed on bedside furniture (nightstand, side table, foot chest) where the user can reach them. The user's reach is an ellipsoid at the right shoulder whose semi-axes are hidden parameters. Clutter occupies the best bedside spots and may be moved anywhere."""

    def _create_variant_markdown_description(self) -> str:
        # pylint: disable=line-too-long
        return "Variants differ in the number of requested items (r) and clutter items (c). For example, BedsideStocking3D-r4-c2 has 4 requested items and 2 clutter items."

    def _create_variant_specific_description(self) -> str:
        return (
            f"This variant has {self._num_requested} requested items and "
            f"{self._num_clutter} clutter items."
        )

    def _create_reward_markdown_description(self) -> str:
        # pylint: disable=line-too-long
        return """The reward is -1 per timestep. The episode terminates when the gripper is empty and every requested item rests on a surface, inside the user's reach ellipsoid, with an unobstructed straight path from the shoulder."""

    def _create_references_markdown_description(self) -> str:
        # pylint: disable=line-too-long
        return """The seated human model and the reachability-personalization setting follow the books environment of [Coloring Between the Lines](https://arxiv.org/abs/2505.15503) (Silver et al., 2025)."""
