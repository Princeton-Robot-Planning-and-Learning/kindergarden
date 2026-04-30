"""Utility functions shared across different types of environments."""

import abc
from collections.abc import Callable

import matplotlib.pyplot as plt
import numpy as np
from gymnasium.spaces import Box
from numpy.typing import NDArray
from prpl_utils.utils import fig2data
from relational_structs import (
    Object,
    ObjectCentricState,
)
from tomsgeoms2d.structs import Circle, Geom2D, Lobject, Rectangle, Tobject
from tomsgeoms2d.utils import geom2ds_intersect

from kinder.envs.dynamic2d.object_types import (
    DotRobotType,
    DynRectangleType,
    KinRectangleType,
    KinRobotType,
)
from kinder.envs.dynamic2d.object_types import LObjectType as LObjectTypeDyn
from kinder.envs.dynamic2d.object_types import (
    SmallCircleType,
    SmallSquareType,
    TObjectType,
)
from kinder.envs.kinematic2d.object_types import (
    CircleType,
    CRVRobotType,
    DoubleRectType,
)
from kinder.envs.kinematic2d.object_types import LObjectType as LObjectTypeGeom
from kinder.envs.kinematic2d.object_types import (
    RectangleType,
)
from kinder.envs.kinematic2d.structs import (
    Body2D,
    MultiBody2D,
    SE2Pose,
    ZOrder,
    z_orders_may_collide,
)

PURPLE: tuple[float, float, float] = (128 / 255, 0 / 255, 128 / 255)
BLACK: tuple[float, float, float] = (0.1, 0.1, 0.1)
BROWN: tuple[float, float, float] = (0.4, 0.2, 0.1)
ORANGE: tuple[float, float, float] = (1.0, 165 / 255, 0.0)


class RobotActionSpace(Box):
    """A space for robot actions."""

    def __init__(self, low, high, dtype=np.float32, **kwargs) -> None:
        super().__init__(
            np.asarray(low, dtype=dtype),
            np.asarray(high, dtype=dtype),
            dtype=dtype,
            **kwargs,
        )

    @abc.abstractmethod
    def create_markdown_description(self) -> str:
        """Create a markdown description of this space."""


def get_se2_pose(state: ObjectCentricState, obj: Object) -> SE2Pose:
    """Get the SE2Pose of an object in a given state."""
    return SE2Pose(
        x=state.get(obj, "x"),
        y=state.get(obj, "y"),
        theta=state.get(obj, "theta"),
    )


def get_relative_se2_transform(
    state: ObjectCentricState, obj1: Object, obj2: Object
) -> SE2Pose:
    """Get the pose of obj2 in the frame of obj1."""
    world_to_obj1 = get_se2_pose(state, obj1)
    world_to_obj2 = get_se2_pose(state, obj2)
    return world_to_obj1.inverse * world_to_obj2


def sample_se2_pose(
    bounds: tuple[SE2Pose, SE2Pose], rng: np.random.Generator
) -> SE2Pose:
    """Sample a SE2Pose uniformly between the bounds."""
    lb, ub = bounds
    x = rng.uniform(lb.x, ub.x)
    y = rng.uniform(lb.y, ub.y)
    theta = rng.uniform(lb.theta, ub.theta)
    return SE2Pose(x, y, theta)


def state_2d_has_collision(
    state: ObjectCentricState,
    group1: set[Object],
    group2: set[Object],
    static_object_cache: dict[Object, MultiBody2D],
    ignore_z_orders: bool = False,
) -> bool:
    """Check for collisions between any objects in two groups."""
    # Create multibodies once.
    obj_to_multibody = {
        o: object_to_multibody2d(o, state, static_object_cache) for o in state
    }
    # Check pairwise collisions.
    for obj1 in group1:
        for obj2 in group2:
            obj1_static = (
                state.get(obj1, "static")
                if "static" in state.type_features[obj1.type]
                else False
            )
            obj2_static = (
                state.get(obj2, "static")
                if "static" in state.type_features[obj2.type]
                else False
            )
            if obj1 == obj2 or (obj1_static and obj2_static):
                # Skip self-collision and static-static collision.
                continue
            multibody1 = obj_to_multibody[obj1]
            multibody2 = obj_to_multibody[obj2]
            for body1 in multibody1.bodies:
                for body2 in multibody2.bodies:
                    if not (
                        ignore_z_orders
                        or z_orders_may_collide(body1.z_order, body2.z_order)
                    ):
                        continue
                    if geom2ds_intersect(body1.geom, body2.geom):
                        return True
    return False


def render_2dstate_on_ax(
    state: ObjectCentricState,
    ax: plt.Axes,
    static_object_body_cache: dict[Object, MultiBody2D] | None = None,
) -> None:
    """Render a state on an existing plt.Axes."""
    if static_object_body_cache is None:
        static_object_body_cache = {}

    # Sort objects by ascending z order, with the robot first.
    def _render_order(obj: Object) -> int:
        if obj.is_instance(CRVRobotType) or obj.is_instance(KinRobotType):
            return -1
        return int(state.get(obj, "z_order"))

    for obj in sorted(state, key=_render_order):
        body = object_to_multibody2d(obj, state, static_object_body_cache)
        body.plot(ax)


def render_2dstate(
    state: ObjectCentricState,
    static_object_body_cache: dict[Object, MultiBody2D] | None = None,
    world_min_x: float = 0.0,
    world_max_x: float = 10.0,
    world_min_y: float = 0.0,
    world_max_y: float = 10.0,
    render_dpi: int = 150,
    ax_callback: Callable[[plt.Axes], None] | None = None,
) -> NDArray[np.uint8]:
    """Render a state.

    Useful for viz and debugging.
    """
    if static_object_body_cache is None:
        static_object_body_cache = {}

    figsize = (
        world_max_x - world_min_x,
        world_max_y - world_min_y,
    )
    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=render_dpi)

    render_2dstate_on_ax(state, ax, static_object_body_cache)

    pad_x = (world_max_x - world_min_x) / 25
    pad_y = (world_max_y - world_min_y) / 25
    ax.set_xlim(world_min_x - pad_x, world_max_x + pad_x)
    ax.set_ylim(world_min_y - pad_y, world_max_y + pad_y)
    ax.axis("off")
    plt.tight_layout()
    if ax_callback is not None:
        ax_callback(ax)
    img = fig2data(fig)
    plt.close()
    return img[:, :, :3]


# ****** State to MultiBody2D conversion helpers ******


def kin_robot_to_multibody2d(obj: Object, state: ObjectCentricState) -> MultiBody2D:
    """Helper for object_to_multibody2d()."""
    assert obj.is_instance(KinRobotType)
    bodies: list[Body2D] = []

    # Base.
    base_x = state.get(obj, "x")
    base_y = state.get(obj, "y")
    base_radius = state.get(obj, "base_radius")
    circ = Circle(
        x=base_x,
        y=base_y,
        radius=base_radius,
    )
    z_order = ZOrder.SURFACE
    rendering_kwargs = {"facecolor": PURPLE, "edgecolor": BLACK}
    base = Body2D(circ, z_order, rendering_kwargs, name="base")
    bodies.append(base)

    # Gripper Base
    theta = state.get(obj, "theta")
    arm_joint = state.get(obj, "arm_joint")
    gripper_base_cx = base_x + np.cos(theta) * arm_joint
    gripper_base_cy = base_y + np.sin(theta) * arm_joint
    gripper_base_height = state.get(obj, "gripper_base_height")
    gripper_base_width = state.get(obj, "gripper_base_width")
    rect = Rectangle.from_center(
        center_x=gripper_base_cx,
        center_y=gripper_base_cy,
        height=gripper_base_height,
        width=gripper_base_width,
        rotation_about_center=theta,
    )
    z_order = ZOrder.SURFACE
    rendering_kwargs = {"facecolor": PURPLE, "edgecolor": BLACK}
    gripper_base = Body2D(rect, z_order, rendering_kwargs, name="gripper_base")
    gripper_base_pose = SE2Pose(
        x=gripper_base_cx,
        y=gripper_base_cy,
        theta=theta,
    )
    bodies.append(gripper_base)

    # Arm
    gripper_base_arm_rel_se2 = SE2Pose(
        x=(-state.get(obj, "arm_length") / 2 - gripper_base_width / 2),
        y=0.0,
        theta=0.0,
    )
    arm_se2 = gripper_base_pose * gripper_base_arm_rel_se2
    rect = Rectangle.from_center(
        center_x=arm_se2.x,
        center_y=arm_se2.y,
        height=gripper_base_width,
        width=state.get(obj, "arm_length"),
        rotation_about_center=theta,
    )
    z_order = ZOrder.SURFACE
    rendering_kwargs = {"facecolor": PURPLE, "edgecolor": BLACK}
    arm = Body2D(rect, z_order, rendering_kwargs, name="arm")
    bodies.append(arm)

    # Fingers
    relative_dx = state.get(obj, "finger_width") / 2
    relative_dy_r = -state.get(obj, "finger_gap") / 2
    relative_dy_l = state.get(obj, "finger_gap") / 2
    finger_r_pose = gripper_base_pose * SE2Pose(
        x=relative_dx,
        y=relative_dy_r,
        theta=0.0,
    )
    finger_l_pose = gripper_base_pose * SE2Pose(
        x=relative_dx,
        y=relative_dy_l,
        theta=0.0,
    )
    finger_r = Rectangle.from_center(
        center_x=finger_r_pose.x,
        center_y=finger_r_pose.y,
        height=state.get(obj, "finger_height"),
        width=state.get(obj, "finger_width"),
        rotation_about_center=finger_r_pose.theta,
    )
    finger_l = Rectangle.from_center(
        center_x=finger_l_pose.x,
        center_y=finger_l_pose.y,
        height=state.get(obj, "finger_height"),
        width=state.get(obj, "finger_width"),
        rotation_about_center=finger_l_pose.theta,
    )
    z_order = ZOrder.SURFACE
    rendering_kwargs = {"facecolor": PURPLE, "edgecolor": BLACK}
    finger_l_body = Body2D(finger_r, z_order, rendering_kwargs, name="arm")
    bodies.append(finger_l_body)
    finger_r_body = Body2D(finger_l, z_order, rendering_kwargs, name="arm")
    bodies.append(finger_r_body)

    multibody = MultiBody2D(obj.name, bodies)
    return multibody


def crv_robot_to_multibody2d(obj: Object, state: ObjectCentricState) -> MultiBody2D:
    """Helper for object_to_multibody2d()."""
    assert obj.is_instance(CRVRobotType)
    bodies: list[Body2D] = []

    # Base.
    base_x = state.get(obj, "x")
    base_y = state.get(obj, "y")
    base_radius = state.get(obj, "base_radius")
    circ = Circle(
        x=base_x,
        y=base_y,
        radius=base_radius,
    )
    z_order = ZOrder.ALL
    rendering_kwargs = {"facecolor": PURPLE, "edgecolor": BLACK}
    base = Body2D(circ, z_order, rendering_kwargs, name="base")
    bodies.append(base)

    # Gripper.
    theta = state.get(obj, "theta")
    arm_joint = state.get(obj, "arm_joint")
    gripper_cx = base_x + np.cos(theta) * arm_joint
    gripper_cy = base_y + np.sin(theta) * arm_joint
    gripper_height = state.get(obj, "gripper_height")
    gripper_width = state.get(obj, "gripper_width")
    rect = Rectangle.from_center(
        center_x=gripper_cx,
        center_y=gripper_cy,
        height=gripper_height,
        width=gripper_width,
        rotation_about_center=theta,
    )
    z_order = ZOrder.SURFACE
    rendering_kwargs = {"facecolor": PURPLE, "edgecolor": BLACK}
    gripper = Body2D(rect, z_order, rendering_kwargs, name="gripper")
    bodies.append(gripper)

    # Arm.
    rect = Rectangle.from_center(
        center_x=(base_x + gripper_cx) / 2,
        center_y=(base_y + gripper_cy) / 2,
        height=np.sqrt((base_x - gripper_cx) ** 2 + (base_y - gripper_cy) ** 2),
        width=(0.5 * gripper_width),
        rotation_about_center=(theta + np.pi / 2),
    )
    z_order = ZOrder.SURFACE
    silver = (128 / 255, 128 / 255, 128 / 255)
    rendering_kwargs = {"facecolor": silver, "edgecolor": BLACK}
    arm = Body2D(rect, z_order, rendering_kwargs, name="arm")
    bodies.append(arm)

    # If the vacuum is on, add a suction area.
    if state.get(obj, "vacuum") > 0.5:
        suction_height = gripper_height
        suction_width = gripper_width
        suction_cx = base_x + np.cos(theta) * (
            arm_joint + gripper_width + suction_width / 2
        )
        suction_cy = base_y + np.sin(theta) * (
            arm_joint + gripper_width + suction_width / 2
        )
        rect = Rectangle.from_center(
            center_x=suction_cx,
            center_y=suction_cy,
            height=suction_height,
            width=suction_width,
            rotation_about_center=theta,
        )
        z_order = ZOrder.NONE  # NOTE: suction collides with nothing
        rendering_kwargs = {"facecolor": PURPLE}
        suction = Body2D(rect, z_order, rendering_kwargs, name="suction")
        bodies.append(suction)

    return MultiBody2D(obj.name, bodies)


def kinematic2d_lobject_to_multibody2d(
    obj: Object, state: ObjectCentricState
) -> MultiBody2D:
    """Helper to create a MultiBody2D for an LObjectType object."""
    assert obj.is_instance(LObjectTypeGeom) or obj.is_instance(LObjectTypeDyn)
    # Get parameters
    x = state.get(obj, "x")
    y = state.get(obj, "y")
    theta = state.get(obj, "theta")
    width = state.get(obj, "width")
    length_side1 = state.get(obj, "length_side1")
    length_side2 = state.get(obj, "length_side2")
    color = (
        state.get(obj, "color_r"),
        state.get(obj, "color_g"),
        state.get(obj, "color_b"),
    )
    z_order = ZOrder(int(state.get(obj, "z_order")))

    geom = Lobject(x, y, width, (length_side1, length_side2), theta)

    rendering_kwargs = {
        "facecolor": color,
        "edgecolor": BLACK,
    }
    body = Body2D(geom, z_order, rendering_kwargs, name="hook")

    return MultiBody2D(obj.name, [body])


def kinematic2d_double_rectangle_to_multibody2d(
    obj: Object, state: ObjectCentricState
) -> MultiBody2D:
    """Helper to create a MultiBody2D for a DoubleRectType object."""
    assert obj.is_instance(DoubleRectType)
    # Note: We need to assume the two rectangles are aligned now.
    # This means theta is the same, relative dy == 0, 0 <= dx < width0 - width1.
    # Such that we can create two obstacles from the base rectangle.
    bodies: list[Body2D] = []

    # First rectangle.
    x0 = state.get(obj, "x")
    y0 = state.get(obj, "y")
    theta0 = state.get(obj, "theta")
    height0 = state.get(obj, "height")
    width0 = state.get(obj, "width")
    pose0 = SE2Pose(x0, y0, theta0)
    # Second rectangle.
    x1 = state.get(obj, "x1")
    y1 = state.get(obj, "y1")
    theta1 = state.get(obj, "theta1")
    width1 = state.get(obj, "width1")
    height1 = state.get(obj, "height1")
    pose1 = SE2Pose(x1, y1, theta1)
    assert theta0 == theta1, f"Expected theta0 == theta1, got {theta0} != {theta1}"
    relative_pose = pose0.inverse * pose1
    assert relative_pose.y == 0.0, f"Expected relative y == 0, got {relative_pose.y}"
    assert relative_pose.x >= 0.0, f"Expected relative x >= 0, got {relative_pose.x}"
    assert relative_pose.x + width1 < width0, "Expected relative x + width1 < width0"
    right_bookend_width = width0 - width1 - relative_pose.x

    # Left bookend.
    geom0 = Rectangle(x0, y0, relative_pose.x, height0, theta0)
    z_order0 = ZOrder(int(state.get(obj, "z_order")))
    rendering_kwargs0 = {
        "facecolor": (
            state.get(obj, "color_r"),
            state.get(obj, "color_g"),
            state.get(obj, "color_b"),
        ),
        "edgecolor": BLACK,
    }
    body0 = Body2D(geom0, z_order0, rendering_kwargs0, name=f"{obj.name}_base0")
    bodies.append(body0)
    # Right bookend.
    right_bookend_pose = pose0 * SE2Pose(relative_pose.x + width1, 0.0, theta0)
    geom0_ = Rectangle(
        right_bookend_pose.x, right_bookend_pose.y, right_bookend_width, height0, theta0
    )
    z_order0_ = ZOrder(int(state.get(obj, "z_order")))
    rendering_kwargs0_ = {
        "facecolor": (
            state.get(obj, "color_r"),
            state.get(obj, "color_g"),
            state.get(obj, "color_b"),
        ),
        "edgecolor": BLACK,
    }
    body0_ = Body2D(geom0_, z_order0_, rendering_kwargs0_, name=f"{obj.name}_base1")
    bodies.append(body0_)

    # Second rectangle.
    x1 = state.get(obj, "x1")
    y1 = state.get(obj, "y1")
    width1 = state.get(obj, "width1")
    height1 = state.get(obj, "height1")
    theta1 = state.get(obj, "theta1")
    geom1 = Rectangle(x1, y1, width1, height1, theta1)
    z_order1 = ZOrder(int(state.get(obj, "z_order1")))
    rendering_kwargs1 = {
        "facecolor": (
            state.get(obj, "color_r1"),
            state.get(obj, "color_g1"),
            state.get(obj, "color_b1"),
        ),
        "edgecolor": BLACK,
        "alpha": 0.5,
    }
    body1 = Body2D(geom1, z_order1, rendering_kwargs1, name=f"{obj.name}_part")
    bodies.append(body1)

    return MultiBody2D(obj.name, bodies)


def double_rectangle_object_to_part_geom(
    state: ObjectCentricState,
    double_rect_obj: Object,
    static_object_cache: dict[Object, MultiBody2D],
) -> Rectangle:
    """Helper to extract the second rectangle for a DoubleRectType object."""
    assert double_rect_obj.is_instance(DoubleRectType)
    multibody = object_to_multibody2d(double_rect_obj, state, static_object_cache)
    assert len(multibody.bodies) == 3
    # The second body is the "part" rectangle.
    assert "part" in multibody.bodies[2].name
    geom = multibody.bodies[2].geom
    assert isinstance(geom, Rectangle)
    return geom


def dot_robot_to_multibody2d(obj: Object, state: ObjectCentricState) -> MultiBody2D:
    """Helper for object_to_multibody2d() for DotRobotType."""
    assert obj.is_instance(DotRobotType)
    bodies: list[Body2D] = []

    # Simple circle robot
    x = state.get(obj, "x")
    y = state.get(obj, "y")
    radius = state.get(obj, "radius")
    circ = Circle(x=x, y=y, radius=radius)
    z_order = ZOrder.ALL
    rendering_kwargs = {
        "facecolor": (50 / 255, 50 / 255, 255 / 255),
        "edgecolor": BLACK,
    }
    base_body = Body2D(
        geom=circ,
        z_order=z_order,
        rendering_kwargs=rendering_kwargs,
        name="base",
    )
    bodies.append(base_body)

    return MultiBody2D(name=obj.name, bodies=bodies)


def tobject_to_multibody2d(obj: Object, state: ObjectCentricState) -> MultiBody2D:
    """Helper for object_to_multibody2d() for TObjectType."""
    assert obj.is_instance(TObjectType)

    # Get parameters
    x = state.get(obj, "x")
    y = state.get(obj, "y")
    theta = state.get(obj, "theta")
    width = state.get(obj, "width")
    length_horizontal = state.get(obj, "length_horizontal")
    length_vertical = state.get(obj, "length_vertical")
    color = (
        state.get(obj, "color_r"),
        state.get(obj, "color_g"),
        state.get(obj, "color_b"),
    )

    # Create Tobject geometry
    tobject_geom = Tobject(
        x=x,
        y=y,
        width=width,
        length_horizontal=length_horizontal,
        length_vertical=length_vertical,
        theta=theta,
    )

    z_order = ZOrder(int(state.get(obj, "z_order")))
    rendering_kwargs = {"facecolor": color, "edgecolor": BLACK}
    body = Body2D(
        geom=tobject_geom,
        z_order=z_order,
        rendering_kwargs=rendering_kwargs,
        name="root",
    )

    return MultiBody2D(name=obj.name, bodies=[body])


def object_to_multibody2d(
    obj: Object,
    state: ObjectCentricState,
    static_object_cache: dict[Object, MultiBody2D],
) -> MultiBody2D:
    """Create a Body2D instance for objects of standard geom types."""
    if obj.is_instance(CRVRobotType):
        return crv_robot_to_multibody2d(obj, state)
    if obj.is_instance(KinRobotType):
        return kin_robot_to_multibody2d(obj, state)
    if obj.is_instance(DotRobotType):
        return dot_robot_to_multibody2d(obj, state)
    if obj.is_instance(TObjectType):
        return tobject_to_multibody2d(obj, state)
    is_static = state.get(obj, "static") > 0.5
    if is_static and obj in static_object_cache:
        return static_object_cache[obj]
    geom: Geom2D  # rectangle or circle
    if obj.is_instance(RectangleType):
        x = state.get(obj, "x")
        y = state.get(obj, "y")
        width = state.get(obj, "width")
        height = state.get(obj, "height")
        theta = state.get(obj, "theta")
        geom = Rectangle(x, y, width, height, theta)
        z_order = ZOrder(int(state.get(obj, "z_order")))
        rendering_kwargs = {
            "facecolor": (
                state.get(obj, "color_r"),
                state.get(obj, "color_g"),
                state.get(obj, "color_b"),
            ),
            "edgecolor": BLACK,
        }
        body = Body2D(geom, z_order, rendering_kwargs)
        multibody = MultiBody2D(obj.name, [body])
    elif obj.is_instance(DynRectangleType) or obj.is_instance(KinRectangleType):
        x = state.get(obj, "x")
        y = state.get(obj, "y")
        width = state.get(obj, "width")
        height = state.get(obj, "height")
        theta = state.get(obj, "theta")
        # Different from RectangleType, use from_center.
        geom = Rectangle.from_center(x, y, width, height, theta)
        z_order = ZOrder(int(state.get(obj, "z_order")))
        rendering_kwargs = {
            "facecolor": (
                state.get(obj, "color_r"),
                state.get(obj, "color_g"),
                state.get(obj, "color_b"),
            ),
            "edgecolor": BLACK,
        }
        body = Body2D(geom, z_order, rendering_kwargs)
        multibody = MultiBody2D(obj.name, [body])
    elif obj.is_instance(CircleType):
        x = state.get(obj, "x")
        y = state.get(obj, "y")
        radius = state.get(obj, "radius")
        geom = Circle(x, y, radius)
        z_order = ZOrder(int(state.get(obj, "z_order")))
        rendering_kwargs = {
            "facecolor": (
                state.get(obj, "color_r"),
                state.get(obj, "color_g"),
                state.get(obj, "color_b"),
            ),
            "edgecolor": BLACK,
        }
        body = Body2D(geom, z_order, rendering_kwargs)
        multibody = MultiBody2D(obj.name, [body])
    elif obj.is_instance(LObjectTypeDyn) or obj.is_instance(LObjectTypeGeom):
        multibody = kinematic2d_lobject_to_multibody2d(obj, state)
    elif obj.is_instance(DoubleRectType):
        multibody = kinematic2d_double_rectangle_to_multibody2d(obj, state)
    elif obj.is_instance(SmallCircleType):
        # Small circle objects (for scoop-pour tasks)
        x = state.get(obj, "x")
        y = state.get(obj, "y")
        radius = state.get(obj, "radius")
        geom = Circle(x, y, radius)
        z_order = ZOrder(int(state.get(obj, "z_order")))
        rendering_kwargs = {
            "facecolor": (
                state.get(obj, "color_r"),
                state.get(obj, "color_g"),
                state.get(obj, "color_b"),
            ),
            "edgecolor": BLACK,
        }
        body = Body2D(geom, z_order, rendering_kwargs)
        multibody = MultiBody2D(obj.name, [body])
    elif obj.is_instance(SmallSquareType):
        # Small square objects (for scoop-pour tasks)
        x = state.get(obj, "x")
        y = state.get(obj, "y")
        size = state.get(obj, "size")
        theta = state.get(obj, "theta")
        # Use from_center for squares
        geom = Rectangle.from_center(x, y, size, size, theta)
        z_order = ZOrder(int(state.get(obj, "z_order")))
        rendering_kwargs = {
            "facecolor": (
                state.get(obj, "color_r"),
                state.get(obj, "color_g"),
                state.get(obj, "color_b"),
            ),
            "edgecolor": BLACK,
        }
        body = Body2D(geom, z_order, rendering_kwargs)
        multibody = MultiBody2D(obj.name, [body])
    else:
        raise NotImplementedError
    if is_static:
        static_object_cache[obj] = multibody
    return multibody


def rectangle_object_to_geom(
    state: ObjectCentricState,
    rect_obj: Object,
    static_object_cache: dict[Object, MultiBody2D],
) -> Rectangle:
    """Helper to extract a rectangle for an object."""
    assert (
        rect_obj.is_instance(RectangleType)
        or rect_obj.is_instance(DynRectangleType)
        or rect_obj.is_instance(KinRectangleType)
    )
    multibody = object_to_multibody2d(rect_obj, state, static_object_cache)
    assert len(multibody.bodies) == 1
    geom = multibody.bodies[0].geom
    assert isinstance(geom, Rectangle)
    return geom
