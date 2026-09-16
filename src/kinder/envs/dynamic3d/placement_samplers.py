"""Placement sampling utilities for dynamic3d environments."""

from typing import Any, Union

import numpy as np
from numpy.typing import NDArray
from shapely import Polygon, box, constrained_delaunay_triangles, union_all

from kinder.envs.dynamic3d import utils
from kinder.envs.dynamic3d.objects import (
    MujocoFixture,
    MujocoObject,
    get_fixture_class,
    get_object_class,
)

# Default yaw range in degrees (full rotation)
DEFAULT_YAW_RANGE = (0.0, 360.0)


def sample_feasible_ground_positions(
    configs: dict[str, dict[str, dict[str, Any]]],
    rng: np.random.Generator,
    region_names: dict[str, str],
    regions: dict[str, Any],
    room_planes: list[tuple[NDArray[np.float64], float]],
    obstacles: list[list[float]],
) -> dict[str, dict[str, dict[str, Any]]]:
    """Sample directly from footprint-safe ground polygons, without retry limits.

    Fixed-yaw footprints use their rotated AABB. Variable-yaw regions use the
    rotation envelope, so every sampled orientation fits. This is conservative,
    consistent with the existing AABB collision model. Objects rest on the ground.
    All poses are prepared before the caller changes the simulator.
    """
    occupied = list(obstacles)
    result: dict[str, dict[str, dict[str, Any]]] = {}
    clearance = 0.005
    for kind, objects in configs.items():
        result[kind] = {}
        for name, config in objects.items():
            bounds = get_object_class(kind).get_bounding_box_from_config(
                np.zeros(3, dtype=np.float32), config
            )
            region = regions[region_names[name]]
            candidates = []
            for index, extent in enumerate(region["ranges"]):
                if len(extent) != 4:
                    raise ValueError("Feasible ground sampling requires 2D regions")
                low, high = region.get(
                    "yaw_ranges", [DEFAULT_YAW_RANGE] * len(region["ranges"])
                )[index]
                yaw = np.deg2rad(rng.uniform(low, high))
                half = (np.array(bounds[3:5]) - bounds[:2]) / 2
                if low == high:
                    c, s = abs(np.cos(yaw)), abs(np.sin(yaw))
                    half = np.array(
                        [c * half[0] + s * half[1], s * half[0] + c * half[1]]
                    )
                else:
                    half = np.full(2, np.linalg.norm(half))
                x0, y0, x1, y1 = extent
                x0, y0 = np.array([x0, y0]) + half + clearance
                x1, y1 = np.array([x1, y1]) - half - clearance
                if x1 <= x0 or y1 <= y0:
                    continue
                vertices = [
                    np.array(p) for p in ((x0, y0), (x1, y0), (x1, y1), (x0, y1))
                ]
                for normal, offset in room_planes:
                    threshold = offset + np.abs(normal) @ half + clearance
                    clipped = []
                    for a, b in zip(vertices, vertices[1:] + vertices[:1]):
                        da, db = normal @ a - threshold, normal @ b - threshold
                        if da >= 0:
                            clipped.append(a)
                        if (da >= 0) != (db >= 0):
                            clipped.append(a + da / (da - db) * (b - a))
                    vertices = clipped
                    if len(vertices) < 3:
                        break
                if len(vertices) < 3:
                    continue
                height = bounds[5] - bounds[2]
                blocked = [
                    box(
                        o[0] - half[0] - clearance,
                        o[1] - half[1] - clearance,
                        o[3] + half[0] + clearance,
                        o[4] + half[1] + clearance,
                    )
                    for o in occupied
                    if o[5] > 0 and o[2] < height
                ]
                feasible = Polygon(vertices).difference(union_all(blocked))
                for triangle in constrained_delaunay_triangles(feasible).geoms:
                    if triangle.area > 0:
                        candidates.append((triangle, yaw, half))
            if not candidates:
                raise RuntimeError(f"No feasible ground placement region for {name!r}")
            areas = np.array([triangle.area for triangle, _, _ in candidates])
            triangle, yaw, half = candidates[
                int(rng.choice(len(candidates), p=areas / areas.sum()))
            ]
            a, b, c = np.array(triangle.exterior.coords)[:3]
            u, v = rng.random(2)
            root = np.sqrt(u)
            xy = (1 - root) * a + root * (1 - v) * b + root * v * c
            position = np.array([*xy, -bounds[2]], dtype=np.float64)
            occupied.append([*(xy - half), 0.0, *(xy + half), bounds[5] - bounds[2]])
            result[kind][name] = {"position": position, "yaw": float(yaw)}
    return result


def sample_collision_free_positions(
    configs: dict[str, dict[str, dict[str, Any]]],
    np_random: np.random.Generator,
    entity_region_names: dict[str, str] | None = None,
    entity_pos_yaw_samplers: dict[str, Any] | None = None,
    entity_check_in_region: dict[str, Any] | None = None,
    initial_placed_bboxes: list[list[float]] | None = None,
    fail_on_exhaustion: bool = False,
) -> dict[str, dict[str, dict[str, Any]]]:
    """Sample collision-free positions and yaws for multiple entities.

    Args:
        configs: Dictionary mapping entity types to entity configurations
                (entity_name -> entity_config). Can be fixture or object
                configurations based on what to place in the environment.
        np_random: Random number generator
        entity_region_names: Dictionary mapping entity names to region names
                           for sampling. If None, no entities will be sampled.
        entity_pos_yaw_samplers: Dictionary mapping entity names to functions
                               that sample positions and yaws within a region.
                               If None, no entities will be sampled.
        entity_check_in_region: Dictionary mapping entity names to functions that
            check whether a position lies within a region. When provided, all four
            bottom corners of the sampled axis-aligned bounding box must lie within
            the region. If None, no region checks are performed.
        initial_placed_bboxes: Bounding boxes that newly sampled entities must
                              avoid, without returning poses for those entities.
        fail_on_exhaustion: Raise instead of returning the legacy origin fallback
                            when no valid sample is found.

    Returns:
        Dictionary mapping entity types to dictionaries of entity poses
        (entity_name -> {"position": position, "yaw": yaw})
    """
    if entity_region_names is None:
        entity_region_names = {}
    if entity_pos_yaw_samplers is None:
        entity_pos_yaw_samplers = {}
    if entity_check_in_region is None:
        entity_check_in_region = {}

    entity_poses: dict[str, dict[str, dict[str, Any]]] = {}
    placed_bboxes = list(initial_placed_bboxes or [])

    for entity_type, entity_configs in configs.items():
        entity_poses[entity_type] = {}
        for entity_name, entity_config in entity_configs.items():

            if entity_name not in entity_pos_yaw_samplers:
                continue
            assert entity_name in entity_region_names, (
                f"Entity '{entity_name}' must have a region name specified in "
                f"entity_region_names if a pos_yaw_sampler is provided."
            )

            # Try to get the entity class (fixture or object)
            entity_class: Union[type[MujocoFixture], type[MujocoObject]]
            try:
                entity_class = get_fixture_class(entity_type)
            except ValueError:
                # If not a fixture, try as an object
                entity_class = get_object_class(entity_type)

            init_bbox = entity_class.get_bounding_box_from_config(
                np.array([0.0, 0.0, 0.0], dtype=np.float32), entity_config
            )
            # Sample a collision-free position and yaw for each entity
            position, yaw, bbox = sample_collision_free_position(
                list(init_bbox),
                placed_bboxes=placed_bboxes,
                np_random=np_random,
                region_name=entity_region_names[entity_name],
                pos_yaw_sampler=entity_pos_yaw_samplers[entity_name],
                check_in_region_func=entity_check_in_region.get(entity_name),
                fail_on_exhaustion=fail_on_exhaustion,
            )
            placed_bboxes.append(list(bbox))
            entity_poses[entity_type][entity_name] = {
                "position": position,
                "yaw": yaw,
            }
    return entity_poses


def sample_collision_free_position(
    bounding_box_at_origin: list[float],
    placed_bboxes: list[list[float]],
    np_random: np.random.Generator,
    region_name: str,
    pos_yaw_sampler: Any,
    check_in_region_func: Any = None,
    max_attempts: int = 100,
    fail_on_exhaustion: bool = False,
) -> tuple[NDArray[np.float32], float, list[float]]:
    """Sample a collision-free position and yaw for an entity.

    This function attempts to sample a position and yaw for an entity such that
    it does not collide with any already placed entities, and optionally the
    bottom face of the bounding box lies within the specified region. To generate
    candidate bounding boxes, the function translates the origin of the bounding
    box to the sampled position and rotates it according to the sampled yaw.

    If no collision-free position is found within the maximum number of attempts,
    either raise ``RuntimeError`` or preserve the legacy behavior of returning a
    fallback position with a warning, according to ``fail_on_exhaustion``.

    Args:
        bounding_box_at_origin: Initial bounding box as
                               [x_min, y_min, z_min, x_max, y_max, z_max]
        placed_bboxes: List of bounding boxes for already placed fixtures
        np_random: Random number generator
        region_name: Name of the region to sample from
        pos_yaw_sampler: Function that samples positions and yaws within a region
        check_in_region_func: Optional function used to require all four bottom
            corners of the sampled axis-aligned bounding box to lie in the region.
        max_attempts: Maximum number of sampling attempts
        fail_on_exhaustion: Raise if no valid placement is found instead of using
            the legacy fallback pose

    Returns:
        Tuple of (position, yaw, bbox) where position is [x, y, z] array,
        yaw is the rotation angle in radians, and bbox is the computed bounding box
        [x_min, y_min, z_min, x_max, y_max, z_max]

    Raises:
        RuntimeError: If no collision-free position is found and
            ``fail_on_exhaustion`` is true.
    """
    for _ in range(max_attempts):
        # Sample a candidate pose
        candidate_x, candidate_y, candidate_z, candidate_yaw = pos_yaw_sampler(
            region_name, np_random
        )

        candidate_pos = np.array(
            [candidate_x, candidate_y, candidate_z], dtype=np.float32
        )

        # Translate the bounding box to the candidate position
        translated_bbox = utils.translate_bounding_box(
            bounding_box_at_origin, candidate_pos
        )

        # Rotate the bounding box around its new center
        new_center_x = (
            translated_bbox[0] + (translated_bbox[3] - translated_bbox[0]) / 2
        )
        new_center_y = (
            translated_bbox[1] + (translated_bbox[4] - translated_bbox[1]) / 2
        )
        new_center = (new_center_x, new_center_y)
        candidate_bbox = utils.rotate_bounding_box_2d(
            translated_bbox, candidate_yaw, new_center
        )

        # Check if it collides with any existing fixture (using 3D overlap)
        collision = False
        for existing_bbox in placed_bboxes:
            if utils.bboxes_overlap(candidate_bbox, existing_bbox, margin=0.0):
                collision = True
                break

        # If no collision, check if all points of bbox are in the region
        if not collision and check_in_region_func is not None:
            # Sample bottom 4 corner points from the bbox to verify they're in the region
            x_min, y_min, z_min, x_max, y_max, _ = candidate_bbox
            test_points = [
                np.array([x_min, y_min, z_min], dtype=np.float32),
                np.array([x_min, y_max, z_min], dtype=np.float32),
                np.array([x_max, y_min, z_min], dtype=np.float32),
                np.array([x_max, y_max, z_min], dtype=np.float32),
            ]

            for point in test_points:
                if not check_in_region_func(point, region_name):
                    collision = True
                    break

        # If no collision, compute final bbox and return
        if not collision:
            return candidate_pos, candidate_yaw, candidate_bbox

    if fail_on_exhaustion:
        raise RuntimeError(
            f"Could not find collision-free position after {max_attempts} attempts"
        )

    # If we couldn't find a collision-free position after max_attempts,
    # return a fallback position (this shouldn't happen often with reasonable
    # fixture sizes)
    print(
        f"Warning: Could not find collision-free position after {max_attempts} "
        f"attempts"
    )
    # pylint: disable=fixme
    fallback_pos = np.array(
        [
            0.0,
            0.0,
            bounding_box_at_origin[2],
        ]
    )
    fallback_yaw_deg = np_random.uniform(DEFAULT_YAW_RANGE[0], DEFAULT_YAW_RANGE[1])
    fallback_yaw = np.radians(fallback_yaw_deg)

    # Translate and rotate the bounding box to get the fallback bbox
    translated_fallback_bbox = utils.translate_bounding_box(
        bounding_box_at_origin, fallback_pos
    )
    new_center_x = (
        translated_fallback_bbox[0]
        + (translated_fallback_bbox[3] - translated_fallback_bbox[0]) / 2
    )
    new_center_y = (
        translated_fallback_bbox[1]
        + (translated_fallback_bbox[4] - translated_fallback_bbox[1]) / 2
    )
    new_center = (new_center_x, new_center_y)
    fallback_bbox = utils.rotate_bounding_box_2d(
        translated_fallback_bbox, fallback_yaw, new_center
    )
    return fallback_pos, fallback_yaw, fallback_bbox
