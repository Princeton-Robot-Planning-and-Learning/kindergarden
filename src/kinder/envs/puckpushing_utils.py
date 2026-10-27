"""Shared pieces of the PuckPushing environments (2D and 3D).

A mobile robot pushes a puck across a floor maze to a goal. The floor is tiled with a
few surface types, each with its own sliding friction coefficient, which the robot does
not know. Everything else (puck mass and size, wall restitution, the map) is known.

This module holds what both envs and the planners share:

- the maps (walls, surface patches, holes, start and goal),
- the friction prior,
- an analytic model of a slide: piecewise Coulomb deceleration across patches, with
  frictionless wall bounces, and failure if the puck's center passes over a hole,
- the robot side of a push: where it stands, whether the strike is collision-free, and
  how far it drives to get there (shortest paths on a grid around the walls).

The envs simulate the same friction (a Coulomb force from the patch under the puck's
center), so the analytic model matches them up to contact effects and integration error.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import cached_property

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from kinder.envs.utils import RobotActionSpace

SURFACE_NAMES: tuple[str, ...] = ("laminate", "wood", "felt")
# Map characters for the surfaces, in SURFACE_NAMES order. '#' is a wall.
SURFACE_CHARS: tuple[str, ...] = ("l", "w", "f")
WALL_CHAR = "#"
# Uniform prior on each surface's sliding friction coefficient, in SURFACE_NAMES order.
DEFAULT_FRICTION_PRIOR: tuple[tuple[float, float], ...] = (
    (0.02, 0.05),
    (0.05, 0.10),
    (0.10, 0.20),
)
SURFACE_RGB: tuple[tuple[float, float, float], ...] = (
    (0.80, 0.86, 0.92),  # laminate: pale blue-gray
    (0.76, 0.60, 0.42),  # wood: tan
    (0.30, 0.50, 0.32),  # felt: green
)
WALL_RGB = (0.25, 0.25, 0.25)
HOLE_RGB = (0.9, 0.1, 0.1)
GOAL_RGB = (0.2, 0.8, 0.2)
PUCK_RGB = (0.95, 0.75, 0.1)

GRAVITY = 9.81


@dataclass(frozen=True)
class PuckMap:
    """A floor maze.

    Rows are listed top to bottom; positions are in meters.     Cell (col,
    row_from_bottom) spans [col * h, (col + 1) * h] x [row * h, (row + 1) *     h] for
    cell size h.
    """

    name: str
    rows: tuple[str, ...]
    cell_size: float
    puck_start: tuple[float, float]
    robot_start: tuple[float, float]
    goal: tuple[float, float]
    holes: tuple[tuple[float, float], ...] = ()

    @property
    def num_cols(self) -> int:
        """Number of cell columns."""
        return len(self.rows[0])

    @property
    def num_rows(self) -> int:
        """Number of cell rows."""
        return len(self.rows)

    @property
    def width(self) -> float:
        """Extent along x (m)."""
        return self.num_cols * self.cell_size

    @property
    def height(self) -> float:
        """Extent along y (m)."""
        return self.num_rows * self.cell_size

    def surface_grid(self) -> NDArray[np.int64]:
        """Surface index per cell, indexed [row_from_bottom, col]; -1 is a wall."""
        grid = np.full((self.num_rows, self.num_cols), -1, dtype=np.int64)
        for r, line in enumerate(self.rows):
            assert len(line) == self.num_cols, f"Ragged map row {r} in {self.name}"
            for c, char in enumerate(line):
                if char == WALL_CHAR:
                    continue
                grid[self.num_rows - 1 - r, c] = SURFACE_CHARS.index(char)
        return grid

    def wall_rects(self) -> list[tuple[float, float, float, float]]:
        """Walls as few axis-aligned rectangles (x0, y0, x1, y1)."""
        return self.cell_rects(-1)

    def surface_rects(self) -> list[tuple[int, tuple[float, float, float, float]]]:
        """Floor patches as (surface index, rectangle) pairs."""
        return [
            (surface, rect)
            for surface in range(len(SURFACE_CHARS))
            for rect in self.cell_rects(surface)
        ]

    def cell_rects(self, value: int) -> list[tuple[float, float, float, float]]:
        """The cells holding `value` as few rectangles (x0, y0, x1, y1).

        Runs of cells are merged along rows, then identical runs on consecutive rows are
        merged.
        """
        grid = self.surface_grid()
        h = self.cell_size
        runs: dict[tuple[int, int], list[int]] = {}
        for row in range(self.num_rows):
            col = 0
            while col < self.num_cols:
                if grid[row, col] != value:
                    col += 1
                    continue
                start = col
                while col < self.num_cols and grid[row, col] == value:
                    col += 1
                runs.setdefault((start, col), []).append(row)
        rects = []
        for (c0, c1), rows in sorted(runs.items()):
            r0 = prev = rows[0]
            for r in rows[1:]:
                if r == prev + 1:
                    prev = r
                    continue
                rects.append(_rect(c0, r0, c1, prev + 1, h))
                r0 = prev = r
            rects.append(_rect(c0, r0, c1, prev + 1, h))
        return rects


def _rect(
    c0: int, r0: int, c1: int, r1: int, h: float
) -> tuple[float, float, float, float]:
    """Cell-index bounds to meters, rounded so shared edges match exactly."""
    return (round(c0 * h, 9), round(r0 * h, 9), round(c1 * h, 9), round(r1 * h, 9))


# Cell size 0.4 m; corridors are three cells (1.2 m) wide, so the 0.55 x 0.51 m TidyBot
# base can stand beside or at an angle behind the puck. Three routes lead from the left
# side to the goal on the right: a slippery laminate corridor with a hole past its far
# end, a slow felt corridor with a hole in its middle, and a one-cell slot through the
# central block that the robot cannot enter. The slot shot reaches the goal only when
# the laminate and wood are slippery enough, and a puck that stops inside the slot can
# no longer be pushed toward the goal.
THREE_ROUTES = PuckMap(
    name="threeroutes",
    rows=(
        "##############",
        "#llllllllllll#",
        "#llllllllllll#",
        "#llllllllllll#",
        "#wwww###wwwww#",
        "#wwwwllllwwww#",
        "#wwww###wwwww#",
        "#ffffffffffff#",
        "#ffffffffffff#",
        "#ffffffffffff#",
        "##############",
    ),
    cell_size=0.4,
    puck_start=(1.2, 2.2),
    robot_start=(1.2, 1.0),
    goal=(3.9, 2.2),
    holes=((4.8, 3.6), (2.6, 1.0)),
)

MAPS: dict[str, PuckMap] = {m.name: m for m in (THREE_ROUTES,)}


@dataclass(frozen=True)
class PuckPhysics:
    """Known physical parameters of the puck, the robot and the strike."""

    puck_radius: float = 0.12
    puck_height: float = 0.12
    puck_mass: float = 1.0
    hole_radius: float = 0.2
    goal_radius: float = 0.3
    wall_restitution: float = 0.5
    # Robot footprint, as a circle for collision checks.
    robot_radius: float = 0.3
    # From the robot's center to the paddle face, along its heading.
    paddle_offset: float = 0.45
    # Base headings tried relative to the push direction, in order. The arm turns the
    # paddle so its face stays square to the push, and the holonomic base translates
    # along the push during the strike.
    base_offsets: tuple[float, ...] = (
        0.0,
        math.radians(30),
        -math.radians(30),
        math.radians(60),
        -math.radians(60),
    )
    paddle_width: float = 0.3
    # The paddle accelerates the puck over this distance, then brakes over the next.
    accel_distance: float = 0.3
    brake_distance: float = 0.1
    max_speed: float = 1.2
    min_speed: float = 0.05
    # Execution noise of a strike: relative speed error and heading error (rad).
    speed_noise: float = 0.05
    angle_noise: float = math.radians(2.0)
    # Measurement noise on the release speed (m/s) and the stop position (m).
    speed_obs_noise: float = 0.02
    position_obs_noise: float = 0.02
    # Grid resolution for the robot's shortest paths.
    nav_resolution: float = 0.1


@dataclass
class SlideResult:
    """Outcome of one slide."""

    stop: NDArray[np.float64]
    # Corner points of the path, from the release point to the stop.
    path: list[NDArray[np.float64]]
    # Index of the hole the puck's center passed over, if any. The slide ends there.
    hole: int | None = None
    distance: float = 0.0


@dataclass(frozen=True)
class Push:
    """A push: the puck's release heading (rad) and speed (m/s)."""

    angle: float
    speed: float

    @property
    def direction(self) -> NDArray[np.float64]:
        """Unit vector of the heading."""
        return np.array([math.cos(self.angle), math.sin(self.angle)])


class PuckWorld:
    """A map with the physics: slides, push feasibility and driving distances."""

    def __init__(self, puck_map: PuckMap, physics: PuckPhysics | None = None) -> None:
        self.puck_map = puck_map
        self.physics = physics if physics is not None else PuckPhysics()
        self.surfaces = self.puck_map.surface_grid()
        self.walls = np.asarray(self.puck_map.wall_rects(), dtype=np.float64)
        self.holes = np.asarray(self.puck_map.holes, dtype=np.float64).reshape(-1, 2)
        self.goal = np.asarray(self.puck_map.goal, dtype=np.float64)
        self._nav_cache: dict[int, NDArray[np.float64]] = {}

    # ------------------------------------------------------------------ surfaces

    def cell_of(self, point: NDArray[np.float64]) -> tuple[int, int]:
        """(row_from_bottom, col) of the cell holding `point`, clamped to the map."""
        h = self.puck_map.cell_size
        col = min(max(int(point[0] // h), 0), self.puck_map.num_cols - 1)
        row = min(max(int(point[1] // h), 0), self.puck_map.num_rows - 1)
        return row, col

    def surface_at(self, point: NDArray[np.float64]) -> int:
        """Surface index under `point`, or -1 inside a wall."""
        return int(self.surfaces[self.cell_of(point)])

    def friction_at(self, point: NDArray[np.float64], frictions) -> float:
        """Friction coefficient under `point`.

        Inside a wall, use the highest one.
        """
        surface = self.surface_at(point)
        if surface < 0:
            return float(max(frictions))
        return float(frictions[surface])

    # ------------------------------------------------------------------ geometry

    def _ray_hit(
        self,
        rects: NDArray[np.float64],
        origin: NDArray[np.float64],
        direction: NDArray[np.float64],
        inflate: float,
    ) -> tuple[float, NDArray[np.float64]]:
        """First entry along the ray into the rects inflated by `inflate`.

        Returns (distance, axes) where axes marks which axis the entry face is normal to
        (both at a corner). Faces the ray leaves through are ignored, so a point resting
        on a face and moving away does not hit it.
        """
        best, best_axes = math.inf, np.zeros(2)
        if len(rects) == 0:
            return best, best_axes
        lo = rects[:, :2] - inflate
        hi = rects[:, 2:] + inflate
        with np.errstate(divide="ignore", invalid="ignore"):
            inv = 1.0 / direction
            t1 = (lo - origin) * inv
            t2 = (hi - origin) * inv
        tmin = np.where(np.isnan(t1), -math.inf, np.minimum(t1, t2))
        tmax = np.where(np.isnan(t2), math.inf, np.maximum(t1, t2))
        # Axes with zero direction: inside the slab or never.
        for axis in range(2):
            if direction[axis] == 0.0:
                inside = (origin[axis] > lo[:, axis]) & (origin[axis] < hi[:, axis])
                tmin[:, axis] = np.where(inside, -math.inf, math.inf)
                tmax[:, axis] = np.where(inside, math.inf, -math.inf)
        enter = tmin.max(axis=1)
        leave = tmax.min(axis=1)
        eps = 1e-9
        candidates = np.nonzero((enter <= leave) & (leave > eps) & (enter > -1e-6))[0]
        for i in candidates:
            t = float(enter[i])
            if t >= best:
                continue
            axes = (np.abs(tmin[i] - t) < 1e-9).astype(float)
            # Moving into the entry face?
            if not np.any(axes * np.abs(direction) > 0):
                continue
            best, best_axes = max(t, 0.0), axes
        return best, best_axes

    def _segment_hits_rects(
        self,
        start: NDArray[np.float64],
        end: NDArray[np.float64],
        inflate: float,
    ) -> bool:
        """Whether a disc of radius `inflate` swept from start to end touches a wall.

        Rectangles are inflated as boxes, a slightly conservative check.
        """
        lo = self.walls[:, :2] - inflate
        hi = self.walls[:, 2:] + inflate
        delta = end - start
        t0 = np.zeros(len(self.walls))
        t1 = np.ones(len(self.walls))
        for axis in range(2):
            d = delta[axis]
            if abs(d) < 1e-12:
                outside = (start[axis] <= lo[:, axis]) | (start[axis] >= hi[:, axis])
                t0 = np.where(outside, 2.0, t0)
                continue
            a = (lo[:, axis] - start[axis]) / d
            b = (hi[:, axis] - start[axis]) / d
            t0 = np.maximum(t0, np.minimum(a, b))
            t1 = np.minimum(t1, np.maximum(a, b))
        return bool(np.any(t0 < t1))

    def _segment_hole(
        self,
        start: NDArray[np.float64],
        end: NDArray[np.float64],
        radius: float,
    ) -> tuple[int, float] | None:
        """First hole whose center lies within `radius` of the segment, with the
        fraction along the segment where the puck's center first enters it."""
        if len(self.holes) == 0:
            return None
        delta = end - start
        length2 = float(delta @ delta)
        best_index, best_t = -1, math.inf
        for i, center in enumerate(self.holes):
            rel = start - center
            if length2 < 1e-18:
                if float(rel @ rel) <= radius**2:
                    return i, 0.0
                continue
            # |start + t delta - center|^2 = radius^2
            b = float(rel @ delta)
            c = float(rel @ rel) - radius**2
            disc = b * b - length2 * c
            if disc < 0:
                continue
            t_in = (-b - math.sqrt(disc)) / length2
            t_out = (-b + math.sqrt(disc)) / length2
            if t_out < 0 or t_in > 1:
                continue
            t = max(t_in, 0.0)
            if t < best_t:
                best_index, best_t = i, t
        return None if best_index < 0 else (best_index, best_t)

    def puck_path_clear(
        self,
        start: NDArray[np.float64],
        end: NDArray[np.float64],
        hole_clearance: float | None = None,
    ) -> bool:
        """Whether the puck can travel straight from `start` to `end` without touching a
        wall or passing within `hole_clearance` (default: the hole radius) of a hole's
        center."""
        if self._segment_hits_rects(start, end, self.physics.puck_radius):
            return False
        clearance = (
            self.physics.hole_radius if hole_clearance is None else hole_clearance
        )
        return self._segment_hole(start, end, clearance) is None

    def nearest_free_puck_position(
        self, point: NDArray[np.float64], max_shift: float = 0.1, step: float = 0.005
    ) -> NDArray[np.float64]:
        """The closest position to `point` where the puck clears the walls, searched on
        rings up to `max_shift` away.

        Use it to clean up noisy measurements.
        """
        point = np.asarray(point, dtype=np.float64)
        if self.puck_position_free(point):
            return point
        for radius in np.arange(step, max_shift + 1e-9, step):
            count = max(8, int(2 * math.pi * radius / step))
            for theta in np.linspace(0.0, 2 * math.pi, count, endpoint=False):
                candidate = point + radius * np.array(
                    [math.cos(theta), math.sin(theta)]
                )
                if self.puck_position_free(candidate):
                    return candidate
        return point

    def puck_position_free(self, point: NDArray[np.float64]) -> bool:
        """Whether the puck can rest at `point` without overlapping a wall."""
        return not self._segment_hits_rects(point, point, self.physics.puck_radius)

    def over_hole(self, point: NDArray[np.float64]) -> bool:
        """Whether the puck's center at `point` is over a hole."""
        if len(self.holes) == 0:
            return False
        gaps = np.linalg.norm(self.holes - point, axis=1)
        return bool(np.any(gaps <= self.physics.hole_radius))

    def in_goal(self, point: NDArray[np.float64]) -> bool:
        """Whether a puck stopped at `point` is in the goal."""
        return float(np.linalg.norm(point - self.goal)) <= self.physics.goal_radius

    # ------------------------------------------------------------------ slides

    def slide(
        self,
        start: NDArray[np.float64],
        speed: float,
        angle: float,
        frictions,
        max_events: int = 1000,
    ) -> SlideResult:
        """Slide the puck from `start` at `speed` along `angle` until it stops.

        Deceleration is friction * g of the patch under the puck's center. Walls reflect
        the normal velocity scaled by the restitution, without friction.
        """
        phys = self.physics
        h = self.puck_map.cell_size
        point = np.array(start, dtype=np.float64)
        direction = np.array([math.cos(angle), math.sin(angle)])
        path = [point.copy()]
        total = 0.0
        v = float(speed)
        for _ in range(max_events):
            if v <= 1e-6:
                break
            probe = point + 1e-7 * direction
            decel = self.friction_at(probe, frictions) * GRAVITY
            s_stop = v * v / (2.0 * decel)
            # Distance to the next cell boundary, so the friction is constant in between.
            s_cell = math.inf
            for axis in range(2):
                d = direction[axis]
                if abs(d) < 1e-12:
                    continue
                idx = math.floor(probe[axis] / h)
                boundary = (idx + 1) * h if d > 0 else idx * h
                s_cell = min(s_cell, (boundary - point[axis]) / d)
            s_cell = max(s_cell, 0.0)
            s_wall, axes = self._ray_hit(self.walls, point, direction, phys.puck_radius)
            step = min(s_stop, s_cell, s_wall)
            end = point + step * direction
            hole = self._segment_hole(point, end, phys.hole_radius)
            if hole is not None:
                index, frac = hole
                point = point + frac * step * direction
                total += frac * step
                path.append(point.copy())
                return SlideResult(point, path, hole=index, distance=total)
            point = end
            total += step
            if step == s_stop:
                v = 0.0
                path.append(point.copy())
                break
            v = math.sqrt(max(v * v - 2.0 * decel * step, 0.0))
            if step == s_wall and step < s_cell + 1e-12:
                path.append(point.copy())
                velocity = v * direction
                velocity = np.where(
                    axes > 0, -phys.wall_restitution * velocity, velocity
                )
                v = float(np.linalg.norm(velocity))
                if v <= 1e-6:
                    break
                direction = velocity / v
        if not np.allclose(path[-1], point):
            path.append(point.copy())
        return SlideResult(point, path, hole=None, distance=total)

    # ------------------------------------------------------------------ pushes

    def push_pose(
        self, puck: NDArray[np.float64], angle: float
    ) -> tuple[NDArray[np.float64], float] | None:
        """Robot base position and heading for pushing the puck along `angle`.

        The paddle face touches the puck from behind. The first base offset (see
        `PuckPhysics.base_offsets`) where the robot fits and its strike sweep clears the
        walls is used. None if the push is impossible, including when the puck would hit
        a wall while still on the paddle.
        """
        phys = self.physics
        direction = np.array([math.cos(angle), math.sin(angle)])
        release = self.release_point(puck, angle)
        if self._segment_hits_rects(puck, release, phys.puck_radius):
            return None
        contact = puck - phys.puck_radius * direction
        sweep = (phys.accel_distance + phys.brake_distance) * direction
        for offset in phys.base_offsets:
            heading = math.atan2(math.sin(angle + offset), math.cos(angle + offset))
            base = contact - phys.paddle_offset * np.array(
                [math.cos(heading), math.sin(heading)]
            )
            if not self.robot_position_free(base):
                continue
            if self._segment_hits_rects(base, base + sweep, phys.robot_radius):
                continue
            return base, heading
        return None

    def push_station(
        self, puck: NDArray[np.float64], angle: float
    ) -> NDArray[np.float64]:
        """Robot base position for pushing along `angle` (straight behind the puck if
        the push is impossible)."""
        pose = self.push_pose(puck, angle)
        if pose is not None:
            return pose[0]
        direction = np.array([math.cos(angle), math.sin(angle)])
        reach = self.physics.puck_radius + self.physics.paddle_offset
        return puck - reach * direction

    def release_point(
        self, puck: NDArray[np.float64], angle: float
    ) -> NDArray[np.float64]:
        """Where the puck leaves the paddle after the acceleration phase."""
        direction = np.array([math.cos(angle), math.sin(angle)])
        return puck + self.physics.accel_distance * direction

    def robot_position_free(self, point: NDArray[np.float64]) -> bool:
        """Whether the robot fits at `point` without touching a wall.

        Holes only swallow the puck; the robot base bridges them.
        """
        return not self._segment_hits_rects(point, point, self.physics.robot_radius)

    def push_feasible(self, puck: NDArray[np.float64], angle: float) -> bool:
        """Whether the robot can push the puck along `angle` (see `push_pose`).

        Whether the robot can drive to the push station is a separate question
        (`drive_distance`).
        """
        return self.push_pose(puck, angle) is not None

    # ------------------------------------------------------------------ driving

    @cached_property
    def _nav_grid(self) -> tuple[NDArray[np.bool_], NDArray[np.int64], csr_matrix]:
        """Free robot positions on a grid, their node ids, and the 8-connected graph."""
        res = self.physics.nav_resolution
        nx = int(round(self.puck_map.width / res))
        ny = int(round(self.puck_map.height / res))
        free = np.zeros((ny, nx), dtype=bool)
        for j in range(ny):
            for i in range(nx):
                free[j, i] = self.robot_position_free(
                    np.array([(i + 0.5) * res, (j + 0.5) * res])
                )
        ids = -np.ones((ny, nx), dtype=np.int64)
        ids[free] = np.arange(int(free.sum()))
        rows, cols, weights = [], [], []
        for j, i in zip(*np.nonzero(free)):
            for dj, di in ((0, 1), (1, 0), (1, 1), (1, -1)):
                jj, ii = j + dj, i + di
                if 0 <= jj < ny and 0 <= ii < nx and free[jj, ii]:
                    # No corner cutting on diagonals.
                    if dj and di and not (free[j, ii] and free[jj, i]):
                        continue
                    w = res * math.hypot(dj, di)
                    rows += [ids[j, i], ids[jj, ii]]
                    cols += [ids[jj, ii], ids[j, i]]
                    weights += [w, w]
        n = int(free.sum())
        graph = csr_matrix((weights, (rows, cols)), shape=(n, n))
        return free, ids, graph

    def nav_node_of(self, point: NDArray[np.float64]) -> int:
        """Node of the nearest free grid point to `point`, or -1 if none is near."""
        free, ids, _ = self._nav_grid
        res = self.physics.nav_resolution
        ny, nx = free.shape
        i0 = int(point[0] // res)
        j0 = int(point[1] // res)
        best, best_d = -1, math.inf
        for j in range(max(j0 - 2, 0), min(j0 + 3, ny)):
            for i in range(max(i0 - 2, 0), min(i0 + 3, nx)):
                if not free[j, i]:
                    continue
                d = math.hypot((i + 0.5) * res - point[0], (j + 0.5) * res - point[1])
                if d < best_d:
                    best, best_d = int(ids[j, i]), d
        return best

    def _distances_from(self, node: int) -> NDArray[np.float64]:
        if node not in self._nav_cache:
            _, _, graph = self._nav_grid
            self._nav_cache[node] = dijkstra(  # type: ignore[call-overload]
                graph, directed=False, indices=node
            )
        return self._nav_cache[node]

    def drive_distance(self, from_node: int, to: NDArray[np.float64]) -> float:
        """Shortest driving distance from a grid node to point `to` (inf if
        unreachable).

        The puck is not treated as an obstacle.
        """
        to_node = self.nav_node_of(to)
        if from_node < 0 or to_node < 0:
            return math.inf
        return float(self._distances_from(from_node)[to_node])

    def drive_path(
        self, start: NDArray[np.float64], goal: NDArray[np.float64]
    ) -> list[NDArray[np.float64]]:
        """Waypoints of a shortest drive from `start` to `goal` (empty if
        unreachable)."""
        free, _, graph = self._nav_grid
        res = self.physics.nav_resolution
        a, b = self.nav_node_of(start), self.nav_node_of(goal)
        if a < 0 or b < 0:
            return []
        _, pred = dijkstra(  # type: ignore[call-overload]
            graph, directed=False, indices=a, return_predecessors=True
        )
        if b != a and pred[b] < 0:
            return []
        centers = np.argwhere(free)  # (j, i) per node, in node order
        nodes = [b]
        while nodes[-1] != a:
            nodes.append(int(pred[nodes[-1]]))
        points = [
            np.array([(centers[n][1] + 0.5) * res, (centers[n][0] + 0.5) * res])
            for n in reversed(nodes)
        ]
        return [np.asarray(start, dtype=np.float64)] + points + [np.asarray(goal)]


def sample_frictions(
    rng: np.random.Generator,
    prior: tuple[tuple[float, float], ...] = DEFAULT_FRICTION_PRIOR,
) -> tuple[float, ...]:
    """Draw one friction coefficient per surface from the uniform prior."""
    return tuple(float(rng.uniform(lo, hi)) for lo, hi in prior)


class PushActionSpace(RobotActionSpace):
    """One push: the puck's release heading and speed."""

    def __init__(self, max_speed: float) -> None:
        super().__init__(
            np.array([-np.pi, 0.0]), np.array([np.pi, max_speed]), dtype=np.float64
        )

    def create_markdown_description(self) -> str:
        """Create a human-readable markdown description of this space."""
        features = [
            ("angle", "Release heading of the puck (rad, 0 is +x)"),
            ("speed", "Release speed of the puck (m/s)"),
        ]
        md = "| **Index** | **Feature** | **Description** | **Min** | **Max** |"
        md += "\n| --- | --- | --- | --- | --- |"
        for idx, (feature, description) in enumerate(features):
            md += (
                f"\n| {idx} | {feature} | {description} | "
                f"{self.low[idx]:.3f} | {self.high[idx]:.3f} |"
            )
        return md + "\n"
