"""Base classes for navigation planners.

Defines the abstract planner interface and the occupancy grid representation
shared by all navigation algorithms.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import math
from typing import Optional

import mujoco
import numpy as np
import cv2


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class NavigationPathError(RuntimeError):
    """Raised when a navigation target cannot be reached."""


# ---------------------------------------------------------------------------
# Occupancy grid
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ObstacleFootprint:
    """Axis-aligned bounding box of an inflated obstacle in world coordinates."""

    geom_name: str
    minimum: tuple[float, float]  # (x_min, y_min)
    maximum: tuple[float, float]  # (x_max, y_max)
    # Inflated XY polygon, when available.  Keeping the AABB fields preserves
    # compatibility with callers that use them for clearance heuristics.
    polygon: tuple[tuple[float, float], ...] | None = None


class OccupancyGrid:
    """2-D occupancy grid rasterised from MuJoCo collision geometry.

    Each collision geom is represented by its world-space XY footprint, not
    its aggregate AABB.  Meshes use the XY projection of MuJoCo's compiled
    convex collision hull; independent mesh geoms remain independent before
    their occupied cells are unioned.

    The grid is row-major: ``occupancy[row, col]`` where *row* indexes the
    y-axis and *col* indexes the x-axis.  ``True`` means occupied (blocked).
    """

    def __init__(
        self,
        bounds: tuple[float, float, float, float],
        occupancy: np.ndarray,
        *,
        resolution: float,
        agent_radius: float,
        obstacles: tuple[ObstacleFootprint, ...],
    ) -> None:
        self.x_min, self.x_max, self.y_min, self.y_max = bounds
        self.occupancy = occupancy  # bool, shape (rows, cols)
        self.resolution = resolution
        self.agent_radius = agent_radius
        self.obstacles = obstacles

    # -- properties ----------------------------------------------------------

    @property
    def width(self) -> int:
        """Number of columns (x-direction)."""
        return self.occupancy.shape[1]

    @property
    def height(self) -> int:
        """Number of rows (y-direction)."""
        return self.occupancy.shape[0]

    # -- factory -------------------------------------------------------------

    @classmethod
    def from_model(
        cls,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        *,
        resolution: float = 0.08,
        agent_radius: float = 0.25,
        bounds: tuple[float, float, float, float] | None = None,
        floor_geom_name: str = "office_floor",
        minimum_obstacle_height: float = 0.08,
        maximum_obstacle_height: float = 1.80,
        require_collision: bool = True,
        exclude_prefixes: tuple[str, ...] = (),
    ) -> "OccupancyGrid":
        """Build an occupancy grid from a MuJoCo model + data pair.

        Parameters
        ----------
        resolution:
            World metres per grid cell.
        agent_radius:
            Extra inflation added around every obstacle.
        bounds:
            Explicit ``(x_min, x_max, y_min, y_max)`` walkable region.
            When *None* the bounds are inferred from the named floor box
            geom.  Required for scenes whose floor is a plane.
        floor_geom_name:
            Name of the box geom that defines the walkable floor bounds
            (ignored when *bounds* is provided).
        minimum_obstacle_height:
            Geoms whose top is below this height (in world Z) are ignored.
        maximum_obstacle_height:
            Geoms whose bottom is above this height are ignored.
        require_collision:
            When True (default), only geoms with collision enabled are
            treated as obstacles.  Set to False for visual-only scenes
            (e.g. Habitat imports) where geoms have ``contype=0``.
        exclude_prefixes:
            Geom names starting with any of these prefixes are skipped.
            Useful for excluding stage/shell geometry from Habitat scenes
            (e.g. ``exclude_prefixes=(\"habitat_stage_\",)``).
        """
        if resolution <= 0 or agent_radius < 0:
            raise ValueError("resolution must be > 0 and agent_radius >= 0")

        if bounds is not None:
            x_min, x_max, y_min, y_max = bounds
        else:
            x_min, x_max, y_min, y_max = cls._walkable_bounds(
                model, data, agent_radius, floor_geom_name
            )

        width = int(math.floor((x_max - x_min) / resolution)) + 1
        height = int(math.floor((y_max - y_min) / resolution)) + 1
        occupancy = np.zeros((height, width), dtype=bool)
        obstacles: list[ObstacleFootprint] = []

        for geom_id in range(model.ngeom):
            geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
            if geom_name == floor_geom_name:
                continue
            if exclude_prefixes and geom_name.startswith(exclude_prefixes):
                continue

            body_id = int(model.geom_bodyid[geom_id])
            # Skip mocap bodies (animated humanoids, etc.)
            if body_id >= 0 and model.body_mocapid[body_id] >= 0:
                continue
            # Skip non-colliding geoms (unless we're told to include all visual
            # geoms, e.g. for Habitat scenes that lack collision attributes).
            if require_collision and (
                model.geom_contype[geom_id] == 0 and model.geom_conaffinity[geom_id] == 0
            ):
                continue

            center = data.geom_xpos[geom_id]
            half_extents = cls._world_half_extents(model, data, geom_id)
            z_min = float(center[2] - half_extents[2])
            z_max = float(center[2] + half_extents[2])

            if z_max <= minimum_obstacle_height or z_min >= maximum_obstacle_height:
                continue

            # Mesh geoms are projected from their compiled vertices instead
            # of using their world AABB.  MuJoCo compiles mesh collisions as
            # convex hulls, so this is also the shape used by physics. For
            # primitive geoms, use their exact transformed box footprint.
            footprint = cls._world_xy_footprint(model, data, geom_id)
            if footprint is None:
                footprint = cls._aabb_polygon(center[:2], half_extents[:2])
            poly = cls._inflate_polygon(footprint, agent_radius)
            minimum = poly.min(axis=0)
            maximum = poly.max(axis=0)
            obstacles.append(
                ObstacleFootprint(
                    geom_name,
                    (float(minimum[0]), float(minimum[1])),
                    (float(maximum[0]), float(maximum[1])),
                    tuple((float(x), float(y)) for x, y in poly),
                )
            )
            cls._rasterize_polygon(occupancy, poly, x_min, y_min, resolution)

        return cls(
            (x_min, x_max, y_min, y_max),
            occupancy,
            resolution=resolution,
            agent_radius=agent_radius,
            obstacles=tuple(obstacles),
        )

    @staticmethod
    def _aabb_polygon(center: np.ndarray, half: np.ndarray) -> np.ndarray:
        x, y = center[:2]
        hx, hy = half[:2]
        return np.asarray(((x-hx, y-hy), (x+hx, y-hy),
                           (x+hx, y+hy), (x-hx, y+hy)), dtype=float)

    @classmethod
    def _world_xy_footprint(cls, model: mujoco.MjModel, data: mujoco.MjData,
                            geom_id: int) -> np.ndarray | None:
        geom_type = int(model.geom_type[geom_id])
        center = data.geom_xpos[geom_id]
        xmat = data.geom_xmat[geom_id].reshape(3, 3)
        # mjGEOM_MESH == 5.  Use all compiled vertices; their convex hull is
        # the same conservative collision representation MuJoCo uses.
        if geom_type == int(mujoco.mjtGeom.mjGEOM_MESH):
            mesh_id = int(model.geom_dataid[geom_id])
            start = int(model.mesh_vertadr[mesh_id])
            count = int(model.mesh_vertnum[mesh_id])
            vertices = np.asarray(model.mesh_vert[start:start + count])
            if len(vertices):
                points = vertices @ xmat.T + center
                return cls._convex_hull(points[:, :2])
        # Exact transformed corners for boxes (including rotated furniture).
        if geom_type == int(mujoco.mjtGeom.mjGEOM_BOX):
            size = np.asarray(model.geom_size[geom_id])
            corners = np.asarray([(sx*size[0], sy*size[1], sz*size[2])
                                  for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
            return cls._convex_hull((corners @ xmat.T + center)[:, :2])
        # Circular/elliptical primitives are sampled in local XY and then
        # transformed, retaining arbitrary body orientation. A capsule's
        # longitudinal axis can be X/Y/Z, so sample both end discs.
        circle_types = {
            int(mujoco.mjtGeom.mjGEOM_SPHERE),
            int(mujoco.mjtGeom.mjGEOM_CYLINDER),
            int(mujoco.mjtGeom.mjGEOM_ELLIPSOID),
            int(mujoco.mjtGeom.mjGEOM_CAPSULE),
        }
        if geom_type in circle_types:
            size = np.asarray(model.geom_size[geom_id])
            angles = np.linspace(0.0, 2.0 * np.pi, 32, endpoint=False)
            # The local XY ellipse gives the exact footprint for spheres,
            # cylinders and ellipsoids in their own frame. For a capsule the
            # union of the two end discs is included conservatively.
            radii = np.asarray((size[0], size[0]))
            if geom_type == int(mujoco.mjtGeom.mjGEOM_ELLIPSOID):
                radii = size[:2]
            local = np.c_[radii[0] * np.cos(angles), radii[1] * np.sin(angles),
                          np.zeros(len(angles))]
            if geom_type == int(mujoco.mjtGeom.mjGEOM_CAPSULE):
                half_length = max(0.0, float(size[1] - size[0]))
                local = np.concatenate((local + (0, 0, half_length),
                                        local - (0, 0, half_length)))
            return cls._convex_hull((local @ xmat.T + center)[:, :2])
        return None

    @staticmethod
    def _convex_hull(points: np.ndarray) -> np.ndarray:
        if len(points) <= 2:
            return points
        hull = cv2.convexHull(np.asarray(points, dtype=np.float32)).reshape(-1, 2)
        return hull.astype(float)

    @staticmethod
    def _inflate_polygon(polygon: np.ndarray, radius: float) -> np.ndarray:
        if radius <= 0:
            return polygon
        # A polygonal Minkowski approximation. Raster dilation below provides
        # the final sub-cell conservative inflation as well.
        samples = []
        angles = np.linspace(0, 2*np.pi, 24, endpoint=False)
        offsets = np.c_[np.cos(angles), np.sin(angles)] * radius
        for point in polygon:
            samples.append(point + offsets)
        return OccupancyGrid._convex_hull(np.concatenate(samples, axis=0))

    @staticmethod
    def _rasterize_polygon(occupancy: np.ndarray, polygon: np.ndarray,
                           x_min: float, y_min: float, resolution: float) -> None:
        points = np.rint((polygon - np.asarray((x_min, y_min))) / resolution).astype(np.int32)
        points[:, 0] = np.clip(points[:, 0], 0, occupancy.shape[1] - 1)
        points[:, 1] = np.clip(points[:, 1], 0, occupancy.shape[0] - 1)
        # OpenCV does not accept bool images.  Draw into a byte mask and merge
        # it back, preserving the public boolean occupancy representation.
        mask = np.zeros(occupancy.shape, dtype=np.uint8)
        cv2.fillPoly(mask, [points[:, [0, 1]]], 1)
        occupancy |= mask.astype(bool)

    # -- coordinate conversion -----------------------------------------------

    def world_to_cell(self, point: np.ndarray) -> tuple[int, int]:
        """Convert a world (x, y) coordinate to a (row, col) cell index."""
        col = int(round((float(point[0]) - self.x_min) / self.resolution))
        row = int(round((float(point[1]) - self.y_min) / self.resolution))
        return row, col

    def cell_to_world(self, cell: tuple[int, int]) -> np.ndarray:
        """Convert a (row, col) cell index back to world (x, y)."""
        row, col = cell
        return np.array([self.x_min + col * self.resolution, self.y_min + row * self.resolution])

    # -- cell queries --------------------------------------------------------

    def is_cell_free(self, cell: tuple[int, int]) -> bool:
        """Return True if *cell* is inside bounds and unoccupied."""
        row, col = cell
        return (
            0 <= row < self.occupancy.shape[0]
            and 0 <= col < self.occupancy.shape[1]
            and not self.occupancy[row, col]
        )

    def is_world_free(self, point: np.ndarray) -> bool:
        """Return True if the world point's cell is free."""
        return self.is_cell_free(self.world_to_cell(np.asarray(point, dtype=float)))

    def is_world_inside_bounds(self, point: np.ndarray) -> bool:
        """Return True when a world point lies within the walkable bounds."""
        x, y = (float(v) for v in np.asarray(point)[:2])
        return self.x_min <= x <= self.x_max and self.y_min <= y <= self.y_max

    def point_inside_obstacle(self, point: np.ndarray) -> bool:
        """Check whether a world point falls inside any inflated obstacle AABB."""
        x, y = (float(v) for v in np.asarray(point)[:2])
        eps = 1e-9
        for obs in self.obstacles:
            if not (obs.minimum[0] + eps < x < obs.maximum[0] - eps
                    and obs.minimum[1] + eps < y < obs.maximum[1] - eps):
                continue
            if obs.polygon is None:
                return True
            contour = np.asarray(obs.polygon, dtype=np.float32)
            if cv2.pointPolygonTest(contour, (float(x), float(y)), False) >= 0:
                return True
        return False

    def nearest_free_cell(
        self,
        origin: tuple[int, int],
        toward: np.ndarray,
    ) -> tuple[int, int]:
        """If *origin* is occupied, return the nearest free cell (toward *toward*)."""
        if self.is_cell_free(origin):
            return origin
        max_radius = max(self.occupancy.shape)
        for radius in range(1, max_radius):
            candidates: list[tuple[int, int]] = []
            for row in range(origin[0] - radius, origin[0] + radius + 1):
                for col in range(origin[1] - radius, origin[1] + radius + 1):
                    if max(abs(row - origin[0]), abs(col - origin[1])) != radius:
                        continue
                    cell = (row, col)
                    if self.is_cell_free(cell):
                        candidates.append(cell)
            if candidates:
                return min(
                    candidates,
                    key=lambda c: float(np.linalg.norm(self.cell_to_world(c) - toward)),
                )
        raise NavigationPathError("No free navigation cell exists near the queried position")

    # -- line-of-sight -------------------------------------------------------

    def line_of_sight(
        self,
        start_cell: tuple[int, int],
        end_cell: tuple[int, int],
    ) -> bool:
        """Return True if the straight line between two cells is collision-free."""
        if any(
            not self.is_cell_free(cell) for cell in self._supercover_cells(start_cell, end_cell)
        ):
            return False
        start_world = self.cell_to_world(start_cell)
        end_world = self.cell_to_world(end_cell)
        return self.world_line_of_sight(start_world, end_world)

    def world_line_of_sight(self, start: np.ndarray, end: np.ndarray) -> bool:
        """Return True if a world-space segment stays in bounds and avoids obstacles."""
        start_world = np.asarray(start, dtype=float)[:2]
        end_world = np.asarray(end, dtype=float)[:2]
        if not self.is_world_inside_bounds(start_world) or not self.is_world_inside_bounds(
            end_world
        ):
            return False
        # Sample at half-cell spacing against the actual polygon. This avoids
        # the old AABB false positives while retaining a conservative check.
        distance = float(np.linalg.norm(end_world - start_world))
        steps = max(1, int(math.ceil(distance / (self.resolution * 0.5))))
        for alpha in np.linspace(0.0, 1.0, steps + 1):
            if self.point_inside_obstacle(start_world * (1-alpha) + end_world * alpha):
                return False
        return True

    @staticmethod
    def _supercover_cells(
        start: tuple[int, int], end: tuple[int, int]
    ) -> tuple[tuple[int, int], ...]:
        """Return every grid cell touched by a segment between two cell centres."""
        row, col = start
        end_row, end_col = end
        delta_row = end_row - row
        delta_col = end_col - col
        steps_row = abs(delta_row)
        steps_col = abs(delta_col)
        sign_row = 0 if delta_row == 0 else (1 if delta_row > 0 else -1)
        sign_col = 0 if delta_col == 0 else (1 if delta_col > 0 else -1)

        cells = [(row, col)]
        row_steps = 0
        col_steps = 0
        while row_steps < steps_row or col_steps < steps_col:
            row_progress = (1 + 2 * row_steps) * steps_col
            col_progress = (1 + 2 * col_steps) * steps_row
            if row_progress == col_progress:
                # At a grid corner, both side cells must be free as well as
                # the diagonal destination. This prevents corner cutting.
                if sign_row and sign_col:
                    cells.append((row + sign_row, col))
                    cells.append((row, col + sign_col))
                row += sign_row
                col += sign_col
                row_steps += 1
                col_steps += 1
            elif row_progress < col_progress:
                row += sign_row
                row_steps += 1
            else:
                col += sign_col
                col_steps += 1
            cells.append((row, col))
        return tuple(cells)

    # -- internal helpers ----------------------------------------------------

    @staticmethod
    def _walkable_bounds(
        model: mujoco.MjModel,
        data: mujoco.MjData,
        margin: float,
        floor_geom_name: str,
    ) -> tuple[float, float, float, float]:
        floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, floor_geom_name)
        if floor_id < 0:
            # Fallback: use the first box geom or a plane
            raise NavigationPathError(
                f"Navigation requires a box geom named '{floor_geom_name}' "
                f"to define walkable bounds."
            )
        if model.geom_type[floor_id] != mujoco.mjtGeom.mjGEOM_BOX:
            raise NavigationPathError(
                f"Floor geom '{floor_geom_name}' must be a box, "
                f"got type {model.geom_type[floor_id]}"
            )
        center = data.geom_xpos[floor_id]
        size = model.geom_size[floor_id]
        return (
            float(center[0] - size[0] + margin),
            float(center[0] + size[0] - margin),
            float(center[1] - size[1] + margin),
            float(center[1] + size[1] - margin),
        )

    @staticmethod
    def _world_half_extents(
        model: mujoco.MjModel,
        data: mujoco.MjData,
        geom_id: int,
    ) -> np.ndarray:
        """Conservative world-aligned half-extents of a geom."""
        geom_type = model.geom_type[geom_id]
        size = model.geom_size[geom_id]
        rotation = data.geom_xmat[geom_id].reshape(3, 3)

        if geom_type == mujoco.mjtGeom.mjGEOM_BOX:
            return np.abs(rotation) @ size
        if geom_type == mujoco.mjtGeom.mjGEOM_SPHERE:
            return np.full(3, size[0])
        if geom_type == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
            return np.sqrt((rotation * size[np.newaxis, :]) ** 2 @ np.ones(3))
        if geom_type in {mujoco.mjtGeom.mjGEOM_CYLINDER, mujoco.mjtGeom.mjGEOM_CAPSULE}:
            radius = float(size[0])
            half_length = float(size[1])
            axis = np.abs(rotation[:, 2])
            if geom_type == mujoco.mjtGeom.mjGEOM_CAPSULE:
                return np.full(3, radius) + half_length * axis
            radial = radius * np.sqrt(np.maximum(0.0, 1.0 - axis**2))
            return radial + half_length * axis
        if geom_type == mujoco.mjtGeom.mjGEOM_MESH:
            # For meshes, geom_size holds the AABB half-extents.
            # Some meshes embed sentinel values (0 or 1e10) — skip those
            # by returning a zero vector so they are filtered out.
            if np.any(size > 1e6) or np.any(size < 0):
                return np.zeros(3)
            clamped = np.clip(size, 0.0, 50.0)
            return np.abs(rotation) @ clamped
        # Fallback: use the bounding radius
        return np.full(3, float(model.geom_rbound[geom_id]))

    @staticmethod
    def _segment_intersects_box(
        start: np.ndarray,
        end: np.ndarray,
        bmin: np.ndarray,
        bmax: np.ndarray,
    ) -> bool:
        """Slab-based test for intersection with the strict AABB interior."""
        # Touching an already-inflated obstacle is valid: it represents
        # exactly the requested agent clearance rather than penetration.
        eps = 1e-9
        bmin = bmin + eps
        bmax = bmax - eps
        if np.any(bmin >= bmax):
            return False
        direction = end - start
        lower = 0.0
        upper = 1.0
        for axis in range(2):
            if abs(direction[axis]) < 1e-12:
                if start[axis] < bmin[axis] or start[axis] > bmax[axis]:
                    return False
                continue
            first = (bmin[axis] - start[axis]) / direction[axis]
            second = (bmax[axis] - start[axis]) / direction[axis]
            entry, exit_ = sorted((first, second))
            lower = max(lower, entry)
            upper = min(upper, exit_)
            if lower > upper:
                return False
        return upper >= 0.0 and lower <= 1.0


# ---------------------------------------------------------------------------
# Abstract planner
# ---------------------------------------------------------------------------


class BasePlanner(ABC):
    """Abstract interface for a grid-based path planner."""

    @staticmethod
    def _resolve_endpoint(
        grid: OccupancyGrid, point: np.ndarray, label: str
    ) -> tuple[np.ndarray, tuple[int, int]]:
        """Validate a requested endpoint and resolve it to a free grid cell."""
        values = np.asarray(point, dtype=float).reshape(-1)
        if values.size < 2 or not np.all(np.isfinite(values[:2])):
            raise ValueError(f"{label} must contain two finite coordinates")
        point_xy = values[:2].copy()
        if not grid.is_world_inside_bounds(point_xy):
            raise NavigationPathError(
                f"{label} {point_xy.tolist()} lies outside the navigation bounds."
            )
        if grid.point_inside_obstacle(point_xy):
            raise NavigationPathError(
                f"{label} {point_xy.tolist()} lies inside an inflated obstacle."
            )

        cell = grid.world_to_cell(point_xy)
        if not grid.is_cell_free(cell):
            cell = grid.nearest_free_cell(cell, point_xy)
            point_xy = grid.cell_to_world(cell)
        return point_xy, cell

    @staticmethod
    def _world_waypoints(
        grid: OccupancyGrid,
        cell_path: list[tuple[int, int]],
        start: np.ndarray,
        goal: np.ndarray,
    ) -> list[np.ndarray]:
        """Convert a cell path to validated, deduplicated world waypoints."""
        waypoints = [start]
        waypoints.extend(grid.cell_to_world(cell) for cell in cell_path[1:-1])
        waypoints.append(goal)

        deduped: list[np.ndarray] = []
        for waypoint in waypoints:
            if not deduped or np.linalg.norm(waypoint - deduped[-1]) > 1e-6:
                deduped.append(waypoint)

        for first, second in zip(deduped, deduped[1:]):
            if not grid.world_line_of_sight(first, second):
                raise NavigationPathError(
                    "Resolved path contains a segment that intersects an inflated obstacle."
                )
        return deduped

    @abstractmethod
    def plan(
        self,
        grid: OccupancyGrid,
        start: np.ndarray,
        goal: np.ndarray,
    ) -> list[np.ndarray]:
        """Compute a path from *start* to *goal*.

        Parameters
        ----------
        grid:
            The occupancy grid describing free / occupied space.
        start:
            World (x, y) start position.
        goal:
            World (x, y) goal position.

        Returns
        -------
        list[np.ndarray]
            Ordered list of 2-D waypoints from start to goal (inclusive).
        """
        ...
