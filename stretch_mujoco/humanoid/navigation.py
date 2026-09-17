"""Automatic 2D navigation grid generated from MuJoCo collision geometry."""

from __future__ import annotations

from dataclasses import dataclass
import heapq
import math

import mujoco
import numpy as np


class NavigationPathError(RuntimeError):
    """Raised when an office target cannot be reached on the navigation grid."""


@dataclass(frozen=True)
class ObstacleFootprint:
    geom_name: str
    minimum: tuple[float, float]
    maximum: tuple[float, float]


class OfficeNavigationMesh:
    """Rasterize MuJoCo geoms, inflate them, and plan smoothed A* paths."""

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
        self.occupancy = occupancy
        self.resolution = resolution
        self.agent_radius = agent_radius
        self.obstacles = obstacles
        self.component_labels, self.component_sizes = self._label_components(occupancy)
        self.primary_component_id = max(
            self.component_sizes,
            key=lambda component_id: (
                self.component_sizes[component_id],
                -component_id,
            ),
        )

    @classmethod
    def from_model(
        cls,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        *,
        resolution: float = 0.08,
        agent_radius: float = 0.25,
        minimum_obstacle_height: float = 0.08,
        maximum_obstacle_height: float = 1.80,
        exclude_body_roots: tuple[str, ...] = (),
        include_mocap_obstacles: bool = False,
        floor_geom_name: str = "office_floor",
    ) -> "OfficeNavigationMesh":
        if resolution <= 0 or agent_radius < 0:
            raise ValueError("Navigation resolution and agent radius must be valid")
        bounds = cls._walkable_bounds(model, data, agent_radius, floor_geom_name)
        x_min, x_max, y_min, y_max = bounds
        width = int(math.floor((x_max - x_min) / resolution)) + 1
        height = int(math.floor((y_max - y_min) / resolution)) + 1
        occupancy = np.zeros((height, width), dtype=bool)
        obstacles: list[ObstacleFootprint] = []

        for geom_id in range(model.ngeom):
            geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
            if geom_name == floor_geom_name:
                continue
            body_id = int(model.geom_bodyid[geom_id])
            # Other mocap actors are live collision proxies.  Exclude only the
            # caller's own root, otherwise multi-NPC plans silently cross them.
            if (
                body_id >= 0
                and model.body_mocapid[body_id] >= 0
                and (
                    not include_mocap_obstacles
                    or cls._body_root_name(model, body_id) in exclude_body_roots
                )
            ):
                continue
            if cls._body_root_name(model, body_id) in exclude_body_roots:
                continue
            if model.geom_contype[geom_id] == 0 and model.geom_conaffinity[geom_id] == 0:
                continue
            half_extents = cls._world_half_extents(model, data, geom_id)
            center = data.geom_xpos[geom_id]
            z_min = float(center[2] - half_extents[2])
            z_max = float(center[2] + half_extents[2])
            if z_max <= minimum_obstacle_height or z_min >= maximum_obstacle_height:
                continue
            minimum = center[:2] - half_extents[:2] - agent_radius
            maximum = center[:2] + half_extents[:2] + agent_radius
            obstacles.append(
                ObstacleFootprint(
                    geom_name,
                    (float(minimum[0]), float(minimum[1])),
                    (float(maximum[0]), float(maximum[1])),
                )
            )
            col_min = max(0, int(math.ceil((minimum[0] - x_min) / resolution)))
            col_max = min(
                width - 1,
                int(math.floor((maximum[0] - x_min) / resolution)),
            )
            row_min = max(0, int(math.ceil((minimum[1] - y_min) / resolution)))
            row_max = min(
                height - 1,
                int(math.floor((maximum[1] - y_min) / resolution)),
            )
            if col_min <= col_max and row_min <= row_max:
                occupancy[row_min : row_max + 1, col_min : col_max + 1] = True

        return cls(
            bounds,
            occupancy,
            resolution=resolution,
            agent_radius=agent_radius,
            obstacles=tuple(obstacles),
        )

    @staticmethod
    def _body_root_name(model: mujoco.MjModel, body_id: int) -> str:
        """Return the top-level body that owns one geom.

        The shared NPC planner normally treats Stretch as a moving obstacle.
        Stretch's own planner instead excludes the ``base_link`` tree so the
        base is not rasterised as an obstacle around its current pose.
        """
        current = body_id
        while current > 0 and int(model.body_parentid[current]) != 0:
            current = int(model.body_parentid[current])
        return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, current) or ""

    @staticmethod
    def _walkable_bounds(
        model: mujoco.MjModel,
        data: mujoco.MjData,
        margin: float,
        floor_geom_name: str = "office_floor",
    ) -> tuple[float, float, float, float]:
        floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, floor_geom_name)
        if floor_id < 0 or model.geom_type[floor_id] != mujoco.mjtGeom.mjGEOM_BOX:
            raise NavigationPathError(f"Navigation requires a box geom named '{floor_geom_name}'")
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
        return np.full(3, float(model.geom_rbound[geom_id]))

    def plan(self, start: np.ndarray, goal: np.ndarray) -> tuple[np.ndarray, ...]:
        start_xy = np.asarray(start, dtype=float)[:2]
        goal_xy = np.asarray(goal, dtype=float)[:2]
        goal_cell = self.world_to_cell(goal_xy)
        if not self.is_cell_free(goal_cell):
            if self.point_inside_obstacle(goal_xy):
                raise NavigationPathError(
                    f"Navigation goal {goal_xy.tolist()} lies inside an inflated obstacle"
                )
            goal_cell = self._nearest_free(goal_cell, start_xy)
        raw_start_cell = self.world_to_cell(start_xy)
        start_cell = self._nearest_free(raw_start_cell, goal_xy)
        cell_path = self._a_star(start_cell, goal_cell)
        smooth_cells = self._smooth(cell_path)
        points = [start_xy.copy()]
        points.extend(self.cell_to_world(cell) for cell in smooth_cells)
        points.append(goal_xy.copy())
        deduplicated = []
        for point in points:
            if not deduplicated or np.linalg.norm(point - deduplicated[-1]) > 1e-6:
                deduplicated.append(np.asarray(point, dtype=float))
        return tuple(deduplicated)

    def world_to_cell(self, point: np.ndarray) -> tuple[int, int]:
        col = int(round((float(point[0]) - self.x_min) / self.resolution))
        row = int(round((float(point[1]) - self.y_min) / self.resolution))
        return row, col

    def cell_to_world(self, cell: tuple[int, int]) -> np.ndarray:
        row, col = cell
        return np.array([self.x_min + col * self.resolution, self.y_min + row * self.resolution])

    def is_cell_free(self, cell: tuple[int, int]) -> bool:
        row, col = cell
        return (
            0 <= row < self.occupancy.shape[0]
            and 0 <= col < self.occupancy.shape[1]
            and not self.occupancy[row, col]
        )

    @staticmethod
    def _label_components(
        occupancy: np.ndarray,
    ) -> tuple[np.ndarray, dict[int, int]]:
        labels = np.full(occupancy.shape, -1, dtype=np.int32)
        sizes: dict[int, int] = {}
        next_label = 0
        for row in range(occupancy.shape[0]):
            for column in range(occupancy.shape[1]):
                if occupancy[row, column] or labels[row, column] >= 0:
                    continue
                frontier = [(row, column)]
                labels[row, column] = next_label
                size = 0
                for current_row, current_column in frontier:
                    size += 1
                    for neighbor_row, neighbor_column in (
                        (current_row - 1, current_column),
                        (current_row + 1, current_column),
                        (current_row, current_column - 1),
                        (current_row, current_column + 1),
                    ):
                        if (
                            0 <= neighbor_row < occupancy.shape[0]
                            and 0 <= neighbor_column < occupancy.shape[1]
                            and not occupancy[neighbor_row, neighbor_column]
                            and labels[neighbor_row, neighbor_column] < 0
                        ):
                            labels[neighbor_row, neighbor_column] = next_label
                            frontier.append((neighbor_row, neighbor_column))
                sizes[next_label] = size
                next_label += 1
        if not sizes:
            raise NavigationPathError("Navigation grid contains no free component")
        return labels, sizes

    def component_id(self, point: np.ndarray) -> int | None:
        cell = self.world_to_cell(np.asarray(point, dtype=float))
        if not self.is_cell_free(cell):
            return None
        return int(self.component_labels[cell])

    def is_world_free(self, point: np.ndarray, *, component_id: int | None = None) -> bool:
        cell = self.world_to_cell(np.asarray(point, dtype=float))
        return self.is_cell_free(cell) and (
            component_id is None or int(self.component_labels[cell]) == component_id
        )

    def nearest_free_world(
        self,
        point: np.ndarray,
        *,
        toward: np.ndarray,
        max_distance: float,
        component_id: int | None = None,
    ) -> np.ndarray:
        """Project an automatic candidate to a nearby free grid cell.

        Callers retain control over how far projection is allowed to move, so
        this cannot silently turn an object-local approach into an unrelated
        point elsewhere in the scene.
        """
        source = np.asarray(point, dtype=float)[:2]
        projected = self.cell_to_world(
            self._nearest_free(
                self.world_to_cell(source),
                np.asarray(toward, dtype=float)[:2],
                component_id=component_id,
            )
        )
        if float(np.linalg.norm(projected - source)) > max_distance:
            raise NavigationPathError(
                f"No free navigation cell within {max_distance:.3f} m of {source.tolist()}"
            )
        return projected

    def point_inside_obstacle(self, point: np.ndarray) -> bool:
        x, y = (float(value) for value in np.asarray(point)[:2])
        return any(
            obstacle.minimum[0] <= x <= obstacle.maximum[0]
            and obstacle.minimum[1] <= y <= obstacle.maximum[1]
            for obstacle in self.obstacles
        )

    def _nearest_free(
        self,
        origin: tuple[int, int],
        toward: np.ndarray,
        *,
        component_id: int | None = None,
    ) -> tuple[int, int]:
        if self.is_cell_free(origin) and (
            component_id is None or int(self.component_labels[origin]) == component_id
        ):
            return origin
        max_radius = max(self.occupancy.shape)
        for radius in range(1, max_radius):
            candidates = []
            for row in range(origin[0] - radius, origin[0] + radius + 1):
                for col in range(origin[1] - radius, origin[1] + radius + 1):
                    if max(abs(row - origin[0]), abs(col - origin[1])) != radius:
                        continue
                    cell = (row, col)
                    if self.is_cell_free(cell) and (
                        component_id is None or int(self.component_labels[cell]) == component_id
                    ):
                        candidates.append(cell)
            if candidates:
                return min(
                    candidates,
                    key=lambda cell: np.linalg.norm(self.cell_to_world(cell) - toward),
                )
        raise NavigationPathError("No free navigation cell exists near the NPC")

    def _a_star(
        self,
        start: tuple[int, int],
        goal: tuple[int, int],
    ) -> tuple[tuple[int, int], ...]:
        frontier = [(0.0, start)]
        came_from: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
        cost_so_far = {start: 0.0}
        moves = (
            (-1, 0, 1.0),
            (1, 0, 1.0),
            (0, -1, 1.0),
            (0, 1, 1.0),
            (-1, -1, math.sqrt(2.0)),
            (-1, 1, math.sqrt(2.0)),
            (1, -1, math.sqrt(2.0)),
            (1, 1, math.sqrt(2.0)),
        )
        while frontier:
            _, current = heapq.heappop(frontier)
            if current == goal:
                break
            for row_delta, col_delta, move_cost in moves:
                neighbor = (current[0] + row_delta, current[1] + col_delta)
                if not self.is_cell_free(neighbor):
                    continue
                if row_delta and col_delta:
                    if not self.is_cell_free((current[0] + row_delta, current[1])):
                        continue
                    if not self.is_cell_free((current[0], current[1] + col_delta)):
                        continue
                new_cost = cost_so_far[current] + move_cost
                if new_cost >= cost_so_far.get(neighbor, math.inf):
                    continue
                cost_so_far[neighbor] = new_cost
                heuristic = math.dist(neighbor, goal)
                heapq.heappush(frontier, (new_cost + heuristic, neighbor))
                came_from[neighbor] = current
        if goal not in came_from:
            raise NavigationPathError("No collision-free office path reaches the target")
        path = []
        current: tuple[int, int] | None = goal
        while current is not None:
            path.append(current)
            current = came_from[current]
        return tuple(reversed(path))

    def _smooth(
        self,
        path: tuple[tuple[int, int], ...],
    ) -> tuple[tuple[int, int], ...]:
        if len(path) <= 2:
            return path
        result = [path[0]]
        index = 0
        while index < len(path) - 1:
            candidate = len(path) - 1
            while candidate > index and not self._line_is_free(path[index], path[candidate]):
                candidate -= 1
            if candidate == index:
                raise NavigationPathError("Grid path grazes an inflated obstacle boundary")
            result.append(path[candidate])
            index = candidate
        return tuple(result)

    def _line_is_free(
        self,
        start: tuple[int, int],
        end: tuple[int, int],
    ) -> bool:
        # The obstacle footprints are continuous approximations, while the
        # raster occupancy is deliberately conservative.  A smoothed segment
        # must satisfy both representations; checking footprints alone can
        # skip an occupied cell at a raster boundary.
        start_world = self.cell_to_world(start)
        end_world = self.cell_to_world(end)
        distance = float(np.linalg.norm(end_world - start_world))
        samples = max(2, int(math.ceil(distance / (self.resolution / 2.0))) + 1)
        for ratio in np.linspace(0.0, 1.0, samples):
            point = start_world + ratio * (end_world - start_world)
            if not self.is_cell_free(self.world_to_cell(point)):
                return False
        for obstacle in self.obstacles:
            if self._segment_intersects_box(
                start_world,
                end_world,
                np.asarray(obstacle.minimum),
                np.asarray(obstacle.maximum),
            ):
                return False
        return True

    @staticmethod
    def _segment_intersects_box(
        start: np.ndarray,
        end: np.ndarray,
        minimum: np.ndarray,
        maximum: np.ndarray,
    ) -> bool:
        direction = end - start
        lower = 0.0
        upper = 1.0
        for axis in range(2):
            if abs(direction[axis]) < 1e-12:
                if start[axis] < minimum[axis] or start[axis] > maximum[axis]:
                    return False
                continue
            first = (minimum[axis] - start[axis]) / direction[axis]
            second = (maximum[axis] - start[axis]) / direction[axis]
            entry, exit_ = sorted((first, second))
            lower = max(lower, entry)
            upper = min(upper, exit_)
            if lower > upper:
                return False
        return upper >= 0.0 and lower <= 1.0
