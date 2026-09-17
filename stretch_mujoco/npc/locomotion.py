"""Per-instance mocap root locomotion."""

from __future__ import annotations

from dataclasses import dataclass
import math

import mujoco
import numpy as np

from stretch_mujoco.humanoid.navigation import NavigationPathError, OfficeNavigationMesh

from .binding import NpcBinding


@dataclass(frozen=True)
class NavigationGeometryContract:
    """Collision geometry contract supplied by a trajectory profile."""

    surface: str
    agent_radius: float
    clearance: float
    resolution: float
    exclude_body_roots: tuple[str, ...] = ()


def yaw_quaternion(yaw: float) -> np.ndarray:
    return np.array([math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)])


def yaw_from_quaternion(quaternion: np.ndarray) -> float:
    w, x, y, z = quaternion
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def angle_delta(source: float, target: float) -> float:
    return (target - source + math.pi) % (2.0 * math.pi) - math.pi


class LocomotionController:
    """Move one NPC root; animation backends never write this pose."""

    def __init__(
        self, model: mujoco.MjModel, binding: NpcBinding, *, dynamic_obstacles: bool = True
    ) -> None:
        self.model = model
        self.binding = binding
        self.dynamic_obstacles = dynamic_obstacles
        self.navigation_geometry: NavigationGeometryContract | None = None
        self.target_site: str | None = None
        self.navigation_site: str | None = None
        # A seat centre is deliberately inside the chair's inflated navigation
        # footprint.  SIT may traverse the final, declared ingress segment at
        # root-motion speed after collision-safe routing reaches that ingress.
        # This is opt-in so ordinary MOVE_TO commands still reject obstacles.
        self.allow_final_ingress = False
        self._final_ingress = False
        self.speed = 1.0
        self.position_tolerance = 0.025
        self.yaw_tolerance = 0.03
        self._last_time: float | None = None
        self.route_revision = 0
        self.replan_attempt = 0
        self.max_replans = 0
        self.progress_timeout = 2.0
        self.last_progress_time: float | None = None
        self._last_progress_position: np.ndarray | None = None
        self.route_tangent: tuple[float, float] | None = None
        self._route_waypoints: tuple[np.ndarray, ...] = ()
        self._route_waypoint_index = 0
        self.failure_reason: str | None = None
        self._last_dynamic_check: float | None = None

    def configure_navigation(self, contract: NavigationGeometryContract | None) -> None:
        """Bind profile geometry; ``None`` retains only the legacy fixture fallback."""
        self.navigation_geometry = contract

    def _navigation_mesh(
        self, data: mujoco.MjData, *, include_mocap_obstacles: bool
    ) -> OfficeNavigationMesh:
        contract = self.navigation_geometry
        surface = "office_floor" if contract is None else contract.surface
        agent_radius = 0.25 if contract is None else contract.agent_radius + contract.clearance
        resolution = 0.08 if contract is None else contract.resolution
        root = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, self.binding.body_id)
        excluded_values = [] if contract is None else list(contract.exclude_body_roots)
        if root is not None:
            excluded_values.append(root)
        return OfficeNavigationMesh.from_model(
            self.model,
            data,
            resolution=resolution,
            agent_radius=agent_radius,
            floor_geom_name=surface,
            exclude_body_roots=tuple(dict.fromkeys(excluded_values)),
            include_mocap_obstacles=include_mocap_obstacles,
        )

    def move_to(
        self,
        site: str,
        speed: float = 1.0,
        *,
        progress_timeout: float = 2.0,
        max_replans: int = 0,
        navigation_site: str | None = None,
        allow_final_ingress: bool = False,
    ) -> None:
        if mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, site) < 0:
            raise ValueError(f"Unknown NPC navigation target site: {site}")
        if not math.isfinite(speed) or speed <= 0:
            raise ValueError("NPC locomotion speed must be positive")
        if not math.isfinite(progress_timeout) or progress_timeout <= 0:
            raise ValueError("NPC locomotion progress_timeout must be positive")
        if max_replans < 0:
            raise ValueError("NPC locomotion max_replans cannot be negative")
        if not isinstance(allow_final_ingress, bool):
            raise ValueError("NPC locomotion allow_final_ingress must be boolean")
        if (
            navigation_site is not None
            and mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, navigation_site) < 0
        ):
            raise ValueError(f"Unknown NPC navigation approach site: {navigation_site}")
        self.target_site = site
        self.navigation_site = navigation_site
        self.allow_final_ingress = allow_final_ingress
        self._final_ingress = False
        self.speed = speed
        self.progress_timeout = progress_timeout
        self.max_replans = max_replans
        self.route_revision = 0
        self.replan_attempt = 0
        self.last_progress_time = None
        self._last_progress_position = None
        self.route_tangent = None
        self._route_waypoints = ()
        self._route_waypoint_index = 0
        self.failure_reason = None
        self._last_dynamic_check = None

    def cancel(self) -> None:
        self.target_site = None
        self.navigation_site = None
        self.allow_final_ingress = False
        self._final_ingress = False
        self.route_tangent = None
        self._route_waypoints = ()
        self._route_waypoint_index = 0

    def step(self, data: mujoco.MjData, sim_time: float) -> bool:
        dt = 0.0 if self._last_time is None else min(max(sim_time - self._last_time, 0.0), 0.1)
        self._last_time = sim_time
        if self.target_site is None:
            return True
        site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, self.target_site)
        if site_id < 0:
            self._fail("route_invalid")
            return False
        target = data.site_xpos[site_id].copy()
        navigation_site_id = (
            site_id
            if self.navigation_site is None
            else mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, self.navigation_site)
        )
        if navigation_site_id < 0:
            self._fail("route_invalid")
            return False
        navigation_target = data.site_xpos[navigation_site_id].copy()
        position = data.mocap_pos[self.binding.mocap_id, :2].copy()
        if not self._route_waypoints and not self._plan_route(data, navigation_target):
            return False
        if self._last_progress_position is None:
            self._last_progress_position = position
            self.last_progress_time = sim_time
        elif (
            float(np.linalg.norm(position - self._last_progress_position))
            >= self.position_tolerance
        ):
            self._last_progress_position = position
            self.last_progress_time = sim_time
        elif (
            self.last_progress_time is not None
            and sim_time - self.last_progress_time > self.progress_timeout
        ):
            if self.replan_attempt < self.max_replans:
                self.replan_attempt += 1
                self.route_revision += 1
                self.last_progress_time = sim_time
                self._last_progress_position = position
                if not self._plan_route(data, navigation_target):
                    return False
            else:
                self._fail("route_blocked")
                return False
        if self._route_waypoint_index < len(self._route_waypoints):
            waypoint = self._route_waypoints[self._route_waypoint_index]
            delta = waypoint[:2] - data.mocap_pos[self.binding.mocap_id, :2]
            distance = float(np.linalg.norm(delta))
            if distance > self.position_tolerance:
                if dt <= 0:
                    return False
                step = min(self.speed * dt, distance)
                direction = delta / distance
                if not self._dynamic_segment_clear(data, sim_time, position, waypoint[:2]):
                    return False
                self.route_tangent = (float(direction[0]), float(direction[1]))
                data.mocap_pos[self.binding.mocap_id, :2] += direction * step
                target_yaw = math.atan2(float(direction[0]), float(-direction[1]))
                self._turn_toward(data, target_yaw, dt)
                return False
            # Only snap the tiny residual inside the declared position tolerance.
            data.mocap_pos[self.binding.mocap_id, :2] = waypoint[:2]
            self._route_waypoint_index += 1
            if self._route_waypoint_index < len(self._route_waypoints):
                return False
        # A chair's navigation ingress is deliberately distinct from its sit
        # site.  Continue the final ingress at normal root-motion speed instead
        # of assigning the mocap pose to the seat in one physics tick.
        if self.navigation_site is not None and self.navigation_site != self.target_site:
            # The final segment into a seat centre crosses the chair's
            # inflated navigation footprint.  It remains opt-in and is used
            # exclusively by the SIT workflow after the approach route has
            # completed at the declared ingress.
            self._final_ingress = self.allow_final_ingress
            self.navigation_site = None
            self._route_waypoints = (
                np.array((target[0], target[1], data.mocap_pos[self.binding.mocap_id, 2])),
            )
            self._route_waypoint_index = 0
            return False
        target_quaternion = np.empty(4)
        mujoco.mju_mat2Quat(target_quaternion, data.site_xmat[site_id])
        target_yaw = yaw_from_quaternion(target_quaternion)
        if not self._turn_toward(data, target_yaw, dt):
            return False
        data.mocap_quat[self.binding.mocap_id] = target_quaternion
        self.target_site = None
        self.navigation_site = None
        self.allow_final_ingress = False
        self._final_ingress = False
        self.route_tangent = None
        self._route_waypoints = ()
        self._route_waypoint_index = 0
        return True

    def _dynamic_segment_clear(
        self,
        data: mujoco.MjData,
        sim_time: float,
        start: np.ndarray,
        waypoint: np.ndarray,
    ) -> bool:
        """Periodically inspect moving collision proxies without rebuilding per tick."""
        if self._last_dynamic_check is not None and sim_time - self._last_dynamic_check < 0.25:
            return True
        self._last_dynamic_check = sim_time
        if self._final_ingress:
            return True
        contract = self.navigation_geometry
        surface = "office_floor" if contract is None else contract.surface
        if mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, surface) < 0:
            if contract is not None:
                self._fail(f"navigation_surface_missing:{surface}")
                return False
            return True
        if not self.dynamic_obstacles:
            static_mesh = self._navigation_mesh(data, include_mocap_obstacles=False)
            return self._segment_is_static_free(static_mesh, start, waypoint)
        try:
            static_mesh = self._navigation_mesh(data, include_mocap_obstacles=False)
            if not self._segment_is_static_free(static_mesh, start, waypoint):
                if self.replan_attempt < self.max_replans:
                    self.replan_attempt += 1
                    self.route_revision += 1
                    target_id = mujoco.mj_name2id(
                        self.model, mujoco.mjtObj.mjOBJ_SITE, self.navigation_site or self.target_site
                    )
                    return target_id >= 0 and self._plan_route(data, data.site_xpos[target_id])
                self._fail("route_static_collision")
                return False
            dynamic_mesh = self._navigation_mesh(data, include_mocap_obstacles=True)
            next_point = start + (waypoint - start) * min(
                1.0, self.speed * 0.25 / max(float(np.linalg.norm(waypoint - start)), 1e-9)
            )
            if not dynamic_mesh.point_inside_obstacle(next_point):
                return True
            # Static geometry has already been accounted for by the route
            # planner.  This periodic guard is specifically for newly moved
            # mocap actors; otherwise an inflated static boundary can be
            # mistaken for a dynamic blockage and repeatedly invalidate an
            # otherwise valid A* route.
            static_mesh = self._navigation_mesh(data, include_mocap_obstacles=False)
            if static_mesh.point_inside_obstacle(next_point):
                return True
        except NavigationPathError:
            pass
        if self.replan_attempt < self.max_replans:
            self.replan_attempt += 1
            self.route_revision += 1
            target_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_SITE, self.navigation_site or self.target_site
            )
            return target_id >= 0 and self._plan_route(data, data.site_xpos[target_id])
        self._fail("route_blocked_dynamic")
        return False

    def _segment_is_static_free(
        self, mesh: OfficeNavigationMesh, start: np.ndarray, waypoint: np.ndarray
    ) -> bool:
        """Check the complete root-motion segment against the raster contract.

        A* cells are conservative, but a smoothed waypoint can span multiple
        cells.  Checking only the next 0.25m allowed a long segment to cross
        an occupied cell before the next periodic dynamic check.
        """
        distance = float(np.linalg.norm(np.asarray(waypoint)[:2] - np.asarray(start)[:2]))
        spacing = max(mesh.resolution / 2.0, 1e-3)
        for ratio in np.linspace(0.0, 1.0, max(2, int(np.ceil(distance / spacing)) + 1)):
            point = np.asarray(start)[:2] + ratio * (np.asarray(waypoint)[:2] - np.asarray(start)[:2])
            if not mesh.is_world_free(point, component_id=mesh.primary_component_id):
                return False
        return True

    def _fail(self, reason: str) -> None:
        self.failure_reason = reason
        self.target_site = None
        self.navigation_site = None
        self.route_tangent = None
        self._route_waypoints = ()
        self._route_waypoint_index = 0

    def _plan_route(self, data: mujoco.MjData, target: np.ndarray) -> bool:
        """Build a collision-aware route from the configured profile geometry.

        Small protocol fixtures intentionally omit the office navigation geometry.
        They retain the historical direct route, while a real office route is rebuilt
        from current collision geometry each time progress monitoring requests a replan.
        """
        start = data.mocap_pos[self.binding.mocap_id].copy()
        contract = self.navigation_geometry
        surface = "office_floor" if contract is None else contract.surface
        floor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, surface)
        if floor_id < 0:
            if contract is not None:
                self._fail(f"navigation_surface_missing:{surface}")
                return False
            self._route_waypoints = (target.copy(),)
            self._route_waypoints[0][2] = start[2]
            self._route_waypoint_index = 0
            return True
        try:
            path = self._navigation_mesh(data, include_mocap_obstacles=self.dynamic_obstacles).plan(
                start, target
            )
        except NavigationPathError:
            self._fail("route_unavailable")
            return False
        waypoints = []
        for point in path[1:]:
            waypoint = np.array((point[0], point[1], start[2]), dtype=float)
            if not waypoints or float(np.linalg.norm(waypoint[:2] - waypoints[-1][:2])) > 1e-6:
                waypoints.append(waypoint)
        if not waypoints:
            waypoints.append(np.array((target[0], target[1], start[2]), dtype=float))
        self._route_waypoints = tuple(waypoints)
        self._route_waypoint_index = 0
        return True

    def align_to(self, data: mujoco.MjData, yaw: float, dt: float) -> bool:
        return self._turn_toward(data, yaw, dt)

    def _turn_toward(self, data: mujoco.MjData, yaw: float, dt: float) -> bool:
        current = yaw_from_quaternion(data.mocap_quat[self.binding.mocap_id])
        delta = angle_delta(current, yaw)
        max_step = 2.2 * dt
        if abs(delta) <= max(max_step, self.yaw_tolerance):
            data.mocap_quat[self.binding.mocap_id] = yaw_quaternion(yaw)
            return True
        data.mocap_quat[self.binding.mocap_id] = yaw_quaternion(
            current + math.copysign(max_step, delta)
        )
        return False
