"""Per-instance mocap root locomotion."""

from __future__ import annotations

import math

import mujoco
import numpy as np

from stretch_mujoco.humanoid.navigation import NavigationPathError, OfficeNavigationMesh

from .binding import NpcBinding


def yaw_quaternion(yaw: float) -> np.ndarray:
    return np.array([math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)])


def yaw_from_quaternion(quaternion: np.ndarray) -> float:
    w, x, y, z = quaternion
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def angle_delta(source: float, target: float) -> float:
    return (target - source + math.pi) % (2.0 * math.pi) - math.pi


class LocomotionController:
    """Move one NPC root; animation backends never write this pose."""

    def __init__(self, model: mujoco.MjModel, binding: NpcBinding) -> None:
        self.model = model
        self.binding = binding
        self.target_site: str | None = None
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

    def move_to(
        self,
        site: str,
        speed: float = 1.0,
        *,
        progress_timeout: float = 2.0,
        max_replans: int = 0,
    ) -> None:
        if mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, site) < 0:
            raise ValueError(f"Unknown NPC navigation target site: {site}")
        if speed <= 0:
            raise ValueError("NPC locomotion speed must be positive")
        if progress_timeout <= 0:
            raise ValueError("NPC locomotion progress_timeout must be positive")
        if max_replans < 0:
            raise ValueError("NPC locomotion max_replans cannot be negative")
        self.target_site = site
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

    def cancel(self) -> None:
        self.target_site = None
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
        position = data.mocap_pos[self.binding.mocap_id, :2].copy()
        if not self._route_waypoints and not self._plan_route(data, target):
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
                if not self._plan_route(data, target):
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
                self.route_tangent = (float(direction[0]), float(direction[1]))
                data.mocap_pos[self.binding.mocap_id, :2] += direction * step
                target_yaw = math.atan2(float(direction[0]), float(-direction[1]))
                self._turn_toward(data, target_yaw, dt)
                return False
            data.mocap_pos[self.binding.mocap_id, :2] = waypoint[:2]
            self._route_waypoint_index += 1
            if self._route_waypoint_index < len(self._route_waypoints):
                return False
        target[2] = data.mocap_pos[self.binding.mocap_id, 2]
        data.mocap_pos[self.binding.mocap_id] = target
        target_quaternion = np.empty(4)
        mujoco.mju_mat2Quat(target_quaternion, data.site_xmat[site_id])
        target_yaw = yaw_from_quaternion(target_quaternion)
        if not self._turn_toward(data, target_yaw, dt):
            return False
        data.mocap_quat[self.binding.mocap_id] = target_quaternion
        self.target_site = None
        self.route_tangent = None
        self._route_waypoints = ()
        self._route_waypoint_index = 0
        return True

    def _fail(self, reason: str) -> None:
        self.failure_reason = reason
        self.target_site = None
        self.route_tangent = None
        self._route_waypoints = ()
        self._route_waypoint_index = 0

    def _plan_route(self, data: mujoco.MjData, target: np.ndarray) -> bool:
        """Build a collision-aware route when the scene exposes an office floor.

        Small protocol fixtures intentionally omit the office navigation geometry.
        They retain the historical direct route, while a real office route is rebuilt
        from current collision geometry each time progress monitoring requests a replan.
        """
        start = data.mocap_pos[self.binding.mocap_id].copy()
        floor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "office_floor")
        if floor_id < 0:
            self._route_waypoints = (target.copy(),)
            self._route_waypoints[0][2] = start[2]
            self._route_waypoint_index = 0
            return True
        try:
            path = OfficeNavigationMesh.from_model(self.model, data).plan(start, target)
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
