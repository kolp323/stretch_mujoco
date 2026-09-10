"""Runtime mesh animation player for pre-baked humanoid clips."""

from __future__ import annotations

import math
import re

import mujoco
import numpy as np

from .navigation import OfficeNavigationMesh


FRAME_GEOM_NAME_PATTERN = re.compile(
    r"^humanoid_preview_frame_(?P<clip>[a-z_]+)_(?P<frame>\d+)_(?P<material>[a-z]+)$"
)
SIT_ROOT_TO_SEAT_HEIGHT = 0.500
APPROACH_DISTANCE = 0.28
DESK_CLEARANCE_DISTANCE = 1.15
WALK_SPEED = 1.0
TURN_SPEED = 2.2
SIT_DOWN_DURATION = 1.4


def _yaw_from_quaternion(quaternion: np.ndarray) -> float:
    w, x, y, z = quaternion
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _yaw_quaternion(yaw: float) -> np.ndarray:
    return np.array([math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)])


def _angle_delta(source: float, target: float) -> float:
    return (target - source + math.pi) % (2.0 * math.pi) - math.pi


class HumanoidMeshAnimator:
    """Play mesh clips and perform the lightweight walk-to-chair sequence."""

    def __init__(self, model: mujoco.MjModel, fps: float = 8.0) -> None:
        self.model = model
        self.fps = fps
        self.clips: dict[str, dict[int, dict[str, int]]] = {}
        self.geom_ids: dict[str, int] = {}
        self._frame_geom_ids: set[int] = set()
        self._active_geom_ids: set[int] = set()
        self._last_key: tuple[str, int] | None = None
        self._mocap_id = -1
        self._standing_position: np.ndarray | None = None
        self._standing_orientation: np.ndarray | None = None
        self._last_update_time: float | None = None
        self._sit_stage: str | None = None
        self._waypoint_index = 0
        self._sit_down_start_time = 0.0
        self._is_seated = False
        self._sit_waypoints: tuple[np.ndarray, ...] = ()
        self._navigation_mesh: OfficeNavigationMesh | None = None
        self._navigation_route_key = ""
        self._navigation_waypoints: tuple[np.ndarray, ...] = ()
        self._navigation_waypoint_index = 0

        humanoid_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "humanoid_preview")
        if humanoid_body_id >= 0:
            self._mocap_id = int(model.body_mocapid[humanoid_body_id])

        for geom_id in range(model.ngeom):
            geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
            if not geom_name:
                continue
            match = FRAME_GEOM_NAME_PATTERN.match(geom_name)
            if match:
                clip = match.group("clip")
                frame = int(match.group("frame"))
                material = match.group("material")
                self.clips.setdefault(clip, {}).setdefault(frame, {})[material] = geom_id
                self._frame_geom_ids.add(geom_id)
                if model.geom_rgba[geom_id, 3] > 0.0:
                    self._active_geom_ids.add(geom_id)
                    self.geom_ids[material] = geom_id

    @property
    def available_clips(self) -> tuple[str, ...]:
        return tuple(sorted(self.clips))

    @property
    def is_available(self) -> bool:
        return bool(self.clips and self.geom_ids)

    @property
    def sit_stage(self) -> str | None:
        return self._sit_stage

    def update(
        self,
        sim_time: float,
        clip: str,
        data: mujoco.MjData | None = None,
        sit_target_site: str = "chair_right_sit",
        navigation_target_site: str = "",
        speed_scale: float = 1.0,
    ) -> None:
        if not self.is_available:
            return
        if clip not in self.clips:
            clip = "idle"
        speed_scale = max(float(speed_scale), 0.01)

        playback_clip = clip
        frame_phase = None
        if data is not None:
            playback_clip, frame_phase = self._update_root_pose(
                data,
                sim_time,
                clip,
                sit_target_site,
                navigation_target_site,
                speed_scale,
            )

        frame_numbers = sorted(self.clips[playback_clip])
        if frame_phase is None:
            animation_speed = speed_scale if playback_clip == "walk" else 1.0
            frame_index = int(sim_time * self.fps * animation_speed) % len(frame_numbers)
        else:
            frame_index = round(frame_phase * (len(frame_numbers) - 1))
        frame_number = frame_numbers[frame_index]
        key = (playback_clip, frame_number)
        if key == self._last_key:
            return

        for geom_id in self._active_geom_ids:
            self.model.geom_rgba[geom_id, 3] = 0.0
        frame_geoms = self.clips[playback_clip][frame_number]
        self._active_geom_ids = set(frame_geoms.values())
        for material, geom_id in frame_geoms.items():
            self.model.geom_rgba[geom_id, 3] = 1.0
            self.geom_ids[material] = geom_id
        self._last_key = key

    def _update_root_pose(
        self,
        data: mujoco.MjData,
        sim_time: float,
        requested_clip: str,
        sit_target_site: str,
        navigation_target_site: str,
        speed_scale: float,
    ) -> tuple[str, float | None]:
        if self._mocap_id < 0:
            return requested_clip, None
        if self._standing_position is None:
            self._standing_position = data.mocap_pos[self._mocap_id].copy()
            self._standing_orientation = data.mocap_quat[self._mocap_id].copy()

        dt = 0.0
        if self._last_update_time is not None:
            dt = min(max(sim_time - self._last_update_time, 0.0), 0.1) * speed_scale
        self._last_update_time = sim_time

        seated_clip = requested_clip in {"sit", "work"}
        if not seated_clip:
            if self._sit_stage is not None or self._is_seated:
                if requested_clip == "walk" and navigation_target_site:
                    data.mocap_pos[self._mocap_id, 2] = self._standing_position[2]
                else:
                    data.mocap_pos[self._mocap_id] = self._standing_position
                    data.mocap_quat[self._mocap_id] = self._standing_orientation
            self._sit_stage = None
            self._is_seated = False
            if requested_clip == "walk" and navigation_target_site:
                return self._walk_route(data, navigation_target_site, dt)
            return requested_clip, None

        site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, sit_target_site)
        if site_id < 0:
            raise ValueError(f"Unknown humanoid sit target site: {sit_target_site}")

        seat_position = data.site_xpos[site_id].copy()
        seat_position[2] -= SIT_ROOT_TO_SEAT_HEIGHT
        seat_orientation = np.empty(4)
        mujoco.mju_mat2Quat(seat_orientation, data.site_xmat[site_id])
        site_rotation = data.site_xmat[site_id].reshape(3, 3)
        approach_offset = site_rotation @ np.array([0.0, -APPROACH_DISTANCE, 0.0])
        approach_position = seat_position + approach_offset
        approach_position[2] = self._standing_position[2]
        clearance_position = approach_position.copy()
        desk_side = 1.0 if seat_position[0] >= 0.0 else -1.0
        clearance_position[0] = seat_position[0] - desk_side * DESK_CLEARANCE_DISTANCE
        clearance_position[2] = self._standing_position[2]

        if self._sit_stage is None:
            self._sit_stage = "walk"
            self._waypoint_index = 0
            planned_waypoints = self._plan_waypoints(data, clearance_position)
            self._sit_waypoints = (*planned_waypoints, approach_position)
            self._navigation_route_key = ""

        if self._sit_stage == "walk":
            target = self._sit_waypoints[self._waypoint_index]
            delta = target[:2] - data.mocap_pos[self._mocap_id, :2]
            distance = float(np.linalg.norm(delta))
            step = WALK_SPEED * dt
            if distance <= max(step, 0.015):
                data.mocap_pos[self._mocap_id] = target
                self._waypoint_index += 1
                if self._waypoint_index == len(self._sit_waypoints):
                    self._sit_stage = "turn"
            elif step > 0.0:
                direction = delta / distance
                data.mocap_pos[self._mocap_id, :2] += direction * step
                target_yaw = math.atan2(float(direction[0]), float(-direction[1]))
                self._turn_toward(data, target_yaw, dt)
            return "walk", None

        seat_yaw = _yaw_from_quaternion(seat_orientation)
        if self._sit_stage == "turn":
            if self._turn_toward(data, seat_yaw, dt):
                data.mocap_quat[self._mocap_id] = seat_orientation
                self._sit_stage = "sit_down"
                self._sit_down_start_time = sim_time
            return "idle", None

        if self._sit_stage == "sit_down":
            phase = min(
                max(
                    (sim_time - self._sit_down_start_time) * speed_scale / SIT_DOWN_DURATION,
                    0.0,
                ),
                1.0,
            )
            blend = phase * phase * (3.0 - 2.0 * phase)
            data.mocap_pos[self._mocap_id] = (
                approach_position * (1.0 - blend) + seat_position * blend
            )
            data.mocap_quat[self._mocap_id] = seat_orientation
            if phase >= 1.0:
                self._sit_stage = "seated"
                self._is_seated = True
            return "sit", phase

        data.mocap_pos[self._mocap_id] = seat_position
        data.mocap_quat[self._mocap_id] = seat_orientation
        if requested_clip == "work":
            return "work", None
        return "sit", 1.0

    def _walk_route(
        self,
        data: mujoco.MjData,
        route_spec: str,
        dt: float,
    ) -> tuple[str, float | None]:
        target_sites = tuple(site for site in route_spec.split("|") if site)
        if not target_sites:
            return "idle", None
        target_site = target_sites[-1]
        if route_spec != self._navigation_route_key:
            site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, target_site)
            if site_id < 0:
                raise ValueError(f"Unknown humanoid navigation target site: {target_site}")
            target = data.site_xpos[site_id].copy()
            target[2] = self._standing_position[2]
            self._navigation_waypoints = self._plan_waypoints(data, target)
            self._navigation_waypoint_index = 0
            self._navigation_route_key = route_spec

        if self._navigation_waypoint_index >= len(self._navigation_waypoints):
            return "idle", None
        target = self._navigation_waypoints[self._navigation_waypoint_index]
        if self._walk_to_position(data, target, dt):
            self._navigation_waypoint_index += 1
            if self._navigation_waypoint_index < len(self._navigation_waypoints):
                return "walk", None
            orientation = np.empty(4)
            site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, target_site)
            mujoco.mju_mat2Quat(orientation, data.site_xmat[site_id])
            data.mocap_quat[self._mocap_id] = orientation
            return "idle", None
        return "walk", None

    def _plan_waypoints(
        self,
        data: mujoco.MjData,
        target: np.ndarray,
    ) -> tuple[np.ndarray, ...]:
        # Rebuild only when a new route starts, so moved furniture and Stretch's
        # current collision pose affect the next plan without entering the frame loop.
        self._navigation_mesh = OfficeNavigationMesh.from_model(self.model, data)
        start = data.mocap_pos[self._mocap_id].copy()
        path = self._navigation_mesh.plan(start, target)
        return tuple(
            np.array([point[0], point[1], self._standing_position[2]]) for point in path[1:]
        )

    def _walk_to_position(
        self,
        data: mujoco.MjData,
        target: np.ndarray,
        dt: float,
    ) -> bool:
        delta = target[:2] - data.mocap_pos[self._mocap_id, :2]
        distance = float(np.linalg.norm(delta))
        step = WALK_SPEED * dt
        if distance <= max(step, 0.025):
            data.mocap_pos[self._mocap_id] = target
            return True
        if step > 0.0:
            direction = delta / distance
            data.mocap_pos[self._mocap_id, :2] += direction * step
            target_yaw = math.atan2(float(direction[0]), float(-direction[1]))
            self._turn_toward(data, target_yaw, dt)
        return False

    def _turn_toward(self, data: mujoco.MjData, target_yaw: float, dt: float) -> bool:
        current_yaw = _yaw_from_quaternion(data.mocap_quat[self._mocap_id])
        delta = _angle_delta(current_yaw, target_yaw)
        max_step = TURN_SPEED * dt
        if abs(delta) <= max(max_step, 0.01):
            new_yaw = target_yaw
            reached = True
        else:
            new_yaw = current_yaw + math.copysign(max_step, delta)
            reached = False
        data.mocap_quat[self._mocap_id] = _yaw_quaternion(new_yaw)
        return reached
