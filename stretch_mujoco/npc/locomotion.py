"""Per-instance mocap root locomotion."""

from __future__ import annotations

import math

import mujoco
import numpy as np

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

    def move_to(self, site: str, speed: float = 1.0) -> None:
        if mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, site) < 0:
            raise ValueError(f"Unknown NPC navigation target site: {site}")
        if speed <= 0:
            raise ValueError("NPC locomotion speed must be positive")
        self.target_site = site
        self.speed = speed

    def cancel(self) -> None:
        self.target_site = None

    def step(self, data: mujoco.MjData, sim_time: float) -> bool:
        dt = 0.0 if self._last_time is None else min(max(sim_time - self._last_time, 0.0), 0.1)
        self._last_time = sim_time
        if self.target_site is None:
            return True
        site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, self.target_site)
        target = data.site_xpos[site_id].copy()
        target[2] = data.mocap_pos[self.binding.mocap_id, 2]
        delta = target[:2] - data.mocap_pos[self.binding.mocap_id, :2]
        distance = float(np.linalg.norm(delta))
        if distance > self.position_tolerance and dt > 0:
            step = min(self.speed * dt, distance)
            direction = delta / distance
            data.mocap_pos[self.binding.mocap_id, :2] += direction * step
            target_yaw = math.atan2(float(direction[0]), float(-direction[1]))
            self._turn_toward(data, target_yaw, dt)
            return False
        data.mocap_pos[self.binding.mocap_id] = target
        target_quaternion = np.empty(4)
        mujoco.mju_mat2Quat(target_quaternion, data.site_xmat[site_id])
        target_yaw = yaw_from_quaternion(target_quaternion)
        if not self._turn_toward(data, target_yaw, dt):
            return False
        data.mocap_quat[self.binding.mocap_id] = target_quaternion
        self.target_site = None
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
