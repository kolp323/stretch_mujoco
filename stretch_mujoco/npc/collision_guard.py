"""Continuous swept-envelope collision checks for NPC motion."""
from __future__ import annotations
from dataclasses import dataclass
import math
from typing import Iterable

@dataclass(frozen=True)
class CollisionResult:
    allowed: bool
    reason: str | None = None
    distance: float | None = None

class CollisionGuard:
    def __init__(self, *, radius: float = 0.28, clearance: float = 0.05) -> None:
        if radius <= 0 or clearance < 0: raise ValueError("collision envelope must be positive")
        self.radius, self.clearance = float(radius), float(clearance)

    def check_swept(self, old_pose: Iterable[float], new_pose: Iterable[float], obstacles: Iterable[Iterable[float]]) -> CollisionResult:
        old = tuple(float(x) for x in old_pose); new = tuple(float(x) for x in new_pose)
        threshold = self.radius + self.clearance
        for obstacle in obstacles:
            point = tuple(float(x) for x in obstacle)
            dx, dy = new[0] - old[0], new[1] - old[1]
            denom = dx * dx + dy * dy
            t = 0.0 if denom == 0 else max(0.0, min(1.0, ((point[0]-old[0])*dx + (point[1]-old[1])*dy) / denom))
            cx, cy = old[0] + t*dx, old[1] + t*dy
            distance = math.hypot(point[0]-cx, point[1]-cy)
            if distance < threshold:
                return CollisionResult(False, "collision_predicted", distance)
        return CollisionResult(True, distance=None)

    def check_mujoco(self, model, data, *, ignore_body_id: int | None = None) -> CollisionResult:
        for index in range(int(data.ncon)):
            contact = data.contact[index]
            if ignore_body_id is not None:
                geom_a = int(model.geom_bodyid[contact.geom1]); geom_b = int(model.geom_bodyid[contact.geom2])
                if geom_a == ignore_body_id or geom_b == ignore_body_id: continue
            return CollisionResult(False, "collision_observed", float(contact.dist))
        return CollisionResult(True)
