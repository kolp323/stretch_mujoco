"""State and component models for deterministic office employee agents."""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field
from typing import Any


def parse_clock(value: str) -> int:
    hour, minute = (int(part) for part in value.split(":"))
    if not 0 <= hour < 24 or not 0 <= minute < 60:
        raise ValueError(f"Invalid clock time '{value}'")
    return hour * 60 + minute


@dataclass(frozen=True)
class EmployeeProfile:
    role: str
    department: str
    personality: dict[str, float]
    preferences: dict[str, Any]


@dataclass
class EmployeeNeeds:
    hunger: float = 0.0
    thirst: float = 0.0
    fatigue: float = 0.0

    def advance(self, minutes: float, activity: str) -> None:
        fatigue_scale = 1.4 if activity in {"work", "use_computer"} else 1.0
        self.hunger = min(1.0, self.hunger + minutes * 0.0009)
        self.thirst = min(1.0, self.thirst + minutes * 0.0013)
        self.fatigue = min(1.0, self.fatigue + minutes * 0.0008 * fatigue_scale)


@dataclass(frozen=True)
class ScheduleItem:
    item_id: str
    start_minute: int
    end_minute: int
    activity: str
    location: str
    variation_minutes: int = 0

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ScheduleItem":
        return cls(
            item_id=payload["id"],
            start_minute=parse_clock(payload["start"]),
            end_minute=parse_clock(payload["end"]),
            activity=payload["activity"],
            location=payload["location"],
            variation_minutes=int(payload.get("variation_minutes", 0)),
        )

    def shifted_window(self, agent_id: str, day: int, seed: int) -> tuple[int, int]:
        if self.variation_minutes <= 0:
            return self.start_minute, self.end_minute
        digest = hashlib.sha256(f"{seed}:{agent_id}:{day}:{self.item_id}".encode("utf-8")).digest()
        rng = random.Random(int.from_bytes(digest[:8], "big"))
        shift = rng.randint(-self.variation_minutes, self.variation_minutes)
        return self.start_minute + shift, self.end_minute + shift


@dataclass
class EmployeeSchedule:
    items: tuple[ScheduleItem, ...]

    def active_item(
        self, agent_id: str, minute_of_day: float, day: int, seed: int
    ) -> ScheduleItem | None:
        for item in self.items:
            start, end = item.shifted_window(agent_id, day, seed)
            if start <= minute_of_day < end:
                return item
        return None


@dataclass(frozen=True)
class MemoryEntry:
    timestamp: float
    event: str
    details: dict[str, Any]


@dataclass
class AgentMemory:
    capacity: int = 100
    entries: list[MemoryEntry] = field(default_factory=list)

    def remember(self, entry: MemoryEntry) -> None:
        self.entries.append(entry)
        if len(self.entries) > self.capacity:
            del self.entries[: len(self.entries) - self.capacity]


@dataclass
class AgentPerception:
    visible_objects: set[str] = field(default_factory=set)
    last_update_time: float = 0.0

    def update(
        self,
        agent_id: str,
        semantic_snapshot: dict[str, Any] | None,
        visibility_range: float = 4.0,
    ) -> None:
        if not semantic_snapshot:
            return
        objects = semantic_snapshot.get("objects", {})
        agent_pose = objects.get(agent_id)
        if agent_pose is None:
            return
        origin = agent_pose["position"]
        self.visible_objects = {
            object_id
            for object_id, pose in objects.items()
            if sum((float(pose["position"][axis]) - float(origin[axis])) ** 2 for axis in range(3))
            <= visibility_range**2
        }
        self.last_update_time = float(semantic_snapshot.get("time", 0.0))


@dataclass
class EmployeeState:
    location: str
    current_action: str = "idle"
    held_object: str | None = None
    hunger: float = 0.0
    thirst: float = 0.0
    fatigue: float = 0.0
    current_goal: str = "follow_schedule"
    schedule_item: str = ""
    mood: float = 1.0
    availability: str = "available"
    attention_target: str | None = None
    blocked_reason: str | None = None
    last_failure: str | None = None
    animation_state: str = "idle"

    def sync_needs(self, needs: EmployeeNeeds) -> None:
        self.hunger = needs.hunger
        self.thirst = needs.thirst
        self.fatigue = needs.fatigue
        self.mood = max(-1.0, min(1.0, 1.0 - (sum((self.hunger, self.thirst, self.fatigue)) / 1.5)))
