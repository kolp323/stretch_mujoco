"""State and component models for deterministic office employee agents."""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field
from enum import Enum
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


class AgentAvailability(str, Enum):
    """The bounded public availability state used by planning and conversations."""

    AVAILABLE = "available"
    EXECUTING = "executing"
    IN_CONVERSATION = "in_conversation"
    BLOCKED = "blocked"


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
    availability: AgentAvailability | str = AgentAvailability.AVAILABLE
    attention_target: str | None = None
    blocked_reason: str | None = None
    last_failure: str | None = None
    animation_state: str = "idle"
    conversation_id: str | None = None
    social_energy: float = 1.0
    stress: float = 0.0
    animation_clip: str = "idle"
    animation_lifecycle: str = "completed"

    def __post_init__(self) -> None:
        self.set_availability(self.availability)
        self.social_energy = self._bounded_social_value("social_energy", self.social_energy)
        self.stress = self._bounded_social_value("stress", self.stress)

    @staticmethod
    def _bounded_social_value(name: str, value: float) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{name} must be a finite value between 0 and 1")
        if not 0.0 <= float(value) <= 1.0:
            raise ValueError(f"{name} must be between 0 and 1")
        return float(value)

    def set_availability(self, availability: AgentAvailability | str) -> None:
        """Accept schema-v1 ``busy`` while retaining a bounded runtime state."""
        if availability == "busy":
            availability = AgentAvailability.EXECUTING
        try:
            self.availability = AgentAvailability(availability)
        except ValueError as error:
            allowed = ", ".join(item.value for item in AgentAvailability)
            raise ValueError(
                f"Unknown agent availability '{availability}'; expected one of {allowed}"
            ) from error

    def sync_needs(self, needs: EmployeeNeeds) -> None:
        self.hunger = needs.hunger
        self.thirst = needs.thirst
        self.fatigue = needs.fatigue
        self.mood = max(-1.0, min(1.0, 1.0 - (sum((self.hunger, self.thirst, self.fatigue)) / 1.5)))

    def begin_conversation(self, session_id: str, partner: str) -> None:
        if not session_id or not partner:
            raise ValueError("Conversation session and partner are required")
        if self.conversation_id not in {None, session_id}:
            raise ValueError("Employee is already in another conversation")
        self.conversation_id = session_id
        self.attention_target = partner
        self.set_availability(AgentAvailability.IN_CONVERSATION)
        self.animation_state = "idle"
        self.animation_clip = "idle"
        self.animation_lifecycle = "running"

    def finish_conversation(self) -> None:
        self.conversation_id = None
        self.attention_target = None
        if self.availability == AgentAvailability.IN_CONVERSATION:
            self.set_availability(AgentAvailability.AVAILABLE)
        self.animation_lifecycle = "completed"

    def record_success(self, stress_delta: float = -0.02) -> None:
        self.stress = min(1.0, max(0.0, self.stress + stress_delta))
        self.last_failure = None
        self.blocked_reason = None

    def record_failure(self, reason: str, stress_delta: float = 0.05) -> None:
        self.stress = min(1.0, max(0.0, self.stress + stress_delta))
        self.last_failure = reason
        self.blocked_reason = reason

    def advance_social(self, minutes: float, conversing: bool) -> None:
        if minutes < 0:
            raise ValueError("Social time cannot move backwards")
        # Conversation is mildly restorative; solitary work slowly consumes
        # social capacity. Both projections are bounded by construction.
        delta = minutes * (0.003 if conversing else -0.0005)
        self.social_energy = min(1.0, max(0.0, self.social_energy + delta))
