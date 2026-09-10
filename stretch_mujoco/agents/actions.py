"""Closed action protocol and execution records for employee agents."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ActionType(str, Enum):
    IDLE = "idle"
    MOVE_TO = "move_to"
    SIT = "sit"
    STAND_UP = "stand_up"
    WORK = "work"
    REST = "rest"
    EAT = "eat"
    DRINK = "drink"
    PICK_UP = "pick_up"
    PUT_DOWN = "put_down"
    REQUEST_ROBOT = "request_robot"
    USE_COMPUTER = "use_computer"
    TALK = "talk"
    GESTURE_POINT = "gesture_point"
    GESTURE_WAVE = "gesture_wave"
    OPEN_CABINET = "open_cabinet"
    HANDOVER = "handover"
    ATTEND_MEETING = "attend_meeting"


class ExecutionStatus(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMED_OUT = "timed_out"


class RobotTaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


@dataclass(frozen=True)
class ActionCommand:
    agent_id: str
    action: ActionType
    target: str | None = None
    parameters: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ActionCommand":
        missing = {"agent_id", "action"} - payload.keys()
        if missing:
            raise ValueError(f"Action command missing fields: {', '.join(sorted(missing))}")
        try:
            action = ActionType(payload["action"])
        except ValueError as error:
            allowed = ", ".join(item.value for item in ActionType)
            raise ValueError(
                f"Action '{payload['action']}' is not allowed. Available: {allowed}"
            ) from error
        parameters = payload.get("parameters", {})
        if not isinstance(parameters, dict):
            raise ValueError("Action parameters must be an object")
        return cls(
            agent_id=str(payload["agent_id"]),
            action=action,
            target=payload.get("target"),
            parameters=dict(parameters),
        )


@dataclass(frozen=True)
class ValidationResult:
    valid: bool
    errors: tuple[str, ...] = ()


@dataclass
class ActionExecution:
    execution_id: str = field(default_factory=lambda: f"exec_{uuid.uuid4().hex}")
    command: ActionCommand | None = None
    status: ExecutionStatus = ExecutionStatus.IDLE
    phase: str = "idle"
    started_at: float = 0.0
    deadline: float | None = None
    driver_handle: str | None = None
    remaining_minutes: float = 0.0
    error: str | None = None

    @property
    def is_busy(self) -> bool:
        return self.status == ExecutionStatus.RUNNING


@dataclass
class RobotTask:
    requester: str
    task: str
    object_id: str
    destination: str
    robot_id: str = "stretch_3"
    task_id: str = field(default_factory=lambda: f"task_{uuid.uuid4().hex[:10]}")
    status: RobotTaskStatus = RobotTaskStatus.PENDING
    error: str | None = None
    conversation_id: str | None = None
    robot_release_confirmed: bool = False
    npc_attachment_confirmed: bool = False
    interaction_confirmed: bool = False
    receipt_ids: set[str] = field(default_factory=set)


@dataclass(frozen=True)
class RuntimeEvent:
    time: float
    event: str
    agent_id: str
    details: dict[str, Any] = field(default_factory=dict)
    # IDs are optional for schema-v1 readers, but all new runtime emissions
    # populate them so a physical receipt can be replayed without duplication.
    event_id: str = ""
    correlation_id: str | None = None
    causation_id: str | None = None


@dataclass(frozen=True)
class ConversationRequest:
    """Validated public input for a conversation transaction.

    ``semantic_snapshot`` is deliberately optional: embodied callers supply an
    interaction driver, while the compatibility path supplies an observation
    snapshot to retain the pre-existing logical conversation API.
    """

    session_id: str
    participants: tuple[str, ...]
    topic: str
    timeout: float = 45.0
    max_turns: int = 8
    interrupt_policy: str = "finish_turn"
    correlation_id: str | None = None
    semantic_snapshot: dict[str, Any] | None = None


@dataclass(frozen=True)
class ConversationReceipt:
    session_id: str
    accepted: bool
    status: str
    error: str | None = None
    correlation_id: str | None = None
