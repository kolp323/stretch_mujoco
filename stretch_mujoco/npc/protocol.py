"""Serializable command, receipt, and observation contracts for NPC control."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import math
from typing import Any, Mapping


class NpcCommandKind(str, Enum):
    MOVE_TO = "move_to"
    ALIGN_TO = "align_to"
    PLAY_ANIMATION = "play_animation"
    ATTACH_OBJECT = "attach_object"
    DETACH_OBJECT = "detach_object"
    INTERACTION_CUE = "interaction_cue"
    CANCEL = "cancel"


class CommandStatus(str, Enum):
    ACCEPTED = "accepted"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMED_OUT = "timed_out"

    @property
    def terminal(self) -> bool:
        return self in {
            CommandStatus.SUCCEEDED,
            CommandStatus.FAILED,
            CommandStatus.CANCELLED,
            CommandStatus.TIMED_OUT,
        }


@dataclass(frozen=True)
class NpcCommand:
    command_id: str
    sequence: int
    npc_id: str
    kind: NpcCommandKind
    payload: dict[str, object]
    issued_at: float
    deadline: float | None = None

    def __post_init__(self) -> None:
        if not self.command_id:
            raise ValueError("NPC command_id cannot be empty")
        if not self.npc_id:
            raise ValueError("NPC npc_id cannot be empty")
        if (
            isinstance(self.sequence, bool)
            or not isinstance(self.sequence, int)
            or self.sequence < 0
        ):
            raise ValueError("NPC command sequence cannot be negative")
        if (
            isinstance(self.issued_at, bool)
            or not isinstance(self.issued_at, (int, float))
            or not math.isfinite(self.issued_at)
        ):
            raise ValueError("NPC command issued_at must be finite")
        if self.deadline is not None and (
            isinstance(self.deadline, bool)
            or not isinstance(self.deadline, (int, float))
            or not math.isfinite(self.deadline)
        ):
            raise ValueError("NPC command deadline must be finite")
        if self.deadline is not None and self.deadline < self.issued_at:
            raise ValueError("NPC command deadline cannot precede issued_at")

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["kind"] = self.kind.value
        return result

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "NpcCommand":
        missing = {
            "command_id",
            "sequence",
            "npc_id",
            "kind",
            "payload",
            "issued_at",
        } - payload.keys()
        if missing:
            raise ValueError(f"NPC command missing fields: {', '.join(sorted(missing))}")
        command_payload = payload["payload"]
        if not isinstance(command_payload, Mapping):
            raise ValueError("NPC command payload must be an object")
        try:
            kind = NpcCommandKind(str(payload["kind"]))
        except ValueError as error:
            raise ValueError(f"Unknown NPC command kind '{payload['kind']}'") from error
        return cls(
            command_id=str(payload["command_id"]),
            sequence=int(payload["sequence"]),
            npc_id=str(payload["npc_id"]),
            kind=kind,
            payload=dict(command_payload),
            issued_at=float(payload["issued_at"]),
            deadline=None if payload.get("deadline") is None else float(payload["deadline"]),
        )


@dataclass(frozen=True)
class NpcCommandReceipt:
    command_id: str
    npc_id: str
    status: CommandStatus
    reason: str | None = None
    started_at: float | None = None
    finished_at: float | None = None

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["status"] = self.status.value
        return result

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "NpcCommandReceipt":
        return cls(
            command_id=str(payload["command_id"]),
            npc_id=str(payload["npc_id"]),
            status=CommandStatus(str(payload["status"])),
            reason=None if payload.get("reason") is None else str(payload["reason"]),
            started_at=(
                None if payload.get("started_at") is None else float(payload["started_at"])
            ),
            finished_at=(
                None if payload.get("finished_at") is None else float(payload["finished_at"])
            ),
        )


@dataclass(frozen=True)
class NpcRuntimeState:
    npc_id: str
    revision: int
    sim_time: float
    position: tuple[float, float, float]
    quaternion: tuple[float, float, float, float]
    locomotion: str
    requested_animation: str
    resolved_clip: str
    clip_phase: float
    phase_seed: int = 0
    phase_offset: float = 0.0
    last_marker: str | None = None
    pending_clip: str | None = None
    deferred_interrupt: bool = False
    route_revision: int = 0
    replan_attempt: int = 0
    stop_marker: str | None = None
    locomotion_failure: str | None = None
    animation_lifecycle: str = "requested"
    transition: str | None = None
    animation_events: tuple[str, ...] = ()
    held_objects: tuple[str, ...] = ()
    interaction_id: str | None = None
    active_command_id: str | None = None
    last_receipt: NpcCommandReceipt | None = None

    def __post_init__(self) -> None:
        if len(self.position) != 3:
            raise ValueError("NPC runtime position must have three components")
        if len(self.quaternion) != 4:
            raise ValueError("NPC runtime quaternion must have four components")
        if self.revision < 0:
            raise ValueError("NPC runtime revision cannot be negative")
        if not 0.0 <= self.clip_phase <= 1.0:
            raise ValueError("NPC animation clip_phase must be between 0 and 1")

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["position"] = list(self.position)
        result["quaternion"] = list(self.quaternion)
        if self.last_receipt is not None:
            result["last_receipt"] = self.last_receipt.to_dict()
        return result

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "NpcRuntimeState":
        receipt = payload.get("last_receipt")
        return cls(
            npc_id=str(payload["npc_id"]),
            revision=int(payload["revision"]),
            sim_time=float(payload["sim_time"]),
            position=tuple(float(value) for value in payload["position"]),  # type: ignore[arg-type]
            quaternion=tuple(float(value) for value in payload["quaternion"]),  # type: ignore[arg-type]
            locomotion=str(payload["locomotion"]),
            requested_animation=str(payload["requested_animation"]),
            resolved_clip=str(payload["resolved_clip"]),
            clip_phase=float(payload["clip_phase"]),
            phase_seed=int(payload.get("phase_seed", 0)),
            phase_offset=float(payload.get("phase_offset", 0.0)),
            last_marker=None if payload.get("last_marker") is None else str(payload["last_marker"]),
            pending_clip=(
                None if payload.get("pending_clip") is None else str(payload["pending_clip"])
            ),
            deferred_interrupt=bool(payload.get("deferred_interrupt", False)),
            route_revision=int(payload.get("route_revision", 0)),
            replan_attempt=int(payload.get("replan_attempt", 0)),
            stop_marker=(
                None if payload.get("stop_marker") is None else str(payload["stop_marker"])
            ),
            locomotion_failure=(
                None
                if payload.get("locomotion_failure") is None
                else str(payload["locomotion_failure"])
            ),
            animation_lifecycle=str(payload.get("animation_lifecycle", "requested")),
            transition=None if payload.get("transition") is None else str(payload["transition"]),
            animation_events=tuple(str(value) for value in payload.get("animation_events", ())),
            held_objects=tuple(str(value) for value in payload.get("held_objects", ())),
            interaction_id=(
                None if payload.get("interaction_id") is None else str(payload["interaction_id"])
            ),
            active_command_id=(
                None
                if payload.get("active_command_id") is None
                else str(payload["active_command_id"])
            ),
            last_receipt=(
                NpcCommandReceipt.from_dict(receipt) if isinstance(receipt, Mapping) else None
            ),
        )
