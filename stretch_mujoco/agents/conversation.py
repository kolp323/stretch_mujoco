"""Bounded, runtime-owned conversations between office participants."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Iterable, Mapping


CONVERSATION_SCHEMA_VERSION = 1
DEFAULT_MAX_OBSERVATION_AGE = 0.5


class ConversationStatus(str, Enum):
    ACTIVE = "active"
    COMPLETED = "completed"
    TIMED_OUT = "timed_out"
    INTERRUPTED = "interrupted"


class ConversationTerminalReason(str, Enum):
    COMPLETED = "completed"
    TIMED_OUT = "timed_out"
    INTERRUPTED = "interrupted"


class InterruptPolicy(str, Enum):
    REJECT = "reject"
    """Reject a request that would interrupt an occupied participant."""

    ALLOW = "allow"
    """Legacy logical-only interrupt policy; never interrupts an embodied action."""

    SAFE_MARKER = "safe_marker"
    IMMEDIATE = "immediate"


class ConversationParticipantKind(str, Enum):
    UNKNOWN = "unknown"
    NPC = "npc"
    STRETCH_ROBOT = "stretch_robot"


class TurnPolicy(str, Enum):
    """Turn policy is contractual; enforcement is added with lifecycle work."""

    FREE_FORM = "free_form"
    ROUND_ROBIN = "round_robin"


class ConversationEvent(str, Enum):
    STARTED = "conversation_started"
    TURN = "conversation_turn"
    COMPLETED = "conversation_completed"
    TIMED_OUT = "conversation_timed_out"


class ConversationObservationSource(str, Enum):
    SEMANTIC_WORLD = "semantic_world"
    OFFLINE_RECORDING = "offline_recording"
    LEGACY_OBJECTS = "legacy_objects"


class ConversationErrorCode(str, Enum):
    INVALID_SNAPSHOT = "invalid_snapshot"
    INVALID_OBSERVATION_TIME = "invalid_observation_time"
    STALE_OBSERVATION = "stale_observation"
    FUTURE_OBSERVATION = "future_observation"
    MISSING_PARTICIPANT = "missing_participant"
    INVALID_POSE = "invalid_pose"
    PARTICIPANT_NOT_VISIBLE = "participant_not_visible"


class ConversationPerceptionError(ValueError):
    """A stable error emitted when an untrusted observation cannot start a session."""

    def __init__(self, code: ConversationErrorCode, message: str) -> None:
        self.code = code
        super().__init__(f"{code.value}: {message}")


class ConversationIntent(str, Enum):
    REQUEST = "request"
    CLARIFY = "clarify"
    ACKNOWLEDGE = "acknowledge"
    HANDOVER_CONFIRM = "handover_confirm"
    GREETING = "greeting"
    PROGRESS_INQUIRY = "progress_inquiry"
    MEETING_INVITATION = "meeting_invitation"
    CONFLICT_RESOLUTION = "conflict_resolution"


ROBOT_INTENTS = frozenset(
    {
        ConversationIntent.REQUEST,
        ConversationIntent.CLARIFY,
        ConversationIntent.ACKNOWLEDGE,
        ConversationIntent.HANDOVER_CONFIRM,
    }
)
NPC_INTENTS = frozenset(
    {
        ConversationIntent.GREETING,
        ConversationIntent.PROGRESS_INQUIRY,
        ConversationIntent.MEETING_INVITATION,
        ConversationIntent.CONFLICT_RESOLUTION,
    }
)

_SENSITIVE_TEXT = re.compile(
    r"\b(?:api[_ -]?key|password|secret|bearer\s+[\w.-]+|token)\b", re.IGNORECASE
)
_FALLBACKS = {
    ConversationIntent.REQUEST: "Could you help with this request?",
    ConversationIntent.CLARIFY: "Could you clarify the next step?",
    ConversationIntent.ACKNOWLEDGE: "Acknowledged.",
    ConversationIntent.HANDOVER_CONFIRM: "Handover confirmed.",
    ConversationIntent.GREETING: "Hello.",
    ConversationIntent.PROGRESS_INQUIRY: "How is the task progressing?",
    ConversationIntent.MEETING_INVITATION: "Would you join the meeting?",
    ConversationIntent.CONFLICT_RESOLUTION: "Let's resolve this safely.",
}


@dataclass(frozen=True)
class SpatialPose:
    position: tuple[float, float, float]
    yaw: float


@dataclass(frozen=True)
class ConversationPerception:
    """Read-only, validated spatial input for a conversation decision.

    The adapter accepts the live ``SemanticWorld.pose_snapshot()`` shape
    (``time`` + ``objects`` + quaternion), the offline recording shape
    (``sim_time`` + ``agents`` + yaw), and the pre-P0 ``objects`` + yaw shape.
    A legacy payload without a timestamp is treated as observed *now* only at
    this boundary so existing callers remain compatible; new producers must
    provide a finite timestamp to receive stale-observation protection.
    """

    schema_version: int
    source: ConversationObservationSource
    observed_at: float
    poses: Mapping[str, SpatialPose]

    @classmethod
    def from_snapshot(
        cls,
        snapshot: Mapping[str, Any],
        participants: Iterable[str],
        *,
        now: float | None = None,
        max_age: float = DEFAULT_MAX_OBSERVATION_AGE,
    ) -> "ConversationPerception":
        if not isinstance(snapshot, Mapping):
            raise ConversationPerceptionError(
                ConversationErrorCode.INVALID_SNAPSHOT,
                "conversation observation must be a mapping",
            )
        if not math.isfinite(max_age) or max_age < 0:
            raise ValueError("Conversation observation max_age must be a non-negative finite value")
        if now is not None and (isinstance(now, bool) or not math.isfinite(now)):
            raise ValueError("Conversation observation clock must be finite")

        participant_tuple = tuple(participants)
        if len(participant_tuple) != len(set(participant_tuple)):
            raise ValueError("Conversation observation participants must be distinct")
        if not participant_tuple:
            raise ValueError("Conversation observation requires at least one participant")

        source, observations, time_key = cls._observation_fields(snapshot)
        raw_time = snapshot.get(time_key)
        has_timestamp = raw_time is not None
        if has_timestamp:
            observed_at = cls._finite_number(
                raw_time,
                ConversationErrorCode.INVALID_OBSERVATION_TIME,
                "observation timestamp",
            )
        elif now is not None:
            observed_at = now
        else:
            observed_at = 0.0

        if now is not None and has_timestamp:
            age = now - observed_at
            if age > max_age:
                raise ConversationPerceptionError(
                    ConversationErrorCode.STALE_OBSERVATION,
                    f"observation age {age:.3f} exceeds {max_age:.3f}",
                )
            if age < -max_age:
                raise ConversationPerceptionError(
                    ConversationErrorCode.FUTURE_OBSERVATION,
                    f"observation is {-age:.3f} ahead of the runtime clock",
                )

        poses: dict[str, SpatialPose] = {}
        for participant in participant_tuple:
            payload = observations.get(participant)
            if not isinstance(payload, Mapping):
                raise ConversationPerceptionError(
                    ConversationErrorCode.MISSING_PARTICIPANT,
                    f"observation is missing participant '{participant}'",
                )
            if payload.get("visible") is False or payload.get("valid") is False:
                raise ConversationPerceptionError(
                    ConversationErrorCode.PARTICIPANT_NOT_VISIBLE,
                    f"participant '{participant}' is not available for face-to-face dialogue",
                )
            poses[participant] = cls._pose_from_payload(participant, payload)

        return cls(
            schema_version=CONVERSATION_SCHEMA_VERSION,
            source=source,
            observed_at=observed_at,
            poses=MappingProxyType(poses),
        )

    @staticmethod
    def _observation_fields(
        snapshot: Mapping[str, Any],
    ) -> tuple[ConversationObservationSource, Mapping[str, Any], str]:
        agents = snapshot.get("agents")
        if isinstance(agents, Mapping):
            return ConversationObservationSource.OFFLINE_RECORDING, agents, "sim_time"
        objects = snapshot.get("objects")
        if not isinstance(objects, Mapping):
            raise ConversationPerceptionError(
                ConversationErrorCode.INVALID_SNAPSHOT,
                "observation requires an 'objects' or 'agents' mapping",
            )
        if any(isinstance(value, Mapping) and "quaternion" in value for value in objects.values()):
            return ConversationObservationSource.SEMANTIC_WORLD, objects, "time"
        return ConversationObservationSource.LEGACY_OBJECTS, objects, "time"

    @classmethod
    def _pose_from_payload(cls, participant: str, payload: Mapping[str, Any]) -> SpatialPose:
        position = payload.get("position")
        if not isinstance(position, (list, tuple)) or len(position) != 3:
            raise ConversationPerceptionError(
                ConversationErrorCode.INVALID_POSE,
                f"participant '{participant}' has no three-axis position",
            )
        coordinates = tuple(
            cls._finite_number(value, ConversationErrorCode.INVALID_POSE, "position")
            for value in position
        )
        raw_yaw = payload.get("yaw")
        if raw_yaw is None:
            raw_yaw = cls._yaw_from_quaternion(participant, payload.get("quaternion"))
        yaw = cls._finite_number(raw_yaw, ConversationErrorCode.INVALID_POSE, "yaw")
        return SpatialPose((coordinates[0], coordinates[1], coordinates[2]), yaw)

    @classmethod
    def _yaw_from_quaternion(cls, participant: str, quaternion: Any) -> float:
        if not isinstance(quaternion, (list, tuple)) or len(quaternion) != 4:
            raise ConversationPerceptionError(
                ConversationErrorCode.INVALID_POSE,
                f"participant '{participant}' has no yaw or quaternion",
            )
        w, x, y, z = (
            cls._finite_number(value, ConversationErrorCode.INVALID_POSE, "quaternion")
            for value in quaternion
        )
        norm = math.sqrt(w * w + x * x + y * y + z * z)
        if norm == 0:
            raise ConversationPerceptionError(
                ConversationErrorCode.INVALID_POSE,
                f"participant '{participant}' has a zero-length quaternion",
            )
        w, x, y, z = (value / norm for value in (w, x, y, z))
        return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

    @staticmethod
    def _finite_number(value: Any, code: ConversationErrorCode, field_name: str) -> float:
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise ConversationPerceptionError(code, f"{field_name} must be a finite number")
        return float(value)


@dataclass(frozen=True)
class ConversationTurn:
    speaker: str
    intent: ConversationIntent
    text: str
    timestamp: float
    used_fallback: bool = False
    turn_id: str = ""


@dataclass
class ConversationSession:
    session_id: str
    participants: tuple[str, ...]
    topic: str
    turn: int
    started_at: float
    timeout: float
    status: ConversationStatus
    transcript: list[ConversationTurn] = field(default_factory=list)
    interrupt_policy: InterruptPolicy = InterruptPolicy.REJECT
    participant_kinds: tuple[ConversationParticipantKind, ...] = ()
    turn_policy: TurnPolicy = TurnPolicy.FREE_FORM
    terminal_reason: ConversationTerminalReason | None = None

    @property
    def deadline(self) -> float:
        return self.started_at + self.timeout


class ConversationCoordinator:
    """Validate, record, and close small dialogue sessions without calling an LLM."""

    def __init__(
        self,
        *,
        max_distance: float = 2.0,
        max_facing_degrees: float = 60.0,
        cooldown_minutes: float = 0.25,
        max_text_length: int = 160,
    ) -> None:
        if max_distance <= 0 or not 0 < max_facing_degrees <= 180:
            raise ValueError("Conversation spatial limits must be positive")
        if cooldown_minutes < 0 or max_text_length <= 0:
            raise ValueError("Conversation text limits are invalid")
        self.max_distance = max_distance
        self.max_facing_degrees = max_facing_degrees
        self.cooldown_minutes = cooldown_minutes
        self.max_text_length = max_text_length
        self.sessions: dict[str, ConversationSession] = {}
        self._last_message_at: dict[str, float] = {}
        self._next_session_number = 1

    def start(
        self,
        participants: Iterable[str],
        topic: str,
        started_at: float,
        poses: Mapping[str, SpatialPose],
        *,
        timeout: float = 2.0,
        interrupt_policy: InterruptPolicy = InterruptPolicy.REJECT,
        participant_kinds: Mapping[str, ConversationParticipantKind | str] | None = None,
        turn_policy: TurnPolicy = TurnPolicy.FREE_FORM,
    ) -> ConversationSession:
        participant_tuple = tuple(dict.fromkeys(participants))
        if len(participant_tuple) != 2:
            raise ValueError("A conversation requires exactly two distinct participants")
        if not topic.strip() or timeout <= 0:
            raise ValueError("Conversation topic and timeout must be positive")
        first, second = participant_tuple
        if first not in poses or second not in poses:
            raise ValueError("Conversation requires semantic poses for both participants")
        if not self._can_speak(poses[first], poses[second]):
            raise ValueError("Participants are too far apart or are not facing each other")
        if any(
            session.status == ConversationStatus.ACTIVE
            and set(session.participants).intersection(participant_tuple)
            for session in self.sessions.values()
        ):
            raise ValueError("A conversation participant already has an active session")
        kinds = tuple(
            ConversationParticipantKind(
                ConversationParticipantKind.UNKNOWN
                if participant_kinds is None
                else participant_kinds.get(participant, ConversationParticipantKind.UNKNOWN)
            )
            for participant in participant_tuple
        )
        session = ConversationSession(
            session_id=f"conversation_{self._next_session_number:06d}",
            participants=participant_tuple,
            topic=topic.strip(),
            turn=0,
            started_at=started_at,
            timeout=timeout,
            status=ConversationStatus.ACTIVE,
            interrupt_policy=interrupt_policy,
            participant_kinds=kinds,
            turn_policy=turn_policy,
        )
        self._next_session_number += 1
        self.sessions[session.session_id] = session
        return session

    def record_candidate(
        self,
        session_id: str,
        speaker: str,
        intent: ConversationIntent | str,
        text: str | None,
        timestamp: float,
        *,
        includes_robot: bool,
    ) -> ConversationTurn:
        session = self._active(session_id)
        if speaker not in session.participants:
            raise ValueError(f"Speaker '{speaker}' is not in the conversation")
        try:
            normalized_intent = ConversationIntent(intent)
        except ValueError as error:
            raise ValueError(f"Unsupported conversation intent '{intent}'") from error
        allowed = ROBOT_INTENTS if includes_robot else NPC_INTENTS
        if normalized_intent not in allowed:
            raise ValueError(
                f"Intent '{normalized_intent.value}' is not valid for this conversation"
            )
        last_message = self._last_message_at.get(speaker)
        if last_message is not None and timestamp - last_message < self.cooldown_minutes:
            raise ValueError(f"Speaker '{speaker}' is in conversation cooldown")
        clean_text, used_fallback = self._safe_text(normalized_intent, text)
        entry = ConversationTurn(
            speaker,
            normalized_intent,
            clean_text,
            timestamp,
            used_fallback,
            f"{session.session_id}:turn:{session.turn + 1}",
        )
        session.transcript.append(entry)
        session.turn += 1
        self._last_message_at[speaker] = timestamp
        return entry

    def complete(self, session_id: str) -> ConversationSession:
        session = self._active(session_id)
        session.status = ConversationStatus.COMPLETED
        session.terminal_reason = ConversationTerminalReason.COMPLETED
        return session

    def expire(self, now: float) -> tuple[ConversationSession, ...]:
        expired: list[ConversationSession] = []
        for session in self.sessions.values():
            if session.status == ConversationStatus.ACTIVE and now >= session.deadline:
                session.status = ConversationStatus.TIMED_OUT
                session.terminal_reason = ConversationTerminalReason.TIMED_OUT
                expired.append(session)
        return tuple(expired)

    def _active(self, session_id: str) -> ConversationSession:
        try:
            session = self.sessions[session_id]
        except KeyError as error:
            raise KeyError(f"Unknown conversation '{session_id}'") from error
        if session.status != ConversationStatus.ACTIVE:
            raise ValueError(f"Conversation '{session_id}' is {session.status.value}")
        return session

    def _can_speak(self, first: SpatialPose, second: SpatialPose) -> bool:
        if not all(
            math.isfinite(value)
            for value in (*first.position, first.yaw, *second.position, second.yaw)
        ):
            return False
        dx = second.position[0] - first.position[0]
        dy = second.position[1] - first.position[1]
        distance = math.hypot(dx, dy)
        if distance > self.max_distance:
            return False
        first_direction = math.atan2(dy, dx)
        second_direction = math.atan2(-dy, -dx)
        limit = math.radians(self.max_facing_degrees)
        return (
            self._angular_distance(first.yaw, first_direction) <= limit
            and self._angular_distance(second.yaw, second_direction) <= limit
        )

    @staticmethod
    def _angular_distance(first: float, second: float) -> float:
        return abs((first - second + math.pi) % (2 * math.pi) - math.pi)

    def _safe_text(self, intent: ConversationIntent, text: str | None) -> tuple[str, bool]:
        candidate = "" if text is None else " ".join(text.split())
        if (
            not candidate
            or len(candidate) > self.max_text_length
            or _SENSITIVE_TEXT.search(candidate)
        ):
            return _FALLBACKS[intent], True
        return candidate, False
