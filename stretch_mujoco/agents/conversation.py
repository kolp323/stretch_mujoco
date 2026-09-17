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
    """Lifecycle states owned by the logical conversation runtime.

    ``APPROACHING`` and ``ALIGNING`` are reserved for the action-driver
    integration.  This branch starts directly in ``ACTIVE`` only for the
    explicitly logical/mock path; it never claims that an embodied cue ran.
    """

    REQUESTED = "requested"
    APPROACHING = "approaching"
    ALIGNING = "aligning"
    ACTIVE = "active"
    COMPLETING = "completing"
    COMPLETED = "completed"
    TIMED_OUT = "timed_out"
    CANCELLED = "cancelled"
    FAILED = "failed"
    INTERRUPTED = "interrupted"  # schema-v1 terminal compatibility

    @property
    def terminal(self) -> bool:
        return self in {
            ConversationStatus.COMPLETED,
            ConversationStatus.TIMED_OUT,
            ConversationStatus.CANCELLED,
            ConversationStatus.FAILED,
            ConversationStatus.INTERRUPTED,
        }


class ConversationTerminalReason(str, Enum):
    COMPLETED = "completed"
    TIMED_OUT = "timed_out"
    CANCELLED = "cancelled"
    FAILED = "failed"
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
    CANCELLED = "conversation_cancelled"
    FAILED = "conversation_failed"


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


class ConversationPhase(str, Enum):
    REQUESTED = "requested"
    APPROACHING = "approaching"
    ALIGNING = "aligning"
    WAITING_FOR_TURN = "waiting_for_turn"
    WAITING_FOR_LLM = "waiting_for_llm"
    PLAYING_TURN = "playing_turn"
    RECOVERING = "recovering"
    TERMINAL = "terminal"


class DialogueAct(str, Enum):
    STATEMENT = "statement"
    CLARIFY = "clarify"
    ACKNOWLEDGE = "acknowledge"
    GREETING = "greeting"
    PROGRESS_QUERY = "progress_query"
    MEETING_INVITE = "meeting_invite"
    CONFLICT_RESOLUTION = "conflict_resolution"
    REQUEST = "request"
    HANDOVER_CONFIRM = "handover_confirm"


class ConversationInterruptPolicy(str, Enum):
    FINISH_TURN = "finish_turn"
    IMMEDIATE = "immediate"
    REJECT_WHILE_HANDOVER = "reject_while_handover"


class TurnStatus(str, Enum):
    PROPOSED = "proposed"
    ACCEPTED = "accepted"
    PLAYING = "playing"
    COMMITTED = "committed"
    REJECTED = "rejected"
    FAILED = "failed"


@dataclass(frozen=True)
class DialogueCandidate:
    request_id: str
    session_id: str
    turn_id: str
    speaker: str
    listener: str
    act: DialogueAct
    text: str
    proposed_at: float


@dataclass
class DialogueTurn:
    turn_id: str
    ordinal: int
    speaker: str
    listener: str
    act: DialogueAct
    text: str
    status: TurnStatus
    llm_request_id: str | None
    execution_id: str | None
    proposed_at: float
    started_at: float | None = None
    completed_at: float | None = None
    fallback_used: bool = False
    error: str | None = None


@dataclass(frozen=True)
class ConversationTransition:
    session_id: str
    phase: ConversationPhase
    status: ConversationStatus
    turn_id: str | None = None
    reason: str | None = None


@dataclass(frozen=True)
class SocialState:
    """Read-only social inputs; the scheduler never mutates agent state."""

    social_energy: float
    stress: float
    available: bool


@dataclass(frozen=True)
class SocialConversationProposal:
    initiator: str
    participant: str
    intent: ConversationIntent
    topic: str
    created_at: float
    metadata: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class SocialConversationDecision:
    proposal: SocialConversationProposal
    accepted: bool
    reason: str


@dataclass
class MeetingInvitation:
    invitation_id: str
    proposal: SocialConversationProposal
    expires_at: float
    accepted: bool | None = None


class NpcConversationScheduler:
    """Deterministically admit low-priority NPC social conversations.

    Sorting by proposal time and both participant IDs gives two simultaneous
    initiators one winner without either waiting on the other.  It is pure
    logical scheduling: embodied approach/alignment stays in the action branch.
    """

    def __init__(self, cooldown_minutes: float = 2.0) -> None:
        if cooldown_minutes < 0:
            raise ValueError("Social conversation cooldown must be non-negative")
        self.cooldown_minutes = cooldown_minutes
        self._cooldown_until: dict[tuple[str, str], float] = {}
        self.invitations: dict[str, MeetingInvitation] = {}
        self._next_invitation_number = 1

    def admit(
        self,
        proposals: Iterable[SocialConversationProposal],
        states: Mapping[str, SocialState],
        now: float,
    ) -> tuple[SocialConversationDecision, ...]:
        admitted: list[SocialConversationDecision] = []
        occupied: set[str] = set()
        for proposal in sorted(
            proposals,
            key=lambda item: (
                item.created_at,
                min(item.initiator, item.participant),
                max(item.initiator, item.participant),
                item.intent.value,
            ),
        ):
            pair = (
                min(proposal.initiator, proposal.participant),
                max(proposal.initiator, proposal.participant),
            )
            initiator = states.get(proposal.initiator)
            participant = states.get(proposal.participant)
            if proposal.initiator == proposal.participant or not proposal.topic.strip():
                admitted.append(SocialConversationDecision(proposal, False, "invalid_proposal"))
            elif proposal.intent not in NPC_INTENTS:
                admitted.append(SocialConversationDecision(proposal, False, "invalid_npc_intent"))
            elif initiator is None or participant is None:
                admitted.append(SocialConversationDecision(proposal, False, "unknown_participant"))
            elif pair in self._cooldown_until and now < self._cooldown_until[pair]:
                admitted.append(SocialConversationDecision(proposal, False, "cooldown"))
            elif proposal.initiator in occupied or proposal.participant in occupied:
                admitted.append(SocialConversationDecision(proposal, False, "participant_busy"))
            elif not initiator.available or not participant.available:
                admitted.append(
                    SocialConversationDecision(proposal, False, "participant_unavailable")
                )
            elif min(initiator.social_energy, participant.social_energy) <= 0.0:
                admitted.append(
                    SocialConversationDecision(proposal, False, "social_energy_depleted")
                )
            elif max(initiator.stress, participant.stress) >= 1.0:
                admitted.append(SocialConversationDecision(proposal, False, "stress_limit"))
            else:
                occupied.update(pair)
                self._cooldown_until[pair] = now + self.cooldown_minutes
                admitted.append(SocialConversationDecision(proposal, True, "accepted"))
        return tuple(admitted)

    def create_invitation(
        self, proposal: SocialConversationProposal, expires_at: float
    ) -> MeetingInvitation:
        if proposal.intent != ConversationIntent.MEETING_INVITATION:
            raise ValueError("Only meeting invitations can create invitation records")
        if expires_at <= proposal.created_at:
            raise ValueError("Meeting invitation expiry must be after creation")
        invitation = MeetingInvitation(
            f"meeting_invitation_{self._next_invitation_number:06d}", proposal, expires_at
        )
        self._next_invitation_number += 1
        self.invitations[invitation.invitation_id] = invitation
        return invitation

    def respond_invitation(
        self, invitation_id: str, accepted: bool, now: float
    ) -> MeetingInvitation:
        invitation = self.invitations[invitation_id]
        if invitation.accepted is not None:
            return invitation
        if now >= invitation.expires_at:
            invitation.accepted = False
        else:
            invitation.accepted = accepted
        return invitation

    def expire_invitations(self, now: float) -> tuple[MeetingInvitation, ...]:
        expired: list[MeetingInvitation] = []
        for invitation in self.invitations.values():
            if invitation.accepted is None and now >= invitation.expires_at:
                invitation.accepted = False
                expired.append(invitation)
        return tuple(expired)

    def cooldown_remaining(self, participant: str, now: float) -> float:
        """Return the largest active social cooldown involving one participant."""
        return max(
            (
                max(0.0, expires_at - now)
                for pair, expires_at in self._cooldown_until.items()
                if participant in pair
            ),
            default=0.0,
        )


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
    expected_speaker: str | None = None
    turn_deadline: float | None = None
    failure_reason: str | None = None
    robot_task_id: str | None = None
    handover_receipts: frozenset[str] = frozenset()
    phase: ConversationPhase = ConversationPhase.REQUESTED
    active_turn_id: str | None = None
    max_turns: int = 8
    parent_event_id: str | None = None
    suspended_plan_ids: dict[str, str] = field(default_factory=dict)
    error: str | None = None
    dialogue_turns: dict[str, DialogueTurn] = field(default_factory=dict)
    preferred_station_id: str | None = None
    station_id: str | None = None
    lease_id: str | None = None
    physical_receipt_ids: list[str] = field(default_factory=list)
    cleanup_evidence_ids: list[str] = field(default_factory=list)
    compatibility_mode: str | None = None
    production_evidence: bool = False

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
        self._seen_request_ids: set[str] = set()
        self._seen_turn_ids: set[str] = set()

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
        turn_policy: TurnPolicy = TurnPolicy.ROUND_ROBIN,
        turn_timeout: float | None = None,
        session_id: str | None = None,
        max_turns: int = 8,
        parent_event_id: str | None = None,
        require_spatial_ready: bool = True,
    ) -> ConversationSession:
        participant_tuple = tuple(dict.fromkeys(participants))
        if len(participant_tuple) != 2:
            raise ValueError("A conversation requires exactly two distinct participants")
        if not topic.strip() or timeout <= 0:
            raise ValueError("Conversation topic and timeout must be positive")
        if turn_timeout is not None and turn_timeout <= 0:
            raise ValueError("Conversation turn timeout must be positive")
        if max_turns <= 0:
            raise ValueError("Conversation max_turns must be positive")
        first, second = participant_tuple
        if first not in poses or second not in poses:
            raise ValueError("Conversation requires semantic poses for both participants")
        # The logical/mock path may only start once speakers already meet the
        # range and mutual-facing gate.  An embodied conversation instead
        # starts in APPROACHING and its action driver moves both people to the
        # rendezvous sites before the same gate is checked by the talk command.
        if require_spatial_ready and not self._can_speak(poses[first], poses[second]):
            raise ValueError("Participants are too far apart or are not facing each other")
        if any(
            session.session_id != session_id
            and not session.status.terminal
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
        resolved_session_id = session_id or f"conversation_{self._next_session_number:06d}"
        existing = self.sessions.get(resolved_session_id)
        if existing is not None:
            if existing.participants != participant_tuple or existing.topic != topic.strip():
                raise ValueError("session_id_conflict")
            return existing
        session = ConversationSession(
            session_id=resolved_session_id,
            participants=participant_tuple,
            topic=topic.strip(),
            turn=0,
            started_at=started_at,
            timeout=timeout,
            status=ConversationStatus.ACTIVE,
            interrupt_policy=interrupt_policy,
            participant_kinds=kinds,
            turn_policy=turn_policy,
            expected_speaker=(
                participant_tuple[0] if turn_policy == TurnPolicy.ROUND_ROBIN else None
            ),
            turn_deadline=(started_at + turn_timeout if turn_timeout is not None else None),
            phase=ConversationPhase.WAITING_FOR_TURN,
            max_turns=max_turns,
            parent_event_id=parent_event_id,
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
        if session.turn_policy == TurnPolicy.ROUND_ROBIN and speaker != session.expected_speaker:
            raise ValueError(
                f"Conversation '{session_id}' is waiting for '{session.expected_speaker}'"
            )
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
        if session.turn_policy == TurnPolicy.ROUND_ROBIN:
            session.expected_speaker = next(
                participant for participant in session.participants if participant != speaker
            )
        return entry

    def mark_approaching(self, session_id: str) -> ConversationTransition:
        return self._set_phase(session_id, ConversationPhase.APPROACHING)

    def mark_aligning(self, session_id: str) -> ConversationTransition:
        """Record that the embodied driver has moved from approach to facing alignment."""
        session = self._active(session_id)
        if session.phase != ConversationPhase.APPROACHING:
            raise ValueError("conversation_not_approaching")
        session.phase = ConversationPhase.ALIGNING
        return ConversationTransition(session_id, session.phase, session.status)

    def mark_aligned(
        self, session_id: str, receipt_ids: Iterable[str] = ()
    ) -> ConversationTransition:
        session = self._active(session_id)
        if session.phase not in {ConversationPhase.APPROACHING, ConversationPhase.ALIGNING}:
            raise ValueError("conversation_not_preparing")
        session.phase = ConversationPhase.WAITING_FOR_TURN
        session.handover_receipts = frozenset(receipt_ids)
        return ConversationTransition(session_id, session.phase, session.status)

    def propose(self, candidate: DialogueCandidate) -> ConversationTransition:
        session = self._active(candidate.session_id)
        if session.phase != ConversationPhase.WAITING_FOR_TURN:
            raise ValueError("conversation_not_ready_for_turn")
        if candidate.request_id in self._seen_request_ids:
            turn = session.dialogue_turns.get(candidate.turn_id)
            return ConversationTransition(
                session.session_id,
                session.phase,
                session.status,
                candidate.turn_id,
                None if turn is not None else "duplicate_request",
            )
        if candidate.turn_id in self._seen_turn_ids:
            raise ValueError("duplicate_turn_id")
        if candidate.speaker != session.expected_speaker:
            raise ValueError("unexpected_speaker")
        if (
            candidate.listener not in session.participants
            or candidate.listener == candidate.speaker
        ):
            raise ValueError("invalid_listener")
        if session.turn >= session.max_turns:
            raise ValueError("max_turns_reached")
        self._seen_request_ids.add(candidate.request_id)
        self._seen_turn_ids.add(candidate.turn_id)
        session.dialogue_turns[candidate.turn_id] = DialogueTurn(
            candidate.turn_id,
            session.turn + 1,
            candidate.speaker,
            candidate.listener,
            candidate.act,
            candidate.text,
            TurnStatus.ACCEPTED,
            candidate.request_id,
            None,
            candidate.proposed_at,
        )
        session.active_turn_id = candidate.turn_id
        session.phase = ConversationPhase.WAITING_FOR_LLM
        return ConversationTransition(
            session.session_id, session.phase, session.status, candidate.turn_id
        )

    def mark_turn_started(
        self, session_id: str, turn_id: str, execution_id: str
    ) -> ConversationTransition:
        session = self._active(session_id)
        turn = self._active_dialogue_turn(session, turn_id)
        if not execution_id:
            raise ValueError("missing_execution_id")
        turn.status = TurnStatus.PLAYING
        turn.execution_id = execution_id
        turn.started_at = turn.started_at or turn.proposed_at
        session.phase = ConversationPhase.PLAYING_TURN
        return ConversationTransition(session_id, session.phase, session.status, turn_id)

    def commit_turn(
        self, session_id: str, turn_id: str, completed_at: float
    ) -> ConversationTransition:
        session = self._active(session_id)
        turn = self._active_dialogue_turn(session, turn_id)
        if turn.status != TurnStatus.PLAYING:
            raise ValueError("turn_not_playing")
        turn.status = TurnStatus.COMMITTED
        turn.completed_at = completed_at
        session.turn += 1
        session.active_turn_id = None
        session.expected_speaker = turn.listener
        session.phase = ConversationPhase.WAITING_FOR_TURN
        return ConversationTransition(session_id, session.phase, session.status, turn_id)

    def reject_turn(self, session_id: str, turn_id: str, reason: str) -> ConversationTransition:
        session = self._active(session_id)
        turn = self._active_dialogue_turn(session, turn_id)
        turn.status = TurnStatus.REJECTED
        turn.error = reason
        session.active_turn_id = None
        session.phase = ConversationPhase.WAITING_FOR_TURN
        return ConversationTransition(session_id, session.phase, session.status, turn_id, reason)

    def interrupt(self, session_id: str, reason: str, now: float) -> ConversationTransition:
        del now
        session = self._terminal(session_id, ConversationStatus.CANCELLED, reason)
        session.phase = ConversationPhase.TERMINAL
        return ConversationTransition(session_id, session.phase, session.status, reason=reason)

    def check_deadlines(self, now: float) -> tuple[ConversationTransition, ...]:
        transitions: list[ConversationTransition] = []
        for session in self.expire(now):
            session.phase = ConversationPhase.TERMINAL
            transitions.append(
                ConversationTransition(
                    session.session_id, session.phase, session.status, reason=session.error
                )
            )
        return tuple(transitions)

    def _set_phase(self, session_id: str, phase: ConversationPhase) -> ConversationTransition:
        session = self._active(session_id)
        session.phase = phase
        return ConversationTransition(session_id, phase, session.status)

    @staticmethod
    def _active_dialogue_turn(session: ConversationSession, turn_id: str) -> DialogueTurn:
        if session.active_turn_id != turn_id:
            raise ValueError("turn_not_active")
        try:
            return session.dialogue_turns[turn_id]
        except KeyError as error:
            raise ValueError("unknown_turn") from error

    def complete(self, session_id: str) -> ConversationSession:
        session = self._session(session_id)
        if session.status.terminal:
            return session
        if session.status != ConversationStatus.ACTIVE:
            raise ValueError(f"Conversation '{session_id}' is {session.status.value}")
        session.status = ConversationStatus.COMPLETING
        session.status = ConversationStatus.COMPLETED
        session.terminal_reason = ConversationTerminalReason.COMPLETED
        session.phase = ConversationPhase.TERMINAL
        return session

    def cancel(self, session_id: str, reason: str = "cancelled") -> ConversationSession:
        return self._terminal(session_id, ConversationStatus.CANCELLED, reason)

    def fail(self, session_id: str, reason: str) -> ConversationSession:
        if not reason.strip():
            raise ValueError("Conversation failure requires a reason")
        return self._terminal(session_id, ConversationStatus.FAILED, reason)

    def expire(self, now: float) -> tuple[ConversationSession, ...]:
        expired: list[ConversationSession] = []
        for session in self.sessions.values():
            if session.status == ConversationStatus.ACTIVE and (
                now >= session.deadline
                or (session.turn_deadline is not None and now >= session.turn_deadline)
            ):
                session.status = ConversationStatus.TIMED_OUT
                session.terminal_reason = ConversationTerminalReason.TIMED_OUT
                session.phase = ConversationPhase.TERMINAL
                if session.turn_deadline is not None and now >= session.turn_deadline:
                    session.failure_reason = "turn_timeout"
                expired.append(session)
        return tuple(expired)

    def _active(self, session_id: str) -> ConversationSession:
        session = self._session(session_id)
        if session.status != ConversationStatus.ACTIVE:
            raise ValueError(f"Conversation '{session_id}' is {session.status.value}")
        return session

    def _session(self, session_id: str) -> ConversationSession:
        try:
            return self.sessions[session_id]
        except KeyError as error:
            raise KeyError(f"Unknown conversation '{session_id}'") from error

    def _terminal(
        self, session_id: str, status: ConversationStatus, reason: str
    ) -> ConversationSession:
        session = self._session(session_id)
        if session.status.terminal:
            return session
        if session.status != ConversationStatus.ACTIVE:
            raise ValueError(f"Conversation '{session_id}' is {session.status.value}")
        session.status = status
        session.terminal_reason = ConversationTerminalReason(status.value)
        session.failure_reason = reason
        session.error = reason
        session.phase = ConversationPhase.TERMINAL
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
            self._angular_distance(first.yaw, first_direction) <= limit + 1e-9
            and self._angular_distance(second.yaw, second_direction) <= limit + 1e-9
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
