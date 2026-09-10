"""Deterministic validation boundary for untrusted dialogue candidates."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any, Mapping

from .conversation import ConversationSession, DialogueAct, DialogueCandidate


@dataclass(frozen=True)
class DialoguePolicyConfig:
    max_chars: int = 240
    max_turns: int = 8
    session_timeout_seconds: float = 45.0
    turn_timeout_seconds: float = 15.0
    pair_cooldown_seconds: float = 30.0
    min_social_energy: float = 0.10
    distance_min: float = 0.45
    distance_max: float = 0.95
    yaw_tolerance: float = 0.30
    fallback_templates: Mapping[DialogueAct, tuple[str, ...]] = field(
        default_factory=lambda: {
            DialogueAct.STATEMENT: ("Let's continue with the agreed plan.",),
            DialogueAct.CLARIFY: ("Could you clarify the next step?",),
            DialogueAct.ACKNOWLEDGE: ("Acknowledged.",),
            DialogueAct.GREETING: ("Hello.",),
            DialogueAct.PROGRESS_QUERY: ("How is the task progressing?",),
            DialogueAct.MEETING_INVITE: ("Would you join the meeting?",),
            DialogueAct.CONFLICT_RESOLUTION: ("Let's resolve this safely.",),
            DialogueAct.REQUEST: ("Could you help with this request?",),
            DialogueAct.HANDOVER_CONFIRM: ("Handover confirmed.",),
        }
    )
    sensitive_patterns: tuple[str, ...] = (
        r"\b(?:api[_ -]?key|password|secret|bearer\s+[\w.-]+|token)\b",
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
        r"\b(?:\+?\d[\d .()-]{7,}\d)\b",
        r"\b[A-Za-z0-9]{24,}\b",
    )

    def __post_init__(self) -> None:
        if self.max_chars <= 0 or self.max_turns <= 0:
            raise ValueError("Dialogue limits must be positive")
        if self.session_timeout_seconds <= 0 or self.turn_timeout_seconds <= 0:
            raise ValueError("Dialogue timeouts must be positive")
        if not 0 <= self.min_social_energy <= 1:
            raise ValueError("min_social_energy must be in [0, 1]")
        if not 0 <= self.distance_min <= self.distance_max:
            raise ValueError("Dialogue distance interval is invalid")
        if not 0 < self.yaw_tolerance <= 3.141592653589793:
            raise ValueError("Dialogue yaw tolerance is invalid")


@dataclass(frozen=True)
class SanitizedDialogue:
    candidate: DialogueCandidate
    fallback_used: bool = False


class DialoguePolicy:
    """Keep provider output as a candidate until the runtime approves it."""

    def __init__(
        self, config: DialoguePolicyConfig | None = None, *, runtime_seed: int = 0
    ) -> None:
        self.config = config or DialoguePolicyConfig()
        self.runtime_seed = runtime_seed
        self._last_pair_at: dict[tuple[str, str], float] = {}
        self._last_speaker_at: dict[str, float] = {}
        self._sensitive = tuple(
            re.compile(pattern, re.IGNORECASE) for pattern in self.config.sensitive_patterns
        )

    def validate_and_sanitize(
        self,
        candidate: DialogueCandidate,
        session: ConversationSession,
        world: Any | None = None,
        *,
        now: float | None = None,
    ) -> SanitizedDialogue:
        del world  # semantic authorization remains in OfficeAgentRuntime.
        if candidate.session_id != session.session_id:
            raise ValueError("session_mismatch")
        if (
            candidate.speaker not in session.participants
            or candidate.listener not in session.participants
        ):
            raise ValueError("participant_mismatch")
        if candidate.speaker == candidate.listener:
            raise ValueError("speaker_listener_mismatch")
        if session.expected_speaker is not None and candidate.speaker != session.expected_speaker:
            raise ValueError("unexpected_speaker")
        if session.turn >= min(session.max_turns, self.config.max_turns):
            raise ValueError("max_turns_reached")
        if not candidate.turn_id.strip() or not candidate.request_id.strip():
            raise ValueError("missing_candidate_id")
        clock = candidate.proposed_at if now is None else now
        pair = tuple(sorted((candidate.speaker, candidate.listener)))
        if clock - self._last_pair_at.get(pair, float("-inf")) < self.config.pair_cooldown_seconds:
            raise ValueError("pair_cooldown")
        text = self._clean_text(candidate.text)
        if not text or any(pattern.search(text) for pattern in self._sensitive):
            return SanitizedDialogue(self.fallback(candidate, "unsafe_or_empty"), True)
        approved = DialogueCandidate(
            candidate.request_id,
            candidate.session_id,
            candidate.turn_id,
            candidate.speaker,
            candidate.listener,
            candidate.act,
            text,
            candidate.proposed_at,
        )
        return SanitizedDialogue(approved)

    def note_committed(self, candidate: DialogueCandidate, now: float) -> None:
        self._last_speaker_at[candidate.speaker] = now
        self._last_pair_at[tuple(sorted((candidate.speaker, candidate.listener)))] = now

    def fallback(self, candidate: DialogueCandidate, reason: str = "fallback") -> DialogueCandidate:
        templates = self.config.fallback_templates.get(candidate.act)
        if not templates:
            raise ValueError(f"No fallback template for {candidate.act.value}")
        seed = hashlib.sha256(
            f"{self.runtime_seed}:{candidate.session_id}:{candidate.turn_id}:{candidate.act.value}:{reason}".encode()
        ).digest()
        text = templates[int.from_bytes(seed[:8], "big") % len(templates)]
        return DialogueCandidate(
            candidate.request_id,
            candidate.session_id,
            candidate.turn_id,
            candidate.speaker,
            candidate.listener,
            candidate.act,
            text,
            candidate.proposed_at,
        )

    def _clean_text(self, text: str) -> str:
        clean = " ".join(str(text).replace("\x00", " ").split())
        if len(clean) <= self.config.max_chars:
            return clean
        boundary = max(clean.rfind(mark, 0, self.config.max_chars) for mark in ".!?。！？")
        if boundary > 0:
            return clean[: boundary + 1]
        return clean[: self.config.max_chars].rstrip()
