"""Participant barriers for handovers and other coordinated interactions."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum


class InteractionStatus(str, Enum):
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMED_OUT = "timed_out"

    @property
    def terminal(self) -> bool:
        return self != InteractionStatus.RUNNING


@dataclass
class InteractionSession:
    kind: str
    participants: tuple[str, ...]
    object_id: str
    phase_requirements: tuple[tuple[str, tuple[str, ...]], ...]
    deadline: float | None = None
    session_id: str = field(default_factory=lambda: f"interaction_{uuid.uuid4().hex}")
    phase_index: int = 0
    acknowledgements: set[str] = field(default_factory=set)
    completed_phases: list[str] = field(default_factory=list)
    status: InteractionStatus = InteractionStatus.RUNNING
    error: str | None = None

    @property
    def phase(self) -> str:
        if self.phase_index >= len(self.phase_requirements):
            return "complete"
        return self.phase_requirements[self.phase_index][0]


class InteractionCoordinator:
    """Advance a session only after the required actors confirm physical evidence."""

    def __init__(self) -> None:
        self.sessions: dict[str, InteractionSession] = {}

    def start(
        self,
        kind: str,
        participants: tuple[str, ...],
        object_id: str,
        phase_requirements: tuple[tuple[str, tuple[str, ...]], ...],
        *,
        deadline: float | None = None,
        session_id: str | None = None,
    ) -> InteractionSession:
        if len(set(participants)) != len(participants) or len(participants) < 2:
            raise ValueError("Interaction requires at least two unique participants")
        participant_set = set(participants)
        for phase, required in phase_requirements:
            if not phase or not required or not set(required) <= participant_set:
                raise ValueError(f"Invalid interaction phase '{phase}'")
        session = InteractionSession(
            kind,
            participants,
            object_id,
            phase_requirements,
            deadline,
            session_id or f"interaction_{uuid.uuid4().hex}",
        )
        previous = self.sessions.get(session.session_id)
        if previous is not None:
            return previous
        self.sessions[session.session_id] = session
        return session

    def acknowledge(self, session_id: str, participant: str, phase: str) -> InteractionSession:
        session = self.sessions[session_id]
        if session.status.terminal:
            return session
        if phase != session.phase:
            raise ValueError(f"Expected interaction phase '{session.phase}', got '{phase}'")
        required = session.phase_requirements[session.phase_index][1]
        if participant not in required:
            raise ValueError(f"Participant '{participant}' is not required for phase '{phase}'")
        session.acknowledgements.add(participant)
        if set(required) <= session.acknowledgements:
            session.completed_phases.append(session.phase)
            session.phase_index += 1
            session.acknowledgements.clear()
            if session.phase_index == len(session.phase_requirements):
                session.status = InteractionStatus.SUCCEEDED
        return session

    def fail(self, session_id: str, reason: str) -> InteractionSession:
        session = self.sessions[session_id]
        if not session.status.terminal:
            session.status = InteractionStatus.FAILED
            session.error = reason
        return session

    def cancel(self, session_id: str, reason: str = "cancelled") -> InteractionSession:
        session = self.sessions[session_id]
        if not session.status.terminal:
            session.status = InteractionStatus.CANCELLED
            session.error = reason
        return session

    def check_deadlines(self, now: float) -> tuple[InteractionSession, ...]:
        timed_out = []
        for session in self.sessions.values():
            if (
                not session.status.terminal
                and session.deadline is not None
                and now > session.deadline
            ):
                session.status = InteractionStatus.TIMED_OUT
                session.error = "deadline_exceeded"
                timed_out.append(session)
        return tuple(timed_out)
