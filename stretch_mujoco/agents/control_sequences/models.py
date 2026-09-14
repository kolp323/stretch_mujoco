"""Immutable public data model for versioned office control sequences."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Mapping


class ControlSequenceError(ValueError):
    """Raised when a sequence cannot be safely loaded or compiled."""


class ControlMode(str, Enum):
    YAML = "yaml"
    LLM = "llm"


class SequencePurpose(str, Enum):
    ACCEPTANCE = "acceptance"
    RUNTIME = "runtime"


class StepKind(str, Enum):
    ACTION = "action"
    ANIMATION = "animation"
    PARALLEL = "parallel"
    CONVERSATION = "conversation"
    LLM_REQUEST = "llm_request"
    WAIT = "wait"
    ASSERT = "assert"
    CHOOSE = "choose"
    REPEAT = "repeat"


class FailurePolicy(str, Enum):
    FAIL_SEQUENCE = "fail_sequence"
    RETRY = "retry"
    SKIP = "skip"
    FALLBACK = "fallback"
    REPLAN = "replan"


class StepStatus(str, Enum):
    PENDING = "pending"
    READY = "ready"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    TIMED_OUT = "timed_out"
    CANCELLED = "cancelled"
    RECOVERING = "recovering"


TERMINAL_STEP_STATUSES = frozenset(
    {StepStatus.SUCCEEDED, StepStatus.FAILED, StepStatus.TIMED_OUT, StepStatus.CANCELLED}
)


@dataclass(frozen=True)
class FailureHandling:
    policy: FailurePolicy = FailurePolicy.FAIL_SEQUENCE
    max_attempts: int = 1
    backoff_seconds: float = 0.0
    fallback_step: str | None = None


@dataclass(frozen=True)
class SequenceStep:
    """A closed step payload. ``payload`` has been structurally validated by the loader."""

    step_id: str
    kind: StepKind
    timeout_seconds: float
    failure: FailureHandling
    payload: Mapping[str, Any]
    caption: str | None = None
    tags: tuple[str, ...] = ()
    children: tuple["SequenceStep", ...] = ()


@dataclass(frozen=True)
class LlmAutonomy:
    allowed_actions: tuple[str, ...]
    allowed_locations: tuple[str, ...]
    decision_interval_seconds: tuple[float, float] = (5.0, 15.0)
    plan_horizon_actions: int = 1
    max_decisions: int | None = None
    max_consecutive_failures: int = 3
    max_replans_per_segment: int = 1
    stop: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LlmSettings:
    enabled: bool = False
    provider_config: Path | None = None
    daily_budget: int | None = None
    failure_policy: FailurePolicy = FailurePolicy.FAIL_SEQUENCE
    record_responses: bool = False
    replay_file: Path | None = None
    autonomy: LlmAutonomy | None = None


@dataclass(frozen=True)
class BehaviorSequence:
    schema_version: int
    sequence_id: str
    purpose: SequencePurpose
    default_control_mode: ControlMode
    seed: int | None
    source_path: Path
    source_sha256: str
    scene: Mapping[str, Path]
    participants: Mapping[str, str]
    locations: Mapping[str, Any]
    defaults: Mapping[str, Any]
    setup: Mapping[str, Any]
    steps: tuple[SequenceStep, ...]
    acceptance: Mapping[str, Any]
    llm: LlmSettings


@dataclass(frozen=True)
class CompiledStep:
    step: SequenceStep
    actor_id: str | None = None
    target_id: str | None = None
    children: tuple["CompiledStep", ...] = ()


@dataclass(frozen=True)
class CompiledSequence:
    sequence: BehaviorSequence
    control_mode: ControlMode
    steps: tuple[CompiledStep, ...]
    plan_sha256: str
    covered_actions: frozenset[str]
    covered_clips: frozenset[str]


@dataclass
class StepExecution:
    step_id: str
    correlation_id: str
    attempt: int = 1
    status: StepStatus = StepStatus.PENDING
    started_at: float | None = None
    finished_at: float | None = None
    receipt_ids: tuple[str, ...] = ()
    error: str | None = None


@dataclass(frozen=True)
class PlanSegment:
    """A source-neutral, already structured portion of a plan."""

    segment_id: str
    steps: tuple[SequenceStep, ...]
    source: str


def resolve_control_mode(
    requested: str | ControlMode | None, default: str | ControlMode | None
) -> ControlMode:
    """Implement the single documented CLI > YAML > safe-default precedence rule."""
    value = requested if requested is not None else default
    if value is None:
        return ControlMode.YAML
    try:
        return value if isinstance(value, ControlMode) else ControlMode(value)
    except ValueError as error:
        raise ControlSequenceError("control mode must be 'yaml' or 'llm'") from error
