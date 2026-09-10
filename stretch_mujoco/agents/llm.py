"""Event-only LLM request gate; never called from physics or animation updates."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable


class LLMTrigger(str, Enum):
    DAY_START = "day_start"
    NEW_TASK = "new_task"
    DIALOGUE = "dialogue"
    REPEATED_FAILURE = "repeated_failure"
    UNEXPECTED_CHANGE = "unexpected_change"
    REINTERPRET_PLAN = "reinterpret_plan"


@dataclass(frozen=True)
class LLMRequest:
    trigger: LLMTrigger
    agent_id: str
    day: int
    minute_of_day: float
    context: dict[str, Any] = field(default_factory=dict)
    request_id: str = ""
    correlation_id: str | None = None
    session_id: str | None = None
    turn_id: str | None = None
    deadline: float | None = None


class EventDrivenLLMGateway:
    """Queue allowed LLM events and enforce a per-agent daily call budget."""

    def __init__(self, daily_budget: int = 30) -> None:
        if daily_budget <= 0:
            raise ValueError("LLM daily budget must be positive")
        self.daily_budget = daily_budget
        self._pending: list[LLMRequest] = []
        self._calls: dict[tuple[int, str], int] = {}
        self._issued: dict[tuple[int, str], int] = {}
        self._completed: dict[str, dict[str, Any]] = {}
        self._next_request_number = 1
        self.metrics: dict[str, int] = {"issued": 0, "skipped": 0, "calls": 0, "rejected": 0}

    def queue(
        self,
        trigger: LLMTrigger,
        agent_id: str,
        day: int,
        minute_of_day: float,
        context: dict[str, Any] | None = None,
    ) -> bool:
        key = (day, agent_id)
        if self._issued.get(key, 0) >= self.daily_budget:
            self.metrics["skipped"] += 1
            return False
        payload = dict(context or {})
        request_id = str(payload.get("request_id") or f"llm_{self._next_request_number:08d}")
        self._next_request_number += 1
        if (
            any(item.request_id == request_id for item in self._pending)
            or request_id in self._completed
        ):
            self.metrics["skipped"] += 1
            return False
        self._pending.append(
            LLMRequest(
                trigger,
                agent_id,
                day,
                minute_of_day,
                payload,
                request_id,
                payload.get("correlation_id"),
                payload.get("session_id"),
                payload.get("turn_id"),
                payload.get("deadline"),
            )
        )
        self._issued[key] = self._issued.get(key, 0) + 1
        self.metrics["issued"] += 1
        return True

    def drain_requests(self) -> tuple[LLMRequest, ...]:
        requests = tuple(self._pending)
        self._pending.clear()
        return requests

    def process(
        self,
        request: LLMRequest,
        provider: Callable[[LLMRequest], dict[str, Any]],
    ) -> dict[str, Any]:
        key = (request.day, request.agent_id)
        if request.request_id and request.request_id in self._completed:
            return dict(self._completed[request.request_id])
        if request.deadline is not None and request.minute_of_day > request.deadline:
            self.metrics["rejected"] += 1
            raise RuntimeError("LLM request deadline exceeded")
        if self._calls.get(key, 0) >= self.daily_budget:
            raise RuntimeError("LLM daily budget exhausted")
        response = provider(request)
        self._calls[key] = self._calls.get(key, 0) + 1
        self.metrics["calls"] += 1
        if request.request_id:
            self._completed[request.request_id] = dict(response)
        return response

    def calls_for(self, day: int, agent_id: str) -> int:
        return self._calls.get((day, agent_id), 0)
