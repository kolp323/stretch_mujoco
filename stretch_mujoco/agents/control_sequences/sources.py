"""The two plan sources share one PlanSegment protocol and no runtime write access."""

from __future__ import annotations

from typing import Any, Callable, Protocol

from .contracts import validate_llm_contract
from .models import ControlSequenceError, FailureHandling, PlanSegment, SequenceStep, StepKind


class PlanSource(Protocol):
    def next_segment(self, observation: Any, prior_result: Any | None) -> PlanSegment | None: ...


class YamlPlanSource:
    def __init__(self, steps: tuple[SequenceStep, ...]) -> None:
        self._steps = steps
        self._issued = False

    def next_segment(self, observation: Any, prior_result: Any | None) -> PlanSegment | None:
        if self._issued:
            return None
        self._issued = True
        return PlanSegment("yaml_0001", self._steps, "yaml")


class LlmPlanSource:
    """Turn a validated short action proposal into the same structured segment as YAML."""

    def __init__(
        self,
        proposal_provider: Callable[[Any], dict[str, Any] | None],
        *,
        allowed_actions: frozenset[str],
        allowed_locations: frozenset[str],
        horizon: int,
    ) -> None:
        self._provider = proposal_provider
        self._actions = allowed_actions
        self._locations = allowed_locations
        self._horizon = horizon
        self._counter = 0

    def next_segment(self, observation: Any, prior_result: Any | None) -> PlanSegment | None:
        raw_response = self._provider(observation)
        if raw_response is None:
            return None
        response = validate_llm_contract("action_proposal_v1", raw_response)
        proposals = response["actions"]
        if len(proposals) > self._horizon:
            raise ControlSequenceError("LLM proposal exceeds configured plan horizon")
        steps: list[SequenceStep] = []

        def append_actions(items: list[dict[str, Any]], prefix: str) -> None:
            for index, proposal in enumerate(items):
                if (
                    proposal["action"] not in self._actions
                    or proposal["target"] not in self._locations
                ):
                    raise ControlSequenceError("LLM proposal is outside the configured allowlist")
                steps.append(
                    SequenceStep(
                        step_id=f"llm_{self._counter}_{prefix}_{index}",
                        kind=StepKind.ACTION,
                        timeout_seconds=30.0,
                        failure=FailureHandling(),
                        payload={
                            "id": f"llm_{self._counter}_{prefix}_{index}",
                            "kind": "action",
                            "actor": proposal["actor"],
                            "action": proposal["action"],
                            "target": proposal["target"],
                            "parameters": dict(proposal.get("parameters", {})),
                        },
                    )
                )

        append_actions(proposals, "action")
        for conversation_index, conversation in enumerate(response.get("conversations", [])):
            turns = []
            for turn_index, turn in enumerate(conversation["turns"]):
                turns.append(
                    {
                        "id": f"llm_{self._counter}_dialogue_{conversation_index}_{turn_index}",
                        "speaker": turn["speaker"],
                        "listener": turn["listener"],
                        "act": turn["act"],
                        "text": turn["text"],
                        # The text is replayed LLM output, already locally validated.
                        "source": "replay",
                    }
                )
            steps.append(
                SequenceStep(
                    step_id=f"llm_{self._counter}_dialogue_{conversation_index}",
                    kind=StepKind.CONVERSATION,
                    timeout_seconds=45.0,
                    failure=FailureHandling(),
                    payload={
                        "id": f"llm_{self._counter}_dialogue_{conversation_index}",
                        "kind": "conversation",
                        "participants": list(conversation["participants"]),
                        "topic": conversation["topic"],
                        "max_turns": len(turns),
                        "turns": turns,
                    },
                )
            )
        append_actions(response.get("post_actions", []), "post_action")
        self._counter += 1
        return PlanSegment(response["proposal_id"], tuple(steps), "llm")
