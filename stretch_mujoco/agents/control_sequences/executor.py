"""Tick-driven, receipt-gated interpreter for compiled control sequences."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any, Callable

from stretch_mujoco.agents.actions import (
    ActionCommand,
    ActionType,
    ConversationRequest,
    ExecutionStatus,
    RobotTaskStatus,
    RobotTaskType,
)
from stretch_mujoco.agents.conversation import (
    ConversationPhase,
    DialogueAct,
    DialogueCandidate,
    TurnStatus,
)
from .audit import SequenceAudit
from .models import (
    CompiledSequence,
    CompiledStep,
    ControlSequenceError,
    StepExecution,
    StepKind,
    StepStatus,
)
from .sources import PlanSource


@dataclass(frozen=True)
class SequenceResult:
    sequence_id: str
    status: StepStatus
    executions: tuple[StepExecution, ...]


class SequenceExecutor:
    """Coordinate public runtime APIs; completion always follows their receipts."""

    def __init__(
        self,
        compiled: CompiledSequence,
        runtime: Any,
        *,
        audit: SequenceAudit | None = None,
        predicate: Callable[[dict[str, Any]], bool] | None = None,
        plan_source: PlanSource | None = None,
        segment_compiler: Callable[[tuple[Any, ...]], tuple[CompiledStep, ...]] | None = None,
        observation: Callable[[], Any] | None = None,
        max_segments: int | None = None,
    ) -> None:
        if plan_source is not None and segment_compiler is None:
            raise ValueError("a plan_source requires a segment_compiler")
        self.compiled, self.runtime, self.audit = compiled, runtime, audit or SequenceAudit()
        self.predicate, self.plan_source, self.segment_compiler = (
            predicate or (lambda _: False),
            plan_source,
            segment_compiler,
        )
        self.observation, self.max_segments = observation or (lambda: None), max_segments
        self._steps, self._segments_issued, self._index = list(compiled.steps), 0, 0
        self._active: StepExecution | None = None
        self._active_step: CompiledStep | None = None
        self._executions: list[StepExecution] = []
        self._status = StepStatus.PENDING
        self._state: dict[str, Any] = {}
        self._bindings: dict[str, dict[str, Any]] = {}
        self.audit.record(
            "sequence_started",
            sequence_id=compiled.sequence.sequence_id,
            control_mode=compiled.control_mode.value,
            sequence_sha256=compiled.sequence.source_sha256,
            compiled_plan_sha256=compiled.plan_sha256,
        )

    @property
    def status(self) -> StepStatus:
        return self._status

    @property
    def executions(self) -> tuple[StepExecution, ...]:
        return tuple(self._executions)

    def result(self) -> SequenceResult:
        return SequenceResult(self.compiled.sequence.sequence_id, self._status, self.executions)

    def tick(self, now: float) -> SequenceResult:
        if self._status in {
            StepStatus.SUCCEEDED,
            StepStatus.FAILED,
            StepStatus.CANCELLED,
            StepStatus.TIMED_OUT,
        }:
            return self.result()
        if self._status is StepStatus.PENDING:
            self._status = StepStatus.RUNNING
        if self._active is None:
            if self._index >= len(self._steps) and not self._request_segment():
                self._status = StepStatus.SUCCEEDED
                self.audit.record(
                    "sequence_succeeded", sequence_id=self.compiled.sequence.sequence_id
                )
                return self.result()
            self._start(self._steps[self._index], now)
        if self._active is None:
            return self.result()
        assert self._active_step is not None and self._active.started_at is not None
        if now - self._active.started_at > self._active_step.step.timeout_seconds:
            self._finish(False, now, "step timed out", StepStatus.TIMED_OUT)
        else:
            done, error = self._advance(self._active_step, now)
            if error:
                self._finish(False, now, error)
            elif done:
                self._finish(True, now)
        return self.result()

    def cancel(self, now: float, reason: str = "sequence cancelled") -> SequenceResult:
        if self._active:
            self._finish(False, now, reason, StepStatus.CANCELLED)
        self._status = StepStatus.CANCELLED
        self.audit.record("sequence_cancelled", reason=reason)
        return self.result()

    def bind_llm_result(self, step_id: str, value: dict[str, Any]) -> None:
        if self._active is None or self._active.step_id != step_id or self._active_step is None:
            raise ValueError("LLM result does not belong to active step")
        self._bindings[self._active_step.step.payload["save_as"]] = dict(value)
        self.audit.record("llm_response_accepted", step_id=step_id)

    def bind_conversation_turn(self, step_id: str, turn_id: str, value: dict[str, Any]) -> None:
        if self._active is None or self._active.step_id != step_id:
            raise ValueError("Dialogue result does not belong to active step")
        self._state.setdefault("dialogue", {})[turn_id] = dict(value)
        self.audit.record("dialogue_response_accepted", step_id=step_id, turn_id=turn_id)

    def _start(self, step: CompiledStep, now: float) -> None:
        self._active = StepExecution(
            step.step.step_id, f"corr_{uuid.uuid4().hex}", status=StepStatus.RUNNING, started_at=now
        )
        self._active_step, self._state = step, {}
        self._executions.append(self._active)
        self.audit.record(
            "step_started", step_id=self._active.step_id, correlation_id=self._active.correlation_id
        )
        if step.step.kind is StepKind.ACTION:
            self._fail_start(self._start_action(step, self._state), now)
        elif step.step.kind is StepKind.PARALLEL:
            self._state["children"] = [
                {"step": x, "state": {}, "done": False} for x in step.children
            ]
            for child in self._state["children"]:
                error = self._start_child(child["step"], child["state"], now)
                if error:
                    self._finish(False, now, error)
                    return
        elif step.step.kind is StepKind.CHOOSE:
            child = next(
                (x for x in step.children if self._condition(x.step.payload.get("when", {}))), None
            )
            if child is None:
                self._finish(False, now, "choose_has_no_matching_option")
                return
            self._state["child"] = {"step": child, "state": {}}
            self._fail_start(self._start_child(child, self._state["child"]["state"], now), now)
        elif step.step.kind is StepKind.REPEAT:
            self._state.update(iteration=0, child_index=0, child_state={})
            self._fail_start(
                self._start_child(step.children[0], self._state["child_state"], now), now
            )
        elif step.step.kind is StepKind.CONVERSATION:
            self._start_conversation(step)
        elif step.step.kind is StepKind.ANIMATION:
            submit = getattr(self.runtime, "submit_animation", None)
            if not callable(submit):
                self._finish(False, now, "runtime_animation_receipt_api_required")
            else:
                value = submit(
                    step.actor_id,
                    step.step.payload["clip"],
                    step.step.payload.get("completion_marker"),
                )
                self._fail_start(None if value.valid else "; ".join(value.errors), now)
                if value.valid:
                    self._state["action_id"] = self.runtime.agents[
                        step.actor_id
                    ].executor.execution_id

    def _fail_start(self, error: str | None, now: float) -> None:
        if error and self._active:
            self._finish(False, now, error)

    def _start_action(self, step: CompiledStep, state: dict[str, Any]) -> str | None:
        assert step.actor_id
        value = self.runtime.submit_action(
            ActionCommand(
                step.actor_id,
                ActionType(step.step.payload["action"]),
                self._substitute(step.target_id),
                self._substitute(step.step.payload.get("parameters", {})),
            )
        )
        if not value.valid:
            return "; ".join(value.errors)
        execution = self.runtime.agents[step.actor_id].executor
        state["action_id"] = execution.execution_id
        session_id = (
            execution.command.parameters.get("_desk_work_session_id") if execution.command else None
        )
        if isinstance(session_id, str):
            state["desk_work_session_id"] = session_id
        return None

    def _start_child(self, step: CompiledStep, state: dict[str, Any], now: float) -> str | None:
        if step.step.kind is StepKind.ACTION:
            return self._start_action(step, state)
        if step.step.kind is StepKind.WAIT:
            state["started"] = now
            return None
        if step.step.kind is StepKind.ASSERT:
            return None
        return f"composite child kind '{step.step.kind.value}' is not supported in v1"

    def _advance(self, step: CompiledStep, now: float) -> tuple[bool, str | None]:
        if step.step.kind is StepKind.WAIT:
            return now - self._active.started_at >= step.step.payload["duration_s"], None
        if step.step.kind is StepKind.ASSERT:
            return self._condition(step.step.payload["condition"]), None
        if step.step.kind is StepKind.LLM_REQUEST:
            return step.step.payload["save_as"] in self._bindings, None
        if step.step.kind in {StepKind.ACTION, StepKind.ANIMATION}:
            return self._action_done(step, self._state)
        if step.step.kind is StepKind.PARALLEL:
            return self._parallel(step, now)
        if step.step.kind is StepKind.CHOOSE:
            child = self._state["child"]
            return self._advance_child(child["step"], child["state"], now)
        if step.step.kind is StepKind.REPEAT:
            return self._repeat(step, now)
        if step.step.kind is StepKind.CONVERSATION:
            return self._conversation(step, now)
        return False, f"unsupported step kind '{step.step.kind.value}'"

    def _advance_child(
        self, step: CompiledStep, state: dict[str, Any], now: float
    ) -> tuple[bool, str | None]:
        if step.step.kind is StepKind.ACTION:
            return self._action_done(step, state)
        if step.step.kind is StepKind.WAIT:
            return now - state["started"] >= step.step.payload["duration_s"], None
        if step.step.kind is StepKind.ASSERT:
            return self._condition(step.step.payload["condition"]), None
        return False, f"composite child kind '{step.step.kind.value}' is not supported in v1"

    def _parallel(self, step: CompiledStep, now: float) -> tuple[bool, str | None]:
        complete = 0
        for child in self._state["children"]:
            if child["done"]:
                complete += 1
                continue
            done, error = self._advance_child(child["step"], child["state"], now)
            if error:
                return False, error
            child["done"] = done
            complete += int(done)
        return (
            complete > 0
            if step.step.payload.get("join", "all") == "any"
            else complete == len(self._state["children"])
        ), None

    def _repeat(self, step: CompiledStep, now: float) -> tuple[bool, str | None]:
        if self._condition(step.step.payload.get("until", {})):
            return True, None
        child = step.children[self._state["child_index"]]
        done, error = self._advance_child(child, self._state["child_state"], now)
        if error or not done:
            return done, error
        self._state["child_index"] += 1
        if self._state["child_index"] == len(step.children):
            self._state["iteration"] += 1
            self._state["child_index"] = 0
            if self._state["iteration"] >= step.step.payload["max_iterations"]:
                return False, "repeat_limit_reached"
        self._state["child_state"] = {}
        return False, self._start_child(
            step.children[self._state["child_index"]], self._state["child_state"], now
        )

    def _start_conversation(self, step: CompiledStep) -> None:
        aliases = self.compiled.sequence.participants
        parts = tuple(aliases.get(x, x) for x in step.step.payload["participants"])
        session_id = f"{self._active.correlation_id}:{step.step.step_id}"
        snapshot = self.observation()
        receipt = self.runtime.begin_conversation(
            ConversationRequest(
                session_id,
                parts,
                step.step.payload["topic"],
                timeout=step.step.timeout_seconds,
                max_turns=step.step.payload["max_turns"],
                correlation_id=self._active.correlation_id,
                semantic_snapshot=snapshot if isinstance(snapshot, dict) else None,
            )
        )
        self._state.update(session_id=session_id, turn_index=0, submitted=False)
        if not receipt.accepted:
            self._state["error"] = receipt.error or receipt.status

    def _conversation(self, step: CompiledStep, now: float) -> tuple[bool, str | None]:
        if self._state.get("error"):
            return False, self._state["error"]
        session = self.runtime.conversation(self._state["session_id"])
        if session.status.terminal:
            return session.status.value == "completed", (
                None
                if session.status.value == "completed"
                else session.error or session.status.value
            )
        turns = step.step.payload.get("turns", [])
        index = self._state["turn_index"]
        if index >= len(turns):
            return False, None
        turn = turns[index]
        if not self._state["submitted"]:
            # A server-backed conversation may still be moving both NPCs to
            # its rendezvous sites.  Do not turn that ordinary physical
            # preparation into a terminal sequence failure; the runtime will
            # expose WAITING_FOR_TURN only after the alignment receipts pass.
            if session.phase is not ConversationPhase.WAITING_FOR_TURN:
                return False, None
            data = self._state.get("dialogue", {}).get(turn["id"])
            if turn.get("source") == "llm" and data is None:
                return False, "dialogue_llm_response_required"
            text = data.get("text") if data else turn.get("text")
            if not isinstance(text, str) or not text:
                return False, "dialogue_text_required"
            aliases = self.compiled.sequence.participants
            value = self.runtime.submit_dialogue_candidate(
                DialogueCandidate(
                    f"{self._active.correlation_id}:{turn['id']}",
                    session.session_id,
                    turn["id"],
                    aliases.get(turn["speaker"], turn["speaker"]),
                    aliases.get(turn["listener"], turn["listener"]),
                    DialogueAct(
                        data.get("act", "acknowledge") if data else turn.get("act", "acknowledge")
                    ),
                    text,
                    now,
                )
            )
            if not value.valid:
                return False, "; ".join(value.errors)
            self._state["submitted"] = True
            return False, None
        dialogue = session.dialogue_turns.get(turn["id"])
        if dialogue and dialogue.status is TurnStatus.COMMITTED:
            self._state["turn_index"] += 1
            self._state["submitted"] = False
        elif dialogue and dialogue.status in {TurnStatus.REJECTED, TurnStatus.FAILED}:
            return False, dialogue.error or "dialogue_turn_failed"
        return False, None

    def _action_done(self, step: CompiledStep, state: dict[str, Any]) -> tuple[bool, str | None]:
        assert step.actor_id
        session_id = state.get("desk_work_session_id")
        status_for = getattr(self.runtime, "desk_work_session_status", None)
        if isinstance(session_id, str) and callable(status_for):
            status, error = status_for(session_id)
            if status is ExecutionStatus.SUCCEEDED:
                return True, None
            if status in {
                ExecutionStatus.FAILED,
                ExecutionStatus.CANCELLED,
                ExecutionStatus.TIMED_OUT,
            }:
                return False, error or "desk_work_session_failed"
            return False, None
        execution = self.runtime.agents[step.actor_id].executor
        if execution.execution_id != state.get("action_id"):
            return False, None
        if execution.status is ExecutionStatus.SUCCEEDED:
            if step.step.kind is StepKind.ACTION and step.step.payload["action"] == "request_robot":
                task = state.get("robot_task")
                if task is None:
                    task = next(
                        (
                            candidate
                            for candidate in self.runtime.robot_tasks.values()
                            if candidate.requester == step.actor_id
                            and candidate.robot_id == self._substitute(step.target_id)
                            and candidate.object_id
                            == self._substitute(step.step.payload.get("parameters", {})).get(
                                "object"
                            )
                        ),
                        None,
                    )
                    if task is None:
                        return False, "robot_task_creation_receipt_missing"
                    state["robot_task"] = task
                if task.status is RobotTaskStatus.FAILED:
                    return False, task.error or "robot_task_failed"
                # Creating a task is only a logical request.  A terminal robot
                # receipt must include the completed task and physical evidence.
                if task.status is not RobotTaskStatus.SUCCEEDED:
                    return False, None
                if not task.receipt_ids:
                    return False, "robot_terminal_physical_receipt_missing"
                if task.task_type is RobotTaskType.ROBOT_TO_NPC_HANDOVER:
                    required_suffixes = (
                        ":robot_release_confirmed",
                        ":npc_attachment_confirmed",
                        ":interaction_completed",
                    )
                    if not all(
                        any(receipt.endswith(suffix) for receipt in task.receipt_ids)
                        for suffix in required_suffixes
                    ):
                        return False, "robot_handover_receipts_missing"
            if state is self._state:
                receipt_ids = [execution.execution_id]
                if (
                    step.step.kind is StepKind.ACTION
                    and step.step.payload["action"] == "request_robot"
                ):
                    receipt_ids.extend(sorted(state["robot_task"].receipt_ids))
                self._active.receipt_ids = tuple(receipt_ids)
            return True, None
        if execution.status in {
            ExecutionStatus.FAILED,
            ExecutionStatus.CANCELLED,
            ExecutionStatus.TIMED_OUT,
        }:
            return False, execution.error or execution.status.value
        return False, None

    def _condition(self, value: dict[str, Any]) -> bool:
        return bool(self.predicate(self._substitute(dict(value))))

    def _finish(
        self,
        success: bool,
        now: float,
        error: str | None = None,
        status: StepStatus = StepStatus.FAILED,
    ) -> None:
        assert self._active
        self._active.status = StepStatus.SUCCEEDED if success else status
        self._active.finished_at = now
        self._active.error = error
        self.audit.record(
            "step_succeeded" if success else "step_failed",
            step_id=self._active.step_id,
            correlation_id=self._active.correlation_id,
            error=error,
            receipt_ids=list(self._active.receipt_ids),
        )
        if success:
            self._index += 1
            self._active = None
            self._active_step = None
            self._state = {}
        else:
            self._status = self._active.status

    def _substitute(self, value: Any) -> Any:
        if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
            binding, field = value[2:-1].split(".", 1)
            return self._bindings[binding][field]
        if isinstance(value, dict):
            return {k: self._substitute(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self._substitute(v) for v in value]
        return value

    def _request_segment(self) -> bool:
        if self.plan_source is None:
            return False
        if self.max_segments is not None and self._segments_issued >= self.max_segments:
            self.audit.record("plan_source_exhausted", reason="segment_limit")
            return False
        segment = self.plan_source.next_segment(self.observation(), self.result())
        if segment is None:
            self.audit.record("plan_source_exhausted", reason="source_exhausted")
            return False
        assert self.segment_compiler
        compiled = self.segment_compiler(segment.steps)
        if not compiled:
            raise ControlSequenceError("plan source produced an empty segment")
        self._steps.extend(compiled)
        self._segments_issued += 1
        self.audit.record(
            "plan_segment_compiled",
            segment_id=segment.segment_id,
            source=segment.source,
            step_ids=[x.step.step_id for x in compiled],
        )
        return True
