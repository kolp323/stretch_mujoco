"""Deterministic in-process robot executor for office-agent integration demos."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .actions import RobotTaskStatus

if TYPE_CHECKING:
    from .runtime import OfficeAgentRuntime


@dataclass
class MockRobotExecutor:
    """Complete pending robot tasks after a fixed simulated delay.

    This is deliberately an adapter around ``OfficeAgentRuntime.complete_robot_task``:
    the runtime remains the sole owner of task state, reservations, and semantic
    relations. A real robot bridge can implement the same polling contract later.
    """

    completion_delay_minutes: float = 1.0
    failed_task_ids: set[str] = field(default_factory=set)
    drop_receipt_task_ids: set[str] = field(default_factory=set)
    duplicate_receipt_task_ids: set[str] = field(default_factory=set)
    handover_ready_task_ids: set[str] = field(default_factory=set)
    _remaining_minutes: dict[str, float] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        if self.completion_delay_minutes <= 0:
            raise ValueError("completion_delay_minutes must be positive")

    def tick(self, runtime: "OfficeAgentRuntime", elapsed_minutes: float) -> tuple[str, ...]:
        """Advance outstanding tasks and return IDs completed during this tick."""
        if elapsed_minutes < 0:
            raise ValueError("elapsed_minutes cannot be negative")

        pending = {task.task_id: task for task in runtime.pending_robot_tasks()}
        for task_id in pending:
            self._remaining_minutes.setdefault(task_id, self.completion_delay_minutes)
        for task_id in set(self._remaining_minutes) - set(pending):
            del self._remaining_minutes[task_id]

        completed: list[str] = []
        for task_id, task in pending.items():
            if task.status == RobotTaskStatus.PENDING:
                task.status = RobotTaskStatus.RUNNING
            self._remaining_minutes[task_id] -= elapsed_minutes
            if self._remaining_minutes[task_id] <= 0:
                if task_id in self.drop_receipt_task_ids:
                    continue
                task_result = runtime.complete_robot_task(
                    task_id,
                    success=task_id not in self.failed_task_ids,
                    error=(
                        "Mock robot reported a delivery failure"
                        if task_id in self.failed_task_ids
                        else None
                    ),
                )
                if task_id in self.duplicate_receipt_task_ids:
                    runtime.complete_robot_task(
                        task_id, success=task_result.status == RobotTaskStatus.SUCCEEDED
                    )
                if (
                    task_id in self.handover_ready_task_ids
                    and task_result.status == RobotTaskStatus.SUCCEEDED
                    and task_result.conversation_id is not None
                ):
                    runtime.record_robot_handover_receipt(
                        task_result.conversation_id,
                        task_id,
                        f"{task_id}:handover",
                        robot_release_confirmed=True,
                        npc_attachment_confirmed=True,
                        interaction_confirmed=True,
                    )
                completed.append(task_id)
        return tuple(completed)
