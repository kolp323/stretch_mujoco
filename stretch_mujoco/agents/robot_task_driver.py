"""Public boundary for externally supplied robot task executors.

This package owns NPC requests, task classification, and runtime receipt
validation. Navigation, grasping, IK, and physical placement are owned by a
production robot integration outside this repository.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from .actions import RobotTaskType

if TYPE_CHECKING:
    from .runtime import OfficeAgentRuntime


@dataclass(frozen=True)
class RobotTaskReceipt:
    """A stable terminal receipt emitted by an external robot executor."""

    receipt_id: str
    task_id: str
    status: str
    phase: str
    evidence: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


@runtime_checkable
class RobotTaskExecutor(Protocol):
    """Minimal lifecycle contract implemented by production robot adapters."""

    supported_task_types: frozenset[RobotTaskType]

    def tick(
        self, runtime: "OfficeAgentRuntime", elapsed_seconds: float
    ) -> tuple[str, ...]: ...

    def cancel(self, task_id: str, reason: str = "robot_task_cancelled") -> RobotTaskReceipt | None: ...


class UnsupportedRobotTaskExecutor:
    """Compatibility adapter that fails tasks rather than simulating hardware.

    This can be used for explicit negative-path tests or deployments that have
    not installed a production ``RobotTaskExecutor`` yet.
    """

    supported_task_types: frozenset[RobotTaskType] = frozenset()

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        self.receipts: dict[str, RobotTaskReceipt] = {}

    def tick(self, runtime: "OfficeAgentRuntime", _elapsed_seconds: float) -> tuple[str, ...]:
        completed: list[str] = []
        for task in runtime.pending_robot_tasks():
            receipt = RobotTaskReceipt(
                f"robot:{task.task_id}:unsupported",
                task.task_id,
                "failed",
                "unsupported",
                error="robot_task_executor_not_configured",
            )
            self.receipts[receipt.receipt_id] = receipt
            task.receipt_ids.add(receipt.receipt_id)
            runtime.complete_robot_task(task.task_id, success=False, error=receipt.error)
            completed.append(task.task_id)
        return tuple(completed)

    def cancel(self, task_id: str, reason: str = "robot_task_cancelled") -> RobotTaskReceipt:
        receipt = RobotTaskReceipt(
            f"robot:{task_id}:cancelled", task_id, "cancelled", "cancelled", error=reason
        )
        self.receipts[receipt.receipt_id] = receipt
        return receipt
