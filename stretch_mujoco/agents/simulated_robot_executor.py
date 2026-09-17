"""MuJoCo-backed simulated robot executor for handover demos and tests."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import ClassVar

from .actions import RobotTaskStatus, RobotTaskType
from .robot_handover import RobotToNpcHandoverBridge
from .robot_task_driver import RobotTaskReceipt

@dataclass
class SimulatedRobotExecutor:
    """Drive a handover through NPC transport receipts and MuJoCo observations.

    This executor deliberately has no logical timer-only success path: terminal
    success requires the bridge's release/attachment receipts and captures the
    current simulator evidence in the task receipt.
    """
    npc_transport: object
    handover_sites: dict[str, str] = field(default_factory=dict)
    handover_yaws: dict[str, float] = field(default_factory=dict)
    timeout_seconds: float = 180.0
    supported_task_types: ClassVar[frozenset[RobotTaskType]] = frozenset({RobotTaskType.ROBOT_TO_NPC_HANDOVER})
    receipts: dict[str, RobotTaskReceipt] = field(default_factory=dict, init=False)
    _bridges: dict[str, RobotToNpcHandoverBridge] = field(default_factory=dict, init=False)
    _stages: dict[str, str] = field(default_factory=dict, init=False)

    def tick(self, runtime, elapsed_seconds: float) -> tuple[str, ...]:
        if elapsed_seconds < 0: raise ValueError("elapsed_seconds cannot be negative")
        completed: list[str] = []
        for task in runtime.pending_robot_tasks():
            if task.task_type is not RobotTaskType.ROBOT_TO_NPC_HANDOVER:
                self._finish(runtime, task, False, "simulated_executor_only_supports_handover")
                completed.append(task.task_id); continue
            if task.recipient is None:
                self._finish(runtime, task, False, "handover_recipient_missing")
                completed.append(task.task_id); continue
            bridge = self._bridges.get(task.task_id)
            if bridge is None:
                bridge = RobotToNpcHandoverBridge(
                    self.npc_transport,
                    handover_sites={task.robot_id: self.handover_sites[task.robot_id]},
                    interaction_yaws={task.robot_id: self.handover_yaws.get(task.robot_id, 0.0)},
                    timeout_seconds=self.timeout_seconds,
                )
                bridge.begin_receive(task.robot_id, task.recipient, task.object_id, session_id=task.task_id)
                self._bridges[task.task_id] = bridge
                self._stages[task.task_id] = "robot_navigate_to_object"
                task.status = RobotTaskStatus.RUNNING
                continue
            session = bridge.poll(task.task_id)
            self._stages[task.task_id] = session.phase
            if session.phase == "released":
                bridge.confirm_robot_release(task.task_id)
                task.receipt_ids.add(f"sim:{task.task_id}:release_confirmed")
                self._stages[task.task_id] = "release_confirmed"
                continue
            if session.status.terminal:
                success = session.status.value == "succeeded"
                if success:
                    task.robot_release_confirmed = True
                    task.npc_attachment_confirmed = True
                    task.interaction_confirmed = True
                    task.receipt_ids.add(f"sim:{task.task_id}:npc_attachment_confirmed")
                    task.receipt_ids.add(f"sim:{task.task_id}:release_confirmed")
                    task.receipt_ids.add(f"sim:{task.task_id}:interaction_completed")
                self._finish(runtime, task, success, None if success else (session.error or "handover_failed"), evidence=self._evidence(task))
                completed.append(task.task_id)
        return tuple(completed)

    def _evidence(self, task) -> dict[str, object]:
        transport = self.npc_transport
        model, data = getattr(transport, "model", None), getattr(transport, "data", None)
        evidence: dict[str, object] = {"executor_kind": "simulated_mujoco", "stage": self._stages.get(task.task_id, "unknown")}
        if model is not None and data is not None:
            evidence["sim_time"] = float(getattr(data, "time", 0.0))
            evidence["ncon"] = int(getattr(data, "ncon", 0))
        return evidence

    def _finish(self, runtime, task, success: bool, error: str | None, *, evidence: dict[str, object] | None = None) -> None:
        receipt_id = f"sim:{task.task_id}:{'succeeded' if success else 'failed'}"
        receipt = RobotTaskReceipt(receipt_id, task.task_id, "succeeded" if success else "failed", self._stages.get(task.task_id, "terminal"), evidence or {}, error)
        self.receipts[receipt_id] = receipt; task.receipt_ids.add(receipt_id)
        runtime.complete_robot_task(task.task_id, success=success, error=error)

    def cancel(self, task_id: str, reason: str = "robot_task_cancelled") -> RobotTaskReceipt:
        receipt = RobotTaskReceipt(f"sim:{task_id}:cancelled", task_id, "cancelled", "cancelled", {"executor_kind": "simulated_mujoco"}, reason)
        self.receipts[receipt.receipt_id] = receipt
        return receipt
