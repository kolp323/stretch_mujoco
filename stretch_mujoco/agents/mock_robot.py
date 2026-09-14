"""Deterministic in-process robot executor for office-agent integration demos."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, ClassVar

from .actions import RobotTaskStatus, RobotTaskType
from .action_recipes import OFFICE_ROBOT_REQUEST_SITES
from .robot_handover import RobotToNpcHandoverBridge
from .robot_task_driver import RobotTaskReceipt
from stretch_mujoco.semantics import RelationType

if TYPE_CHECKING:
    from .runtime import OfficeAgentRuntime


@dataclass
class MockRobotExecutor:
    """Non-production adapter for deterministic tests and NPC handover demos.

    This is deliberately an adapter around ``OfficeAgentRuntime.complete_robot_task``:
    the runtime remains the sole owner of task state, reservations, and semantic
    relations. A real robot integration implements ``RobotTaskExecutor``.
    """

    completion_delay_minutes: float = 1.0
    supported_task_types: ClassVar[frozenset[RobotTaskType]] = frozenset(RobotTaskType)
    failed_task_ids: set[str] = field(default_factory=set)
    drop_receipt_task_ids: set[str] = field(default_factory=set)
    duplicate_receipt_task_ids: set[str] = field(default_factory=set)
    handover_ready_task_ids: set[str] = field(default_factory=set)
    # Keep the completion notification visible long enough for the renderer's
    # observation cadence to capture it before the receiver starts walking.
    notification_delay_minutes: float = 1.0
    npc_transport: object | None = None
    receipt_source: object | None = None
    handover_sites: dict[str, str] | None = None
    handover_yaws: dict[str, float] | None = None
    _remaining_minutes: dict[str, float] = field(default_factory=dict, init=False)
    _handovers: dict[str, RobotToNpcHandoverBridge] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        if self.completion_delay_minutes <= 0:
            raise ValueError("completion_delay_minutes must be positive")
        if self.notification_delay_minutes < 0:
            raise ValueError("notification_delay_minutes cannot be negative")
        self.handover_sites = dict(self.handover_sites or {})
        self.handover_yaws = dict(self.handover_yaws or {})

    def tick(self, runtime: "OfficeAgentRuntime", elapsed_seconds: float) -> tuple[str, ...]:
        """Advance outstanding tasks and return IDs completed during this tick."""
        if elapsed_seconds < 0:
            raise ValueError("elapsed_seconds cannot be negative")
        elapsed_minutes = elapsed_seconds * runtime.minutes_per_second

        pending = {task.task_id: task for task in runtime.pending_robot_tasks()}
        for task_id in pending:
            self._remaining_minutes.setdefault(task_id, self.completion_delay_minutes)
        for task_id in set(self._remaining_minutes) - set(pending):
            del self._remaining_minutes[task_id]
            self._handovers.pop(task_id, None)

        completed: list[str] = []
        for task_id, task in pending.items():
            if task.status == RobotTaskStatus.PENDING:
                task.status = RobotTaskStatus.RUNNING
            self._remaining_minutes[task_id] -= elapsed_minutes
            if self._remaining_minutes[task_id] <= 0:
                if task_id in self.drop_receipt_task_ids:
                    continue
                success = task_id not in self.failed_task_ids
                if success and task.task_type is RobotTaskType.ROBOT_TO_NPC_HANDOVER:
                    if self.npc_transport is None:
                        task.error = "robot_to_npc_handover_transport_unsupported"
                        task.receipt_ids.add(f"mock:{task_id}:handover_failed")
                        runtime.complete_robot_task(task_id, success=False, error=task.error)
                        completed.append(task_id)
                        continue
                    bridge = self._handovers.get(task_id)
                    if bridge is None:
                        bridge = RobotToNpcHandoverBridge(
                            self.npc_transport,
                            handover_sites={task.robot_id: self.handover_sites.get(
                                task.robot_id, OFFICE_ROBOT_REQUEST_SITES[task.robot_id]
                            )},
                            interaction_yaws={task.robot_id: self.handover_yaws.get(task.robot_id, 0.0)},
                        )
                        self._handovers[task_id] = bridge
                        bridge.begin_receive(task.robot_id, task.recipient or "", task.object_id, session_id=task_id)
                        continue
                    session = bridge.poll(task_id)
                    if session.status.terminal and session.status.value != "succeeded":
                        task.error = session.error or f"npc_receive_{session.status.value}"
                        task.receipt_ids.add(f"mock:{task_id}:handover_failed")
                        runtime.complete_robot_task(task_id, success=False, error=task.error)
                        completed.append(task_id)
                        continue
                    if session.phase == "released":
                        # This namespaced mock receipt is the explicit release evidence;
                        # it is not inferred from elapsed logical delivery time.
                        task.receipt_ids.add(f"mock:{task_id}:robot_release_confirmed")
                        bridge.confirm_robot_release(task_id)
                        continue
                    if not session.status.terminal:
                        continue
                    receiver = runtime.agents[task.recipient or ""]
                    receiver.state.held_object = task.object_id
                    for relation in runtime.world.find_relations(subject=task.object_id):
                        if relation.relation in {RelationType.INSIDE, RelationType.ON}:
                            runtime.world.remove_relation(
                                task.object_id, relation.relation, relation.object
                            )
                    runtime.world.add_relation(receiver.agent_id, RelationType.HOLDS, task.object_id)
                    runtime.world.object(task.object_id).attributes.pop("location", None)
                    task.receipt_ids.update({
                        f"mock:{task_id}:npc_attachment_confirmed",
                        f"mock:{task_id}:interaction_completed",
                    })
                    task.robot_release_confirmed = True
                    task.npc_attachment_confirmed = True
                    task.interaction_confirmed = True
                # A mock completion still has to satisfy the same terminal
                # receipt contract consumed by SequenceExecutor.  The mock
                # does not fabricate physical evidence; it only supplies a
                # clearly namespaced terminal receipt for its configured
                # logical delay.
                if task.task_type is RobotTaskType.PLACE_DELIVERY:
                    task.receipt_ids.add(f"mock:{task_id}:{'delivery_verified' if success else 'delivery_failed'}")
                task_result = runtime.complete_robot_task(
                    task_id,
                    success=success,
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

    def cancel(self, task_id: str, reason: str = "robot_task_cancelled") -> RobotTaskReceipt:
        """Return an explicit cancellation receipt for the owning runtime to commit."""
        return RobotTaskReceipt(
            f"mock:{task_id}:cancelled", task_id, "cancelled", "cancelled", error=reason
        )
