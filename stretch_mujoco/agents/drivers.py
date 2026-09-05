"""Narrow execution drivers separating action intent from physical completion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from stretch_mujoco.npc.protocol import CommandStatus, NpcCommand, NpcCommandKind, NpcCommandReceipt

from .actions import ActionExecution, ActionType, ExecutionStatus
from .interactions import InteractionCoordinator


@dataclass(frozen=True)
class DriverResult:
    status: ExecutionStatus
    phase: str
    handle: str | None = None
    error: str | None = None


@dataclass
class _HandoverWorkflow:
    session_id: str
    giver: str
    receiver: str
    object_name: str
    stage: str
    command_id: str


@dataclass
class _SitWorkflow:
    seat: str
    yaw: float
    stage: str
    command_id: str


class ActionDriver(Protocol):
    supported_actions: frozenset[ActionType]

    def start(self, execution: ActionExecution) -> DriverResult:
        ...

    def poll(self, execution: ActionExecution) -> DriverResult:
        ...

    def cancel(self, execution: ActionExecution, reason: str) -> DriverResult:
        ...


class SimulatorStatus(Protocol):
    time: float


class NpcSimulatorClient(Protocol):
    def pull_status(self) -> SimulatorStatus:
        ...

    def submit_npc_command(self, command: NpcCommand) -> str:
        ...

    def pull_npc_receipts(self) -> tuple[NpcCommandReceipt, ...]:
        ...

    def cancel_npc_command(self, npc_id: str, command_id: str) -> None:
        ...


class MujocoNpcActionDriver:
    """Lower embodied actions into commands and wait for physical receipts."""

    supported_actions = frozenset(
        {
            ActionType.MOVE_TO,
            ActionType.SIT,
            ActionType.PICK_UP,
            ActionType.PUT_DOWN,
            ActionType.HANDOVER,
        }
    )

    def __init__(
        self,
        simulator: NpcSimulatorClient,
        location_sites: dict[str, str],
        placement_sites: dict[str, str] | None = None,
        *,
        seat_yaws: dict[str, float] | None = None,
        timeout_seconds: float = 30.0,
    ) -> None:
        self.simulator = simulator
        self.location_sites = dict(location_sites)
        self.placement_sites = dict(placement_sites or {})
        self.seat_yaws = dict(seat_yaws or {})
        self.timeout_seconds = timeout_seconds
        self._sequences: dict[str, int] = {}
        self._receipts: dict[str, NpcCommandReceipt] = {}
        self.interactions = InteractionCoordinator()
        self._handovers: dict[str, _HandoverWorkflow] = {}
        self._sits: dict[str, _SitWorkflow] = {}

    def start(self, execution: ActionExecution) -> DriverResult:
        command = execution.command
        if command is None or command.action not in self.supported_actions:
            return DriverResult(ExecutionStatus.FAILED, "start", error="unsupported_action")
        if command.action == ActionType.HANDOVER:
            return self._start_handover(execution)
        if command.action == ActionType.SIT:
            return self._start_sit(execution)
        status = self.simulator.pull_status()
        issued_at = float(status.time)
        sequence = self._sequences.get(command.agent_id, -1) + 1
        self._sequences[command.agent_id] = sequence
        try:
            kind, payload, phase = self._physical_command(execution)
        except ValueError as error:
            return DriverResult(ExecutionStatus.FAILED, "prepare", error=str(error))
        npc_command = NpcCommand(
            command_id=execution.execution_id,
            sequence=sequence,
            npc_id=command.agent_id,
            kind=kind,
            payload=payload,
            issued_at=issued_at,
            deadline=issued_at + self.timeout_seconds,
        )
        handle = self.simulator.submit_npc_command(npc_command)
        return DriverResult(ExecutionStatus.RUNNING, phase, handle=handle)

    def poll(self, execution: ActionExecution) -> DriverResult:
        for new_receipt in self.simulator.pull_npc_receipts():
            self._receipts[new_receipt.command_id] = new_receipt
        workflow = self._handovers.get(execution.execution_id)
        sit = self._sits.get(execution.execution_id)
        handle = (
            workflow.command_id
            if workflow is not None
            else (sit.command_id if sit is not None else str(execution.driver_handle))
        )
        receipt = self._receipts.get(handle)
        if receipt is None or receipt.status in {CommandStatus.ACCEPTED, CommandStatus.RUNNING}:
            return DriverResult(
                ExecutionStatus.RUNNING,
                (
                    workflow.stage
                    if workflow is not None
                    else (sit.stage if sit is not None else execution.phase)
                ),
                handle,
            )
        if workflow is not None:
            return self._advance_handover(execution, workflow, receipt)
        if sit is not None:
            return self._advance_sit(execution, sit, receipt)
        if receipt.status == CommandStatus.SUCCEEDED:
            return DriverResult(ExecutionStatus.SUCCEEDED, "arrived", execution.driver_handle)
        status = (
            ExecutionStatus.TIMED_OUT
            if receipt.status == CommandStatus.TIMED_OUT
            else (
                ExecutionStatus.CANCELLED
                if receipt.status == CommandStatus.CANCELLED
                else ExecutionStatus.FAILED
            )
        )
        return DriverResult(status, "terminal", execution.driver_handle, receipt.reason)

    def _physical_command(
        self, execution: ActionExecution
    ) -> tuple[NpcCommandKind, dict[str, object], str]:
        assert execution.command is not None
        command = execution.command
        if command.action == ActionType.MOVE_TO:
            site = self.location_sites.get(str(command.target))
            if site is None:
                raise ValueError(f"No NPC site configured for location '{command.target}'")
            return NpcCommandKind.MOVE_TO, {"site": site, "arrival_clip": "idle"}, "navigate"
        if command.action == ActionType.PICK_UP:
            return NpcCommandKind.ATTACH_OBJECT, {"object": str(command.target)}, "attach"
        if command.action == ActionType.PUT_DOWN:
            object_name = command.parameters.get("object")
            if not object_name:
                raise ValueError("Put-down command has no held object")
            site = self.placement_sites.get(str(command.target))
            if site is None:
                raise ValueError(f"No placement site configured for '{command.target}'")
            return (
                NpcCommandKind.DETACH_OBJECT,
                {"object": str(object_name), "site": site},
                "detach",
            )
        raise ValueError(f"Unsupported embodied action '{command.action.value}'")

    def _start_sit(self, execution: ActionExecution) -> DriverResult:
        assert execution.command is not None
        command = execution.command
        seat = str(command.target or "")
        site = self.location_sites.get(seat)
        yaw = self.seat_yaws.get(seat)
        if site is None:
            return DriverResult(
                ExecutionStatus.FAILED, "prepare", error=f"No seat site for '{seat}'"
            )
        if yaw is None:
            return DriverResult(
                ExecutionStatus.FAILED, "prepare", error=f"No seat yaw for '{seat}'"
            )
        handle = self._submit_stage(
            command.agent_id,
            execution.execution_id,
            "approach_seat",
            NpcCommandKind.MOVE_TO,
            {"site": site, "arrival_clip": "idle"},
        )
        self._sits[execution.execution_id] = _SitWorkflow(seat, yaw, "approach_seat", handle)
        return DriverResult(ExecutionStatus.RUNNING, "approach_seat", handle)

    def _advance_sit(
        self,
        execution: ActionExecution,
        workflow: _SitWorkflow,
        receipt: NpcCommandReceipt,
    ) -> DriverResult:
        if receipt.status != CommandStatus.SUCCEEDED:
            self._sits.pop(execution.execution_id, None)
            status = (
                ExecutionStatus.TIMED_OUT
                if receipt.status == CommandStatus.TIMED_OUT
                else (
                    ExecutionStatus.CANCELLED
                    if receipt.status == CommandStatus.CANCELLED
                    else ExecutionStatus.FAILED
                )
            )
            return DriverResult(
                status,
                "terminal",
                workflow.command_id,
                receipt.reason or receipt.status.value,
            )
        assert execution.command is not None
        if workflow.stage == "approach_seat":
            workflow.stage = "align_seat"
            workflow.command_id = self._submit_stage(
                execution.command.agent_id,
                execution.execution_id,
                workflow.stage,
                NpcCommandKind.ALIGN_TO,
                {"yaw": workflow.yaw},
            )
            return DriverResult(ExecutionStatus.RUNNING, workflow.stage, workflow.command_id)
        if workflow.stage == "align_seat":
            workflow.stage = "sit_transition"
            workflow.command_id = self._submit_stage(
                execution.command.agent_id,
                execution.execution_id,
                workflow.stage,
                NpcCommandKind.PLAY_ANIMATION,
                {"clip": "sit", "completion_marker": "seated"},
            )
            return DriverResult(ExecutionStatus.RUNNING, workflow.stage, workflow.command_id)
        self._sits.pop(execution.execution_id, None)
        return DriverResult(ExecutionStatus.SUCCEEDED, "seated", workflow.command_id)

    def _start_handover(self, execution: ActionExecution) -> DriverResult:
        assert execution.command is not None
        command = execution.command
        object_name = str(command.parameters.get("object", ""))
        receiver = str(command.target or "")
        if not object_name or not receiver:
            return DriverResult(
                ExecutionStatus.FAILED, "prepare", error="Handover requires receiver and object"
            )
        session = self.interactions.start(
            "handover",
            (command.agent_id, receiver),
            object_name,
            (
                ("ready", (command.agent_id, receiver)),
                ("released", (command.agent_id,)),
                ("received", (receiver,)),
            ),
            session_id=execution.execution_id,
        )
        handle = self._submit_stage(
            command.agent_id,
            execution.execution_id,
            "giver_ready",
            NpcCommandKind.INTERACTION_CUE,
            {"clip": "idle", "duration": 0.1, "interaction_id": session.session_id},
        )
        self._handovers[execution.execution_id] = _HandoverWorkflow(
            session.session_id,
            command.agent_id,
            receiver,
            object_name,
            "giver_ready",
            handle,
        )
        return DriverResult(ExecutionStatus.RUNNING, "giver_ready", handle)

    def _advance_handover(
        self,
        execution: ActionExecution,
        workflow: _HandoverWorkflow,
        receipt: NpcCommandReceipt,
    ) -> DriverResult:
        if receipt.status != CommandStatus.SUCCEEDED:
            self.interactions.fail(workflow.session_id, receipt.reason or receipt.status.value)
            return DriverResult(
                ExecutionStatus.FAILED,
                "terminal",
                workflow.command_id,
                receipt.reason or receipt.status.value,
            )
        if workflow.stage == "giver_ready":
            self.interactions.acknowledge(workflow.session_id, workflow.giver, "ready")
            workflow.stage = "receiver_ready"
            workflow.command_id = self._submit_stage(
                workflow.receiver,
                execution.execution_id,
                workflow.stage,
                NpcCommandKind.INTERACTION_CUE,
                {"clip": "idle", "duration": 0.1, "interaction_id": workflow.session_id},
            )
        elif workflow.stage == "receiver_ready":
            self.interactions.acknowledge(workflow.session_id, workflow.receiver, "ready")
            workflow.stage = "release"
            workflow.command_id = self._submit_stage(
                workflow.giver,
                execution.execution_id,
                workflow.stage,
                NpcCommandKind.DETACH_OBJECT,
                {"object": workflow.object_name},
            )
        elif workflow.stage == "release":
            self.interactions.acknowledge(workflow.session_id, workflow.giver, "released")
            workflow.stage = "receive"
            workflow.command_id = self._submit_stage(
                workflow.receiver,
                execution.execution_id,
                workflow.stage,
                NpcCommandKind.ATTACH_OBJECT,
                {"object": workflow.object_name},
            )
        else:
            session = self.interactions.acknowledge(
                workflow.session_id, workflow.receiver, "received"
            )
            self._handovers.pop(execution.execution_id, None)
            if session.status.value == "succeeded":
                return DriverResult(ExecutionStatus.SUCCEEDED, "received", workflow.command_id)
            return DriverResult(ExecutionStatus.FAILED, "terminal", workflow.command_id)
        return DriverResult(ExecutionStatus.RUNNING, workflow.stage, workflow.command_id)

    def _submit_stage(
        self,
        npc_id: str,
        execution_id: str,
        stage: str,
        kind: NpcCommandKind,
        payload: dict[str, object],
    ) -> str:
        issued_at = float(self.simulator.pull_status().time)
        sequence = self._sequences.get(npc_id, -1) + 1
        self._sequences[npc_id] = sequence
        command = NpcCommand(
            command_id=f"{execution_id}:{stage}",
            sequence=sequence,
            npc_id=npc_id,
            kind=kind,
            payload=payload,
            issued_at=issued_at,
            deadline=issued_at + self.timeout_seconds,
        )
        return self.simulator.submit_npc_command(command)

    def cancel(self, execution: ActionExecution, reason: str) -> DriverResult:
        workflow = self._handovers.get(execution.execution_id)
        if workflow is not None:
            npc_id = (
                workflow.receiver
                if workflow.stage in {"receiver_ready", "receive"}
                else workflow.giver
            )
            self.simulator.cancel_npc_command(npc_id, workflow.command_id)
            self.interactions.cancel(workflow.session_id, reason)
        sit = self._sits.pop(execution.execution_id, None)
        if sit is not None and execution.command is not None:
            self.simulator.cancel_npc_command(execution.command.agent_id, sit.command_id)
        elif execution.driver_handle is not None and execution.command is not None:
            self.simulator.cancel_npc_command(execution.command.agent_id, execution.driver_handle)
        return DriverResult(
            ExecutionStatus.CANCELLED,
            "cancelled",
            execution.driver_handle,
            reason,
        )
