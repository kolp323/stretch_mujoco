"""Narrow execution drivers separating action intent from physical completion."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Protocol

from stretch_mujoco.npc.protocol import CommandStatus, NpcCommand, NpcCommandKind, NpcCommandReceipt
from stretch_mujoco.npc.trajectory_profile import NpcTrajectoryProfile

from .actions import ActionExecution, ActionType, ExecutionStatus
from .conversation import ConversationSession, DialogueTurn
from .action_recipes import ACTION_RECIPES, ActionRecipe
from .desk_work import (
    WORK_DURATION_SECONDS_PARAMETER,
    WORK_SESSION_ID_PARAMETER,
    WORK_SESSION_SEAT_PARAMETER,
)
from .interactions import InteractionCoordinator
from .interaction_stations import (
    InteractionStationAllocationError,
    InteractionStationAllocator,
    InteractionStationLease,
    SeatSlotAllocator,
)


EMBODIED_ACTION_REQUIRED_CLIPS: dict[ActionType, frozenset[str]] = {
    ActionType.IDLE: frozenset({ACTION_RECIPES[ActionType.IDLE].animation}),
    ActionType.MOVE_TO: frozenset({ACTION_RECIPES[ActionType.MOVE_TO].animation}),
    ActionType.SIT: frozenset({ACTION_RECIPES[ActionType.SIT].animation}),
    ActionType.STAND_UP: frozenset({ACTION_RECIPES[ActionType.STAND_UP].animation}),
    ActionType.WORK: frozenset({ACTION_RECIPES[ActionType.WORK].animation}),
    ActionType.PICK_UP: frozenset({ACTION_RECIPES[ActionType.PICK_UP].animation}),
    ActionType.PUT_DOWN: frozenset({ACTION_RECIPES[ActionType.PUT_DOWN].animation}),
    # A handover cannot complete unless both participants' clips are available.
    ActionType.HANDOVER: frozenset({"give", "receive"}),
    ActionType.USE_COMPUTER: frozenset({ACTION_RECIPES[ActionType.USE_COMPUTER].animation}),
    ActionType.TALK: frozenset({ACTION_RECIPES[ActionType.TALK].animation}),
    ActionType.GESTURE_POINT: frozenset({ACTION_RECIPES[ActionType.GESTURE_POINT].animation}),
    ActionType.GESTURE_WAVE: frozenset({ACTION_RECIPES[ActionType.GESTURE_WAVE].animation}),
    ActionType.EAT: frozenset({ACTION_RECIPES[ActionType.EAT].animation}),
    ActionType.DRINK: frozenset({ACTION_RECIPES[ActionType.DRINK].animation}),
    ActionType.REQUEST_ROBOT: frozenset({ACTION_RECIPES[ActionType.REQUEST_ROBOT].animation}),
}


@dataclass(frozen=True)
class DriverResult:
    status: ExecutionStatus
    phase: str
    handle: str | None = None
    error: str | None = None
    receipt_ids: tuple[str, ...] = ()
    cleanup_evidence_id: str | None = None
    station_id: str | None = None
    lease_id: str | None = None
    compatibility_mode: str | None = None
    production_evidence: bool = False


class LocationSlotAllocationError(ValueError):
    """Raised when a configured multi-occupancy location has no free slot."""


class LocationSlotAllocator:
    """Own physical location slots for one embodied NPC driver.

    Semantic locations (for example ``meeting_table``) remain broad regions in
    :class:`OfficeAgentRuntime`.  This allocator owns only the concrete MuJoCo
    destination sites declared for a region, so it deliberately does not share
    the runtime's chair/object ``ReservationManager`` namespace.
    """

    def __init__(self, slots_by_target: dict[str, dict[str, str]] | None = None) -> None:
        self._slots_by_target: dict[str, tuple[str, ...]] = {}
        self._owners: dict[str, str] = {}
        declared_targets: dict[str, str] = {}
        for target, configured_slots in (slots_by_target or {}).items():
            if not isinstance(target, str) or not target:
                raise ValueError("Location slot target must be a non-empty string")
            if not configured_slots:
                raise ValueError(f"Location '{target}' must declare at least one slot")
            slots: list[str] = []
            seen_sites: set[str] = set()
            for slot_id, site in configured_slots.items():
                if not isinstance(slot_id, str) or not slot_id:
                    raise ValueError(f"Location '{target}' has an invalid slot_id")
                if not isinstance(site, str) or not site:
                    raise ValueError(f"Location slot '{target}/{slot_id}' requires a site")
                if site in seen_sites:
                    raise ValueError(
                        f"Location '{target}' assigns site '{site}' to more than one slot"
                    )
                other_target = declared_targets.get(site)
                if other_target is not None:
                    raise ValueError(
                        f"Location slot site '{site}' is shared by '{other_target}' and '{target}'"
                    )
                seen_sites.add(site)
                declared_targets[site] = target
                slots.append(site)
            self._slots_by_target[target] = tuple(slots)

    def has_target(self, target: str) -> bool:
        return target in self._slots_by_target

    def acquire(self, target: str, npc_id: str) -> str:
        """Lease the first free configured slot, idempotently per NPC/target."""
        try:
            slots = self._slots_by_target[target]
        except KeyError as error:
            raise LocationSlotAllocationError(f"location_slots_unknown_target:{target}") from error
        for site in slots:
            if self._owners.get(site) == npc_id:
                return site
        for site in slots:
            if site not in self._owners:
                self._owners[site] = npc_id
                return site
        raise LocationSlotAllocationError(f"location_slots_exhausted:{target}")

    def release(self, site: str, npc_id: str) -> bool:
        if self._owners.get(site) != npc_id:
            return False
        self._owners.pop(site)
        return True

    def owner(self, site: str) -> str | None:
        return self._owners.get(site)

    def snapshot(self) -> dict[str, str]:
        return dict(self._owners)


@dataclass(frozen=True)
class _LocationSlotLease:
    target: str
    site: str


@dataclass(frozen=True)
class _PendingLocationSlotMovement:
    npc_id: str
    previous: _LocationSlotLease | None
    next_lease: _LocationSlotLease | None


@dataclass
class _HandoverWorkflow:
    session_id: str
    giver: str
    receiver: str
    object_name: str
    giver_site: str
    receiver_site: str
    giver_yaw: float
    receiver_yaw: float
    stage: str
    command_ids: dict[str, str]
    lease: InteractionStationLease | None = None
    transfer_site: str | None = None
    distance_min_m: float = 0.45
    distance_max_m: float = 0.95
    yaw_tolerance_rad: float = 0.30
    physical_receipt_ids: list[str] = field(default_factory=list)
    released: bool = False
    failure_error: str | None = None
    terminal_outcome: str = "failed"
    compatibility_mode: str = "population_v2_legacy_adapter"
    production_evidence: bool = False


@dataclass
class _SitWorkflow:
    seat: str
    yaw: float
    stage: str
    command_id: str
    session_id: str
    ingress_site: str
    sit_site: str
    receipt_ids: list[str] = field(default_factory=list)
    failure_error: str | None = None


@dataclass
class _StandWorkflow:
    agent_id: str
    seat: str
    session_id: str
    ingress_site: str | None
    stage: str
    command_id: str
    receipt_ids: list[str] = field(default_factory=list)


@dataclass
class _ObjectWorkflow:
    object_name: str
    stage: str
    command_id: str
    clip: str
    marker: str
    yaw: float
    detach_site: str | None = None


@dataclass
class _RecipeWorkflow:
    recipe: ActionRecipe
    stage: str
    command_id: str
    yaw: float
    target_site: str


@dataclass
class _ConversationWorkflow:
    session_id: str
    participants: tuple[str, ...]
    sites: dict[str, str]
    stage: str
    command_ids: dict[str, str]
    lease: InteractionStationLease | None = None
    distance_min_m: float = 0.45
    distance_max_m: float = 0.95
    yaw_tolerance_rad: float = 0.30
    last_physical_receipt_ids: tuple[str, ...] = ()


class _ConversationCommandSubmitError(RuntimeError):
    def __init__(self, cause: Exception, submitted: dict[str, str]) -> None:
        super().__init__(str(cause))
        self.cause = cause
        self.submitted = submitted


@dataclass
class _RobotRequestWorkflow:
    request_site: str
    stage: str
    command_id: str


class ActionDriver(Protocol):
    candidate_actions: frozenset[ActionType]
    supported_actions: frozenset[ActionType]

    def start(self, execution: ActionExecution) -> DriverResult: ...

    def poll(self, execution: ActionExecution) -> DriverResult: ...

    def cancel(self, execution: ActionExecution, reason: str) -> DriverResult: ...


class InteractionDriver(Protocol):
    """Receipt-owning bridge used by the strict conversation runtime path.

    Implementations may return ``RUNNING`` while movement/alignment is in
    progress, but may return ``SUCCEEDED`` only after every participant's
    physical receipt and range/yaw gate have been checked.
    """

    def prepare_conversation(self, session: ConversationSession) -> DriverResult: ...

    def play_turn(self, session: ConversationSession, turn: DialogueTurn) -> DriverResult: ...

    def poll_conversation(self, session_id: str) -> DriverResult: ...

    def cancel_conversation(self, session_id: str, reason: str) -> DriverResult: ...


class SimulatorStatus(Protocol):
    time: float


class NpcSimulatorClient(Protocol):
    def pull_status(self) -> SimulatorStatus: ...

    def submit_npc_command(self, command: NpcCommand) -> str: ...

    def pull_npc_receipts(self) -> tuple[NpcCommandReceipt, ...]: ...

    def cancel_npc_command(self, npc_id: str, command_id: str) -> None: ...


class MujocoNpcActionDriver:
    """Lower embodied actions into commands and wait for physical receipts."""

    # Semantic recipes without an approved physical marker contract must not
    # fall through to runtime duration completion when this driver is active.
    embodied_only_unsupported_actions = frozenset(
        {ActionType.REST, ActionType.ATTEND_MEETING, ActionType.OPEN_CABINET}
    )

    # These are the actions this driver knows how to lower.  ``supported_actions``
    # is narrowed per instance when a production animation manifest is supplied.
    candidate_actions = frozenset(EMBODIED_ACTION_REQUIRED_CLIPS)

    def __init__(
        self,
        simulator: NpcSimulatorClient,
        location_sites: dict[str, str],
        placement_sites: dict[str, str] | None = None,
        *,
        object_approach_sites: dict[str, str] | None = None,
        handover_sites: dict[str, str] | None = None,
        handover_role_sites: dict[tuple[str, str], tuple[str, str]] | None = None,
        handover_role_yaws: dict[tuple[str, str], tuple[float, float]] | None = None,
        conversation_role_sites: dict[tuple[str, str], tuple[str, str]] | None = None,
        conversation_site_yaws: dict[str, float] | None = None,
        interaction_station_allocator: InteractionStationAllocator | None = None,
        seat_slot_allocator: SeatSlotAllocator | None = None,
        seat_verifier: Callable[[str, str, str, str], str | None] | None = None,
        cue_sites: dict[str, str] | None = None,
        seat_yaws: dict[str, float] | None = None,
        seat_navigation_sites: dict[str, str] | None = None,
        interaction_yaws: dict[str, float] | None = None,
        robot_request_sites: dict[str, str] | None = None,
        location_slot_sites: dict[str, dict[str, str]] | None = None,
        available_clips: set[str] | None = None,
        trajectory_profile: NpcTrajectoryProfile | None = None,
        agent_locations: dict[str, str] | None = None,
        timeout_seconds: float = 30.0,
    ) -> None:
        self.simulator = simulator
        self.location_sites = dict(location_sites)
        self.placement_sites = dict(placement_sites or {})
        self.object_approach_sites = dict(object_approach_sites or {})
        self.handover_sites = dict(handover_sites or {})
        self.handover_role_sites = dict(handover_role_sites or {})
        self.handover_role_yaws = dict(handover_role_yaws or {})
        self.conversation_role_sites = dict(conversation_role_sites or {})
        self.conversation_site_yaws = dict(conversation_site_yaws or {})
        self.interaction_station_allocator = interaction_station_allocator
        self.seat_slot_allocator = seat_slot_allocator
        self.seat_verifier = seat_verifier
        self.cue_sites = dict(cue_sites or {})
        self.seat_yaws = dict(seat_yaws or {})
        self.seat_navigation_sites = dict(seat_navigation_sites or {})
        self.interaction_yaws = dict(interaction_yaws or {})
        self.robot_request_sites = dict(robot_request_sites or {})
        self.location_slots = LocationSlotAllocator(location_slot_sites)
        self.available_clips = None if available_clips is None else set(available_clips)
        self.supported_actions = (
            self.candidate_actions
            if self.available_clips is None
            else frozenset(
                action
                for action, required_clips in EMBODIED_ACTION_REQUIRED_CLIPS.items()
                if required_clips <= self.available_clips
            )
        )
        self.trajectory_profile = trajectory_profile
        self.agent_locations = dict(agent_locations or {})
        self._movement_destinations: dict[str, tuple[str, str]] = {}
        # A completed MOVE_TO owns its physical region slot until its owner
        # successfully reaches another location (or the caller explicitly
        # releases it).  Pending moves retain the prior slot until the receipt
        # is terminal, which avoids a failed navigation silently evicting an
        # NPC from the region it still occupies.
        self._active_location_slots: dict[str, _LocationSlotLease] = {}
        self._pending_location_slot_movements: dict[str, _PendingLocationSlotMovement] = {}
        self._location_slot_movements: dict[str, _PendingLocationSlotMovement] = {}
        self.timeout_seconds = timeout_seconds
        self._sequences: dict[str, int] = {}
        self._receipts: dict[str, NpcCommandReceipt] = {}
        self.interactions = InteractionCoordinator()
        self._handovers: dict[str, _HandoverWorkflow] = {}
        self._handover_terminal_results: dict[str, DriverResult] = {}
        self._sits: dict[str, _SitWorkflow] = {}
        self._objects: dict[str, _ObjectWorkflow] = {}
        self._recipes: dict[str, _RecipeWorkflow] = {}
        self._conversations: dict[str, _ConversationWorkflow] = {}
        self._robot_requests: dict[str, _RobotRequestWorkflow] = {}
        self._seated_agents: dict[str, str] = {}
        # execution_id -> (agent, seat, optional navigation exit, exit submitted)
        self._standing_up: dict[str, _StandWorkflow] = {}
        # Role sites are physical resources, not merely semantic labels.  A
        # workflow owns both endpoints until its terminal cleanup path runs.
        self._interaction_site_leases: dict[str, str] = {}

    def start(self, execution: ActionExecution) -> DriverResult:
        command = execution.command
        if command is None or command.action not in self.supported_actions:
            return DriverResult(ExecutionStatus.FAILED, "start", error="unsupported_action")
        if command.action == ActionType.MOVE_TO and command.agent_id in self._seated_agents:
            return DriverResult(ExecutionStatus.FAILED, "prepare", error="move_requires_stand_up")
        if command.action == ActionType.HANDOVER:
            return self._start_handover(execution)
        if command.action == ActionType.SIT:
            return self._start_sit(execution)
        if command.action == ActionType.STAND_UP:
            return self._start_stand_up(execution)
        if command.action == ActionType.WORK:
            return self._start_desk_work(execution)
        if command.action == ActionType.REQUEST_ROBOT:
            return self._start_robot_request(execution)
        if command.action == ActionType.USE_COMPUTER:
            return DriverResult(
                ExecutionStatus.FAILED,
                "start",
                error="use_computer_must_expand_to_desk_work",
            )
        if command.action == ActionType.IDLE:
            duration = command.parameters.get("duration_seconds", 0.25)
            if (
                isinstance(duration, bool)
                or not isinstance(duration, (int, float))
                or not math.isfinite(duration)
                or duration <= 0
            ):
                return DriverResult(
                    ExecutionStatus.FAILED, "prepare", error="idle_duration_invalid"
                )
            handle = self._submit_stage(
                command.agent_id,
                execution.execution_id,
                "idle",
                NpcCommandKind.PLAY_ANIMATION,
                {"clip": "idle", "duration": float(duration), "arrival_clip": "idle"},
                timeout_seconds=ACTION_RECIPES[ActionType.IDLE].timeout_seconds,
            )
            return DriverResult(ExecutionStatus.RUNNING, "idle", handle)
        if command.action in {
            ActionType.TALK,
            ActionType.GESTURE_POINT,
            ActionType.GESTURE_WAVE,
            ActionType.EAT,
            ActionType.DRINK,
        }:
            return self._start_recipe(execution)
        if command.action in {ActionType.PICK_UP, ActionType.PUT_DOWN}:
            return self._start_object_action(execution)
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
            deadline=issued_at + self._timeout_for(command.action),
        )
        try:
            handle = self.simulator.submit_npc_command(npc_command)
        except Exception:
            self._settle_pending_location_slot(execution.execution_id, succeeded=False)
            raise
        if kind == NpcCommandKind.MOVE_TO:
            self._remember_movement(handle, command.agent_id, str(payload["site"]))
            pending_slot = self._pending_location_slot_movements.pop(execution.execution_id, None)
            if pending_slot is not None:
                self._location_slot_movements[handle] = pending_slot
        return DriverResult(ExecutionStatus.RUNNING, phase, handle=handle)

    def poll(self, execution: ActionExecution) -> DriverResult:
        for new_receipt in self.simulator.pull_npc_receipts():
            self._receipts[new_receipt.command_id] = new_receipt
            if new_receipt.status == CommandStatus.SUCCEEDED:
                self._record_arrival(new_receipt.command_id)
        workflow = self._handovers.get(execution.execution_id)
        sit = self._sits.get(execution.execution_id)
        object_workflow = self._objects.get(execution.execution_id)
        recipe_workflow = self._recipes.get(execution.execution_id)
        robot_request = self._robot_requests.get(execution.execution_id)
        handle = (
            self._workflow_handle(workflow)
            if workflow is not None
            else (
                sit.command_id
                if sit is not None
                else (
                    object_workflow.command_id
                    if object_workflow is not None
                    else str(execution.driver_handle)
                )
            )
        )
        if recipe_workflow is not None:
            handle = recipe_workflow.command_id
        if robot_request is not None:
            handle = robot_request.command_id
        self.interactions.check_deadlines(float(self.simulator.pull_status().time))
        if workflow is not None:
            session = self.interactions.sessions[workflow.session_id]
            if session.status.value == "timed_out" and workflow.stage != "rollback":
                if workflow.released:
                    return self._start_handover_reconciliation(
                        execution.execution_id,
                        workflow,
                        "timed_out",
                        session.error or "deadline_exceeded",
                    )
                self._cancel_workflow_commands(workflow)
                cleanup_evidence_id = (
                    "cleanup:handover_timed_out_unconfirmed:"
                    f"{self._workflow_handle(workflow)}"
                )
                return self._terminal_handover(
                    execution.execution_id,
                    workflow,
                    ExecutionStatus.TIMED_OUT,
                    "timed_out",
                    session.error or "deadline_exceeded",
                    cleanup_evidence_id=cleanup_evidence_id,
                )
        if workflow is not None:
            return self._advance_handover(execution, workflow)
        if robot_request is not None:
            receipt = self._receipts.get(robot_request.command_id)
            if receipt is None or receipt.status in {CommandStatus.ACCEPTED, CommandStatus.RUNNING}:
                return DriverResult(
                    ExecutionStatus.RUNNING, robot_request.stage, robot_request.command_id
                )
            return self._advance_robot_request(execution, robot_request, receipt)
        receipt = self._receipts.get(handle)
        if receipt is None or receipt.status in {CommandStatus.ACCEPTED, CommandStatus.RUNNING}:
            return DriverResult(
                ExecutionStatus.RUNNING,
                (
                    workflow.stage
                    if workflow is not None
                    else (
                        sit.stage
                        if sit is not None
                        else (
                            object_workflow.stage
                            if object_workflow is not None
                            else execution.phase
                        )
                    )
                ),
                handle,
            )
        if sit is not None:
            return self._advance_sit(execution, sit, receipt)
        if object_workflow is not None:
            return self._advance_object_action(execution, object_workflow, receipt)
        if recipe_workflow is not None:
            return self._advance_recipe(execution, recipe_workflow, receipt)
        if receipt.status == CommandStatus.SUCCEEDED:
            self._settle_location_slot_movement(handle, succeeded=True)
            stood = self._standing_up.get(execution.execution_id)
            if stood is not None:
                stood.receipt_ids.append(receipt.command_id)
            if stood is not None and stood.stage == "stand_transition" and stood.ingress_site:
                exit_handle = self._submit_stage(
                    stood.agent_id, execution.execution_id, "stand_exit", NpcCommandKind.MOVE_TO,
                    self._move_payload(
                        stood.ingress_site, agent_id=stood.agent_id, action=ActionType.STAND_UP
                    ),
                    timeout_seconds=ACTION_RECIPES[ActionType.STAND_UP].timeout_seconds,
                )
                stood.stage = "stand_exit"
                stood.command_id = exit_handle
                execution.driver_handle = exit_handle
                return DriverResult(
                    ExecutionStatus.RUNNING,
                    "stand_exit",
                    exit_handle,
                    receipt_ids=tuple(stood.receipt_ids),
                )
            self._standing_up.pop(execution.execution_id, None)
            if stood is not None:
                verification_id = self._verify_seat(
                    stood.agent_id, stood.seat, "standing", stood.ingress_site or ""
                )
                if verification_id is None:
                    return DriverResult(
                        ExecutionStatus.FAILED,
                        "stand_verification",
                        execution.driver_handle,
                        "seat_stand_verification_failed",
                        tuple(stood.receipt_ids),
                    )
                stood.receipt_ids.append(verification_id)
                if self.seat_slot_allocator is not None:
                    self.seat_slot_allocator.release_occupied(
                        stood.seat,
                        stood.agent_id,
                        stood.session_id,
                        verification_id,
                    )
                if self._seated_agents.get(stood.agent_id) == stood.seat:
                    self._seated_agents.pop(stood.agent_id, None)
                return DriverResult(
                    ExecutionStatus.SUCCEEDED,
                    "standing",
                    execution.driver_handle,
                    receipt_ids=tuple(stood.receipt_ids),
                    station_id=stood.seat,
                    lease_id=stood.session_id,
                    compatibility_mode=(
                        "population_v3_seat_slot"
                        if self.seat_slot_allocator is not None
                        else "legacy_chair_adapter"
                    ),
                )
            return DriverResult(
                ExecutionStatus.SUCCEEDED,
                "arrived",
                execution.driver_handle,
                receipt_ids=(receipt.command_id,),
            )
        status = self._execution_status(receipt.status)
        self._settle_location_slot_movement(handle, succeeded=False)
        stood = self._standing_up.pop(execution.execution_id, None)
        if stood is not None:
            stood.receipt_ids.append(receipt.command_id)
            return DriverResult(
                status,
                "terminal",
                execution.driver_handle,
                receipt.reason,
                tuple(stood.receipt_ids),
                station_id=stood.seat,
                lease_id=stood.session_id,
            )
        return DriverResult(
            status,
            "terminal",
            execution.driver_handle,
            receipt.reason,
            (receipt.command_id,),
        )

    @staticmethod
    def _execution_status(status: CommandStatus) -> ExecutionStatus:
        if status == CommandStatus.TIMED_OUT:
            return ExecutionStatus.TIMED_OUT
        if status == CommandStatus.CANCELLED:
            return ExecutionStatus.CANCELLED
        return ExecutionStatus.FAILED

    @staticmethod
    def _workflow_handle(workflow: _HandoverWorkflow) -> str | None:
        return next(iter(workflow.command_ids.values()), None)

    def _cancel_workflow_commands(self, workflow: _HandoverWorkflow) -> None:
        for npc_id, command_id in workflow.command_ids.items():
            self.simulator.cancel_npc_command(npc_id, command_id)

    def _lease_interaction_sites(self, owner: str, sites: tuple[str, ...]) -> bool:
        if any(self._interaction_site_leases.get(site) not in {None, owner} for site in sites):
            return False
        for site in sites:
            self._interaction_site_leases[site] = owner
        return True

    def _release_interaction_sites(self, owner: str) -> None:
        for site, lease_owner in tuple(self._interaction_site_leases.items()):
            if lease_owner == owner:
                self._interaction_site_leases.pop(site, None)

    def _physical_command(
        self, execution: ActionExecution
    ) -> tuple[NpcCommandKind, dict[str, object], str]:
        assert execution.command is not None
        command = execution.command
        if command.action == ActionType.MOVE_TO:
            target = str(command.target)
            # Chair sit sites are inside their inflated collision footprints.
            # Navigation ends at the associated ingress; only SIT may enter it.
            location_site = self._location_site_for_move(execution, target)
            site = self.seat_navigation_sites.get(target) or location_site
            if site is None:
                raise ValueError(f"No NPC site configured for location '{command.target}'")
            return (
                NpcCommandKind.MOVE_TO,
                self._move_payload(
                    site,
                    agent_id=command.agent_id,
                    action=command.action,
                    requested_route=command.parameters.get("trajectory_route"),
                    requested_source=command.parameters.get("trajectory_source"),
                ),
                "navigate",
            )
        raise ValueError(f"Unsupported embodied action '{command.action.value}'")

    def _location_site_for_move(self, execution: ActionExecution, target: str) -> str | None:
        """Resolve a MOVE_TO region to a leased slot or its legacy single site."""
        assert execution.command is not None
        npc_id = execution.command.agent_id
        previous = self._active_location_slots.get(npc_id)
        if self.location_slots.has_target(target):
            if previous is not None and previous.target == target:
                return previous.site
            try:
                site = self.location_slots.acquire(target, npc_id)
            except LocationSlotAllocationError as error:
                raise ValueError(str(error)) from error
            self._pending_location_slot_movements[execution.execution_id] = (
                _PendingLocationSlotMovement(
                    npc_id=npc_id,
                    previous=previous,
                    next_lease=_LocationSlotLease(target, site),
                )
            )
            return site

        site = self.location_sites.get(target)
        if previous is not None and previous.target != target:
            # A legacy target has no lease of its own, but a successful move to
            # it still means the NPC has left the slot-backed region.
            self._pending_location_slot_movements[execution.execution_id] = (
                _PendingLocationSlotMovement(npc_id=npc_id, previous=previous, next_lease=None)
            )
        return site

    def _settle_pending_location_slot(self, execution_id: str, *, succeeded: bool) -> None:
        movement = self._pending_location_slot_movements.pop(execution_id, None)
        if movement is not None:
            self._settle_location_slot(movement, succeeded=succeeded)

    def _settle_location_slot_movement(self, command_id: str | None, *, succeeded: bool) -> None:
        if command_id is None:
            return
        movement = self._location_slot_movements.pop(command_id, None)
        if movement is not None:
            self._settle_location_slot(movement, succeeded=succeeded)

    def _settle_location_slot(
        self, movement: _PendingLocationSlotMovement, *, succeeded: bool
    ) -> None:
        previous = movement.previous
        next_lease = movement.next_lease
        if succeeded:
            if previous is not None and (next_lease is None or previous.site != next_lease.site):
                self.location_slots.release(previous.site, movement.npc_id)
            if next_lease is None:
                self._active_location_slots.pop(movement.npc_id, None)
            else:
                self._active_location_slots[movement.npc_id] = next_lease
            return
        if next_lease is not None and (previous is None or previous.site != next_lease.site):
            self.location_slots.release(next_lease.site, movement.npc_id)

    def _timeout_for(self, action: ActionType) -> float:
        configured = ACTION_RECIPES[action].timeout_seconds
        return self.timeout_seconds if configured is None else configured

    def _move_payload(
        self,
        site: str,
        *,
        agent_id: str | None = None,
        action: ActionType | None = None,
        requested_route: object | None = None,
        requested_source: object | None = None,
    ) -> dict[str, object]:
        """Every embodied approach has a named target and one local replan."""
        payload: dict[str, object] = {"site": site, "arrival_clip": "idle", "max_replans": 1}
        if requested_route is not None:
            if self.trajectory_profile is None or agent_id is None or action is None:
                raise ValueError(f"trajectory_route_unknown:{requested_route}")
            routes = [
                route
                for route in self.trajectory_profile.routes
                if route.route_id == str(requested_route)
            ]
            if len(routes) != 1:
                raise ValueError(f"trajectory_route_unknown:{requested_route}")
            route = routes[0]
            destination_site = self.trajectory_profile.anchors[route.destination].site
            if (
                destination_site != site
                or action.value not in route.actions
                or requested_source is None
                or route.source != str(requested_source)
            ):
                raise ValueError(f"trajectory_route_contract_mismatch:{requested_route}")
            payload["trajectory_route"] = route.route_id
            payload["trajectory_source"] = route.source
            payload["route_mode"] = "audited"
            return payload
        if self.trajectory_profile is not None and agent_id is not None and action is not None:
            route = self.trajectory_profile.route_for(
                self.agent_locations.get(agent_id), site, action.value
            )
            if route is not None:
                payload["trajectory_route"] = route.route_id
                payload["trajectory_source"] = route.source
                payload["route_mode"] = "audited"
            else:
                # The scene contract audits only declared profile routes.  This
                # remains a live collision-checked dynamic route, not a claim
                # that it passed profile preflight.
                payload["route_mode"] = "dynamic"
        return payload

    def _remember_movement(self, command_id: str, npc_id: str, site: str) -> None:
        self._movement_destinations[command_id] = (npc_id, site)

    def _record_arrival(self, command_id: str) -> None:
        movement = self._movement_destinations.pop(command_id, None)
        if movement is None or self.trajectory_profile is None:
            return
        npc_id, site = movement
        anchors = [
            anchor_id
            for anchor_id, anchor in self.trajectory_profile.anchors.items()
            if anchor.site == site
        ]
        if len(anchors) == 1:
            self.agent_locations[npc_id] = anchors[0]

    def _start_robot_request(self, execution: ActionExecution) -> DriverResult:
        """Require an NPC to reach and address the live Stretch body first."""
        assert execution.command is not None
        command = execution.command
        request_site = self.robot_request_sites.get(str(command.target))
        recipe = ACTION_RECIPES[ActionType.REQUEST_ROBOT]
        if request_site is None:
            return DriverResult(
                ExecutionStatus.FAILED, "prepare", error="robot_request_site_missing"
            )
        if self.available_clips is not None and recipe.animation not in self.available_clips:
            return DriverResult(
                ExecutionStatus.FAILED, "prepare", error="robot_request_clip_unavailable"
            )
        handle = self._submit_stage(
            command.agent_id,
            execution.execution_id,
            "approach_stretch",
            NpcCommandKind.MOVE_TO,
            self._move_payload(request_site, agent_id=command.agent_id, action=command.action),
            timeout_seconds=recipe.timeout_seconds,
        )
        self._robot_requests[execution.execution_id] = _RobotRequestWorkflow(
            request_site, "approach_stretch", handle
        )
        return DriverResult(ExecutionStatus.RUNNING, "approach_stretch", handle)

    def _advance_robot_request(
        self,
        execution: ActionExecution,
        workflow: _RobotRequestWorkflow,
        receipt: NpcCommandReceipt,
    ) -> DriverResult:
        if receipt.status != CommandStatus.SUCCEEDED:
            self._robot_requests.pop(execution.execution_id, None)
            return DriverResult(
                self._execution_status(receipt.status),
                "terminal",
                workflow.command_id,
                receipt.reason or receipt.status.value,
            )
        assert execution.command is not None
        if workflow.stage == "approach_stretch":
            # The robot-owned site supplies the NPC's final locomotion yaw.  The
            # controller then observes actual body distance and both yaws before
            # allowing the talk clip to advance to its completion marker.
            workflow.stage = "speak_request"
            workflow.command_id = self._submit_stage(
                execution.command.agent_id,
                execution.execution_id,
                workflow.stage,
                NpcCommandKind.PLAY_ANIMATION,
                {
                    "clip": "talk",
                    "completion_marker": "talk_cycle",
                    "arrival_clip": "idle",
                    "target_site": workflow.request_site,
                    "gaze_target": execution.command.target,
                    "interaction_target": execution.command.target,
                    "interaction_distance_min": 0.45,
                    "interaction_distance_max": 1.45,
                    # The robot's base starts facing +y while the collision-free
                    # stand point is left/front of it; keep a bounded facing gate
                    # that accepts this authored approach pose without disabling
                    # orientation validation.
                    "interaction_yaw_tolerance": 1.20,
                },
                timeout_seconds=ACTION_RECIPES[ActionType.REQUEST_ROBOT].timeout_seconds,
            )
            return DriverResult(ExecutionStatus.RUNNING, workflow.stage, workflow.command_id)
        self._robot_requests.pop(execution.execution_id, None)
        return DriverResult(ExecutionStatus.SUCCEEDED, "request_accepted", workflow.command_id)

    def _start_recipe(self, execution: ActionExecution) -> DriverResult:
        assert execution.command is not None
        command = execution.command
        recipe = ACTION_RECIPES[command.action]
        if command.action == ActionType.USE_COMPUTER:
            sites = self.location_sites
        elif command.action in {ActionType.EAT, ActionType.DRINK}:
            sites = self.object_approach_sites
        else:
            sites = self.cue_sites
        error = recipe.validate(
            target=command.target,
            location_sites=sites,
            yaws=self.interaction_yaws,
            available_clips=self.available_clips,
        )
        if error is not None:
            return DriverResult(ExecutionStatus.FAILED, "prepare", error=error)
        assert command.target is not None
        handle = self._submit_stage(
            command.agent_id,
            execution.execution_id,
            "approach",
            NpcCommandKind.MOVE_TO,
            self._move_payload(
                sites[command.target], agent_id=command.agent_id, action=command.action
            ),
            timeout_seconds=recipe.timeout_seconds,
        )
        self._recipes[execution.execution_id] = _RecipeWorkflow(
            recipe, "approach", handle, self.interaction_yaws[command.target], sites[command.target]
        )
        return DriverResult(ExecutionStatus.RUNNING, "approach", handle)

    def _advance_recipe(
        self, execution: ActionExecution, workflow: _RecipeWorkflow, receipt: NpcCommandReceipt
    ) -> DriverResult:
        if receipt.status != CommandStatus.SUCCEEDED:
            self._recipes.pop(execution.execution_id, None)
            return DriverResult(
                self._execution_status(receipt.status),
                "terminal",
                workflow.command_id,
                receipt.reason,
            )
        assert execution.command is not None
        kind: NpcCommandKind
        payload: dict[str, object]
        if workflow.stage == "approach":
            workflow.stage = "align"
            kind = NpcCommandKind.ALIGN_TO
            payload = {"yaw": workflow.yaw, "target_site": workflow.target_site}
        elif workflow.stage == "align":
            workflow.stage = "play"
            kind = NpcCommandKind.PLAY_ANIMATION
            payload = {
                "clip": workflow.recipe.animation,
                "completion_marker": workflow.recipe.completion_marker,
                "arrival_clip": "idle",
                "target_site": workflow.target_site,
            }
            if workflow.recipe.target_kind == "participant":
                payload.update(
                    gaze_target=execution.command.target,
                    interaction_target=execution.command.target,
                    interaction_distance_min=0.45,
                    interaction_distance_max=0.95,
                    interaction_yaw_tolerance=0.30,
                )
        else:
            self._recipes.pop(execution.execution_id, None)
            return DriverResult(ExecutionStatus.SUCCEEDED, "completed", workflow.command_id)
        workflow.command_id = self._submit_stage(
            execution.command.agent_id,
            execution.execution_id,
            workflow.stage,
            kind,
            payload,
            timeout_seconds=workflow.recipe.timeout_seconds,
        )
        return DriverResult(ExecutionStatus.RUNNING, workflow.stage, workflow.command_id)

    def _start_object_action(self, execution: ActionExecution) -> DriverResult:
        assert execution.command is not None
        command = execution.command
        recipe = ACTION_RECIPES[command.action]
        if command.action == ActionType.PICK_UP:
            object_name, clip, marker, detach_site = (
                str(command.target or ""),
                "pick_up",
                "grasp",
                None,
            )
            approach_site = self.object_approach_sites.get(object_name)
        else:
            object_name = str(command.parameters.get("object", ""))
            detach_site = self.placement_sites.get(str(command.target))
            approach_site = self.location_sites.get(str(command.target))
            clip, marker = "place", "release"
        if (
            not object_name
            or approach_site is None
            or (command.action == ActionType.PUT_DOWN and detach_site is None)
        ):
            return DriverResult(
                ExecutionStatus.FAILED, "prepare", error="Missing object approach or placement site"
            )
        validation_target = str(command.target or "")
        error = recipe.validate(
            target=validation_target,
            location_sites={validation_target: approach_site},
            yaws=self.interaction_yaws,
            available_clips=self.available_clips,
        )
        if error is not None:
            return DriverResult(ExecutionStatus.FAILED, "prepare", error=error)
        handle = self._submit_stage(
            command.agent_id,
            execution.execution_id,
            "approach",
            NpcCommandKind.MOVE_TO,
            self._move_payload(approach_site, agent_id=command.agent_id, action=command.action),
            timeout_seconds=recipe.timeout_seconds,
        )
        self._objects[execution.execution_id] = _ObjectWorkflow(
            object_name,
            "approach",
            handle,
            clip,
            marker,
            self.interaction_yaws[validation_target],
            detach_site,
        )
        return DriverResult(ExecutionStatus.RUNNING, "approach", handle)

    def _advance_object_action(
        self, execution: ActionExecution, workflow: _ObjectWorkflow, receipt: NpcCommandReceipt
    ) -> DriverResult:
        payload: dict[str, object]
        if receipt.status != CommandStatus.SUCCEEDED:
            self._objects.pop(execution.execution_id, None)
            return DriverResult(
                self._execution_status(receipt.status),
                "terminal",
                workflow.command_id,
                receipt.reason,
            )
        assert execution.command is not None
        recipe = ACTION_RECIPES[execution.command.action]
        if workflow.stage == "approach":
            workflow.stage, kind, payload = (
                "align",
                NpcCommandKind.ALIGN_TO,
                {"yaw": workflow.yaw, "target_site": self._object_target_site(execution)},
            )
        elif workflow.stage == "align":
            workflow.stage, kind, payload = (
                "grasp" if execution.command.action == ActionType.PICK_UP else "release",
                NpcCommandKind.PLAY_ANIMATION,
                {
                    "clip": workflow.clip,
                    "completion_marker": workflow.marker,
                    "arrival_clip": "idle",
                    "target_site": self._object_target_site(execution),
                },
            )
        elif workflow.stage == "grasp":
            workflow.stage, kind, payload = (
                "attach",
                NpcCommandKind.ATTACH_OBJECT,
                {"object": workflow.object_name, "visible": False},
            )
        elif workflow.stage == "release":
            assert workflow.detach_site is not None
            workflow.stage, kind, payload = (
                "detach",
                NpcCommandKind.DETACH_OBJECT,
                {"object": workflow.object_name, "site": workflow.detach_site},
            )
        else:
            self._objects.pop(execution.execution_id, None)
            return DriverResult(ExecutionStatus.SUCCEEDED, workflow.stage, workflow.command_id)
        workflow.command_id = self._submit_stage(
            execution.command.agent_id,
            execution.execution_id,
            workflow.stage,
            kind,
            payload,
            timeout_seconds=recipe.timeout_seconds,
        )
        return DriverResult(ExecutionStatus.RUNNING, workflow.stage, workflow.command_id)

    def _object_target_site(self, execution: ActionExecution) -> str:
        assert execution.command is not None
        if execution.command.action == ActionType.PICK_UP:
            return self.object_approach_sites[str(execution.command.target)]
        return self.location_sites[str(execution.command.target)]

    def _start_sit(self, execution: ActionExecution) -> DriverResult:
        assert execution.command is not None
        command = execution.command
        seat = str(command.target or "")
        site = self.location_sites.get(seat)
        yaw = self.seat_yaws.get(seat)
        session_id = str(command.parameters.get(WORK_SESSION_ID_PARAMETER) or execution.execution_id)
        navigation_site = self.seat_navigation_sites.get(seat)
        if self.seat_slot_allocator is not None:
            if self.seat_verifier is None:
                return DriverResult(
                    ExecutionStatus.FAILED,
                    "prepare",
                    error="seat_verification_unavailable",
                )
            try:
                definition = self.seat_slot_allocator.definition(seat)
                sit_role = definition.role("sit")
                ingress_role = definition.role("ingress")
                site = sit_role.site
                navigation_site = ingress_role.site
                yaw = sit_role.yaw
                self.seat_slot_allocator.reserve(seat, command.agent_id, session_id)
            except (InteractionStationAllocationError, KeyError) as error:
                return DriverResult(ExecutionStatus.FAILED, "prepare", error=str(error))
        if site is None:
            return DriverResult(
                ExecutionStatus.FAILED, "prepare", error=f"No seat site for '{seat}'"
            )
        if yaw is None:
            return DriverResult(
                ExecutionStatus.FAILED, "prepare", error=f"No seat yaw for '{seat}'"
            )
        payload = self._move_payload(
            navigation_site if self.seat_slot_allocator is not None and navigation_site else site,
            agent_id=command.agent_id,
            action=command.action,
        )
        if navigation_site is not None:
            payload["navigation_site"] = navigation_site
            if self.seat_slot_allocator is None:
                payload["allow_final_ingress"] = True
        try:
            handle = self._submit_stage(
                command.agent_id,
                execution.execution_id,
                "approach_seat",
                NpcCommandKind.MOVE_TO,
                payload,
                timeout_seconds=ACTION_RECIPES[ActionType.SIT].timeout_seconds,
            )
        except Exception:
            if self.seat_slot_allocator is not None:
                self.seat_slot_allocator.release_reservation(
                    seat, command.agent_id, session_id
                )
            raise
        self._sits[execution.execution_id] = _SitWorkflow(
            seat,
            yaw,
            "approach_seat",
            handle,
            session_id,
            navigation_site or site,
            site,
        )
        return DriverResult(ExecutionStatus.RUNNING, "approach_seat", handle)

    def _advance_sit(
        self,
        execution: ActionExecution,
        workflow: _SitWorkflow,
        receipt: NpcCommandReceipt,
    ) -> DriverResult:
        if receipt.status != CommandStatus.SUCCEEDED:
            self._sits.pop(execution.execution_id, None)
            workflow.receipt_ids.append(receipt.command_id)
            if self.seat_slot_allocator is not None:
                lease = self.seat_slot_allocator.lease(workflow.seat)
                if lease is not None and not lease.occupied:
                    self.seat_slot_allocator.release_reservation(
                        workflow.seat, execution.command.agent_id, workflow.session_id
                    )
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
                tuple(workflow.receipt_ids),
                station_id=workflow.seat,
                lease_id=workflow.session_id,
            )
        assert execution.command is not None
        workflow.receipt_ids.append(receipt.command_id)
        if workflow.stage == "approach_seat":
            workflow.stage = "align_seat"
            workflow.command_id = self._submit_stage(
                execution.command.agent_id,
                execution.execution_id,
                workflow.stage,
                NpcCommandKind.ALIGN_TO,
                {"yaw": workflow.yaw, "target_site": workflow.ingress_site},
                timeout_seconds=ACTION_RECIPES[ActionType.SIT].timeout_seconds,
            )
            return DriverResult(ExecutionStatus.RUNNING, workflow.stage, workflow.command_id)
        if workflow.stage == "align_seat":
            workflow.stage = "sit_transition"
            workflow.command_id = self._submit_stage(
                execution.command.agent_id,
                execution.execution_id,
                workflow.stage,
                NpcCommandKind.PLAY_ANIMATION,
                {
                    "clip": ACTION_RECIPES[ActionType.SIT].animation,
                    "completion_marker": ACTION_RECIPES[ActionType.SIT].completion_marker,
                    "target_site": (workflow.ingress_site if execution.command.parameters.get("_acceptance_stationary_sit") else workflow.sit_site),
                },
                timeout_seconds=ACTION_RECIPES[ActionType.SIT].timeout_seconds,
            )
            return DriverResult(ExecutionStatus.RUNNING, workflow.stage, workflow.command_id)
        verification_id = self._verify_seat(
            execution.command.agent_id, workflow.seat, "seated", workflow.sit_site
        )
        if verification_id is None:
            self._sits.pop(execution.execution_id, None)
            if self.seat_slot_allocator is not None:
                self.seat_slot_allocator.release_reservation(
                    workflow.seat, execution.command.agent_id, workflow.session_id
                )
            return DriverResult(
                ExecutionStatus.FAILED,
                "seat_verification",
                workflow.command_id,
                "seat_contact_verification_failed",
                tuple(workflow.receipt_ids),
                station_id=workflow.seat,
                lease_id=workflow.session_id,
            )
        workflow.receipt_ids.append(verification_id)
        if self.seat_slot_allocator is not None:
            self.seat_slot_allocator.occupy(
                workflow.seat,
                execution.command.agent_id,
                workflow.session_id,
                verification_id,
            )
        self._sits.pop(execution.execution_id, None)
        self._seated_agents[execution.command.agent_id] = workflow.seat
        return DriverResult(
            ExecutionStatus.SUCCEEDED,
            "seated",
            workflow.command_id,
            receipt_ids=tuple(workflow.receipt_ids),
            station_id=workflow.seat,
            lease_id=workflow.session_id,
            compatibility_mode=(
                "population_v3_seat_slot"
                if self.seat_slot_allocator is not None
                else "legacy_chair_adapter"
            ),
        )

    def _start_stand_up(self, execution: ActionExecution) -> DriverResult:
        assert execution.command is not None
        command = execution.command
        session_id = execution.execution_id
        if self.seat_slot_allocator is not None:
            lease = self.seat_slot_allocator.lease(str(command.target))
            if lease is None or lease.owner_id != command.agent_id or not lease.occupied:
                return DriverResult(
                    ExecutionStatus.FAILED, "prepare", error="seat_slot_not_occupied"
                )
            session_id = lease.session_id
        target_site = (self.seat_navigation_sites.get(str(command.target)) if command.parameters.get("_acceptance_stationary_sit") else self.location_sites.get(str(command.target)))
        if target_site is None:
            return DriverResult(ExecutionStatus.FAILED, "prepare", error="recipe_missing_site")
        recipe = ACTION_RECIPES[ActionType.STAND_UP]
        error = recipe.validate(
            target=command.target,
            location_sites={str(command.target): target_site},
            yaws=self.interaction_yaws,
            available_clips=self.available_clips,
        )
        if error is not None:
            return DriverResult(ExecutionStatus.FAILED, "prepare", error=error)
        handle = self._submit_stage(
            command.agent_id,
            execution.execution_id,
            "stand_transition",
            NpcCommandKind.PLAY_ANIMATION,
            {
                "clip": "stand_up",
                "completion_marker": "standing",
                "arrival_clip": "idle",
                "target_site": target_site,
            },
            timeout_seconds=recipe.timeout_seconds,
        )
        self._standing_up[execution.execution_id] = _StandWorkflow(
            command.agent_id,
            str(command.target),
            session_id,
            self.seat_navigation_sites.get(str(command.target)),
            "stand_transition",
            handle,
        )
        return DriverResult(ExecutionStatus.RUNNING, "stand_transition", handle)

    def _start_desk_work(self, execution: ActionExecution) -> DriverResult:
        """Play work in the seat established by the preceding sit receipt."""
        assert execution.command is not None
        command = execution.command
        seat = command.parameters.get(WORK_SESSION_SEAT_PARAMETER)
        duration = command.parameters.get(WORK_DURATION_SECONDS_PARAMETER)
        if not isinstance(seat, str) or seat not in self.location_sites:
            return DriverResult(
                ExecutionStatus.FAILED, "prepare", error="desk_work_missing_seat_site"
            )
        if self._seated_agents.get(command.agent_id) != seat:
            return DriverResult(
                ExecutionStatus.FAILED,
                "prepare",
                error="desk_work_requires_seat_receipt",
            )
        if (
            not isinstance(duration, (int, float))
            or isinstance(duration, bool)
            or not math.isfinite(duration)
            or duration <= 0
        ):
            return DriverResult(
                ExecutionStatus.FAILED, "prepare", error="desk_work_invalid_duration"
            )
        work_payload: dict[str, object] = {
            "clip": ACTION_RECIPES[ActionType.WORK].animation,
            "arrival_clip": "seated_idle",
        }
        if self.seat_slot_allocator is None:
            work_payload.update(
                {"duration": float(duration), "target_site": self.location_sites[seat]}
            )
        else:
            target_site = self.location_sites.get(str(command.target))
            if target_site is None:
                return DriverResult(
                    ExecutionStatus.FAILED,
                    "prepare",
                    error="desk_work_missing_workstation_site",
                )
            work_payload.update(
                {
                    "completion_marker": ACTION_RECIPES[
                        ActionType.USE_COMPUTER
                    ].completion_marker,
                    "target_site": target_site,
                }
            )
        handle = self._submit_stage(
            command.agent_id,
            execution.execution_id,
            "work",
            NpcCommandKind.PLAY_ANIMATION,
            work_payload,
            timeout_seconds=ACTION_RECIPES[ActionType.WORK].timeout_seconds,
        )
        return DriverResult(ExecutionStatus.RUNNING, "work", handle)

    def _verify_seat(
        self, agent_id: str, seat: str, phase: str, target_site: str
    ) -> str | None:
        if self.seat_slot_allocator is None:
            return f"legacy_seat:{agent_id}:{seat}:{phase}"
        if self.seat_verifier is None:
            return None
        receipt_id = self.seat_verifier(agent_id, seat, phase, target_site)
        return receipt_id if isinstance(receipt_id, str) and receipt_id else None

    def _start_handover(self, execution: ActionExecution) -> DriverResult:
        assert execution.command is not None
        previous = self._handover_terminal_results.get(execution.execution_id)
        if previous is not None:
            return previous
        command = execution.command
        object_name = str(command.parameters.get("object", ""))
        receiver = str(command.target or "")
        actor_mode = command.parameters.get("actor_mode", "npc_to_npc")
        preferred_station_id = command.parameters.get("preferred_station_id")
        if actor_mode != "npc_to_npc":
            return DriverResult(
                ExecutionStatus.FAILED,
                "prepare",
                error="phase_2c_npc_handover_only",
            )
        if not object_name:
            return DriverResult(
                ExecutionStatus.FAILED, "prepare", error="handover_missing_object"
            )
        lease = None
        transfer_site = None
        distance_min_m, distance_max_m, yaw_tolerance_rad = 0.45, 0.95, 0.30
        compatibility_mode = "population_v2_legacy_adapter"
        if self.interaction_station_allocator is not None:
            try:
                lease = self.interaction_station_allocator.acquire(
                    session_id=execution.execution_id,
                    kind="handover",
                    participants=(command.agent_id, receiver),
                    actor_kinds=("npc", "npc"),
                    object_id=object_name,
                    preferred_station_id=preferred_station_id,
                )
            except InteractionStationAllocationError as error:
                return DriverResult(ExecutionStatus.FAILED, "prepare", error=str(error))
            if lease.mode != "npc_to_npc":
                cleanup_evidence_id = (
                    f"cleanup:handover_actor_mode_rejected:{lease.lease_id}"
                )
                self.interaction_station_allocator.release_unconfirmed(
                    lease.lease_id,
                    cleanup_evidence_id=cleanup_evidence_id,
                    outcome="failed",
                )
                return DriverResult(
                    ExecutionStatus.FAILED,
                    "prepare",
                    error="phase_2c_npc_handover_only",
                    cleanup_evidence_id=cleanup_evidence_id,
                    station_id=lease.station_id,
                    lease_id=lease.lease_id,
                    compatibility_mode="population_v3_candidate",
                )
            station = self.interaction_station_allocator.catalog.station(lease.station_id)
            assignments = {item.role: item for item in lease.assignments}
            giver_site = assignments["giver"].site
            receiver_site = assignments["receiver"].site
            giver_yaw = station.role("giver").yaw
            receiver_yaw = station.role("receiver").yaw
            if giver_yaw is None or receiver_yaw is None:
                cleanup_evidence_id = f"cleanup:handover_yaw_missing:{lease.lease_id}"
                self.interaction_station_allocator.release_unconfirmed(
                    lease.lease_id,
                    cleanup_evidence_id=cleanup_evidence_id,
                    outcome="failed",
                )
                return DriverResult(
                    ExecutionStatus.FAILED,
                    "prepare",
                    error="handover_station_yaw_missing",
                    cleanup_evidence_id=cleanup_evidence_id,
                    station_id=lease.station_id,
                    lease_id=lease.lease_id,
                    compatibility_mode="population_v3_candidate",
                )
            assert station.distance_min_m is not None
            assert station.distance_max_m is not None
            assert station.yaw_tolerance_rad is not None
            assert station.transfer_site is not None
            distance_min_m = station.distance_min_m
            distance_max_m = station.distance_max_m
            yaw_tolerance_rad = station.yaw_tolerance_rad
            transfer_site = station.transfer_site
            compatibility_mode = "population_v3_candidate"
        else:
            recipe = ACTION_RECIPES[ActionType.HANDOVER]
            error = recipe.validate(
                target=receiver or None,
                location_sites=self.handover_sites,
                yaws=self.interaction_yaws,
                available_clips=self.available_clips,
            )
            if error is not None:
                return DriverResult(ExecutionStatus.FAILED, "prepare", error=error)
            role_sites = self.handover_role_sites.get((command.agent_id, receiver))
            if role_sites is None:
                return DriverResult(
                    ExecutionStatus.FAILED, "prepare", error="handover_missing_role_sites"
                )
            if not self._lease_interaction_sites(execution.execution_id, role_sites):
                return DriverResult(
                    ExecutionStatus.FAILED, "prepare", error="interaction_sites_busy"
                )
            giver_site, receiver_site = role_sites
            giver_yaw, receiver_yaw = self.handover_role_yaws.get(
                (command.agent_id, receiver),
                (
                    self.interaction_yaws[command.agent_id],
                    math.remainder(
                        self.interaction_yaws[command.agent_id] + math.pi,
                        2 * math.pi,
                    ),
                ),
            )
        timeout_seconds = self._timeout_for(ActionType.HANDOVER)
        issued_at = float(self.simulator.pull_status().time)
        session = self.interactions.start(
            "handover",
            (command.agent_id, receiver),
            object_name,
            (
                ("rendezvous", (command.agent_id, receiver)),
                ("aligned", (command.agent_id, receiver)),
                ("giver_ready", (command.agent_id,)),
                ("receiver_ready", (receiver,)),
                ("released", (command.agent_id,)),
                ("received", (receiver,)),
            ),
            deadline=issued_at + timeout_seconds,
            session_id=execution.execution_id,
        )
        workflow = _HandoverWorkflow(
            session_id=session.session_id,
            giver=command.agent_id,
            receiver=receiver,
            object_name=object_name,
            giver_site=giver_site,
            receiver_site=receiver_site,
            giver_yaw=giver_yaw,
            receiver_yaw=receiver_yaw,
            stage="rendezvous",
            command_ids={},
            lease=lease,
            transfer_site=transfer_site,
            distance_min_m=distance_min_m,
            distance_max_m=distance_max_m,
            yaw_tolerance_rad=yaw_tolerance_rad,
            compatibility_mode=compatibility_mode,
        )
        try:
            workflow.command_ids = self._submit_handover_stage(
                execution.execution_id,
                "rendezvous",
                (
                    (command.agent_id, NpcCommandKind.MOVE_TO, self._move_payload(giver_site)),
                    (receiver, NpcCommandKind.MOVE_TO, self._move_payload(receiver_site)),
                ),
                timeout_seconds=timeout_seconds,
            )
        except _ConversationCommandSubmitError as error:
            return self._handover_submit_failure(workflow, "rendezvous", error)
        self._handovers[execution.execution_id] = workflow
        return DriverResult(
            ExecutionStatus.RUNNING,
            "rendezvous",
            self._workflow_handle(workflow),
            station_id=None if lease is None else lease.station_id,
            lease_id=None if lease is None else lease.lease_id,
            compatibility_mode=compatibility_mode,
        )

    def _advance_handover(
        self,
        execution: ActionExecution,
        workflow: _HandoverWorkflow,
    ) -> DriverResult:
        receipts = {
            npc_id: self._receipts.get(command_id)
            for npc_id, command_id in workflow.command_ids.items()
        }
        failed = next(
            (
                receipt
                for receipt in receipts.values()
                if receipt is not None
                and receipt.status.terminal
                and receipt.status != CommandStatus.SUCCEEDED
            ),
            None,
        )
        if failed is not None:
            self._append_handover_receipts(
                workflow,
                tuple(
                    receipt.command_id
                    for receipt in receipts.values()
                    if receipt is not None and receipt.status.terminal
                ),
            )
            failure_error = failed.reason or failed.status.value
            terminal_outcome = {
                CommandStatus.CANCELLED: "cancelled",
                CommandStatus.TIMED_OUT: "timed_out",
            }.get(failed.status, "failed")
            if workflow.stage == "receive" and workflow.released:
                return self._start_handover_reconciliation(
                    execution.execution_id,
                    workflow,
                    terminal_outcome,
                    failure_error,
                )
            if workflow.stage == "rollback":
                cleanup_evidence_id = (
                    f"cleanup:handover_reconciliation_required:{failed.command_id}"
                )
                return self._terminal_handover(
                    execution.execution_id,
                    workflow,
                    self._execution_status(failed.status),
                    terminal_outcome,
                    failure_error,
                    cleanup_evidence_id=cleanup_evidence_id,
                )
            self._cancel_workflow_commands(workflow)
            return self._terminal_handover(
                execution.execution_id,
                workflow,
                self._execution_status(failed.status),
                terminal_outcome,
                failure_error,
                terminal_receipt_id=failed.command_id,
            )
        if not all(
            receipt is not None and receipt.status == CommandStatus.SUCCEEDED
            for receipt in receipts.values()
        ):
            return DriverResult(
                ExecutionStatus.RUNNING, workflow.stage, self._workflow_handle(workflow)
            )
        phase_receipt_ids = tuple(receipt.command_id for receipt in receipts.values())
        self._append_handover_receipts(workflow, phase_receipt_ids)
        timeout_seconds = self._timeout_for(ActionType.HANDOVER)
        if workflow.stage == "rendezvous":
            self.interactions.acknowledge(workflow.session_id, workflow.giver, "rendezvous")
            self.interactions.acknowledge(workflow.session_id, workflow.receiver, "rendezvous")
            workflow.stage = "aligned"
            try:
                workflow.command_ids = self._submit_handover_stage(
                    execution.execution_id,
                    "aligned",
                    (
                        (
                            workflow.giver,
                            NpcCommandKind.ALIGN_TO,
                            {"yaw": workflow.giver_yaw, "target_site": workflow.giver_site},
                        ),
                        (
                            workflow.receiver,
                            NpcCommandKind.ALIGN_TO,
                            {"yaw": workflow.receiver_yaw, "target_site": workflow.receiver_site},
                        ),
                    ),
                    timeout_seconds=timeout_seconds,
                )
            except _ConversationCommandSubmitError as error:
                return self._handover_submit_failure(workflow, "aligned", error)
        elif workflow.stage == "aligned":
            self.interactions.acknowledge(workflow.session_id, workflow.giver, "aligned")
            self.interactions.acknowledge(workflow.session_id, workflow.receiver, "aligned")
            workflow.stage = "giver_ready"
            try:
                workflow.command_ids = self._submit_handover_stage(
                    execution.execution_id,
                    "ready",
                    (
                        (
                            workflow.giver,
                            NpcCommandKind.PLAY_ANIMATION,
                            {
                                "clip": "give",
                                "completion_marker": "handover_ready",
                                "interaction_id": workflow.session_id,
                                "interaction_target": workflow.receiver,
                                "interaction_distance_min": workflow.distance_min_m,
                                "interaction_distance_max": workflow.distance_max_m,
                                "interaction_yaw_tolerance": workflow.yaw_tolerance_rad,
                                "arrival_clip": "idle",
                                "reveal_object": workflow.object_name,
                                "target_site": workflow.giver_site,
                                "position_tolerance": 0.12,
                            },
                        ),
                        (
                            workflow.receiver,
                            NpcCommandKind.PLAY_ANIMATION,
                            {
                                "clip": "receive",
                                "completion_marker": "handover_ready",
                                "interaction_id": workflow.session_id,
                                "interaction_target": workflow.giver,
                                "interaction_distance_min": workflow.distance_min_m,
                                "interaction_distance_max": workflow.distance_max_m,
                                "interaction_yaw_tolerance": workflow.yaw_tolerance_rad,
                                "arrival_clip": "idle",
                                "target_site": workflow.receiver_site,
                                "position_tolerance": 0.12,
                            },
                        ),
                    ),
                    timeout_seconds=timeout_seconds,
                )
            except _ConversationCommandSubmitError as error:
                return self._handover_submit_failure(workflow, "ready", error)
        elif workflow.stage == "giver_ready":
            self.interactions.acknowledge(workflow.session_id, workflow.giver, "giver_ready")
            self.interactions.acknowledge(workflow.session_id, workflow.receiver, "receiver_ready")
            workflow.stage = "release"
            detach_payload: dict[str, object] = {
                "object": workflow.object_name,
                "interaction_id": workflow.session_id,
            }
            if workflow.transfer_site is not None:
                detach_payload["site"] = workflow.transfer_site
            try:
                workflow.command_ids = self._submit_handover_stage(
                    execution.execution_id,
                    "release",
                    ((workflow.giver, NpcCommandKind.DETACH_OBJECT, detach_payload),),
                    timeout_seconds=timeout_seconds,
                )
            except _ConversationCommandSubmitError as error:
                return self._handover_submit_failure(workflow, "release", error)
        elif workflow.stage == "release":
            self.interactions.acknowledge(workflow.session_id, workflow.giver, "released")
            workflow.released = True
            workflow.stage = "receive"
            try:
                workflow.command_ids = self._submit_handover_stage(
                    execution.execution_id,
                    "receive",
                    (
                        (
                            workflow.receiver,
                            NpcCommandKind.ATTACH_OBJECT,
                            {"object": workflow.object_name, "interaction_id": workflow.session_id},
                        ),
                    ),
                    timeout_seconds=timeout_seconds,
                )
            except _ConversationCommandSubmitError as error:
                return self._start_handover_reconciliation(
                    execution.execution_id,
                    workflow,
                    "failed",
                    f"handover_receive_submit_failed:{error.cause}",
                )
        elif workflow.stage == "rollback":
            status = {
                "cancelled": ExecutionStatus.CANCELLED,
                "timed_out": ExecutionStatus.TIMED_OUT,
            }.get(workflow.terminal_outcome, ExecutionStatus.FAILED)
            return self._terminal_handover(
                execution.execution_id,
                workflow,
                status,
                workflow.terminal_outcome,
                workflow.failure_error,
                terminal_receipt_id=phase_receipt_ids[-1],
            )
        else:
            if workflow.lease is not None:
                observed_owner = self._observed_attachment_owner(workflow.object_name)
                if observed_owner != workflow.receiver:
                    cleanup_evidence_id = (
                        "cleanup:handover_reconciliation_required:"
                        f"{phase_receipt_ids[-1]}"
                    )
                    return self._terminal_handover(
                        execution.execution_id,
                        workflow,
                        ExecutionStatus.FAILED,
                        "failed",
                        "handover_owner_observation_failed",
                        cleanup_evidence_id=cleanup_evidence_id,
                    )
            session = self.interactions.acknowledge(
                workflow.session_id, workflow.receiver, "received"
            )
            if session.status.value == "succeeded":
                return self._terminal_handover(
                    execution.execution_id,
                    workflow,
                    ExecutionStatus.SUCCEEDED,
                    "succeeded",
                    None,
                    terminal_receipt_id=phase_receipt_ids[-1],
                )
            return self._terminal_handover(
                execution.execution_id,
                workflow,
                ExecutionStatus.FAILED,
                "failed",
                "handover_interaction_barrier_failed",
                terminal_receipt_id=phase_receipt_ids[-1],
            )
        return DriverResult(
            ExecutionStatus.RUNNING,
            workflow.stage,
            self._workflow_handle(workflow),
            receipt_ids=tuple(workflow.physical_receipt_ids),
            station_id=None if workflow.lease is None else workflow.lease.station_id,
            lease_id=None if workflow.lease is None else workflow.lease.lease_id,
            compatibility_mode=workflow.compatibility_mode,
        )

    def _submit_handover_stage(
        self,
        execution_id: str,
        stage: str,
        commands: tuple[tuple[str, NpcCommandKind, dict[str, object]], ...],
        *,
        timeout_seconds: float,
    ) -> dict[str, str]:
        submitted: dict[str, str] = {}
        try:
            for npc_id, kind, payload in commands:
                submitted[npc_id] = self._submit_stage(
                    npc_id,
                    execution_id,
                    f"{stage}_{npc_id}",
                    kind,
                    payload,
                    timeout_seconds=timeout_seconds,
                )
        except Exception as error:
            for npc_id, command_id in submitted.items():
                self.simulator.cancel_npc_command(npc_id, command_id)
            raise _ConversationCommandSubmitError(error, submitted) from error
        return submitted

    @staticmethod
    def _append_handover_receipts(
        workflow: _HandoverWorkflow, receipt_ids: tuple[str, ...]
    ) -> None:
        for receipt_id in receipt_ids:
            if receipt_id not in workflow.physical_receipt_ids:
                workflow.physical_receipt_ids.append(receipt_id)

    def _handover_submit_failure(
        self,
        workflow: _HandoverWorkflow,
        stage: str,
        error: _ConversationCommandSubmitError,
    ) -> DriverResult:
        if workflow.released:
            return self._start_handover_reconciliation(
                workflow.session_id,
                workflow,
                "failed",
                f"handover_{stage}_submit_failed:{error.cause}",
            )
        evidence_source = (
            next(iter(error.submitted.values()))
            if error.submitted
            else workflow.lease.lease_id
            if workflow.lease is not None
            else workflow.session_id
        )
        cleanup_evidence_id = (
            f"cleanup:handover_{stage}_submit_failed:{evidence_source}"
        )
        return self._terminal_handover(
            workflow.session_id,
            workflow,
            ExecutionStatus.FAILED,
            "failed",
            f"handover_command_submit_failed:{error.cause}",
            cleanup_evidence_id=cleanup_evidence_id,
        )

    def _start_handover_reconciliation(
        self,
        execution_id: str,
        workflow: _HandoverWorkflow,
        outcome: str,
        error: str,
    ) -> DriverResult:
        self._cancel_workflow_commands(workflow)
        workflow.stage = "rollback"
        workflow.failure_error = error
        workflow.terminal_outcome = outcome
        try:
            workflow.command_ids = self._submit_handover_stage(
                execution_id,
                "rollback",
                (
                    (
                        workflow.giver,
                        NpcCommandKind.ATTACH_OBJECT,
                        {
                            "object": workflow.object_name,
                            "interaction_id": workflow.session_id,
                        },
                    ),
                ),
                timeout_seconds=self._timeout_for(ActionType.HANDOVER),
            )
        except _ConversationCommandSubmitError as submit_error:
            evidence_source = (
                next(iter(submit_error.submitted.values()))
                if submit_error.submitted
                else workflow.lease.lease_id
                if workflow.lease is not None
                else workflow.session_id
            )
            cleanup_evidence_id = (
                f"cleanup:handover_reconciliation_required:{evidence_source}"
            )
            return self._terminal_handover(
                execution_id,
                workflow,
                ExecutionStatus.FAILED,
                outcome,
                f"handover_reconciliation_submit_failed:{submit_error.cause}",
                cleanup_evidence_id=cleanup_evidence_id,
            )
        self._handovers[execution_id] = workflow
        return DriverResult(
            ExecutionStatus.RUNNING,
            "rollback",
            self._workflow_handle(workflow),
            error,
            tuple(workflow.physical_receipt_ids),
            station_id=None if workflow.lease is None else workflow.lease.station_id,
            lease_id=None if workflow.lease is None else workflow.lease.lease_id,
            compatibility_mode=workflow.compatibility_mode,
        )

    def _terminal_handover(
        self,
        execution_id: str,
        workflow: _HandoverWorkflow,
        status: ExecutionStatus,
        outcome: str,
        error: str | None,
        *,
        terminal_receipt_id: str | None = None,
        cleanup_evidence_id: str | None = None,
    ) -> DriverResult:
        self._handovers.pop(execution_id, None)
        session = self.interactions.sessions.get(workflow.session_id)
        if session is not None and not session.status.terminal:
            if outcome == "cancelled":
                self.interactions.cancel(workflow.session_id, error or outcome)
            elif outcome != "succeeded":
                self.interactions.fail(workflow.session_id, error or outcome)
        if workflow.lease is not None and self.interaction_station_allocator is not None:
            if terminal_receipt_id is not None:
                self.interaction_station_allocator.release(
                    workflow.lease.lease_id,
                    terminal_receipt_id=terminal_receipt_id,
                    outcome=outcome,
                )
            else:
                assert cleanup_evidence_id is not None
                self.interaction_station_allocator.release_unconfirmed(
                    workflow.lease.lease_id,
                    cleanup_evidence_id=cleanup_evidence_id,
                    outcome=outcome,
                )
        else:
            self._release_interaction_sites(execution_id)
        result = DriverResult(
            status,
            "completed" if status == ExecutionStatus.SUCCEEDED else "terminal",
            terminal_receipt_id,
            error,
            tuple(workflow.physical_receipt_ids),
            cleanup_evidence_id,
            None if workflow.lease is None else workflow.lease.station_id,
            None if workflow.lease is None else workflow.lease.lease_id,
            workflow.compatibility_mode,
            workflow.production_evidence,
        )
        self._handover_terminal_results[execution_id] = result
        return result

    def _observed_attachment_owner(self, object_name: str) -> str | None:
        pull_states = getattr(self.simulator, "pull_npc_states", None)
        if not callable(pull_states):
            return None
        states = pull_states()
        owners = tuple(
            sorted(
                npc_id
                for npc_id, state in states.items()
                if object_name in tuple(getattr(state, "held_objects", ()))
            )
        )
        return owners[0] if len(owners) == 1 else None

    def prepare_conversation(self, session: ConversationSession) -> DriverResult:
        """Physically approach and align every participant before turns may commit."""
        if len(session.participants) != 2:
            return DriverResult(
                ExecutionStatus.FAILED, "prepare", error="two_participants_required"
        )
        first, second = session.participants
        lease = None
        distance_min_m, distance_max_m, yaw_tolerance_rad = 0.45, 0.95, 0.30
        if self.interaction_station_allocator is not None:
            try:
                lease = self.interaction_station_allocator.acquire(
                    session_id=session.session_id,
                    kind="conversation",
                    participants=(first, second),
                    actor_kinds=("npc", "npc"),
                    preferred_station_id=session.preferred_station_id,
                )
            except InteractionStationAllocationError as error:
                return DriverResult(ExecutionStatus.FAILED, "prepare", error=str(error))
            sites = {
                assignment.participant_id: assignment.site
                for assignment in lease.assignments
            }
            station = self.interaction_station_allocator.catalog.station(lease.station_id)
            assert station.distance_min_m is not None
            assert station.distance_max_m is not None
            assert station.yaw_tolerance_rad is not None
            distance_min_m = station.distance_min_m
            distance_max_m = station.distance_max_m
            yaw_tolerance_rad = station.yaw_tolerance_rad
            for assignment in lease.assignments:
                role = station.role(assignment.role)
                if role.yaw is not None:
                    self.conversation_site_yaws[assignment.site] = role.yaw
            session.station_id = lease.station_id
            session.lease_id = lease.lease_id
        else:
            sites = self._conversation_sites(first, second)
            if sites is None:
                return DriverResult(
                    ExecutionStatus.FAILED, "prepare", error="conversation_sites_missing"
                )
            if not self._lease_interaction_sites(session.session_id, tuple(sites.values())):
                return DriverResult(
                    ExecutionStatus.FAILED, "prepare", error="interaction_sites_busy"
                )
        commands = tuple(
            (npc_id, NpcCommandKind.MOVE_TO, self._move_payload(site))
            for npc_id, site in sites.items()
        )
        try:
            command_ids = self._submit_conversation_commands(
                session.session_id,
                "conversation_approach",
                commands,
                timeout_seconds=30.0,
            )
        except _ConversationCommandSubmitError as error:
            cleanup_evidence_id = (
                "cleanup:conversation_prepare_submit_failed:"
                + (
                    next(iter(error.submitted.values()))
                    if error.submitted
                    else lease.lease_id
                    if lease is not None
                    else session.session_id
                )
            )
            if lease is not None and self.interaction_station_allocator is not None:
                self.interaction_station_allocator.release_unconfirmed(
                    lease.lease_id,
                    cleanup_evidence_id=cleanup_evidence_id,
                    outcome="failed",
                )
            else:
                self._release_interaction_sites(session.session_id)
            return DriverResult(
                ExecutionStatus.FAILED,
                "prepare",
                error=f"conversation_command_submit_failed:{error.cause}",
                cleanup_evidence_id=cleanup_evidence_id,
            )
        self._conversations[session.session_id] = _ConversationWorkflow(
            session.session_id,
            session.participants,
            sites,
            "approach",
            command_ids,
            lease,
            distance_min_m,
            distance_max_m,
            yaw_tolerance_rad,
        )
        return DriverResult(ExecutionStatus.RUNNING, "approach", next(iter(command_ids.values())))

    def poll_conversation(self, session_id: str) -> DriverResult:
        workflow = self._conversations.get(session_id)
        if workflow is None:
            return DriverResult(ExecutionStatus.FAILED, "terminal", error="unknown_conversation")
        for receipt in self.simulator.pull_npc_receipts():
            self._receipts[receipt.command_id] = receipt
        receipts = [
            self._receipts.get(workflow.command_ids[participant])
            for participant in workflow.command_ids
        ]
        if any(
            receipt is None or receipt.status in {CommandStatus.ACCEPTED, CommandStatus.RUNNING}
            for receipt in receipts
        ):
            return DriverResult(
                ExecutionStatus.RUNNING, workflow.stage, next(iter(workflow.command_ids.values()))
            )
        if any(
            receipt.status != CommandStatus.SUCCEEDED for receipt in receipts if receipt is not None
        ):
            failed_receipt = next(
                receipt
                for receipt in receipts
                if receipt is not None and receipt.status != CommandStatus.SUCCEEDED
            )
            self._terminal_conversation_workflow(
                workflow,
                failed_receipt.status.value,
                failed_receipt.command_id,
            )
            failed_status = {
                CommandStatus.CANCELLED: ExecutionStatus.CANCELLED,
                CommandStatus.TIMED_OUT: ExecutionStatus.TIMED_OUT,
            }.get(failed_receipt.status, ExecutionStatus.FAILED)
            return DriverResult(
                failed_status,
                "terminal",
                failed_receipt.command_id,
                "conversation_physical_receipt_failed",
                tuple(receipt.command_id for receipt in receipts if receipt is not None),
            )
        phase_receipt_ids = tuple(receipt.command_id for receipt in receipts)
        workflow.last_physical_receipt_ids = phase_receipt_ids
        if workflow.stage == "approach":
            workflow.stage = "align"
            try:
                workflow.command_ids = self._submit_conversation_commands(
                    session_id,
                    "conversation_align",
                    tuple(
                    (
                        npc_id,
                        NpcCommandKind.ALIGN_TO,
                        {
                            "yaw": self.conversation_site_yaws.get(
                                site, self.interaction_yaws.get(npc_id, 0.0)
                            ),
                            "target_site": site,
                        },
                    )
                    for npc_id, site in workflow.sites.items()
                    ),
                    timeout_seconds=15.0,
                )
            except _ConversationCommandSubmitError as error:
                return self._conversation_submit_failure(
                    workflow, "align", error, phase_receipt_ids
                )
            return DriverResult(
                ExecutionStatus.RUNNING,
                "align",
                next(iter(workflow.command_ids.values())),
                receipt_ids=phase_receipt_ids,
            )
        if workflow.stage == "align":
            workflow.stage = "alignment_gate"
            try:
                workflow.command_ids = self._submit_conversation_commands(
                    session_id,
                    "conversation_alignment_gate",
                    tuple(
                    (
                        npc_id,
                        NpcCommandKind.INTERACTION_CUE,
                        {
                            "clip": "idle",
                            "duration": 0.05,
                            "target_site": workflow.sites[npc_id],
                            "interaction_target": next(
                                participant
                                for participant in workflow.participants
                                if participant != npc_id
                            ),
                            "interaction_distance_min": workflow.distance_min_m,
                            "interaction_distance_max": workflow.distance_max_m,
                            "interaction_yaw_tolerance": workflow.yaw_tolerance_rad,
                        },
                    )
                    for npc_id in workflow.participants
                    ),
                    timeout_seconds=15.0,
                )
            except _ConversationCommandSubmitError as error:
                return self._conversation_submit_failure(
                    workflow, "alignment_gate", error, phase_receipt_ids
                )
            return DriverResult(
                ExecutionStatus.RUNNING,
                "alignment_gate",
                next(iter(workflow.command_ids.values())),
                receipt_ids=phase_receipt_ids,
            )
        # A completed talk receipt leaves the participants physically aligned
        # for the next round-robin turn.  Restore the prepared stage rather
        # than treating a two-turn conversation as a one-turn-only workflow.
        if workflow.stage == "turn":
            workflow.stage = "ready"
        elif workflow.stage == "alignment_gate":
            workflow.stage = "ready"
        return DriverResult(
            ExecutionStatus.SUCCEEDED,
            "aligned",
            next(iter(workflow.command_ids.values())),
            receipt_ids=phase_receipt_ids,
        )

    def command_receipts(self) -> tuple[NpcCommandReceipt, ...]:
        """Return the receipt evidence retained by this driver instance."""
        return tuple(self._receipts.values())

    def play_turn(self, session: ConversationSession, turn: DialogueTurn) -> DriverResult:
        workflow = self._conversations.get(session.session_id)
        if workflow is None or workflow.stage != "ready":
            return DriverResult(ExecutionStatus.FAILED, "turn", error="conversation_not_aligned")
        commands = (
                (
                    turn.speaker,
                    NpcCommandKind.PLAY_ANIMATION,
                    {
                        "clip": "talk",
                        "completion_marker": "talk_cycle",
                        "target_site": workflow.sites[turn.speaker],
                        "arrival_clip": "idle",
                        "interaction_target": turn.listener,
                        "interaction_distance_min": workflow.distance_min_m,
                        "interaction_distance_max": workflow.distance_max_m,
                        "interaction_yaw_tolerance": workflow.yaw_tolerance_rad,
                    },
                ),
                (
                    turn.listener,
                    NpcCommandKind.INTERACTION_CUE,
                    {
                        "clip": "idle",
                        "duration": 0.05,
                        "target_site": workflow.sites[turn.listener],
                        "interaction_target": turn.speaker,
                        "interaction_distance_min": workflow.distance_min_m,
                        "interaction_distance_max": workflow.distance_max_m,
                        "interaction_yaw_tolerance": workflow.yaw_tolerance_rad,
                    },
                ),
        )
        try:
            command_ids = self._submit_conversation_commands(
                f"{session.session_id}:{turn.turn_id}",
                "conversation_turn",
                commands,
                timeout_seconds=30.0,
            )
        except _ConversationCommandSubmitError as error:
            cleanup_evidence_id = (
                "cleanup:conversation_turn_submit_failed:"
                + (
                    next(iter(error.submitted.values()))
                    if error.submitted
                    else workflow.lease.lease_id
                    if workflow.lease is not None
                    else session.session_id
                )
            )
            self._terminal_conversation_workflow(
                workflow,
                "failed",
                None,
                cleanup_evidence_id=cleanup_evidence_id,
            )
            return DriverResult(
                ExecutionStatus.FAILED,
                "talk",
                error=f"conversation_command_submit_failed:{error.cause}",
                cleanup_evidence_id=cleanup_evidence_id,
            )
        workflow.stage = "turn"
        workflow.command_ids = command_ids
        return DriverResult(
            ExecutionStatus.RUNNING, "talk", command_ids[turn.speaker]
        )

    def cancel_conversation(self, session_id: str, reason: str) -> DriverResult:
        return self.terminate_conversation(session_id, "cancelled", reason)

    def terminate_conversation(
        self, session_id: str, outcome: str, reason: str
    ) -> DriverResult:
        workflow = self._conversations.get(session_id)
        if workflow is not None:
            for npc_id, command_id in workflow.command_ids.items():
                self.simulator.cancel_npc_command(npc_id, command_id)
            receipt_ids = workflow.last_physical_receipt_ids
            # The transport exposes no cancellation ack.  Prior phase receipts
            # stay in the trace but cannot confirm this terminal transition.
            cleanup_evidence_id = (
                f"cleanup:conversation_{outcome}_unconfirmed:"
                f"{next(iter(workflow.command_ids.values()))}"
            )
            self._terminal_conversation_workflow(
                workflow,
                outcome,
                None,
                cleanup_evidence_id=cleanup_evidence_id,
            )
            status = {
                "cancelled": ExecutionStatus.CANCELLED,
                "timed_out": ExecutionStatus.TIMED_OUT,
            }.get(outcome, ExecutionStatus.FAILED)
            return DriverResult(
                status,
                "cleanup_unconfirmed",
                None,
                reason,
                receipt_ids,
                cleanup_evidence_id,
            )
        return DriverResult(ExecutionStatus.CANCELLED, "cancelled", error=reason)

    def finish_conversation(self, session_id: str, outcome: str) -> DriverResult:
        workflow = self._conversations.get(session_id)
        if workflow is None:
            return DriverResult(ExecutionStatus.FAILED, "terminal", error="unknown_conversation")
        receipt_ids = workflow.last_physical_receipt_ids
        if not receipt_ids:
            return DriverResult(
                ExecutionStatus.FAILED,
                "terminal",
                error="conversation_terminal_physical_receipt_missing",
            )
        self._terminal_conversation_workflow(workflow, outcome, receipt_ids[-1])
        return DriverResult(
            ExecutionStatus.SUCCEEDED,
            "terminal",
            receipt_ids[-1],
            receipt_ids=receipt_ids,
        )

    def _submit_conversation_commands(
        self,
        execution_id: str,
        stage: str,
        commands: tuple[tuple[str, NpcCommandKind, dict[str, object]], ...],
        *,
        timeout_seconds: float,
    ) -> dict[str, str]:
        submitted: dict[str, str] = {}
        try:
            for npc_id, kind, payload in commands:
                submitted[npc_id] = self._submit_stage(
                    npc_id,
                    execution_id,
                    f"{stage}_{npc_id}",
                    kind,
                    payload,
                    timeout_seconds=timeout_seconds,
                )
        except Exception as error:
            for npc_id, command_id in submitted.items():
                self.simulator.cancel_npc_command(npc_id, command_id)
            raise _ConversationCommandSubmitError(error, submitted) from error
        return submitted

    def _conversation_submit_failure(
        self,
        workflow: _ConversationWorkflow,
        stage: str,
        error: _ConversationCommandSubmitError,
        prior_receipt_ids: tuple[str, ...],
    ) -> DriverResult:
        evidence_source = (
            next(iter(error.submitted.values()))
            if error.submitted
            else workflow.lease.lease_id
            if workflow.lease is not None
            else workflow.session_id
        )
        cleanup_evidence_id = (
            f"cleanup:conversation_{stage}_submit_failed:{evidence_source}"
        )
        self._terminal_conversation_workflow(
            workflow,
            "failed",
            None,
            cleanup_evidence_id=cleanup_evidence_id,
        )
        return DriverResult(
            ExecutionStatus.FAILED,
            stage,
            error=f"conversation_command_submit_failed:{error.cause}",
            receipt_ids=prior_receipt_ids,
            cleanup_evidence_id=cleanup_evidence_id,
        )

    def _terminal_conversation_workflow(
        self,
        workflow: _ConversationWorkflow,
        outcome: str,
        terminal_receipt_id: str | None,
        *,
        cleanup_evidence_id: str | None = None,
    ) -> None:
        if workflow.lease is not None and self.interaction_station_allocator is not None:
            if terminal_receipt_id is not None:
                self.interaction_station_allocator.release(
                    workflow.lease.lease_id,
                    terminal_receipt_id=terminal_receipt_id,
                    outcome=outcome,
                )
            else:
                assert cleanup_evidence_id is not None
                self.interaction_station_allocator.release_unconfirmed(
                    workflow.lease.lease_id,
                    cleanup_evidence_id=cleanup_evidence_id,
                    outcome=outcome,
                )
        else:
            self._release_interaction_sites(workflow.session_id)
        self._conversations.pop(workflow.session_id, None)

    def _conversation_sites(self, first: str, second: str) -> dict[str, str] | None:
        pair = self.conversation_role_sites.get((first, second))
        if pair is not None:
            return {first: pair[0], second: pair[1]}
        pair = self.handover_role_sites.get((first, second))
        if pair is not None:
            return {first: pair[0], second: pair[1]}
        if first in self.cue_sites and second in self.cue_sites:
            return {first: self.cue_sites[first], second: self.cue_sites[second]}
        return None

    def _submit_stage(
        self,
        npc_id: str,
        execution_id: str,
        stage: str,
        kind: NpcCommandKind,
        payload: dict[str, object],
        *,
        timeout_seconds: float | None = None,
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
            deadline=issued_at
            + (self.timeout_seconds if timeout_seconds is None else timeout_seconds),
        )
        handle = self.simulator.submit_npc_command(command)
        if kind == NpcCommandKind.MOVE_TO:
            self._remember_movement(handle, npc_id, str(payload["site"]))
        return handle

    def cancel(self, execution: ActionExecution, reason: str) -> DriverResult:
        self._settle_pending_location_slot(execution.execution_id, succeeded=False)
        self._settle_location_slot_movement(execution.driver_handle, succeeded=False)
        workflow = self._handovers.get(execution.execution_id)
        cancelled_workflow = workflow is not None
        if workflow is not None:
            if workflow.released:
                return self._start_handover_reconciliation(
                    execution.execution_id,
                    workflow,
                    "cancelled",
                    reason,
                )
            self._cancel_workflow_commands(workflow)
            cleanup_evidence_id = (
                "cleanup:handover_cancelled_unconfirmed:"
                f"{self._workflow_handle(workflow)}"
            )
            return self._terminal_handover(
                execution.execution_id,
                workflow,
                ExecutionStatus.CANCELLED,
                "cancelled",
                reason,
                cleanup_evidence_id=cleanup_evidence_id,
            )
        sit = self._sits.pop(execution.execution_id, None)
        if sit is not None and execution.command is not None:
            self.simulator.cancel_npc_command(execution.command.agent_id, sit.command_id)
            if self.seat_slot_allocator is not None:
                lease = self.seat_slot_allocator.lease(sit.seat)
                if lease is not None and not lease.occupied:
                    self.seat_slot_allocator.release_reservation(
                        sit.seat, execution.command.agent_id, sit.session_id
                    )
            cancelled_workflow = True
        stood = self._standing_up.pop(execution.execution_id, None)
        if stood is not None:
            self.simulator.cancel_npc_command(stood.agent_id, stood.command_id)
            cancelled_workflow = True
        object_workflow = self._objects.pop(execution.execution_id, None)
        if object_workflow is not None and execution.command is not None:
            self.simulator.cancel_npc_command(
                execution.command.agent_id, object_workflow.command_id
            )
            cancelled_workflow = True
        recipe_workflow = self._recipes.pop(execution.execution_id, None)
        if recipe_workflow is not None and execution.command is not None:
            self.simulator.cancel_npc_command(
                execution.command.agent_id, recipe_workflow.command_id
            )
            cancelled_workflow = True
        robot_request = self._robot_requests.pop(execution.execution_id, None)
        if robot_request is not None and execution.command is not None:
            self.simulator.cancel_npc_command(execution.command.agent_id, robot_request.command_id)
            cancelled_workflow = True
        if (
            not cancelled_workflow
            and execution.driver_handle is not None
            and execution.command is not None
        ):
            self.simulator.cancel_npc_command(execution.command.agent_id, execution.driver_handle)
        return DriverResult(
            ExecutionStatus.CANCELLED,
            "cancelled",
            execution.driver_handle,
            reason,
        )
