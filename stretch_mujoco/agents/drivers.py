"""Narrow execution drivers separating action intent from physical completion."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

from stretch_mujoco.npc.protocol import CommandStatus, NpcCommand, NpcCommandKind, NpcCommandReceipt
from stretch_mujoco.npc.trajectory_profile import NpcTrajectoryProfile

from .actions import ActionExecution, ActionType, ExecutionStatus
from .conversation import ConversationSession, DialogueTurn
from .action_recipes import ACTION_RECIPES, ActionRecipe
from .desk_work import WORK_DURATION_SECONDS_PARAMETER, WORK_SESSION_SEAT_PARAMETER
from .interactions import InteractionCoordinator


EMBODIED_ACTION_REQUIRED_CLIPS: dict[ActionType, frozenset[str]] = {
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
    released: bool = False
    failure_error: str | None = None


@dataclass
class _SitWorkflow:
    seat: str
    yaw: float
    stage: str
    command_id: str


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
        cue_sites: dict[str, str] | None = None,
        seat_yaws: dict[str, float] | None = None,
        seat_navigation_sites: dict[str, str] | None = None,
        interaction_yaws: dict[str, float] | None = None,
        robot_request_sites: dict[str, str] | None = None,
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
        self.cue_sites = dict(cue_sites or {})
        self.seat_yaws = dict(seat_yaws or {})
        self.seat_navigation_sites = dict(seat_navigation_sites or {})
        self.interaction_yaws = dict(interaction_yaws or {})
        self.robot_request_sites = dict(robot_request_sites or {})
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
        self.timeout_seconds = timeout_seconds
        self._sequences: dict[str, int] = {}
        self._receipts: dict[str, NpcCommandReceipt] = {}
        self.interactions = InteractionCoordinator()
        self._handovers: dict[str, _HandoverWorkflow] = {}
        self._sits: dict[str, _SitWorkflow] = {}
        self._objects: dict[str, _ObjectWorkflow] = {}
        self._recipes: dict[str, _RecipeWorkflow] = {}
        self._conversations: dict[str, _ConversationWorkflow] = {}
        self._robot_requests: dict[str, _RobotRequestWorkflow] = {}
        self._seated_agents: dict[str, str] = {}
        self._standing_up: dict[str, tuple[str, str]] = {}
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
        handle = self.simulator.submit_npc_command(npc_command)
        if kind == NpcCommandKind.MOVE_TO:
            self._remember_movement(handle, command.agent_id, str(payload["site"]))
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
            if session.status.value == "timed_out":
                self._handovers.pop(execution.execution_id, None)
                self._cancel_workflow_commands(workflow)
                self._release_interaction_sites(execution.execution_id)
                return DriverResult(
                    ExecutionStatus.TIMED_OUT,
                    "terminal",
                    self._workflow_handle(workflow),
                    session.error,
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
            stood = self._standing_up.pop(execution.execution_id, None)
            if stood is not None and self._seated_agents.get(stood[0]) == stood[1]:
                self._seated_agents.pop(stood[0], None)
            return DriverResult(ExecutionStatus.SUCCEEDED, "arrived", execution.driver_handle)
        status = self._execution_status(receipt.status)
        return DriverResult(status, "terminal", execution.driver_handle, receipt.reason)

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
            site = self.seat_navigation_sites.get(target) or self.location_sites.get(target)
            if site is None:
                raise ValueError(f"No NPC site configured for location '{command.target}'")
            return (
                NpcCommandKind.MOVE_TO,
                self._move_payload(site, agent_id=command.agent_id, action=command.action),
                "navigate",
            )
        raise ValueError(f"Unsupported embodied action '{command.action.value}'")

    def _timeout_for(self, action: ActionType) -> float:
        configured = ACTION_RECIPES[action].timeout_seconds
        return self.timeout_seconds if configured is None else configured

    def _move_payload(
        self,
        site: str,
        *,
        agent_id: str | None = None,
        action: ActionType | None = None,
    ) -> dict[str, object]:
        """Every embodied approach has a named target and one local replan."""
        payload: dict[str, object] = {"site": site, "arrival_clip": "idle", "max_replans": 1}
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
        if site is None:
            return DriverResult(
                ExecutionStatus.FAILED, "prepare", error=f"No seat site for '{seat}'"
            )
        if yaw is None:
            return DriverResult(
                ExecutionStatus.FAILED, "prepare", error=f"No seat yaw for '{seat}'"
            )
        payload = self._move_payload(site, agent_id=command.agent_id, action=command.action)
        navigation_site = self.seat_navigation_sites.get(seat)
        if navigation_site is not None:
            payload["navigation_site"] = navigation_site
        handle = self._submit_stage(
            command.agent_id,
            execution.execution_id,
            "approach_seat",
            NpcCommandKind.MOVE_TO,
            payload,
            timeout_seconds=ACTION_RECIPES[ActionType.SIT].timeout_seconds,
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
                {"yaw": workflow.yaw, "target_site": self.location_sites[workflow.seat]},
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
                    "target_site": self.location_sites[workflow.seat],
                },
                timeout_seconds=ACTION_RECIPES[ActionType.SIT].timeout_seconds,
            )
            return DriverResult(ExecutionStatus.RUNNING, workflow.stage, workflow.command_id)
        self._sits.pop(execution.execution_id, None)
        self._seated_agents[execution.command.agent_id] = workflow.seat
        return DriverResult(ExecutionStatus.SUCCEEDED, "seated", workflow.command_id)

    def _start_stand_up(self, execution: ActionExecution) -> DriverResult:
        assert execution.command is not None
        command = execution.command
        target_site = self.location_sites.get(str(command.target))
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
        self._standing_up[execution.execution_id] = (command.agent_id, str(command.target))
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
        handle = self._submit_stage(
            command.agent_id,
            execution.execution_id,
            "work",
            NpcCommandKind.PLAY_ANIMATION,
            {
                "clip": ACTION_RECIPES[ActionType.WORK].animation,
                "duration": float(duration),
                "arrival_clip": "seated_idle",
                "target_site": self.location_sites[seat],
            },
            timeout_seconds=ACTION_RECIPES[ActionType.WORK].timeout_seconds,
        )
        return DriverResult(ExecutionStatus.RUNNING, "work", handle)

    def _start_handover(self, execution: ActionExecution) -> DriverResult:
        assert execution.command is not None
        command = execution.command
        object_name = str(command.parameters.get("object", ""))
        receiver = str(command.target or "")
        recipe = ACTION_RECIPES[ActionType.HANDOVER]
        error = recipe.validate(
            target=receiver or None,
            location_sites=self.handover_sites,
            yaws=self.interaction_yaws,
            available_clips=self.available_clips,
        )
        if not object_name or error is not None:
            return DriverResult(
                ExecutionStatus.FAILED, "prepare", error=error or "handover_missing_object"
            )
        role_sites = self.handover_role_sites.get((command.agent_id, receiver))
        if role_sites is None:
            return DriverResult(
                ExecutionStatus.FAILED, "prepare", error="handover_missing_role_sites"
            )
        if not self._lease_interaction_sites(execution.execution_id, role_sites):
            return DriverResult(ExecutionStatus.FAILED, "prepare", error="interaction_sites_busy")
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
        giver_site, receiver_site = role_sites
        giver_yaw, receiver_yaw = self.handover_role_yaws.get(
            (command.agent_id, receiver),
            (
                self.interaction_yaws[command.agent_id],
                math.remainder(self.interaction_yaws[command.agent_id] + math.pi, 2 * math.pi),
            ),
        )
        command_ids = self._submit_handover_stage(
            execution.execution_id,
            "rendezvous",
            (
                (command.agent_id, NpcCommandKind.MOVE_TO, self._move_payload(giver_site)),
                (receiver, NpcCommandKind.MOVE_TO, self._move_payload(receiver_site)),
            ),
            timeout_seconds=timeout_seconds,
        )
        self._handovers[execution.execution_id] = _HandoverWorkflow(
            session.session_id,
            command.agent_id,
            receiver,
            object_name,
            giver_site,
            receiver_site,
            giver_yaw,
            receiver_yaw,
            "rendezvous",
            command_ids,
        )
        return DriverResult(
            ExecutionStatus.RUNNING,
            "rendezvous",
            self._workflow_handle(self._handovers[execution.execution_id]),
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
            if workflow.stage == "receive" and workflow.released:
                self.interactions.fail(workflow.session_id, failed.reason or failed.status.value)
                workflow.failure_error = failed.reason or failed.status.value
                workflow.stage = "rollback"
                workflow.command_ids = self._submit_handover_stage(
                    execution.execution_id,
                    "rollback",
                    (
                        (
                            workflow.giver,
                            NpcCommandKind.ATTACH_OBJECT,
                            {"object": workflow.object_name, "interaction_id": workflow.session_id},
                        ),
                    ),
                    timeout_seconds=self._timeout_for(ActionType.HANDOVER),
                )
                return DriverResult(
                    ExecutionStatus.RUNNING, workflow.stage, self._workflow_handle(workflow)
                )
            self.interactions.fail(workflow.session_id, failed.reason or failed.status.value)
            self._cancel_workflow_commands(workflow)
            self._handovers.pop(execution.execution_id, None)
            self._release_interaction_sites(execution.execution_id)
            return DriverResult(
                self._execution_status(failed.status),
                "terminal",
                self._workflow_handle(workflow),
                failed.reason or failed.status.value,
            )
        if not all(
            receipt is not None and receipt.status == CommandStatus.SUCCEEDED
            for receipt in receipts.values()
        ):
            return DriverResult(
                ExecutionStatus.RUNNING, workflow.stage, self._workflow_handle(workflow)
            )
        timeout_seconds = self._timeout_for(ActionType.HANDOVER)
        if workflow.stage == "rendezvous":
            self.interactions.acknowledge(workflow.session_id, workflow.giver, "rendezvous")
            self.interactions.acknowledge(workflow.session_id, workflow.receiver, "rendezvous")
            workflow.stage = "aligned"
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
        elif workflow.stage == "aligned":
            self.interactions.acknowledge(workflow.session_id, workflow.giver, "aligned")
            self.interactions.acknowledge(workflow.session_id, workflow.receiver, "aligned")
            workflow.stage = "giver_ready"
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
                            "interaction_distance_min": 0.45,
                            "interaction_distance_max": 0.95,
                            "interaction_yaw_tolerance": 0.30,
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
                            "interaction_distance_min": 0.45,
                            "interaction_distance_max": 0.95,
                            "interaction_yaw_tolerance": 0.30,
                            "arrival_clip": "idle",
                            "target_site": workflow.receiver_site,
                            "position_tolerance": 0.12,
                        },
                    ),
                ),
                timeout_seconds=timeout_seconds,
            )
        elif workflow.stage == "giver_ready":
            self.interactions.acknowledge(workflow.session_id, workflow.giver, "giver_ready")
            self.interactions.acknowledge(workflow.session_id, workflow.receiver, "receiver_ready")
            workflow.stage = "release"
            workflow.command_ids = self._submit_handover_stage(
                execution.execution_id,
                "release",
                (
                    (
                        workflow.giver,
                        NpcCommandKind.DETACH_OBJECT,
                        {"object": workflow.object_name, "interaction_id": workflow.session_id},
                    ),
                ),
                timeout_seconds=timeout_seconds,
            )
        elif workflow.stage == "release":
            self.interactions.acknowledge(workflow.session_id, workflow.giver, "released")
            workflow.released = True
            workflow.stage = "receive"
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
        elif workflow.stage == "rollback":
            self._handovers.pop(execution.execution_id, None)
            self._release_interaction_sites(execution.execution_id)
            return DriverResult(
                ExecutionStatus.FAILED,
                "terminal",
                self._workflow_handle(workflow),
                workflow.failure_error,
            )
        else:
            session = self.interactions.acknowledge(
                workflow.session_id, workflow.receiver, "received"
            )
            self._handovers.pop(execution.execution_id, None)
            self._release_interaction_sites(execution.execution_id)
            if session.status.value == "succeeded":
                return DriverResult(
                    ExecutionStatus.SUCCEEDED, "completed", self._workflow_handle(workflow)
                )
            return DriverResult(ExecutionStatus.FAILED, "terminal", self._workflow_handle(workflow))
        return DriverResult(
            ExecutionStatus.RUNNING, workflow.stage, self._workflow_handle(workflow)
        )

    def _submit_handover_stage(
        self,
        execution_id: str,
        stage: str,
        commands: tuple[tuple[str, NpcCommandKind, dict[str, object]], ...],
        *,
        timeout_seconds: float,
    ) -> dict[str, str]:
        return {
            npc_id: self._submit_stage(
                npc_id,
                execution_id,
                f"{stage}_{npc_id}",
                kind,
                payload,
                timeout_seconds=timeout_seconds,
            )
            for npc_id, kind, payload in commands
        }

    def prepare_conversation(self, session: ConversationSession) -> DriverResult:
        """Physically approach and align every participant before turns may commit."""
        if len(session.participants) != 2:
            return DriverResult(
                ExecutionStatus.FAILED, "prepare", error="two_participants_required"
            )
        first, second = session.participants
        sites = self._conversation_sites(first, second)
        if sites is None:
            return DriverResult(
                ExecutionStatus.FAILED, "prepare", error="conversation_sites_missing"
            )
        if not self._lease_interaction_sites(session.session_id, tuple(sites.values())):
            return DriverResult(ExecutionStatus.FAILED, "prepare", error="interaction_sites_busy")
        commands = tuple(
            (npc_id, NpcCommandKind.MOVE_TO, self._move_payload(site))
            for npc_id, site in sites.items()
        )
        command_ids = self._submit_handover_stage(
            session.session_id, "conversation_approach", commands, timeout_seconds=30.0
        )
        self._conversations[session.session_id] = _ConversationWorkflow(
            session.session_id, session.participants, sites, "approach", command_ids
        )
        return DriverResult(ExecutionStatus.RUNNING, "approach", next(iter(command_ids.values())))

    def poll_conversation(self, session_id: str) -> DriverResult:
        workflow = self._conversations.get(session_id)
        if workflow is None:
            return DriverResult(ExecutionStatus.FAILED, "terminal", error="unknown_conversation")
        for receipt in self.simulator.pull_npc_receipts():
            self._receipts[receipt.command_id] = receipt
        receipts = [self._receipts.get(command_id) for command_id in workflow.command_ids.values()]
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
            self._conversations.pop(session_id, None)
            self._release_interaction_sites(session_id)
            return DriverResult(
                ExecutionStatus.FAILED, "terminal", error="conversation_physical_receipt_failed"
            )
        if workflow.stage == "approach":
            workflow.stage = "align"
            workflow.command_ids = self._submit_handover_stage(
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
            return DriverResult(
                ExecutionStatus.RUNNING, "align", next(iter(workflow.command_ids.values()))
            )
        # A completed talk receipt leaves the participants physically aligned
        # for the next round-robin turn.  Restore the prepared stage rather
        # than treating a two-turn conversation as a one-turn-only workflow.
        if workflow.stage == "turn":
            workflow.stage = "align"
        return DriverResult(
            ExecutionStatus.SUCCEEDED, "aligned", next(iter(workflow.command_ids.values()))
        )

    def command_receipts(self) -> tuple[NpcCommandReceipt, ...]:
        """Return the receipt evidence retained by this driver instance."""
        return tuple(self._receipts.values())

    def play_turn(self, session: ConversationSession, turn: DialogueTurn) -> DriverResult:
        workflow = self._conversations.get(session.session_id)
        if workflow is None or workflow.stage != "align":
            return DriverResult(ExecutionStatus.FAILED, "turn", error="conversation_not_aligned")
        command_id = self._submit_stage(
            turn.speaker,
            f"{session.session_id}:{turn.turn_id}",
            "talk",
            NpcCommandKind.PLAY_ANIMATION,
            {
                "clip": "talk",
                "completion_marker": "talk_cycle",
                "target_site": workflow.sites[turn.speaker],
                "arrival_clip": "idle",
                "interaction_target": turn.listener,
                "interaction_distance_min": 0.45,
                "interaction_distance_max": 0.95,
                "interaction_yaw_tolerance": 0.30,
            },
            timeout_seconds=30.0,
        )
        workflow.stage = "turn"
        workflow.command_ids = {turn.speaker: command_id}
        return DriverResult(ExecutionStatus.RUNNING, "talk", command_id)

    def cancel_conversation(self, session_id: str, reason: str) -> DriverResult:
        workflow = self._conversations.pop(session_id, None)
        if workflow is not None:
            for npc_id, command_id in workflow.command_ids.items():
                self.simulator.cancel_npc_command(npc_id, command_id)
        self._release_interaction_sites(session_id)
        return DriverResult(ExecutionStatus.CANCELLED, "cancelled", error=reason)

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
        workflow = self._handovers.pop(execution.execution_id, None)
        cancelled_workflow = workflow is not None
        if workflow is not None:
            self._cancel_workflow_commands(workflow)
            self.interactions.cancel(workflow.session_id, reason)
            self._release_interaction_sites(execution.execution_id)
        sit = self._sits.pop(execution.execution_id, None)
        if sit is not None and execution.command is not None:
            self.simulator.cancel_npc_command(execution.command.agent_id, sit.command_id)
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
