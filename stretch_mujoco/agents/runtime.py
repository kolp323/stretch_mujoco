"""Deterministic office runtime for employee actions, conflicts, and robot tasks."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

from stretch_mujoco.semantics import ObjectType, RelationType, SemanticWorld

from .actions import (
    ActionCommand,
    ActionExecution,
    ActionType,
    ExecutionStatus,
    RobotTask,
    RobotTaskStatus,
    RuntimeEvent,
    ValidationResult,
)
from .employee import EmployeeAgent
from .conversation import (
    DEFAULT_MAX_OBSERVATION_AGE,
    ConversationCoordinator,
    ConversationEvent,
    ConversationIntent,
    ConversationParticipantKind,
    ConversationPerception,
    ConversationSession,
    ConversationStatus,
    ConversationTurn,
    InterruptPolicy,
    NpcConversationScheduler,
    SocialConversationDecision,
    SocialConversationProposal,
    SocialState,
)
from .events import DailyOfficeEvent, DailyOfficeEventGenerator
from .llm import EventDrivenLLMGateway, LLMRequest, LLMTrigger
from .llm_config import LLMProviderConfig
from .llm_provider import OpenAICompatibleProvider
from .models import AgentAvailability, EmployeeSchedule, MemoryEntry, ScheduleItem

ACTION_DURATIONS_MINUTES = {
    ActionType.IDLE: 1.0,
    ActionType.MOVE_TO: 3.0,
    ActionType.SIT: 0.5,
    ActionType.WORK: 15.0,
    ActionType.REST: 10.0,
    ActionType.EAT: 4.0,
    ActionType.DRINK: 2.0,
    ActionType.PICK_UP: 0.5,
    ActionType.PUT_DOWN: 0.5,
    ActionType.REQUEST_ROBOT: 0.2,
    ActionType.USE_COMPUTER: 10.0,
    ActionType.OPEN_CABINET: 0.5,
    ActionType.HANDOVER: 0.5,
    ActionType.ATTEND_MEETING: 15.0,
}


class ReservationManager:
    def __init__(self) -> None:
        self._owners: dict[str, str] = {}

    def owner(self, resource_id: str) -> str | None:
        return self._owners.get(resource_id)

    def is_available(self, resource_id: str, requester: str) -> bool:
        return self.owner(resource_id) in {None, requester}

    def reserve(self, resource_id: str, requester: str) -> bool:
        if not self.is_available(resource_id, requester):
            return False
        self._owners[resource_id] = requester
        return True

    def release(self, resource_id: str, requester: str | None = None) -> bool:
        owner = self.owner(resource_id)
        if owner is None or (requester is not None and owner != requester):
            return False
        del self._owners[resource_id]
        return True

    def transfer(self, resource_id: str, current_owner: str, new_owner: str) -> bool:
        if self.owner(resource_id) != current_owner:
            return False
        self._owners[resource_id] = new_owner
        return True

    def snapshot(self) -> dict[str, str]:
        return dict(self._owners)


class OfficeAgentRuntime:
    """Advance office time and execute only validated built-in actions."""

    def __init__(
        self,
        world: SemanticWorld,
        agents: dict[str, EmployeeAgent],
        *,
        start_minute: float = 9 * 60,
        minutes_per_second: float = 1.0,
        seed: int = 0,
        auto_plan: bool = True,
        state_machine_hz: float = 4.0,
        needs_hz: float = 0.5,
        utility_interval_seconds: tuple[float, float] = (5.0, 15.0),
        llm_daily_budget: int = 30,
        daily_events: bool = True,
        llm_provider_config: LLMProviderConfig | None = None,
        action_driver: object | None = None,
        conversation_max_observation_age: float = DEFAULT_MAX_OBSERVATION_AGE,
    ) -> None:
        if state_machine_hz <= 0 or needs_hz <= 0:
            raise ValueError("Agent update frequencies must be positive")
        if (
            utility_interval_seconds[0] <= 0
            or utility_interval_seconds[1] < utility_interval_seconds[0]
        ):
            raise ValueError("Invalid utility decision interval")
        if conversation_max_observation_age < 0 or not math.isfinite(
            conversation_max_observation_age
        ):
            raise ValueError("Conversation observation age must be a non-negative finite value")
        self.world = world
        self.agents = agents
        self.minute_of_day = float(start_minute)
        self.day = 0
        self.elapsed_minutes = 0.0
        self.minutes_per_second = minutes_per_second
        self.seed = seed
        self.auto_plan = auto_plan
        self.state_machine_hz = state_machine_hz
        self.needs_hz = needs_hz
        self.utility_interval_seconds = utility_interval_seconds
        self.reservations = ReservationManager()
        self.robot_tasks: dict[str, RobotTask] = {}
        self.conversations = ConversationCoordinator()
        self.social_conversations = NpcConversationScheduler()
        self.conversation_max_observation_age = conversation_max_observation_age
        self._conversation_locks: dict[
            str, dict[str, tuple[str | None, AgentAvailability, str]]
        ] = {}
        self.events: list[RuntimeEvent] = []
        self.llm = EventDrivenLLMGateway(llm_daily_budget)
        self.llm_provider_config = llm_provider_config
        self.action_driver = action_driver
        self.office_event_generator = DailyOfficeEventGenerator(seed)
        self.daily_events_enabled = daily_events
        self.daily_office_events: list[DailyOfficeEvent] = []
        self._state_machine_accumulator = 0.0
        self._needs_accumulator = 0.0
        self._utility_seconds_remaining = 0.0
        self._utility_decision_count = 0
        self._consecutive_failures: dict[str, int] = {}
        self._committed_execution_ids: set[str] = set()
        for agent in self.agents.values():
            agent.validate_identity(world)
            world.object(agent.state.location)
        self._start_day()

    @classmethod
    def from_json(
        cls,
        world: SemanticWorld,
        path: str | Path,
        *,
        seed: int | None = None,
        auto_plan: bool = True,
        action_driver: object | None = None,
    ) -> "OfficeAgentRuntime":
        source_path = Path(path).resolve()
        payload = json.loads(source_path.read_text(encoding="utf-8"))
        clock = payload.get("clock", {})
        frequencies = payload.get("frequencies", {})
        utility_interval = frequencies.get("utility_interval_seconds", [5.0, 15.0])
        start_hour, start_minute = (int(part) for part in clock.get("start", "09:00").split(":"))
        version = payload.get("schema_version")
        if version == 2:
            from stretch_mujoco.npc.schema import NpcPopulation

            population = NpcPopulation.from_dict(
                payload,
                source_path=source_path,
                locations=set(world.objects),
                sites={point.site for point in world.interaction_points.values()},
            )
            agents = {
                npc_id: EmployeeAgent.from_definition(definition)
                for npc_id, definition in population.npcs.items()
            }
        elif version == 1:
            agents = {
                agent_id: EmployeeAgent.from_dict(agent_id, definition)
                for agent_id, definition in payload.get("employees", {}).items()
            }
        else:
            raise ValueError(
                f"Unsupported office agent schema_version {version!r}; expected 1 or 2"
            )
        llm_provider_config = None
        llm_config_name = payload.get("llm_config")
        if llm_config_name:
            llm_config_path = source_path.parent / llm_config_name
            if llm_config_path.exists():
                llm_provider_config = LLMProviderConfig.from_json(llm_config_path)
        return cls(
            world,
            agents,
            start_minute=start_hour * 60 + start_minute,
            minutes_per_second=float(clock.get("minutes_per_second", 1.0)),
            seed=int(payload.get("seed", 0) if seed is None else seed),
            auto_plan=auto_plan,
            state_machine_hz=float(frequencies.get("state_machine_hz", 4.0)),
            needs_hz=float(frequencies.get("needs_hz", 0.5)),
            utility_interval_seconds=(
                float(utility_interval[0]),
                float(utility_interval[1]),
            ),
            llm_daily_budget=int(payload.get("llm_daily_budget", 30)),
            daily_events=bool(payload.get("daily_events", True)),
            llm_provider_config=llm_provider_config,
            action_driver=action_driver,
        )

    def llm_config_summary(self) -> dict[str, Any]:
        if self.llm_provider_config is None:
            return {"enabled": False, "configured": False}
        return {
            "configured": True,
            **self.llm_provider_config.safe_summary(),
        }

    def create_llm_provider(self) -> OpenAICompatibleProvider:
        """Build the explicitly configured provider for out-of-loop LLM events."""
        if self.llm_provider_config is None:
            raise ValueError("No local LLM provider config was found")
        return OpenAICompatibleProvider(self.llm_provider_config)

    def validate_action(self, command: ActionCommand) -> ValidationResult:
        errors: list[str] = []
        agent = self.agents.get(command.agent_id)
        if agent is None:
            return ValidationResult(False, (f"Unknown agent '{command.agent_id}'",))
        if agent.executor.is_busy:
            errors.append(
                f"Agent '{command.agent_id}' is already executing "
                f"'{agent.executor.command.action.value}'"
            )

        target = self._normalize_target(command.target)
        if target is not None and target not in self.world.objects:
            errors.append(f"Target '{target}' does not exist")

        if command.action == ActionType.MOVE_TO:
            self._require_target(target, errors)
        elif command.action == ActionType.SIT:
            self._require_type(target, ObjectType.CHAIR, errors)
            supported = getattr(self.action_driver, "supported_actions", frozenset())
            if self.action_driver is None or ActionType.SIT not in supported:
                self._require_location(agent, target, errors)
            self._require_available(target, command.agent_id, errors)
        elif command.action == ActionType.WORK:
            self._require_type(target, ObjectType.WORKSTATION, errors)
            self._require_location(agent, target, errors)
        elif command.action == ActionType.REST:
            self._require_type(target, ObjectType.CHAIR, errors)
            self._require_location(agent, target, errors)
            self._require_available(target, command.agent_id, errors)
        elif command.action == ActionType.ATTEND_MEETING:
            self._require_type(target, ObjectType.MEETING_TABLE, errors)
            self._require_location(agent, target, errors)
        elif command.action == ActionType.OPEN_CABINET:
            self._require_type(target, ObjectType.STORAGE_CABINET, errors)
            self._require_location(agent, target, errors)
        elif command.action == ActionType.USE_COMPUTER:
            self._require_type(target, ObjectType.COMPUTER, errors)
            if target in self.world.objects:
                workstations = self.world.related_objects(target, RelationType.ON)
                if not workstations or agent.state.location != workstations[0].object_id:
                    errors.append("Agent is not at the computer's workstation")
        elif command.action == ActionType.PICK_UP:
            self._validate_pick_up(agent, target, errors)
        elif command.action in {ActionType.EAT, ActionType.DRINK}:
            if target is None or agent.state.held_object != target:
                errors.append("Agent must hold the target object")
            self._require_type(target, ObjectType.SNACK, errors)
        elif command.action == ActionType.PUT_DOWN:
            if agent.state.held_object is None:
                errors.append("Agent is not holding an object")
            self._require_target(target, errors)
        elif command.action == ActionType.HANDOVER:
            if agent.state.held_object is None:
                errors.append("Agent is not holding an object")
            self._require_target(target, errors)
        elif command.action == ActionType.REQUEST_ROBOT:
            self._validate_robot_request(agent, target, command.parameters, errors)

        return ValidationResult(not errors, tuple(errors))

    def submit_action(self, command: ActionCommand | dict[str, Any]) -> ValidationResult:
        if isinstance(command, dict):
            command = ActionCommand.from_dict(command)
        parameters = dict(command.parameters)
        if command.action in {ActionType.PUT_DOWN, ActionType.HANDOVER}:
            held_object = self.agents.get(command.agent_id)
            if held_object is not None and held_object.state.held_object is not None:
                parameters.setdefault("object", held_object.state.held_object)
        normalized = ActionCommand(
            command.agent_id,
            command.action,
            self._normalize_target(command.target),
            parameters,
        )
        validation = self.validate_action(normalized)
        if not validation.valid:
            self._emit(
                "action_rejected",
                normalized.agent_id,
                {"action": normalized.action.value, "errors": list(validation.errors)},
            )
            if normalized.agent_id in self.agents:
                self._record_failure(
                    normalized.agent_id,
                    {"action": normalized.action.value, "errors": validation.errors},
                )
            return validation

        self._reserve_for_action(normalized)
        agent = self.agents[normalized.agent_id]
        agent.executor = ActionExecution(
            command=normalized,
            status=ExecutionStatus.RUNNING,
            phase="start",
            started_at=self.elapsed_minutes,
            remaining_minutes=ACTION_DURATIONS_MINUTES[normalized.action],
        )
        agent.state.current_action = normalized.action.value
        agent.state.animation_state = normalized.action.value
        agent.state.set_availability(AgentAvailability.EXECUTING)
        agent.state.attention_target = normalized.target
        agent.state.blocked_reason = None
        self._emit(
            "action_started",
            normalized.agent_id,
            {
                "action": normalized.action.value,
                "target": normalized.target,
                "execution_id": agent.executor.execution_id,
            },
        )
        supported = getattr(self.action_driver, "supported_actions", frozenset())
        if self.action_driver is not None and normalized.action in supported:
            result = self.action_driver.start(agent.executor)
            agent.executor.status = result.status
            agent.executor.phase = result.phase
            agent.executor.driver_handle = result.handle
            agent.executor.error = result.error
            if result.status in {
                ExecutionStatus.FAILED,
                ExecutionStatus.CANCELLED,
                ExecutionStatus.TIMED_OUT,
            }:
                self._fail_action(agent, result.error or result.status.value)
        return validation

    def tick(
        self, seconds: float, semantic_snapshot: dict[str, Any] | None = None
    ) -> tuple[RuntimeEvent, ...]:
        if seconds < 0:
            raise ValueError("Runtime tick duration cannot be negative")
        minutes = seconds * self.minutes_per_second
        self._advance_clock(minutes)

        for session in self.conversations.expire(self.elapsed_minutes):
            self._close_conversation(session, ConversationEvent.TIMED_OUT.value)

        for agent in self.agents.values():
            agent.perception.update(agent.agent_id, semantic_snapshot)

        self._needs_accumulator += seconds
        needs_period = 1.0 / self.needs_hz
        while self._needs_accumulator >= needs_period:
            needs_minutes = needs_period * self.minutes_per_second
            for agent in self.agents.values():
                agent.needs.advance(needs_minutes, agent.state.current_action)
                agent.state.sync_needs(agent.needs)
            self._needs_accumulator -= needs_period

        self._utility_seconds_remaining -= seconds
        if self.auto_plan and self._utility_seconds_remaining <= 0.0:
            self._run_utility_decisions()
            self._utility_seconds_remaining = self._next_utility_interval()

        self._state_machine_accumulator += seconds
        state_machine_period = 1.0 / self.state_machine_hz
        while self._state_machine_accumulator >= state_machine_period:
            self._state_machine_step(state_machine_period * self.minutes_per_second)
            self._state_machine_accumulator -= state_machine_period
        return self.drain_events()

    def queue_llm_event(
        self,
        trigger: LLMTrigger,
        agent_id: str,
        context: dict[str, Any] | None = None,
    ) -> bool:
        if agent_id not in self.agents:
            raise KeyError(f"Unknown agent '{agent_id}'")
        queued = self.llm.queue(
            trigger,
            agent_id,
            self.day,
            self.minute_of_day,
            context,
        )
        if queued:
            self._emit(
                "llm_event_queued",
                agent_id,
                {"trigger": trigger.value},
            )
        return queued

    def drain_llm_requests(self) -> tuple[LLMRequest, ...]:
        """Drain event requests for processing outside the simulation tick."""
        return self.llm.drain_requests()

    def apply_llm_response(self, request: LLMRequest, response: dict[str, Any]) -> ValidationResult:
        """Apply a schedule draft or closed action after normal validation."""
        agent = self.agents[request.agent_id]
        errors: list[str] = []
        if request.trigger == LLMTrigger.DIALOGUE and "session_id" in request.context:
            try:
                entry = self.record_conversation_candidate(
                    str(request.context["session_id"]),
                    request.agent_id,
                    response.get("intent", "acknowledge"),
                    response.get("text", response.get("dialogue")),
                )
            except (KeyError, TypeError, ValueError) as error:
                return ValidationResult(False, (f"Invalid dialogue candidate: {error}",))
            self._remember(
                agent,
                "conversation_candidate_applied",
                {"session_id": request.context["session_id"], "turn": entry.intent.value},
            )
            return ValidationResult(True)
        if "schedule" in response:
            try:
                items = tuple(ScheduleItem.from_dict(item) for item in response["schedule"])
            except (KeyError, TypeError, ValueError) as error:
                return ValidationResult(False, (f"Invalid schedule draft: {error}",))
            for item in items:
                if item.location not in self.world.objects:
                    errors.append(
                        f"Schedule item '{item.item_id}' has unknown location " f"'{item.location}'"
                    )
                if item.end_minute <= item.start_minute:
                    errors.append(f"Schedule item '{item.item_id}' has an invalid time window")
            ordered = sorted(items, key=lambda item: item.start_minute)
            for previous, current in zip(ordered, ordered[1:]):
                if current.start_minute < previous.end_minute:
                    errors.append(
                        f"Schedule items '{previous.item_id}' and " f"'{current.item_id}' overlap"
                    )
            if not errors:
                agent.schedule = EmployeeSchedule(items)
        action_result = ValidationResult(True)
        if "action" in response and not errors:
            try:
                action_payload = dict(response["action"])
                action_payload["agent_id"] = request.agent_id
                event_target = request.context.get("target")
                parameters = action_payload.get("parameters", {})
                requested_object = (
                    parameters.get("object") if isinstance(parameters, dict) else None
                )
                if (
                    request.trigger == LLMTrigger.NEW_TASK
                    and event_target is not None
                    and event_target not in {action_payload.get("target"), requested_object}
                ):
                    action_result = ValidationResult(
                        False,
                        (f"New-task action does not address '{event_target}'",),
                    )
                else:
                    action_result = self.submit_action(action_payload)
            except (TypeError, ValueError) as error:
                action_result = ValidationResult(False, (str(error),))
        errors.extend(action_result.errors)
        result = ValidationResult(not errors, tuple(errors))
        self._remember(
            agent,
            "llm_response_applied" if result.valid else "llm_response_rejected",
            {"trigger": request.trigger.value, "errors": list(result.errors)},
        )
        return result

    def notify_new_task(self, agent_id: str, task: dict[str, Any]) -> bool:
        return self.queue_llm_event(LLMTrigger.NEW_TASK, agent_id, task)

    def request_dialogue(self, agent_id: str, partner: str, topic: str) -> bool:
        return self.queue_llm_event(
            LLMTrigger.DIALOGUE,
            agent_id,
            {"partner": partner, "topic": topic},
        )

    def start_conversation(
        self,
        initiator: str,
        participant: str,
        topic: str,
        semantic_snapshot: dict[str, Any],
        *,
        timeout: float = 2.0,
        interrupt_policy: InterruptPolicy = InterruptPolicy.REJECT,
    ) -> ConversationSession:
        """Start a spatially valid dialogue and pause involved NPC plans.

        The snapshot is runtime observation, never LLM output.  Both participants
        must provide a position and yaw so the runtime can reject implausible speech.
        """
        if initiator not in self.agents:
            raise KeyError(f"Unknown NPC initiator '{initiator}'")
        if participant not in self.agents and participant not in self.world.objects:
            raise KeyError(f"Unknown conversation participant '{participant}'")
        if (
            participant not in self.agents
            and self.world.object(participant).object_type != ObjectType.STRETCH_ROBOT
        ):
            raise ValueError("Conversation participant must be an NPC or Stretch robot")
        agent_participants = tuple(
            agent_id for agent_id in (initiator, participant) if agent_id in self.agents
        )
        if any(self.agents[agent_id].executor.is_busy for agent_id in agent_participants):
            raise ValueError("Cannot interrupt a busy NPC conversation plan")
        if any(
            self.agents[agent_id].state.availability == "in_conversation"
            for agent_id in agent_participants
        ):
            raise ValueError("An NPC may only participate in one active conversation")
        perception = self._conversation_perception(semantic_snapshot, (initiator, participant))
        session = self.conversations.start(
            (initiator, participant),
            topic,
            self.elapsed_minutes,
            perception.poses,
            timeout=timeout,
            interrupt_policy=interrupt_policy,
            participant_kinds={
                initiator: ConversationParticipantKind.NPC,
                participant: (
                    ConversationParticipantKind.NPC
                    if participant in self.agents
                    else ConversationParticipantKind.STRETCH_ROBOT
                ),
            },
        )
        locks: dict[str, tuple[str | None, AgentAvailability, str]] = {}
        for agent_id in agent_participants:
            agent = self.agents[agent_id]
            locks[agent_id] = (
                agent.state.attention_target,
                AgentAvailability(agent.state.availability),
                agent.state.current_goal,
            )
            agent.state.attention_target = participant if agent_id == initiator else initiator
            agent.state.set_availability(AgentAvailability.IN_CONVERSATION)
            agent.state.conversation_id = session.session_id
            agent.state.social_energy = max(0.0, agent.state.social_energy - 0.02)
            self._remember(
                agent, ConversationEvent.STARTED.value, {"session_id": session.session_id}
            )
            self._emit(
                ConversationEvent.STARTED.value,
                agent_id,
                {"session_id": session.session_id, "topic": session.topic},
            )
        self._conversation_locks[session.session_id] = locks
        return session

    def record_conversation_candidate(
        self,
        session_id: str,
        speaker: str,
        intent: ConversationIntent | str,
        text: str | None,
    ) -> ConversationTurn:
        session = self.conversations.sessions.get(session_id)
        if session is None:
            raise KeyError(f"Unknown conversation '{session_id}'")
        includes_robot = any(participant not in self.agents for participant in session.participants)
        if includes_robot:
            self._validate_robot_dialogue_turn(session, intent)
        entry = self.conversations.record_candidate(
            session_id,
            speaker,
            intent,
            text,
            self.elapsed_minutes,
            includes_robot=includes_robot,
        )
        for participant in session.participants:
            agent = self.agents.get(participant)
            if agent is not None:
                self._remember(
                    agent,
                    ConversationEvent.TURN.value,
                    {
                        "session_id": session_id,
                        "speaker": speaker,
                        "intent": entry.intent.value,
                        "used_fallback": entry.used_fallback,
                    },
                )
        self._emit(
            ConversationEvent.TURN.value,
            speaker,
            {
                "session_id": session_id,
                "intent": entry.intent.value,
                "used_fallback": entry.used_fallback,
            },
        )
        return entry

    def complete_conversation(self, session_id: str) -> ConversationSession:
        session = self.conversations.complete(session_id)
        self._close_conversation(session, ConversationEvent.COMPLETED.value)
        return session

    def cancel_conversation(
        self, session_id: str, reason: str = "cancelled"
    ) -> ConversationSession:
        """Cancel once and restore the exact locked plan state without touching reservations."""
        session = self.conversations.cancel(session_id, reason)
        self._close_conversation(session, ConversationEvent.CANCELLED.value)
        return session

    def fail_conversation(self, session_id: str, reason: str) -> ConversationSession:
        """Record a logical/receipt failure and use the common terminal cleanup path."""
        session = self.conversations.fail(session_id, reason)
        self._close_conversation(session, ConversationEvent.FAILED.value)
        return session

    def request_robot_task(
        self,
        session_id: str,
        requester: str,
        *,
        task: str,
        object_id: str,
        destination: str,
    ) -> ValidationResult:
        """Submit the only supported robot-request path for a dialogue session.

        It deliberately delegates to ``submit_action``: the same permission,
        graspability, reservation, and target validation protects conversational
        requests and ordinary planner requests.  The task receives its session
        link only when that validated action actually commits.
        """
        session = self.conversations.sessions.get(session_id)
        if session is None or session.status.terminal:
            raise ValueError(f"Conversation '{session_id}' is not active")
        if requester not in self.agents or requester not in session.participants:
            raise ValueError("Robot task requester must be an NPC conversation participant")
        if not any(participant not in self.agents for participant in session.participants):
            raise ValueError("Robot task requests require a Stretch conversation participant")
        if session.robot_task_id is not None:
            raise ValueError(f"Conversation '{session_id}' already has a robot task")
        command = ActionCommand(
            requester,
            ActionType.REQUEST_ROBOT,
            "stretch_3",
            {
                "task": task,
                "object": object_id,
                "destination": destination,
                "conversation_id": session_id,
            },
        )
        validation = self.validate_action(command)
        if not validation.valid:
            self._emit(
                "action_rejected",
                requester,
                {"action": command.action.value, "errors": list(validation.errors)},
            )
            return validation
        # A request is an accepted logical RPC, not an embodied NPC action.
        # Running it through the normal executor would release the conversation
        # attention lock while the session is still active.
        self._reserve_for_action(command)
        requester_agent = self.agents[requester]
        self._apply_action_effect(requester_agent, command)
        self._verify_action_effect(requester_agent, command)
        self._emit(
            "robot_task_requested",
            requester,
            {"session_id": session_id, "task_id": session.robot_task_id},
        )
        return validation

    def record_robot_handover_receipt(
        self,
        session_id: str,
        task_id: str,
        receipt_id: str,
        *,
        robot_release_confirmed: bool = False,
        npc_attachment_confirmed: bool = False,
        interaction_confirmed: bool = False,
    ) -> RobotTask:
        """Record idempotent physical evidence; dialogue text cannot supply it."""
        if not receipt_id.strip():
            raise ValueError("Handover receipt requires a stable receipt_id")
        session = self.conversations.sessions.get(session_id)
        task = self.robot_tasks[task_id]
        if (
            session is None
            or session.robot_task_id != task_id
            or task.conversation_id != session_id
        ):
            raise ValueError("Handover receipt does not match the conversation task")
        if task.status != RobotTaskStatus.SUCCEEDED:
            raise ValueError("Robot handover receipt requires a successful task")
        if receipt_id in task.receipt_ids:
            return task
        task.receipt_ids.add(receipt_id)
        task.robot_release_confirmed = task.robot_release_confirmed or robot_release_confirmed
        task.npc_attachment_confirmed = task.npc_attachment_confirmed or npc_attachment_confirmed
        task.interaction_confirmed = task.interaction_confirmed or interaction_confirmed
        session.handover_receipts = frozenset(task.receipt_ids)
        self._emit(
            "robot_handover_receipt",
            task.requester,
            {"session_id": session_id, "task_id": task_id, "receipt_id": receipt_id},
        )
        return task

    def queue_conversation_candidate(self, session_id: str, speaker: str) -> bool:
        """Queue one LLM candidate; acceptance remains in ``record_conversation_candidate``."""
        session = self.conversations.sessions.get(session_id)
        if session is None:
            raise KeyError(f"Unknown conversation '{session_id}'")
        if speaker not in self.agents:
            raise ValueError("Only an NPC can request an LLM conversation candidate")
        if speaker not in session.participants:
            raise ValueError(f"Speaker '{speaker}' is not in the conversation")
        partner = next(
            participant for participant in session.participants if participant != speaker
        )
        return self.queue_llm_event(
            LLMTrigger.DIALOGUE,
            speaker,
            {
                "session_id": session_id,
                "partner": partner,
                "topic": session.topic,
                "allowed_intents": sorted(
                    intent.value
                    for intent in (
                        {
                            ConversationIntent.REQUEST,
                            ConversationIntent.CLARIFY,
                            ConversationIntent.ACKNOWLEDGE,
                            ConversationIntent.HANDOVER_CONFIRM,
                        }
                        if partner not in self.agents
                        else {
                            ConversationIntent.GREETING,
                            ConversationIntent.PROGRESS_INQUIRY,
                            ConversationIntent.MEETING_INVITATION,
                            ConversationIntent.CONFLICT_RESOLUTION,
                        }
                    )
                ),
            },
        )

    def schedule_npc_conversations(
        self,
        proposals: list[SocialConversationProposal],
        semantic_snapshot: dict[str, Any],
    ) -> tuple[SocialConversationDecision, ...]:
        """Admit deterministic low-priority NPC dialogue without changing schedules.

        Accepted meeting invitations remain proposals.  No schedule or table
        reservation is changed here; a later planner/action receipt owns that
        transition.  The runtime does validate a named meeting location before
        it starts the dialogue that discusses it.
        """
        states = {
            agent_id: SocialState(
                agent.state.social_energy,
                agent.state.stress,
                agent.state.availability == AgentAvailability.AVAILABLE
                and not agent.executor.is_busy,
            )
            for agent_id, agent in self.agents.items()
        }
        decisions = self.social_conversations.admit(proposals, states, self.elapsed_minutes)
        results: list[SocialConversationDecision] = []
        for decision in decisions:
            proposal = decision.proposal
            if not decision.accepted:
                self._emit(
                    "npc_conversation_rejected",
                    proposal.initiator,
                    {"participant": proposal.participant, "reason": decision.reason},
                )
                results.append(decision)
                continue
            if proposal.intent == ConversationIntent.MEETING_INVITATION:
                location = proposal.metadata.get("location")
                if location is not None and (
                    location not in self.world.objects
                    or self.world.object(location).object_type != ObjectType.MEETING_TABLE
                    or not self.reservations.is_available(location, proposal.initiator)
                ):
                    rejected = SocialConversationDecision(
                        proposal, False, "invalid_meeting_location"
                    )
                    self._emit(
                        "npc_conversation_rejected",
                        proposal.initiator,
                        {"participant": proposal.participant, "reason": rejected.reason},
                    )
                    results.append(rejected)
                    continue
            try:
                session = self.start_conversation(
                    proposal.initiator,
                    proposal.participant,
                    proposal.topic,
                    semantic_snapshot,
                )
                self.record_conversation_candidate(
                    session.session_id, proposal.initiator, proposal.intent, None
                )
            except (KeyError, ValueError):
                rejected = SocialConversationDecision(proposal, False, "runtime_rejected")
                self._emit(
                    "npc_conversation_rejected",
                    proposal.initiator,
                    {"participant": proposal.participant, "reason": rejected.reason},
                )
                results.append(rejected)
            else:
                if proposal.intent == ConversationIntent.MEETING_INVITATION:
                    self.social_conversations.create_invitation(
                        proposal, self.elapsed_minutes + 15.0
                    )
                results.append(decision)
        return tuple(results)

    def report_unexpected_change(self, agent_id: str, description: str) -> bool:
        return self.queue_llm_event(
            LLMTrigger.UNEXPECTED_CHANGE,
            agent_id,
            {"description": description},
        )

    def request_plan_reinterpretation(self, agent_id: str, reason: str) -> bool:
        return self.queue_llm_event(
            LLMTrigger.REINTERPRET_PLAN,
            agent_id,
            {"reason": reason},
        )

    def complete_robot_task(
        self,
        task_id: str,
        success: bool,
        error: str | None = None,
        semantic_snapshot: dict[str, Any] | None = None,
    ) -> RobotTask:
        task = self.robot_tasks[task_id]
        if task.status not in {RobotTaskStatus.PENDING, RobotTaskStatus.RUNNING}:
            return task
        if success and semantic_snapshot is not None:
            verification = self.verify_robot_task_result(task_id, semantic_snapshot)
            if not verification.valid:
                success = False
                error = "; ".join(verification.errors)
        requester = self.agents[task.requester]
        if success:
            task.status = RobotTaskStatus.SUCCEEDED
            self.world.set_location(task.object_id, RelationType.ON, task.destination)
            self.world.remove_relation(task.object_id, RelationType.REQUESTED_BY, task.requester)
            self.reservations.release(task.object_id, task.requester)
            self._remember(
                requester,
                "robot_task_succeeded",
                {
                    "task_id": task_id,
                    "object": task.object_id,
                    "conversation_id": task.conversation_id,
                },
            )
        else:
            task.status = RobotTaskStatus.FAILED
            task.error = error or "Robot task failed"
            self.reservations.release(task.object_id, task.requester)
            self._remember(
                requester,
                "robot_task_failed",
                {"task_id": task_id, "error": task.error, "conversation_id": task.conversation_id},
            )
        self._emit(
            "robot_task_completed",
            task.requester,
            {
                "task_id": task_id,
                "status": task.status.value,
                "conversation_id": task.conversation_id,
            },
        )
        return task

    def verify_robot_task_result(
        self,
        task_id: str,
        semantic_snapshot: dict[str, Any],
        horizontal_tolerance: float = 0.9,
    ) -> ValidationResult:
        task = self.robot_tasks[task_id]
        objects = semantic_snapshot.get("objects", {})
        object_pose = objects.get(task.object_id)
        destination_pose = objects.get(task.destination)
        errors: list[str] = []
        if object_pose is None:
            errors.append(f"Snapshot is missing object '{task.object_id}'")
        if destination_pose is None:
            errors.append(f"Snapshot is missing destination '{task.destination}'")
        if not errors:
            object_xy = object_pose["position"][:2]
            destination_xy = destination_pose["position"][:2]
            distance = math.dist(object_xy, destination_xy)
            if distance > horizontal_tolerance:
                errors.append(
                    f"Object is {distance:.3f} m from destination; "
                    f"tolerance is {horizontal_tolerance:.3f} m"
                )
        return ValidationResult(not errors, tuple(errors))

    def drain_events(self) -> tuple[RuntimeEvent, ...]:
        events = tuple(self.events)
        self.events.clear()
        return events

    def pending_robot_tasks(self) -> tuple[RobotTask, ...]:
        return tuple(
            task
            for task in self.robot_tasks.values()
            if task.status in {RobotTaskStatus.PENDING, RobotTaskStatus.RUNNING}
        )

    def _validate_pick_up(
        self, agent: EmployeeAgent, target: str | None, errors: list[str]
    ) -> None:
        self._require_target(target, errors)
        if target is None or target not in self.world.objects:
            return
        semantic_object = self.world.object(target)
        if not semantic_object.get("graspable", False):
            errors.append(f"Object '{target}' is not graspable")
        if agent.state.held_object is not None:
            errors.append("Agent already holds an object")
        location_object = self.world.location_of(target)
        location = None if location_object is None else location_object.object_id
        if location is not None and agent.state.location != location:
            errors.append(f"Agent is at '{agent.state.location}', but object is at '{location}'")
        if not self.reservations.is_available(target, agent.agent_id):
            errors.append(f"Object '{target}' is reserved by '{self.reservations.owner(target)}'")
        if not self.world.can_access(agent.agent_id, target):
            errors.append(f"Agent '{agent.agent_id}' is not allowed to access '{target}'")
        if (
            agent.perception.last_update_time > 0.0
            and target not in agent.perception.visible_objects
        ):
            errors.append(f"Object '{target}' is not currently visible")

    def _validate_robot_request(
        self,
        agent: EmployeeAgent,
        target: str | None,
        parameters: dict[str, Any],
        errors: list[str],
    ) -> None:
        if target != "stretch_3":
            errors.append("request_robot target must be 'stretch' or 'stretch_3'")
        task = parameters.get("task")
        object_id = parameters.get("object")
        destination = parameters.get("destination")
        if task not in {"deliver", "fetch"}:
            errors.append("Robot task must be 'deliver' or 'fetch'")
        if object_id not in self.world.objects:
            errors.append(f"Requested object '{object_id}' does not exist")
        if destination not in self.world.objects:
            errors.append(f"Destination '{destination}' does not exist")
        if object_id in self.world.objects:
            semantic_object = self.world.object(object_id)
            if not semantic_object.get("graspable", False):
                errors.append(f"Requested object '{object_id}' is not graspable")
            if semantic_object.get("consumed", False):
                errors.append(f"Requested object '{object_id}' has been consumed")
            if not semantic_object.get("available", True):
                errors.append(f"Requested object '{object_id}' is unavailable")
            if not self.world.can_access(agent.agent_id, object_id):
                errors.append(f"Agent '{agent.agent_id}' is not allowed to request '{object_id}'")
            if not self.reservations.is_available(object_id, agent.agent_id):
                errors.append(
                    f"Object '{object_id}' is reserved by "
                    f"'{self.reservations.owner(object_id)}'"
                )
            if any(
                task.object_id == object_id
                and task.status in {RobotTaskStatus.PENDING, RobotTaskStatus.RUNNING}
                for task in self.robot_tasks.values()
            ):
                errors.append(f"Object '{object_id}' already has an active robot task")

    def _complete_action(self, agent: EmployeeAgent) -> None:
        command = agent.executor.command
        if command is None:
            return
        if agent.executor.execution_id in self._committed_execution_ids:
            return
        try:
            self._apply_action_effect(agent, command)
            self._verify_action_effect(agent, command)
        except (KeyError, ValueError) as error:
            self._fail_action(agent, str(error))
            return
        else:
            self._committed_execution_ids.add(agent.executor.execution_id)
            agent.executor.status = ExecutionStatus.SUCCEEDED
            self._consecutive_failures[agent.agent_id] = 0
            self._remember(
                agent,
                "action_succeeded",
                {
                    "action": command.action.value,
                    "target": command.target,
                    "execution_id": agent.executor.execution_id,
                },
            )
            self._emit(
                "action_succeeded",
                agent.agent_id,
                {
                    "action": command.action.value,
                    "target": command.target,
                    "execution_id": agent.executor.execution_id,
                },
            )
            self._emit(
                "semantic_commit",
                agent.agent_id,
                {
                    "action": command.action.value,
                    "execution_id": agent.executor.execution_id,
                },
            )
        agent.state.current_action = "idle"
        agent.state.animation_state = "idle"
        agent.state.set_availability(AgentAvailability.AVAILABLE)
        agent.state.attention_target = None

    def _fail_action(self, agent: EmployeeAgent, error: str) -> None:
        command = agent.executor.command
        if command is None:
            return
        if agent.executor.status not in {
            ExecutionStatus.CANCELLED,
            ExecutionStatus.TIMED_OUT,
        }:
            agent.executor.status = ExecutionStatus.FAILED
        agent.executor.error = error
        self._record_failure(
            agent.agent_id,
            {"action": command.action.value, "error": error},
        )
        self._release_failed_action(command)
        self._remember(
            agent,
            "action_failed",
            {"action": command.action.value, "error": error},
        )
        self._emit(
            "action_failed",
            agent.agent_id,
            {
                "action": command.action.value,
                "error": error,
                "execution_id": agent.executor.execution_id,
            },
        )
        agent.state.current_action = "idle"
        agent.state.animation_state = "idle"
        agent.state.set_availability(AgentAvailability.AVAILABLE)
        agent.state.attention_target = None

    def _apply_action_effect(self, agent: EmployeeAgent, command: ActionCommand) -> None:
        action = command.action
        target = command.target
        if action == ActionType.MOVE_TO:
            self._leave_occupied_location(agent)
            agent.state.location = target
        elif action == ActionType.SIT:
            self._leave_occupied_location(agent)
            agent.state.location = target
            self.world.replace_relation(target, RelationType.OCCUPIED_BY, agent.agent_id)
        elif action == ActionType.REST:
            agent.needs.fatigue = max(0.0, agent.needs.fatigue - 0.35)
        elif action == ActionType.EAT:
            agent.needs.hunger = max(0.0, agent.needs.hunger - 0.55)
            self._consume_held_object(agent, target)
        elif action == ActionType.DRINK:
            agent.needs.thirst = max(0.0, agent.needs.thirst - 0.60)
            self._consume_held_object(agent, target)
        elif action == ActionType.PICK_UP:
            agent.state.held_object = target
            self.world.add_relation(agent.agent_id, RelationType.HOLDS, target)
            for relation in self.world.find_relations(subject=target):
                if relation.relation in {RelationType.INSIDE, RelationType.ON}:
                    self.world.remove_relation(target, relation.relation, relation.object)
            self.world.object(target).attributes.pop("location", None)
        elif action == ActionType.PUT_DOWN:
            held_object = agent.state.held_object
            self.world.remove_relation(agent.agent_id, RelationType.HOLDS, held_object)
            self.world.set_location(held_object, RelationType.ON, target)
            self.reservations.release(held_object, agent.agent_id)
            agent.state.held_object = None
        elif action == ActionType.REQUEST_ROBOT:
            task = RobotTask(
                requester=agent.agent_id,
                task=command.parameters["task"],
                object_id=command.parameters["object"],
                destination=command.parameters["destination"],
                conversation_id=command.parameters.get("conversation_id"),
            )
            self.robot_tasks[task.task_id] = task
            if task.conversation_id is not None:
                session = self.conversations.sessions.get(task.conversation_id)
                if session is None or session.status.terminal or session.robot_task_id is not None:
                    raise ValueError("Robot task has an invalid conversation association")
                session.robot_task_id = task.task_id
            self.world.add_relation(task.object_id, RelationType.REQUESTED_BY, agent.agent_id)
            self._emit(
                "robot_task_created",
                agent.agent_id,
                {
                    "task_id": task.task_id,
                    "object": task.object_id,
                    "conversation_id": task.conversation_id,
                },
            )
        elif action == ActionType.OPEN_CABINET:
            self.world.object(target).attributes["open"] = True
        elif action == ActionType.HANDOVER:
            held_object = agent.state.held_object
            self.world.remove_relation(agent.agent_id, RelationType.HOLDS, held_object)
            self.world.add_relation(target, RelationType.HOLDS, held_object)
            self.reservations.transfer(held_object, agent.agent_id, target)
            agent.state.held_object = None
            if target in self.agents:
                self.agents[target].state.held_object = held_object
        agent.state.sync_needs(agent.needs)

    def _verify_action_effect(self, agent: EmployeeAgent, command: ActionCommand) -> None:
        if command.action == ActionType.MOVE_TO and agent.state.location != command.target:
            raise ValueError("Location update verification failed")
        if command.action == ActionType.SIT and agent.state.location != command.target:
            raise ValueError("Seat occupancy location verification failed")
        if command.action == ActionType.PICK_UP:
            if agent.state.held_object != command.target or not self.world.find_relations(
                subject=agent.agent_id,
                relation=RelationType.HOLDS,
                object_id=command.target,
            ):
                raise ValueError("Pick-up result verification failed")
        if command.action == ActionType.REQUEST_ROBOT and not any(
            task.requester == agent.agent_id
            and task.object_id == command.parameters["object"]
            and task.status == RobotTaskStatus.PENDING
            for task in self.robot_tasks.values()
        ):
            raise ValueError("Robot request verification failed")
        if command.action == ActionType.HANDOVER:
            receiver = self.agents.get(str(command.target))
            if receiver is not None and receiver.state.held_object != command.parameters.get(
                "object"
            ):
                raise ValueError("Handover result verification failed")

    def _consume_held_object(self, agent: EmployeeAgent, object_id: str | None) -> None:
        self.world.remove_relation(agent.agent_id, RelationType.HOLDS, object_id)
        attributes = self.world.object(object_id).attributes
        attributes["consumed"] = True
        attributes["available"] = False
        self.reservations.release(object_id, agent.agent_id)
        agent.state.held_object = None

    def _reserve_for_action(self, command: ActionCommand) -> None:
        resource = None
        if command.action in {ActionType.SIT, ActionType.REST, ActionType.PICK_UP}:
            resource = command.target
        elif command.action == ActionType.REQUEST_ROBOT:
            resource = command.parameters["object"]
        if resource is not None and not self.reservations.reserve(resource, command.agent_id):
            raise RuntimeError(f"Reservation race for '{resource}'")

    def _release_failed_action(self, command: ActionCommand) -> None:
        if command.target is not None:
            self.reservations.release(command.target, command.agent_id)
        if command.action == ActionType.REQUEST_ROBOT:
            self.reservations.release(command.parameters["object"], command.agent_id)

    def _leave_occupied_location(self, agent: EmployeeAgent) -> None:
        location = agent.state.location
        if location in self.world.objects:
            self.world.remove_relation(location, RelationType.OCCUPIED_BY, agent.agent_id)
            self.reservations.release(location, agent.agent_id)

    def _require_target(self, target: str | None, errors: list[str]) -> None:
        if target is None:
            errors.append("Action requires a target")

    def _require_type(self, target: str | None, object_type: ObjectType, errors: list[str]) -> None:
        self._require_target(target, errors)
        if target in self.world.objects and self.world.object(target).object_type != object_type:
            errors.append(f"Target '{target}' must be {object_type.value}")

    @staticmethod
    def _require_location(agent: EmployeeAgent, target: str | None, errors: list[str]) -> None:
        if target is not None and agent.state.location != target:
            errors.append(
                f"Agent is at '{agent.state.location}', not at required location '{target}'"
            )

    def _require_available(self, target: str | None, requester: str, errors: list[str]) -> None:
        if target is not None and not self.reservations.is_available(target, requester):
            errors.append(f"Target '{target}' is reserved by '{self.reservations.owner(target)}'")

    def _has_active_robot_task(self, agent_id: str) -> bool:
        return any(
            task.requester == agent_id
            and task.status in {RobotTaskStatus.PENDING, RobotTaskStatus.RUNNING}
            for task in self.robot_tasks.values()
        )

    @staticmethod
    def _normalize_target(target: str | None) -> str | None:
        return "stretch_3" if target == "stretch" else target

    def _advance_clock(self, minutes: float) -> None:
        self.elapsed_minutes += minutes
        self.minute_of_day += minutes
        while self.minute_of_day >= 24 * 60:
            self.minute_of_day -= 24 * 60
            self.day += 1
            self._start_day()

    def _state_machine_step(self, elapsed_minutes: float) -> None:
        for agent in self.agents.values():
            if agent.state.availability == "in_conversation":
                continue
            if agent.executor.is_busy:
                supported = getattr(self.action_driver, "supported_actions", frozenset())
                if (
                    self.action_driver is not None
                    and agent.executor.command is not None
                    and agent.executor.command.action in supported
                ):
                    result = self.action_driver.poll(agent.executor)
                    agent.executor.status = result.status
                    agent.executor.phase = result.phase
                    if result.handle is not None:
                        agent.executor.driver_handle = result.handle
                    agent.executor.error = result.error
                    if result.status == ExecutionStatus.SUCCEEDED:
                        self._complete_action(agent)
                    elif result.status in {
                        ExecutionStatus.FAILED,
                        ExecutionStatus.CANCELLED,
                        ExecutionStatus.TIMED_OUT,
                    }:
                        self._fail_action(agent, result.error or result.status.value)
                else:
                    agent.executor.remaining_minutes -= elapsed_minutes
                    if agent.executor.remaining_minutes <= 0.0:
                        self._complete_action(agent)
            if not agent.executor.is_busy and agent.planner.action_queue:
                command = agent.planner.pop_action()
                if command is not None:
                    validation = self.submit_action(command)
                    if not validation.valid:
                        agent.planner.clear_plan()

    def _run_utility_decisions(self) -> None:
        for agent in self.agents.values():
            if (
                agent.executor.is_busy
                or agent.state.availability == "in_conversation"
                or agent.planner.action_queue
                or self._has_active_robot_task(agent.agent_id)
            ):
                continue
            plan = agent.planner.choose_plan(
                agent,
                self.world,
                self.minute_of_day,
                self.day,
                self.seed,
            )
            self._emit(
                "plan_selected",
                agent.agent_id,
                {
                    "goal": plan.goal.value,
                    "score": plan.score,
                    "variant": plan.variant,
                    "actions": [action.action.value for action in plan.actions],
                },
            )

    def _next_utility_interval(self) -> float:
        low, high = self.utility_interval_seconds
        digest = hashlib.sha256(
            f"{self.seed}:{self.day}:utility:{self._utility_decision_count}".encode("utf-8")
        ).digest()
        ratio = int.from_bytes(digest[:8], "big") / float(2**64 - 1)
        self._utility_decision_count += 1
        return low + (high - low) * ratio

    def _start_day(self) -> None:
        for snack in self.world.objects_of_type(ObjectType.SNACK):
            snack.attributes["available"] = True
        for document in self.world.objects_of_type(ObjectType.DOCUMENT):
            document.attributes["urgent"] = False
        for agent_id, agent in self.agents.items():
            self.queue_llm_event(
                LLMTrigger.DAY_START,
                agent_id,
                self._day_start_llm_context(agent),
            )
        if not self.daily_events_enabled:
            return
        generated = self.office_event_generator.generate(
            self.day,
            self.world,
            tuple(self.agents),
        )
        for event in generated:
            self.office_event_generator.apply(event, self.world)
            self.daily_office_events.append(event)
            self._emit(
                "daily_office_event",
                event.details.get("requester", "system"),
                {"event_type": event.event_type, "target": event.target},
            )
            requester = event.details.get("requester")
            if requester in self.agents:
                self.queue_llm_event(
                    LLMTrigger.NEW_TASK,
                    requester,
                    {
                        "event_type": event.event_type,
                        "target": event.target,
                        "current_location": self.agents[requester].state.location,
                        "valid_objects": self._llm_object_catalog(),
                    },
                )

    def _day_start_llm_context(self, agent: EmployeeAgent) -> dict[str, Any]:
        return {
            "purpose": "generate_schedule_draft",
            "workday": {"start": "09:00", "end": "18:00"},
            "profile": {
                "role": agent.profile.role,
                "department": agent.profile.department,
                "personality": agent.profile.personality,
                "preferences": agent.profile.preferences,
            },
            "needs": {
                "hunger": agent.needs.hunger,
                "thirst": agent.needs.thirst,
                "fatigue": agent.needs.fatigue,
            },
            "current_location": agent.state.location,
            "existing_schedule": [
                {
                    "id": item.item_id,
                    "start": f"{item.start_minute // 60:02d}:{item.start_minute % 60:02d}",
                    "end": f"{item.end_minute // 60:02d}:{item.end_minute % 60:02d}",
                    "activity": item.activity,
                    "location": item.location,
                    "variation_minutes": item.variation_minutes,
                }
                for item in agent.schedule.items
            ],
            "valid_objects": self._llm_object_catalog(),
        }

    def _llm_object_catalog(self) -> dict[str, list[str]]:
        catalog: dict[str, list[str]] = {}
        for semantic_object in self.world.objects.values():
            catalog.setdefault(semantic_object.object_type.value, []).append(
                semantic_object.object_id
            )
        return {object_type: sorted(object_ids) for object_type, object_ids in catalog.items()}

    def _record_failure(self, agent_id: str, context: dict[str, Any]) -> None:
        agent = self.agents.get(agent_id)
        if agent is not None:
            reason = context.get("error") or "; ".join(context.get("errors", ()))
            agent.state.last_failure = str(reason) if reason else "Action validation failed"
            agent.state.blocked_reason = agent.state.last_failure
        failures = self._consecutive_failures.get(agent_id, 0) + 1
        self._consecutive_failures[agent_id] = failures
        if failures >= 3:
            self.queue_llm_event(
                LLMTrigger.REPEATED_FAILURE,
                agent_id,
                {"failures": failures, **context},
            )
            self._consecutive_failures[agent_id] = 0

    def _conversation_perception(
        self, semantic_snapshot: dict[str, Any], participants: tuple[str, str]
    ) -> ConversationPerception:
        """Adapt live, offline, and legacy observation payloads at one boundary."""
        return ConversationPerception.from_snapshot(
            semantic_snapshot,
            participants,
            now=self.elapsed_minutes,
            max_age=self.conversation_max_observation_age,
        )

    def _close_conversation(self, session: ConversationSession, event: str) -> None:
        locks = self._conversation_locks.pop(session.session_id, {})
        for agent_id, (attention_target, availability, current_goal) in locks.items():
            agent = self.agents[agent_id]
            if agent.state.conversation_id == session.session_id:
                agent.state.attention_target = attention_target
                agent.state.set_availability(availability)
                agent.state.current_goal = current_goal
                agent.state.conversation_id = None
                if session.status == ConversationStatus.FAILED:
                    agent.state.stress = min(1.0, agent.state.stress + 0.05)
            self._remember(
                agent,
                event,
                {
                    "session_id": session.session_id,
                    "status": session.status.value,
                    "reason": session.failure_reason,
                },
            )
            self._emit(
                event,
                agent_id,
                {
                    "session_id": session.session_id,
                    "status": session.status.value,
                    "reason": session.failure_reason,
                },
            )

    def _validate_robot_dialogue_turn(
        self, session: ConversationSession, intent: ConversationIntent | str
    ) -> None:
        """Restrict robot dialogue to facts already accepted by the runtime."""
        try:
            normalized = ConversationIntent(intent)
        except ValueError:
            return
        task = (
            None if session.robot_task_id is None else self.robot_tasks.get(session.robot_task_id)
        )
        if normalized == ConversationIntent.REQUEST:
            if task is None:
                raise ValueError("Robot request text requires a validated RobotTask")
        elif normalized == ConversationIntent.CLARIFY:
            if task is None or task.status not in {
                RobotTaskStatus.PENDING,
                RobotTaskStatus.RUNNING,
            }:
                raise ValueError("Robot clarification requires a pending accepted task")
        elif normalized == ConversationIntent.ACKNOWLEDGE:
            if task is None:
                raise ValueError("Robot acknowledgement requires an accepted RobotTask")
        elif normalized == ConversationIntent.HANDOVER_CONFIRM:
            if task is None or not (
                task.status == RobotTaskStatus.SUCCEEDED
                and task.robot_release_confirmed
                and task.npc_attachment_confirmed
                and task.interaction_confirmed
            ):
                raise ValueError(
                    "Handover confirmation requires release, attachment, and barrier receipts"
                )

    def _remember(self, agent: EmployeeAgent, event: str, details: dict[str, Any]) -> None:
        agent.memory.remember(MemoryEntry(self.elapsed_minutes, event, details))

    def _emit(self, event: str, agent_id: str, details: dict[str, Any]) -> None:
        self.events.append(RuntimeEvent(self.elapsed_minutes, event, agent_id, details))
