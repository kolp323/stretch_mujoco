"""Deterministic office runtime for employee actions, conflicts, and robot tasks."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable

from stretch_mujoco.semantics import ObjectType, RelationType, SemanticWorld

from .actions import (
    ActionCommand,
    ActionExecution,
    ActionType,
    ConversationReceipt,
    ConversationRequest,
    ExecutionStatus,
    RobotTask,
    RobotTaskStatus,
    RobotTaskType,
    compile_robot_task_type,
    RuntimeEvent,
    ValidationResult,
    required_capability,
)
if TYPE_CHECKING:
    from .robot_task_driver import RobotTaskExecutor
from .employee import EmployeeAgent, PlanCheckpoint
from .dialogue_policy import DialoguePolicy, DialoguePolicyConfig
from .conversation import (
    DEFAULT_MAX_OBSERVATION_AGE,
    ConversationCoordinator,
    ConversationEvent,
    ConversationPhase,
    DialogueAct,
    DialogueCandidate,
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
from .utility import UtilityContext
from .desk_work import (
    WORK_DURATION_SECONDS_PARAMETER,
    WORK_SESSION_ID_PARAMETER,
    WORK_SESSION_SEAT_PARAMETER,
    WORK_SESSION_COMPUTER_PARAMETER,
    WORK_SESSION_WORKSTATION_PARAMETER,
    REQUESTED_WORK_DURATION_SECONDS_PARAMETER,
    chair_for_workstation,
    desk_work_session_actions,
    is_desk_work_session,
)

ACTION_DURATIONS_MINUTES = {
    ActionType.IDLE: 1.0,
    ActionType.MOVE_TO: 3.0,
    ActionType.SIT: 0.5,
    ActionType.STAND_UP: 0.5,
    ActionType.WORK: 15.0,
    ActionType.REST: 10.0,
    ActionType.EAT: 4.0,
    ActionType.DRINK: 2.0,
    ActionType.PICK_UP: 0.5,
    ActionType.PUT_DOWN: 0.5,
    ActionType.REQUEST_ROBOT: 0.2,
    ActionType.USE_COMPUTER: 10.0,
    ActionType.TALK: 0.5,
    ActionType.GESTURE_POINT: 0.25,
    ActionType.GESTURE_WAVE: 0.25,
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
        robot_task_driver: "RobotTaskExecutor | None" = None,
        interaction_driver: object | None = None,
        conversation_policy: DialoguePolicy | None = None,
        population_npc_ids: Iterable[str] | None = None,
        population_interaction_templates: dict[str, object] | None = None,
        interaction_station_allocator: object | None = None,
        population_schema_version: int | None = None,
        trajectory_profile_path: Path | None = None,
        conversation_max_observation_age: float = DEFAULT_MAX_OBSERVATION_AGE,
    ) -> None:
        if state_machine_hz <= 0 or needs_hz <= 0 or minutes_per_second <= 0:
            raise ValueError("Agent update frequencies and clock rate must be positive")
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
        self._desk_work_sessions: dict[str, dict[str, str | None]] = {}
        self._next_desk_work_session_number = 1
        self.conversations = ConversationCoordinator()
        self.social_conversations = NpcConversationScheduler()
        self.conversation_max_observation_age = conversation_max_observation_age
        self._conversation_locks: dict[
            str, dict[str, tuple[str | None, AgentAvailability, str, PlanCheckpoint]]
        ] = {}
        self.events: list[RuntimeEvent] = []
        self._pending_events: list[RuntimeEvent] = []
        self.llm = EventDrivenLLMGateway(llm_daily_budget)
        self.llm_provider_config = llm_provider_config
        self.action_driver = action_driver
        # The driver owns real robot terminal receipts.  Runtime owns only the
        # semantic commit after that receipt has been observed.
        self.robot_task_driver = robot_task_driver
        self.interaction_driver = interaction_driver
        self.conversation_policy = conversation_policy or DialoguePolicy(runtime_seed=seed)
        # ``None`` preserves the legacy schema-v1 employee configuration.
        # Schema-v2 callers use this to select population-aware MuJoCo bindings.
        self.population_npc_ids = (
            None if population_npc_ids is None else frozenset(population_npc_ids)
        )
        self.population_interaction_templates = dict(population_interaction_templates or {})
        self.interaction_station_allocator = interaction_station_allocator
        self.population_schema_version = population_schema_version
        self.trajectory_profile_path = trajectory_profile_path
        self.office_event_generator = DailyOfficeEventGenerator(seed)
        self.daily_events_enabled = daily_events
        self.daily_office_events: list[DailyOfficeEvent] = []
        self._state_machine_accumulator = 0.0
        self._needs_accumulator = 0.0
        self._utility_seconds_remaining = 0.0
        self._utility_decision_count = 0
        self._consecutive_failures: dict[str, int] = {}
        self._committed_execution_ids: set[str] = set()
        self._participant_reservations: dict[str, str] = {}
        self._event_sequence = 0
        self._applied_event_ids: set[str] = set()
        self._conversation_receipts: dict[str, ConversationReceipt] = {}
        self._finished_conversation_ids: set[str] = set()
        # A candidate is retained only while its physical talk command is in flight.
        # Transcript/memory effects are delayed until the terminal receipt arrives.
        self._pending_dialogue_candidates: dict[tuple[str, str], tuple[DialogueCandidate, bool]] = (
            {}
        )
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
        daily_events: bool | None = None,
    ) -> "OfficeAgentRuntime":
        source_path = Path(path).resolve()
        payload = json.loads(source_path.read_text(encoding="utf-8"))
        clock = payload.get("clock", {})
        frequencies = payload.get("frequencies", {})
        utility_interval = frequencies.get("utility_interval_seconds", [5.0, 15.0])
        start_hour, start_minute = (int(part) for part in clock.get("start", "09:00").split(":"))
        version = payload.get("schema_version")
        if version in {2, 3}:
            from stretch_mujoco.npc.schema import NpcPopulation
            from .interaction_stations import (
                InteractionStationAllocator,
                InteractionStationCatalog,
            )

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
            world.register_population_npcs(population.npcs)
            # The report is confidential to the owning employee in the base
            # graph, but Alex is the authored NPC requester in the control
            # sequence.  Add that permission only after population binding so
            # the schema-v1 world remains valid for legacy scenes without NPC
            # bodies.
            if "npc_alex_chen" in agents and "document_report" in world.objects:
                world.add_relation("document_report", RelationType.ALLOWED_FOR, "npc_alex_chen")
            population_npc_ids = population.npcs.keys()
            population_interaction_templates = population.interaction_templates
            interaction_station_allocator = (
                InteractionStationAllocator(
                    InteractionStationCatalog.from_population(population),
                    route_cost=lambda _participant, _actor_kind, _site: None,
                )
                if version == 3
                else None
            )
            trajectory_profile_path = (
                None
                if population.trajectory_profile is None
                else population.resolve_path(population.trajectory_profile)
            )
            try:
                conversation_policy = DialoguePolicy(
                    DialoguePolicyConfig.from_dict(population.dialogue_policy or {}),
                    runtime_seed=int(payload.get("seed", 0) if seed is None else seed),
                )
            except (TypeError, ValueError) as error:
                raise ValueError(f"Invalid population dialogue_policy: {error}") from error
        elif version == 1:
            agents = {
                agent_id: EmployeeAgent.from_dict(agent_id, definition)
                for agent_id, definition in payload.get("employees", {}).items()
            }
            population_npc_ids = None
            population_interaction_templates = None
            interaction_station_allocator = None
            trajectory_profile_path = None
            conversation_policy = None
        else:
            raise ValueError(
                f"Unsupported office agent schema_version {version!r}; expected 1, 2, or 3"
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
            daily_events=bool(payload.get("daily_events", True)) if daily_events is None else daily_events,
            llm_provider_config=llm_provider_config,
            action_driver=action_driver,
            population_npc_ids=population_npc_ids,
            population_interaction_templates=population_interaction_templates,
            interaction_station_allocator=interaction_station_allocator,
            population_schema_version=int(version),
            trajectory_profile_path=trajectory_profile_path,
            conversation_policy=conversation_policy,
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
        capability = required_capability(command.action)
        if (
            capability is not None
            and agent.capabilities is not None
            and capability not in agent.capabilities
        ):
            errors.append(
                f"Action '{command.action.value}' requires capability '{capability}'"
            )
        if agent.executor.is_busy:
            errors.append(
                f"Agent '{command.agent_id}' is already executing "
                f"'{agent.executor.command.action.value}'"
            )

        target = self._normalize_target(command.target)
        social_cue = command.action in {
            ActionType.TALK,
            ActionType.GESTURE_POINT,
            ActionType.GESTURE_WAVE,
        }
        if (
            target is not None
            and target not in self.world.objects
            and not (social_cue and target in self.agents)
        ):
            errors.append(f"Target '{target}' does not exist")

        candidate_actions = getattr(self.action_driver, "candidate_actions", frozenset())
        supported_actions = getattr(self.action_driver, "supported_actions", frozenset())
        embodied_only_unsupported = getattr(
            self.action_driver, "embodied_only_unsupported_actions", frozenset()
        )
        if command.action in embodied_only_unsupported:
            errors.append(f"Action '{command.action.value}' has no embodied receipt implementation")
        if (
            self.action_driver is not None
            and command.action in candidate_actions
            and command.action not in supported_actions
            and command.action != ActionType.REQUEST_ROBOT
        ):
            errors.append(
                f"Action '{command.action.value}' is unavailable in the active NPC asset bundle"
            )

        if command.action == ActionType.MOVE_TO:
            self._require_target(target, errors)
        elif command.action == ActionType.SIT:
            self._require_seat_target(target, errors)
            if self.action_driver is None or ActionType.SIT not in supported_actions:
                self._require_location(agent, target, errors)
            self._require_available(target, command.agent_id, errors)
        elif command.action == ActionType.STAND_UP:
            self._require_seat_target(target, errors)
            self._require_location(agent, target, errors)
            if not self.world.find_relations(
                subject=target,
                relation=RelationType.OCCUPIED_BY,
                object_id=command.agent_id,
            ):
                errors.append("Agent is not occupying the target chair")
        elif command.action == ActionType.WORK:
            self._require_type(target, ObjectType.WORKSTATION, errors)
            if target in self.world.objects:
                try:
                    chair = chair_for_workstation(self.world, target)
                except ValueError as error:
                    errors.append(str(error))
                else:
                    if is_desk_work_session(command):
                        session_id = command.parameters.get(WORK_SESSION_ID_PARAMETER)
                        session = (
                            self._desk_work_sessions.get(session_id)
                            if isinstance(session_id, str)
                            else None
                        )
                        if session is None or session["agent_id"] != command.agent_id:
                            errors.append("Desk work requires a runtime-issued session ID")
                        if command.parameters.get(WORK_SESSION_SEAT_PARAMETER) != chair:
                            errors.append("Desk work session seat does not match workstation")
                        self._require_location(agent, chair, errors)
                        if not self.world.find_relations(
                            subject=chair,
                            relation=RelationType.OCCUPIED_BY,
                            object_id=command.agent_id,
                        ):
                            errors.append(
                                "Agent must be seated at the workstation chair before work"
                            )
                        duration = command.parameters.get(WORK_DURATION_SECONDS_PARAMETER)
                        if (
                            not isinstance(duration, (int, float))
                            or isinstance(duration, bool)
                            or not math.isfinite(duration)
                            or duration <= 0
                        ):
                            errors.append(
                                "Desk work session duration must be a positive finite number"
                            )
                    elif not self.reservations.is_available(chair, command.agent_id):
                        errors.append(
                            f"Target '{chair}' is reserved by "
                            f"'{self.reservations.owner(chair)}'"
                        )
        elif command.action == ActionType.REST:
            self._require_seat_target(target, errors)
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
                if len(workstations) != 1 or workstations[0].object_type != ObjectType.WORKSTATION:
                    errors.append("Computer must belong to exactly one workstation")
        elif social_cue:
            if target is None or target not in self.agents:
                errors.append("Social cue target must be an agent")
            elif target == command.agent_id:
                errors.append("Agent cannot target itself with a social cue")
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
            if target not in self.agents:
                errors.append("phase_2c_npc_handover_only")
            actor_mode = command.parameters.get("actor_mode", "npc_to_npc")
            if actor_mode != "npc_to_npc":
                errors.append("phase_2c_npc_handover_only")
            preferred_station_id = command.parameters.get("preferred_station_id")
            if preferred_station_id is not None and (
                not isinstance(preferred_station_id, str) or not preferred_station_id
            ):
                errors.append("preferred_station_id must be a non-empty string")
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
        if normalized.action == ActionType.USE_COMPUTER:
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
            assert normalized.target is not None
            workstation = self.world.related_objects(normalized.target, RelationType.ON)[0]
            normalized = ActionCommand(
                normalized.agent_id,
                ActionType.WORK,
                workstation.object_id,
                {
                    "_requested_action": ActionType.USE_COMPUTER.value,
                    WORK_SESSION_COMPUTER_PARAMETER: normalized.target,
                    **normalized.parameters,
                },
            )
        if normalized.action == ActionType.WORK and not is_desk_work_session(normalized):
            return self._submit_desk_work_session(normalized)
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
            self._record_action_driver_evidence(agent.executor, result)
            if result.status in {
                ExecutionStatus.FAILED,
                ExecutionStatus.CANCELLED,
                ExecutionStatus.TIMED_OUT,
            }:
                self._fail_action(agent, result.error or result.status.value)
        return validation

    def _submit_desk_work_session(self, command: ActionCommand) -> ValidationResult:
        """Expand a public work request into receipt-gated seated work actions."""
        validation = self.validate_action(command)
        if not validation.valid:
            self._emit(
                "action_rejected",
                command.agent_id,
                {"action": command.action.value, "errors": list(validation.errors)},
            )
            if command.agent_id in self.agents:
                self._record_failure(
                    command.agent_id,
                    {"action": command.action.value, "errors": validation.errors},
                )
            return validation
        agent = self.agents[command.agent_id]
        if agent.planner.action_queue:
            return ValidationResult(False, ("Agent has a pending plan",))
        assert command.target is not None
        default_duration_seconds = (
            ACTION_DURATIONS_MINUTES[ActionType.WORK] / self.minutes_per_second
        )
        requested_duration = command.parameters.get(
            REQUESTED_WORK_DURATION_SECONDS_PARAMETER, default_duration_seconds
        )
        if (
            not isinstance(requested_duration, (int, float))
            or isinstance(requested_duration, bool)
            or not math.isfinite(requested_duration)
            or requested_duration <= 0
        ):
            return ValidationResult(False, ("duration_seconds must be a positive finite number",))
        session_id = f"desk_work_{self._next_desk_work_session_number:06d}"
        self._next_desk_work_session_number += 1
        try:
            actions = desk_work_session_actions(
                agent,
                self.world,
                command.target,
                duration_seconds=float(requested_duration),
                session_id=session_id,
                computer_id=(
                    str(command.parameters[WORK_SESSION_COMPUTER_PARAMETER])
                    if WORK_SESSION_COMPUTER_PARAMETER in command.parameters
                    else None
                ),
            )
        except ValueError as error:
            return ValidationResult(False, (str(error),))
        chair = str(actions[-2].target)
        self._desk_work_sessions[session_id] = {
            "agent_id": command.agent_id,
            "status": ExecutionStatus.RUNNING.value,
            "error": None,
        }
        self._emit(
            "desk_work_session_started",
            command.agent_id,
            {
                "workstation": command.target,
                "computer": command.parameters.get(WORK_SESSION_COMPUTER_PARAMETER),
                "seat_slot": chair,
                "session_id": session_id,
            },
        )
        agent.planner.action_queue.extend(actions[1:])
        first_result = self.submit_action(actions[0])
        if not first_result.valid:
            agent.planner.clear_plan()
            self._desk_work_sessions[session_id]["status"] = ExecutionStatus.FAILED.value
            self._desk_work_sessions[session_id]["error"] = "; ".join(first_result.errors)
        return first_result

    def desk_work_session_status(self, session_id: str) -> tuple[ExecutionStatus, str | None]:
        """Expose terminal receipt-chain state to sequence executors."""
        session = self._desk_work_sessions.get(session_id)
        if session is None:
            return ExecutionStatus.FAILED, "unknown_desk_work_session"
        return ExecutionStatus(str(session["status"])), session["error"]

    def tick(
        self, seconds: float, semantic_snapshot: dict[str, Any] | None = None
    ) -> tuple[RuntimeEvent, ...]:
        if seconds < 0:
            raise ValueError("Runtime tick duration cannot be negative")
        minutes = seconds * self.minutes_per_second
        self._advance_clock(minutes)

        for session in self.conversations.expire(self.elapsed_minutes):
            if any(
                self._participant_reservations.get(participant) == session.session_id
                for participant in session.participants
            ):
                self._finish_conversation(session.session_id, "conversation_turn_timeout")
            else:
                self._close_conversation(session, ConversationEvent.TIMED_OUT.value)

        # A timeout wins over a receipt observed in the same tick: terminal effects
        # must not be committed after the session deadline.
        self._advance_embodied_conversations()

        if self.robot_task_driver is not None:
            self.robot_task_driver.tick(self, seconds)

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
        agent = self.agents[agent_id]
        event_context = dict(context or {})
        profile_context = dict(event_context.get("profile") or {})
        profile_context.update(
            {
                "role": agent.profile.role,
                "department": agent.profile.department,
                "personality": dict(agent.profile.personality),
                "preferences": dict(agent.profile.preferences),
            }
        )
        event_context["profile"] = profile_context
        queued = self.llm.queue(
            trigger,
            agent_id,
            self.day,
            self.minute_of_day,
            event_context,
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
        effects = [key for key in ("schedule", "action", "dialogue") if key in response]
        if len(effects) > 1:
            return ValidationResult(False, ("ambiguous_llm_response",))
        if "dialogue" in response and isinstance(response["dialogue"], dict):
            payload = response["dialogue"]
            try:
                candidate = DialogueCandidate(
                    request_id=str(payload.get("request_id", request.request_id)),
                    session_id=str(payload["session_id"]),
                    turn_id=str(payload["turn_id"]),
                    speaker=str(payload["speaker"]),
                    listener=str(payload["listener"]),
                    act=DialogueAct(str(payload["act"])),
                    text=str(payload["text"]),
                    proposed_at=self.elapsed_minutes,
                )
            except (KeyError, TypeError, ValueError) as error:
                return ValidationResult(False, (f"invalid_dialogue_candidate:{error}",))
            return self.submit_dialogue_candidate(candidate)
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

    def begin_conversation(self, request: ConversationRequest) -> ConversationReceipt:
        """Atomically reserve participants and begin an embodied conversation.

        New callers use this strict entry point.  The older ``start_conversation``
        remains a logical/mock compatibility adapter and is intentionally not
        upgraded to claim an embodied preparation receipt.
        """
        prior = self._conversation_receipts.get(request.session_id)
        if prior is not None:
            session = self.conversations.sessions.get(request.session_id)
            if (
                session is not None
                and session.participants == request.participants
                and session.topic == request.topic
            ):
                return prior
            return ConversationReceipt(request.session_id, False, "rejected", "session_id_conflict")
        if len(request.participants) < 2 or len(request.participants) != len(
            set(request.participants)
        ):
            receipt = ConversationReceipt(
                request.session_id, False, "rejected", "invalid_participants"
            )
            self._conversation_receipts[request.session_id] = receipt
            return receipt
        if not request.topic.strip() or request.timeout <= 0 or request.max_turns <= 0:
            receipt = ConversationReceipt(
                request.session_id, False, "rejected", "invalid_conversation_request"
            )
            self._conversation_receipts[request.session_id] = receipt
            return receipt
        if self.interaction_driver is None:
            return ConversationReceipt(
                request.session_id, False, "rejected", "physical_driver_required"
            )
        if any(participant not in self.agents for participant in request.participants):
            return ConversationReceipt(
                request.session_id,
                False,
                "rejected",
                "phase_2b_npc_conversation_only",
            )
        if request.semantic_snapshot is None:
            return ConversationReceipt(
                request.session_id, False, "rejected", "observation_required"
            )
        for participant in request.participants:
            if participant not in self.world.objects:
                return ConversationReceipt(
                    request.session_id, False, "rejected", "unknown_participant"
                )
            if participant in self._participant_reservations:
                return ConversationReceipt(
                    request.session_id, False, "rejected", "participant_busy"
                )
        agent_participants = tuple(
            participant for participant in request.participants if participant in self.agents
        )
        if any(self.agents[participant].executor.is_busy for participant in agent_participants):
            return ConversationReceipt(request.session_id, False, "rejected", "participant_busy")
        try:
            perception = self._conversation_perception(
                request.semantic_snapshot, request.participants
            )
            session = self.conversations.start(
                request.participants,
                request.topic,
                self.elapsed_minutes,
                perception.poses,
                timeout=request.timeout,
                session_id=request.session_id,
                max_turns=request.max_turns,
                parent_event_id=request.correlation_id,
                require_spatial_ready=False,
            )
            session.preferred_station_id = request.preferred_station_id
            session.compatibility_mode = {
                2: "population_v2_legacy_adapter",
                3: "population_v3_candidate",
            }.get(self.population_schema_version)
            session.production_evidence = False
            self.conversations.mark_approaching(session.session_id)
            for participant in request.participants:
                self._participant_reservations[participant] = session.session_id
            self._lock_conversation_agents(session)
            result = self.interaction_driver.prepare_conversation(session)
            self._record_conversation_driver_evidence(session, result)
            if result.status == ExecutionStatus.SUCCEEDED:
                self.conversations.mark_aligned(session.session_id, (result.handle or "ready",))
                receipt = self._conversation_receipt(session, True, "ready")
            elif result.status == ExecutionStatus.RUNNING:
                receipt = self._conversation_receipt(session, True, "approaching")
            else:
                raise RuntimeError(result.error or "conversation_prepare_failed")
        except (KeyError, RuntimeError, ValueError) as error:
            session = self.conversations.sessions.get(request.session_id)
            if session is not None:
                self.conversations.fail(session.session_id, str(error))
                self._finish_conversation(session.session_id, "conversation_prepare_failed")
            else:
                for participant in request.participants:
                    self._participant_reservations.pop(participant, None)
            session = self.conversations.sessions.get(request.session_id)
            receipt = (
                ConversationReceipt(request.session_id, False, "failed", str(error))
                if session is None
                else self._conversation_receipt(session, False, "failed", str(error))
            )
        self._conversation_receipts[request.session_id] = receipt
        return receipt

    def conversation(self, session_id: str) -> ConversationSession:
        return self.conversations.sessions[session_id]

    def conversation_receipt(self, session_id: str) -> ConversationReceipt:
        return self._conversation_receipts[session_id]

    def active_conversations(self) -> tuple[ConversationSession, ...]:
        return tuple(
            session
            for session in self.conversations.sessions.values()
            if not session.status.terminal
        )

    def submit_dialogue_candidate(self, candidate: DialogueCandidate) -> ValidationResult:
        """Commit a turn only after the interaction driver returns a real receipt."""
        try:
            session = self.conversation(candidate.session_id)
            sanitized = self.conversation_policy.validate_and_sanitize(
                candidate, session, self.world, now=self.elapsed_minutes
            )
            self.conversations.propose(sanitized.candidate)
            if self.interaction_driver is None:
                raise RuntimeError("physical_driver_required")
            turn = session.dialogue_turns[sanitized.candidate.turn_id]
            result = self.interaction_driver.play_turn(session, turn)
            self._record_conversation_driver_evidence(session, result)
            if result.status not in {ExecutionStatus.RUNNING, ExecutionStatus.SUCCEEDED}:
                self.conversations.reject_turn(
                    session.session_id, turn.turn_id, result.error or result.status.value
                )
                self.conversations.fail(
                    session.session_id, result.error or "turn_physical_receipt_required"
                )
                self._finish_conversation(
                    session.session_id, result.error or "turn_physical_receipt_required"
                )
                return ValidationResult(False, (result.error or "turn_physical_receipt_required",))
            execution_id = result.handle or f"conversation:{session.session_id}:{turn.turn_id}"
            self.conversations.mark_turn_started(session.session_id, turn.turn_id, execution_id)
            self._pending_dialogue_candidates[(session.session_id, turn.turn_id)] = (
                sanitized.candidate,
                sanitized.fallback_used,
            )
            if result.status == ExecutionStatus.SUCCEEDED:
                self._commit_dialogue_turn(session, turn.turn_id)
            return ValidationResult(True)
        except (KeyError, RuntimeError, ValueError) as error:
            return ValidationResult(False, (str(error),))

    def _advance_embodied_conversations(self) -> None:
        """Consume only physical terminal receipts for embodied conversation effects."""
        if self.interaction_driver is None:
            return
        poll = getattr(self.interaction_driver, "poll_conversation", None)
        if not callable(poll):
            return
        for session in self.active_conversations():
            if session.phase not in {
                ConversationPhase.APPROACHING,
                ConversationPhase.ALIGNING,
                ConversationPhase.PLAYING_TURN,
            }:
                continue
            result = poll(session.session_id)
            self._record_conversation_driver_evidence(session, result)
            self._conversation_receipts[session.session_id] = self._conversation_receipt(
                session, True, result.phase, result.error
            )
            if result.status == ExecutionStatus.RUNNING:
                if session.phase == ConversationPhase.APPROACHING and result.phase == "align":
                    self.conversations.mark_aligning(session.session_id)
                continue
            if result.status != ExecutionStatus.SUCCEEDED:
                if session.active_turn_id is not None:
                    self.conversations.reject_turn(
                        session.session_id,
                        session.active_turn_id,
                        result.error or "conversation_physical_receipt_failed",
                    )
                self._finish_conversation(
                    session.session_id, result.error or "conversation_physical_receipt_failed"
                )
                continue
            if session.phase in {ConversationPhase.APPROACHING, ConversationPhase.ALIGNING}:
                self.conversations.mark_aligned(session.session_id, (result.handle or "ready",))
            elif session.active_turn_id is not None:
                self._commit_dialogue_turn(session, session.active_turn_id)
            self._conversation_receipts[session.session_id] = self._conversation_receipt(
                session, True, "ready"
            )

    def _commit_dialogue_turn(self, session: ConversationSession, turn_id: str) -> None:
        """Apply transcript, memory, and events after a successful talk receipt only."""
        turn = session.dialogue_turns[turn_id]
        pending = self._pending_dialogue_candidates.pop((session.session_id, turn_id), None)
        candidate, fallback_used = pending if pending is not None else (None, False)
        turn.fallback_used = fallback_used
        self.conversations.commit_turn(session.session_id, turn_id, self.elapsed_minutes)
        if candidate is not None:
            self.conversation_policy.note_committed(candidate, self.elapsed_minutes)
        event_id = self._new_event_id()
        for participant in session.participants:
            agent = self.agents.get(participant)
            if agent is not None:
                self._remember(
                    agent,
                    "dialogue_turn_committed",
                    {
                        "event_id": event_id,
                        "session_id": session.session_id,
                        "turn_id": turn.turn_id,
                        "text": turn.text,
                        "fallback_used": fallback_used,
                    },
                )
        self._emit(
            "dialogue_turn_committed",
            turn.speaker,
            {
                "session_id": session.session_id,
                "turn_id": turn.turn_id,
                "listener": turn.listener,
                "act": turn.act.value,
                "text": turn.text,
                "fallback_used": fallback_used,
                "station_id": session.station_id,
                "lease_id": session.lease_id,
                "physical_receipt_ids": list(session.physical_receipt_ids),
                "compatibility_mode": session.compatibility_mode,
                "production_evidence": session.production_evidence,
            },
            event_id=event_id,
            correlation_id=session.parent_event_id or session.session_id,
        )
        if session.turn >= session.max_turns:
            # Reaching the sequence-declared number of turns is the normal
            # completion condition, not a physical or policy failure.
            self.complete_conversation(session.session_id)

    def interrupt_conversation(self, session_id: str, reason: str) -> ConversationReceipt:
        session = self.conversation(session_id)
        self.conversations.interrupt(session_id, reason, self.elapsed_minutes)
        self._finish_conversation(session_id, reason)
        return self._conversation_receipts[session_id]

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
        locks: dict[str, tuple[str | None, AgentAvailability, str, PlanCheckpoint]] = {}
        for agent_id in agent_participants:
            agent = self.agents[agent_id]
            locks[agent_id] = (
                agent.state.attention_target,
                AgentAvailability(agent.state.availability),
                agent.state.current_goal,
                agent.planner.suspend(session.session_id),
            )
            agent.planner.clear_plan()
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
        self._finish_conversation(session_id, ConversationEvent.COMPLETED.value)
        return session

    def cancel_conversation(
        self, session_id: str, reason: str = "cancelled"
    ) -> ConversationSession:
        """Cancel once and restore the exact locked plan state without touching reservations."""
        session = self.conversations.cancel(session_id, reason)
        self._finish_conversation(session_id, reason)
        return session

    def fail_conversation(self, session_id: str, reason: str) -> ConversationSession:
        """Record a logical/receipt failure and use the common terminal cleanup path."""
        session = self.conversations.fail(session_id, reason)
        self._finish_conversation(session_id, reason)
        return session

    def request_robot_task(
        self,
        session_id: str,
        requester: str,
        *,
        task: str,
        object_id: str,
        destination: str,
        recipient: str | None = None,
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
        supported = getattr(self.action_driver, "supported_actions", frozenset())
        if self.action_driver is not None and ActionType.REQUEST_ROBOT in supported:
            # The old dialogue helper has no NPC request-motion receipt.  In an
            # embodied runtime it must not bypass the request_robot workflow.
            return ValidationResult(False, ("request_robot_requires_embodied_action",))
        command = ActionCommand(
            requester,
            ActionType.REQUEST_ROBOT,
            "stretch_3",
            {
                "task": task,
                "object": object_id,
                "destination": destination,
                "recipient": recipient,
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
        if success and task.task_type is RobotTaskType.ROBOT_TO_NPC_HANDOVER:
            missing = tuple(
                name
                for name, confirmed in (
                    ("robot_release_confirmed", task.robot_release_confirmed),
                    ("npc_attachment_confirmed", task.npc_attachment_confirmed),
                    ("interaction_confirmed", task.interaction_confirmed),
                )
                if not confirmed
            )
            if missing:
                success = False
                error = "robot_handover_evidence_missing:" + ",".join(missing)
        if success and semantic_snapshot is not None:
            verification = self.verify_robot_task_result(task_id, semantic_snapshot)
            if not verification.valid:
                success = False
                error = "; ".join(verification.errors)
        requester = self.agents[task.requester]
        if success:
            task.status = RobotTaskStatus.SUCCEEDED
            if task.task_type is RobotTaskType.PLACE_DELIVERY:
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
            # The renderer/UI may show delivery only after this verified
            # terminal commit; an accepted request is intentionally distinct.
            self._emit(
                "robot_delivery_verified",
                task.requester,
                {
                    "task_id": task_id,
                    "object": task.object_id,
                    "destination": task.destination,
                    "recipient": task.recipient,
                    "task_type": task.task_type.value,
                    "receipt_ids": sorted(task.receipt_ids),
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
        events = tuple(self._pending_events)
        self._pending_events.clear()
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
        recipient = parameters.get("recipient")
        try:
            task_type = compile_robot_task_type(task, recipient)
        except ValueError as error:
            errors.append(str(error))
            task_type = None
        if object_id not in self.world.objects:
            errors.append(f"Requested object '{object_id}' does not exist")
        if destination not in self.world.objects:
            errors.append(f"Destination '{destination}' does not exist")
        if destination in self.agents:
            errors.append("Robot destination must be a location, not an NPC")
        if recipient is not None and not isinstance(recipient, str):
            errors.append("Robot task recipient must be an NPC ID")
        if task_type is RobotTaskType.ROBOT_TO_NPC_HANDOVER:
            if recipient not in self.agents:
                errors.append(f"Recipient '{recipient}' is not an NPC")
            elif (
                self.agents[recipient].capabilities is not None
                and "object_handover" not in self.agents[recipient].capabilities
            ):
                errors.append(f"Recipient '{recipient}' lacks object_handover capability")
        executor = self.robot_task_driver
        if task_type is not None and executor is not None:
            supported = getattr(executor, "supported_task_types", frozenset())
            if task_type not in supported:
                errors.append(f"Robot executor does not support task type '{task_type.value}'")
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
        conversation_id = parameters.get("conversation_id")
        if conversation_id is not None:
            if not isinstance(conversation_id, str):
                errors.append("Robot request conversation_id must be a string")
            else:
                session = self.conversations.sessions.get(conversation_id)
                if (
                    session is None
                    or session.status.terminal
                    or agent.agent_id not in session.participants
                ):
                    errors.append("Robot task has an invalid conversation association")
                elif session.robot_task_id is not None:
                    errors.append("Conversation already has a robot task")

    @staticmethod
    def _record_action_driver_evidence(execution: ActionExecution, result: object) -> None:
        for receipt_id in getattr(result, "receipt_ids", ()):
            if receipt_id not in execution.physical_receipt_ids:
                execution.physical_receipt_ids += (receipt_id,)
        cleanup_evidence_id = getattr(result, "cleanup_evidence_id", None)
        if cleanup_evidence_id is not None:
            execution.cleanup_evidence_id = cleanup_evidence_id
        for attribute in ("station_id", "lease_id", "compatibility_mode"):
            value = getattr(result, attribute, None)
            if value is not None:
                setattr(execution, attribute, value)
        execution.production_evidence = bool(
            getattr(result, "production_evidence", False)
        )

    @staticmethod
    def _handover_event_evidence(execution: ActionExecution) -> dict[str, object]:
        if execution.command is None:
            return {}
        evidence: dict[str, object] = {
            "station_id": execution.station_id,
            "lease_id": execution.lease_id,
            "physical_receipt_ids": list(execution.physical_receipt_ids),
            "cleanup_evidence_id": execution.cleanup_evidence_id,
            "compatibility_mode": execution.compatibility_mode,
            "production_evidence": execution.production_evidence,
        }
        if execution.command.action in {
            ActionType.SIT,
            ActionType.REST,
            ActionType.STAND_UP,
            ActionType.WORK,
        }:
            evidence.update(
                {
                    "seat_slot": execution.command.parameters.get(
                        WORK_SESSION_SEAT_PARAMETER, execution.command.target
                    ),
                    "computer": execution.command.parameters.get(
                        WORK_SESSION_COMPUTER_PARAMETER
                    ),
                    "workstation": execution.command.parameters.get(
                        WORK_SESSION_WORKSTATION_PARAMETER
                    ),
                    "session_id": execution.command.parameters.get(
                        WORK_SESSION_ID_PARAMETER, execution.lease_id
                    ),
                }
            )
        elif execution.command.action != ActionType.HANDOVER:
            return {}
        return evidence

    def _complete_action(self, agent: EmployeeAgent) -> None:
        command = agent.executor.command
        if command is None:
            return
        if agent.executor.execution_id in self._committed_execution_ids:
            return
        try:
            if command.action == ActionType.REQUEST_ROBOT:
                self._prepare_robot_task_commit(agent, command)
            self._apply_action_effect(agent, command)
            self._verify_action_effect(agent, command)
        except (KeyError, ValueError) as error:
            self._fail_action(agent, str(error))
            return
        else:
            self._committed_execution_ids.add(agent.executor.execution_id)
            agent.executor.status = ExecutionStatus.SUCCEEDED
            self._consecutive_failures[agent.agent_id] = 0
            event_id = self._new_event_id()
            handover_evidence = self._handover_event_evidence(agent.executor)
            terminal_receipt_id = (
                agent.executor.physical_receipt_ids[-1]
                if command.action == ActionType.HANDOVER
                and agent.executor.physical_receipt_ids
                else None
            )
            self._remember(
                agent,
                "action_succeeded",
                {
                    "action": command.action.value,
                    "target": command.target,
                    "execution_id": agent.executor.execution_id,
                    "event_id": event_id,
                    **handover_evidence,
                },
            )
            self._emit(
                "action_succeeded",
                agent.agent_id,
                {
                    "action": command.action.value,
                    "target": command.target,
                    "execution_id": agent.executor.execution_id,
                    **handover_evidence,
                },
                event_id=event_id,
                correlation_id=agent.executor.lease_id,
                causation_id=terminal_receipt_id,
            )
            self._emit(
                "semantic_commit",
                agent.agent_id,
                {
                    "action": command.action.value,
                    "execution_id": agent.executor.execution_id,
                    **handover_evidence,
                },
                correlation_id=agent.executor.lease_id,
                causation_id=event_id,
            )
            session_id = command.parameters.get(WORK_SESSION_ID_PARAMETER)
            if command.action == ActionType.IDLE and isinstance(session_id, str):
                session = self._desk_work_sessions.get(session_id)
                if session is not None and session["agent_id"] == agent.agent_id:
                    session["status"] = ExecutionStatus.SUCCEEDED.value
        agent.state.current_action = "idle"
        agent.state.animation_state = "idle"
        agent.state.set_availability(AgentAvailability.AVAILABLE)
        agent.state.attention_target = None

    def _prepare_robot_task_commit(self, agent: EmployeeAgent, command: ActionCommand) -> None:
        """Reject invalid conversation ownership before any semantic mutation."""
        conversation_id = command.parameters.get("conversation_id")
        if conversation_id is None:
            return
        if not isinstance(conversation_id, str):
            raise ValueError("Robot task has an invalid conversation association")
        session = self.conversations.sessions.get(conversation_id)
        if (
            session is None
            or session.status.terminal
            or agent.agent_id not in session.participants
            or session.robot_task_id is not None
        ):
            raise ValueError("Robot task has an invalid conversation association")

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
        session_id = command.parameters.get(WORK_SESSION_ID_PARAMETER)
        if isinstance(session_id, str):
            session = self._desk_work_sessions.get(session_id)
            if session is not None and session["agent_id"] == agent.agent_id:
                session["status"] = ExecutionStatus.FAILED.value
                session["error"] = error
                if command.action != ActionType.WORK:
                    agent.planner.clear_plan()
        handover_evidence = self._handover_event_evidence(agent.executor)
        self._record_failure(
            agent.agent_id,
            {"action": command.action.value, "error": error, **handover_evidence},
        )
        self._release_failed_action(command)
        self._remember(
            agent,
            "action_failed",
            {"action": command.action.value, "error": error, **handover_evidence},
        )
        self._emit(
            "action_failed",
            agent.agent_id,
            {
                "action": command.action.value,
                "error": error,
                "execution_id": agent.executor.execution_id,
                **handover_evidence,
            },
            correlation_id=agent.executor.lease_id,
            causation_id=(
                agent.executor.physical_receipt_ids[-1]
                if agent.executor.physical_receipt_ids
                else None
            ),
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
        elif action == ActionType.STAND_UP:
            self.world.remove_relation(target, RelationType.OCCUPIED_BY, agent.agent_id)
            self.reservations.release(target, agent.agent_id)
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
            supported = getattr(self.action_driver, "supported_actions", frozenset())
            if (
                self.action_driver is not None
                and ActionType.REQUEST_ROBOT in supported
                and agent.executor.phase != "request_accepted"
            ):
                raise ValueError("Robot task requires a live request acceptance receipt")
            task = RobotTask(
                requester=agent.agent_id,
                task=command.parameters["task"],
                object_id=command.parameters["object"],
                destination=command.parameters["destination"],
                recipient=command.parameters.get("recipient"),
                robot_id=str(command.target),
                conversation_id=command.parameters.get("conversation_id"),
                request_receipt_id=f"npc:{agent.executor.execution_id}:request_accepted",
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
            self._emit(
                "robot_request_accepted",
                agent.agent_id,
                {
                    "task_id": task.task_id,
                    "receipt_id": task.request_receipt_id,
                    "robot_id": task.robot_id,
                    "object": task.object_id,
                    "destination": task.destination,
                    "recipient": task.recipient,
                    "task_type": task.task_type.value,
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
        if command.action == ActionType.STAND_UP and self.world.find_relations(
            subject=command.target,
            relation=RelationType.OCCUPIED_BY,
            object_id=agent.agent_id,
        ):
            raise ValueError("Chair occupancy release verification failed")
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

    def _require_seat_target(self, target: str | None, errors: list[str]) -> None:
        self._require_target(target, errors)
        if target not in self.world.objects:
            return
        object_type = self.world.object(target).object_type
        allowed = (
            {ObjectType.SEAT_SLOT}
            if self.population_schema_version == 3
            else {ObjectType.SEAT_SLOT, ObjectType.CHAIR}
        )
        if object_type not in allowed:
            errors.append(f"Target '{target}' must be SeatSlot")

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
                    self._record_action_driver_evidence(agent.executor, result)
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
            context = self._utility_context(agent)
            plan = agent.planner.choose_plan(
                agent,
                self.world,
                self.minute_of_day,
                self.day,
                self.seed,
                context=context,
            )
            self._emit(
                "plan_selected",
                agent.agent_id,
                {
                    "goal": plan.goal.value,
                    "score": plan.score,
                    "variant": plan.variant,
                    "actions": [action.action.value for action in plan.actions],
                    "utility_context": {
                        "schedule_urgency": context.schedule_urgency,
                        "task_priority": context.task_priority,
                        "pending_invitation_priority": context.pending_invitation_priority,
                        "consecutive_failures": context.consecutive_failures,
                        "resource_availability": context.resource_availability,
                        "cooldown_remaining": context.cooldown_remaining,
                    },
                },
            )

    def _utility_context(self, agent: EmployeeAgent) -> UtilityContext:
        """Project runtime-owned constraints into the planner's value-only input."""
        schedule_item = agent.schedule.active_item(
            agent.agent_id, self.minute_of_day, self.day, self.seed
        )
        schedule_urgency = 0.0
        task_priority = 0.0
        if schedule_item is not None:
            _, end = schedule_item.shifted_window(agent.agent_id, self.day, self.seed)
            schedule_urgency = max(0.0, min(1.0, 1.0 - max(0.0, end - self.minute_of_day) / 60.0))
            if schedule_item.activity == "meeting":
                task_priority = schedule_urgency
        pending_invitation_priority = max(
            (
                1.0
                for invitation in self.social_conversations.invitations.values()
                if invitation.accepted is None
                and invitation.expires_at > self.elapsed_minutes
                and invitation.proposal.participant == agent.agent_id
            ),
            default=0.0,
        )
        preferred_objects = (
            agent.profile.preferences.get("snack", "bread_snack"),
            agent.profile.preferences.get("drink", "soda_can"),
        )
        resource_availability = 1.0
        for object_id in preferred_objects:
            if object_id in self.world.objects and not self.reservations.is_available(
                object_id, agent.agent_id
            ):
                resource_availability = 0.0
                break
        return UtilityContext(
            schedule_urgency=schedule_urgency,
            task_priority=task_priority,
            hunger=agent.needs.hunger,
            thirst=agent.needs.thirst,
            fatigue=agent.needs.fatigue,
            social_energy=agent.state.social_energy,
            stress=agent.state.stress,
            pending_invitation_priority=pending_invitation_priority,
            consecutive_failures=self._consecutive_failures.get(agent.agent_id, 0),
            resource_availability=resource_availability,
            cooldown_remaining=self.social_conversations.cooldown_remaining(
                agent.agent_id, self.elapsed_minutes
            ),
            repetition_penalty=agent.planner.recent_goals.count(agent.state.current_goal) * 0.04,
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
        for agent_id, (attention_target, availability, current_goal, checkpoint) in locks.items():
            agent = self.agents[agent_id]
            if agent.state.conversation_id == session.session_id:
                agent.state.attention_target = attention_target
                agent.state.set_availability(availability)
                agent.state.current_goal = current_goal
                agent.state.conversation_id = None
                if not agent.planner.resume(checkpoint, self.world, self.elapsed_minutes):
                    agent.planner.invalidate("conversation_plan_invalidated")
                    self._emit(
                        "plan_resume_invalidated",
                        agent_id,
                        {"session_id": session.session_id, "reason": "invalid_target"},
                    )
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

    def _lock_conversation_agents(self, session: ConversationSession) -> None:
        locks: dict[str, tuple[str | None, AgentAvailability, str, PlanCheckpoint]] = {}
        for participant in session.participants:
            agent = self.agents.get(participant)
            if agent is None:
                continue
            partner = next(
                candidate for candidate in session.participants if candidate != participant
            )
            locks[participant] = (
                agent.state.attention_target,
                AgentAvailability(agent.state.availability),
                agent.state.current_goal,
                agent.planner.suspend(session.session_id),
            )
            agent.planner.clear_plan()
            agent.state.begin_conversation(session.session_id, partner)
        self._conversation_locks[session.session_id] = locks

    def _finish_conversation(self, session_id: str, reason: str) -> None:
        """Single cleanup path for receipt failure, timeout, completion, and interruption."""
        if session_id in self._finished_conversation_ids:
            return
        session = self.conversations.sessions.get(session_id)
        if session is None:
            return
        for key in tuple(self._pending_dialogue_candidates):
            if key[0] == session_id:
                self._pending_dialogue_candidates.pop(key, None)
        if not session.status.terminal:
            self.conversations.fail(session_id, reason)
        if self.interaction_driver is not None:
            if session.status == ConversationStatus.COMPLETED:
                finish = getattr(self.interaction_driver, "finish_conversation", None)
                result = finish(session_id, "succeeded") if callable(finish) else None
            else:
                outcome = {
                    ConversationStatus.CANCELLED: "cancelled",
                    ConversationStatus.TIMED_OUT: "timed_out",
                }.get(session.status, "failed")
                terminate = getattr(self.interaction_driver, "terminate_conversation", None)
                if callable(terminate):
                    result = terminate(session_id, outcome, reason)
                else:
                    cancel = getattr(self.interaction_driver, "cancel_conversation", None)
                    result = cancel(session_id, reason) if callable(cancel) else None
            if result is not None:
                self._record_conversation_driver_evidence(session, result)
        for participant in session.participants:
            if self._participant_reservations.get(participant) == session_id:
                del self._participant_reservations[participant]
        locks = self._conversation_locks.pop(session_id, {})
        self._finished_conversation_ids.add(session_id)
        for participant, (attention, availability, goal, checkpoint) in locks.items():
            agent = self.agents[participant]
            if agent.state.conversation_id == session_id:
                event_id = self._new_event_id()
                agent.state.conversation_id = None
                agent.state.attention_target = attention
                agent.state.set_availability(availability)
                agent.state.current_goal = goal
                if not agent.planner.resume(checkpoint, self.world, self.elapsed_minutes):
                    agent.planner.invalidate("conversation_plan_invalidated")
                    self._emit(
                        "plan_resume_invalidated",
                        participant,
                        {"session_id": session_id, "reason": "invalid_target"},
                    )
                if session.status == ConversationStatus.COMPLETED:
                    agent.state.record_success()
                else:
                    agent.state.record_failure(reason)
                self._remember(
                    agent,
                    "conversation_terminal",
                    {
                        "event_id": event_id,
                        "session_id": session_id,
                        "status": session.status.value,
                        "reason": reason,
                        "station_id": session.station_id,
                        "lease_id": session.lease_id,
                        "physical_receipt_ids": list(session.physical_receipt_ids),
                        "cleanup_evidence_ids": list(session.cleanup_evidence_ids),
                        "compatibility_mode": session.compatibility_mode,
                        "production_evidence": session.production_evidence,
                    },
                )
                self._emit(
                    "conversation_terminal",
                    participant,
                    {
                        "session_id": session_id,
                        "status": session.status.value,
                        "reason": reason,
                        "station_id": session.station_id,
                        "lease_id": session.lease_id,
                        "physical_receipt_ids": list(session.physical_receipt_ids),
                        "cleanup_evidence_ids": list(session.cleanup_evidence_ids),
                        "compatibility_mode": session.compatibility_mode,
                        "production_evidence": session.production_evidence,
                    },
                    event_id=event_id,
                    correlation_id=session.parent_event_id or session_id,
                )
        self._conversation_receipts[session_id] = self._conversation_receipt(
            session,
            True,
            session.status.value,
            session.failure_reason,
        )

    @staticmethod
    def _record_conversation_driver_evidence(
        session: ConversationSession, result: object
    ) -> None:
        for receipt_id in getattr(result, "receipt_ids", ()):
            if receipt_id not in session.physical_receipt_ids:
                session.physical_receipt_ids.append(receipt_id)
        cleanup_evidence_id = getattr(result, "cleanup_evidence_id", None)
        if (
            cleanup_evidence_id is not None
            and cleanup_evidence_id not in session.cleanup_evidence_ids
        ):
            session.cleanup_evidence_ids.append(cleanup_evidence_id)

    def _conversation_receipt(
        self,
        session: ConversationSession,
        accepted: bool,
        status: str,
        error: str | None = None,
    ) -> ConversationReceipt:
        return ConversationReceipt(
            session.session_id,
            accepted,
            status,
            error,
            session.parent_event_id,
            session.station_id,
            session.lease_id,
            tuple(session.physical_receipt_ids),
            session.cleanup_evidence_ids[-1] if session.cleanup_evidence_ids else None,
            session.compatibility_mode,
            session.production_evidence,
        )

    def _new_event_id(self) -> str:
        self._event_sequence += 1
        return f"evt_{self.seed}_{self._event_sequence:08d}"

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

    def _emit(
        self,
        event: str,
        agent_id: str,
        details: dict[str, Any],
        *,
        event_id: str | None = None,
        correlation_id: str | None = None,
        causation_id: str | None = None,
    ) -> RuntimeEvent:
        resolved_event_id = event_id or self._new_event_id()
        runtime_event = RuntimeEvent(
            self.elapsed_minutes,
            event,
            agent_id,
            details,
            resolved_event_id,
            correlation_id,
            causation_id,
        )
        self.events.append(runtime_event)
        self._pending_events.append(runtime_event)
        return runtime_event
