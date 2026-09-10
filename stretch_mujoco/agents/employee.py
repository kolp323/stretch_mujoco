"""Employee agent composition and deterministic local planner."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from stretch_mujoco.semantics import ObjectType, RelationType, SemanticWorld

from .actions import ActionCommand, ActionExecution, ActionType
from .models import (
    AgentMemory,
    AgentPerception,
    EmployeeNeeds,
    EmployeeProfile,
    EmployeeSchedule,
    EmployeeState,
    ScheduleItem,
)
from .utility import UtilityGoal, UtilityScore, UtilitySystem

if TYPE_CHECKING:
    from stretch_mujoco.npc.schema import NpcDefinition


@dataclass(frozen=True)
class BehaviorPlan:
    goal: UtilityGoal
    score: float
    variant: str
    actions: tuple[ActionCommand, ...]


@dataclass(frozen=True)
class PlanCheckpoint:
    plan: BehaviorPlan | None
    remaining_actions: tuple[ActionCommand, ...]
    goal: str


@dataclass
class EmployeePlanner:
    utility: UtilitySystem = field(default_factory=UtilitySystem)
    action_queue: list[ActionCommand] = field(default_factory=list)
    recent_goals: list[str] = field(default_factory=list)
    decision_count: int = 0
    current_plan: BehaviorPlan | None = None
    last_scores: tuple[UtilityScore, ...] = ()

    def choose_plan(
        self,
        agent: "EmployeeAgent",
        world: SemanticWorld,
        minute_of_day: float,
        day: int,
        seed: int,
    ) -> BehaviorPlan:
        schedule_item = agent.schedule.active_item(agent.agent_id, minute_of_day, day, seed)
        agent.state.schedule_item = "" if schedule_item is None else schedule_item.item_id
        self.last_scores = self.utility.evaluate(agent, world, schedule_item)
        choice = self.utility.choose(
            self.last_scores,
            agent_id=agent.agent_id,
            day=day,
            decision_index=self.decision_count,
            seed=seed,
        )
        plan = self._build_plan(agent, world, schedule_item, choice, seed, day)
        self.decision_count += 1
        self.current_plan = plan
        self.action_queue = list(plan.actions)
        self.recent_goals.append(plan.goal.value)
        self.recent_goals = self.recent_goals[-8:]
        agent.state.current_goal = plan.goal.value
        return plan

    def pop_action(self) -> ActionCommand | None:
        return self.action_queue.pop(0) if self.action_queue else None

    def clear_plan(self) -> None:
        self.action_queue.clear()
        self.current_plan = None

    def suspend(self, plan_id: str = "") -> PlanCheckpoint:
        del plan_id  # IDs are carried by the owning conversation session.
        return PlanCheckpoint(
            self.current_plan,
            tuple(self.action_queue),
            "" if self.current_plan is None else self.current_plan.goal.value,
        )

    def resume(self, checkpoint: PlanCheckpoint, world: SemanticWorld, now: float) -> bool:
        del now
        # Revalidate targets that are semantic objects; commands with an agent
        # target remain valid only when the caller's session lock has released.
        if any(
            command.target is not None
            and command.target not in world.objects
            and command.action
            not in {ActionType.TALK, ActionType.GESTURE_POINT, ActionType.GESTURE_WAVE}
            for command in checkpoint.remaining_actions
        ):
            return False
        self.current_plan = checkpoint.plan
        self.action_queue = list(checkpoint.remaining_actions)
        return True

    def invalidate(self, reason: str = "invalidated") -> None:
        del reason
        self.clear_plan()

    def _build_plan(
        self,
        agent: "EmployeeAgent",
        world: SemanticWorld,
        schedule_item: ScheduleItem | None,
        choice: UtilityScore,
        seed: int,
        day: int,
    ) -> BehaviorPlan:
        goal = choice.goal
        if goal == UtilityGoal.WORK:
            location = (
                schedule_item.location
                if schedule_item and schedule_item.activity == "work"
                else agent.profile.preferences.get("workstation", "workstation_right")
            )
            actions = EmployeePlanner._move_if_needed(agent, location)
            computers = [
                computer.object_id
                for computer in world.objects_of_type(ObjectType.COMPUTER)
                if world.find_relations(
                    subject=computer.object_id,
                    relation=RelationType.ON,
                    object_id=location,
                )
            ]
            use_computer = computers and (seed + day + self.decision_count) % 2
            if use_computer:
                actions.append(ActionCommand(agent.agent_id, ActionType.USE_COMPUTER, computers[0]))
                variant = "computer_work"
            else:
                actions.append(ActionCommand(agent.agent_id, ActionType.WORK, location))
                variant = "desk_work"
        elif goal in {UtilityGoal.EAT, UtilityGoal.DRINK}:
            object_id = agent.profile.preferences.get(
                "snack" if goal == UtilityGoal.EAT else "drink",
                "bread_snack" if goal == UtilityGoal.EAT else "soda_can",
            )
            actions = EmployeePlanner._consumption_plan(agent, world, object_id, goal)
            variant = "local_consume" if len(actions) > 1 else "robot_delivery"
        elif goal == UtilityGoal.REST:
            chair = agent.profile.preferences.get("chair", "chair_right")
            actions = EmployeePlanner._move_if_needed(agent, chair)
            actions.extend(
                (
                    ActionCommand(agent.agent_id, ActionType.SIT, chair),
                    ActionCommand(agent.agent_id, ActionType.REST, chair),
                )
            )
            variant = "chair_break"
        elif goal == UtilityGoal.MEETING:
            meeting_table = schedule_item.location if schedule_item else "meeting_table"
            actions, variant = EmployeePlanner._meeting_plan(
                agent, world, meeting_table, seed + day + self.decision_count
            )
        elif goal == UtilityGoal.REQUEST_ROBOT:
            object_id = EmployeePlanner._request_object(agent, schedule_item)
            actions = [
                EmployeePlanner._delivery_request(agent.agent_id, object_id, agent.state.location)
            ]
            variant = "stay_at_workstation"
        else:
            actions = [ActionCommand(agent.agent_id, ActionType.IDLE)]
            variant = "wait"
        return BehaviorPlan(goal, choice.score, variant, tuple(actions))

    @staticmethod
    def _move_if_needed(agent: "EmployeeAgent", location: str) -> list[ActionCommand]:
        if agent.state.location == location:
            return []
        return [ActionCommand(agent.agent_id, ActionType.MOVE_TO, location)]

    @staticmethod
    def _consumption_plan(
        agent: "EmployeeAgent",
        world: SemanticWorld,
        object_id: str,
        goal: UtilityGoal,
    ) -> list[ActionCommand]:
        consume_action = ActionType.EAT if goal == UtilityGoal.EAT else ActionType.DRINK
        if agent.state.held_object == object_id:
            return [ActionCommand(agent.agent_id, consume_action, object_id)]
        location = world.location_of(object_id)
        if location is not None and location.object_id == agent.state.location:
            return [
                ActionCommand(agent.agent_id, ActionType.PICK_UP, object_id),
                ActionCommand(agent.agent_id, consume_action, object_id),
            ]
        return [EmployeePlanner._delivery_request(agent.agent_id, object_id, agent.state.location)]

    @staticmethod
    def _meeting_plan(
        agent: "EmployeeAgent",
        world: SemanticWorld,
        meeting_table: str,
        variation_key: int,
    ) -> tuple[list[ActionCommand], str]:
        document = "document_report"
        location = world.location_of(document)
        if variation_key % 2 == 0 and location is not None:
            actions = EmployeePlanner._move_if_needed(agent, location.object_id)
            if location.object_type == ObjectType.STORAGE_CABINET:
                actions.append(
                    ActionCommand(agent.agent_id, ActionType.OPEN_CABINET, location.object_id)
                )
            actions.append(ActionCommand(agent.agent_id, ActionType.PICK_UP, document))
            if location.object_id != meeting_table:
                actions.append(ActionCommand(agent.agent_id, ActionType.MOVE_TO, meeting_table))
            actions.append(ActionCommand(agent.agent_id, ActionType.ATTEND_MEETING, meeting_table))
            actions.append(ActionCommand(agent.agent_id, ActionType.PUT_DOWN, meeting_table))
            return actions, "self_fetch_document"
        return (
            [
                EmployeePlanner._delivery_request(agent.agent_id, document, meeting_table),
                *EmployeePlanner._move_if_needed(agent, meeting_table),
                ActionCommand(agent.agent_id, ActionType.ATTEND_MEETING, meeting_table),
            ],
            "robot_fetch_document",
        )

    @staticmethod
    def _request_object(agent: "EmployeeAgent", schedule_item: ScheduleItem | None) -> str:
        if schedule_item and schedule_item.activity == "meeting":
            return "document_report"
        if agent.needs.thirst >= agent.needs.hunger:
            return agent.profile.preferences.get("drink", "soda_can")
        return agent.profile.preferences.get("snack", "bread_snack")

    @staticmethod
    def _delivery_request(agent_id: str, object_id: str, destination: str) -> ActionCommand:
        return ActionCommand(
            agent_id=agent_id,
            action=ActionType.REQUEST_ROBOT,
            target="stretch_3",
            parameters={
                "task": "deliver",
                "object": object_id,
                "destination": destination,
            },
        )


@dataclass
class EmployeeAgent:
    agent_id: str
    profile: EmployeeProfile
    needs: EmployeeNeeds
    schedule: EmployeeSchedule
    state: EmployeeState
    memory: AgentMemory = field(default_factory=AgentMemory)
    planner: EmployeePlanner = field(default_factory=EmployeePlanner)
    perception: AgentPerception = field(default_factory=AgentPerception)
    executor: ActionExecution = field(default_factory=ActionExecution)

    @classmethod
    def from_definition(cls, definition: "NpcDefinition") -> "EmployeeAgent":
        """Build the behavior projection of an already validated NPC definition."""
        state = EmployeeState(location=definition.spawn.location)
        state.sync_needs(definition.needs)
        return cls(
            agent_id=definition.npc_id,
            profile=definition.profile,
            needs=definition.needs,
            schedule=definition.schedule,
            state=state,
        )

    @classmethod
    def from_dict(cls, agent_id: str, payload: dict[str, Any]) -> "EmployeeAgent":
        """Load the legacy schema-v1 employee shape."""
        profile_data = payload["profile"]
        needs_data = payload.get("needs", {})
        needs = EmployeeNeeds(
            hunger=float(needs_data.get("hunger", 0.0)),
            thirst=float(needs_data.get("thirst", 0.0)),
            fatigue=float(needs_data.get("fatigue", 0.0)),
        )
        state = EmployeeState(location=payload["initial_location"])
        state.sync_needs(needs)
        return cls(
            agent_id=agent_id,
            profile=EmployeeProfile(
                role=profile_data["role"],
                department=profile_data["department"],
                personality=dict(profile_data.get("personality", {})),
                preferences=dict(profile_data.get("preferences", {})),
            ),
            needs=needs,
            schedule=EmployeeSchedule(
                tuple(ScheduleItem.from_dict(item) for item in payload.get("schedule", []))
            ),
            state=state,
        )

    def validate_identity(self, world: SemanticWorld) -> None:
        semantic_object = world.object(self.agent_id)
        if semantic_object.object_type != ObjectType.EMPLOYEE:
            raise ValueError(f"Agent '{self.agent_id}' is not bound to an Employee object")
