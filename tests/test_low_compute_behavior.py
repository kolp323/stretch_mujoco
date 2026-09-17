from pathlib import Path

import pytest

from stretch_mujoco.agents import (
    ActionCommand,
    ActionType,
    DailyOfficeEventGenerator,
    EventDrivenLLMGateway,
    LLMTrigger,
    OfficeAgentRuntime,
    UtilityGoal,
    UtilityScore,
)
from stretch_mujoco.semantics import SemanticWorld


MODELS_PATH = Path(__file__).resolve().parents[1] / "stretch_mujoco" / "models"


def load_runtime(*, auto_plan: bool = False) -> OfficeAgentRuntime:
    world = SemanticWorld.from_json(MODELS_PATH / "office_semantics.json")
    return OfficeAgentRuntime.from_json(
        world,
        MODELS_PATH / "office_agents.json",
        auto_plan=auto_plan,
    )


def test_runtime_uses_separate_state_machine_and_needs_rates() -> None:
    runtime = load_runtime()
    agent = runtime.agents["employee_01"]
    initial_hunger = agent.needs.hunger
    runtime.submit_action(ActionCommand("employee_01", ActionType.IDLE))

    runtime.tick(0.24)
    assert agent.executor.remaining_minutes == 1.0
    assert agent.needs.hunger == initial_hunger

    runtime.tick(0.01)
    assert agent.executor.remaining_minutes == 0.75
    runtime.tick(1.75)
    assert agent.needs.hunger > initial_hunger


def test_utility_scores_include_repetition_penalty() -> None:
    runtime = load_runtime()
    agent = runtime.agents["employee_01"]
    schedule_item = agent.schedule.active_item(
        agent.agent_id, runtime.minute_of_day, runtime.day, runtime.seed
    )

    initial = {
        score.goal: score
        for score in agent.planner.utility.evaluate(agent, runtime.world, schedule_item)
    }
    agent.planner.recent_goals.extend([UtilityGoal.WORK.value] * 2)
    repeated = {
        score.goal: score
        for score in agent.planner.utility.evaluate(agent, runtime.world, schedule_item)
    }

    assert initial[UtilityGoal.WORK].score - repeated[UtilityGoal.WORK].score == pytest.approx(0.16)


def test_consumed_preference_does_not_create_robot_request_utility() -> None:
    runtime = load_runtime()
    agent = runtime.agents["employee_01"]
    runtime.world.object("soda_can").attributes["consumed"] = True
    runtime.world.object("bread_snack").attributes["consumed"] = True
    agent.needs.hunger = 1.0
    agent.needs.thirst = 1.0

    scores = {
        score.goal: score for score in agent.planner.utility.evaluate(agent, runtime.world, None)
    }

    assert scores[UtilityGoal.REQUEST_ROBOT].score == 0.0


def test_zero_score_goals_are_never_randomly_selected() -> None:
    runtime = load_runtime()
    utility = runtime.agents["employee_01"].planner.utility
    scores = (
        UtilityScore(UtilityGoal.WAIT, 0.08, {}),
        UtilityScore(UtilityGoal.DRINK, 0.0, {}),
        UtilityScore(UtilityGoal.MEETING, 0.0, {}),
        UtilityScore(UtilityGoal.REQUEST_ROBOT, 0.0, {}),
    )

    choices = {
        utility.choose(
            scores,
            agent_id="employee_01",
            day=0,
            decision_index=index,
            seed=23,
        ).goal
        for index in range(100)
    }

    assert choices == {UtilityGoal.WAIT}


def test_seeded_meeting_plans_use_multiple_constrained_paths() -> None:
    variants = set()
    for day in (0, 1):
        runtime = load_runtime()
        runtime.day = day
        runtime.minute_of_day = 11 * 60 + 10
        agent = runtime.agents["employee_01"]
        plan = agent.planner.choose_plan(
            agent,
            runtime.world,
            runtime.minute_of_day,
            runtime.day,
            runtime.seed,
        )
        variants.add(plan.variant)
        assert plan.goal == UtilityGoal.MEETING
        assert any(action.action == ActionType.ATTEND_MEETING for action in plan.actions)
        if plan.variant == "self_fetch_document":
            assert plan.actions[-1].action == ActionType.PUT_DOWN

    assert variants == {"robot_fetch_document", "self_fetch_document"}


def test_utility_decision_interval_stays_in_configured_range() -> None:
    runtime = load_runtime(auto_plan=True)

    runtime.tick(0.25)

    assert 5.0 <= runtime._utility_seconds_remaining <= 15.0
    assert runtime.agents["employee_01"].planner.current_plan is not None


def test_daily_office_events_are_seeded_and_semantically_constrained() -> None:
    runtime = load_runtime()
    generator = DailyOfficeEventGenerator(runtime.seed)

    first = generator.generate(3, runtime.world, tuple(runtime.agents))
    second = generator.generate(3, runtime.world, tuple(runtime.agents))

    assert first == second
    assert first
    assert all(event.target in runtime.world.objects for event in first)


def test_llm_gateway_is_event_only_and_budgeted() -> None:
    gateway = EventDrivenLLMGateway(daily_budget=2)
    calls = []
    assert gateway.queue(LLMTrigger.DAY_START, "employee_01", 0, 540.0)
    assert gateway.queue(LLMTrigger.NEW_TASK, "employee_01", 0, 550.0)
    assert not gateway.queue(LLMTrigger.DIALOGUE, "employee_01", 0, 560.0)
    assert calls == []

    requests = gateway.drain_requests()
    result = gateway.process(requests[0], lambda request: calls.append(request) or {"ok": True})

    assert result == {"ok": True}
    assert len(calls) == 1
    assert gateway.calls_for(0, "employee_01") == 1


def test_simulation_tick_never_processes_llm_requests() -> None:
    runtime = load_runtime(auto_plan=True)

    runtime.tick(30.0)

    assert runtime.llm.calls_for(0, "employee_01") == 0
    assert runtime.drain_llm_requests()


def test_day_start_llm_request_has_constrained_office_context() -> None:
    runtime = load_runtime()

    request = runtime.drain_llm_requests()[0]

    assert request.trigger == LLMTrigger.DAY_START
    assert request.context["workday"] == {"start": "09:00", "end": "18:00"}
    assert request.context["profile"]["role"] == "Operations Specialist"
    assert "workstation_right" in request.context["valid_objects"]["Workstation"]
    assert request.context["existing_schedule"]


@pytest.mark.parametrize(
    "trigger",
    (
        LLMTrigger.NEW_TASK,
        LLMTrigger.DIALOGUE,
        LLMTrigger.REPEATED_FAILURE,
        LLMTrigger.UNEXPECTED_CHANGE,
        LLMTrigger.REINTERPRET_PLAN,
    ),
)
def test_local_llm_requests_include_authoritative_personality_and_preferences(
    trigger: LLMTrigger,
) -> None:
    runtime = load_runtime()
    runtime.drain_llm_requests()
    agent = runtime.agents["employee_01"]

    assert runtime.queue_llm_event(
        trigger,
        agent.agent_id,
        {
            "event_detail": "local",
            "profile": {
                "personality": {"impersonated": 1.0},
                "preferences": {"workstation": "imaginary_workstation"},
                "event_specific": "preserved",
            },
        },
    )
    request = runtime.drain_llm_requests()[0]

    assert request.context["event_detail"] == "local"
    assert request.context["profile"]["personality"] == agent.profile.personality
    assert request.context["profile"]["preferences"] == agent.profile.preferences
    assert request.context["profile"]["event_specific"] == "preserved"


def test_llm_schedule_response_is_validated_before_application() -> None:
    runtime = load_runtime()
    request = runtime.drain_llm_requests()[0]
    original_schedule = runtime.agents["employee_01"].schedule

    invalid = runtime.apply_llm_response(
        request,
        {
            "schedule": [
                {
                    "id": "bad_location",
                    "start": "09:00",
                    "end": "10:00",
                    "activity": "work",
                    "location": "imaginary_office",
                }
            ]
        },
    )
    valid = runtime.apply_llm_response(
        request,
        {
            "schedule": [
                {
                    "id": "draft_work",
                    "start": "09:10",
                    "end": "10:00",
                    "activity": "work",
                    "location": "workstation_right",
                    "variation_minutes": 5,
                }
            ]
        },
    )

    assert not invalid.valid
    assert runtime.agents["employee_01"].schedule is not original_schedule
    assert valid.valid
    assert runtime.agents["employee_01"].schedule.items[0].item_id == "draft_work"


def test_new_task_llm_action_must_address_event_target() -> None:
    runtime = load_runtime()
    requests = runtime.drain_llm_requests()
    request = next(item for item in requests if item.trigger == LLMTrigger.NEW_TASK)

    result = runtime.apply_llm_response(
        request,
        {
            "action": {
                "action": "request_robot",
                "target": "stretch_3",
                "parameters": {
                    "task": "deliver",
                    "object": "bread_snack",
                    "destination": "workstation_right",
                },
            }
        },
    )

    assert not result.valid
    assert "does not address" in result.errors[0]
    assert not runtime.pending_robot_tasks()


def test_three_consecutive_rejections_queue_llm_failure_event() -> None:
    runtime = load_runtime()
    runtime.drain_llm_requests()
    invalid = ActionCommand("employee_01", ActionType.MOVE_TO, "missing_place")

    for _ in range(3):
        runtime.submit_action(invalid)

    requests = runtime.drain_llm_requests()
    assert requests[-1].trigger == LLMTrigger.REPEATED_FAILURE
    assert requests[-1].context["failures"] == 3
