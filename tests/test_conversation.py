from math import pi
from pathlib import Path

import pytest

from stretch_mujoco.agents import (
    ConversationCoordinator,
    ConversationIntent,
    ConversationStatus,
    InterruptPolicy,
    OfficeAgentRuntime,
    SpatialPose,
)
from stretch_mujoco.semantics import SemanticWorld


MODELS_PATH = Path(__file__).resolve().parents[1] / "stretch_mujoco" / "models"


def load_runtime() -> OfficeAgentRuntime:
    world = SemanticWorld.from_json(MODELS_PATH / "office_semantics.json")
    return OfficeAgentRuntime.from_json(world, MODELS_PATH / "office_agents.json", auto_plan=False)


def facing_snapshot() -> dict[str, object]:
    return {
        "objects": {
            "employee_01": {"position": [0.0, 0.0, 0.0], "yaw": 0.0},
            "stretch_3": {"position": [1.0, 0.0, 0.0], "yaw": pi},
        }
    }


def test_conversation_requires_distance_and_mutual_facing() -> None:
    coordinator = ConversationCoordinator(max_distance=1.5)
    poses = {
        "a": SpatialPose((0.0, 0.0, 0.0), 0.0),
        "b": SpatialPose((2.0, 0.0, 0.0), pi),
    }

    with pytest.raises(ValueError, match="too far apart"):
        coordinator.start(("a", "b"), "status", 0.0, poses)


def test_robot_conversation_filters_candidate_and_restores_agent_plan() -> None:
    runtime = load_runtime()
    runtime.drain_llm_requests()
    agent = runtime.agents["employee_01"]
    agent.state.current_goal = "work"
    session = runtime.start_conversation(
        "employee_01",
        "stretch_3",
        "document handover",
        facing_snapshot(),
        timeout=3.0,
    )

    assert session.status == ConversationStatus.ACTIVE
    assert agent.state.availability == "in_conversation"
    assert agent.state.attention_target == "stretch_3"
    turn = runtime.record_conversation_candidate(
        session.session_id,
        "employee_01",
        ConversationIntent.REQUEST,
        "Bearer private-token",
    )

    assert turn.used_fallback
    assert turn.text == "Could you help with this request?"
    completed = runtime.complete_conversation(session.session_id)

    assert completed.status == ConversationStatus.COMPLETED
    assert agent.state.availability == "available"
    assert agent.state.current_goal == "work"
    assert agent.state.attention_target is None
    assert any(entry.event == "conversation_turn" for entry in agent.memory.entries)


def test_robot_conversation_times_out_and_records_runtime_event() -> None:
    runtime = load_runtime()
    runtime.drain_llm_requests()
    session = runtime.start_conversation(
        "employee_01", "stretch_3", "delivery", facing_snapshot(), timeout=0.5
    )

    events = runtime.tick(0.5)

    assert session.status == ConversationStatus.TIMED_OUT
    assert runtime.agents["employee_01"].state.availability == "available"
    assert any(event.event == "conversation_timed_out" for event in events)


def test_llm_dialogue_is_only_a_candidate_and_uses_allowed_intent() -> None:
    runtime = load_runtime()
    runtime.drain_llm_requests()
    session = runtime.start_conversation("employee_01", "stretch_3", "delivery", facing_snapshot())

    assert runtime.queue_conversation_candidate(session.session_id, "employee_01")
    request = runtime.drain_llm_requests()[0]
    result = runtime.apply_llm_response(
        request,
        {"intent": "handover_confirm", "text": "Item released at the handover point."},
    )

    assert result.valid
    assert session.transcript[0].intent == ConversationIntent.HANDOVER_CONFIRM
    assert session.transcript[0].text == "Item released at the handover point."


def test_npc_conversation_intents_and_cooldown_are_enforced() -> None:
    coordinator = ConversationCoordinator(cooldown_minutes=1.0)
    poses = {
        "a": SpatialPose((0.0, 0.0, 0.0), 0.0),
        "b": SpatialPose((1.0, 0.0, 0.0), pi),
    }
    session = coordinator.start(("a", "b"), "meeting", 0.0, poses)

    turn = coordinator.record_candidate(
        session.session_id,
        "a",
        "meeting_invitation",
        "Can we meet at 11?",
        0.0,
        includes_robot=False,
    )
    assert turn.intent == ConversationIntent.MEETING_INVITATION
    with pytest.raises(ValueError, match="cooldown"):
        coordinator.record_candidate(
            session.session_id,
            "a",
            "greeting",
            "Hello again.",
            0.5,
            includes_robot=False,
        )
    with pytest.raises(ValueError, match="not valid"):
        coordinator.record_candidate(
            session.session_id,
            "b",
            "request",
            "Please deliver this.",
            1.0,
            includes_robot=False,
        )


def test_busy_plan_is_not_interrupted_by_default() -> None:
    runtime = load_runtime()
    runtime.drain_llm_requests()
    runtime.submit_action(
        {"agent_id": "employee_01", "action": "work", "target": "workstation_right"}
    )

    with pytest.raises(ValueError, match="Cannot interrupt"):
        runtime.start_conversation(
            "employee_01",
            "stretch_3",
            "status",
            facing_snapshot(),
            interrupt_policy=InterruptPolicy.REJECT,
        )
