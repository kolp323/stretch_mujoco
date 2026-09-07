from dataclasses import dataclass
from math import nan, pi
from pathlib import Path

import pytest

from stretch_mujoco.agents import (
    AgentAvailability,
    AgentMemory,
    ConversationCoordinator,
    ConversationErrorCode,
    ConversationIntent,
    ConversationObservationSource,
    ConversationParticipantKind,
    ConversationPerception,
    ConversationPerceptionError,
    ConversationStatus,
    EmployeeAgent,
    EmployeeState,
    InterruptPolicy,
    MockRobotExecutor,
    MemoryEntry,
    OfficeAgentRuntime,
    SpatialPose,
)
from stretch_mujoco.semantics import ObjectType, SemanticObject, SemanticWorld


MODELS_PATH = Path(__file__).resolve().parents[1] / "stretch_mujoco" / "models"


@dataclass
class DeterministicConversationFixture:
    """A no-asset conversation fixture for P0/P3 lifecycle tests."""

    runtime: OfficeAgentRuntime
    semantic_snapshot: dict[str, object]
    robot: MockRobotExecutor


@pytest.fixture
def deterministic_conversation_fixture() -> DeterministicConversationFixture:
    world = SemanticWorld.from_json(MODELS_PATH / "office_semantics.json")
    employee_binding = world.object("employee_01").binding
    for employee_id in ("employee_02", "employee_03"):
        world.objects[employee_id] = SemanticObject(
            employee_id,
            ObjectType.EMPLOYEE,
            employee_binding,
            {"display_name": employee_id},
        )

    def employee(employee_id: str) -> EmployeeAgent:
        return EmployeeAgent.from_dict(
            employee_id,
            {
                "profile": {"role": "Tester", "department": "QA"},
                "initial_location": "workstation_right",
            },
        )

    runtime = OfficeAgentRuntime(
        world,
        {
            employee_id: employee(employee_id)
            for employee_id in ("employee_01", "employee_02", "employee_03")
        },
        auto_plan=False,
        daily_events=False,
    )
    return DeterministicConversationFixture(
        runtime=runtime,
        semantic_snapshot={
            "time": 0.0,
            "objects": {
                "employee_01": {"position": [0.0, 0.0, 0.0], "yaw": 0.0},
                "employee_02": {"position": [1.0, 0.0, 0.0], "yaw": pi},
                "employee_03": {"position": [0.0, 2.0, 0.0], "yaw": -pi / 2},
                "stretch_3": {"position": [0.0, -1.0, 0.0], "yaw": pi / 2},
            },
        },
        robot=MockRobotExecutor(completion_delay_minutes=1.0),
    )


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


def test_conversation_contract_uses_deterministic_ids_and_rejects_repeated_participants() -> None:
    coordinator = ConversationCoordinator()
    poses = {
        "employee_01": SpatialPose((0.0, 0.0, 0.0), 0.0),
        "stretch_3": SpatialPose((1.0, 0.0, 0.0), pi),
    }
    session = coordinator.start(
        ("employee_01", "stretch_3"),
        "delivery",
        0.0,
        poses,
        participant_kinds={
            "employee_01": ConversationParticipantKind.NPC,
            "stretch_3": ConversationParticipantKind.STRETCH_ROBOT,
        },
    )

    assert session.session_id == "conversation_000001"
    assert session.participant_kinds == (
        ConversationParticipantKind.NPC,
        ConversationParticipantKind.STRETCH_ROBOT,
    )
    turn = coordinator.record_candidate(
        session.session_id,
        "employee_01",
        ConversationIntent.REQUEST,
        "Please bring the document.",
        0.0,
        includes_robot=True,
    )
    assert turn.turn_id == "conversation_000001:turn:1"

    with pytest.raises(ValueError, match="already has an active session"):
        coordinator.start(("employee_01", "stretch_3"), "delivery", 0.1, poses)


def test_conversation_perception_adapts_semantic_and_offline_snapshots_read_only() -> None:
    semantic = ConversationPerception.from_snapshot(
        {
            "time": 3.0,
            "objects": {
                "employee_01": {
                    "position": [0.0, 0.0, 0.0],
                    "quaternion": [1.0, 0.0, 0.0, 0.0],
                },
                "stretch_3": {
                    "position": [1.0, 0.0, 0.0],
                    "quaternion": [0.0, 0.0, 0.0, 1.0],
                },
            },
        },
        ("employee_01", "stretch_3"),
        now=3.25,
    )
    offline = ConversationPerception.from_snapshot(
        {
            "sim_time": 4.0,
            "agents": {
                "employee_01": {"position": [0.0, 0.0, 0.0], "yaw": 0.0},
                "stretch_3": {"position": [1.0, 0.0, 0.0], "yaw": pi},
            },
        },
        ("employee_01", "stretch_3"),
        now=4.25,
    )

    assert semantic.source == ConversationObservationSource.SEMANTIC_WORLD
    assert semantic.poses["stretch_3"].yaw == pytest.approx(pi)
    assert offline.source == ConversationObservationSource.OFFLINE_RECORDING
    with pytest.raises(TypeError):
        semantic.poses["other"] = SpatialPose((0.0, 0.0, 0.0), 0.0)  # type: ignore[index]


@pytest.mark.parametrize(
    ("snapshot", "code"),
    [
        (
            {
                "time": 0.0,
                "objects": {
                    "employee_01": {"position": [0.0, 0.0, 0.0], "yaw": 0.0},
                    "stretch_3": {"position": [1.0, 0.0, 0.0], "yaw": pi},
                },
            },
            ConversationErrorCode.STALE_OBSERVATION,
        ),
        (
            {
                "time": 1.0,
                "objects": {
                    "employee_01": {"position": [nan, 0.0, 0.0], "yaw": 0.0},
                    "stretch_3": {"position": [1.0, 0.0, 0.0], "yaw": pi},
                },
            },
            ConversationErrorCode.INVALID_POSE,
        ),
        (
            {
                "time": 1.0,
                "objects": {
                    "employee_01": {"position": [0.0, 0.0, 0.0], "yaw": 0.0},
                    "stretch_3": {"position": [1.0, 0.0, 0.0]},
                },
            },
            ConversationErrorCode.INVALID_POSE,
        ),
        (
            {
                "time": 1.0,
                "objects": {
                    "employee_01": {"position": [0.0, 0.0, 0.0], "yaw": 0.0},
                    "stretch_3": {
                        "position": [1.0, 0.0, 0.0],
                        "yaw": pi,
                        "visible": False,
                    },
                },
            },
            ConversationErrorCode.PARTICIPANT_NOT_VISIBLE,
        ),
    ],
)
def test_conversation_perception_rejects_unusable_observations(
    snapshot: dict[str, object], code: ConversationErrorCode
) -> None:
    with pytest.raises(ConversationPerceptionError) as error:
        ConversationPerception.from_snapshot(
            snapshot,
            ("employee_01", "stretch_3"),
            now=1.0,
            max_age=0.25,
        )

    assert error.value.code == code


def test_runtime_rejects_stale_observation_and_unknown_participant(
    deterministic_conversation_fixture: DeterministicConversationFixture,
) -> None:
    runtime = deterministic_conversation_fixture.runtime
    runtime.tick(1.0)

    with pytest.raises(ConversationPerceptionError, match="stale_observation"):
        runtime.start_conversation(
            "employee_01",
            "employee_02",
            "status",
            deterministic_conversation_fixture.semantic_snapshot,
        )
    with pytest.raises(KeyError, match="Unknown conversation participant"):
        runtime.start_conversation(
            "employee_01",
            "employee_missing",
            "status",
            deterministic_conversation_fixture.semantic_snapshot,
        )


def test_employee_state_uses_bounded_availability_and_legacy_busy_value() -> None:
    assert EmployeeState("desk", availability="busy").availability == AgentAvailability.EXECUTING
    with pytest.raises(ValueError, match="Unknown agent availability"):
        EmployeeState("desk", availability="unbounded")


def test_agent_memory_capacity_discards_oldest_entries() -> None:
    memory = AgentMemory(capacity=2)
    for timestamp in range(3):
        memory.remember(MemoryEntry(float(timestamp), "conversation_turn", {}))

    assert [entry.timestamp for entry in memory.entries] == [1, 2]


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
