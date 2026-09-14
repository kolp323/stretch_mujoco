from dataclasses import dataclass
from math import nan, pi
from pathlib import Path

import pytest

from stretch_mujoco.agents import (
    AgentAvailability,
    AgentMemory,
    ActionCommand,
    ActionType,
    ConversationCoordinator,
    ConversationRequest,
    DialogueAct,
    DialogueCandidate,
    DialoguePolicy,
    DialoguePolicyConfig,
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
    SocialConversationProposal,
    SocialState,
    TurnPolicy,
)
from stretch_mujoco.agents.drivers import DriverResult
from stretch_mujoco.agents.actions import ExecutionStatus
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


class _ReadyConversationDriver:
    def prepare_conversation(self, _session: object) -> DriverResult:
        return DriverResult(ExecutionStatus.SUCCEEDED, "ready", "prepare-1")

    def play_turn(self, _session: object, _turn: object) -> DriverResult:
        return DriverResult(ExecutionStatus.SUCCEEDED, "played", "turn-1")

    def cancel_conversation(self, _session_id: str, _reason: str) -> DriverResult:
        return DriverResult(ExecutionStatus.SUCCEEDED, "cancelled")


class _AsyncTurnConversationDriver(_ReadyConversationDriver):
    """A receipt-producing driver: talk remains pending until runtime ticks it."""

    def play_turn(self, _session: object, _turn: object) -> DriverResult:
        return DriverResult(ExecutionStatus.RUNNING, "talk", "turn-pending")

    def poll_conversation(self, _session_id: str) -> DriverResult:
        return DriverResult(ExecutionStatus.SUCCEEDED, "terminal", "turn-receipt")


def test_strict_conversation_commits_only_after_driver_receipt(
    deterministic_conversation_fixture: DeterministicConversationFixture,
) -> None:
    runtime = deterministic_conversation_fixture.runtime
    runtime.interaction_driver = _ReadyConversationDriver()
    receipt = runtime.begin_conversation(
        ConversationRequest(
            "session-receipt",
            ("employee_01", "employee_02"),
            "status",
            semantic_snapshot=deterministic_conversation_fixture.semantic_snapshot,
        )
    )
    assert receipt.accepted and receipt.status == "ready"
    assert runtime.agents["employee_01"].state.conversation_id == "session-receipt"
    candidate = DialogueCandidate(
        "request-1",
        "session-receipt",
        "turn-1",
        "employee_01",
        "employee_02",
        DialogueAct.GREETING,
        "Hello.",
        0.0,
    )
    assert runtime.submit_dialogue_candidate(candidate).valid
    session = runtime.conversation("session-receipt")
    assert session.dialogue_turns["turn-1"].status.value == "committed"
    assert any(event.event == "dialogue_turn_committed" for event in runtime.events)


def test_embodied_conversation_defers_spatial_gate_until_driver_approach(
    deterministic_conversation_fixture: DeterministicConversationFixture,
) -> None:
    """A real driver must be allowed to bring separated people together."""
    runtime = deterministic_conversation_fixture.runtime
    runtime.interaction_driver = _ReadyConversationDriver()
    snapshot = dict(deterministic_conversation_fixture.semantic_snapshot)
    snapshot["objects"] = dict(snapshot["objects"])
    snapshot["objects"]["employee_02"] = {
        "position": [9.0, 0.0, 0.0],
        "yaw": pi,
    }

    receipt = runtime.begin_conversation(
        ConversationRequest(
            "session-approach-before-gate",
            ("employee_01", "employee_02"),
            "status",
            semantic_snapshot=snapshot,
        )
    )

    assert receipt.accepted and receipt.status == "ready"


def test_strict_conversation_waits_for_async_turn_receipt(
    deterministic_conversation_fixture: DeterministicConversationFixture,
) -> None:
    runtime = deterministic_conversation_fixture.runtime
    runtime.interaction_driver = _AsyncTurnConversationDriver()
    receipt = runtime.begin_conversation(
        ConversationRequest(
            "session-async-receipt",
            ("employee_01", "employee_02"),
            "status",
            semantic_snapshot=deterministic_conversation_fixture.semantic_snapshot,
        )
    )
    assert receipt.accepted and receipt.status == "ready"
    candidate = DialogueCandidate(
        "request-async",
        "session-async-receipt",
        "turn-async",
        "employee_01",
        "employee_02",
        DialogueAct.GREETING,
        "Hello.",
        0.0,
    )
    assert runtime.submit_dialogue_candidate(candidate).valid
    session = runtime.conversation("session-async-receipt")
    assert session.dialogue_turns["turn-async"].status.value == "playing"
    assert not any(event.event == "dialogue_turn_committed" for event in runtime.events)

    events = runtime.tick(0.1, deterministic_conversation_fixture.semantic_snapshot)
    assert session.dialogue_turns["turn-async"].status.value == "committed"
    assert any(event.event == "dialogue_turn_committed" for event in events)


def test_dialogue_policy_fallback_is_deterministic() -> None:
    policy = DialoguePolicy(DialoguePolicyConfig(max_chars=20), runtime_seed=7)
    candidate = DialogueCandidate("r", "s", "t", "a", "b", DialogueAct.GREETING, "", 0.0)
    session = ConversationCoordinator().start(
        ("a", "b"), "topic", 0.0, {"a": SpatialPose((0, 0, 0), 0), "b": SpatialPose((1, 0, 0), pi)}
    )
    session.session_id = "s"
    first = policy.validate_and_sanitize(candidate, session)
    second = policy.validate_and_sanitize(candidate, session)
    assert first.fallback_used and first.candidate.text == second.candidate.text


def test_dialogue_pair_cooldown_does_not_block_later_turns_in_one_session() -> None:
    policy = DialoguePolicy(DialoguePolicyConfig(pair_cooldown_seconds=30.0))
    session = ConversationCoordinator().start(
        ("a", "b"),
        "topic",
        0.0,
        {"a": SpatialPose((0, 0, 0), 0), "b": SpatialPose((1, 0, 0), pi)},
    )
    policy.note_committed(
        DialogueCandidate(
            "first", session.session_id, "first", "a", "b", DialogueAct.GREETING, "Hi.", 0.0
        ),
        0.0,
    )
    session.turn = 1
    session.expected_speaker = "b"
    candidate = DialogueCandidate(
        "second", session.session_id, "second", "b", "a", DialogueAct.ACKNOWLEDGE, "Hello.", 1.0
    )

    assert policy.validate_and_sanitize(candidate, session, now=1.0).candidate == candidate


def test_conversation_spatial_boundaries_accept_exact_limits_and_reject_overage() -> None:
    coordinator = ConversationCoordinator(max_distance=1.0, max_facing_degrees=60.0)
    boundary_poses = {
        "a": SpatialPose((0.0, 0.0, 0.0), pi / 3),
        "b": SpatialPose((1.0, 0.0, 0.0), 2 * pi / 3),
    }
    assert (
        coordinator.start(("a", "b"), "boundary", 0.0, boundary_poses).status
        == ConversationStatus.ACTIVE
    )
    with pytest.raises(ValueError, match="too far apart"):
        coordinator.start(
            ("c", "d"),
            "too far",
            1.0,
            {"c": boundary_poses["a"], "d": SpatialPose((1.001, 0.0, 0.0), pi)},
        )


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
    assert runtime.request_robot_task(
        session.session_id,
        "employee_01",
        task="deliver",
        object_id="document_report",
        destination="workstation_right",
    ).valid
    runtime.tick(1.0)
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

    assert not result.valid
    assert "requires release" in result.errors[0]
    assert not session.transcript


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


def test_conversation_terminal_cleanup_is_idempotent_and_preserves_planner_queue() -> None:
    runtime = load_runtime()
    runtime.drain_llm_requests()
    agent = runtime.agents["employee_01"]
    agent.planner.action_queue = []
    agent.state.current_goal = "work"
    session = runtime.start_conversation("employee_01", "stretch_3", "status", facing_snapshot())

    assert agent.state.conversation_id == session.session_id
    completed = runtime.complete_conversation(session.session_id)
    events_after_first = tuple(
        event for event in runtime.events if event.event == "conversation_completed"
    )
    assert runtime.complete_conversation(session.session_id) is completed
    assert runtime.cancel_conversation(session.session_id) is completed

    assert completed.status == ConversationStatus.COMPLETED
    assert agent.state.conversation_id is None
    assert agent.state.current_goal == "work"
    assert len(
        tuple(event for event in runtime.events if event.event == "conversation_completed")
    ) == len(events_after_first)
    assert not [event for event in runtime.events if event.event == "conversation_cancelled"]


def test_conversation_suspends_and_resumes_pending_plan() -> None:
    runtime = load_runtime()
    agent = runtime.agents["employee_01"]
    queued = ActionCommand("employee_01", ActionType.MOVE_TO, "meeting_table")
    agent.planner.action_queue = [queued]

    session = runtime.start_conversation("employee_01", "stretch_3", "status", facing_snapshot())

    assert agent.planner.action_queue == []
    runtime.complete_conversation(session.session_id)
    assert agent.planner.action_queue == [queued]


def test_round_robin_turns_and_turn_timeout_are_enforced() -> None:
    coordinator = ConversationCoordinator(cooldown_minutes=0.0)
    poses = {
        "a": SpatialPose((0.0, 0.0, 0.0), 0.0),
        "b": SpatialPose((1.0, 0.0, 0.0), pi),
    }
    session = coordinator.start(
        ("a", "b"), "status", 0.0, poses, turn_policy=TurnPolicy.ROUND_ROBIN, turn_timeout=0.5
    )
    with pytest.raises(ValueError, match="waiting for 'a'"):
        coordinator.record_candidate(
            session.session_id, "b", "greeting", "Hello.", 0.0, includes_robot=False
        )
    coordinator.record_candidate(
        session.session_id, "a", "greeting", "Hello.", 0.0, includes_robot=False
    )
    assert session.expected_speaker == "b"
    assert coordinator.expire(0.5) == (session,)
    assert session.status == ConversationStatus.TIMED_OUT
    assert session.failure_reason == "turn_timeout"


def test_busy_embodied_actions_reject_allow_interrupt_policy() -> None:
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
            interrupt_policy=InterruptPolicy.ALLOW,
        )


def test_robot_dialogue_requires_task_and_physical_handover_receipts() -> None:
    runtime = load_runtime()
    runtime.drain_llm_requests()
    session = runtime.start_conversation("employee_01", "stretch_3", "delivery", facing_snapshot())
    with pytest.raises(ValueError, match="validated RobotTask"):
        runtime.record_conversation_candidate(
            session.session_id, "employee_01", "request", "Deliver it."
        )

    assert runtime.request_robot_task(
        session.session_id,
        "employee_01",
        task="deliver",
        object_id="document_report",
        destination="workstation_right",
    ).valid
    runtime.tick(1.0)
    task_id = session.robot_task_id
    assert task_id is not None
    assert runtime.agents["employee_01"].state.conversation_id == session.session_id
    assert runtime.agents["employee_01"].state.availability == "in_conversation"
    runtime.record_conversation_candidate(
        session.session_id, "employee_01", "request", "Deliver it."
    )
    runtime.record_conversation_candidate(session.session_id, "stretch_3", "clarify", "Which desk?")
    runtime.tick(0.25)
    runtime.record_conversation_candidate(
        session.session_id, "employee_01", "acknowledge", "The right desk."
    )

    MockRobotExecutor(
        completion_delay_minutes=0.1,
        duplicate_receipt_task_ids={task_id},
        handover_ready_task_ids={task_id},
    ).tick(runtime, 0.1)
    assert runtime.robot_tasks[task_id].status.value == "succeeded"
    runtime.record_conversation_candidate(
        session.session_id, "stretch_3", "handover_confirm", "Handover complete."
    )
    assert len([event for event in runtime.events if event.event == "robot_task_completed"]) == 1


def test_mock_robot_dropped_receipt_leaves_task_running() -> None:
    runtime = load_runtime()
    runtime.submit_action(
        {
            "agent_id": "employee_01",
            "action": "request_robot",
            "target": "stretch_3",
            "parameters": {
                "task": "deliver",
                "object": "document_report",
                "destination": "workstation_right",
            },
        }
    )
    runtime.tick(1.0)
    task = runtime.pending_robot_tasks()[0]
    assert MockRobotExecutor(0.1, drop_receipt_task_ids={task.task_id}).tick(runtime, 0.1) == ()
    assert task.status.value == "running"


def test_three_npc_social_scheduler_has_deterministic_single_winner(
    deterministic_conversation_fixture: DeterministicConversationFixture,
) -> None:
    runtime = deterministic_conversation_fixture.runtime
    proposals = [
        SocialConversationProposal(
            "employee_02", "employee_03", ConversationIntent.GREETING, "hello", 0.0
        ),
        SocialConversationProposal(
            "employee_01", "employee_02", ConversationIntent.PROGRESS_INQUIRY, "status", 0.0
        ),
        SocialConversationProposal(
            "employee_03", "employee_01", ConversationIntent.CONFLICT_RESOLUTION, "resolve", 0.0
        ),
    ]
    decisions = runtime.schedule_npc_conversations(
        proposals, deterministic_conversation_fixture.semantic_snapshot
    )
    assert [decision.accepted for decision in decisions] == [True, False, False]
    assert decisions[0].proposal.initiator == "employee_01"
    assert decisions[1].reason == "participant_busy"
    assert decisions[2].reason == "participant_busy"
    assert (
        len(
            [
                session
                for session in runtime.conversations.sessions.values()
                if not session.status.terminal
            ]
        )
        == 1
    )


def test_social_scheduler_cooldown_and_meeting_responses_are_deterministic() -> None:
    from stretch_mujoco.agents import NpcConversationScheduler

    scheduler = NpcConversationScheduler(cooldown_minutes=2.0)
    states = {"a": SocialState(1.0, 0.0, True), "b": SocialState(1.0, 0.0, True)}
    proposal = SocialConversationProposal(
        "a", "b", ConversationIntent.MEETING_INVITATION, "planning", 0.0
    )
    assert scheduler.admit((proposal,), states, 0.0)[0].accepted
    assert scheduler.admit((proposal,), states, 1.0)[0].reason == "cooldown"
    invitation = scheduler.create_invitation(proposal, 1.5)
    assert scheduler.respond_invitation(invitation.invitation_id, True, 1.0).accepted is True
    expired = scheduler.create_invitation(proposal, 2.0)
    assert scheduler.expire_invitations(2.0) == (expired,)
    assert expired.accepted is False
