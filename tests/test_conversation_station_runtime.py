from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from stretch_mujoco.agents.actions import (
    ConversationRequest,
    ExecutionStatus,
)
from stretch_mujoco.agents.conversation import DialogueAct, DialogueCandidate
from stretch_mujoco.agents.drivers import MujocoNpcActionDriver
from stretch_mujoco.agents.employee import EmployeeAgent
from stretch_mujoco.agents.drivers import DriverResult
from stretch_mujoco.agents.interaction_stations import (
    InteractionLeaseOutcome,
    InteractionStationAllocator,
    InteractionStationCatalog,
)
from stretch_mujoco.agents.runtime import OfficeAgentRuntime
from stretch_mujoco import stretch_mujoco_simulator as simulator_module
from stretch_mujoco.npc.protocol import CommandStatus, NpcCommandReceipt
from stretch_mujoco.npc.schema import (
    NpcInteractionCatalogEntry,
    NpcInteractionStation,
    NpcPopulation,
)
from stretch_mujoco.semantics import ObjectType, SemanticObject, SemanticWorld


MODELS = Path(__file__).resolve().parents[1] / "stretch_mujoco" / "models"


class _Simulator:
    def __init__(self) -> None:
        self.commands = []
        self.receipts = []
        self.cancelled = []
        self.time = 10.0

    def pull_status(self):
        return SimpleNamespace(time=self.time)

    def submit_npc_command(self, command):
        self.commands.append(command)
        return command.command_id

    def pull_npc_receipts(self):
        receipts = tuple(self.receipts)
        self.receipts.clear()
        return receipts

    def cancel_npc_command(self, npc_id, command_id):
        self.cancelled.append((npc_id, command_id))


class _FailNthSubmitSimulator(_Simulator):
    def __init__(self, fail_at: int) -> None:
        super().__init__()
        self.fail_at = fail_at
        self.attempts = 0

    def submit_npc_command(self, command):
        self.attempts += 1
        if self.attempts == self.fail_at:
            raise RuntimeError("transport_down")
        return super().submit_npc_command(command)


def _conversation(station_id: str, prefix: str) -> NpcInteractionCatalogEntry:
    return NpcInteractionCatalogEntry(
        station_id,
        "conversation",
        {
            "speaker": NpcInteractionStation(f"{prefix}.speaker", 0.5),
            "listener": NpcInteractionStation(f"{prefix}.listener", -0.5),
        },
        {
            "allowed_actor_pairs": ["npc_npc"],
            "distance_m": {"min": 0.55, "max": 1.05},
            "yaw_tolerance_rad": 0.35,
        },
    )


def _population(*, schema_version: int = 3) -> NpcPopulation:
    return NpcPopulation(
        scene="fixture.xml",
        asset_manifest="fixture.json",
        npcs={},
        clock={},
        interaction_stations={
            "conversation": {
                "conversation.a": _conversation("conversation.a", "a"),
                "conversation.b": _conversation("conversation.b", "b"),
            },
            "handover": {},
            "seat": {},
            "workstation": {},
        },
        schema_version=schema_version,
    )


def _allocator(**kwargs) -> InteractionStationAllocator:
    return InteractionStationAllocator(
        InteractionStationCatalog.from_population(_population()), **kwargs
    )


def _session(session_id: str, participants=("npc.1", "npc.2"), preferred=None):
    return SimpleNamespace(
        session_id=session_id,
        participants=participants,
        preferred_station_id=preferred,
        station_id=None,
        lease_id=None,
    )


def _succeed_current_commands(
    simulator: _Simulator, driver: MujocoNpcActionDriver, session_id: str
):
    workflow = driver._conversations[session_id]
    expected_receipt_ids = tuple(workflow.command_ids.values())
    simulator.receipts.extend(
        NpcCommandReceipt(command_id, npc_id, CommandStatus.SUCCEEDED)
        for npc_id, command_id in workflow.command_ids.items()
    )
    result = driver.poll_conversation(session_id)
    assert result.receipt_ids == expected_receipt_ids
    return result


def _ready_conversation(
    simulator: _Simulator,
    driver: MujocoNpcActionDriver,
    session_id: str,
    participants=("npc.1", "npc.2"),
    preferred=None,
):
    session = _session(session_id, participants, preferred)
    assert driver.prepare_conversation(session).phase == "approach"
    assert _succeed_current_commands(simulator, driver, session_id).phase == "align"
    gate = _succeed_current_commands(simulator, driver, session_id)
    assert gate.phase == "alignment_gate"
    aligned = _succeed_current_commands(simulator, driver, session_id)
    assert aligned.status is ExecutionStatus.SUCCEEDED
    return session


def test_two_conversations_lease_distinct_stations_and_preferred_busy_fails() -> None:
    simulator = _Simulator()
    allocator = _allocator()
    driver = MujocoNpcActionDriver(
        simulator, {}, interaction_station_allocator=allocator
    )
    first = _session("session.1", preferred="conversation.a")
    second = _session(
        "session.2", ("npc.3", "npc.4"), preferred="conversation.b"
    )
    assert driver.prepare_conversation(first).status is ExecutionStatus.RUNNING
    assert driver.prepare_conversation(second).status is ExecutionStatus.RUNNING
    assert {first.station_id, second.station_id} == {
        "conversation.a",
        "conversation.b",
    }
    busy = driver.prepare_conversation(
        _session("session.3", ("npc.5", "npc.6"), "conversation.a")
    )
    assert busy.status is ExecutionStatus.FAILED
    assert busy.error == "interaction_station_busy"
    assert len(allocator.snapshot().active_leases) == 2


def test_lease_assignments_and_authored_contract_drive_alignment_gate() -> None:
    simulator = _Simulator()
    allocator = _allocator()
    driver = MujocoNpcActionDriver(
        simulator, {}, interaction_station_allocator=allocator
    )
    session = _session("session.contract", preferred="conversation.b")
    driver.prepare_conversation(session)
    assert [command.payload["site"] for command in simulator.commands] == [
        "b.speaker",
        "b.listener",
    ]
    _succeed_current_commands(simulator, driver, session.session_id)
    _succeed_current_commands(simulator, driver, session.session_id)
    gate_commands = simulator.commands[-2:]
    assert {command.payload["target_site"] for command in gate_commands} == {
        "b.speaker",
        "b.listener",
    }
    assert all(command.payload["interaction_distance_min"] == 0.55 for command in gate_commands)
    assert all(command.payload["interaction_distance_max"] == 1.05 for command in gate_commands)
    assert all(command.payload["interaction_yaw_tolerance"] == 0.35 for command in gate_commands)


def test_alignment_failure_releases_every_station_resource_with_failed_receipt() -> None:
    simulator = _Simulator()
    allocator = _allocator()
    driver = MujocoNpcActionDriver(
        simulator, {}, interaction_station_allocator=allocator
    )
    session = _session("session.alignment-fail")
    driver.prepare_conversation(session)
    _succeed_current_commands(simulator, driver, session.session_id)
    _succeed_current_commands(simulator, driver, session.session_id)
    workflow = driver._conversations[session.session_id]
    failed_id = next(iter(workflow.command_ids.values()))
    simulator.receipts.extend(
        NpcCommandReceipt(
            command_id,
            npc_id,
            CommandStatus.FAILED if command_id == failed_id else CommandStatus.SUCCEEDED,
        )
        for npc_id, command_id in workflow.command_ids.items()
    )
    result = driver.poll_conversation(session.session_id)
    assert result.status is ExecutionStatus.FAILED
    assert result.handle == failed_id
    assert allocator.snapshot().resource_owners == ()
    assert allocator.lease(session.lease_id).outcome is InteractionLeaseOutcome.FAILED


def test_partial_prepare_submit_rolls_back_commands_and_lease_without_fake_receipt() -> None:
    simulator = _Simulator()
    original_submit = simulator.submit_npc_command
    attempts = 0

    def fail_second(command):
        nonlocal attempts
        attempts += 1
        if attempts == 2:
            raise RuntimeError("transport_down")
        return original_submit(command)

    simulator.submit_npc_command = fail_second
    allocator = _allocator()
    driver = MujocoNpcActionDriver(
        simulator, {}, interaction_station_allocator=allocator
    )
    session = _session("session.submit-fail")
    result = driver.prepare_conversation(session)
    assert result.status is ExecutionStatus.FAILED
    assert result.handle is None
    assert result.receipt_ids == ()
    assert result.cleanup_evidence_id.startswith(
        "cleanup:conversation_prepare_submit_failed:"
    )
    assert simulator.cancelled == [("npc.1", simulator.commands[0].command_id)]
    assert allocator.snapshot().resource_owners == ()
    terminal = allocator.lease(session.lease_id)
    assert terminal.terminal_receipt_id is None
    assert terminal.cleanup_evidence_id == result.cleanup_evidence_id


@pytest.mark.parametrize(
    ("failed_stage", "fail_at"),
    [("align", 4), ("alignment_gate", 6)],
)
def test_partial_phase_submit_rolls_back_first_command_and_active_lease(
    failed_stage: str, fail_at: int
) -> None:
    simulator = _FailNthSubmitSimulator(fail_at)
    allocator = _allocator()
    driver = MujocoNpcActionDriver(
        simulator, {}, interaction_station_allocator=allocator
    )
    session = _session(f"session.submit-{failed_stage}")
    driver.prepare_conversation(session)
    result = _succeed_current_commands(simulator, driver, session.session_id)
    if failed_stage == "alignment_gate":
        assert result.phase == "align"
        result = _succeed_current_commands(simulator, driver, session.session_id)
    assert result.status is ExecutionStatus.FAILED
    assert result.phase == failed_stage
    assert result.handle is None
    assert result.receipt_ids
    assert result.cleanup_evidence_id.startswith(
        f"cleanup:conversation_{failed_stage}_submit_failed:"
    )
    assert result.cleanup_evidence_id not in result.receipt_ids
    assert simulator.cancelled[-1][1].endswith(f"{failed_stage}_npc.1")
    assert allocator.snapshot().resource_owners == ()
    terminal = allocator.lease(session.lease_id)
    assert terminal.terminal_receipt_id is None
    assert terminal.cleanup_evidence_id == result.cleanup_evidence_id
    assert session.session_id not in driver._conversations


@pytest.mark.parametrize(
    ("outcome", "expected"),
    [
        ("succeeded", InteractionLeaseOutcome.SUCCEEDED),
        ("failed", InteractionLeaseOutcome.FAILED),
        ("cancelled", InteractionLeaseOutcome.CANCELLED),
        ("timed_out", InteractionLeaseOutcome.TIMED_OUT),
    ],
)
def test_external_terminal_outcomes_distinguish_receipts_from_cleanup_evidence(
    outcome, expected
) -> None:
    simulator = _Simulator()
    allocator = _allocator()
    driver = MujocoNpcActionDriver(
        simulator, {}, interaction_station_allocator=allocator
    )
    session = _ready_conversation(simulator, driver, f"session.{outcome}")
    if outcome == "succeeded":
        result = driver.finish_conversation(session.session_id, outcome)
    else:
        result = driver.terminate_conversation(session.session_id, outcome, outcome)
    assert allocator.snapshot().resource_owners == ()
    terminal = allocator.lease(session.lease_id)
    assert terminal.outcome is expected
    if outcome == "succeeded":
        assert result.handle.startswith("session.")
        assert terminal.terminal_receipt_id == result.handle
        assert terminal.cleanup_evidence_id is None
    else:
        assert result.handle is None
        assert result.receipt_ids
        assert result.cleanup_evidence_id.startswith(
            f"cleanup:conversation_{outcome}_unconfirmed:"
        )
        assert terminal.terminal_receipt_id is None
        assert terminal.cleanup_evidence_id == result.cleanup_evidence_id


def test_cancel_without_physics_ack_is_explicit_unconfirmed_cleanup() -> None:
    simulator = _Simulator()
    allocator = _allocator()
    driver = MujocoNpcActionDriver(
        simulator, {}, interaction_station_allocator=allocator
    )
    session = _session("session.cancel-before-receipt")
    driver.prepare_conversation(session)
    result = driver.terminate_conversation(session.session_id, "cancelled", "user_cancelled")
    assert result.phase == "cleanup_unconfirmed"
    assert result.handle is None
    assert result.receipt_ids == ()
    assert result.cleanup_evidence_id.startswith(
        "cleanup:conversation_cancelled_unconfirmed:session.cancel-before-receipt:"
    )
    terminal = allocator.lease(session.lease_id)
    assert terminal.outcome is InteractionLeaseOutcome.CANCELLED
    assert terminal.terminal_receipt_id is None
    assert terminal.cleanup_evidence_id == result.cleanup_evidence_id
    assert allocator.snapshot().resource_owners == ()


@pytest.mark.parametrize("failed_participant", ["npc.1", "npc.2"])
def test_failed_talk_or_listener_hold_does_not_report_success_and_releases_lease(
    failed_participant: str,
) -> None:
    simulator = _Simulator()
    allocator = _allocator()
    driver = MujocoNpcActionDriver(
        simulator, {}, interaction_station_allocator=allocator
    )
    session = _ready_conversation(simulator, driver, "session.talk-fail")
    turn = SimpleNamespace(
        speaker="npc.1", listener="npc.2", turn_id="turn.1"
    )
    driver.play_turn(session, turn)
    workflow = driver._conversations[session.session_id]
    talk_id = workflow.command_ids["npc.1"]
    failed_id = workflow.command_ids[failed_participant]
    simulator.receipts.append(
        NpcCommandReceipt(
            talk_id,
            "npc.1",
            CommandStatus.FAILED
            if failed_participant == "npc.1"
            else CommandStatus.SUCCEEDED,
        )
    )
    listener_id = workflow.command_ids["npc.2"]
    simulator.receipts.append(
        NpcCommandReceipt(
            listener_id,
            "npc.2",
            CommandStatus.FAILED
            if failed_participant == "npc.2"
            else CommandStatus.SUCCEEDED,
        )
    )
    result = driver.poll_conversation(session.session_id)
    assert result.status is ExecutionStatus.FAILED
    assert result.handle == failed_id
    assert allocator.snapshot().resource_owners == ()


@pytest.mark.parametrize(
    ("command_status", "execution_status", "lease_outcome"),
    [
        (CommandStatus.CANCELLED, ExecutionStatus.CANCELLED, InteractionLeaseOutcome.CANCELLED),
        (CommandStatus.TIMED_OUT, ExecutionStatus.TIMED_OUT, InteractionLeaseOutcome.TIMED_OUT),
    ],
)
def test_current_command_terminal_receipt_confirms_cancel_or_timeout(
    command_status, execution_status, lease_outcome
) -> None:
    simulator = _Simulator()
    allocator = _allocator()
    driver = MujocoNpcActionDriver(
        simulator, {}, interaction_station_allocator=allocator
    )
    session = _ready_conversation(simulator, driver, f"session.{command_status.value}.receipt")
    driver.play_turn(
        session,
        SimpleNamespace(speaker="npc.1", listener="npc.2", turn_id="turn.terminal"),
    )
    workflow = driver._conversations[session.session_id]
    terminal_id = workflow.command_ids["npc.1"]
    simulator.receipts.extend(
        (
            NpcCommandReceipt(terminal_id, "npc.1", command_status),
            NpcCommandReceipt(
                workflow.command_ids["npc.2"], "npc.2", CommandStatus.SUCCEEDED
            ),
        )
    )
    result = driver.poll_conversation(session.session_id)
    terminal = allocator.lease(session.lease_id)
    assert result.status is execution_status
    assert result.handle == terminal_id
    assert terminal.outcome is lease_outcome
    assert terminal.terminal_receipt_id == terminal_id
    assert terminal.cleanup_evidence_id is None


def test_v3_runtime_loader_constructs_allocator_and_v2_strict_path_rejects(
    tmp_path: Path
) -> None:
    world = SemanticWorld.from_json(MODELS / "office_semantics.json")
    payload = json.loads((MODELS / "office_population.json").read_text(encoding="utf-8"))
    payload["schema_version"] = 3
    payload.pop("interaction_templates", None)
    payload.pop("appearance_catalog", None)
    payload.pop("trajectory_profile", None)
    sites = sorted(point.site for point in world.interaction_points.values())
    assert len(sites) >= 9
    payload["interaction_stations"] = {
        "conversation": {
            "conversation.a": {
                "roles": {
                    "speaker": {"site": sites[0], "yaw": 0.0},
                    "listener": {"site": sites[1], "yaw": 3.141592653589793},
                },
                "allowed_actor_pairs": ["npc_npc"],
                "distance_m": {"min": 0.55, "max": 1.05},
                "yaw_tolerance_rad": 0.35,
            }
        },
        "handover": {
            "handover.a": {
                "roles": {
                    "giver": {"site": sites[2], "yaw": 0.0},
                    "receiver": {"site": sites[3], "yaw": 3.141592653589793},
                    "robot": {"site": sites[4], "yaw": 0.0},
                },
                "modes": ["npc_to_npc"],
                "object_ids": ["cup"],
                "distance_m": {"min": 0.6, "max": 1.0},
                "yaw_tolerance_rad": 0.35,
                "transfer_site": sites[5],
            }
        },
        "seat": {
            "seat.a": {
                "roles": {
                    "ingress": {"site": sites[6], "yaw": 0.0},
                    "sit": {"site": sites[7], "yaw": 0.0},
                },
                "owner_entity": "chair_right",
                "seat_type": "chair",
                "slot_index": 1,
                "clearance_radius_m": 0.28,
            }
        },
        "workstation": {
            "workstation.a": {
                "roles": {"work": {"site": sites[8], "yaw": 0.0}},
                "workstation_entity": "workstation_right",
                "computer_entity": "computer_right",
                "seat_slot": "seat.a",
            }
        },
    }
    path = tmp_path / "population.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    runtime = OfficeAgentRuntime.from_json(world, path, auto_plan=False)
    assert isinstance(runtime.interaction_station_allocator, InteractionStationAllocator)
    assert runtime.population_schema_version == 3
    driver = MujocoNpcActionDriver(
        _Simulator(), {}, interaction_station_allocator=runtime.interaction_station_allocator
    )
    assert driver.prepare_conversation(_session("session.no-live-router")).error == (
        "interaction_station_route_unavailable"
    )

    legacy = _runtime_with_two_agents(_ReceiptDriver())
    legacy.population_schema_version = 2
    receipt = legacy.begin_conversation(_request("legacy.session"))
    assert receipt.accepted
    assert receipt.compatibility_mode == "population_v2_legacy_adapter"
    assert receipt.production_evidence is False
    session = legacy.conversation("legacy.session")
    assert session.compatibility_mode == "population_v2_legacy_adapter"
    assert session.production_evidence is False
    assert legacy.submit_dialogue_candidate(_candidate("legacy.session")).valid
    terminal_events = [
        event for event in legacy.events if event.event == "conversation_terminal"
    ]
    assert terminal_events
    assert all(
        event.details["compatibility_mode"] == "population_v2_legacy_adapter"
        and event.details["production_evidence"] is False
        for event in terminal_events
    )


def test_phase_2b_rejects_robot_before_npc_transport() -> None:
    driver = _ReceiptDriver()
    runtime = _runtime_with_two_agents(driver)
    runtime.world.objects["stretch_3"] = SemanticObject(
        "stretch_3",
        ObjectType.STRETCH_ROBOT,
        runtime.world.object("employee_01").binding,
        {},
    )
    receipt = runtime.begin_conversation(
        ConversationRequest(
            "session.robot",
            ("employee_01", "stretch_3"),
            "status",
            semantic_snapshot={"objects": {}},
        )
    )
    assert not receipt.accepted
    assert receipt.error == "phase_2b_npc_conversation_only"


def test_embodied_simulator_injects_allocator_and_interaction_driver(monkeypatch) -> None:
    allocator = _allocator()
    runtime = SimpleNamespace(
        trajectory_profile_path=None,
        population_npc_ids=frozenset({"npc.1", "npc.2"}),
        population_interaction_templates={},
        interaction_station_allocator=allocator,
        agents={},
        action_driver=None,
        interaction_driver=None,
    )
    driver = object()
    captured = {}
    simulator = SimpleNamespace(semantic_world=object(), agent_runtime=None)
    monkeypatch.setattr(
        simulator_module.OfficeAgentRuntime,
        "from_json",
        lambda *args, **kwargs: runtime,
    )
    monkeypatch.setattr(
        "stretch_mujoco.agents.simulation_bridge.create_mujoco_action_driver",
        lambda *args, **kwargs: captured.update(kwargs) or driver,
    )

    result = simulator_module.StretchMujocoSimulator.create_office_agent_runtime(
        simulator, "population.json", embodied=True
    )

    assert captured["interaction_station_allocator"] is allocator
    assert result.action_driver is driver
    assert result.interaction_driver is driver


def _runtime_with_two_agents(driver) -> OfficeAgentRuntime:
    world = SemanticWorld.from_json(MODELS / "office_semantics.json")
    binding = world.object("employee_01").binding
    world.objects["employee_02"] = SemanticObject(
        "employee_02", ObjectType.EMPLOYEE, binding, {"display_name": "employee_02"}
    )

    def employee(employee_id: str) -> EmployeeAgent:
        return EmployeeAgent.from_dict(
            employee_id,
            {
                "profile": {"role": "Tester", "department": "QA"},
                "initial_location": "workstation_right",
            },
        )

    return OfficeAgentRuntime(
        world,
        {employee_id: employee(employee_id) for employee_id in ("employee_01", "employee_02")},
        auto_plan=False,
        daily_events=False,
        interaction_driver=driver,
    )


class _ReceiptDriver:
    def __init__(self, *, fail_talk: bool = False) -> None:
        self.fail_talk = fail_talk
        self.finished = []

    def prepare_conversation(self, _session):
        _session.station_id = "conversation.a"
        _session.lease_id = f"interaction_lease:{_session.session_id}"
        return DriverResult(
            ExecutionStatus.SUCCEEDED,
            "aligned",
            "alignment.receipt.2",
            receipt_ids=("alignment.receipt.1", "alignment.receipt.2"),
        )

    def play_turn(self, _session, _turn):
        if self.fail_talk:
            return DriverResult(
                ExecutionStatus.FAILED,
                "talk",
                "talk.failed",
                "talk_failed",
                ("talk.failed", "listener.hold"),
            )
        return DriverResult(
            ExecutionStatus.SUCCEEDED,
            "talk",
            "talk.succeeded",
            receipt_ids=("talk.succeeded", "listener.hold"),
        )

    def finish_conversation(self, session_id, outcome):
        self.finished.append((session_id, outcome))
        return DriverResult(
            ExecutionStatus.SUCCEEDED,
            "terminal",
            "talk.succeeded",
            receipt_ids=("talk.succeeded", "listener.hold"),
        )

    def cancel_conversation(self, _session_id, reason):
        return DriverResult(ExecutionStatus.CANCELLED, "cancelled", "alignment.receipt", reason)


def _request(session_id: str, *, max_turns: int = 1) -> ConversationRequest:
    return ConversationRequest(
        session_id,
        ("employee_01", "employee_02"),
        "status",
        max_turns=max_turns,
        semantic_snapshot={
            "objects": {
                "employee_01": {"position": [0, 0, 0], "yaw": 0},
                "employee_02": {"position": [1, 0, 0], "yaw": 3.141592653589793},
            }
        },
    )


def _candidate(session_id: str) -> DialogueCandidate:
    return DialogueCandidate(
        "request.1",
        session_id,
        "turn.1",
        "employee_01",
        "employee_02",
        DialogueAct.GREETING,
        "Hello.",
        0.0,
    )


def test_failed_talk_never_commits_transcript_memory_or_semantic_event() -> None:
    runtime = _runtime_with_two_agents(_ReceiptDriver(fail_talk=True))
    assert runtime.begin_conversation(_request("session.runtime-fail")).accepted
    result = runtime.submit_dialogue_candidate(_candidate("session.runtime-fail"))
    assert not result.valid
    session = runtime.conversation("session.runtime-fail")
    assert session.turn == 0
    assert session.transcript == []
    assert not [event for event in runtime.events if event.event == "dialogue_turn_committed"]
    assert all(
        not [entry for entry in agent.memory.entries if entry.event == "dialogue_turn_committed"]
        for agent in runtime.agents.values()
    )


def test_success_uses_one_cleanup_path_and_does_not_double_emit() -> None:
    driver = _ReceiptDriver()
    runtime = _runtime_with_two_agents(driver)
    assert runtime.begin_conversation(_request("session.runtime-success")).accepted
    assert runtime.submit_dialogue_candidate(_candidate("session.runtime-success")).valid
    session = runtime.conversation("session.runtime-success")
    assert session.status.value == "completed"
    assert runtime._participant_reservations == {}
    assert driver.finished == [("session.runtime-success", "succeeded")]
    receipt = runtime.conversation_receipt("session.runtime-success")
    assert receipt.physical_receipt_ids == (
        "alignment.receipt.1",
        "alignment.receipt.2",
        "talk.succeeded",
        "listener.hold",
    )
    assert receipt.station_id == "conversation.a"
    assert receipt.lease_id == "interaction_lease:session.runtime-success"
    assert receipt.cleanup_evidence_id is None
    terminal_events = [
        event
        for event in runtime.events
        if event.event == "conversation_terminal"
        and event.details["session_id"] == "session.runtime-success"
    ]
    assert len(terminal_events) == 2
    assert len({event.event_id for event in terminal_events}) == 2
    assert all(
        event.details["physical_receipt_ids"] == list(receipt.physical_receipt_ids)
        for event in terminal_events
    )
    event_history = tuple(runtime.events)
    assert runtime.drain_events()
    assert tuple(runtime.events) == event_history
    assert runtime.drain_events() == ()
    runtime.complete_conversation("session.runtime-success")
    assert driver.finished == [("session.runtime-success", "succeeded")]
    assert len(
        [
            event
            for event in runtime.events
            if event.event == "conversation_terminal"
            and event.details["session_id"] == "session.runtime-success"
        ]
    ) == 2
