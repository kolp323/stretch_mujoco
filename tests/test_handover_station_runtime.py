from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from stretch_mujoco.agents.actions import (
    ActionCommand,
    ActionExecution,
    ActionType,
    ExecutionStatus,
)
from stretch_mujoco.agents.drivers import MujocoNpcActionDriver
from stretch_mujoco.agents.interaction_stations import (
    InMemoryInteractionLeaseJournal,
    InteractionLeaseOutcome,
    InteractionStationAllocator,
    InteractionStationCatalog,
)
from stretch_mujoco.agents.runtime import OfficeAgentRuntime
from stretch_mujoco.npc.protocol import CommandStatus, NpcCommandKind, NpcCommandReceipt
from stretch_mujoco.npc.schema import (
    NpcInteractionCatalogEntry,
    NpcInteractionStation,
    NpcPopulation,
)
from stretch_mujoco.semantics import RelationType, SemanticWorld


MODELS = Path(__file__).resolve().parents[1] / "stretch_mujoco" / "models"


class _Simulator:
    def __init__(self, *, fail_at: int | None = None) -> None:
        self.commands = []
        self.receipts = []
        self.cancelled = []
        self.owners: dict[str, str] = {}
        self.time = 10.0
        self.fail_at = fail_at
        self.attempts = 0

    def pull_status(self):
        return SimpleNamespace(time=self.time)

    def submit_npc_command(self, command):
        self.attempts += 1
        if self.attempts == self.fail_at:
            raise RuntimeError("transport_down")
        self.commands.append(command)
        return command.command_id

    def pull_npc_receipts(self):
        receipts = tuple(self.receipts)
        self.receipts.clear()
        return receipts

    def pull_npc_states(self):
        npc_ids = {command.npc_id for command in self.commands}
        return {
            npc_id: SimpleNamespace(
                held_objects=tuple(
                    sorted(
                        object_id
                        for object_id, owner in self.owners.items()
                        if owner == npc_id
                    )
                )
            )
            for npc_id in npc_ids
        }

    def cancel_npc_command(self, npc_id, command_id):
        self.cancelled.append((npc_id, command_id))


def _handover(station_id: str, prefix: str) -> NpcInteractionCatalogEntry:
    return NpcInteractionCatalogEntry(
        station_id,
        "handover",
        {
            "giver": NpcInteractionStation(f"{prefix}.giver", 0.4),
            "receiver": NpcInteractionStation(f"{prefix}.receiver", -2.7),
            "robot": NpcInteractionStation(f"{prefix}.robot", 0.4),
        },
        {
            "modes": ["npc_to_npc"],
            "object_ids": ["parcel", "report", "soda_can"],
            "transfer_site": f"{prefix}.transfer",
            "distance_m": {"min": 0.58, "max": 0.92},
            "yaw_tolerance_rad": 0.22,
        },
    )


def _population() -> NpcPopulation:
    return NpcPopulation(
        scene="fixture.xml",
        asset_manifest="fixture.json",
        npcs={},
        clock={},
        interaction_stations={
            "conversation": {},
            "handover": {
                "handover.a": _handover("handover.a", "a"),
                "handover.b": _handover("handover.b", "b"),
            },
            "seat": {},
            "workstation": {},
        },
        schema_version=3,
    )


def _allocator(*, route_cost=None, journal=None) -> InteractionStationAllocator:
    return InteractionStationAllocator(
        InteractionStationCatalog.from_population(_population()),
        route_cost=route_cost,
        journal=journal,
    )


def _execution(
    execution_id: str,
    giver: str,
    receiver: str,
    object_id: str,
    preferred: str | None = None,
) -> ActionExecution:
    parameters = {"object": object_id}
    if preferred is not None:
        parameters["preferred_station_id"] = preferred
    return ActionExecution(
        execution_id=execution_id,
        command=ActionCommand(giver, ActionType.HANDOVER, receiver, parameters),
        status=ExecutionStatus.RUNNING,
    )


def _command(simulator: _Simulator, command_id: str):
    return next(command for command in simulator.commands if command.command_id == command_id)


def _finish_stage(
    simulator: _Simulator,
    driver: MujocoNpcActionDriver,
    execution: ActionExecution,
    *,
    failed_status: CommandStatus | None = None,
    failed_reason: str = "injected_failure",
):
    workflow = driver._handovers[execution.execution_id]
    first = True
    for npc_id, command_id in workflow.command_ids.items():
        command = _command(simulator, command_id)
        status = failed_status if first and failed_status is not None else CommandStatus.SUCCEEDED
        first = False
        if status == CommandStatus.SUCCEEDED:
            if command.kind == NpcCommandKind.DETACH_OBJECT:
                simulator.owners.pop(str(command.payload["object"]), None)
            elif command.kind == NpcCommandKind.ATTACH_OBJECT:
                simulator.owners[str(command.payload["object"])] = npc_id
        simulator.receipts.append(
            NpcCommandReceipt(
                command_id,
                npc_id,
                status,
                None if status == CommandStatus.SUCCEEDED else failed_reason,
            )
        )
    return driver.poll(execution)


def _queue_stage_receipts(
    simulator: _Simulator,
    driver: MujocoNpcActionDriver,
    execution: ActionExecution,
    *,
    failed_status: CommandStatus | None = None,
    failed_reason: str = "injected_failure",
) -> None:
    workflow = driver._handovers[execution.execution_id]
    first = True
    for npc_id, command_id in workflow.command_ids.items():
        command = _command(simulator, command_id)
        status = failed_status if first and failed_status is not None else CommandStatus.SUCCEEDED
        first = False
        if status == CommandStatus.SUCCEEDED:
            if command.kind == NpcCommandKind.DETACH_OBJECT:
                simulator.owners.pop(str(command.payload["object"]), None)
            elif command.kind == NpcCommandKind.ATTACH_OBJECT:
                simulator.owners[str(command.payload["object"])] = npc_id
        simulator.receipts.append(
            NpcCommandReceipt(
                command_id,
                npc_id,
                status,
                None if status == CommandStatus.SUCCEEDED else failed_reason,
            )
        )


def _advance_to_receive(
    simulator: _Simulator,
    driver: MujocoNpcActionDriver,
    execution: ActionExecution,
):
    assert driver.start(execution).phase == "rendezvous"
    assert _finish_stage(simulator, driver, execution).phase == "aligned"
    assert _finish_stage(simulator, driver, execution).phase == "giver_ready"
    assert _finish_stage(simulator, driver, execution).phase == "release"
    result = _finish_stage(simulator, driver, execution)
    assert result.phase == "receive"
    return result


def test_v3_selects_distinct_stations_and_claims_object_atomically() -> None:
    simulator = _Simulator()
    allocator = _allocator(
        route_cost=lambda _participant, _actor_kind, site: (
            1.0 if site.startswith("b.") else 5.0
        )
    )
    driver = MujocoNpcActionDriver(simulator, {}, interaction_station_allocator=allocator)
    first = _execution("handover.1", "npc.1", "npc.2", "parcel")
    second = _execution("handover.2", "npc.3", "npc.4", "report")

    assert driver.start(first).status is ExecutionStatus.RUNNING
    assert allocator.lease_for_session("handover.1").station_id == "handover.b"
    preferred_busy = driver.start(
        _execution("handover.busy", "npc.5", "npc.6", "report", "handover.b")
    )
    assert preferred_busy.status is ExecutionStatus.FAILED
    assert preferred_busy.error == "interaction_station_busy"

    object_conflict = driver.start(
        _execution("handover.object", "npc.5", "npc.6", "parcel", "handover.a")
    )
    assert object_conflict.status is ExecutionStatus.FAILED
    assert object_conflict.error == "interaction_station_busy"

    assert driver.start(second).status is ExecutionStatus.RUNNING
    assert allocator.lease_for_session("handover.2").station_id == "handover.a"
    owners = dict(allocator.snapshot().resource_owners)
    assert owners["object:parcel"] == "interaction_lease:handover.1"
    assert owners["object:report"] == "interaction_lease:handover.2"
    assert owners["station:handover.b"] == "interaction_lease:handover.1"
    assert owners["station:handover.a"] == "interaction_lease:handover.2"
    assert len(allocator.snapshot().active_leases) == 2


def test_v3_authored_contract_transfer_site_and_full_receipt_chain() -> None:
    simulator = _Simulator()
    simulator.owners["parcel"] = "npc.1"
    allocator = _allocator()
    driver = MujocoNpcActionDriver(simulator, {}, interaction_station_allocator=allocator)
    execution = _execution(
        "handover.contract", "npc.1", "npc.2", "parcel", "handover.b"
    )

    assert driver.start(execution).phase == "rendezvous"
    assert [command.payload["site"] for command in simulator.commands] == [
        "b.giver",
        "b.receiver",
    ]
    _finish_stage(simulator, driver, execution)
    assert [command.payload["yaw"] for command in simulator.commands[-2:]] == [0.4, -2.7]
    _finish_stage(simulator, driver, execution)
    for command in simulator.commands[-2:]:
        assert command.payload["completion_marker"] == "handover_ready"
        assert command.payload["interaction_distance_min"] == 0.58
        assert command.payload["interaction_distance_max"] == 0.92
        assert command.payload["interaction_yaw_tolerance"] == 0.22
    _finish_stage(simulator, driver, execution)
    detach = simulator.commands[-1]
    assert detach.kind is NpcCommandKind.DETACH_OBJECT
    assert detach.payload["site"] == "b.transfer"
    _finish_stage(simulator, driver, execution)
    result = _finish_stage(simulator, driver, execution)

    assert result.status is ExecutionStatus.SUCCEEDED
    assert result.station_id == "handover.b"
    assert result.lease_id == "interaction_lease:handover.contract"
    assert result.compatibility_mode == "population_v3_candidate"
    assert result.production_evidence is False
    assert len(result.receipt_ids) == 8
    assert result.handle == result.receipt_ids[-1]
    assert simulator.owners == {"parcel": "npc.2"}
    terminal = allocator.lease(result.lease_id)
    assert terminal.outcome is InteractionLeaseOutcome.SUCCEEDED
    assert terminal.terminal_receipt_id == result.receipt_ids[-1]
    assert allocator.snapshot().resource_owners == ()
    assert driver.start(execution) == result


@pytest.mark.parametrize(
    ("fail_at", "failed_stage"),
    ((2, "rendezvous"), (4, "aligned"), (6, "ready")),
)
def test_every_two_actor_stage_rolls_back_partial_submit(
    fail_at: int, failed_stage: str
) -> None:
    simulator = _Simulator(fail_at=fail_at)
    allocator = _allocator()
    driver = MujocoNpcActionDriver(simulator, {}, interaction_station_allocator=allocator)
    execution = _execution("handover.partial", "npc.1", "npc.2", "parcel")

    result = driver.start(execution)
    while result.status is ExecutionStatus.RUNNING and simulator.attempts < fail_at:
        result = _finish_stage(simulator, driver, execution)

    assert result.status is ExecutionStatus.FAILED
    assert failed_stage in result.cleanup_evidence_id
    assert result.cleanup_evidence_id not in result.receipt_ids
    assert simulator.cancelled[-1][1] == simulator.commands[-1].command_id
    assert allocator.snapshot().resource_owners == ()
    terminal = allocator.lease("interaction_lease:handover.partial")
    assert terminal.terminal_receipt_id is None
    assert terminal.cleanup_evidence_id == result.cleanup_evidence_id


@pytest.mark.parametrize(
    ("outcome", "expected_status", "expected_outcome"),
    (
        ("cancel", ExecutionStatus.CANCELLED, InteractionLeaseOutcome.CANCELLED),
        ("timeout", ExecutionStatus.TIMED_OUT, InteractionLeaseOutcome.TIMED_OUT),
    ),
)
def test_detach_then_cancel_or_timeout_requires_confirmed_giver_reattach(
    outcome: str,
    expected_status: ExecutionStatus,
    expected_outcome: InteractionLeaseOutcome,
) -> None:
    simulator = _Simulator()
    simulator.owners["parcel"] = "npc.1"
    allocator = _allocator()
    driver = MujocoNpcActionDriver(simulator, {}, interaction_station_allocator=allocator)
    execution = _execution("handover.reconcile", "npc.1", "npc.2", "parcel")
    _advance_to_receive(simulator, driver, execution)
    assert simulator.owners.get("parcel") is None

    if outcome == "cancel":
        result = driver.cancel(execution, "operator_cancelled")
    else:
        simulator.time = 1000.0
        result = driver.poll(execution)
    assert result.status is ExecutionStatus.RUNNING
    assert result.phase == "rollback"
    result = _finish_stage(simulator, driver, execution)

    assert result.status is expected_status
    assert simulator.owners == {"parcel": "npc.1"}
    terminal = allocator.lease("interaction_lease:handover.reconcile")
    assert terminal.outcome is expected_outcome
    assert terminal.terminal_receipt_id == result.receipt_ids[-1]
    assert terminal.cleanup_evidence_id is None
    assert allocator.snapshot().resource_owners == ()


def test_receive_failure_rolls_back_and_rollback_failure_is_explicit() -> None:
    simulator = _Simulator()
    simulator.owners["parcel"] = "npc.1"
    allocator = _allocator()
    driver = MujocoNpcActionDriver(simulator, {}, interaction_station_allocator=allocator)
    execution = _execution("handover.rollback-fails", "npc.1", "npc.2", "parcel")
    _advance_to_receive(simulator, driver, execution)

    result = _finish_stage(
        simulator,
        driver,
        execution,
        failed_status=CommandStatus.FAILED,
        failed_reason="receiver_grasp_failed",
    )
    assert result.phase == "rollback"
    result = _finish_stage(
        simulator,
        driver,
        execution,
        failed_status=CommandStatus.FAILED,
        failed_reason="giver_reattach_failed",
    )

    assert result.status is ExecutionStatus.FAILED
    assert result.error == "giver_reattach_failed"
    assert result.cleanup_evidence_id.startswith(
        "cleanup:handover_reconciliation_required:"
    )
    assert simulator.owners.get("parcel") is None
    terminal = allocator.lease("interaction_lease:handover.rollback-fails")
    assert terminal.terminal_receipt_id is None
    assert terminal.cleanup_evidence_id == result.cleanup_evidence_id
    assert allocator.snapshot().resource_owners == ()


def test_receive_submit_exception_reconciles_before_terminal_failure() -> None:
    simulator = _Simulator(fail_at=8)
    simulator.owners["parcel"] = "npc.1"
    allocator = _allocator()
    driver = MujocoNpcActionDriver(simulator, {}, interaction_station_allocator=allocator)
    execution = _execution("handover.receive-submit", "npc.1", "npc.2", "parcel")

    assert driver.start(execution).phase == "rendezvous"
    _finish_stage(simulator, driver, execution)
    _finish_stage(simulator, driver, execution)
    _finish_stage(simulator, driver, execution)
    result = _finish_stage(simulator, driver, execution)
    assert result.phase == "rollback"
    result = _finish_stage(simulator, driver, execution)

    assert result.status is ExecutionStatus.FAILED
    assert result.error == "handover_receive_submit_failed:transport_down"
    assert simulator.owners == {"parcel": "npc.1"}
    assert allocator.snapshot().resource_owners == ()


def test_release_submit_exception_cleans_lease_without_changing_owner() -> None:
    simulator = _Simulator(fail_at=7)
    simulator.owners["parcel"] = "npc.1"
    allocator = _allocator()
    driver = MujocoNpcActionDriver(simulator, {}, interaction_station_allocator=allocator)
    execution = _execution("handover.release-submit", "npc.1", "npc.2", "parcel")
    assert driver.start(execution).phase == "rendezvous"
    _finish_stage(simulator, driver, execution)
    _finish_stage(simulator, driver, execution)
    result = _finish_stage(simulator, driver, execution)

    assert result.status is ExecutionStatus.FAILED
    assert result.cleanup_evidence_id.startswith("cleanup:handover_release_submit_failed:")
    assert simulator.owners == {"parcel": "npc.1"}
    assert allocator.snapshot().resource_owners == ()


def test_rollback_submit_exception_is_reconciliation_required() -> None:
    simulator = _Simulator()
    simulator.owners["parcel"] = "npc.1"
    allocator = _allocator()
    driver = MujocoNpcActionDriver(simulator, {}, interaction_station_allocator=allocator)
    execution = _execution("handover.rollback-submit", "npc.1", "npc.2", "parcel")
    _advance_to_receive(simulator, driver, execution)
    simulator.fail_at = simulator.attempts + 1

    result = _finish_stage(
        simulator,
        driver,
        execution,
        failed_status=CommandStatus.FAILED,
        failed_reason="receiver_grasp_failed",
    )

    assert result.status is ExecutionStatus.FAILED
    assert result.error == "handover_reconciliation_submit_failed:transport_down"
    assert result.cleanup_evidence_id.startswith(
        "cleanup:handover_reconciliation_required:"
    )
    assert simulator.owners.get("parcel") is None
    assert allocator.snapshot().resource_owners == ()


def test_cancel_before_detach_uses_unconfirmed_cleanup_not_physical_receipt() -> None:
    simulator = _Simulator()
    allocator = _allocator()
    driver = MujocoNpcActionDriver(simulator, {}, interaction_station_allocator=allocator)
    execution = _execution("handover.cancel-early", "npc.1", "npc.2", "parcel")
    assert driver.start(execution).status is ExecutionStatus.RUNNING

    result = driver.cancel(execution, "operator_cancelled")

    assert result.status is ExecutionStatus.CANCELLED
    assert result.receipt_ids == ()
    assert result.cleanup_evidence_id.startswith(
        "cleanup:handover_cancelled_unconfirmed:"
    )
    terminal = allocator.lease("interaction_lease:handover.cancel-early")
    assert terminal.terminal_receipt_id is None
    assert terminal.cleanup_evidence_id == result.cleanup_evidence_id
    assert allocator.snapshot().resource_owners == ()


def test_reconstructed_active_object_claim_blocks_competing_handover() -> None:
    journal = InMemoryInteractionLeaseJournal()
    first_allocator = _allocator(journal=journal)
    first_driver = MujocoNpcActionDriver(
        _Simulator(), {}, interaction_station_allocator=first_allocator
    )
    assert first_driver.start(
        _execution("handover.persisted", "npc.1", "npc.2", "parcel", "handover.a")
    ).status is ExecutionStatus.RUNNING

    recovered = InteractionStationAllocator.reconstruct(
        first_allocator.catalog,
        journal,
    )
    competing_driver = MujocoNpcActionDriver(
        _Simulator(), {}, interaction_station_allocator=recovered
    )
    result = competing_driver.start(
        _execution("handover.competing", "npc.3", "npc.4", "parcel", "handover.b")
    )

    assert result.status is ExecutionStatus.FAILED
    assert result.error == "interaction_station_busy"
    assert dict(recovered.snapshot().resource_owners)["object:parcel"] == (
        "interaction_lease:handover.persisted"
    )


def test_v3_without_live_route_callback_fails_closed() -> None:
    allocator = _allocator(route_cost=lambda *_args: None)
    driver = MujocoNpcActionDriver(
        _Simulator(), {}, interaction_station_allocator=allocator
    )
    result = driver.start(_execution("handover.no-route", "npc.1", "npc.2", "parcel"))
    assert result.status is ExecutionStatus.FAILED
    assert result.error == "interaction_station_route_unavailable"
    assert allocator.snapshot().resource_owners == ()


def test_v2_legacy_adapter_runs_but_never_produces_production_evidence() -> None:
    simulator = _Simulator()
    simulator.owners["parcel"] = "npc.1"
    driver = MujocoNpcActionDriver(
        simulator,
        {},
        handover_sites={"npc.2": "legacy.receiver"},
        handover_role_sites={("npc.1", "npc.2"): ("legacy.giver", "legacy.receiver")},
        interaction_yaws={"npc.1": 0.4, "npc.2": -2.7},
    )
    execution = _execution("handover.legacy", "npc.1", "npc.2", "parcel")
    assert driver.start(execution).status is ExecutionStatus.RUNNING
    for _ in range(5):
        result = _finish_stage(simulator, driver, execution)
    assert result.status is ExecutionStatus.SUCCEEDED
    assert result.compatibility_mode == "population_v2_legacy_adapter"
    assert result.production_evidence is False
    assert result.station_id is None
    assert result.lease_id is None


def _runtime_handover(*, legacy: bool = False):
    world = SemanticWorld.from_json(MODELS / "office_semantics.json")
    simulator = _Simulator()
    simulator.owners["soda_can"] = "npc_alex_chen"
    allocator = None if legacy else _allocator()
    driver = MujocoNpcActionDriver(
        simulator,
        {},
        handover_sites={"npc_morgan_lee": "legacy.receiver"} if legacy else None,
        handover_role_sites=(
            {("npc_alex_chen", "npc_morgan_lee"): ("legacy.giver", "legacy.receiver")}
            if legacy
            else None
        ),
        interaction_yaws=(
            {"npc_alex_chen": 0.4, "npc_morgan_lee": -2.7} if legacy else None
        ),
        interaction_station_allocator=allocator,
    )
    runtime = OfficeAgentRuntime.from_json(
        world,
        MODELS / "office_population.production.example.json",
        auto_plan=False,
        action_driver=driver,
    )
    giver = runtime.agents["npc_alex_chen"]
    giver.state.held_object = "soda_can"
    runtime.reservations.reserve("soda_can", giver.agent_id)
    for relation in tuple(world.find_relations(subject="soda_can")):
        if relation.relation in {RelationType.ON, RelationType.INSIDE}:
            world.remove_relation("soda_can", relation.relation, relation.object)
    world.add_relation(giver.agent_id, RelationType.HOLDS, "soda_can")
    command = ActionCommand(
        giver.agent_id,
        ActionType.HANDOVER,
        "npc_morgan_lee",
        {} if legacy else {"preferred_station_id": "handover.a"},
    )
    assert runtime.submit_action(command).valid
    return runtime, world, simulator, driver, allocator, giver.executor


def test_runtime_commits_owner_only_after_observed_attachment_terminal_receipt() -> None:
    runtime, world, simulator, driver, allocator, execution = _runtime_handover()
    for _ in range(5):
        _queue_stage_receipts(simulator, driver, execution)
        runtime.tick(0.3)

    assert execution.status is ExecutionStatus.SUCCEEDED
    assert world.find_relations(
        subject="npc_morgan_lee",
        relation=RelationType.HOLDS,
        object_id="soda_can",
    )
    assert not world.find_relations(
        subject="npc_alex_chen",
        relation=RelationType.HOLDS,
        object_id="soda_can",
    )
    action_event = [
        event
        for event in runtime.events
        if event.event == "action_succeeded" and event.details["action"] == "handover"
    ][-1]
    semantic_event = [
        event
        for event in runtime.events
        if event.event == "semantic_commit" and event.details["action"] == "handover"
    ][-1]
    assert len(action_event.details["physical_receipt_ids"]) == 8
    assert action_event.causation_id == action_event.details["physical_receipt_ids"][-1]
    assert semantic_event.causation_id == action_event.event_id
    assert action_event.details["compatibility_mode"] == "population_v3_candidate"
    assert action_event.details["production_evidence"] is False
    assert allocator.snapshot().resource_owners == ()


def test_marker_failure_preserves_semantic_owner_and_releases_all_resources() -> None:
    runtime, world, simulator, driver, allocator, execution = _runtime_handover()
    _queue_stage_receipts(simulator, driver, execution)
    runtime.tick(0.3)
    _queue_stage_receipts(simulator, driver, execution)
    runtime.tick(0.3)
    _queue_stage_receipts(
        simulator,
        driver,
        execution,
        failed_status=CommandStatus.FAILED,
        failed_reason="handover_marker_missing",
    )
    runtime.tick(0.3)

    assert execution.status is ExecutionStatus.FAILED
    assert world.find_relations(
        subject="npc_alex_chen",
        relation=RelationType.HOLDS,
        object_id="soda_can",
    )
    assert not world.find_relations(
        subject="npc_morgan_lee",
        relation=RelationType.HOLDS,
        object_id="soda_can",
    )
    assert not [
        event
        for event in runtime.events
        if event.event == "semantic_commit" and event.details["action"] == "handover"
    ]
    failed_event = [
        event
        for event in runtime.events
        if event.event == "action_failed" and event.details["action"] == "handover"
    ][-1]
    assert failed_event.details["production_evidence"] is False
    assert allocator.snapshot().resource_owners == ()


def test_v2_runtime_event_is_compatibility_only() -> None:
    runtime, _world, simulator, driver, allocator, execution = _runtime_handover(
        legacy=True
    )
    assert allocator is None
    for _ in range(5):
        _queue_stage_receipts(simulator, driver, execution)
        runtime.tick(0.3)

    event = [
        item
        for item in runtime.events
        if item.event == "action_succeeded" and item.details["action"] == "handover"
    ][-1]
    assert event.details["compatibility_mode"] == "population_v2_legacy_adapter"
    assert event.details["production_evidence"] is False


def test_phase_2c_rejects_robot_actor_mode_before_transport() -> None:
    simulator = _Simulator()
    driver = MujocoNpcActionDriver(simulator, {}, interaction_station_allocator=_allocator())
    execution = ActionExecution(
        execution_id="handover.robot-rejected",
        command=ActionCommand(
            "npc.1",
            ActionType.HANDOVER,
            "stretch_3",
            {"object": "parcel", "actor_mode": "npc_to_robot"},
        ),
        status=ExecutionStatus.RUNNING,
    )
    result = driver.start(execution)
    assert result.status is ExecutionStatus.FAILED
    assert result.error == "phase_2c_npc_handover_only"
    assert simulator.commands == []


def test_legacy_conversation_pair_preserves_speaker_listener_order() -> None:
    driver = MujocoNpcActionDriver(
        _Simulator(),
        {},
        conversation_role_sites={("npc.1", "npc.2"): ("speaker.site", "listener.site")},
    )
    assert driver._conversation_sites("npc.1", "npc.2") == {
        "npc.1": "speaker.site",
        "npc.2": "listener.site",
    }
