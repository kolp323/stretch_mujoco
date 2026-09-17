from types import SimpleNamespace

import pytest

from stretch_mujoco.agents.actions import ActionCommand, ActionExecution, ActionType, ExecutionStatus
from stretch_mujoco.agents.desk_work import (
    WORK_SESSION_COMPUTER_PARAMETER,
    WORK_SESSION_SEAT_PARAMETER,
    desk_work_session_actions,
)
from stretch_mujoco.agents.drivers import MujocoNpcActionDriver
from stretch_mujoco.agents.interaction_stations import (
    InteractionStationAllocationError,
    InteractionStationCatalog,
    SeatSlotAllocator,
)
from stretch_mujoco.npc.protocol import CommandStatus, NpcCommandReceipt
from stretch_mujoco.npc.schema import NpcInteractionCatalogEntry, NpcInteractionStation, NpcPopulation
from stretch_mujoco.semantics import ObjectType, RelationType, SemanticRelation


class FakeSimulator:
    def __init__(self) -> None:
        self.time = 1.0
        self.commands = []
        self.receipts = []
        self.cancelled = []

    def pull_status(self):
        return SimpleNamespace(time=self.time)

    def submit_npc_command(self, command):
        self.commands.append(command)
        return command.command_id

    def pull_npc_receipts(self):
        result = tuple(self.receipts)
        self.receipts.clear()
        return result

    def cancel_npc_command(self, npc_id, command_id):
        self.cancelled.append((npc_id, command_id))

    def succeed_latest(self, npc_id: str) -> None:
        self.receipts.append(
            NpcCommandReceipt(self.commands[-1].command_id, npc_id, CommandStatus.SUCCEEDED)
        )


def _entry(station_id, kind, roles, attributes):
    return NpcInteractionCatalogEntry(
        station_id,
        kind,
        {name: NpcInteractionStation(site, yaw) for name, (site, yaw) in roles.items()},
        attributes,
    )


def _catalog() -> InteractionStationCatalog:
    seats = {
        f"seat.sofa.{index}": _entry(
            f"seat.sofa.{index}",
            "seat",
            {
                "ingress": (f"sofa_{index}_ingress", 0.0),
                "sit": (f"sofa_{index}_sit", 1.57),
            },
            {
                "owner_entity": "sofa",
                "seat_type": "sofa",
                "slot_index": index,
                "clearance_radius_m": 0.45,
            },
        )
        for index in (0, 1)
    }
    population = NpcPopulation(
        scene="fixture.xml",
        asset_manifest="fixture.json",
        npcs={},
        clock={},
        interaction_stations={
            "conversation": {},
            "handover": {},
            "seat": seats,
            "workstation": {
                "workstation.desk": _entry(
                    "workstation.desk",
                    "workstation",
                    {"work": ("desk_work", 1.57)},
                    {
                        "workstation_entity": "desk",
                        "computer_entity": "computer",
                        "seat_slot": "seat.sofa.0",
                    },
                )
            },
        },
        schema_version=3,
    )
    return InteractionStationCatalog.from_population(population)


def _execution(execution_id, npc, action, target, parameters=None):
    return ActionExecution(
        execution_id=execution_id,
        command=ActionCommand(npc, action, target, parameters or {}),
        status=ExecutionStatus.RUNNING,
    )


def _driver(simulator, allocator, verifier=lambda npc, slot, phase, site: f"verify:{npc}:{phase}"):
    return MujocoNpcActionDriver(
        simulator,
        {
            "seat.sofa.0": "sofa_0_sit",
            "seat.sofa.1": "sofa_1_sit",
            "desk": "desk_work",
        },
        seat_yaws={"seat.sofa.0": 1.57, "seat.sofa.1": 1.57},
        seat_navigation_sites={
            "seat.sofa.0": "sofa_0_ingress",
            "seat.sofa.1": "sofa_1_ingress",
        },
        seat_slot_allocator=allocator,
        seat_verifier=verifier,
    )


def _complete_sit(driver, simulator, execution):
    result = driver.start(execution)
    execution.driver_handle = result.handle
    for _ in range(3):
        simulator.succeed_latest(execution.command.agent_id)
        result = driver.poll(execution)
        execution.driver_handle = result.handle
    return result


def test_sofa_slots_are_independent_and_same_slot_conflicts() -> None:
    allocator = SeatSlotAllocator(_catalog())
    first = allocator.reserve("seat.sofa.0", "npc.1", "sit.1")
    second = allocator.reserve("seat.sofa.1", "npc.2", "sit.2")
    assert first.slot_id != second.slot_id
    with pytest.raises(InteractionStationAllocationError, match="^seat_slot_busy:seat.sofa.0$"):
        allocator.reserve("seat.sofa.0", "npc.3", "sit.3")


def test_sit_receipts_gate_occupancy_move_and_stand_release() -> None:
    simulator = FakeSimulator()
    allocator = SeatSlotAllocator(_catalog())
    driver = _driver(simulator, allocator)
    sit = _execution("sit.1", "npc.1", ActionType.SIT, "seat.sofa.0")

    started = driver.start(sit)
    assert started.phase == "approach_seat"
    assert allocator.lease("seat.sofa.0").occupied is False
    for _ in range(3):
        simulator.succeed_latest("npc.1")
        result = driver.poll(sit)
        sit.driver_handle = result.handle
    assert result.status == ExecutionStatus.SUCCEEDED
    assert len(result.receipt_ids) == 4
    assert allocator.lease("seat.sofa.0").occupied is True

    move = _execution("move.1", "npc.1", ActionType.MOVE_TO, "desk")
    assert driver.start(move).error == "move_requires_stand_up"

    stand = _execution("stand.1", "npc.1", ActionType.STAND_UP, "seat.sofa.0")
    stand_result = driver.start(stand)
    stand.driver_handle = stand_result.handle
    for _ in range(2):
        simulator.succeed_latest("npc.1")
        stand_result = driver.poll(stand)
        stand.driver_handle = stand_result.handle
    assert stand_result.status == ExecutionStatus.SUCCEEDED
    assert allocator.lease("seat.sofa.0") is None


def test_missing_contact_receipt_fails_without_occupancy_commit() -> None:
    simulator = FakeSimulator()
    allocator = SeatSlotAllocator(_catalog())
    driver = _driver(simulator, allocator, verifier=lambda *_args: None)
    sit = _execution("sit.failed", "npc.1", ActionType.SIT, "seat.sofa.0")
    result = _complete_sit(driver, simulator, sit)
    assert result.status == ExecutionStatus.FAILED
    assert result.error == "seat_contact_verification_failed"
    assert allocator.lease("seat.sofa.0") is None


def test_work_cycle_receipt_keeps_slot_occupied_until_stand() -> None:
    simulator = FakeSimulator()
    allocator = SeatSlotAllocator(_catalog())
    driver = _driver(simulator, allocator)
    sit = _execution("work.session", "npc.1", ActionType.SIT, "seat.sofa.0")
    assert _complete_sit(driver, simulator, sit).status == ExecutionStatus.SUCCEEDED

    work = _execution(
        "work.1",
        "npc.1",
        ActionType.WORK,
        "desk",
        {WORK_SESSION_SEAT_PARAMETER: "seat.sofa.0", "_desk_work_duration_seconds": 10.0},
    )
    started = driver.start(work)
    work.driver_handle = started.handle
    assert started.phase == "work"
    assert simulator.commands[-1].payload["completion_marker"] == "work_cycle"
    assert simulator.commands[-1].payload["target_site"] == "desk_work"
    simulator.succeed_latest("npc.1")
    finished = driver.poll(work)
    assert finished.status == ExecutionStatus.SUCCEEDED
    assert finished.receipt_ids == (simulator.commands[-1].command_id,)
    assert allocator.lease("seat.sofa.0").occupied is True


class DeskWorld:
    def object(self, object_id):
        return SimpleNamespace(
            object_id=object_id,
            object_type={"desk": ObjectType.WORKSTATION, "slot": ObjectType.SEAT_SLOT}[object_id],
        )

    def find_relations(self, *, relation=None, object_id=None, subject=None):
        relations = [SemanticRelation("slot", RelationType.NEAR, "desk")]
        return tuple(
            item
            for item in relations
            if (relation is None or item.relation == relation)
            and (object_id is None or item.object == object_id)
            and (subject is None or item.subject == subject)
        )


def test_use_computer_lowering_is_sit_work_cycle_stand_idle_with_ids() -> None:
    agent = SimpleNamespace(
        agent_id="npc.1", state=SimpleNamespace(location="aisle")
    )
    actions = desk_work_session_actions(
        agent,
        DeskWorld(),
        "desk",
        duration_seconds=10.0,
        session_id="session.1",
        computer_id="computer",
    )
    assert [action.action for action in actions] == [
        ActionType.SIT,
        ActionType.WORK,
        ActionType.STAND_UP,
        ActionType.IDLE,
    ]
    assert all(action.parameters["_desk_work_session_id"] == "session.1" for action in actions)
    assert actions[1].parameters[WORK_SESSION_COMPUTER_PARAMETER] == "computer"
    assert actions[1].parameters[WORK_SESSION_SEAT_PARAMETER] == "slot"
