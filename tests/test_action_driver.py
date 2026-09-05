from pathlib import Path
from types import SimpleNamespace

from stretch_mujoco.agents import (
    ActionCommand,
    ActionExecution,
    ActionType,
    ExecutionStatus,
    OfficeAgentRuntime,
    RobotToNpcHandoverBridge,
)
from stretch_mujoco.agents.drivers import MujocoNpcActionDriver
from stretch_mujoco.npc import CommandStatus, NpcCommandReceipt
from stretch_mujoco.semantics import SemanticWorld

MODELS = Path(__file__).resolve().parents[1] / "stretch_mujoco" / "models"


class FakeSimulator:
    def __init__(self) -> None:
        self.command = None
        self.commands = []
        self.receipts = []

    def pull_status(self):
        return SimpleNamespace(time=10.0)

    def submit_npc_command(self, command):
        self.command = command
        self.commands.append(command)
        return command.command_id

    def pull_npc_receipts(self):
        receipts = tuple(self.receipts)
        self.receipts.clear()
        return receipts

    def cancel_npc_command(self, npc_id, command_id):
        pass


def runtime_with_driver(simulator: FakeSimulator) -> OfficeAgentRuntime:
    world = SemanticWorld.from_json(MODELS / "office_semantics.json")
    driver = MujocoNpcActionDriver(
        simulator,
        {"meeting_table": "meeting_human_stand_site"},
    )
    return OfficeAgentRuntime.from_json(
        world,
        MODELS / "office_agents.json",
        auto_plan=False,
        action_driver=driver,
    )


def test_move_commits_location_only_after_physical_receipt() -> None:
    simulator = FakeSimulator()
    runtime = runtime_with_driver(simulator)
    agent = runtime.agents["employee_01"]
    original_location = agent.state.location

    result = runtime.submit_action(
        ActionCommand("employee_01", ActionType.MOVE_TO, "meeting_table")
    )
    runtime.tick(5.0)

    assert result.valid
    assert agent.state.location == original_location
    simulator.receipts.append(
        NpcCommandReceipt(
            simulator.command.command_id,
            "employee_01",
            CommandStatus.SUCCEEDED,
            started_at=10.0,
            finished_at=12.0,
        )
    )
    runtime.tick(0.25)

    assert agent.state.location == "meeting_table"


def test_failed_physical_move_does_not_commit_location() -> None:
    simulator = FakeSimulator()
    runtime = runtime_with_driver(simulator)
    agent = runtime.agents["employee_01"]
    original_location = agent.state.location
    runtime.submit_action(ActionCommand("employee_01", ActionType.MOVE_TO, "meeting_table"))
    simulator.receipts.append(
        NpcCommandReceipt(
            simulator.command.command_id,
            "employee_01",
            CommandStatus.TIMED_OUT,
            reason="deadline_exceeded",
            started_at=10.0,
            finished_at=40.0,
        )
    )

    runtime.tick(0.25)

    assert agent.state.location == original_location
    assert agent.executor.status.value == "timed_out"
    assert agent.state.availability == "available"


def test_sit_approaches_aligns_and_marks_seated_before_semantic_commit() -> None:
    simulator = FakeSimulator()
    world = SemanticWorld.from_json(MODELS / "office_semantics.json")
    driver = MujocoNpcActionDriver(
        simulator,
        {"chair_right": "chair_right_sit"},
        seat_yaws={"chair_right": 3.1415926},
    )
    runtime = OfficeAgentRuntime.from_json(
        world,
        MODELS / "office_agents.json",
        auto_plan=False,
        action_driver=driver,
    )
    agent = runtime.agents["employee_01"]

    result = runtime.submit_action(ActionCommand("employee_01", ActionType.SIT, "chair_right"))
    assert result.valid
    assert simulator.command.kind.value == "move_to"
    assert agent.state.location == "workstation_right"

    for expected_kind, expected_phase in (
        ("align_to", "align_seat"),
        ("play_animation", "sit_transition"),
    ):
        simulator.receipts.append(
            NpcCommandReceipt(simulator.command.command_id, "employee_01", CommandStatus.SUCCEEDED)
        )
        runtime.tick(0.25)
        assert simulator.command.kind.value == expected_kind
        assert agent.executor.phase == expected_phase
        assert agent.state.location == "workstation_right"

    simulator.receipts.append(
        NpcCommandReceipt(simulator.command.command_id, "employee_01", CommandStatus.SUCCEEDED)
    )
    runtime.tick(0.25)

    assert agent.state.location == "chair_right"
    assert agent.executor.status == ExecutionStatus.SUCCEEDED


def test_npc_handover_waits_for_ready_release_and_receive_receipts() -> None:
    simulator = FakeSimulator()
    driver = MujocoNpcActionDriver(simulator, {})
    execution = ActionExecution(
        execution_id="handover_1",
        command=ActionCommand(
            "employee_01",
            ActionType.HANDOVER,
            "employee_02",
            {"object": "parcel"},
        ),
        status=ExecutionStatus.RUNNING,
    )

    result = driver.start(execution)
    for expected_phase in ("receiver_ready", "release", "receive", "received"):
        simulator.receipts.append(
            NpcCommandReceipt(
                simulator.command.command_id, simulator.command.npc_id, CommandStatus.SUCCEEDED
            )
        )
        result = driver.poll(execution)
        assert result.phase == expected_phase

    assert result.status == ExecutionStatus.SUCCEEDED
    assert [command.kind.value for command in simulator.commands] == [
        "interaction_cue",
        "interaction_cue",
        "detach_object",
        "attach_object",
    ]


def test_robot_handover_requires_release_confirmation_before_npc_attach() -> None:
    simulator = FakeSimulator()
    bridge = RobotToNpcHandoverBridge(simulator)

    handover = bridge.accept_released_object(
        "stretch_3",
        "employee_01",
        "parcel",
        robot_release_confirmed=True,
        session_id="robot_handover_1",
    )
    simulator.receipts.append(
        NpcCommandReceipt(handover.command_id, "employee_01", CommandStatus.SUCCEEDED)
    )

    assert bridge.poll("robot_handover_1").status.value == "succeeded"
