from pathlib import Path

from stretch_mujoco.agents import (
    ActionCommand,
    ActionType,
    MockRobotExecutor,
    OfficeAgentRuntime,
    ReservationManager,
    RobotTaskStatus,
)
from stretch_mujoco.agents.drivers import MujocoNpcActionDriver
from stretch_mujoco.agents.conversation import SpatialPose
from stretch_mujoco.npc import CommandStatus, NpcCommandReceipt
from stretch_mujoco.semantics import RelationType, SemanticWorld

MODELS_PATH = Path(__file__).resolve().parents[1] / "stretch_mujoco" / "models"


def load_runtime(*, auto_plan: bool = False) -> OfficeAgentRuntime:
    world = SemanticWorld.from_json(MODELS_PATH / "office_semantics.json")
    return OfficeAgentRuntime.from_json(
        world,
        MODELS_PATH / "office_agents.json",
        auto_plan=auto_plan,
    )


def test_action_protocol_is_closed_and_rejects_unknown_actions() -> None:
    try:
        ActionCommand.from_dict({"agent_id": "employee_01", "action": "invent_new_action"})
    except ValueError as error:
        assert "is not allowed" in str(error)
    else:
        raise AssertionError("Unknown action was accepted")


def test_population_runtime_registers_generated_npc_semantics() -> None:
    world = SemanticWorld.from_json(MODELS_PATH / "office_semantics.json")
    runtime = OfficeAgentRuntime.from_json(
        world,
        MODELS_PATH / "office_population.production.example.json",
        auto_plan=False,
    )

    assert runtime.population_npc_ids == frozenset(runtime.agents)
    alex = world.object("npc_alex_chen")
    assert alex.object_type.value == "Employee"
    assert alex.binding.name == "npc__npc_alex_chen"
    assert world.interaction_points["npc_alex_chen_handover"].site == "npc__npc_alex_chen__handover"
    assert set(runtime.population_interaction_templates) == {"conversation", "handover"}


def test_action_validation_checks_location_and_target() -> None:
    runtime = load_runtime()

    sit_result = runtime.submit_action(ActionCommand("employee_01", ActionType.SIT, "chair_right"))
    missing_result = runtime.submit_action(
        ActionCommand("employee_01", ActionType.MOVE_TO, "missing_room")
    )

    assert not sit_result.valid
    assert "not at required location" in sit_result.errors[0]
    assert not missing_result.valid
    assert "does not exist" in missing_result.errors[0]


def test_work_session_inserts_sit_and_stand_using_the_workstation_chair_contract() -> None:
    runtime = load_runtime()
    agent = runtime.agents["employee_01"]

    result = runtime.submit_action(
        ActionCommand("employee_01", ActionType.WORK, "workstation_right")
    )

    assert result.valid
    assert agent.executor.command is not None
    assert agent.executor.command.action == ActionType.MOVE_TO
    assert agent.executor.command.target == "chair_right"
    session_id = agent.executor.command.parameters["_desk_work_session_id"]
    assert [command.action for command in agent.planner.action_queue] == [
        ActionType.SIT,
        ActionType.WORK,
        ActionType.STAND_UP,
        ActionType.IDLE,
    ]
    assert agent.planner.action_queue[1].parameters["_desk_work_seat"] == "chair_right"
    assert all(
        command.parameters["_desk_work_session_id"] == session_id
        for command in agent.planner.action_queue
    )

    runtime.tick(3.0)
    assert agent.executor.command is not None
    assert agent.executor.command.action == ActionType.SIT
    runtime.tick(0.5)
    assert agent.state.location == "chair_right"
    assert runtime.world.find_relations(
        subject="chair_right", relation=RelationType.OCCUPIED_BY, object_id="employee_01"
    )
    assert agent.executor.command is not None
    assert agent.executor.command.action == ActionType.WORK

    runtime.tick(15.0)
    assert agent.executor.command is not None
    assert agent.executor.command.action == ActionType.STAND_UP
    runtime.tick(0.5)
    assert not runtime.world.find_relations(
        subject="chair_right", relation=RelationType.OCCUPIED_BY, object_id="employee_01"
    )
    assert agent.executor.command is not None
    assert agent.executor.command.action == ActionType.IDLE


def test_use_computer_always_expands_to_the_workstation_chair_session() -> None:
    runtime = load_runtime()
    agent = runtime.agents["employee_01"]

    result = runtime.submit_action(
        ActionCommand("employee_01", ActionType.USE_COMPUTER, "computer_right")
    )

    assert result.valid
    assert agent.executor.command is not None
    assert agent.executor.command.action == ActionType.MOVE_TO
    assert agent.executor.command.target == "chair_right"
    assert [command.action for command in agent.planner.action_queue] == [
        ActionType.SIT,
        ActionType.WORK,
        ActionType.STAND_UP,
        ActionType.IDLE,
    ]


def test_work_request_duration_is_applied_only_after_runtime_session_expansion() -> None:
    runtime = load_runtime()
    agent = runtime.agents["employee_01"]

    result = runtime.submit_action(
        ActionCommand(
            "employee_01",
            ActionType.USE_COMPUTER,
            "computer_right",
            {"duration_seconds": 1.25},
        )
    )

    assert result.valid
    work = next(
        command for command in agent.planner.action_queue if command.action == ActionType.WORK
    )
    assert work.parameters["_desk_work_duration_seconds"] == 1.25
    assert "_desk_work_session_id" in work.parameters


def test_work_rejects_an_invalid_requested_duration() -> None:
    runtime = load_runtime()

    result = runtime.submit_action(
        ActionCommand("employee_01", ActionType.WORK, "workstation_right", {"duration_seconds": 0})
    )

    assert not result.valid
    assert result.errors == ("duration_seconds must be a positive finite number",)


def test_work_rejects_scene_without_one_unambiguous_workstation_chair_relation() -> None:
    runtime = load_runtime()
    runtime.world.remove_relation("chair_right", RelationType.NEAR, "workstation_right")

    missing = runtime.submit_action(
        ActionCommand("employee_01", ActionType.WORK, "workstation_right")
    )

    assert not missing.valid
    assert "requires exactly one Chair --NEAR--> Workstation relation; found 0" in missing.errors[0]

    runtime = load_runtime()
    runtime.world.add_relation("chair_left", RelationType.NEAR, "workstation_right")
    ambiguous = runtime.submit_action(
        ActionCommand("employee_01", ActionType.WORK, "workstation_right")
    )

    assert not ambiguous.valid
    assert (
        "requires exactly one Chair --NEAR--> Workstation relation; found 2" in ambiguous.errors[0]
    )


def test_reservations_reject_conflicting_agents() -> None:
    reservations = ReservationManager()

    assert reservations.reserve("soda_can", "employee_01")
    assert not reservations.reserve("soda_can", "employee_02")
    assert reservations.owner("soda_can") == "employee_01"


def test_robot_request_creates_verified_task_and_updates_world() -> None:
    runtime = load_runtime()
    result = runtime.submit_action(
        {
            "agent_id": "employee_01",
            "action": "request_robot",
            "target": "stretch",
            "parameters": {
                "task": "deliver",
                "object": "document_report",
                "destination": "workstation_right",
            },
        }
    )

    runtime.tick(1.0)
    task = runtime.pending_robot_tasks()[0]

    assert result.valid
    assert task.status == RobotTaskStatus.PENDING
    assert task.robot_id == "stretch_3"
    assert runtime.reservations.owner("document_report") == "employee_01"
    assert runtime.world.find_relations(
        subject="document_report",
        relation=RelationType.REQUESTED_BY,
        object_id="employee_01",
    )

    runtime.complete_robot_task(task.task_id, success=True)

    assert task.status == RobotTaskStatus.SUCCEEDED
    assert runtime.world.location_of("document_report").object_id == "workstation_right"
    assert runtime.reservations.owner("document_report") is None


def test_robot_request_with_unknown_conversation_leaves_no_task_or_reservation() -> None:
    runtime = load_runtime()
    result = runtime.submit_action(
        ActionCommand(
            "employee_01",
            ActionType.REQUEST_ROBOT,
            "stretch_3",
            {
                "task": "deliver",
                "object": "document_report",
                "destination": "workstation_right",
                "conversation_id": "missing_conversation",
            },
        )
    )

    assert not result.valid
    assert result.errors == ("Robot task has an invalid conversation association",)
    assert runtime.pending_robot_tasks() == ()
    assert runtime.reservations.owner("document_report") is None


def test_embodied_robot_request_creates_no_task_until_speech_receipt() -> None:
    class Simulator:
        def __init__(self) -> None:
            self.time = 0.0
            self.commands = []
            self.receipts = []

        def pull_status(self):
            return type("Status", (), {"time": self.time})()

        def submit_npc_command(self, command):
            self.commands.append(command)
            return command.command_id

        def pull_npc_receipts(self):
            value = tuple(self.receipts)
            self.receipts.clear()
            return value

        def cancel_npc_command(self, _npc_id, _command_id):
            return None

    simulator = Simulator()
    world = SemanticWorld.from_json(MODELS_PATH / "office_semantics.json")
    driver = MujocoNpcActionDriver(
        simulator,
        {},
        robot_request_sites={"stretch_3": "stretch_request_stand_site"},
        available_clips={"talk"},
    )
    runtime = OfficeAgentRuntime.from_json(
        world, MODELS_PATH / "office_agents.json", auto_plan=False, action_driver=driver
    )

    assert runtime.submit_action(
        ActionCommand(
            "employee_01",
            ActionType.REQUEST_ROBOT,
            "stretch_3",
            {"task": "deliver", "object": "document_report", "destination": "workstation_right"},
        )
    ).valid
    assert not runtime.robot_tasks

    simulator.receipts.append(
        NpcCommandReceipt(simulator.commands[-1].command_id, "employee_01", CommandStatus.SUCCEEDED)
    )
    runtime.tick(0.25)
    assert not runtime.robot_tasks

    simulator.receipts.append(
        NpcCommandReceipt(simulator.commands[-1].command_id, "employee_01", CommandStatus.SUCCEEDED)
    )
    events = runtime.tick(0.25)

    task = next(iter(runtime.robot_tasks.values()))
    assert task.request_receipt_id is not None
    assert any(event.event == "robot_request_accepted" for event in events)
    assert not any(event.event == "robot_delivery_verified" for event in events)
    completion_events = runtime.complete_robot_task(task.task_id, success=True)
    assert completion_events.status is RobotTaskStatus.SUCCEEDED
    assert any(event.event == "robot_delivery_verified" for event in runtime.events)


def test_embodied_runtime_rejects_the_legacy_remote_robot_request_helper() -> None:
    class Simulator:
        def pull_status(self):
            return type("Status", (), {"time": 0.0})()

        def submit_npc_command(self, command):
            return command.command_id

        def pull_npc_receipts(self):
            return ()

        def cancel_npc_command(self, _npc_id, _command_id):
            return None

    world = SemanticWorld.from_json(MODELS_PATH / "office_semantics.json")
    runtime = OfficeAgentRuntime.from_json(
        world,
        MODELS_PATH / "office_agents.json",
        auto_plan=False,
        action_driver=MujocoNpcActionDriver(
            Simulator(),
            {},
            robot_request_sites={"stretch_3": "stretch_request_stand_site"},
            available_clips={"talk"},
        ),
    )
    session = runtime.conversations.start(
        ("employee_01", "stretch_3"),
        "delivery",
        0.0,
        {
            "employee_01": SpatialPose((0.0, 0.0, 0.0), 0.0),
            "stretch_3": SpatialPose((0.0, 0.8, 0.0), 3.141592653589793),
        },
        require_spatial_ready=False,
    )

    assert not runtime.request_robot_task(
        session.session_id,
        "employee_01",
        task="deliver",
        object_id="document_report",
        destination="workstation_right",
    ).valid
    assert not runtime.robot_tasks


def test_mock_robot_completes_a_request_without_bypassing_runtime_state() -> None:
    runtime = load_runtime()
    result = runtime.submit_action(
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
    mock_robot = MockRobotExecutor(completion_delay_minutes=2.0)

    assert result.valid
    assert mock_robot.tick(runtime, 1.0) == ()
    assert task.status == RobotTaskStatus.RUNNING
    assert mock_robot.tick(runtime, 1.0) == (task.task_id,)
    assert task.status == RobotTaskStatus.SUCCEEDED
    assert task.receipt_ids == {f"mock:{task.task_id}:delivery_verified"}
    assert runtime.world.location_of("document_report").object_id == "workstation_right"


def test_mock_robot_failure_releases_the_reserved_object() -> None:
    runtime = load_runtime()
    runtime.submit_action(
        ActionCommand(
            "employee_01",
            ActionType.REQUEST_ROBOT,
            "stretch_3",
            {"task": "deliver", "object": "document_report", "destination": "workstation_right"},
        )
    )
    runtime.tick(1.0)
    task = runtime.pending_robot_tasks()[0]

    MockRobotExecutor(0.1, {task.task_id}).tick(runtime, 0.1)

    assert task.status == RobotTaskStatus.FAILED
    assert runtime.reservations.owner("document_report") is None
    assert "Mock robot reported" in task.error


def test_failed_action_records_recovery_state() -> None:
    runtime = load_runtime()
    agent = runtime.agents["employee_01"]

    result = runtime.submit_action(ActionCommand("employee_01", ActionType.SIT, "chair_right"))

    assert not result.valid
    assert agent.state.availability == "available"
    assert agent.state.animation_state == "idle"
    assert agent.state.blocked_reason is not None
    assert "not at required location" in agent.state.last_failure


def test_robot_result_can_be_verified_against_physical_snapshot() -> None:
    runtime = load_runtime()
    runtime.submit_action(
        {
            "agent_id": "employee_01",
            "action": "request_robot",
            "target": "stretch",
            "parameters": {
                "task": "deliver",
                "object": "soda_can",
                "destination": "workstation_right",
            },
        }
    )
    runtime.tick(1.0)
    task = runtime.pending_robot_tasks()[0]
    far_snapshot = {
        "objects": {
            "soda_can": {"position": [0.0, -1.25, 0.8]},
            "workstation_right": {"position": [2.25, 1.25, 0.0]},
        }
    }

    runtime.complete_robot_task(
        task.task_id,
        success=True,
        semantic_snapshot=far_snapshot,
    )

    assert task.status == RobotTaskStatus.FAILED
    assert "from destination" in task.error


def test_permission_and_graspability_are_checked_before_robot_request() -> None:
    runtime = load_runtime()
    runtime.world.remove_relation("document_report", RelationType.ALLOWED_FOR, "employee_01")

    denied = runtime.submit_action(
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
    not_graspable = runtime.submit_action(
        {
            "agent_id": "employee_01",
            "action": "request_robot",
            "target": "stretch_3",
            "parameters": {
                "task": "deliver",
                "object": "coffee_machine",
                "destination": "workstation_right",
            },
        }
    )

    assert not denied.valid
    assert any("not allowed" in error for error in denied.errors)
    assert not not_graspable.valid
    assert any("not graspable" in error for error in not_graspable.errors)


def test_move_pick_up_and_drink_update_agent_needs_and_memory() -> None:
    runtime = load_runtime()
    agent = runtime.agents["employee_01"]

    assert runtime.submit_action(
        ActionCommand("employee_01", ActionType.MOVE_TO, "snack_counter")
    ).valid
    runtime.tick(3.0)
    assert agent.state.location == "snack_counter"

    assert runtime.submit_action(ActionCommand("employee_01", ActionType.PICK_UP, "soda_can")).valid
    runtime.tick(0.5)
    assert agent.state.held_object == "soda_can"

    agent.needs.thirst = 0.9
    assert runtime.submit_action(ActionCommand("employee_01", ActionType.DRINK, "soda_can")).valid
    runtime.tick(2.0)

    assert agent.state.held_object is None
    assert agent.needs.thirst < 0.4
    assert runtime.world.object("soda_can").attributes["consumed"] is True
    assert runtime.world.object("soda_can").attributes["available"] is False
    assert agent.memory.entries[-1].event == "action_succeeded"


def test_action_success_and_semantic_commit_share_one_event_id() -> None:
    runtime = load_runtime()

    assert runtime.submit_action(
        ActionCommand("employee_01", ActionType.MOVE_TO, "meeting_table")
    ).valid
    events = runtime.tick(3.0)

    succeeded = [event for event in events if event.event == "action_succeeded"][-1]
    committed = [event for event in events if event.event == "semantic_commit"][-1]
    memory = runtime.agents["employee_01"].memory.entries[-1]
    assert succeeded.event_id
    assert succeeded.event_id == committed.event_id == memory.details["event_id"]
    assert succeeded.details["execution_id"] == committed.details["execution_id"]


def test_utility_event_uses_runtime_owned_constraints() -> None:
    runtime = load_runtime(auto_plan=True)
    agent = runtime.agents["employee_01"]
    runtime._consecutive_failures[agent.agent_id] = 2
    assert runtime.reservations.reserve("soda_can", "employee_02")

    runtime._run_utility_decisions()

    event = [event for event in runtime.events if event.event == "plan_selected"][-1]
    context = event.details["utility_context"]
    assert context["consecutive_failures"] == 2
    assert context["resource_availability"] == 0.0


def test_schedule_has_deterministic_daily_variation() -> None:
    runtime = load_runtime()
    item = runtime.agents["employee_01"].schedule.items[0]

    windows = {item.shifted_window("employee_01", day, runtime.seed) for day in range(5)}

    assert len(windows) > 1
    assert item.shifted_window("employee_01", 2, runtime.seed) == item.shifted_window(
        "employee_01", 2, runtime.seed
    )


def test_auto_planner_does_not_duplicate_an_active_robot_request() -> None:
    runtime = load_runtime(auto_plan=True)
    runtime.minute_of_day = 13 * 60
    runtime.agents["employee_01"].needs.thirst = 0.9

    runtime.tick(1.0)
    runtime.tick(1.0)
    runtime.tick(10.0)

    assert len(runtime.robot_tasks) == 1
    assert len(runtime.pending_robot_tasks()) == 1
