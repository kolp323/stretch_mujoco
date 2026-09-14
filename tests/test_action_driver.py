import math
import json
from pathlib import Path
from types import SimpleNamespace

import mujoco
import pytest

from stretch_mujoco.agents import (
    ActionCommand,
    ActionExecution,
    ActionType,
    ExecutionStatus,
    OfficeAgentRuntime,
    EmployeeAgent,
    MockRobotExecutor,
    RobotTask,
    RobotToNpcHandoverBridge,
)
from stretch_mujoco.agents.drivers import MujocoNpcActionDriver
from stretch_mujoco.agents.action_recipes import ACTION_RECIPES
from stretch_mujoco.agents.simulation_bridge import create_mujoco_action_driver
from stretch_mujoco.npc import CommandStatus, NpcCommand, NpcCommandKind, NpcCommandReceipt
from stretch_mujoco.npc.system import NpcSystem
from stretch_mujoco.npc.locomotion import yaw_quaternion
from stretch_mujoco.npc.trajectory_profile import NpcTrajectoryProfile
from stretch_mujoco.semantics import ObjectType, SemanticObject, SemanticWorld

MODELS = Path(__file__).resolve().parents[1] / "stretch_mujoco" / "models"
OFFICE_PROFILE = (
    Path(__file__).resolve().parents[1] / "stretch_mujoco/npc/trajectory_profiles/office_v1.json"
)
PRODUCTION_POPULATION = MODELS / "office_population.production.example.json"


class FakeSimulator:
    def __init__(self) -> None:
        self.command = None
        self.commands = []
        self.receipts = []
        self.time = 10.0
        self.cancelled = []

    def pull_status(self):
        return SimpleNamespace(time=self.time)

    def submit_npc_command(self, command):
        self.command = command
        self.commands.append(command)
        return command.command_id

    def pull_npc_receipts(self):
        receipts = tuple(self.receipts)
        self.receipts.clear()
        return receipts

    def cancel_npc_command(self, npc_id, command_id):
        self.cancelled.append((npc_id, command_id))


class NpcSystemSimulator:
    """Minimal in-process transport used to exercise the real NPC controller."""

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, system: NpcSystem) -> None:
        self.model = model
        self.data = data
        self.system = system
        self.time = 0.0

    def pull_status(self):
        return SimpleNamespace(time=self.time)

    def submit_npc_command(self, command):
        self.system.submit(command)
        return command.command_id

    def pull_npc_receipts(self):
        return self.system.drain_receipts()

    def cancel_npc_command(self, npc_id, command_id):
        sequence = 100 + sum(1 for _ in self.system.drain_receipts())
        self.system.submit(
            NpcCommand(
                f"cancel:{command_id}",
                sequence,
                npc_id,
                NpcCommandKind.CANCEL,
                {"command_id": command_id},
                self.time,
            )
        )

    def step(self, seconds: float) -> None:
        self.time += seconds
        self.system.step(self.model, self.data, self.time)


def _handover_model() -> mujoco.MjModel:
    def frames(npc_id: str, clip: str, count: int = 8) -> str:
        return "\n".join(
            f'<geom name="npc__{npc_id}__clip__{clip}__frame__{index:03d}__slot__body" '
            f'type="sphere" size=".08" rgba="1 1 1 0"/>'
            for index in range(count)
        )

    return mujoco.MjModel.from_xml_string(
        f"""
        <mujoco>
          <worldbody>
            <geom type="plane" size="5 5 .1"/>
            <body name="npc__employee_01" mocap="true">
              {frames("employee_01", "idle", 1)}
              {frames("employee_01", "walk")}
              {frames("employee_01", "give")}
              <site name="npc__employee_01__handover" pos="0 0 1"/>
            </body>
            <body name="npc__employee_02" mocap="true" pos="0 1 0">
              {frames("employee_02", "idle", 1)}
              {frames("employee_02", "walk")}
              {frames("employee_02", "receive")}
              <site name="npc__employee_02__handover" pos="0 0 1"/>
            </body>
            <site name="giver_stand" pos="-.35 0 0"/>
            <site name="receiver_stand" pos=".35 0 0"/>
            <body name="parcel" pos="0 .2 .2"><freejoint/><geom type="box" size=".04 .04 .04"/></body>
          </worldbody>
        </mujoco>
        """
    )


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

    assert ACTION_RECIPES[ActionType.SIT].animation == "sit"

    result = runtime.submit_action(ActionCommand("employee_01", ActionType.SIT, "chair_right"))
    assert result.valid
    assert simulator.command.kind.value == "move_to"
    assert simulator.command.payload == {
        "site": "chair_right_sit",
        "arrival_clip": "idle",
        "max_replans": 1,
    }
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

    result = runtime.submit_action(ActionCommand("employee_01", ActionType.STAND_UP, "chair_right"))
    assert result.valid
    assert simulator.command.kind.value == "play_animation"
    assert simulator.command.payload == {
        "clip": "stand_up",
        "completion_marker": "standing",
        "arrival_clip": "idle",
        "target_site": "chair_right_sit",
    }

    simulator.receipts.append(
        NpcCommandReceipt(simulator.command.command_id, "employee_01", CommandStatus.SUCCEEDED)
    )
    runtime.tick(0.25)

    assert agent.executor.status == ExecutionStatus.SUCCEEDED
    assert not world.find_relations(
        subject="chair_right", relation="OCCUPIED_BY", object_id="employee_01"
    )


def test_move_to_chair_uses_the_chair_ingress_not_the_sit_center() -> None:
    simulator = FakeSimulator()
    driver = MujocoNpcActionDriver(
        simulator,
        {"chair_right": "chair_right_sit"},
        seat_navigation_sites={"chair_right": "chair_right_approach_site"},
    )
    execution = ActionExecution(
        command=ActionCommand("employee_01", ActionType.MOVE_TO, "chair_right"),
        status=ExecutionStatus.RUNNING,
    )

    result = driver.start(execution)

    assert result.phase == "navigate"
    assert simulator.command.kind == NpcCommandKind.MOVE_TO
    assert simulator.command.payload["site"] == "chair_right_approach_site"


def test_robot_request_requires_npc_approach_and_spoken_receipt_before_success() -> None:
    simulator = FakeSimulator()
    driver = MujocoNpcActionDriver(
        simulator,
        {},
        robot_request_sites={"stretch_3": "stretch_request_stand_site"},
        available_clips={"talk"},
    )
    execution = ActionExecution(
        execution_id="robot_request_1",
        command=ActionCommand(
            "employee_01",
            ActionType.REQUEST_ROBOT,
            "stretch_3",
            {"task": "deliver", "object": "parcel", "destination": "desk"},
        ),
        status=ExecutionStatus.RUNNING,
    )

    assert driver.start(execution).phase == "approach_stretch"
    assert simulator.command.kind is NpcCommandKind.MOVE_TO
    assert simulator.command.payload["site"] == "stretch_request_stand_site"

    simulator.receipts.append(
        NpcCommandReceipt(simulator.command.command_id, "employee_01", CommandStatus.SUCCEEDED)
    )
    assert driver.poll(execution).phase == "speak_request"
    assert simulator.command.kind is NpcCommandKind.PLAY_ANIMATION
    assert simulator.command.payload == {
        "clip": "talk",
        "completion_marker": "talk_cycle",
        "arrival_clip": "idle",
        "target_site": "stretch_request_stand_site",
        "gaze_target": "stretch_3",
        "interaction_target": "stretch_3",
        "interaction_distance_min": 0.45,
        "interaction_distance_max": 1.45,
        "interaction_yaw_tolerance": 1.20,
    }

    simulator.receipts.append(
        NpcCommandReceipt(simulator.command.command_id, "employee_01", CommandStatus.SUCCEEDED)
    )
    result = driver.poll(execution)

    assert result.status is ExecutionStatus.SUCCEEDED
    assert result.phase == "request_accepted"


def test_npc_handover_waits_for_ready_release_and_receive_receipts() -> None:
    simulator = FakeSimulator()
    driver = MujocoNpcActionDriver(
        simulator,
        {},
        handover_sites={"employee_02": "employee_02_handover_site"},
        handover_role_sites={("employee_01", "employee_02"): ("giver_stand", "receiver_stand")},
        interaction_yaws={"employee_01": math.pi / 2, "employee_02": -math.pi / 2},
    )
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
    assert result.phase == "rendezvous"
    assert [command.kind.value for command in simulator.commands] == ["move_to", "move_to"]
    assert simulator.commands[0].payload == {
        "site": "giver_stand",
        "arrival_clip": "idle",
        "max_replans": 1,
    }
    assert simulator.commands[1].payload["site"] == "receiver_stand"
    for expected_phase, expected_count in (
        ("aligned", 2),
        ("giver_ready", 2),
        ("release", 2),
        ("receive", 1),
    ):
        for command in simulator.commands[-expected_count:]:
            simulator.receipts.append(
                NpcCommandReceipt(command.command_id, command.npc_id, CommandStatus.SUCCEEDED)
            )
        result = driver.poll(execution)
        assert result.phase == expected_phase

    simulator.receipts.append(
        NpcCommandReceipt(
            simulator.command.command_id, simulator.command.npc_id, CommandStatus.SUCCEEDED
        )
    )
    result = driver.poll(execution)

    assert result.status == ExecutionStatus.SUCCEEDED
    assert [command.kind.value for command in simulator.commands] == [
        "move_to",
        "move_to",
        "align_to",
        "align_to",
        "play_animation",
        "play_animation",
        "detach_object",
        "attach_object",
    ]
    assert simulator.commands[2].payload == {"yaw": math.pi / 2, "target_site": "giver_stand"}
    assert simulator.commands[4].payload["completion_marker"] == "handover_ready"
    assert simulator.commands[4].payload["arrival_clip"] == "idle"
    assert simulator.commands[4].payload["target_site"] == "giver_stand"
    assert simulator.commands[4].payload["reveal_object"] == "parcel"
    assert simulator.commands[5].payload["target_site"] == "receiver_stand"
    # A handover-ready marker is meaningful only while both physical actors
    # are at their role stations, within reach, and facing one another.
    assert simulator.commands[4].payload["interaction_target"] == "employee_02"
    assert simulator.commands[5].payload["interaction_target"] == "employee_01"
    for command in simulator.commands[4:6]:
        assert command.payload["interaction_distance_min"] == 0.45
        assert command.payload["interaction_distance_max"] == 0.95
        assert command.payload["interaction_yaw_tolerance"] == 0.30
        assert command.payload["position_tolerance"] == 0.12
    assert simulator.commands[6].payload["interaction_id"] == "handover_1"
    assert simulator.commands[7].payload["interaction_id"] == "handover_1"
    assert driver.interactions.sessions["handover_1"].completed_phases == [
        "rendezvous",
        "aligned",
        "giver_ready",
        "receiver_ready",
        "released",
        "received",
    ]


def test_production_handover_roles_use_close_opposing_stations() -> None:
    population = json.loads(PRODUCTION_POPULATION.read_text(encoding="utf-8"))
    handover = population["interaction_templates"]["handover"]

    assert handover["giver"] == {
        "site": "npc_handover_giver_stand_site",
        "yaw": 1.5708,
    }
    assert handover["receiver"] == {
        "site": "npc_handover_receiver_stand_site",
        "yaw": -1.5708,
    }


def test_handover_role_sites_are_leased_until_cancelled() -> None:
    simulator = FakeSimulator()
    driver = MujocoNpcActionDriver(
        simulator,
        {},
        handover_sites={
            "employee_02": "employee_02_handover_site",
            "employee_04": "employee_04_handover_site",
        },
        handover_role_sites={
            ("employee_01", "employee_02"): ("giver_stand", "receiver_stand"),
            ("employee_03", "employee_04"): ("giver_stand", "receiver_stand"),
        },
        interaction_yaws={
            "employee_01": 0.0,
            "employee_02": math.pi,
            "employee_03": 0.0,
            "employee_04": math.pi,
        },
    )
    first = ActionExecution(
        execution_id="lease_first",
        command=ActionCommand(
            "employee_01", ActionType.HANDOVER, "employee_02", {"object": "parcel"}
        ),
        status=ExecutionStatus.RUNNING,
    )
    second = ActionExecution(
        execution_id="lease_second",
        command=ActionCommand(
            "employee_03", ActionType.HANDOVER, "employee_04", {"object": "parcel_2"}
        ),
        status=ExecutionStatus.RUNNING,
    )
    assert driver.start(first).status is ExecutionStatus.RUNNING
    assert driver.start(second).error == "interaction_sites_busy"
    driver.cancel(first, "test")
    assert driver.start(second).status is ExecutionStatus.RUNNING


def test_handover_contract_completes_on_real_npc_controllers_and_returns_to_idle() -> None:
    model = _handover_model()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    system = NpcSystem.from_model(model)
    simulator = NpcSystemSimulator(model, data, system)

    # A preceding pickup owns the free object before the handover starts.
    system.submit(
        NpcCommand(
            "preheld", 0, "employee_01", NpcCommandKind.ATTACH_OBJECT, {"object": "parcel"}, 0.0
        )
    )
    system.step(model, data, 0.0)
    system.drain_receipts()
    driver = MujocoNpcActionDriver(
        simulator,
        {},
        handover_sites={"employee_02": "npc__employee_02__handover"},
        handover_role_sites={("employee_01", "employee_02"): ("giver_stand", "receiver_stand")},
        interaction_yaws={"employee_01": math.pi / 2, "employee_02": -math.pi / 2},
        available_clips={"give", "receive"},
    )
    # Preserve the monotonic sequence established by the preceding physical pickup.
    driver._sequences["employee_01"] = 0
    execution = ActionExecution(
        execution_id="real_handover",
        command=ActionCommand(
            "employee_01", ActionType.HANDOVER, "employee_02", {"object": "parcel"}
        ),
        status=ExecutionStatus.RUNNING,
    )

    result = driver.start(execution)
    for _ in range(360):
        simulator.step(0.05)
        result = driver.poll(execution)
        if result.status != ExecutionStatus.RUNNING:
            break

    assert result.status == ExecutionStatus.SUCCEEDED
    assert system.controllers["employee_01"].attachments.held_objects == ()
    assert system.controllers["employee_02"].attachments.held_objects == ("parcel",)
    for _ in range(2):
        simulator.step(0.05)
    states = system.states(data)
    assert states["employee_01"].resolved_clip == "idle"
    assert states["employee_02"].resolved_clip == "idle"


def test_real_handover_gate_blocks_far_or_misaligned_participants_before_transfer() -> None:
    model = _handover_model()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    system = NpcSystem.from_model(model)
    system.submit(
        NpcCommand(
            "preheld", 0, "employee_01", NpcCommandKind.ATTACH_OBJECT, {"object": "parcel"}, 0.0
        )
    )
    system.step(model, data, 0.0)
    system.drain_receipts()
    gate = {
        "interaction_distance_min": 0.45,
        "interaction_distance_max": 0.95,
        "interaction_yaw_tolerance": 0.30,
    }
    system.submit(
        NpcCommand(
            "give", 1, "employee_01", NpcCommandKind.PLAY_ANIMATION,
            {"clip": "give", "completion_marker": "handover_ready", "interaction_target": "employee_02", **gate},
            0.0,
        )
    )
    system.submit(
        NpcCommand(
            "receive", 0, "employee_02", NpcCommandKind.PLAY_ANIMATION,
            {"clip": "receive", "completion_marker": "handover_ready", "interaction_target": "employee_01", **gate},
            0.0,
        )
    )
    for step in range(1, 5):
        system.step(model, data, step * 0.1)
    states = system.states(data)
    assert states["employee_01"].resolved_clip == "idle"
    assert states["employee_02"].resolved_clip == "idle"
    assert states["employee_01"].last_marker != "handover_ready"
    assert states["employee_02"].last_marker != "handover_ready"
    assert states["employee_01"].held_objects == ("parcel",)
    assert states["employee_02"].held_objects == ()

    first_mocap = model.body("npc__employee_01").mocapid[0]
    second_mocap = model.body("npc__employee_02").mocapid[0]
    data.mocap_pos[first_mocap] = (-0.35, 0.0, 0.0)
    data.mocap_pos[second_mocap] = (0.35, 0.0, 0.0)
    data.mocap_quat[first_mocap] = yaw_quaternion(0.0)
    data.mocap_quat[second_mocap] = yaw_quaternion(0.0)
    mujoco.mj_forward(model, data)
    for step in range(5, 9):
        system.step(model, data, step * 0.1)
    states = system.states(data)
    assert states["employee_01"].resolved_clip == "idle"
    assert states["employee_02"].resolved_clip == "idle"
    assert states["employee_01"].last_marker != "handover_ready"
    assert states["employee_02"].last_marker != "handover_ready"
    assert states["employee_01"].held_objects == ("parcel",)
    assert states["employee_02"].held_objects == ()


def test_pick_up_waits_for_grasp_marker_before_attachment() -> None:
    simulator = FakeSimulator()
    driver = MujocoNpcActionDriver(
        simulator,
        {},
        object_approach_sites={"parcel": "parcel_human_stand_site"},
        interaction_yaws={"parcel": math.pi},
    )
    execution = ActionExecution(
        execution_id="pick_1",
        command=ActionCommand("employee_01", ActionType.PICK_UP, "parcel"),
        status=ExecutionStatus.RUNNING,
    )

    assert driver.start(execution).phase == "approach"
    assert simulator.command.kind.value == "move_to"
    assert simulator.command.payload == {
        "site": "parcel_human_stand_site",
        "arrival_clip": "idle",
        "max_replans": 1,
    }
    simulator.receipts.append(
        NpcCommandReceipt(simulator.command.command_id, "employee_01", CommandStatus.SUCCEEDED)
    )

    assert driver.poll(execution).phase == "align"
    assert simulator.command.kind.value == "align_to"
    assert simulator.command.payload == {
        "yaw": math.pi,
        "target_site": "parcel_human_stand_site",
    }
    simulator.receipts.append(
        NpcCommandReceipt(simulator.command.command_id, "employee_01", CommandStatus.SUCCEEDED)
    )

    assert driver.poll(execution).phase == "grasp"
    assert simulator.command.kind.value == "play_animation"
    assert simulator.command.payload == {
        "clip": "pick_up",
        "completion_marker": "grasp",
        "arrival_clip": "idle",
        "target_site": "parcel_human_stand_site",
    }
    simulator.receipts.append(
        NpcCommandReceipt(simulator.command.command_id, "employee_01", CommandStatus.SUCCEEDED)
    )
    assert driver.poll(execution).phase == "attach"
    assert simulator.command.kind.value == "attach_object"
    assert simulator.command.payload == {"object": "parcel", "visible": False}


def test_object_and_handover_actions_reject_missing_approach_sites() -> None:
    simulator = FakeSimulator()
    driver = MujocoNpcActionDriver(simulator, {})

    pick = ActionExecution(
        execution_id="pick_missing_site",
        command=ActionCommand("employee_01", ActionType.PICK_UP, "parcel"),
        status=ExecutionStatus.RUNNING,
    )
    handover = ActionExecution(
        execution_id="handover_missing_site",
        command=ActionCommand(
            "employee_01",
            ActionType.HANDOVER,
            "employee_02",
            {"object": "parcel"},
        ),
        status=ExecutionStatus.RUNNING,
    )

    assert driver.start(pick).error == "Missing object approach or placement site"
    assert driver.start(handover).error == "recipe_missing_site"


def test_handover_session_timeout_cancels_current_physical_stage() -> None:
    simulator = FakeSimulator()
    driver = MujocoNpcActionDriver(
        simulator,
        {},
        handover_sites={"employee_02": "employee_02_handover_site"},
        handover_role_sites={("employee_01", "employee_02"): ("giver_stand", "receiver_stand")},
        interaction_yaws={"employee_01": math.pi / 2, "employee_02": -math.pi / 2},
    )
    execution = ActionExecution(
        execution_id="handover_timeout",
        command=ActionCommand(
            "employee_01",
            ActionType.HANDOVER,
            "employee_02",
            {"object": "parcel"},
        ),
        status=ExecutionStatus.RUNNING,
    )

    driver.start(execution)
    assert {command.npc_id for command in simulator.commands} == {"employee_01", "employee_02"}
    simulator.time = 56.0
    result = driver.poll(execution)

    assert result.status == ExecutionStatus.TIMED_OUT
    assert result.error == "deadline_exceeded"
    assert simulator.cancelled == [
        ("employee_01", "handover_timeout:rendezvous_employee_01"),
        ("employee_02", "handover_timeout:rendezvous_employee_02"),
    ]


def test_production_handover_uses_population_npc_ids_and_generic_role_sites() -> None:
    simulator = FakeSimulator()
    driver = create_mujoco_action_driver(simulator, npc_ids={"npc_alex_chen", "npc_morgan_lee"})
    execution = ActionExecution(
        execution_id="production_handover",
        command=ActionCommand(
            "npc_alex_chen", ActionType.HANDOVER, "npc_morgan_lee", {"object": "parcel"}
        ),
        status=ExecutionStatus.RUNNING,
    )

    result = driver.start(execution)

    assert result.phase == "rendezvous"
    assert {command.npc_id for command in simulator.commands} == {
        "npc_alex_chen",
        "npc_morgan_lee",
    }
    assert {str(command.payload["site"]) for command in simulator.commands} == {
        "npc_handover_giver_stand_site",
        "npc_handover_receiver_stand_site",
    }
    assert driver.cue_sites == {
        "npc_alex_chen": "npc_handover_giver_stand_site",
        "npc_morgan_lee": "npc_handover_giver_stand_site",
    }
    assert driver.interaction_yaws["npc_alex_chen"] == math.pi / 2
    assert driver.interaction_yaws["npc_morgan_lee"] == math.pi / 2


def test_production_driver_excludes_unregistered_actions_before_submitting_command() -> None:
    simulator = FakeSimulator()
    driver = create_mujoco_action_driver(simulator)
    assert driver.supported_actions == frozenset(
        {
            ActionType.MOVE_TO,
            ActionType.SIT,
            ActionType.STAND_UP,
            ActionType.WORK,
            ActionType.PICK_UP,
            ActionType.PUT_DOWN,
            ActionType.HANDOVER,
            ActionType.USE_COMPUTER,
            ActionType.TALK,
            ActionType.GESTURE_WAVE,
            ActionType.EAT,
            ActionType.DRINK,
            ActionType.REQUEST_ROBOT,
        }
    )
    assert {
        ActionType.GESTURE_POINT,
    }.isdisjoint(driver.supported_actions)
    execution = ActionExecution(
        command=ActionCommand("employee_01", ActionType.GESTURE_POINT, "employee_02"),
        status=ExecutionStatus.RUNNING,
    )

    result = driver.start(execution)

    assert result.error == "unsupported_action"
    assert simulator.commands == []


def test_driver_reads_desk_work_sites_and_seat_yaws_from_scene_semantics() -> None:
    simulator = FakeSimulator()
    world = SemanticWorld.from_json(MODELS / "office_semantics.json")

    driver = create_mujoco_action_driver(simulator, world=world)

    assert driver.location_sites["workstation_right"] == "desk_right_work_site"
    assert driver.location_sites["chair_right"] == "chair_right_sit"
    assert driver.seat_yaws["chair_right"] == pytest.approx(math.pi)


def test_desk_work_driver_plays_at_the_seated_chair_until_session_duration() -> None:
    simulator = FakeSimulator()
    driver = MujocoNpcActionDriver(
        simulator,
        {"chair_right": "chair_right_sit"},
        seat_yaws={"chair_right": math.pi},
        available_clips={"sit", "work"},
    )
    execution = ActionExecution(
        execution_id="desk_work_1",
        command=ActionCommand(
            "employee_01",
            ActionType.WORK,
            "workstation_right",
            {
                "_desk_work_seat": "chair_right",
                "_desk_work_duration_seconds": 15.0,
                "_desk_work_session_id": "desk_work_1",
            },
        ),
        status=ExecutionStatus.RUNNING,
    )

    assert driver.start(execution).error == "desk_work_requires_seat_receipt"

    sit = ActionExecution(
        execution_id="sit_1",
        command=ActionCommand("employee_01", ActionType.SIT, "chair_right"),
        status=ExecutionStatus.RUNNING,
    )
    assert driver.start(sit).phase == "approach_seat"
    for _ in range(3):
        simulator.receipts.append(
            NpcCommandReceipt(simulator.command.command_id, "employee_01", CommandStatus.SUCCEEDED)
        )
        result = driver.poll(sit)
    assert result.status == ExecutionStatus.SUCCEEDED

    result = driver.start(execution)

    assert result.phase == "work"
    assert simulator.command.kind == NpcCommandKind.PLAY_ANIMATION
    assert simulator.command.payload == {
        "clip": "work",
        "duration": 15.0,
        "arrival_clip": "seated_idle",
        "target_site": "chair_right_sit",
    }


def test_runtime_rejects_candidate_without_registered_production_clip() -> None:
    simulator = FakeSimulator()
    world = SemanticWorld.from_json(MODELS / "office_semantics.json")
    runtime = OfficeAgentRuntime.from_json(
        world,
        MODELS / "office_agents.json",
        auto_plan=False,
        action_driver=create_mujoco_action_driver(simulator),
    )

    result = runtime.submit_action(
        ActionCommand("employee_01", ActionType.GESTURE_POINT, "employee_02")
    )

    assert not result.valid
    assert any("unavailable in the active NPC asset bundle" in error for error in result.errors)
    assert simulator.commands == []


def test_driver_attaches_declared_trajectory_route_for_matching_agent_transition() -> None:
    simulator = FakeSimulator()
    driver = create_mujoco_action_driver(
        simulator,
        trajectory_profile=NpcTrajectoryProfile.from_json(OFFICE_PROFILE),
        agent_locations={"employee_01": "workstation_left"},
    )
    execution = ActionExecution(
        execution_id="profile_move",
        command=ActionCommand("employee_01", ActionType.MOVE_TO, "meeting_table"),
        status=ExecutionStatus.RUNNING,
    )

    result = driver.start(execution)

    assert result.status == ExecutionStatus.RUNNING
    assert simulator.commands[0].payload["trajectory_route"] == "workstation_left_to_meeting"


def test_handover_receive_failure_reattaches_to_giver_before_reporting_failure() -> None:
    simulator = FakeSimulator()
    driver = MujocoNpcActionDriver(
        simulator,
        {},
        handover_sites={"employee_02": "employee_02_handover_site"},
        handover_role_sites={("employee_01", "employee_02"): ("giver_stand", "receiver_stand")},
        interaction_yaws={"employee_01": 0.0, "employee_02": math.pi},
    )
    execution = ActionExecution(
        execution_id="handover_rollback",
        command=ActionCommand(
            "employee_01", ActionType.HANDOVER, "employee_02", {"object": "parcel"}
        ),
        status=ExecutionStatus.RUNNING,
    )

    driver.start(execution)
    for command_count in (2, 2, 2, 1):
        for command in simulator.commands[-command_count:]:
            simulator.receipts.append(
                NpcCommandReceipt(command.command_id, command.npc_id, CommandStatus.SUCCEEDED)
            )
        driver.poll(execution)
    assert simulator.command.kind == NpcCommandKind.ATTACH_OBJECT
    assert simulator.command.npc_id == "employee_02"

    simulator.receipts.append(
        NpcCommandReceipt(
            simulator.command.command_id,
            "employee_02",
            CommandStatus.FAILED,
            "receiver_grasp_failed",
        )
    )
    result = driver.poll(execution)
    assert result.phase == "rollback"
    assert simulator.command.kind == NpcCommandKind.ATTACH_OBJECT
    assert simulator.command.npc_id == "employee_01"

    simulator.receipts.append(
        NpcCommandReceipt(simulator.command.command_id, "employee_01", CommandStatus.SUCCEEDED)
    )
    result = driver.poll(execution)
    assert result.status == ExecutionStatus.FAILED
    assert result.error == "receiver_grasp_failed"


def test_cancel_removes_object_workflow_and_cancels_its_current_stage() -> None:
    simulator = FakeSimulator()
    driver = MujocoNpcActionDriver(
        simulator,
        {},
        object_approach_sites={"parcel": "parcel_human_stand_site"},
        interaction_yaws={"parcel": math.pi},
    )
    execution = ActionExecution(
        execution_id="pick_cancel",
        command=ActionCommand("employee_01", ActionType.PICK_UP, "parcel"),
        status=ExecutionStatus.RUNNING,
    )

    driver.start(execution)
    result = driver.cancel(execution, "user_cancelled")

    assert result.status == ExecutionStatus.CANCELLED
    assert simulator.cancelled == [("employee_01", "pick_cancel:approach")]
    assert "pick_cancel" not in driver._objects


def test_direct_use_computer_driver_command_is_rejected_without_a_desk_work_session() -> None:
    simulator = FakeSimulator()
    execution = ActionExecution(
        execution_id="computer_1",
        command=ActionCommand("employee_01", ActionType.USE_COMPUTER, "workstation_left"),
        status=ExecutionStatus.RUNNING,
    )
    driver = MujocoNpcActionDriver(simulator, {"workstation_left": "desk_site"})

    assert driver.start(execution).error == "use_computer_must_expand_to_desk_work"
    assert simulator.commands == []


def test_eat_requires_a_physical_consume_marker_before_driver_success() -> None:
    simulator = FakeSimulator()
    driver = MujocoNpcActionDriver(
        simulator,
        {},
        object_approach_sites={"bread_snack": "snack_human_stand_site"},
        interaction_yaws={"bread_snack": math.pi},
        available_clips={"eat"},
    )
    execution = ActionExecution(
        execution_id="eat_1",
        command=ActionCommand("employee_01", ActionType.EAT, "bread_snack"),
        status=ExecutionStatus.RUNNING,
    )

    assert driver.start(execution).phase == "approach"
    for expected_kind in ("align_to", "play_animation"):
        simulator.receipts.append(
            NpcCommandReceipt(simulator.command.command_id, "employee_01", CommandStatus.SUCCEEDED)
        )
        assert driver.poll(execution).status is ExecutionStatus.RUNNING
        assert simulator.command.kind.value == expected_kind
    simulator.receipts.append(
        NpcCommandReceipt(simulator.command.command_id, "employee_01", CommandStatus.SUCCEEDED)
    )
    assert driver.poll(execution).status is ExecutionStatus.SUCCEEDED
    assert simulator.commands[-1].payload["completion_marker"] == "consume"


def test_talk_cue_requires_participant_site_then_carries_gaze_contract() -> None:
    simulator = FakeSimulator()
    driver = MujocoNpcActionDriver(
        simulator,
        {},
        cue_sites={"employee_02": "employee_02_conversation_site"},
        interaction_yaws={"employee_02": 1.57},
        available_clips={"talk"},
    )
    execution = ActionExecution(
        execution_id="talk_1",
        command=ActionCommand("employee_01", ActionType.TALK, "employee_02"),
        status=ExecutionStatus.RUNNING,
    )

    assert driver.start(execution).phase == "approach"
    assert simulator.command.payload == {
        "site": "employee_02_conversation_site",
        "arrival_clip": "idle",
        "max_replans": 1,
    }
    for phase in ("align", "play"):
        simulator.receipts.append(
            NpcCommandReceipt(simulator.command.command_id, "employee_01", CommandStatus.SUCCEEDED)
        )
        assert driver.poll(execution).phase == phase
    assert simulator.command.payload == {
        "clip": "talk",
        "completion_marker": "talk_cycle",
        "arrival_clip": "idle",
        "target_site": "employee_02_conversation_site",
        "gaze_target": "employee_02",
        "interaction_target": "employee_02",
        "interaction_distance_min": 0.45,
        "interaction_distance_max": 0.95,
        "interaction_yaw_tolerance": 0.30,
    }


def test_social_cue_rejects_missing_participant_site() -> None:
    simulator = FakeSimulator()
    driver = MujocoNpcActionDriver(
        simulator,
        {},
        interaction_yaws={"employee_02": 1.57},
        available_clips={"gesture_wave"},
    )
    execution = ActionExecution(
        execution_id="wave_missing_site",
        command=ActionCommand("employee_01", ActionType.GESTURE_WAVE, "employee_02"),
        status=ExecutionStatus.RUNNING,
    )

    assert driver.start(execution).error == "recipe_missing_site"


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


def test_robot_handover_prepares_npc_before_accepting_release() -> None:
    simulator = FakeSimulator()
    bridge = RobotToNpcHandoverBridge(
        simulator,
        handover_sites={"stretch_3": "stretch_receive_stand"},
        interaction_yaws={"stretch_3": 0.0},
    )
    handover = bridge.begin_receive(
        "stretch_3", "employee_01", "parcel", session_id="robot_prepared_1"
    )
    assert simulator.command.kind == NpcCommandKind.MOVE_TO
    with pytest.raises(ValueError, match="receive marker"):
        bridge.confirm_robot_release(handover.session.session_id)

    for expected_kind in (NpcCommandKind.ALIGN_TO, NpcCommandKind.PLAY_ANIMATION):
        simulator.receipts.append(
            NpcCommandReceipt(simulator.command.command_id, "employee_01", CommandStatus.SUCCEEDED)
        )
        bridge.poll(handover.session.session_id)
        assert simulator.command.kind == expected_kind
    simulator.receipts.append(
        NpcCommandReceipt(simulator.command.command_id, "employee_01", CommandStatus.SUCCEEDED)
    )
    assert bridge.poll(handover.session.session_id).phase == "released"

    bridge.confirm_robot_release(handover.session.session_id)
    assert simulator.command.kind == NpcCommandKind.ATTACH_OBJECT
    simulator.receipts.append(
        NpcCommandReceipt(simulator.command.command_id, "employee_01", CommandStatus.SUCCEEDED)
    )
    session = bridge.poll(handover.session.session_id)
    assert session.status.value == "succeeded"
    assert session.completed_phases == [
        "rendezvous",
        "aligned",
        "receiver_ready",
        "released",
        "received",
    ]


def test_mock_robot_handover_waits_for_receive_marker_before_release_and_attach() -> None:
    simulator = FakeSimulator()
    world = SemanticWorld.from_json(MODELS / "office_semantics.json")
    world.objects["employee_02"] = SemanticObject(
        "employee_02", ObjectType.EMPLOYEE, world.object("employee_01").binding, {}
    )
    runtime = OfficeAgentRuntime.from_json(world, MODELS / "office_agents.json", auto_plan=False)
    runtime.agents["employee_02"] = EmployeeAgent.from_dict(
        "employee_02",
        {"profile": {"role": "Receiver", "department": "QA"}, "initial_location": "workstation_right"},
    )
    task = RobotTask(
        "employee_01", "deliver", "document_report", "workstation_right", recipient="employee_02"
    )
    runtime.robot_tasks[task.task_id] = task
    robot = MockRobotExecutor(0.1, npc_transport=simulator)

    robot.tick(runtime, 0.1)
    assert simulator.command.kind is NpcCommandKind.MOVE_TO
    for expected in (NpcCommandKind.ALIGN_TO, NpcCommandKind.PLAY_ANIMATION):
        simulator.receipts.append(
            NpcCommandReceipt(simulator.command.command_id, "employee_02", CommandStatus.SUCCEEDED)
        )
        robot.tick(runtime, 0.1)
        assert simulator.command.kind is expected
        assert f"mock:{task.task_id}:robot_release_confirmed" not in task.receipt_ids
    simulator.receipts.append(
        NpcCommandReceipt(simulator.command.command_id, "employee_02", CommandStatus.SUCCEEDED)
    )
    robot.tick(runtime, 0.1)
    assert task.receipt_ids == {f"mock:{task.task_id}:robot_release_confirmed"}
    robot.tick(runtime, 0.1)
    assert simulator.command.kind is NpcCommandKind.ATTACH_OBJECT
    assert f"mock:{task.task_id}:robot_release_confirmed" in task.receipt_ids
    simulator.receipts.append(
        NpcCommandReceipt(simulator.command.command_id, "employee_02", CommandStatus.SUCCEEDED)
    )
    assert robot.tick(runtime, 0.1) == (task.task_id,)
    assert task.status.value == "succeeded"
    assert task.receipt_ids >= {
        f"mock:{task.task_id}:robot_release_confirmed",
        f"mock:{task.task_id}:npc_attachment_confirmed",
        f"mock:{task.task_id}:interaction_completed",
    }


def test_mock_robot_handover_timeout_fails_without_attachment() -> None:
    simulator = FakeSimulator()
    world = SemanticWorld.from_json(MODELS / "office_semantics.json")
    world.objects["employee_02"] = SemanticObject(
        "employee_02", ObjectType.EMPLOYEE, world.object("employee_01").binding, {}
    )
    runtime = OfficeAgentRuntime.from_json(world, MODELS / "office_agents.json", auto_plan=False)
    runtime.agents["employee_02"] = EmployeeAgent.from_dict(
        "employee_02",
        {"profile": {"role": "Receiver", "department": "QA"}, "initial_location": "workstation_right"},
    )
    task = RobotTask("employee_01", "deliver", "document_report", "workstation_right", recipient="employee_02")
    runtime.robot_tasks[task.task_id] = task
    robot = MockRobotExecutor(0.1, npc_transport=simulator)

    robot.tick(runtime, 0.1)
    simulator.time = 100.0
    assert robot.tick(runtime, 0.1) == (task.task_id,)
    assert task.status.value == "failed"
    assert not task.npc_attachment_confirmed


def test_runtime_rejects_handover_success_without_all_physical_evidence() -> None:
    world = SemanticWorld.from_json(MODELS / "office_semantics.json")
    runtime = OfficeAgentRuntime.from_json(world, MODELS / "office_agents.json", auto_plan=False)
    task = RobotTask("employee_01", "deliver", "document_report", "workstation_right", recipient="employee_01")
    runtime.robot_tasks[task.task_id] = task

    completed = runtime.complete_robot_task(task.task_id, success=True)

    assert completed.status.value == "failed"
    assert completed.error == (
        "robot_handover_evidence_missing:robot_release_confirmed,"
        "npc_attachment_confirmed,interaction_confirmed"
    )
