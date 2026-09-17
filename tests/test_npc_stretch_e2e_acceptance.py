"""Headless, single-scene acceptance for three NPCs and the live Stretch body.

This deliberately uses a compiled office MJCF and ``NpcSystem`` rather than
the mock transport used by deterministic conversation recordings.  The
in-process transport advances the same controller and MuJoCo ``MjData`` that
the server owns; it exists to keep the acceptance test bounded and does not
claim a robot grasp/delivery that the public Stretch API cannot execute yet.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import mujoco

from stretch_mujoco.agents import (
    ActionCommand,
    ActionExecution,
    ActionType,
    ConversationRequest,
    DialogueAct,
    DialogueCandidate,
    ExecutionStatus,
    RobotTaskType,
)
from stretch_mujoco.agents.robot_task_driver import UnsupportedRobotTaskExecutor
from stretch_mujoco.agents.simulation_bridge import create_mujoco_action_driver
from stretch_mujoco.npc import NpcCommand, NpcCommandKind
from stretch_mujoco.npc.composition import load_composed_npc_runtime


MODELS = Path(__file__).resolve().parents[1] / "stretch_mujoco" / "models"
PRODUCTION_POPULATION = MODELS / "office_population.production.example.json"
NPC_IDS = ("npc_morgan_lee", "npc_jordan_patell", "npc_priya_narayanan")


class _TerminalUnavailableRobotExecutor(UnsupportedRobotTaskExecutor):
    """Allow request dispatch, then emit the production unavailable receipt.

    This is a negative transport fixture, not a simulated robot: it never
    creates a delivery/grasp success and deliberately delegates the terminal
    failed receipt to ``UnsupportedRobotTaskExecutor``.
    """

    supported_task_types = frozenset(RobotTaskType)


class _InProcessMujocoNpcTransport:
    """Transport facade over one real compiled scene and its NPC system.

    Rendering and full robot physics are intentionally outside this focused
    acceptance check.  Each advance still uses the compiled MuJoCo model/data,
    calls the production NPC controller, and records the controller's actual
    command receipts.  The Stretch ``base_link`` remains the physical target
    for the request interaction gate.
    """

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, npc_system: object) -> None:
        self.model = model
        self.data = data
        self.npc_system = npc_system
        self.time = 0.0
        self._receipts = []

    def pull_status(self) -> SimpleNamespace:
        return SimpleNamespace(time=self.time)

    def submit_npc_command(self, command):
        self.npc_system.submit(command)
        self._receipts.extend(self.npc_system.drain_receipts())
        return command.command_id

    def pull_npc_receipts(self):
        receipts = tuple(self._receipts)
        self._receipts.clear()
        return receipts

    def cancel_npc_command(self, npc_id: str, command_id: str) -> None:
        # The acceptance workflows are expected to terminate normally; retain
        # the production cancellation protocol for an unexpected cleanup path.
        self.npc_system.submit(
            NpcCommand(
                f"cancel:{command_id}",
                10_000 + len(self._receipts),
                npc_id,
                NpcCommandKind.CANCEL,
                {"command_id": command_id},
                self.time,
            )
        )
        self._receipts.extend(self.npc_system.drain_receipts())

    def advance(self, seconds: float = 0.05) -> None:
        self.time += seconds
        # Keep the observation clock and controller clock identical.  This is
        # the same data object used for all scene-site, collision, attachment,
        # and Stretch-body interaction checks.
        self.data.time = self.time
        self.npc_system.step(self.model, self.data, self.time)
        # Navigation planning observes the latest collision-proxy transforms;
        # mocap updates made by the controller are not reflected in geom_xpos
        # until MuJoCo's forward pass runs.
        mujoco.mj_forward(self.model, self.data)
        self._receipts.extend(self.npc_system.drain_receipts())


def _three_npc_population(tmp_path: Path) -> Path:
    """Derive a compact acceptance roster without altering production config."""

    payload = json.loads(PRODUCTION_POPULATION.read_text(encoding="utf-8"))
    payload["npcs"] = {npc_id: payload["npcs"][npc_id] for npc_id in NPC_IDS}
    for field in ("scene", "asset_manifest", "appearance_catalog", "trajectory_profile"):
        if field in payload:
            payload[field] = str((MODELS / payload[field]).resolve())
    path = tmp_path / "three_npc_office_population.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _advance_until(transport, predicate, *, steps: int = 1_400) -> None:
    for _ in range(steps):
        transport.advance()
        if predicate():
            return
    raise AssertionError("e2e workflow did not reach a terminal state")


def test_three_npc_stretch_single_scene_receipt_chain(tmp_path: Path) -> None:
    """Exercise navigation, dialogue, Stretch request, and handover in one MJCF.

    A real robot delivery is intentionally *not* asserted: the present public
    Stretch transport has no semantic waypoint or grasp-IK receipt endpoint.
    ``UnsupportedRobotTaskExecutor`` therefore produces the required terminal
    failure instead of allowing a logical completion to stand in for physics.
    """

    population_path = _three_npc_population(tmp_path)
    loaded = load_composed_npc_runtime(population_path, tmp_path / "three_npc_office.xml")
    model = loaded.model
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    transport = _InProcessMujocoNpcTransport(model, data, loaded.npc_system)
    runtime = loaded.office_runtime
    driver = create_mujoco_action_driver(
        transport,
        npc_ids=NPC_IDS,
        world=loaded.semantic_world,
        interaction_templates=runtime.population_interaction_templates,
    )
    runtime.action_driver = driver
    runtime.interaction_driver = driver
    runtime.robot_task_driver = _TerminalUnavailableRobotExecutor()

    assert model.body("base_link").id >= 0
    assert set(loaded.npc_system.controllers) == set(NPC_IDS)

    # Two independent controllers first navigate concurrently in the same
    # office.  The third command is interleaved after they free the central
    # aisle; this exercises dynamic occupancy without treating a blocked route
    # as a collision-free success.
    navigation = tuple(
        ActionExecution(
            execution_id=f"e2e-navigation-{npc_id}",
            command=ActionCommand(npc_id, ActionType.MOVE_TO, target),
            status=ExecutionStatus.RUNNING,
        )
        for npc_id, target in (
            ("npc_morgan_lee", "snack_counter"),
            ("npc_jordan_patell", "meeting_table"),
        )
    )
    for execution in navigation:
        started = driver.start(execution)
        assert started.status is ExecutionStatus.RUNNING
        execution.driver_handle = started.handle
        execution.phase = started.phase
    navigation_results = {}

    def navigation_finished() -> bool:
        for execution in navigation:
            navigation_results[execution.execution_id] = driver.poll(execution)
        return all(result.status is not ExecutionStatus.RUNNING for result in navigation_results.values())

    _advance_until(transport, navigation_finished)
    assert {result.status for result in navigation_results.values()} == {ExecutionStatus.SUCCEEDED}, {
        key: result.error for key, result in navigation_results.items()
    }
    priya_navigation = ActionExecution(
        execution_id="e2e-navigation-npc_priya_narayanan",
        command=ActionCommand("npc_priya_narayanan", ActionType.MOVE_TO, "storage_cabinet"),
        status=ExecutionStatus.RUNNING,
    )
    started = driver.start(priya_navigation)
    assert started.status is ExecutionStatus.RUNNING
    priya_navigation.driver_handle = started.handle

    def priya_navigation_finished() -> bool:
        nonlocal started
        started = driver.poll(priya_navigation)
        return started.status is not ExecutionStatus.RUNNING

    _advance_until(transport, priya_navigation_finished)
    assert started.status is ExecutionStatus.SUCCEEDED, started.error
    # Direct driver navigation above advances the physical clock; align the
    # runtime clock before it consumes the corresponding live observation.
    runtime.tick(transport.time, loaded.semantic_world.pose_snapshot(model, data))

    # Conversation is receipt-gated: no transcript is committed before the
    # approach, align, and talk commands run on the same NPC controllers.
    conversation_id = "e2e-npc-conversation"
    conversation_receipt = runtime.begin_conversation(
        ConversationRequest(
            conversation_id,
            # The first participant owns the authored speaker role. Priya's
            # storage-side route reaches that left station without crossing
            # Jordan's meeting-side route to the listener station.
            ("npc_priya_narayanan", "npc_jordan_patell"),
            "office status",
            timeout=120.0,
            max_turns=1,
            semantic_snapshot=loaded.semantic_world.pose_snapshot(model, data),
        )
    )
    assert conversation_receipt.accepted

    def conversation_ready() -> bool:
        runtime.tick(0.05, loaded.semantic_world.pose_snapshot(model, data))
        return runtime.conversation(conversation_id).phase.value == "waiting_for_turn"

    _advance_until(transport, conversation_ready)
    candidate = DialogueCandidate(
        "e2e-dialogue-request",
        conversation_id,
        "e2e-dialogue-turn-1",
        "npc_priya_narayanan",
        "npc_jordan_patell",
        DialogueAct.GREETING,
        "Status check complete.",
        runtime.elapsed_minutes,
    )
    assert runtime.submit_dialogue_candidate(candidate).valid

    def conversation_finished() -> bool:
        runtime.tick(0.05, loaded.semantic_world.pose_snapshot(model, data))
        return runtime.conversation(conversation_id).status.terminal

    _advance_until(transport, conversation_finished)
    session = runtime.conversation(conversation_id)
    assert session.status.value == "completed"
    assert session.dialogue_turns["e2e-dialogue-turn-1"].status.value == "committed"

    # The NPC physically approaches and addresses the *actual* Stretch body.
    # The robot executor then rejects the follow-up task because no live
    # waypoint/IK transport is installed, which is the only honest terminal
    # outcome currently available.
    assert runtime.submit_action(
        ActionCommand(
            "npc_morgan_lee",
            ActionType.REQUEST_ROBOT,
            "stretch_3",
            {"task": "deliver", "object": "soda_can", "destination": "workstation_right"},
        )
    ).valid

    def robot_task_finished() -> bool:
        runtime.tick(0.05, loaded.semantic_world.pose_snapshot(model, data))
        return bool(runtime.robot_tasks) and all(task.status.value != "pending" for task in runtime.robot_tasks.values())

    _advance_until(transport, robot_task_finished)
    task = next(iter(runtime.robot_tasks.values()))
    assert task.status.value == "failed"
    assert task.error == "robot_task_executor_not_configured"
    assert not any(event.event == "robot_delivery_verified" for event in runtime.events)

    # Finish with a physical NPC-to-NPC handover.  Ownership starts only after
    # an actual attachment command on the same model; the driver then enforces
    # rendezvous, opposing yaw, give/receive markers, detach, and reattach.
    loaded.npc_system.submit(
        NpcCommand(
            "e2e-attach-report",
            100,
            "npc_morgan_lee",
            NpcCommandKind.ATTACH_OBJECT,
            {"object": "document_report", "visible": False},
            transport.time,
        )
    )
    transport._receipts.extend(loaded.npc_system.drain_receipts())
    transport.advance()
    transport.pull_npc_receipts()
    # The direct attachment establishes the production transport sequence
    # boundary before the driver submits the handover workflow.
    driver._sequences["npc_morgan_lee"] = 100
    handover = ActionExecution(
        execution_id="e2e-npc-handover",
        command=ActionCommand(
            "npc_morgan_lee",
            ActionType.HANDOVER,
            "npc_jordan_patell",
            {"object": "document_report"},
        ),
        status=ExecutionStatus.RUNNING,
    )
    assert driver.start(handover).status is ExecutionStatus.RUNNING
    handover_result = None

    def handover_finished() -> bool:
        nonlocal handover_result
        handover_result = driver.poll(handover)
        return handover_result.status is not ExecutionStatus.RUNNING

    _advance_until(transport, handover_finished)
    assert handover_result is not None and handover_result.status is ExecutionStatus.SUCCEEDED
    assert loaded.npc_system.controllers["npc_morgan_lee"].attachments.held_objects == ()
    assert loaded.npc_system.controllers["npc_jordan_patell"].attachments.held_objects == (
        "document_report",
    )

    # Runtime events are an ordered audit trail: every emitted event has one
    # unique ID and the numeric suffix is strictly increasing.
    event_ids = [event.event_id for event in runtime.events]
    assert event_ids and len(event_ids) == len(set(event_ids))
    assert [int(event_id.rsplit("_", 1)[1]) for event_id in event_ids] == sorted(
        int(event_id.rsplit("_", 1)[1]) for event_id in event_ids
    )
