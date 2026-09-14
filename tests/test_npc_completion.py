import mujoco
import numpy as np
import math

from stretch_mujoco.npc import CommandStatus, NpcCommand, NpcCommandKind
from stretch_mujoco.npc.animation import MeshSequenceBackend
from stretch_mujoco.npc.binding import NpcBinding
from stretch_mujoco.npc.locomotion import LocomotionController, yaw_quaternion
from stretch_mujoco.npc.system import NpcSystem


def _model() -> mujoco.MjModel:
    def frames(clip: str, count: int) -> str:
        return "\n".join(
            f'<geom name="npc__employee_01__clip__{clip}__frame__{index:03d}__slot__body" '
            f'type="sphere" size=".1" rgba="1 1 1 0"/>'
            for index in range(count)
        )

    sit_down_frames = frames("sit_down", 8)
    seated_idle_frames = frames("seated_idle", 4)
    stand_up_frames = frames("stand_up", 8)
    talk_frames = frames("talk", 8)
    walk_frames = frames("walk", 8)
    give_frames = frames("give", 8)
    return mujoco.MjModel.from_xml_string(
        f"""
        <mujoco>
          <worldbody>
            <body name="npc__employee_01" mocap="true">
              <geom name="npc__employee_01__clip__idle__frame__000__slot__body"
                    type="sphere" size=".1"/>
              {sit_down_frames}
              {seated_idle_frames}
              {stand_up_frames}
              {talk_frames}
              {walk_frames}
              {give_frames}
              <site name="npc__employee_01__handover" pos="0 0 1"/>
            </body>
            <body name="npc__employee_02" mocap="true" pos="0 1 0">
              <geom name="npc__employee_02__clip__idle__frame__000__slot__body"
                    type="sphere" size=".1"/>
              <site name="npc__employee_02__handover" pos="0 0 1"/>
            </body>
            <body name="parcel" pos="1 0 .2">
              <freejoint/>
              <geom type="box" size=".05 .05 .05"/>
            </body>
            <site name="drop_site" pos="2 0 .5"/>
          </worldbody>
        </mujoco>
        """
    )


def _command(kind: NpcCommandKind, payload: dict[str, object], sequence: int) -> NpcCommand:
    return NpcCommand(f"command_{sequence}", sequence, "employee_01", kind, payload, 0.0)


def test_sit_completes_at_animation_marker() -> None:
    model = _model()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    system = NpcSystem.from_model(model)
    system.submit(
        _command(
            NpcCommandKind.PLAY_ANIMATION,
            {"clip": "sit_down", "completion_marker": "seated"},
            0,
        )
    )

    for index in range(9):
        system.step(model, data, index * 0.125)

    state = system.states(data)["employee_01"]
    assert state.last_receipt is not None
    assert state.last_receipt.status == CommandStatus.SUCCEEDED
    assert state.resolved_clip == "seated_idle"


def test_stand_up_marker_returns_to_idle_only_after_completion() -> None:
    model = _model()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    system = NpcSystem.from_model(model)
    system.submit(
        _command(
            NpcCommandKind.PLAY_ANIMATION,
            {"clip": "sit_down", "completion_marker": "seated"},
            0,
        )
    )
    for index in range(10):
        system.step(model, data, index * 0.125)
    system.submit(
        _command(
            NpcCommandKind.PLAY_ANIMATION,
            {"clip": "stand_up", "completion_marker": "standing"},
            1,
        )
    )
    for index in range(10, 17):
        system.step(model, data, index * 0.125)

    state = system.states(data)["employee_01"]
    assert state.last_receipt is not None
    assert state.last_receipt.status == CommandStatus.SUCCEEDED
    assert state.resolved_clip == "stand_up"
    assert state.requested_animation == "idle"
    system.step(model, data, 2.5)
    assert system.states(data)["employee_01"].resolved_clip == "idle"


def test_social_animation_waits_for_distance_and_mutual_facing() -> None:
    model = _model()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    system = NpcSystem.from_model(model)
    system.submit(
        _command(
            NpcCommandKind.PLAY_ANIMATION,
            {
                "clip": "talk",
                "interaction_target": "employee_02",
                "interaction_distance_min": 0.45,
                "interaction_distance_max": 0.95,
                "interaction_yaw_tolerance": 0.30,
            },
            0,
        )
    )
    system.step(model, data, 0.0)
    assert system.states(data)["employee_01"].resolved_clip == "idle"

    first_mocap = model.body("npc__employee_01").mocapid[0]
    second_mocap = model.body("npc__employee_02").mocapid[0]
    data.mocap_pos[first_mocap] = (-0.35, 0.0, 0.0)
    data.mocap_pos[second_mocap] = (0.35, 0.0, 0.0)
    data.mocap_quat[first_mocap] = yaw_quaternion(math.pi / 2)
    data.mocap_quat[second_mocap] = yaw_quaternion(-math.pi / 2)
    mujoco.mj_forward(model, data)
    system.step(model, data, 0.125)
    assert system.states(data)["employee_01"].resolved_clip == "talk"


def test_attach_and_detach_receipts_follow_physical_state() -> None:
    model = _model()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    system = NpcSystem.from_model(model)

    system.submit(_command(NpcCommandKind.ATTACH_OBJECT, {"object": "parcel"}, 0))
    system.step(model, data, 0.1)
    state = system.states(data)["employee_01"]
    assert state.held_objects == ("parcel",)
    parcel_joint = model.body("parcel").jntadr[0]
    qpos_address = model.jnt_qposadr[parcel_joint]
    np.testing.assert_allclose(
        data.qpos[qpos_address : qpos_address + 3], (0.0, 0.0, 1.0), atol=1e-6
    )
    parcel_body = model.body("parcel").id
    np.testing.assert_allclose(data.xpos[parcel_body], (0.0, 0.0, 1.0), atol=1e-6)

    # A held free body must update both qpos and the renderer-visible xpos in
    # the same NPC system step as its owner moves.
    employee_mocap = model.body("npc__employee_01").mocapid[0]
    data.mocap_pos[employee_mocap] = (0.5, -0.25, 0.0)
    system.step(model, data, 0.15)
    np.testing.assert_allclose(
        data.qpos[qpos_address : qpos_address + 3], (0.5, -0.25, 1.0), atol=1e-6
    )
    np.testing.assert_allclose(data.xpos[parcel_body], (0.5, -0.25, 1.0), atol=1e-6)

    system.submit(
        _command(
            NpcCommandKind.DETACH_OBJECT,
            {"object": "parcel", "site": "drop_site"},
            1,
        )
    )
    system.step(model, data, 0.2)
    assert system.states(data)["employee_01"].held_objects == ()
    np.testing.assert_allclose(data.qpos[qpos_address : qpos_address + 3], (2.0, 0.0, 0.5))


def test_hidden_pickup_attachment_reveals_when_give_animation_starts() -> None:
    model = _model()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    system = NpcSystem.from_model(model)
    parcel_geom = next(
        geom_id
        for geom_id in range(model.ngeom)
        if int(model.geom_bodyid[geom_id]) == model.body("parcel").id
    )

    system.submit(
        _command(
            NpcCommandKind.ATTACH_OBJECT,
            {"object": "parcel", "visible": False},
            0,
        )
    )
    system.step(model, data, 0.1)
    assert model.geom_rgba[parcel_geom, 3] == 0.0

    system.submit(
        _command(
            NpcCommandKind.PLAY_ANIMATION,
            {"clip": "give", "reveal_object": "parcel"},
            1,
        )
    )
    system.step(model, data, 0.2)
    assert model.geom_rgba[parcel_geom, 3] == 1.0


def test_visible_mesh_frame_moves_attachment_site_to_its_hand_anchor() -> None:
    model = mujoco.MjModel.from_xml_string(
        """
        <mujoco><worldbody>
          <body name="npc__employee_01" mocap="true">
            <geom name="npc__employee_01__clip__idle__frame__000__slot__body"
                  type="sphere" size=".1"/>
            <geom name="npc__employee_01__clip__walk__frame__000__slot__body"
                  type="sphere" size=".1" rgba="1 1 1 0"/>
            <geom name="npc__employee_01__clip__walk__frame__001__slot__body"
                  type="sphere" size=".1" rgba="1 1 1 0"/>
            <site name="npc__employee_01__handover" pos="-.2 0 .9"/>
            <site name="npc__employee_01__anchor__handover__clip__walk__frame__000"
                  pos="-.25 -.1 .85"/>
            <site name="npc__employee_01__anchor__handover__clip__walk__frame__001"
                  pos="-.25 .15 .88"/>
          </body>
        </worldbody></mujoco>
        """
    )
    binding = NpcBinding.from_model(model, "employee_01")
    backend = MeshSequenceBackend(model, binding)

    backend.sample("walk", 0.0)
    np.testing.assert_allclose(model.site("npc__employee_01__handover").pos, (-.25, -.1, .85))
    backend.sample("walk", 0.75)
    np.testing.assert_allclose(model.site("npc__employee_01__handover").pos, (-.25, .15, .88))


def test_missing_clip_emits_one_fallback_event() -> None:
    model = _model()
    data = mujoco.MjData(model)
    system = NpcSystem.from_model(model)
    system.submit(_command(NpcCommandKind.PLAY_ANIMATION, {"clip": "missing", "duration": 0.2}, 0))

    system.step(model, data, 0.0)
    state = system.states(data)["employee_01"]
    assert state.animation_events == ("clip_fallback",)
    assert state.animation_lifecycle == "failed"
    assert state.last_receipt is not None
    assert state.last_receipt.status == CommandStatus.FAILED
    assert state.last_receipt.reason == "clip_unavailable:missing"


def test_interaction_animation_rejects_unknown_target_site_and_settles_to_idle() -> None:
    model = _model()
    data = mujoco.MjData(model)
    system = NpcSystem.from_model(model)

    invalid = system.submit(
        _command(
            NpcCommandKind.PLAY_ANIMATION,
            {"clip": "sit_down", "target_site": "missing_interaction_site"},
            0,
        )
    )
    assert invalid.status == CommandStatus.FAILED
    assert invalid.reason == "unknown_target_site:missing_interaction_site"

    system.submit(
        _command(
            NpcCommandKind.PLAY_ANIMATION,
            {
                "clip": "sit_down",
                "completion_marker": "seated",
                "arrival_clip": "idle",
                "target_site": "drop_site",
            },
            1,
        )
    )
    for index in range(10):
        system.step(model, data, index * 0.125)
    system.step(model, data, 1.5)

    state = system.states(data)["employee_01"]
    assert state.last_receipt is not None
    assert state.last_receipt.status == CommandStatus.SUCCEEDED
    assert state.requested_animation == "idle"
    assert state.resolved_clip == "idle"


def test_align_and_animation_fail_if_the_npc_leaves_the_target_site() -> None:
    model = _model()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    system = NpcSystem.from_model(model)
    assert (
        system.submit(
            _command(NpcCommandKind.ALIGN_TO, {"yaw": 0.0, "target_site": "drop_site"}, 0)
        ).status
        == CommandStatus.ACCEPTED
    )
    system.step(model, data, 0.1)
    receipt = system.states(data)["employee_01"].last_receipt
    assert receipt is not None
    assert receipt.status == CommandStatus.FAILED
    assert receipt.reason == "target_site_not_reached:drop_site"


def test_seat_ingress_never_assigns_a_large_mocap_jump() -> None:
    model = mujoco.MjModel.from_xml_string(
        """
        <mujoco><worldbody>
          <body name="npc__employee_01" mocap="true">
            <geom name="npc__employee_01__clip__idle__frame__000__slot__body" type="sphere" size=".1"/>
          </body>
          <site name="approach" pos="0.4 0 0"/>
          <site name="seat" pos="1.0 0 0"/>
        </worldbody></mujoco>
        """
    )
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    controller = LocomotionController(model, NpcBinding.from_model(model, "employee_01"))
    controller.move_to("seat", speed=0.5, navigation_site="approach")
    previous = data.mocap_pos[0, :2].copy()
    for index in range(1, 50):
        controller.step(data, index * 0.1)
        current = data.mocap_pos[0, :2].copy()
        assert np.linalg.norm(current - previous) <= 0.050001
        previous = current
        if controller.target_site is None:
            break
    assert controller.target_site is None
    np.testing.assert_allclose(data.mocap_pos[0, :2], (1.0, 0.0), atol=1e-6)


def test_interaction_animation_timeout_recovers_to_idle() -> None:
    model = _model()
    data = mujoco.MjData(model)
    system = NpcSystem.from_model(model)
    system.submit(
        NpcCommand(
            "interaction_timeout",
            0,
            "employee_01",
            NpcCommandKind.PLAY_ANIMATION,
            {
                "clip": "sit_down",
                "completion_marker": "seated",
                "target_site": "drop_site",
            },
            issued_at=0.0,
            deadline=0.01,
        )
    )

    system.step(model, data, 0.0)
    system.step(model, data, 0.02)
    system.step(model, data, 0.03)

    state = system.states(data)["employee_01"]
    assert state.last_receipt is not None
    assert state.last_receipt.status == CommandStatus.TIMED_OUT
    assert state.requested_animation == "idle"
    assert state.resolved_clip == "idle"


def test_move_waits_for_walk_marker_before_start_and_stop() -> None:
    model = _model()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    system = NpcSystem.from_model(model)
    system.submit(_command(NpcCommandKind.MOVE_TO, {"site": "drop_site"}, 0))

    system.step(model, data, 0.0)
    np.testing.assert_allclose(data.mocap_pos[0, :2], (0.0, 0.0))
    for index in range(1, 100):
        system.step(model, data, index * 0.125)
        state = system.states(data)["employee_01"]
        if state.last_receipt is not None and state.last_receipt.status.terminal:
            break

    state = system.states(data)["employee_01"]
    assert state.last_receipt is not None
    assert state.last_receipt.status == CommandStatus.SUCCEEDED
    assert state.stop_marker in {"left_foot", "right_foot"}
    np.testing.assert_allclose(data.mocap_pos[0, :2], (2.0, 0.0), atol=1e-6)


def test_delayed_walk_cancel_finalizes_both_target_and_cancel_receipts() -> None:
    model = _model()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    system = NpcSystem.from_model(model)
    target = _command(NpcCommandKind.MOVE_TO, {"site": "drop_site"}, 0)
    assert system.submit(target).status == CommandStatus.ACCEPTED
    for index in range(1, 30):
        system.step(model, data, index * 0.125)
        if (
            system.controllers["employee_01"].active_command is not None
            and data.mocap_pos[0, 0] > 0
        ):
            break
    cancel = NpcCommand(
        "cancel_delayed",
        1,
        "employee_01",
        NpcCommandKind.CANCEL,
        {"command_id": target.command_id},
        index * 0.125,
    )
    assert system.submit(cancel).status == CommandStatus.RUNNING
    system.drain_receipts()
    for index in range(index + 1, 80):
        system.step(model, data, index * 0.125)
        if system._receipts["cancel_delayed"].status.terminal:
            break
    terminal = {
        receipt.command_id: receipt
        for receipt in system.drain_receipts()
        if receipt.status.terminal
    }
    assert terminal[target.command_id].status == CommandStatus.CANCELLED
    assert terminal["cancel_delayed"].status == CommandStatus.CANCELLED
    assert terminal["cancel_delayed"].finished_at is not None
    assert system.submit(cancel) == terminal["cancel_delayed"]


def test_fail_active_commands_finalizes_a_pending_walk_cancel() -> None:
    model = _model()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    system = NpcSystem.from_model(model)
    target = _command(NpcCommandKind.MOVE_TO, {"site": "drop_site"}, 0)
    system.submit(target)
    for index in range(1, 30):
        system.step(model, data, index * 0.125)
        if data.mocap_pos[0, 0] > 0:
            break
    cancel = NpcCommand(
        "cancel_shutdown",
        1,
        "employee_01",
        NpcCommandKind.CANCEL,
        {"command_id": target.command_id},
        index * 0.125,
    )
    assert system.submit(cancel).status == CommandStatus.RUNNING
    system.fail_active_commands(index * 0.125 + 0.01, "shutdown")
    receipt = system.submit(cancel)
    assert receipt.status == CommandStatus.FAILED
    assert receipt.finished_at == index * 0.125 + 0.01


def test_dynamic_mocap_obstacle_stops_a_preplanned_segment() -> None:
    model = mujoco.MjModel.from_xml_string(
        """
        <mujoco><worldbody>
          <geom name="office_floor" type="box" pos="0 0 -.05" size="3 3 .05"/>
          <body name="npc__employee_01" mocap="true">
            <geom name="npc__employee_01__clip__idle__frame__000__slot__body" type="sphere" size=".08"/>
            <geom name="npc__employee_01__collision__body" type="sphere" size=".10"/>
          </body>
          <body name="npc__employee_02" mocap="true" pos="0 2 0">
            <geom name="npc__employee_02__clip__idle__frame__000__slot__body" type="sphere" size=".08"/>
            <geom name="npc__employee_02__collision__body" type="sphere" size=".10"/>
          </body>
          <site name="target" pos="2 0 0"/>
        </worldbody></mujoco>
        """
    )
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    mover = LocomotionController(model, NpcBinding.from_model(model, "employee_01"))
    mover.move_to("target", speed=0.5, max_replans=0)
    assert not mover.step(data, 0.0)  # Initial route has no blocker in its segment.
    data.mocap_pos[model.body("npc__employee_02").mocapid[0], :2] = (0.30, 0.0)
    mujoco.mj_forward(model, data)
    assert not mover.step(data, 0.1)
    assert mover.failure_reason == "route_blocked_dynamic"
    assert data.mocap_pos[model.body("npc__employee_01").mocapid[0], 0] < 0.05


def test_office_move_routes_around_collision_geometry() -> None:
    model = mujoco.MjModel.from_xml_string(
        """
        <mujoco>
          <worldbody>
            <geom name="office_floor" type="box" pos="0 0 -.05" size="3 3 .05"/>
            <body name="npc__employee_01" mocap="true">
              <geom name="npc__employee_01__clip__idle__frame__000__slot__body"
                    type="sphere" size=".1"/>
              <site name="npc__employee_01__handover" pos="0 0 1"/>
            </body>
            <body name="blocking_desk" pos="1 0 .5">
              <geom type="box" size=".15 .45 .5"/>
            </body>
            <site name="drop_site" pos="2 0 0" euler="0 0 0"/>
          </worldbody>
        </mujoco>
        """
    )
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    controller = LocomotionController(model, NpcBinding.from_model(model, "employee_01"))
    controller.move_to("drop_site")
    visited_y = []

    for index in range(100):
        controller.step(data, index * 0.1)
        visited_y.append(float(data.mocap_pos[0, 1]))
        if controller.target_site is None:
            break

    assert controller.failure_reason is None
    assert controller.target_site is None
    assert max(abs(value) for value in visited_y) > 0.5
    np.testing.assert_allclose(data.mocap_pos[0, :2], (2.0, 0.0), atol=1e-6)


def test_blocked_move_replans_once_then_fails_without_location_commit() -> None:
    model = _model()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    system = NpcSystem.from_model(model)
    system.submit(
        _command(
            NpcCommandKind.MOVE_TO,
            {"site": "drop_site", "speed": 0.001, "progress_timeout": 0.01, "max_replans": 1},
            0,
        )
    )

    for index in range(100):
        system.step(model, data, index * 0.125)
        state = system.states(data)["employee_01"]
        if state.last_receipt is not None and state.last_receipt.status.terminal:
            break

    state = system.states(data)["employee_01"]
    assert state.last_receipt is not None
    assert state.last_receipt.status == CommandStatus.FAILED
    assert state.last_receipt.reason == "route_blocked"
    assert state.replan_attempt == 1
    assert state.locomotion_failure == "route_blocked"


def test_lifecycle_tracks_requested_navigation_alignment_playback_and_completion() -> None:
    model = _model()
    data = mujoco.MjData(model)
    system = NpcSystem.from_model(model)

    system.submit(_command(NpcCommandKind.MOVE_TO, {"site": "drop_site"}, 0))
    assert system.states(data)["employee_01"].animation_lifecycle == "navigating"
    system.submit(
        NpcCommand(
            "cancel_0", 1, "employee_01", NpcCommandKind.CANCEL, {"command_id": "command_0"}, 0.0
        )
    )

    system.submit(_command(NpcCommandKind.ALIGN_TO, {"yaw": 1.0}, 2))
    assert system.states(data)["employee_01"].animation_lifecycle == "aligning"
    system.submit(
        NpcCommand(
            "cancel_1", 3, "employee_01", NpcCommandKind.CANCEL, {"command_id": "command_2"}, 0.0
        )
    )

    system.submit(_command(NpcCommandKind.PLAY_ANIMATION, {"clip": "sit_down"}, 4))
    assert system.states(data)["employee_01"].animation_lifecycle == "requested"
    system.step(model, data, 0.0)
    assert system.states(data)["employee_01"].animation_lifecycle == "playing"

    system.submit(
        NpcCommand(
            "cancel_2", 5, "employee_01", NpcCommandKind.CANCEL, {"command_id": "command_4"}, 0.0
        )
    )
    assert system.states(data)["employee_01"].animation_lifecycle == "failed"

    system.submit(
        _command(
            NpcCommandKind.PLAY_ANIMATION,
            {"clip": "sit_down", "completion_marker": "seated"},
            6,
        )
    )
    for index in range(1, 10):
        system.step(model, data, index * 0.125)
    assert system.states(data)["employee_01"].animation_lifecycle == "completed"


def test_cancelled_attach_releases_cross_npc_object_claim() -> None:
    model = _model()
    system = NpcSystem.from_model(model)
    attach = _command(NpcCommandKind.ATTACH_OBJECT, {"object": "parcel"}, 0)
    system.submit(attach)
    cancel = NpcCommand(
        "cancel_1",
        1,
        "employee_01",
        NpcCommandKind.CANCEL,
        {"command_id": attach.command_id},
        0.0,
    )

    assert system.submit(cancel).status == CommandStatus.CANCELLED
    second_attach = NpcCommand(
        "second_attach",
        0,
        "employee_02",
        NpcCommandKind.ATTACH_OBJECT,
        {"object": "parcel"},
        0.0,
    )
    assert system.submit(second_attach).status == CommandStatus.ACCEPTED
