import mujoco
import numpy as np

from stretch_mujoco.npc import CommandStatus, NpcCommand, NpcCommandKind
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
