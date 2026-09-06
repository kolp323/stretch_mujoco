from pathlib import Path
from types import SimpleNamespace

import numpy as np

from stretch_mujoco.recording.native_scene import build_native_multi_npc_scene
from stretch_mujoco.recording.renderers import (
    _transcode_h264,
    annotate_3d_frame,
    event_lines,
    render_topdown_video,
)
from stretch_mujoco.recording.snapshots import (
    JsonlSnapshotWriter,
    build_office_snapshot,
    read_snapshots,
)


class FakeCommand:
    target = "meeting_table"

    class action:
        value = "move_to"


class FakeExecutor:
    command = FakeCommand()
    remaining_minutes = 1.5


class FakeState:
    location = "workstation_left"
    current_action = "move_to"


class FakeProfile:
    role = "Operations Specialist"


class FakeAgent:
    state = FakeState()
    executor = FakeExecutor()
    profile = FakeProfile()


class FakeRuntime:
    minute_of_day = 9 * 60
    agents = {"employee_01": FakeAgent()}
    robot_tasks = {}


def test_snapshot_interpolates_active_navigation() -> None:
    snapshot = build_office_snapshot(
        FakeRuntime(),
        {"workstation_left": (-3.0, -3.0), "meeting_table": (-5.0, 3.0)},
    )

    agent = snapshot["agents"]["employee_01"]
    assert agent["animation"] == "walk"
    assert agent["target"] == "meeting_table"
    assert agent["position"] == [-4.0, 0.0, 0.0]
    assert snapshot["robot_tasks"] == []


def test_jsonl_snapshots_round_trip_and_render_to_2d_video(tmp_path: Path) -> None:
    input_path = tmp_path / "snapshots.jsonl"
    snapshot = {
        "schema_version": 1,
        "minute_of_day": 9 * 60,
        "agents": {
            "employee_01": {
                "position": [-3.0, -3.0, 0.0],
                "action": "work",
                "label": "Operations",
            }
        },
        "events": [
            {
                "time": 9 * 60,
                "event": "action_started",
                "agent_id": "employee_01",
                "details": {"action": "work"},
            }
        ],
    }
    with JsonlSnapshotWriter(input_path) as writer:
        writer.write(snapshot)

    assert list(read_snapshots(input_path)) == [snapshot]
    assert event_lines(snapshot) == ["[09:00] employee_01: action_started work"]
    assert annotate_3d_frame(np.zeros((80, 160, 3), dtype="uint8"), snapshot).shape == (
        80,
        160,
        3,
    )
    assert annotate_3d_frame(np.zeros((80, 160, 3), dtype="uint8"), snapshot).sum() > 0

    output_path = tmp_path / "office.mp4"
    assert render_topdown_video(read_snapshots(input_path), output_path, fps=5) == 1
    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_native_multi_npc_scene_compiles_with_two_poseable_employees(tmp_path: Path) -> None:
    import mujoco

    scene_path = build_native_multi_npc_scene(tmp_path / "native_office_multi_npc.xml")
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "employee_01_body") >= 0
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "employee_02_body") >= 0
    assert any(
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) == "em02_frame_walk_00_body"
        for geom_id in range(model.ngeom)
    )
    material_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MATERIAL, "humanoid_employee")
    employee_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "em01_frame_idle_00_body")
    assert model.geom_matid[employee_geom_id] == material_id


def test_3d_video_transcode_uses_browser_compatible_h264(tmp_path: Path, monkeypatch) -> None:
    commands: list[list[str]] = []

    def fake_run(command: list[str], *, check: bool) -> SimpleNamespace:
        commands.append(command)
        assert check is True
        return SimpleNamespace()

    monkeypatch.setattr("stretch_mujoco.recording.renderers.subprocess.run", fake_run)
    _transcode_h264(tmp_path / "intermediate.mp4", tmp_path / "preview.mp4")

    assert commands == [
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(tmp_path / "intermediate.mp4"),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            "-an",
            str(tmp_path / "preview.mp4"),
        ]
    ]
