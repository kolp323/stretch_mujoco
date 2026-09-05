from pathlib import Path

from stretch_mujoco.agents.mp4_recorder import OfficeMp4Recorder


def test_recorder_renders_status_cards_and_robot_task(tmp_path: Path) -> None:
    output_path = tmp_path / "inspection.mp4"
    recorder = OfficeMp4Recorder(output_path, fps=5)
    recorder.start()
    recorder.record_frame(
        9 * 60,
        {
            "employee_01": {
                "position": (-3.0, -3.0),
                "action": "move_to",
                "label": "NPC 01",
                "role": "Operations Specialist",
                "location": "workstation_left",
                "target": "meeting_table",
                "target_position": (-5.5, 3.0),
            }
        },
        events=["09:00  NPC 01: walking started"],
        robot_tasks=[
            {"status": "running", "object": "document_report", "destination": "meeting_table"}
        ],
    )
    recorder.close()

    assert recorder._frame_count == 1
    assert output_path.exists()
    assert output_path.stat().st_size > 0
