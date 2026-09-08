import json
from pathlib import Path

from examples.npc_conversation_acceptance import build_deterministic_conversation_recording
from stretch_mujoco.recording.snapshots import read_snapshots


def test_deterministic_conversation_recording_covers_social_and_robot_contracts(
    tmp_path: Path,
) -> None:
    artifacts = build_deterministic_conversation_recording(tmp_path)

    assert artifacts.snapshots >= 14
    assert artifacts.scene_path.exists()
    assert artifacts.snapshot_path.exists()
    assert artifacts.manifest_path.exists()
    assert artifacts.report_path.exists()
    # NPC-NPC lifecycle events are written once per participating NPC; the
    # robot session has one runtime-owned NPC participant.
    assert artifacts.event_counts["conversation_started"] == 9
    assert artifacts.event_counts["conversation_completed"] == 9
    assert artifacts.event_counts["robot_task_completed"] == 1
    assert artifacts.event_counts["robot_handover_receipt"] == 1

    snapshots = list(read_snapshots(artifacts.snapshot_path))
    assert {"employee_01", "employee_02", "employee_03"} == set(snapshots[0]["agents"])
    turn_intents = {
        event["details"]["intent"]
        for snapshot in snapshots
        for event in snapshot["events"]
        if event["event"] == "conversation_turn"
    }
    assert turn_intents == {
        "greeting",
        "progress_inquiry",
        "meeting_invitation",
        "conflict_resolution",
        "request",
        "clarify",
        "acknowledge",
        "handover_confirm",
    }
    assert any(
        agent["action"] == "conversation"
        for snapshot in snapshots
        for agent in snapshot["agents"].values()
    )
    report = json.loads(artifacts.report_path.read_text(encoding="utf-8"))
    assert report["deterministic"] is True
    assert report["scenarios"][-1] == "robot_request_clarify_acknowledge_handover_confirm"
