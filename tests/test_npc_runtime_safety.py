from __future__ import annotations

from stretch_mujoco.npc.collision_guard import CollisionGuard
from stretch_mujoco.npc.journal import JsonlCommandJournal
from stretch_mujoco.npc.protocol import NpcCommand, NpcCommandKind
from stretch_mujoco.npc.traffic import TrafficManager


def test_traffic_reservation_is_idempotent_and_released() -> None:
    traffic = TrafficManager("fixture")
    first = traffic.acquire("hallway", "alex", 1)
    assert first is not None
    assert traffic.acquire("hallway", "alex", 1) == first
    assert traffic.acquire("hallway", "jordan", 1) is None
    assert traffic.release("alex", "hallway", 1)
    assert traffic.acquire("hallway", "jordan", 1) is not None


def test_collision_guard_rejects_swept_path_but_not_clear_path() -> None:
    guard = CollisionGuard(radius=0.2, clearance=0.05)
    assert guard.check_swept((0.0, 0.0), (1.0, 0.0), ((0.5, 0.0),)).reason == "collision_predicted"
    assert guard.check_swept((0.0, 0.0), (1.0, 0.0), ((0.5, 1.0),)).allowed


def test_jsonl_command_journal_retains_payload_hash_and_terminal_receipt(tmp_path) -> None:
    journal = JsonlCommandJournal(tmp_path / "commands.jsonl", execution_epoch=4)
    command = NpcCommand("move-1", 2, "alex", NpcCommandKind.MOVE_TO, {"site": "desk"}, 1.0)
    accepted = journal.append_command(command)
    terminal = journal.append_command(command, event="terminal", receipt={"status": "succeeded"})
    records = journal.records()
    assert (accepted.event, terminal.event) == ("accepted", "terminal")
    assert len(records) == 2
    assert records[0].payload_hash == records[1].payload_hash
    assert records[1].receipt == {"status": "succeeded"}
    assert records[0].execution_epoch == 4
