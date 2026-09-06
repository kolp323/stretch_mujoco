import pytest

from stretch_mujoco.npc import NpcCommand, NpcCommandKind


def test_command_round_trip_preserves_ordering_contract() -> None:
    command = NpcCommand("cmd-1", 4, "employee_01", NpcCommandKind.MOVE_TO, {"site": "a"}, 1.0, 2.0)

    assert NpcCommand.from_dict(command.to_dict()) == command


def test_command_rejects_deadline_before_issue_time() -> None:
    with pytest.raises(ValueError, match="deadline"):
        NpcCommand("cmd-1", 0, "employee_01", NpcCommandKind.MOVE_TO, {}, 2.0, 1.0)
