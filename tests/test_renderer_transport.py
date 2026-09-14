from types import SimpleNamespace

import pytest

from stretch_mujoco.agents.renderer_transport import ServerBackedRendererTransport
from stretch_mujoco.npc import NpcCommand, NpcCommandKind


class _Server:
    def __init__(self):
        self.status = SimpleNamespace(time=4.0)
        self.calls = []

    def pull_status(self):
        return self.status

    def submit_npc_command(self, command):
        self.calls.append(command.command_id)
        return command.command_id

    def pull_npc_receipts(self):
        return ()

    def pull_npc_states(self): return {}

    def cancel_npc_command(self, *_args): pass

    def pull_semantic_state(self): return {"objects": {}}


def test_renderer_transport_exposes_only_npc_and_observation_boundary():
    server = _Server()
    transport = ServerBackedRendererTransport(server)
    command = NpcCommand("c1", 0, "npc", NpcCommandKind.PLAY_ANIMATION, {"clip": "idle"}, 4.0)
    assert transport.pull_status() is server.status
    assert transport.submit_npc_command(command) == "c1"
    assert not hasattr(transport, "solve_grasp_ik")
    assert not hasattr(transport, "plan_base_path")


def test_renderer_transport_rejects_incomplete_npc_transport():
    with pytest.raises(TypeError, match="live StretchMujocoSimulator"):
        ServerBackedRendererTransport(SimpleNamespace(pull_status=lambda: None))
