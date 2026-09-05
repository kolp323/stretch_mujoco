import mujoco
import numpy as np

from stretch_mujoco.npc import CommandStatus, NpcCommand, NpcCommandKind
from stretch_mujoco.npc.system import NpcSystem

MODEL_XML = """
<mujoco>
  <worldbody>
    <body name="npc__employee_01" mocap="true" pos="0 0 0">
      <geom name="npc__employee_01__clip__idle__frame__000__slot__body" type="sphere" size=".1"/>
      <geom name="npc__employee_01__clip__walk__frame__000__slot__body" type="sphere" size=".1" rgba="1 1 1 0"/>
    </body>
    <body name="npc__employee_02" mocap="true" pos="0 1 0">
      <geom name="npc__employee_02__clip__idle__frame__000__slot__body" type="sphere" size=".1"/>
      <geom name="npc__employee_02__clip__walk__frame__000__slot__body" type="sphere" size=".1" rgba="1 1 1 0"/>
    </body>
    <site name="target_left" pos="-1 0 0"/>
    <site name="target_right" pos="1 1 0"/>
  </worldbody>
</mujoco>
"""


def command(command_id: str, sequence: int, npc_id: str, site: str) -> NpcCommand:
    return NpcCommand(command_id, sequence, npc_id, NpcCommandKind.MOVE_TO, {"site": site}, 0.0)


def test_two_npcs_move_and_animate_independently() -> None:
    model = mujoco.MjModel.from_xml_string(MODEL_XML)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    system = NpcSystem.from_model(model)
    first = command("left", 0, "employee_01", "target_left")
    second = command("right", 0, "employee_02", "target_right")

    assert system.submit(first).status == CommandStatus.ACCEPTED
    assert system.submit(second).status == CommandStatus.ACCEPTED
    for step in range(30):
        system.step(model, data, step * 0.1)

    states = system.states(data)
    np.testing.assert_allclose(states["employee_01"].position[:2], (-1.0, 0.0))
    np.testing.assert_allclose(states["employee_02"].position[:2], (1.0, 1.0))
    assert states["employee_01"].last_receipt.status == CommandStatus.SUCCEEDED
    assert states["employee_02"].last_receipt.status == CommandStatus.SUCCEEDED


def test_duplicate_and_stale_commands_do_not_execute_twice() -> None:
    model = mujoco.MjModel.from_xml_string(MODEL_XML)
    system = NpcSystem.from_model(model)
    original = command("move", 2, "employee_01", "target_left")

    accepted = system.submit(original)
    assert system.submit(original) == accepted
    stale = command("stale", 1, "employee_01", "target_right")
    assert system.submit(stale).reason == "stale_sequence"
