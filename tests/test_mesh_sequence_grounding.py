import mujoco
from stretch_mujoco.npc.animation.mesh_sequence import MeshSequenceBackend
from stretch_mujoco.npc.binding import NpcBinding


def test_stand_up_plays_the_registered_sit_sequence_in_reverse() -> None:
    model = mujoco.MjModel.from_xml_string(
        """
        <mujoco>
          <asset>
            <mesh name="sit_00" vertex="0 0 0  1 0 0  0 1 0  0 0 1" face="0 1 2 0 1 3 0 2 3 1 2 3"/>
            <mesh name="sit_01" vertex="0 0 0  2 0 0  0 2 0  0 0 2" face="0 1 2 0 1 3 0 2 3 1 2 3"/>
            <mesh name="stand_00" vertex="0 0 0  3 0 0  0 3 0  0 0 3" face="0 1 2 0 1 3 0 2 3 1 2 3"/>
          </asset>
          <worldbody><body name="npc__demo" mocap="true">
            <geom name="npc__demo__clip__sit__frame__000__slot__body" type="mesh" mesh="sit_00" rgba="1 1 1 0"/>
            <geom name="npc__demo__clip__sit__frame__001__slot__body" type="mesh" mesh="sit_01" rgba="1 1 1 0"/>
            <geom name="npc__demo__clip__stand_up__frame__000__slot__body" type="mesh" mesh="stand_00" rgba="1 1 1 0"/>
          </body></worldbody>
        </mujoco>
        """
    )
    backend = MeshSequenceBackend(model, NpcBinding.from_model(model, "demo"))
    sit_first = model.geom("npc__demo__clip__sit__frame__000__slot__body").id
    sit_last = model.geom("npc__demo__clip__sit__frame__001__slot__body").id
    original_stand_up = model.geom("npc__demo__clip__stand_up__frame__000__slot__body").id

    backend.sample("stand_up", 0.0)
    assert model.geom_rgba[sit_last, 3] == 1.0
    assert model.geom_rgba[sit_first, 3] == 0.0
    assert model.geom_rgba[original_stand_up, 3] == 0.0

    backend.sample("stand_up", 1.0)
    assert model.geom_rgba[sit_first, 3] == 1.0
    assert model.geom_rgba[sit_last, 3] == 0.0


def test_seated_idle_freezes_the_terminal_sit_frame() -> None:
    model = mujoco.MjModel.from_xml_string(
        """
        <mujoco>
          <asset>
            <mesh name="sit_00" vertex="0 0 0  1 0 0  0 1 0  0 0 1" face="0 1 2 0 1 3 0 2 3 1 2 3"/>
            <mesh name="sit_01" vertex="0 0 0  2 0 0  0 2 0  0 0 2" face="0 1 2 0 1 3 0 2 3 1 2 3"/>
            <mesh name="idle_00" vertex="0 0 0  3 0 0  0 3 0  0 0 3" face="0 1 2 0 1 3 0 2 3 1 2 3"/>
          </asset>
          <worldbody><body name="npc__demo" mocap="true">
            <geom name="npc__demo__clip__sit__frame__000__slot__body" type="mesh" mesh="sit_00" rgba="1 1 1 0"/>
            <geom name="npc__demo__clip__sit__frame__001__slot__body" type="mesh" mesh="sit_01" rgba="1 1 1 0"/>
            <geom name="npc__demo__clip__idle__frame__000__slot__body" type="mesh" mesh="idle_00" rgba="1 1 1 0"/>
          </body></worldbody>
        </mujoco>
        """
    )
    backend = MeshSequenceBackend(model, NpcBinding.from_model(model, "demo"))
    sit_last = model.geom("npc__demo__clip__sit__frame__001__slot__body").id
    standing_idle = model.geom("npc__demo__clip__idle__frame__000__slot__body").id

    backend.sample("seated_idle", 0.0)

    assert "seated_idle" in backend.available_clips
    assert backend.frame_count("seated_idle") == 1
    assert model.geom_rgba[sit_last, 3] == 1.0
    assert model.geom_rgba[standing_idle, 3] == 0.0
