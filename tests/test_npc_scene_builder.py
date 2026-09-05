from pathlib import Path

import mujoco

from stretch_mujoco.npc.scene_builder import build_npc_scene

MODELS = Path(__file__).resolve().parents[1] / "stretch_mujoco" / "models"


def test_scene_builder_outputs_compilable_two_npc_include(tmp_path: Path) -> None:
    output = build_npc_scene(MODELS / "office_population.json", tmp_path / "npcs.xml")
    model = mujoco.MjModel.from_xml_path(str(output))

    assert model.nmocap == 2
    assert model.body("npc__employee_01").mocapid[0] != model.body("npc__employee_02").mocapid[0]
    assert model.geom("npc__employee_01__clip__idle__frame__000__slot__body").id >= 0
