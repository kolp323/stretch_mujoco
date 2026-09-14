from pathlib import Path

import mujoco
import numpy as np

from stretch_mujoco.npc.scene_builder import build_npc_scene

MODELS = Path(__file__).resolve().parents[1] / "stretch_mujoco" / "models"


def test_scene_builder_outputs_compilable_two_npc_include(tmp_path: Path) -> None:
    output = build_npc_scene(MODELS / "office_population.json", tmp_path / "npcs.xml")
    model = mujoco.MjModel.from_xml_path(str(output))

    assert model.nmocap == 2
    assert model.body("npc__employee_01").mocapid[0] != model.body("npc__employee_02").mocapid[0]
    assert model.geom("npc__employee_01__clip__idle__frame__000__slot__body").id >= 0


def test_production_roster_spawns_all_npcs_at_separate_collision_free_positions(
    tmp_path: Path,
) -> None:
    output = build_npc_scene(
        MODELS / "office_population.production.example.json", tmp_path / "production_npcs.xml"
    )
    model = mujoco.MjModel.from_xml_path(str(output))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    npc_ids = (
        "npc_alex_chen",
        "npc_morgan_lee",
        "npc_jordan_patell",
        "npc_priya_narayanan",
        "npc_marco_silva",
        "npc_olivia_bennett",
        "npc_daniel_kim",
        "npc_samira_haddad",
        "npc_wei_zhang",
        "npc_lena_fischer",
    )
    positions = np.array(
        [data.mocap_pos[int(model.body(f"npc__{npc_id}").mocapid[0]), :2] for npc_id in npc_ids]
    )
    pairwise_distances = np.linalg.norm(positions[:, None] - positions[None, :], axis=-1)

    assert len({tuple(position) for position in positions}) == len(npc_ids)
    assert np.min(pairwise_distances[np.triu_indices(len(npc_ids), k=1)]) >= 0.5
    assert data.ncon == 0
    for npc_id in npc_ids:
        hand_site = model.site(f"npc__{npc_id}__handover")
        assert hand_site.id >= 0
        assert model.site(
            f"npc__{npc_id}__anchor__handover__clip__walk__frame__002"
        ).id >= 0
