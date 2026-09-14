import json
from pathlib import Path

from stretch_mujoco.npc.schema import NpcPopulation


MODELS = Path(__file__).parents[1] / "stretch_mujoco/models"
PRODUCTION_POPULATION = MODELS / "office_population.production.example.json"
TEXTILE_SPEC = MODELS / "appearance_recipes/office_personas_v1.textile_layers.json"
ROSTER = MODELS / "appearance_recipes/office_personas_v1.roster.json"


def test_production_roster_has_ten_named_npcs_with_explicit_distinct_appearances() -> None:
    population = NpcPopulation.from_json(PRODUCTION_POPULATION)

    assert tuple(population.npcs)[:2] == ("npc_alex_chen", "npc_morgan_lee")
    assert len(population.npcs) == 10
    assert len({definition.agent_id for definition in population.npcs.values()}) == 10
    assert len({definition.embodiment.appearance for definition in population.npcs.values()}) == 10
    assert all(
        definition.embodiment.appearance_config is not None
        for definition in population.npcs.values()
    )
    assert all(
        definition.embodiment.scale == definition.embodiment.appearance_config.scale
        for definition in population.npcs.values()
        if definition.embodiment.appearance_config is not None
    )
    assert population.trajectory_profile == "../npc/trajectory_profiles/office_v1.json"


def test_production_roster_assigns_distinct_top_layers_to_npcs() -> None:
    population = NpcPopulation.from_json(PRODUCTION_POPULATION)

    assert population.npcs["npc_jordan_patell"].embodiment.appearance_config.top == (
        "top_poly_wool_herringbone_v1"
    )
    assert population.npcs["npc_olivia_bennett"].embodiment.appearance_config.top == (
        "top_cotton_jersey_v1"
    )
    assert population.npcs["npc_wei_zhang"].embodiment.appearance_config.top == (
        "top_stretch_poplin_v1"
    )


def test_new_textile_sources_are_registered_as_top_appearance_elements() -> None:
    textile_spec = json.loads(TEXTILE_SPEC.read_text(encoding="utf-8"))
    roster = json.loads(ROSTER.read_text(encoding="utf-8"))
    registered = {entry["id"] for entry in roster["layers"]}
    source_ids = {
        "bi_stretch",
        "caban",
        "cotton_jersey",
        "crepe_georgette",
        "denim_fabric_04",
        "denim_fabric_06",
        "fabric_leather_02",
        "poly_wool_herringbone",
        "rough_linen",
        "stretch_poplin",
        "velour_velvet",
    }

    for source_id in source_ids:
        layer_id = f"top_{source_id}_v1"
        layer = next(layer for layer in textile_spec["layers"] if layer["id"] == layer_id)
        assert layer_id in registered
        assert layer["source_archive"] == (
            f"../../sources/npc/textures/{source_id}/{source_id}_1k.zip"
        )
        assert layer["archive_member"] == f"textures/{source_id}_diff_1k.jpg"
        assert layer["license"] == "CC0-1.0"
