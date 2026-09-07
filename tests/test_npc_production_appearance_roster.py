from pathlib import Path

from stretch_mujoco.npc.schema import NpcPopulation


MODELS = Path(__file__).parents[1] / "stretch_mujoco/models"
PRODUCTION_POPULATION = MODELS / "office_population.production.example.json"


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
