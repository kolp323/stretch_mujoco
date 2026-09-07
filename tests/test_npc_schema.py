import json
import hashlib
from pathlib import Path

import pytest

from stretch_mujoco.npc.schema import NpcAppearance, NpcPopulation

MODELS = Path(__file__).resolve().parents[1] / "stretch_mujoco" / "models"


def test_example_population_is_strict_v2() -> None:
    population = NpcPopulation.from_json(MODELS / "office_population.json")

    assert tuple(population.npcs) == ("employee_01", "employee_02")
    assert population.npcs["employee_01"].spawn.site == "desk_left_work_site"
    assert population.npcs["employee_01"].capabilities == frozenset(
        {"locomotion", "sit", "conversation", "use_computer"}
    )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda payload: payload.update(schema_version=99), "schema_version"),
        (
            lambda payload: payload["npcs"]["employee_01"]["embodiment"].update(scale=0),
            "scale",
        ),
        (
            lambda payload: payload["npcs"]["employee_01"].update(capabilities=["teleport"]),
            "unknown capabilities",
        ),
    ],
)
def test_population_rejects_invalid_contract(mutation, message: str) -> None:
    payload = json.loads((MODELS / "office_population.json").read_text())
    mutation(payload)

    with pytest.raises(ValueError, match=message):
        NpcPopulation.from_dict(payload)


def test_population_validates_semantic_location_and_site() -> None:
    payload = json.loads((MODELS / "office_population.json").read_text())

    with pytest.raises(ValueError, match="unknown spawn location"):
        NpcPopulation.from_dict(payload, locations={"workstation_right"})
    with pytest.raises(ValueError, match="unknown spawn site"):
        NpcPopulation.from_dict(
            payload,
            locations={"workstation_left", "workstation_right"},
            sites={"desk_right_work_site"},
        )


def test_population_validates_visual_identity_against_catalog(tmp_path: Path) -> None:
    base = tmp_path / "base.png"
    layer = tmp_path / "hair.png"
    base.write_bytes(b"base")
    layer.write_bytes(b"hair")
    catalog = {
        "schema_version": 1,
        "texture_topology_id": "topology-v1",
        "base": "base.png",
        "base_sha256": hashlib.sha256(base.read_bytes()).hexdigest(),
        "layers": {
            "hair_v1": {
                "category": "hair",
                "image": "hair.png",
                "sha256": hashlib.sha256(layer.read_bytes()).hexdigest(),
            }
        },
        "identities": {"alex_v1": {"appearance_id": "alex_appearance_v1", "layers": ["hair_v1"]}},
    }
    (tmp_path / "catalog.json").write_text(json.dumps(catalog))
    payload = json.loads((MODELS / "office_population.json").read_text())
    payload["appearance_catalog"] = "catalog.json"
    embodiment = payload["npcs"]["employee_01"]["embodiment"]
    embodiment["appearance"] = "alex_appearance_v1"
    embodiment["visual_identity"] = "alex_v1"
    payload["npcs"]["employee_02"]["embodiment"]["appearance"] = "alex_appearance_v1"
    payload["npcs"]["employee_02"]["embodiment"]["visual_identity"] = "alex_v1"
    population_path = tmp_path / "population.json"
    population_path.write_text(json.dumps(payload))

    population = NpcPopulation.from_json(population_path)

    assert population.npcs["employee_01"].embodiment.visual_identity == "alex_v1"
    payload["npcs"]["employee_01"]["embodiment"]["appearance"] = "wrong_v1"
    population_path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="produces appearance"):
        NpcPopulation.from_json(population_path)


def test_explicit_appearance_slots_bind_to_npc_agent() -> None:
    appearance = NpcAppearance.from_dict(
        {
            "skin": "skin_warm_v1",
            "hair": "hair_short_brown_v1",
            "top": "top_teal_v1",
            "bottom": "bottom_charcoal_v1",
            "shoes": "shoes_black_v1",
            "accessories": ["cap_simple_v1"],
            "scale": 1.0,
        }
    )
    payload = json.loads((MODELS / "office_population.json").read_text())
    npc = payload["npcs"]["employee_01"]
    npc["agent_id"] = "agent_alex"
    npc["embodiment"]["appearance_config"] = {
        "skin": appearance.skin,
        "hair": appearance.hair,
        "top": appearance.top,
        "bottom": appearance.bottom,
        "shoes": appearance.shoes,
        "accessories": list(appearance.accessories),
        "scale": appearance.scale,
    }
    npc["embodiment"]["accessories"] = list(appearance.accessories)
    npc["embodiment"]["scale"] = appearance.scale

    population = NpcPopulation.from_dict(payload)

    definition = population.npcs["employee_01"]
    assert definition.agent_id == "agent_alex"
    assert definition.embodiment.appearance_config == appearance
