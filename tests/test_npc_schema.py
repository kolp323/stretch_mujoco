import json
from pathlib import Path

import pytest

from stretch_mujoco.npc.schema import NpcPopulation

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
