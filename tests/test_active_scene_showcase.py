import json
from pathlib import Path

import pytest

from tools.render_active_scene_showcase import _display_label, load_scenario, validate_scenario


ROOT = Path(__file__).resolve().parents[1]
SCENARIOS = ROOT / "aaa_workspace/showcases"


@pytest.mark.parametrize("name", ["office_02_day_in_the_life", "home_04_household_assistance"])
def test_showcase_scenario_validates_against_active_profile(name: str) -> None:
    catalog = json.loads((ROOT / "stretch_mujoco/models/generated_scene_npc/active/active_catalog.json").read_text())
    scenario = load_scenario(SCENARIOS / f"{name}.json")
    record = catalog["scenes"][scenario["scene_id"]]
    build = ROOT / "stretch_mujoco/models/generated_scene_npc/active" / catalog["build_root"]
    result = validate_scenario(scenario, build / record["trajectory_profile"], build / record["population"])
    assert all(item["status"] == "declared" for item in result["phase_receipts"])
    assert all(item.get("reason") == "awaiting_runtime_execution" for item in result["phase_receipts"])


def test_partial_scenario_cannot_be_reported_as_showcase_success() -> None:
    scenario = load_scenario(SCENARIOS / "home_04_household_assistance.json")
    statuses = ["executed", *["not_executed"] * (len(scenario["phases"]) - 1)]
    assert any(status == "not_executed" for status in statuses)
    assert not all(status == "executed" for status in statuses)


def test_showcase_rejects_unknown_route(tmp_path: Path) -> None:
    scenario = json.loads((SCENARIOS / "office_02_day_in_the_life.json").read_text())
    scenario["phases"][0]["route"] = "through_the_sofa"
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(scenario))
    loaded = load_scenario(path)
    catalog = json.loads((ROOT / "stretch_mujoco/models/generated_scene_npc/active/active_catalog.json").read_text())
    record = catalog["scenes"][loaded["scene_id"]]
    build = ROOT / "stretch_mujoco/models/generated_scene_npc/active" / catalog["build_root"]
    with pytest.raises(ValueError, match="showcase_route_action_invalid"):
        validate_scenario(loaded, build / record["trajectory_profile"], build / record["population"])


def test_runtime_phase_label_is_one_based_and_human_readable() -> None:
    assert _display_label("0:move:npc_jordan_patell") == "Phase 1 | Move | Jordan Patell"
    assert _display_label("2:conversation") == "Phase 3 | Conversation"
