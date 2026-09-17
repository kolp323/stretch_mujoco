import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from stretch_mujoco.npc.schema import NpcPopulation
from stretch_mujoco.npc.scene_site_config import load_scene_site_plans
from stretch_mujoco.semantics import SemanticWorld


ROOT = Path(__file__).resolve().parents[1]
SCENES = ROOT / "stretch_mujoco" / "models" / "assets" / "home_scenes"
GENERATED = ROOT / "stretch_mujoco" / "models" / "generated_home_npc"
SITE_CONFIG = SCENES / "npc_slot_overrides.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_ten_home_npc_artifacts_project_the_named_site_configuration() -> None:
    catalog = json.loads((GENERATED / "catalog.json").read_text(encoding="utf-8"))
    plans = load_scene_site_plans(SITE_CONFIG)
    assert catalog["scene_count"] == 10
    assert len(catalog["scenes"]) == 10
    assert set(plans) == {Path(entry["scene"]).stem.removesuffix("_npc") for entry in catalog["scenes"]}

    for entry in catalog["scenes"]:
        stem = Path(entry["scene"]).stem
        scene = SCENES / entry["scene"]
        population_path = GENERATED / entry["population"]
        semantic_path = GENERATED / entry["semantic_world"]
        receipt_path = GENERATED / entry["receipt"]
        assert scene.is_file()
        plan = plans[stem.removesuffix("_npc")]
        population = NpcPopulation.from_json(population_path)
        assert set(population.npcs) == {entry.npc_id for entry in plan.roster}
        assert {
            npc_id: definition.spawn.site for npc_id, definition in population.npcs.items()
        } == {
            entry.npc_id: entry.site for entry in plan.roster
        }
        world = SemanticWorld.from_json(semantic_path)
        spawn_points = [
            point
            for point in world.interaction_points_for(owner="home")
            if point.attributes.get("usage") == "spawn"
        ]
        assert {point.site for point in spawn_points} == {entry.site for entry in plan.roster}
        if plan.demo is not None:
            assert world.interaction_points["home_activity"].site == plan.demo.activity_site

        xml_root = ET.parse(scene).getroot()
        names = {node.get("name") for node in xml_root.findall(".//site")}
        assert set(plan.sites) <= names
        assert xml_root.find(".//geom[@name='office_floor']") is not None
        assert xml_root.find(".//camera[@name='office_overview']") is not None
        bed_objects = [obj for obj in world.objects.values() if obj.object_type.value == "Bed"]
        bed_sit_sites = {
            point.site
            for point in world.interaction_points_for(role="chair_sit_site")
            if point.owner in {obj.object_id for obj in bed_objects}
        }
        assert bed_sit_sites <= names
        for site in bed_sit_sites:
            index = site.removeprefix("npc_home_bed_").removesuffix("_sit_site")
            assert f"npc_home_bed_{index}_approach_site" in names
        for bed in bed_objects:
            sit_point = next(
                point
                for point in world.interaction_points_for(role="chair_sit_site")
                if point.owner == bed.object_id
            )
            approach_point = next(
                point
                for point in world.interaction_points_for(owner=bed.object_id)
                if point.attributes.get("binding") == "seat_navigation"
            )
            assert sit_point.attributes["seat_type"] == "bed"
            assert sit_point.site != approach_point.site

        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        assert receipt["scene"] == scene.name
        assert receipt["sha256"]["generated_scene"] == _sha256(scene)
        assert receipt["sha256"]["population"] == _sha256(population_path)
        assert receipt["sha256"]["semantic"] == _sha256(semantic_path)


def test_home_npc_population_paths_are_portable() -> None:
    for population_path in sorted((GENERATED / "populations").glob("*.json")):
        population = NpcPopulation.from_json(population_path)
        assert not Path(population.scene).is_absolute()
        assert not Path(population.asset_manifest).is_absolute()
        assert population.resolve_path(population.scene).is_file()


def test_named_site_plan_allows_nonsequential_names_and_a_variable_roster(tmp_path: Path) -> None:
    config_path = tmp_path / "scene_sites.json"
    config_path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "scenes": {
                    "home_custom": {
                        "sites": {
                            "reading_nook": {"position": [1.0, 2.0], "tags": ["activity"]},
                            "guest_entry": {"position": [3.0, 4.0, 0.1], "yaw": 1.2},
                        },
                        "roster": [
                            {"npc_id": "npc_guest", "site": "guest_entry", "location": "lobby"}
                        ],
                        "demo": {
                            "activity_site": "reading_nook",
                            "walker_npc_id": "npc_guest",
                            "conversation_partner_npc_id": "npc_guest",
                        },
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unknown site or roster NPC"):
        load_scene_site_plans(config_path)

    payload = json.loads(config_path.read_text(encoding="utf-8"))
    payload["scenes"]["home_custom"].pop("demo")
    config_path.write_text(json.dumps(payload), encoding="utf-8")
    plan = load_scene_site_plans(config_path)["home_custom"]
    assert set(plan.sites) == {"reading_nook", "guest_entry"}
    assert plan.sites["reading_nook"].position == (1.0, 2.0, 0.025)
    assert plan.roster == (plan.roster[0],)
    assert plan.roster[0].npc_id == "npc_guest"
    assert plan.roster[0].site == "guest_entry"
