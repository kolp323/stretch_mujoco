from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import pytest

from stretch_mujoco.npc.interaction_projection import PHYSICAL_VALIDATORS
from stretch_mujoco.npc.interaction_station import InteractionSitePlanError
from stretch_mujoco.npc.scene_compiler import compile_scene_npc_config
from stretch_mujoco.npc.schema import NpcPopulation
from stretch_mujoco.npc.trajectory_profile import NpcTrajectoryProfile
from stretch_mujoco.semantics import (
    InteractionRole,
    ObjectType,
    RelationType,
    SemanticRelation,
    SemanticWorld,
)


ROOT = Path(__file__).resolve().parents[1]
OFFICE_CONFIG = (
    ROOT / "stretch_mujoco/models/scene_npc_configs/office/office_02_cross_axis.json"
)
OFFICE_PLAN = (
    ROOT
    / "stretch_mujoco/models/scene_interaction_plans/office/office_02_cross_axis.json"
)
FIXTURES = ROOT / "stretch_mujoco/models/scene_npc_configs/fixtures"
ACTIVE_CATALOG = (
    ROOT / "stretch_mujoco/models/generated_scene_npc/active/active_catalog.json"
)
ACTIVE_CATALOG_SHA256 = "bf1d638c6ae56305f84c0780b2f3c5bdc3dd2eb23c46d9b7cfe326d029871365"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _v2_config(
    tmp_path: Path,
    *,
    plan_path: Path = OFFICE_PLAN,
    manifest_path: Path | None = None,
    mutate=None,
) -> Path:
    payload = json.loads(OFFICE_CONFIG.read_text(encoding="utf-8"))
    payload["schema"] = "scene_npc_config/v2"
    payload.pop("interactions")
    for key in ("source_mjcf", "source_manifest", "semantic_policy", "npc_catalog"):
        payload["scene"][key] = str(
            (OFFICE_CONFIG.parent / payload["scene"][key]).resolve()
        )
    if manifest_path is not None:
        payload["scene"]["source_manifest"] = str(manifest_path)
    payload["interaction_plan"] = {
        "path": str(plan_path),
        "required_capabilities": ["conversation", "handover", "sit", "work"],
    }
    payload["robot_navigation"] = {
        "footprint_radius": 0.32,
        "clearance": 0.08,
        "resolution": 0.08,
    }
    if mutate is not None:
        mutate(payload)
    path = tmp_path / "office_02_candidate.json"
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return path


def _v1_config(tmp_path: Path) -> Path:
    payload = json.loads((FIXTURES / "minimal_scene.json").read_text(encoding="utf-8"))
    payload["scene"]["source_mjcf"] = str(FIXTURES / "minimal_scene.xml")
    payload["scene"]["source_manifest"] = str(FIXTURES / "minimal_scene.manifest.json")
    payload["scene"]["semantic_policy"] = str(
        ROOT / "stretch_mujoco/models/semantic_policies/fixture.json"
    )
    payload["scene"]["npc_catalog"] = str(
        ROOT / "stretch_mujoco/models/office_population.production.example.json"
    )
    path = tmp_path / "legacy_v1.json"
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return path


def _mutated_plan(tmp_path: Path, name: str, mutate) -> Path:
    payload = json.loads(OFFICE_PLAN.read_text(encoding="utf-8"))
    mutate(payload)
    path = tmp_path / name
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return path


def test_v2_compiler_emits_all_transparent_interaction_sites_and_population_v3(
    tmp_path: Path,
) -> None:
    outputs = compile_scene_npc_config(_v2_config(tmp_path), tmp_path / "candidate.xml")
    assert set(outputs) == {
        "scene",
        "semantic_v2",
        "semantic_v1",
        "coverage",
        "population",
        "trajectory_profile",
        "receipt",
    }
    root = ET.parse(outputs["scene"]).getroot()
    sites = {
        site.get("name"): site
        for site in root.findall(".//site")
        if site.get("name", "").startswith("interaction__office_02_cross_axis__")
    }
    assert len(sites) == 11
    assert {
        "interaction__office_02_cross_axis__conversation__meeting__01__speaker",
        "interaction__office_02_cross_axis__conversation__meeting__01__listener",
        "interaction__office_02_cross_axis__handover__work__01__giver",
        "interaction__office_02_cross_axis__handover__work__01__receiver",
        "interaction__office_02_cross_axis__handover__work__01__robot",
        "interaction__office_02_cross_axis__handover__work__01__transfer",
        "interaction__office_02_cross_axis__seat__object__asset_013_new_dini__01__ingress",
        "interaction__office_02_cross_axis__seat__object__asset_013_new_dini__01__sit",
        "interaction__office_02_cross_axis__seat__object__asset_011_new_dini__01__ingress",
        "interaction__office_02_cross_axis__seat__object__asset_011_new_dini__01__sit",
        "interaction__office_02_cross_axis__workstation__office_02__01__work",
    } == set(sites)
    assert all(site.get("rgba") == "0 0 0 0" for site in sites.values())

    population = NpcPopulation.from_json(outputs["population"])
    assert population.schema_version == 3
    assert population.interaction_stations is not None
    assert set(population.interaction_stations["conversation"]) == {
        "conversation.meeting.01"
    }
    assert population.interaction_stations["handover"][
        "handover.work.01"
    ].attributes["transfer_site"].endswith("__transfer")

    payload = json.loads(outputs["semantic_v2"].read_text(encoding="utf-8"))
    workstation = payload["entities"]["object.new_teaming_table.8"]
    assert workstation["source"]["xml_binding"]["name"] == "asset_008_new_team"
    assert workstation["labels"]["source_category"] == "meeting_tables"
    assert workstation["points"]["navigation"]
    assert workstation["points"]["action"] == ["point.workstation.office_02.01.work"]
    world = SemanticWorld.from_json(outputs["semantic_v2"])
    assert world.objects["seat.object.asset_013_new_dini.01"].object_type == ObjectType.SEAT_SLOT
    assert world.interaction_points["point.handover.work.01.transfer"].role == (
        InteractionRole.HANDOVER_TRANSFER
    )
    assert world.interaction_points["point.workstation.office_02.01.work"].role == (
        InteractionRole.DESK_WORK
    )
    assert SemanticRelation(
        "object.new_imac_24.9",
        RelationType.ON,
        "object.new_teaming_table.8",
    ) in world.relations

    semantic_v1 = json.loads(outputs["semantic_v1"].read_text(encoding="utf-8"))
    assert semantic_v1["projection_scope"] == "base_compatibility_only"
    assert semantic_v1["interaction_station_support"] is False
    coverage = json.loads(outputs["coverage"].read_text(encoding="utf-8"))
    assert coverage["navigation_preflight"] == "passed"
    assert coverage["interaction_navigation_preflight"] == {
        "requested": True,
        "candidate_only": False,
        "npc": {"status": "passed", "target_count": 6, "route_count": 24},
        "robot": {"status": "passed", "target_count": 3, "route_count": 3},
    }
    trajectory = NpcTrajectoryProfile.from_json(outputs["trajectory_profile"])
    assert trajectory.scene == outputs["scene"].name


def test_v2_receipt_hashes_every_input_and_nonreceipt_output(tmp_path: Path) -> None:
    config_path = _v2_config(tmp_path)
    outputs = compile_scene_npc_config(config_path, tmp_path / "candidate.xml")
    receipt = json.loads(outputs["receipt"].read_text(encoding="utf-8"))
    config = json.loads(config_path.read_text(encoding="utf-8"))
    expected_inputs = {
        "scene_config": config_path,
        "interaction_plan": Path(config["interaction_plan"]["path"]),
        "source_mjcf": Path(config["scene"]["source_mjcf"]),
        "source_manifest": Path(config["scene"]["source_manifest"]),
        "semantic_policy": Path(config["scene"]["semantic_policy"]),
        "npc_catalog": Path(config["scene"]["npc_catalog"]),
    }
    assert receipt["sha256"]["inputs"] == {
        name: _sha256(path) for name, path in expected_inputs.items()
    }
    assert receipt["sha256"]["outputs"] == {
        name: _sha256(path) for name, path in outputs.items() if name != "receipt"
    }
    assert "receipt" not in receipt["sha256"]["outputs"]
    assert receipt["validated_for_runtime"] is False
    assert receipt["validators"]["source_bindings"] == {"status": "passed"}
    assert receipt["validators"]["npc_navigation"]["status"] == "passed"
    assert receipt["validators"]["robot_navigation"]["status"] == "passed"
    assert all(
        receipt["validators"][name] == {"status": "not_run"}
        for name in PHYSICAL_VALIDATORS
        if name not in {"npc_navigation", "robot_navigation"}
    )
    assert receipt["projection"]["validators"]["source_bindings"] == {
        "status": "not_run"
    }
    assert receipt["counts"]["derived_interaction_sites"] == 11
    assert receipt["semantic_v1"] == {
        "scope": "base_compatibility_only",
        "interaction_station_support": False,
    }


def test_v2_navigation_receipt_uses_real_meshes_and_unique_targets(
    tmp_path: Path,
) -> None:
    outputs = compile_scene_npc_config(_v2_config(tmp_path), tmp_path / "candidate.xml")
    model = mujoco.MjModel.from_xml_path(str(outputs["scene"]))
    assert model.nsite > 0

    population = NpcPopulation.from_json(outputs["population"])
    receipt = json.loads(outputs["receipt"].read_text(encoding="utf-8"))
    npc = receipt["validators"]["npc_navigation"]
    robot = receipt["validators"]["robot_navigation"]
    assert npc["status"] == robot["status"] == "passed"
    assert npc["footprint"] == {
        "radius": 0.16,
        "clearance": 0.06,
        "effective_radius": 0.22,
        "resolution": 0.08,
    }
    assert robot["footprint"] == {
        "radius": 0.32,
        "clearance": 0.08,
        "effective_radius": 0.4,
        "resolution": 0.08,
    }
    assert npc["primary_component"]["cell_count"] > 0
    assert robot["primary_component"]["cell_count"] > 0
    assert len(npc["targets"]) == len({target["site"] for target in npc["targets"]}) == 6
    assert len(npc["routes"]) == len(population.npcs) * len(npc["targets"]) == 24
    assert len(robot["routes"]) == len(robot["targets"]) == 3
    seat_target = next(
        target
        for target in npc["targets"]
        if target["point"] == "point.seat.object.asset_013_new_dini.01.ingress"
    )
    assert seat_target["usages"] == ["seat_ingress", "workstation_seat_ingress"]
    assert seat_target["consumers"] == [
        "seat.object.asset_013_new_dini.01",
        "workstation.office_02.01",
    ]
    assert all(route["path_length_m"] > 0 for route in [*npc["routes"], *robot["routes"]])


def test_v2_navigation_rejects_npc_target_outside_primary_component(
    tmp_path: Path,
) -> None:
    def move_conversation_outside(payload):
        roles = payload["conversation_stations"][0]["roles"]
        roles["speaker"]["position"] = [-5.1, 3.0, 0.025]
        roles["listener"]["position"] = [-5.9, 3.0, 0.025]

    plan = _mutated_plan(tmp_path, "npc-target-outside.json", move_conversation_outside)
    with pytest.raises(
        ValueError,
        match=r"^npc_navigation_target_unavailable:point\.conversation\.meeting\.01\.speaker$",
    ):
        compile_scene_npc_config(
            _v2_config(tmp_path, plan_path=plan), tmp_path / "candidate.xml"
        )


def test_v2_navigation_rejects_robot_only_footprint_failure(tmp_path: Path) -> None:
    baseline = compile_scene_npc_config(
        _v2_config(tmp_path), tmp_path / "baseline-candidate.xml"
    )
    baseline_receipt = json.loads(baseline["receipt"].read_text(encoding="utf-8"))
    assert baseline_receipt["validators"]["npc_navigation"]["status"] == "passed"

    def enlarge_robot(payload):
        payload["robot_navigation"]["footprint_radius"] = 5.5

    with pytest.raises(ValueError, match=r"^robot_navigation_mesh_unavailable$"):
        compile_scene_npc_config(
            _v2_config(tmp_path, mutate=enlarge_robot),
            tmp_path / "large-robot-candidate.xml",
        )


def test_v2_check_navigation_false_is_explicitly_candidate_only(tmp_path: Path) -> None:
    outputs = compile_scene_npc_config(
        _v2_config(tmp_path), tmp_path / "candidate.xml", check_navigation=False
    )
    receipt = json.loads(outputs["receipt"].read_text(encoding="utf-8"))
    disabled = {
        "status": "not_run",
        "reason": "check_navigation_false",
        "candidate_only": True,
    }
    assert receipt["validators"]["npc_navigation"] == disabled
    assert receipt["validators"]["robot_navigation"] == disabled
    coverage = json.loads(outputs["coverage"].read_text(encoding="utf-8"))
    assert coverage["navigation_preflight"] == "not_run"
    assert coverage["strongly_connected"] is False
    assert coverage["interaction_navigation_preflight"] == {
        "requested": False,
        "candidate_only": True,
        "npc": {"status": "not_run", "target_count": 0, "route_count": 0},
        "robot": {"status": "not_run", "target_count": 0, "route_count": 0},
    }
    assert receipt["validated_for_runtime"] is False


def test_v2_candidate_output_bytes_are_deterministic(tmp_path: Path) -> None:
    active_before = _sha256(ACTIVE_CATALOG)
    assert active_before == ACTIVE_CATALOG_SHA256
    config = _v2_config(tmp_path)
    first_outputs = compile_scene_npc_config(config, tmp_path / "candidate.xml")
    first = {name: path.read_bytes() for name, path in first_outputs.items()}
    second_outputs = compile_scene_npc_config(config, tmp_path / "candidate.xml")
    assert {name: path.read_bytes() for name, path in second_outputs.items()} == first
    assert _sha256(ACTIVE_CATALOG) == active_before


def test_v2_compiler_rejects_missing_required_plan_resource(tmp_path: Path) -> None:
    plan = json.loads(OFFICE_PLAN.read_text(encoding="utf-8"))
    plan["conversation_stations"] = []
    plan_path = tmp_path / "missing-conversation.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    with pytest.raises(ValueError, match="required_capability_missing:conversation"):
        compile_scene_npc_config(
            _v2_config(tmp_path, plan_path=plan_path), tmp_path / "candidate.xml"
        )


def test_v2_compiler_rejects_semantic_entity_id_conflict(tmp_path: Path) -> None:
    plan = json.loads(OFFICE_PLAN.read_text(encoding="utf-8"))
    plan["seat_slots"][0]["id"] = "object.asset_013_new_dini"
    plan["workstations"][0]["seat_slot"] = "object.asset_013_new_dini"
    plan_path = tmp_path / "entity-conflict.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    with pytest.raises(
        ValueError, match="interaction_entity_id_conflict:object.asset_013_new_dini"
    ):
        compile_scene_npc_config(
            _v2_config(tmp_path, plan_path=plan_path), tmp_path / "candidate.xml"
        )


def test_v2_compiler_rejects_conflicting_relation(tmp_path: Path) -> None:
    plan = json.loads(OFFICE_PLAN.read_text(encoding="utf-8"))
    second = copy.deepcopy(plan["workstations"][0])
    second["id"] = "workstation.office_02.02"
    second["workstation_entity"] = "object.asset_001_new_cb_d"
    second["seat_slot"] = "seat.object.asset_011_new_dini.01"
    second["work_site"]["position"][0] += 2.0
    plan["workstations"].append(second)
    plan_path = tmp_path / "relation-conflict.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    source_manifest = (
        OFFICE_CONFIG.parent
        / json.loads(OFFICE_CONFIG.read_text(encoding="utf-8"))["scene"][
            "source_manifest"
        ]
    ).resolve()
    manifest = json.loads(source_manifest.read_text(encoding="utf-8"))
    asset = next(item for item in manifest["assets"] if item["instance"] == 1)
    asset["position"] = [-5.5, 3.0, 0.0]
    changed_manifest = tmp_path / "relation-conflict-manifest.json"
    changed_manifest.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(
        ValueError,
        match="conflicting_semantic_relation:object.new_imac_24.9:ON",
    ):
        compile_scene_npc_config(
            _v2_config(
                tmp_path,
                plan_path=plan_path,
                manifest_path=changed_manifest,
            ),
            tmp_path / "candidate.xml",
        )


def test_v2_compiler_propagates_printer_as_computer_failure(tmp_path: Path) -> None:
    manifest_path = Path(
        json.loads(OFFICE_CONFIG.read_text(encoding="utf-8"))["scene"]["source_manifest"]
    )
    manifest_path = (OFFICE_CONFIG.parent / manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    next(asset for asset in manifest["assets"] if asset["instance"] == 9)[
        "category"
    ] = "printers"
    changed_manifest = tmp_path / "printer-manifest.json"
    changed_manifest.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(InteractionSitePlanError, match="computer_entity_not_computer"):
        compile_scene_npc_config(
            _v2_config(tmp_path, manifest_path=changed_manifest),
            tmp_path / "candidate.xml",
        )


def test_v1_compiler_output_contract_remains_legacy(tmp_path: Path) -> None:
    outputs = compile_scene_npc_config(_v1_config(tmp_path), tmp_path / "legacy.xml")
    assert set(outputs) == {
        "scene",
        "semantic_v2",
        "semantic_v1",
        "coverage",
        "population",
        "trajectory_profile",
        "receipt",
    }
    assert json.loads(outputs["population"].read_text(encoding="utf-8"))[
        "schema_version"
    ] == 2
    assert json.loads(outputs["receipt"].read_text(encoding="utf-8"))[
        "generator"
    ] == "stretch_mujoco.npc.scene_compiler"
