"""Strict integration coverage for generated Office NPC preprocessing."""

from __future__ import annotations

import json
import hashlib
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import pytest

from stretch_mujoco.humanoid.navigation import OfficeNavigationMesh
from stretch_mujoco.npc.assets import NpcAssetManifest
from stretch_mujoco.npc.composition import _bind_population_semantics, compose_npc_scene
from stretch_mujoco.npc.schema import NpcPopulation
from stretch_mujoco.npc.system import NpcSystem
from stretch_mujoco.npc.trajectory_profile import NpcTrajectoryProfile
from stretch_mujoco.semantics import SemanticWorld
from tools.preprocess_generated_office_npcs import _json_write, _population, _trajectory


MODELS = Path(__file__).resolve().parents[1] / "stretch_mujoco" / "models"
SCENES = MODELS / "assets" / "office_scenes"
GENERATED = MODELS / "generated_office_npc"
SCENE_IDS = tuple(
    f"office_{index:02d}_{suffix}"
    for index, suffix in (
        (1, "linear_bench"),
        (2, "cross_axis"),
        (3, "long_gallery"),
        (4, "central_meeting"),
        (5, "team_clusters"),
        (6, "diagonal_flow"),
        (7, "u_bench"),
        (8, "dual_island"),
        (9, "staggered_rows"),
        (10, "social_core"),
    )
)


def test_custom_output_population_paths_are_resolved_from_population_directory(
    tmp_path: Path,
) -> None:
    """The public --output root must not rely on DEFAULT_OUTPUT's layout."""
    output = tmp_path / "arbitrary-output-root"
    scene = SCENES / "office_01_linear_bench_npc.xml"
    sites = {
        name: (0.0, 0.0, 0.0)
        for name in (
            "npc_spawn_01_site",
            "npc_spawn_02_site",
            "npc_spawn_03_site",
            "npc_conversation_speaker_site",
            "npc_conversation_listener_site",
            "npc_handover_giver_site",
            "npc_handover_receiver_site",
        )
    }
    profile_path = output / "trajectory_profiles" / "office_01_linear_bench_npc.json"
    _json_write(
        profile_path, _trajectory(scene.name, hashlib.sha256(scene.read_bytes()).hexdigest())
    )
    population_path = output / "populations" / "office_01_linear_bench.population.json"
    _json_write(population_path, _population(scene.name, output, sites))

    population = NpcPopulation.from_json(population_path)
    assert population.resolve_path(population.scene).is_file()
    assert population.resolve_path(population.asset_manifest).is_file()
    assert population.resolve_path(population.appearance_catalog or "").is_file()
    assert population.resolve_path(population.trajectory_profile or "").is_file()


@pytest.mark.parametrize("scene_id", SCENE_IDS)
def test_generated_office_npc_artifacts_are_portable_and_schema_valid(scene_id: str) -> None:
    scene = SCENES / f"{scene_id}_npc.xml"
    population_path = GENERATED / "populations" / f"{scene_id}.population.json"
    profile_path = GENERATED / "trajectory_profiles" / f"{scene_id}_npc.json"
    semantic_path = GENERATED / "semantics" / f"{scene_id}.semantic.json"
    receipt_path = GENERATED / "receipts" / f"{scene_id}.json"

    assert scene.is_file() and population_path.is_file() and profile_path.is_file()
    assert semantic_path.is_file() and receipt_path.is_file()
    path_attributes = [
        value
        for node in ET.parse(scene).getroot().iter()
        for attribute, value in node.attrib.items()
        if attribute in {"file", "assetdir"}
    ]
    assert path_attributes and all(not Path(value).is_absolute() for value in path_attributes)

    population = NpcPopulation.from_json(population_path)
    assert population.resolve_path(population.scene) == scene.resolve()
    assert len(population.npcs) == 3
    manifest = NpcAssetManifest.from_json(population.resolve_path(population.asset_manifest))
    manifest.validate_population(population)
    profile = NpcTrajectoryProfile.from_json(profile_path)
    profile.validate_scene(scene)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["sha256"]["generated_scene"] == hashlib.sha256(scene.read_bytes()).hexdigest()
    semantic_payload = json.loads(semantic_path.read_text(encoding="utf-8"))
    shared_attributes = semantic_payload["objects"]["shared_object"]["attributes"]
    assert shared_attributes["source_category"] == "interactive_objects"
    assert all(shared_attributes[key] for key in ("source_asset_id", "source_name", "grasp_site"))

    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    profile.preflight(model, data)
    navigation = OfficeNavigationMesh.from_model(model, data)
    spawn_positions = []
    for definition in population.npcs.values():
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, definition.spawn.site)
        assert site_id >= 0
        point = data.site_xpos[site_id][:2]
        assert navigation.is_world_free(point)
        spawn_positions.append(point)
    assert (
        min(
            math.dist(left, right)
            for index, left in enumerate(spawn_positions)
            for right in spawn_positions[index + 1 :]
        )
        >= 1.0
    )

    for first, second in (
        ("npc_conversation_speaker_site", "npc_conversation_listener_site"),
        ("npc_handover_giver_site", "npc_handover_receiver_site"),
    ):
        first_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, first)
        second_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, second)
        distance = math.dist(data.site_xpos[first_id][:2], data.site_xpos[second_id][:2])
        assert 0.75 <= distance <= 1.10
        first_x = data.site_xmat[first_id].reshape(3, 3)[:, 0][:2]
        second_x = data.site_xmat[second_id].reshape(3, 3)[:, 0][:2]
        assert float(first_x @ second_x) < -0.99


@pytest.mark.parametrize("scene_id", SCENE_IDS)
def test_every_generated_office_composes_and_validates_runtime_contract(
    scene_id: str, tmp_path: Path
) -> None:
    population_path = GENERATED / "populations" / f"{scene_id}.population.json"
    semantic_path = GENERATED / "semantics" / f"{scene_id}.semantic.json"
    composition = compose_npc_scene(population_path, tmp_path / f"{scene_id}.xml", reuse=False)
    model = mujoco.MjModel.from_xml_path(str(composition.scene_path))
    population = NpcPopulation.from_json(population_path)
    manifest = NpcAssetManifest.from_json(population.resolve_path(population.asset_manifest))
    # This invokes the population-authoritative profile digest and full route preflight.
    assert (
        len(
            NpcSystem.from_population(
                model, population, manifest, scene_path=composition.base_scene_path
            ).controllers
        )
        == 3
    )
    profile = NpcTrajectoryProfile.from_json(
        GENERATED / "trajectory_profiles" / f"{scene_id}_npc.json"
    )
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    # Use the runtime locomotion controller against every declared route, not
    # just the 2-D planner, so capsule contacts with furniture fail loudly.
    assert len(profile.audit_npc_clearance(model, data, "npc_alex_chen", sample_period=0.1)) == 4
    world = SemanticWorld.from_json(semantic_path)
    _bind_population_semantics(world, population)
    world.validate_model(model)
