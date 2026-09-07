from pathlib import Path

import pytest

from stretch_mujoco.humanoid.smplx_animation_baker import write_npc_asset_manifest
from stretch_mujoco.npc.animation import OFFICE_CLIPS
from stretch_mujoco.npc.assets import NpcAssetManifest
from stretch_mujoco.npc.schema import NpcPopulation

MODELS = Path(__file__).resolve().parents[1] / "stretch_mujoco" / "models"


def test_example_preview_assets_are_explicit_and_valid() -> None:
    population = NpcPopulation.from_json(MODELS / "office_population.json")
    manifest = NpcAssetManifest.from_json(MODELS / "npc_assets.example.json")

    manifest.validate_population(population)
    assert manifest.bundles["cesium_man_preview_v1"].asset_quality == "preview"


def test_runtime_animation_graph_is_derived_from_validated_bundle() -> None:
    manifest = NpcAssetManifest.from_json(MODELS / "npc_assets.example.json")

    graph = manifest.animation_graph("cesium_man_preview_v1", "office_preview_v1")

    assert graph.graph_id == "office_preview_v1"
    assert graph.initial_clip == "idle"
    assert graph.clips["idle"].fps > 0


def test_restricted_bundle_requires_the_complete_clip_contract(tmp_path: Path) -> None:
    source = (MODELS / "npc_assets.example.json").read_text()
    source = source.replace('"asset_quality": "preview"', '"asset_quality": "restricted"')
    path = tmp_path / "manifest.json"
    path.write_text(source)

    with pytest.raises(ValueError, match="restricted production clips are incomplete"):
        NpcAssetManifest.from_json(path)


def test_texture_topology_mismatch_is_rejected(tmp_path: Path) -> None:
    source = (MODELS / "npc_assets.example.json").read_text()
    source = source.replace(
        '"texture_topology_id": "cesium-man-preview-v1"',
        '"texture_topology_id": "smplx-neutral-10475-v1"',
    )
    path = tmp_path / "manifest.json"
    path.write_text(source)

    with pytest.raises(ValueError, match="does not match bundle topology"):
        NpcAssetManifest.from_json(path)


def test_smplx_baker_writes_formal_production_clip_contract(tmp_path: Path) -> None:
    (tmp_path / "smplx_employee_diffuse.png").write_bytes(b"texture")
    for clip_name in OFFICE_CLIPS:
        (tmp_path / f"humanoid_{clip_name}_00_body.obj").write_text("mesh")

    manifest = write_npc_asset_manifest(tmp_path, 1.72)
    bundle = manifest["bundles"]["smplx_office_neutral_v1"]

    assert bundle["asset_quality"] == "restricted"
    assert {
        "idle",
        "walk",
        "sit_down",
        "seated_idle",
        "stand_up",
        "work",
        "use_computer",
        "eat",
        "pick_up",
        "place",
        "give",
        "receive",
        "talk",
        "gesture_wave",
        "gesture_point",
    } <= set(bundle["clips"])
    assert bundle["clips"]["sit_down"]["markers"] == [{"name": "seated", "phase": 0.875}]
    assert bundle["clips"]["stand_up"]["markers"] == [{"name": "standing", "phase": 0.875}]
    assert bundle["clips"]["pick_up"]["markers"] == [{"name": "grasp", "phase": 0.75}]
    assert len(bundle["sha256"]["humanoid_idle_00_body.obj"]) == 64
