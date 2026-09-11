from pathlib import Path

import pytest

from stretch_mujoco.humanoid.smplx_animation_baker import (
    register_approved_interaction_clips,
    write_npc_asset_manifest,
)
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
    assert set(bundle["clips"]) == set(OFFICE_CLIPS)
    assert bundle["clips"]["sit"]["markers"] == [{"name": "seated", "phase": 0.875}]
    assert bundle["clips"]["pick_up"]["markers"] == [{"name": "grasp", "phase": 0.75}]
    assert len(bundle["sha256"]["humanoid_idle_00_body.obj"]) == 64


def test_approved_interaction_registration_closes_production_clip_contract() -> None:
    population = NpcPopulation.from_json(MODELS / "office_population.production.example.json")
    manifest_path = MODELS / "assets/humanoid/generated/animations/manifest.json"
    manifest = NpcAssetManifest.from_json(manifest_path)

    manifest.validate_population(population)
    bundle = manifest.bundles["smplx_office_neutral_v1"]
    assert set(bundle.clips) == set(OFFICE_CLIPS)
    assert {"pick_up", "give", "receive", "talk"} <= set(bundle.clips)
    assert bundle.clips["stand_up"].markers == ({"name": "standing", "phase": 0.875},)
    assert bundle.clips["place"].markers == ({"name": "release", "phase": 0.75},)
    assert bundle.clips["gesture_wave"].markers == (
        {"name": "gesture_wave_complete", "phase": 0.932},
    )


def test_registered_stand_up_frames_are_the_reversed_sit_sequence() -> None:
    manifest_paths = (
        MODELS / "assets/humanoid/generated/animations/manifest.json",
        MODELS
        / "assets/humanoid/generated/animations/manifest.beautiful_hair_v1.npc_jordan_patell.runtime.json",
        MODELS
        / "assets/humanoid/generated/animations/manifest.ponytail_hair_v1.npc_priya_narayanan.runtime.json",
    )
    for manifest_path in manifest_paths:
        manifest = NpcAssetManifest.from_json(manifest_path)
        for bundle in manifest.bundles.values():
            sit = bundle.clips.get("sit") or bundle.clips.get("sit_down")
            stand_up = bundle.clips.get("stand_up")
            if sit is not None and stand_up is not None:
                assert stand_up.frames == tuple(reversed(sit.frames))


def test_asset_loader_rejects_a_stand_up_sequence_that_is_not_reversed_sit(tmp_path: Path) -> None:
    for name in ("idle.obj", "sit_00.obj", "sit_01.obj"):
        (tmp_path / name).write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        """
        {
          "schema_version": 1,
          "bundles": {
            "bundle": {
              "format": "mesh_sequence", "topology_id": "test", "coordinate_system": "mujoco_z_up",
              "unit": "meter", "height_m": 1.7, "material_slots": ["body"], "asset_quality": "preview",
              "appearances": {},
              "clips": {
                "idle": {"fps": 8, "loop": true, "root_motion": "in_place", "frames": ["idle.obj"]},
                "sit": {"fps": 8, "loop": false, "root_motion": "in_place", "frames": ["sit_00.obj", "sit_01.obj"]},
                "stand_up": {"fps": 8, "loop": false, "root_motion": "in_place", "frames": ["sit_00.obj", "sit_01.obj"]}
              },
              "sha256": {}
            }
          }
        }
        """
    )

    with pytest.raises(ValueError, match="stand_up frames must be the reverse"):
        NpcAssetManifest.from_json(manifest_path)


def test_interaction_registrar_projects_reviewed_clips_and_hashes(tmp_path: Path) -> None:
    frame = tmp_path / "humanoid_give_00_body.obj"
    frame.write_text("v 0 0 0\n")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        '{"bundles":{"bundle":{"topology_id":"topology","coordinate_system":"mujoco_z_up","unit":"meter","clips":{},"sha256":{}}}}'
    )
    approved_path = tmp_path / "approved.json"
    approved_path.write_text(
        '{"bundle_id":"bundle","topology_id":"topology","coordinate_system":"mujoco_z_up","unit":"meter","clips":{"give":{"fps":8,"loop":false,"root_motion":"in_place","frames":["humanoid_give_00_body.obj"],"markers":[{"name":"handover_ready","phase":0.6}]}}}'
    )

    registered = register_approved_interaction_clips(manifest_path, approved_path)

    bundle = registered["bundles"]["bundle"]
    assert bundle["clips"]["give"]["markers"] == [{"name": "handover_ready", "phase": 0.6}]
    assert len(bundle["sha256"]["humanoid_give_00_body.obj"]) == 64
