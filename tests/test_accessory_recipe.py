import importlib.util
import json
from pathlib import Path

import pytest

from stretch_mujoco.npc.appearance_pipeline.accessory_recipe import FusedAccessoryRecipe


def _builder_module():
    path = Path(__file__).parents[1] / "tools" / "build_npc_fused_accessory.py"
    spec = importlib.util.spec_from_file_location("build_npc_fused_accessory", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_recipe_requires_head_follow_fused_and_exact_fields(tmp_path: Path) -> None:
    recipe = tmp_path / "cap.recipe.json"
    recipe.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "accessory_id": "cap",
                "attachment_mode": "head_follow_fused",
                "mesh_scale": 1.2,
                "head_clearance_m": -0.01,
                "back_offset_m": 0.1,
                "back_tilt_degrees": 13,
            }
        )
    )

    parsed = FusedAccessoryRecipe.from_json(recipe)

    assert parsed.mesh_scale == 1.2
    assert parsed.as_dict()["attachment_mode"] == "head_follow_fused"
    recipe.write_text("{}")
    with pytest.raises(ValueError, match="fields mismatch"):
        FusedAccessoryRecipe.from_json(recipe)


def test_runtime_population_removes_only_fused_accessory() -> None:
    module = _builder_module()
    population = {
        "asset_manifest": "old.json",
        "npcs": {
            "alex": {
                "embodiment": {
                    "accessories": ["cap", "glasses"],
                    "appearance_config": {"accessories": ["cap", "glasses"]},
                }
            }
        },
    }

    result = module.runtime_population_payload(
        population, npc_id="alex", accessory_id="cap", manifest_path=Path("/tmp/fused.json")
    )

    embodiment = result["npcs"]["alex"]["embodiment"]
    assert embodiment["accessories"] == ["glasses"]
    assert embodiment["appearance_config"]["accessories"] == ["glasses"]


def test_recipe_accessory_binding_replaces_stale_manifest_reference(tmp_path: Path) -> None:
    module = _builder_module()
    recipe_path = tmp_path / "recipe.json"
    recipe_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "accessory_id": "cap",
                "attachment_mode": "head_follow_fused",
                "mesh_scale": 1.0,
                "head_clearance_m": 0.0,
                "back_offset_m": 0.0,
                "back_tilt_degrees": 0.0,
            }
        )
    )
    recipe = FusedAccessoryRecipe.from_json(recipe_path)
    mesh, anchors = tmp_path / "output" / "cap.obj", tmp_path / "output" / "cap.anchors.json"
    mesh.parent.mkdir()
    mesh.write_text("mesh")
    anchors.write_text("{}")
    manifest_path = tmp_path / "manifest.json"
    manifest = {"bundles": {"bundle": {"accessories": {"cap": {"mesh": "old", "anchors": "old"}}, "sha256": {}}}}

    module.bind_recipe_accessory(
        manifest,
        bundle_id="bundle",
        recipe=recipe,
        mesh_path=mesh,
        anchors_path=anchors,
        manifest_path=manifest_path,
    )

    bound = manifest["bundles"]["bundle"]
    assert bound["accessories"]["cap"] == {
        "mesh": "output/cap.obj",
        "anchors": "output/cap.anchors.json",
    }
    assert bound["sha256"]["output/cap.obj"] == module._sha256(mesh)
