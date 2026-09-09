import importlib.util
import json
from pathlib import Path

import pytest

from stretch_mujoco.npc.appearance_pipeline.accessory_recipe import (
    FusedAccessoryRecipe,
    FusedAccessoryRuntimeConfig,
)


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
        population,
        source_population_path=Path("/tmp/models/office_population.json"),
        npc_id="alex",
        accessory_id="cap",
        manifest_path=Path("/tmp/fused.json"),
    )

    embodiment = result["npcs"]["alex"]["embodiment"]
    assert embodiment["accessories"] == ["glasses"]
    assert embodiment["appearance_config"]["accessories"] == ["glasses"]


def test_runtime_population_resolves_paths_before_moving_projection() -> None:
    module = _builder_module()
    population = {
        "scene": "office_scene.xml",
        "appearance_catalog": "assets/appearance_catalog.json",
        "npcs": {"alex": {"embodiment": {"accessories": []}}},
    }

    result = module.runtime_population_payload(
        population,
        source_population_path=Path("/tmp/models/office_population.json"),
        npc_id="alex",
        accessory_id="cap",
        manifest_path=Path("/tmp/generated/manifest.json"),
    )

    assert result["scene"] == "/tmp/models/office_scene.xml"
    assert result["appearance_catalog"] == "/tmp/models/assets/appearance_catalog.json"


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
    manifest = {
        "bundles": {
            "bundle": {"accessories": {"cap": {"mesh": "old", "anchors": "old"}}, "sha256": {}}
        }
    }

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


def test_runtime_config_has_exact_paths_and_resolves_from_its_file(tmp_path: Path) -> None:
    path = tmp_path / "cap.runtime.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "recipe": "look/cap.recipe.json",
                "source_archive": "source/cap.zip",
                "source_manifest": "assets/manifest.json",
                "source_population": "office_population.json",
                "npc_id": "employee_01",
                "bundle": "smplx_office_neutral_v1",
                "output_dir": "generated/cap",
                "output_manifest": "generated/manifest.json",
                "output_population": "generated/population.json",
            }
        )
    )

    config = FusedAccessoryRuntimeConfig.from_json(path)

    assert config.resolve_path(path, "recipe") == tmp_path / "look/cap.recipe.json"
    path.write_text("{}")
    with pytest.raises(ValueError, match="runtime config fields mismatch"):
        FusedAccessoryRuntimeConfig.from_json(path)


def test_runtime_scene_build_rebuilds_recipe_every_invocation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "cap.runtime.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "recipe": "cap.recipe.json",
                "source_archive": "cap.zip",
                "source_manifest": "manifest.json",
                "source_population": "population.json",
                "npc_id": "employee_01",
                "bundle": "bundle",
                "output_dir": "generated/cap",
                "output_manifest": "generated/manifest.json",
                "output_population": "generated/population.json",
            }
        )
    )
    scene_module_path = Path(__file__).parents[1] / "tools" / "build_npc_scene.py"
    spec = importlib.util.spec_from_file_location("build_npc_scene_tool", scene_module_path)
    assert spec and spec.loader
    scene_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(scene_module)
    calls: list[dict[str, object]] = []

    class Builder:
        @staticmethod
        def build_fused_accessory(**kwargs: object) -> dict[str, str]:
            calls.append(kwargs)
            return {"population": str(tmp_path / "generated/population.json")}

    monkeypatch.setattr(scene_module, "_tool_module", lambda _: Builder)
    monkeypatch.setattr(
        scene_module,
        "build_npc_scene",
        lambda population, output, include_base_scene=False: Path(output),
    )

    scene_module.build_scene_from_accessory_runtime_config(path, tmp_path / "one.xml")
    scene_module.build_scene_from_accessory_runtime_config(path, tmp_path / "two.xml")

    assert len(calls) == 2
    assert calls[0]["recipe_path"] == tmp_path / "cap.recipe.json"


def test_builder_creates_nested_runtime_population_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
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
    source_population = tmp_path / "population.json"
    source_population.write_text("{}")
    output_population = tmp_path / "runtime_populations/cap/employee.json"

    class StopBuild(Exception):
        pass

    monkeypatch.setattr(module, "_tool_module", lambda _: (_ for _ in ()).throw(StopBuild()))
    with pytest.raises(StopBuild):
        module.build_fused_accessory(
            recipe_path=recipe_path,
            source_archive=tmp_path / "source.zip",
            source_manifest=tmp_path / "manifest.json",
            source_population=source_population,
            npc_id="employee",
            output_dir=tmp_path / "runtime/cap/employee",
            output_manifest=tmp_path / "manifests/cap.json",
            output_population=output_population,
            bundle_id="bundle",
        )

    assert output_population.parent.is_dir()


def test_checked_in_cap_runtime_config_references_the_canonical_recipe() -> None:
    models = Path(__file__).parents[1] / "stretch_mujoco/models/accessories"
    recipe_path = models / "cap_source_v1.recipe.json"
    runtime_path = models / "cap_source_v1.runtime.json"

    recipe = FusedAccessoryRecipe.from_json(recipe_path)
    runtime = FusedAccessoryRuntimeConfig.from_json(runtime_path)

    assert runtime.resolve_path(runtime_path, "recipe") == recipe_path
    assert recipe.accessory_id == "cap_source_v1"
    assert runtime.source_archive == (
        "../assets/humanoid/sources/npc/accessories/cap_source_v1/cap_source_v1.zip"
    )
    assert runtime.output_dir.endswith("runtime/cap_source_v1/employee_01")
    assert runtime.source_manifest == "../assets/humanoid/generated/animations/manifest.json"
    assert runtime.output_manifest == (
        "../assets/humanoid/generated/animations/"
        "manifest.cap_source_v1.employee_01.runtime.preview.json"
    )
    assert runtime.output_population.endswith("runtime_populations/cap_source_v1/employee_01.json")
