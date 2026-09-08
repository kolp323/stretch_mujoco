import importlib.util
import io
import json
import sys
import zipfile
from dataclasses import replace
from pathlib import Path

import numpy as np

from stretch_mujoco.npc.appearance_pipeline.accessory_recipe import FusedAccessoryRecipe


def _module():
    path = Path(__file__).parents[1] / "tools" / "tune_npc_obj_accessory.py"
    spec = importlib.util.spec_from_file_location("tune_npc_obj_accessory", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _fixture(tmp_path: Path) -> Path:
    body = tmp_path / "body.obj"
    body.write_text("v -0.1 -0.1 1.60\n" "v 0.1 -0.1 1.60\n" "v 0.0 0.1 1.75\n" "f 1 2 3\n")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({"bundles": {"bundle": {"clips": {"idle": {"frames": ["body.obj"]}}}}})
    )
    accessory_obj = b"v 0 0 0\nv 1 0 0\nv 0 1 1\nf 1 2 3\n"
    nested = io.BytesIO()
    with zipfile.ZipFile(nested, "w") as archive:
        archive.writestr("accessory.obj", accessory_obj)
    source = tmp_path / "accessory.zip"
    with zipfile.ZipFile(source, "w") as archive:
        archive.writestr("source/accessory.zip", nested.getvalue())
    recipe = tmp_path / "accessory.recipe.json"
    recipe.write_text(
        json.dumps(
            {
                "schema_version": 4,
                "accessory_id": "test_accessory_v1",
                "attachment_mode": "head_follow_fused",
                "mesh_scale": 0.1,
                "lateral_offset_m": 0.02,
                "head_clearance_m": -0.01,
                "back_offset_m": 0.03,
                "back_tilt_degrees": 0.0,
                "roll_degrees": 0.0,
                "yaw_degrees": 0.0,
                "source_vertical_anchor": 0.0,
                "accessory_uv": [0.0, 0.0],
            }
        )
    )
    runtime = tmp_path / "accessory.runtime.json"
    runtime.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "recipe": recipe.name,
                "source_archive": source.name,
                "source_format": "nested_zip_obj",
                "nested_archive_member": "source/accessory.zip",
                "obj_member": "accessory.obj",
                "source_unit_scale": 1.0,
                "source_up_axis": "y",
                "source_manifest": manifest.name,
                "source_population": "population.json",
                "npc_id": "npc_test",
                "bundle": "bundle",
                "output_dir": "generated/accessory",
                "output_manifest": "generated/manifest.json",
                "output_population": "generated/population.json",
            }
        )
    )
    return runtime


def test_tuner_uses_production_transform_and_persists_full_pose(tmp_path: Path) -> None:
    module = _module()
    context = module.load_tuning_context(_fixture(tmp_path))

    vertices = module.transformed_accessory(context, context.recipe)

    assert np.isclose(vertices[:, 0].min(), -0.03)
    assert np.isclose(vertices[:, 1].min(), -0.12)
    assert np.isclose(vertices[:, 2].min(), 1.74)

    changed = replace(
        context.recipe,
        mesh_scale=0.2,
        lateral_offset_m=-0.04,
        back_offset_m=-0.05,
        head_clearance_m=0.02,
        back_tilt_degrees=12.0,
        roll_degrees=-7.0,
        yaw_degrees=90.0,
    )
    module.save_recipe(context.recipe_path, changed)

    saved = FusedAccessoryRecipe.from_json(context.recipe_path)
    assert saved == changed


def test_tuner_can_render_a_headless_snapshot(tmp_path: Path) -> None:
    module = _module()
    context = module.load_tuning_context(_fixture(tmp_path))
    snapshot = tmp_path / "preview.png"

    module.run_tuner(context, snapshot=snapshot, maximum_faces=20)

    assert snapshot.is_file()
    assert snapshot.stat().st_size > 1000
