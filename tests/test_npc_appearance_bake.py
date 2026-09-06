import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from stretch_mujoco.npc.appearance_pipeline.bake import bake_appearance, register_baked_appearance
from stretch_mujoco.npc.appearance_pipeline.slots import require_split_geometry
from stretch_mujoco.npc.assets import NpcAssetManifest


def _png(path: Path, color: tuple[int, int, int, int]) -> None:
    image = np.full((2, 2, 4), color, dtype=np.uint8)
    assert cv2.imwrite(str(path), image)


def test_bake_composites_layer_and_writes_manifest_fragment(tmp_path: Path) -> None:
    _png(tmp_path / "base.png", (10, 20, 30, 255))
    _png(tmp_path / "shirt.png", (100, 110, 120, 128))
    recipe = {
        "schema_version": 1,
        "appearance_id": "employee_blue_v1",
        "texture_topology_id": "test-topology-v1",
        "textures": {
            "body": {"base": "base.png", "layers": [{"image": "shirt.png", "opacity": 0.5}]}
        },
    }
    recipe_path = tmp_path / "recipe.json"
    recipe_path.write_text(json.dumps(recipe))

    baked = bake_appearance(recipe_path, tmp_path / "generated")

    image = cv2.imread(str(baked.textures["body"]), cv2.IMREAD_UNCHANGED)
    assert image is not None
    assert tuple(image[0, 0]) == (33, 43, 53, 255)
    sidecar = json.loads(baked.manifest_path.read_text())
    assert sidecar["appearance_id"] == "employee_blue_v1"
    assert len(sidecar["sha256"]["body.png"]) == 64


def test_register_baked_appearance_updates_manifest_and_validates(tmp_path: Path) -> None:
    _png(tmp_path / "base.png", (10, 20, 30, 255))
    mesh = tmp_path / "idle.obj"
    mesh.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nvt 0 0\nvt 1 0\nvt 0 1\nf 1/1 2/2 3/3\n")
    recipe_path = tmp_path / "recipe.json"
    recipe_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "appearance_id": "blue_v1",
                "texture_topology_id": "test-topology-v1",
                "textures": {"body": {"base": "base.png"}},
            }
        )
    )
    manifest_path = tmp_path / "assets.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "bundles": {
                    "test": {
                        "format": "mesh_sequence",
                        "topology_id": "test-topology-v1",
                        "coordinate_system": "mujoco_z_up",
                        "unit": "meter",
                        "height_m": 1.7,
                        "material_slots": ["body"],
                        "asset_quality": "preview",
                        "appearances": {},
                        "clips": {
                            "idle": {
                                "fps": 1,
                                "loop": True,
                                "root_motion": "in_place",
                                "frames": ["idle.obj"],
                            }
                        },
                        "sha256": {"idle.obj": hashlib.sha256(mesh.read_bytes()).hexdigest()},
                    }
                },
            }
        )
    )
    baked = bake_appearance(recipe_path, tmp_path / "generated")
    register_baked_appearance(baked, manifest_path, "test")

    manifest = NpcAssetManifest.from_json(manifest_path)
    assert manifest.bundles["test"].appearances["blue_v1"].textures == {
        "body": "generated/body.png"
    }


def test_split_geometry_is_explicitly_reserved() -> None:
    with pytest.raises(NotImplementedError, match="not implemented"):
        require_split_geometry()
