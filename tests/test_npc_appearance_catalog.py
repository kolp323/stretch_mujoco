import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from stretch_mujoco.npc.appearance_pipeline.catalog import AppearanceCatalog
from stretch_mujoco.npc.schema import NpcAppearance


def _png(path: Path, color: tuple[int, int, int, int]) -> str:
    assert cv2.imwrite(str(path), np.full((2, 2, 4), color, dtype=np.uint8))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_catalog_bakes_named_identity_and_verifies_layer_hashes(tmp_path: Path) -> None:
    base_hash = _png(tmp_path / "base.png", (10, 20, 30, 255))
    hair_hash = _png(tmp_path / "hair.png", (40, 50, 60, 255))
    face_hash = _png(tmp_path / "face.png", (255, 255, 255, 255))
    mask_manifest = tmp_path / "semantic_masks.json"
    mask_manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "texture_topology_id": "topology-v1",
                "masks": {"face": {"file": "face.png", "sha256": face_hash}},
            }
        )
    )
    mask_manifest_hash = hashlib.sha256(mask_manifest.read_bytes()).hexdigest()
    catalog_path = tmp_path / "catalog.json"
    catalog_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "texture_topology_id": "topology-v1",
                "base": "base.png",
                "base_sha256": base_hash,
                "semantic_mask_manifest": "semantic_masks.json",
                "semantic_mask_manifest_sha256": mask_manifest_hash,
                "layers": {
                    f"{category}_v1": {
                        "category": category,
                        "image": "hair.png",
                        "sha256": hair_hash,
                    }
                    for category in ("skin", "hair", "top", "bottom", "shoes")
                },
                "identities": {
                    "alex_v1": {
                        "appearance_id": "alex_v1",
                        "layers": ["skin_v1", "hair_v1", "top_v1", "bottom_v1", "shoes_v1"],
                        "traits": {"hair_style": "short"},
                    }
                },
            }
        )
    )

    catalog = AppearanceCatalog.from_json(catalog_path)
    baked = catalog.bake("alex_v1", tmp_path / "output")

    assert baked.appearance_id == "alex_v1"
    assert baked.textures["body"].is_file()
    assert baked.thumbnail_path.is_file()
    receipt = json.loads(baked.manifest_path.read_text())
    assert receipt["seed"] == catalog.identities["alex_v1"].seed
    assert receipt["artifacts"]["thumbnail"] == baked.thumbnail_path.name
    catalog.validate_appearance_slots(
        "alex_v1",
        NpcAppearance.from_dict(
            {
                "skin": "skin_v1",
                "hair": "hair_v1",
                "top": "top_v1",
                "bottom": "bottom_v1",
                "shoes": "shoes_v1",
                "accessories": [],
                "scale": 1.0,
            }
        ),
    )
    catalog.identity_for_appearance("alex_v1", "alex_v1")
    with pytest.raises(ValueError, match="not 'other_v1'"):
        catalog.identity_for_appearance("alex_v1", "other_v1")


def test_catalog_rejects_modified_layer(tmp_path: Path) -> None:
    base_hash = _png(tmp_path / "base.png", (10, 20, 30, 255))
    _png(tmp_path / "hair.png", (40, 50, 60, 255))
    catalog_path = tmp_path / "catalog.json"
    catalog_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "texture_topology_id": "topology-v1",
                "base": "base.png",
                "base_sha256": base_hash,
                "layers": {
                    "hair": {
                        "category": "hair",
                        "image": "hair.png",
                        "sha256": "0" * 64,
                    }
                },
                "identities": {"alex": {"appearance_id": "alex", "layers": ["hair"]}},
            }
        )
    )

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        AppearanceCatalog.from_json(catalog_path)
