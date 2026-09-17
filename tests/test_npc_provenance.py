import json
from pathlib import Path

import pytest

from stretch_mujoco.npc.provenance import (
    ProvenanceError,
    load_active_catalog,
    validate_active_catalog,
)


def _catalog(tmp_path: Path) -> Path:
    root = tmp_path / "active"
    build = root / "builds" / ("a" * 64) / "scene"
    build.mkdir(parents=True)
    artifact = build / "scene.json"
    artifact.write_text("{}", encoding="utf-8")
    import hashlib
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    receipt = build / "scene.receipt.json"
    receipt.write_text(json.dumps({"sha256": {"semantic_v2": digest}}), encoding="utf-8")
    rid = hashlib.sha256(receipt.read_bytes()).hexdigest()
    catalog = root / "active_catalog.json"
    catalog.write_text(json.dumps({"schema_version": 1, "build_id": rid, "build_root": f"builds/{'a'*64}", "scene_count": 1, "scenes": {"scene": {"receipt": "scene/scene.receipt.json", "semantic_v2": "scene/scene.json"}}}), encoding="utf-8")
    return catalog


def test_active_catalog_validates_receipt_and_build_digest(tmp_path: Path):
    catalog = _catalog(tmp_path)
    result = validate_active_catalog(catalog)
    assert result["scene_count"] == 1


def test_active_catalog_rejects_receipt_hash_drift(tmp_path: Path):
    catalog = _catalog(tmp_path)
    payload = json.loads(catalog.read_text())
    receipt = catalog.parent / payload["build_root"] / "scene/scene.receipt.json"
    data = json.loads(receipt.read_text())
    data["sha256"]["semantic_v2"] = "0" * 64
    receipt.write_text(json.dumps(data))
    with pytest.raises(ProvenanceError, match="receipt_hash_mismatch"):
        validate_active_catalog(catalog)
