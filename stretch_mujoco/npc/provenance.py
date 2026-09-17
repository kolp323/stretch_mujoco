"""Content-addressed provenance helpers for the active NPC catalog."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


class ProvenanceError(ValueError):
    """Raised when an active artifact or sidecar is stale or unbound."""


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_active_catalog(path: str | Path) -> dict[str, Any]:
    catalog_path = Path(path).resolve()
    try:
        catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProvenanceError(f"active_catalog_unreadable:{catalog_path}") from exc
    if not isinstance(catalog, dict) or catalog.get("schema_version") != 1:
        raise ProvenanceError("active_catalog_schema_invalid")
    if not isinstance(catalog.get("build_id"), str) or len(catalog["build_id"]) != 64:
        raise ProvenanceError("active_catalog_build_id_invalid")
    scenes = catalog.get("scenes")
    if not isinstance(scenes, dict) or catalog.get("scene_count") != len(scenes):
        raise ProvenanceError("active_catalog_scene_count_invalid")
    build_root = catalog.get("build_root")
    if not isinstance(build_root, str) or Path(build_root).is_absolute():
        raise ProvenanceError("active_catalog_build_root_invalid")
    return catalog


def resolve_active_artifact(catalog_path: str | Path, scene_id: str, artifact: str) -> Path:
    catalog_file = Path(catalog_path).resolve()
    catalog = load_active_catalog(catalog_file)
    record = catalog["scenes"].get(scene_id)
    if not isinstance(record, dict) or artifact not in record:
        raise ProvenanceError(f"active_artifact_unbound:{scene_id}:{artifact}")
    root = (catalog_file.parent / catalog["build_root"]).resolve()
    target = (root / str(record[artifact])).resolve()
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise ProvenanceError(f"active_artifact_escapes_build:{scene_id}:{artifact}") from exc
    if not target.is_file():
        raise ProvenanceError(f"active_artifact_missing:{scene_id}:{artifact}")
    return target


def validate_active_catalog(path: str | Path) -> dict[str, Any]:
    catalog_file = Path(path).resolve()
    catalog = load_active_catalog(catalog_file)
    receipt_bytes: list[bytes] = []
    checked = 0
    for scene_id in sorted(catalog["scenes"]):
        record = catalog["scenes"][scene_id]
        if not isinstance(record, dict):
            raise ProvenanceError(f"active_scene_record_invalid:{scene_id}")
        receipt_path = resolve_active_artifact(catalog_file, scene_id, "receipt")
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        hashes = receipt.get("sha256", {})
        if not isinstance(hashes, dict):
            raise ProvenanceError(f"receipt_hashes_missing:{scene_id}")
        receipt_bytes.append(receipt_path.read_bytes())
        for artifact, expected in hashes.items():
            catalog_artifact = {"generated_scene": "scene"}.get(artifact, artifact)
            if catalog_artifact not in record:
                continue
            actual = sha256_file(resolve_active_artifact(catalog_file, scene_id, catalog_artifact))
            if actual != expected:
                raise ProvenanceError(f"receipt_hash_mismatch:{scene_id}:{artifact}")
            checked += 1
    build_id = hashlib.sha256(b"".join(receipt_bytes)).hexdigest()
    if build_id != catalog["build_id"]:
        raise ProvenanceError("active_catalog_build_id_mismatch")
    return {"build_id": build_id, "scene_count": len(catalog["scenes"]), "checked_hashes": checked}


def validate_showcase_sidecar(sidecar_path: str | Path, catalog_path: str | Path) -> dict[str, Any]:
    sidecar_file = Path(sidecar_path).resolve()
    try:
        payload = json.loads(sidecar_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProvenanceError(f"showcase_sidecar_unreadable:{sidecar_file}") from exc
    catalog = load_active_catalog(catalog_path)
    if payload.get("build_id") != catalog["build_id"]:
        raise ProvenanceError("showcase_sidecar_build_id_mismatch")
    scene_id = payload.get("scene_id")
    if scene_id not in catalog["scenes"]:
        raise ProvenanceError("showcase_sidecar_scene_unbound")
    expected_catalog_hash = payload.get("active_catalog_sha256")
    if expected_catalog_hash and expected_catalog_hash != sha256_file(Path(catalog_path)):
        raise ProvenanceError("showcase_sidecar_catalog_hash_mismatch")
    return payload
