#!/usr/bin/env python3
"""Materialize and register a reproducible catalog of NPC persona appearances."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from stretch_mujoco.npc.appearance_pipeline.catalog import AppearanceCatalog
from stretch_mujoco.npc.appearance_pipeline.face_details import generate_face_detail_layers
from stretch_mujoco.npc.appearance_pipeline.flat_layers import generate_flat_layers
from stretch_mujoco.npc.appearance_pipeline.hair_layers import generate_short_hair_layers
from stretch_mujoco.npc.appearance_pipeline.textile_layers import generate_textile_layers
from stretch_mujoco.npc.assets import NpcAssetManifest


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _required_string(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"Roster must define non-empty string '{key}'")
    return value


def _relative(root: Path, relative: str) -> Path:
    return root / relative


def build_persona_roster(
    roster_path: str | Path,
    asset_root: str | Path,
    *,
    catalog_path: str | Path | None = None,
    manifest_path: str | Path | None = None,
    bundle_id: str = "smplx_office_neutral_v1",
) -> tuple[Path, tuple[str, ...]]:
    """Generate palette layers, bake identities, and atomically register them.

    The versioned roster is authoritative. Catalog PNGs, thumbnails, receipts,
    and manifest entries are local projections that this function may replace
    on every invocation.
    """
    roster_source = Path(roster_path).resolve()
    roster = json.loads(roster_source.read_text(encoding="utf-8"))
    if not isinstance(roster, dict) or roster.get("schema_version") != 1:
        raise ValueError("Unsupported persona roster schema_version")
    root = Path(asset_root).resolve()
    catalog_destination = (
        Path(catalog_path).resolve()
        if catalog_path is not None
        else root / "appearance_catalog.json"
    )
    manifest_destination = (
        Path(manifest_path).resolve() if manifest_path is not None else root / "manifest.json"
    )
    if not manifest_destination.is_file():
        raise ValueError(f"Asset manifest is missing: {manifest_destination}")

    generate_flat_layers(
        roster_source.parent / _required_string(roster, "flat_layer_spec"),
        _relative(root, _required_string(roster, "flat_layer_output")),
    )
    generate_short_hair_layers(
        roster_source.parent / _required_string(roster, "hair_layer_spec"),
        _relative(root, _required_string(roster, "hair_layer_output")),
    )
    face_detail_spec = roster.get("face_detail_spec")
    if face_detail_spec is not None:
        generate_face_detail_layers(
            _relative(root, _required_string(roster, "face_detail_spec")),
            _relative(root, _required_string(roster, "face_detail_output")),
        )
    textile_layer_spec = roster.get("textile_layer_spec")
    if textile_layer_spec is not None:
        generate_textile_layers(
            roster_source.parent / _required_string(roster, "textile_layer_spec"), root
        )

    raw_layers = roster.get("layers")
    raw_identities = roster.get("identities")
    if not isinstance(raw_layers, list) or not isinstance(raw_identities, list):
        raise ValueError("Roster layers and identities must be arrays")
    layers: dict[str, dict[str, str]] = {}
    for item in raw_layers:
        if not isinstance(item, dict):
            raise ValueError("Roster layer entries must be objects")
        layer_id = _required_string(item, "id")
        image = _required_string(item, "image")
        category = _required_string(item, "category")
        image_path = _relative(root, image)
        if not image_path.is_file():
            raise ValueError(f"Roster layer is missing after generation: {image_path}")
        layers[layer_id] = {"category": category, "image": image, "sha256": _sha256(image_path)}

    identities: dict[str, dict[str, object]] = {}
    for item in raw_identities:
        if not isinstance(item, dict):
            raise ValueError("Roster identity entries must be objects")
        identity_id = _required_string(item, "id")
        slots = item.get("slots")
        details = item.get("details", [])
        traits = item.get("traits", {})
        seed = item.get("seed")
        if (
            not isinstance(slots, list)
            or len(slots) != 5
            or not all(isinstance(value, str) for value in slots)
            or not isinstance(details, list)
            or not all(isinstance(value, str) for value in details)
            or not isinstance(traits, dict)
            or not isinstance(seed, int)
        ):
            raise ValueError(
                f"Roster identity '{identity_id}' has invalid slots, details, traits, or seed"
            )
        selected = [*slots, *details]
        if len(set(selected)) != len(selected) or set(selected) - layers.keys():
            raise ValueError(f"Roster identity '{identity_id}' has duplicate or unknown layers")
        categories = {layers[layer_id]["category"] for layer_id in slots}
        if categories != {"skin", "hair", "top", "bottom", "shoes"}:
            raise ValueError(
                f"Roster identity '{identity_id}' must select every material slot once"
            )
        identities[identity_id] = {
            "appearance_id": identity_id,
            "layers": selected,
            "traits": {str(key): str(value) for key, value in traits.items()},
            "seed": seed,
        }

    base = _required_string(roster, "base")
    semantic_mask_manifest = _required_string(roster, "semantic_mask_manifest")
    catalog_payload = {
        "schema_version": 1,
        "texture_topology_id": _required_string(roster, "texture_topology_id"),
        "base": base,
        "base_sha256": _sha256(_relative(root, base)),
        "semantic_mask_manifest": semantic_mask_manifest,
        "semantic_mask_manifest_sha256": _sha256(_relative(root, semantic_mask_manifest)),
        "layers": layers,
        "identities": identities,
    }
    catalog_destination.parent.mkdir(parents=True, exist_ok=True)
    _write_json(catalog_destination, catalog_payload)
    catalog = AppearanceCatalog.from_json(catalog_destination)

    baked = [
        catalog.bake(identity_id, root / "appearances" / identity_id) for identity_id in identities
    ]
    manifest = json.loads(manifest_destination.read_text(encoding="utf-8"))
    bundle = manifest["bundles"][bundle_id]
    appearances = bundle.setdefault("appearances", {})
    hashes = bundle.setdefault("sha256", {})
    for item in baked:
        appearances[item.appearance_id] = item.manifest_fragment(relative_to=root)
        for _, image in item.textures.items():
            hashes[str(image.relative_to(root))] = _sha256(image)
    candidate = manifest_destination.with_suffix(manifest_destination.suffix + ".candidate")
    _write_json(candidate, manifest)
    try:
        NpcAssetManifest.from_json(candidate)
        os.replace(candidate, manifest_destination)
    finally:
        candidate.unlink(missing_ok=True)
    return catalog_destination, tuple(identities)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roster", required=True, type=Path)
    parser.add_argument("--asset-root", required=True, type=Path)
    parser.add_argument("--catalog", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--bundle", default="smplx_office_neutral_v1")
    args = parser.parse_args()
    catalog, identities = build_persona_roster(
        args.roster,
        args.asset_root,
        catalog_path=args.catalog,
        manifest_path=args.manifest,
        bundle_id=args.bundle,
    )
    print(json.dumps({"catalog": str(catalog), "identities": identities}, indent=2))


if __name__ == "__main__":
    main()
