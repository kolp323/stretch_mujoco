"""Versioned, hash-verified visual identity catalogs for NPC appearance baking."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .bake import BakedAppearance, bake_appearance_definition, register_baked_appearance

IDENTITY_CATALOG_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class AppearanceLayer:
    layer_id: str
    category: str
    image: str
    sha256: str


@dataclass(frozen=True)
class VisualIdentity:
    identity_id: str
    appearance_id: str
    layers: tuple[str, ...]
    traits: dict[str, str]


@dataclass(frozen=True)
class AppearanceCatalog:
    topology_id: str
    base: str
    base_sha256: str
    layers: dict[str, AppearanceLayer]
    identities: dict[str, VisualIdentity]
    source_path: Path
    semantic_mask_manifest: str | None = None
    semantic_mask_manifest_sha256: str | None = None

    @classmethod
    def from_json(cls, path: str | Path) -> "AppearanceCatalog":
        source = Path(path).resolve()
        payload = _mapping(json.loads(source.read_text(encoding="utf-8")), "Appearance catalog")
        if payload.get("schema_version") != IDENTITY_CATALOG_SCHEMA_VERSION:
            raise ValueError("Unsupported appearance catalog schema_version")
        topology_id = _string(payload.get("texture_topology_id"), "Appearance catalog topology")
        base = _string(payload.get("base"), "Appearance catalog base")
        base_sha256 = _sha256_value(payload.get("base_sha256"), "Appearance catalog base_sha256")
        raw_layers = _mapping(payload.get("layers"), "Appearance catalog layers")
        layers: dict[str, AppearanceLayer] = {}
        for layer_id, raw_layer in raw_layers.items():
            item = _mapping(raw_layer, f"Appearance layer '{layer_id}'")
            layers[str(layer_id)] = AppearanceLayer(
                layer_id=str(layer_id),
                category=_string(item.get("category"), f"Appearance layer '{layer_id}' category"),
                image=_string(item.get("image"), f"Appearance layer '{layer_id}' image"),
                sha256=_sha256_value(item.get("sha256"), f"Appearance layer '{layer_id}' sha256"),
            )
        raw_identities = _mapping(payload.get("identities"), "Appearance catalog identities")
        identities: dict[str, VisualIdentity] = {}
        for identity_id, raw_identity in raw_identities.items():
            item = _mapping(raw_identity, f"Visual identity '{identity_id}'")
            raw_selected = item.get("layers")
            if not isinstance(raw_selected, list) or not all(
                isinstance(layer_id, str) for layer_id in raw_selected
            ):
                raise ValueError(f"Visual identity '{identity_id}' layers must be string IDs")
            selected = tuple(raw_selected)
            if not selected or len(set(selected)) != len(selected):
                raise ValueError(
                    f"Visual identity '{identity_id}' layers must be unique and non-empty"
                )
            traits = _mapping(item.get("traits", {}), f"Visual identity '{identity_id}' traits")
            identities[str(identity_id)] = VisualIdentity(
                identity_id=str(identity_id),
                appearance_id=_string(
                    item.get("appearance_id"), f"Visual identity '{identity_id}' appearance_id"
                ),
                layers=selected,
                traits={str(key): str(value) for key, value in traits.items()},
            )
        semantic_mask_manifest = payload.get("semantic_mask_manifest")
        semantic_mask_manifest_sha256 = payload.get("semantic_mask_manifest_sha256")
        if (semantic_mask_manifest is None) != (semantic_mask_manifest_sha256 is None):
            raise ValueError(
                "Appearance catalog semantic_mask_manifest and semantic_mask_manifest_sha256 "
                "must be supplied together"
            )
        catalog = cls(
            topology_id,
            base,
            base_sha256,
            layers,
            identities,
            source,
            (
                _string(semantic_mask_manifest, "Appearance catalog semantic_mask_manifest")
                if semantic_mask_manifest is not None
                else None
            ),
            (
                _sha256_value(
                    semantic_mask_manifest_sha256,
                    "Appearance catalog semantic_mask_manifest_sha256",
                )
                if semantic_mask_manifest_sha256 is not None
                else None
            ),
        )
        catalog.validate()
        return catalog

    def validate(self) -> None:
        _validate_file_hash(self.source_path.parent / self.base, self.base_sha256, "catalog base")
        for layer in self.layers.values():
            _validate_file_hash(
                self.source_path.parent / layer.image,
                layer.sha256,
                f"appearance layer '{layer.layer_id}'",
            )
        for identity in self.identities.values():
            unknown = set(identity.layers) - self.layers.keys()
            if unknown:
                raise ValueError(
                    f"Visual identity '{identity.identity_id}' references unknown layers: "
                    f"{', '.join(sorted(unknown))}"
                )
        if self.semantic_mask_manifest is not None:
            assert self.semantic_mask_manifest_sha256 is not None
            manifest_path = self.source_path.parent / self.semantic_mask_manifest
            _validate_file_hash(
                manifest_path,
                self.semantic_mask_manifest_sha256,
                "semantic mask manifest",
            )
            mask_payload = _mapping(
                json.loads(manifest_path.read_text(encoding="utf-8")), "Semantic mask manifest"
            )
            if mask_payload.get("texture_topology_id") != self.topology_id:
                raise ValueError(
                    "Semantic mask manifest topology does not match appearance catalog"
                )
            for mask_id, raw_mask in _mapping(
                mask_payload.get("masks"), "Semantic mask entries"
            ).items():
                item = _mapping(raw_mask, f"Semantic mask '{mask_id}'")
                _validate_file_hash(
                    manifest_path.parent
                    / _string(item.get("file"), f"Semantic mask '{mask_id}' file"),
                    _sha256_value(item.get("sha256"), f"Semantic mask '{mask_id}' sha256"),
                    f"semantic mask '{mask_id}'",
                )

    def recipe_for(self, identity_id: str) -> dict[str, object]:
        identity = self.identities.get(identity_id)
        if identity is None:
            raise ValueError(f"Unknown visual identity '{identity_id}'")
        return {
            "schema_version": 1,
            "appearance_id": identity.appearance_id,
            "texture_topology_id": self.topology_id,
            "textures": {
                "body": {
                    "base": self.base,
                    "layers": [
                        {"image": self.layers[layer_id].image} for layer_id in identity.layers
                    ],
                }
            },
        }

    def bake(self, identity_id: str, output_dir: str | Path) -> BakedAppearance:
        recipe = self.recipe_for(identity_id)
        digest = hashlib.sha256(
            self.source_path.read_bytes() + identity_id.encode("utf-8")
        ).hexdigest()
        return bake_appearance_definition(
            recipe, self.source_path.parent, output_dir, recipe_sha=digest
        )

    def identity_for_appearance(self, identity_id: str, appearance_id: str) -> None:
        identity = self.identities.get(identity_id)
        if identity is None:
            raise ValueError(f"Unknown visual identity '{identity_id}'")
        if identity.appearance_id != appearance_id:
            raise ValueError(
                f"Visual identity '{identity_id}' produces appearance '{identity.appearance_id}', "
                f"not '{appearance_id}'"
            )


def _mapping(value: object, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be an object")
    return value


def _string(value: object, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{context} must be a non-empty string")
    return value


def _sha256_value(value: object, context: str) -> str:
    result = _string(value, context)
    if len(result) != 64 or any(character not in "0123456789abcdef" for character in result):
        raise ValueError(f"{context} must be a lowercase SHA-256 hex digest")
    return result


def _validate_file_hash(path: Path, expected: str, context: str) -> None:
    if not path.is_file():
        raise ValueError(f"{context} is missing: {path}")
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        raise ValueError(f"{context} SHA-256 mismatch: {path}")


def main() -> None:
    """Bake a named visual identity and optionally register its final appearance."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--identity", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--asset-manifest")
    parser.add_argument("--bundle")
    args = parser.parse_args()
    if bool(args.asset_manifest) != bool(args.bundle):
        parser.error("--asset-manifest and --bundle must be supplied together")
    catalog = AppearanceCatalog.from_json(args.catalog)
    baked = catalog.bake(args.identity, args.output_dir)
    if args.asset_manifest:
        register_baked_appearance(baked, args.asset_manifest, args.bundle)
    print(baked.manifest_path)
