"""Bake layered PNG textures into an NPC appearance manifest projection."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np

from ..assets import NpcAssetManifest

BAKE_RECIPE_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class BakedAppearance:
    """The generated files and manifest fragment for one baked appearance."""

    appearance_id: str
    texture_topology_id: str
    textures: dict[str, Path]
    sha256: dict[str, str]
    manifest_path: Path

    def manifest_fragment(self, *, relative_to: Path) -> dict[str, object]:
        """Return the fragment accepted by ``NpcAssetManifest.appearances``."""
        return {
            "textures": {
                slot: str(path.relative_to(relative_to)) for slot, path in self.textures.items()
            },
            "texture_topology_id": self.texture_topology_id,
        }


def bake_appearance(recipe_path: str | Path, output_dir: str | Path) -> BakedAppearance:
    """Bake every texture slot in *recipe_path* into ``output_dir``.

    Recipe files are the source of truth. Generated PNGs and their sidecar
    manifest are projections and should be regenerated instead of hand-edited.
    """
    source = Path(recipe_path).resolve()
    output = Path(output_dir).resolve()
    recipe = _read_recipe(source)
    output.mkdir(parents=True, exist_ok=True)

    appearance_id = _required_string(recipe, "appearance_id", "Bake recipe")
    topology_id = _required_string(recipe, "texture_topology_id", "Bake recipe")
    textures = _mapping(recipe.get("textures"), "Bake recipe textures")
    if not textures:
        raise ValueError("Bake recipe must define at least one texture slot")

    written: dict[str, Path] = {}
    digests: dict[str, str] = {}
    for slot, raw_texture in textures.items():
        slot_name = str(slot)
        texture = _mapping(raw_texture, f"Texture slot '{slot_name}'")
        image = _bake_texture(source.parent, texture, slot_name)
        filename = str(texture.get("output", f"{slot_name}.png"))
        destination = (output / filename).resolve()
        if output not in destination.parents:
            raise ValueError(f"Texture slot '{slot_name}' output must stay below output_dir")
        destination.parent.mkdir(parents=True, exist_ok=True)
        _write_png(destination, image)
        written[slot_name] = destination
        digests[slot_name] = _sha256(destination)

    sidecar = output / f"{appearance_id}.appearance.json"
    sidecar_payload = {
        "schema_version": BAKE_RECIPE_SCHEMA_VERSION,
        "appearance_id": appearance_id,
        "texture_topology_id": topology_id,
        "textures": {
            slot: path.name if path.parent == output else str(path.relative_to(output))
            for slot, path in written.items()
        },
        "sha256": {
            str(path.name if path.parent == output else path.relative_to(output)): digests[slot]
            for slot, path in written.items()
        },
        "recipe_sha256": _sha256(source),
    }
    _write_json(sidecar, sidecar_payload)
    return BakedAppearance(appearance_id, topology_id, written, digests, sidecar)


def register_baked_appearance(
    baked: BakedAppearance, asset_manifest_path: str | Path, bundle_id: str
) -> None:
    """Register generated files in a bundle, then run normal asset validation.

    The generated paths must be below the asset manifest's directory so their
    references remain portable with the rest of the bundle.
    """
    manifest_path = Path(asset_manifest_path).resolve()
    root = manifest_path.parent
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    bundles = _mutable_mapping(payload.get("bundles"), "NPC asset manifest bundles")
    bundle = _mutable_mapping(bundles.get(bundle_id), f"Asset bundle '{bundle_id}'")
    slots = set(bundle.get("material_slots", []))
    if set(baked.textures) - slots:
        raise ValueError(f"Baked appearance uses slots absent from bundle '{bundle_id}'")
    if baked.texture_topology_id != bundle.get("topology_id"):
        raise ValueError("Baked appearance topology does not match the target bundle")
    appearances = _mutable_mapping(
        bundle.get("appearances"), f"Asset bundle '{bundle_id}' appearances"
    )
    if baked.appearance_id in appearances:
        raise ValueError(
            f"Appearance '{baked.appearance_id}' already exists in bundle '{bundle_id}'"
        )

    fragment = baked.manifest_fragment(relative_to=root)
    appearances[baked.appearance_id] = fragment
    hashes = _mutable_mapping(bundle.get("sha256"), f"Asset bundle '{bundle_id}' sha256")
    for slot, path in baked.textures.items():
        hashes[str(path.relative_to(root))] = baked.sha256[slot]
    candidate = manifest_path.with_suffix(manifest_path.suffix + ".candidate")
    _write_json(candidate, payload)
    try:
        NpcAssetManifest.from_json(candidate)
        os.replace(candidate, manifest_path)
    finally:
        candidate.unlink(missing_ok=True)


def _read_recipe(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    recipe = _mapping(payload, "Bake recipe")
    if recipe.get("schema_version") != BAKE_RECIPE_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported bake recipe schema_version {recipe.get('schema_version')!r}; "
            f"expected {BAKE_RECIPE_SCHEMA_VERSION}"
        )
    return recipe


def _bake_texture(root: Path, texture: Mapping[str, Any], slot: str) -> np.ndarray:
    base_name = _required_string(texture, "base", f"Texture slot '{slot}'")
    canvas = _read_rgba(root / base_name, f"Texture slot '{slot}' base")
    layers = texture.get("layers", [])
    if not isinstance(layers, list):
        raise ValueError(f"Texture slot '{slot}' layers must be a list")
    for index, raw_layer in enumerate(layers):
        layer = _mapping(raw_layer, f"Texture slot '{slot}' layer {index}")
        image_name = _required_string(layer, "image", f"Texture slot '{slot}' layer {index}")
        overlay = _read_rgba(root / image_name, f"Texture slot '{slot}' layer {index}")
        if overlay.shape != canvas.shape:
            raise ValueError(
                f"Texture slot '{slot}' layer {index} dimensions do not match its base"
            )
        alpha = overlay[..., 3].astype(np.float32) / 255.0
        if "mask" in layer:
            mask = _read_mask(root / _required_string(layer, "mask", "Layer"), slot, index)
            if mask.shape != canvas.shape[:2]:
                raise ValueError(
                    f"Texture slot '{slot}' layer {index} mask dimensions do not match"
                )
            alpha *= mask.astype(np.float32) / 255.0
        opacity = float(layer.get("opacity", 1.0))
        if not 0.0 <= opacity <= 1.0:
            raise ValueError(f"Texture slot '{slot}' layer {index} opacity must be in [0, 1]")
        alpha *= opacity
        blend = str(layer.get("blend", "normal"))
        if blend == "normal":
            source = overlay[..., :3].astype(np.float32)
        elif blend == "multiply":
            source = (
                canvas[..., :3].astype(np.float32) * overlay[..., :3].astype(np.float32) / 255.0
            )
        else:
            raise ValueError(f"Texture slot '{slot}' layer {index} has unsupported blend '{blend}'")
        destination = canvas[..., :3].astype(np.float32)
        canvas[..., :3] = np.round(
            source * alpha[..., None] + destination * (1 - alpha[..., None])
        ).astype(np.uint8)
        canvas[..., 3] = np.maximum(canvas[..., 3], np.round(alpha * 255).astype(np.uint8))
    return canvas


def _read_rgba(path: Path, context: str) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None or image.ndim not in {2, 3}:
        raise ValueError(f"{context} PNG cannot be decoded: {path}")
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGRA)
    elif image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
    elif image.shape[2] != 4:
        raise ValueError(f"{context} PNG has unsupported channels: {path}")
    return image


def _read_mask(path: Path, slot: str, index: int) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise ValueError(f"Texture slot '{slot}' layer {index} mask cannot be decoded: {path}")
    if mask.ndim == 2:
        return mask
    if mask.ndim == 3 and mask.shape[2] in {3, 4}:
        return mask[..., 3] if mask.shape[2] == 4 else cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    raise ValueError(f"Texture slot '{slot}' layer {index} mask has unsupported channels")


def _write_png(path: Path, image: np.ndarray) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp.png")
    if not cv2.imwrite(str(temporary), image, [cv2.IMWRITE_PNG_COMPRESSION, 9]):
        raise OSError(f"Could not write baked texture '{path}'")
    os.replace(temporary, path)


def _write_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _mapping(value: object, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be an object")
    return value


def _mutable_mapping(value: object, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be an object")
    return value


def _required_string(payload: Mapping[str, Any], key: str, context: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{context} must define non-empty string '{key}'")
    return value


def main() -> None:
    """Console entry point for baking and optionally registering an appearance."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--recipe", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--asset-manifest")
    parser.add_argument("--bundle")
    args = parser.parse_args()
    if bool(args.asset_manifest) != bool(args.bundle):
        parser.error("--asset-manifest and --bundle must be supplied together")
    baked = bake_appearance(args.recipe, args.output_dir)
    if args.asset_manifest:
        register_baked_appearance(baked, args.asset_manifest, args.bundle)
    print(baked.manifest_path)
