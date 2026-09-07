#!/usr/bin/env python3
"""Build a receipt-backed, head-follow fused NPC accessory runtime projection."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

from stretch_mujoco.npc.appearance_pipeline.accessory_recipe import (
    FusedAccessoryRecipe,
    recipe_sha256,
)
from stretch_mujoco.npc.assets import NpcAssetManifest
from stretch_mujoco.npc.schema import NpcPopulation


def _tool_module(filename: str) -> ModuleType:
    path = Path(__file__).with_name(filename)
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def runtime_population_payload(
    payload: dict[str, object], *, npc_id: str, accessory_id: str, manifest_path: Path
) -> dict[str, object]:
    """Bind one NPC to fused frames and retire only its duplicate accessory geom."""
    npcs = payload.get("npcs")
    if not isinstance(npcs, dict) or npc_id not in npcs or not isinstance(npcs[npc_id], dict):
        raise ValueError(f"Population does not define NPC '{npc_id}'")
    embodiment = npcs[npc_id].get("embodiment")
    if not isinstance(embodiment, dict):
        raise ValueError(f"NPC '{npc_id}' has no embodiment")
    raw_accessories = embodiment.get("accessories", [])
    if not isinstance(raw_accessories, list):
        raise ValueError(f"NPC '{npc_id}' accessories must be a list")
    embodiment["accessories"] = [item for item in raw_accessories if item != accessory_id]
    config = embodiment.get("appearance_config")
    if isinstance(config, dict) and isinstance(config.get("accessories"), list):
        config["accessories"] = [item for item in config["accessories"] if item != accessory_id]
    payload["asset_manifest"] = str(manifest_path.resolve())
    return payload


def bind_recipe_accessory(
    manifest: dict[str, object],
    *,
    bundle_id: str,
    recipe: FusedAccessoryRecipe,
    mesh_path: Path,
    anchors_path: Path,
    manifest_path: Path,
) -> None:
    """Make the manifest's provenance declaration point at this build's inputs."""
    bundle = manifest["bundles"][bundle_id]
    assert isinstance(bundle, dict)
    accessories = bundle.setdefault("accessories", {})
    hashes = bundle.setdefault("sha256", {})
    mesh_relative = mesh_path.relative_to(manifest_path.parent).as_posix()
    anchors_relative = anchors_path.relative_to(manifest_path.parent).as_posix()
    accessories[recipe.accessory_id] = {"mesh": mesh_relative, "anchors": anchors_relative}
    hashes[mesh_relative] = _sha256(mesh_path)
    hashes[anchors_relative] = _sha256(anchors_path)


def build_fused_accessory(
    *,
    recipe_path: Path,
    source_archive: Path,
    source_manifest: Path,
    source_population: Path,
    npc_id: str,
    output_dir: Path,
    output_manifest: Path,
    output_population: Path,
    bundle_id: str,
) -> dict[str, str]:
    """Materialize every derived projection from one immutable pose recipe."""
    recipe = FusedAccessoryRecipe.from_json(recipe_path)
    prepare = _tool_module("prepare_obj_accessory.py")
    fuse = _tool_module("fuse_obj_accessory.py")
    accessory_dir = output_dir / "accessory"
    prepared = prepare.prepare(
        source_archive.resolve(),
        source_manifest.resolve(),
        accessory_dir,
        recipe.accessory_id,
        mesh_scale=recipe.mesh_scale,
        head_clearance_m=recipe.head_clearance_m,
        back_offset_m=recipe.back_offset_m,
        back_tilt_degrees=recipe.back_tilt_degrees,
    )
    receipt_path = Path(prepared["receipt"])
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt_fields = {
        "asset_id": recipe.accessory_id,
        "mesh_scale": recipe.mesh_scale,
        "head_clearance_m": recipe.head_clearance_m,
        "back_offset_m": recipe.back_offset_m,
        "back_tilt_degrees": recipe.back_tilt_degrees,
    }
    for field, expected in receipt_fields.items():
        if receipt.get(field) != expected:
            raise ValueError(f"Generated receipt does not match recipe field '{field}'")
    receipt.update(
        {
            "recipe": str(recipe_path.resolve()),
            "recipe_sha256": recipe_sha256(recipe_path),
            "attachment_mode": recipe.attachment_mode,
        }
    )
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    fused = fuse.fuse_manifest(
        source_manifest.resolve(),
        Path(prepared["mesh"]),
        Path(prepared["anchors"]),
        output_dir / "frames",
        output_manifest.resolve(),
        bundle_id=bundle_id,
    )
    manifest = json.loads(output_manifest.read_text(encoding="utf-8"))
    bind_recipe_accessory(
        manifest,
        bundle_id=bundle_id,
        recipe=recipe,
        mesh_path=Path(prepared["mesh"]),
        anchors_path=Path(prepared["anchors"]),
        manifest_path=output_manifest,
    )
    manifest["fused_accessory"].update(
        {
            "recipe": str(recipe_path.resolve()),
            "recipe_sha256": recipe_sha256(recipe_path),
            "receipt": str(receipt_path.resolve()),
            "receipt_sha256": _sha256(receipt_path),
            "attachment_mode": recipe.attachment_mode,
        }
    )
    output_manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    population = runtime_population_payload(
        json.loads(source_population.read_text(encoding="utf-8")),
        npc_id=npc_id,
        accessory_id=recipe.accessory_id,
        manifest_path=output_manifest,
    )
    output_population.write_text(json.dumps(population, indent=2) + "\n", encoding="utf-8")
    checked_manifest = NpcAssetManifest.from_json(output_manifest)
    checked_manifest.validate_population(NpcPopulation.from_json(output_population))
    return {
        "receipt": str(receipt_path),
        "manifest": str(output_manifest),
        "population": str(output_population),
        "fused_receipt": str(fused["receipt"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--source-archive", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--source-population", type=Path, required=True)
    parser.add_argument("--npc-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--output-population", type=Path, required=True)
    parser.add_argument("--bundle", default="smplx_office_neutral_v1")
    args = parser.parse_args()
    result = build_fused_accessory(
        recipe_path=args.recipe,
        source_archive=args.source_archive,
        source_manifest=args.source_manifest,
        source_population=args.source_population,
        npc_id=args.npc_id,
        output_dir=args.output_dir,
        output_manifest=args.output_manifest,
        output_population=args.output_population,
        bundle_id=args.bundle,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
