#!/usr/bin/env python3
"""Build a receipt-backed, head-follow fused NPC accessory runtime projection."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import shutil
import sys
from pathlib import Path
from types import ModuleType

from stretch_mujoco.npc.appearance_pipeline.accessory_recipe import (
    FusedAccessoryRecipe,
    recipe_sha256,
)
from stretch_mujoco.npc.appearance_pipeline.obj_accessory_surface import (
    SurfaceFallbackArtifacts,
    bake_smplx_head_surface_fallback,
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
    payload: dict[str, object],
    *,
    source_population_path: Path,
    npc_id: str,
    accessory_id: str,
    fused_bundle_id: str | None = None,
    fused_appearance_id: str | None = None,
    fused_visual_identity: str | None = None,
    appearance_catalog_path: Path | None = None,
    manifest_path: Path,
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
    if fused_bundle_id is not None:
        embodiment["bundle"] = fused_bundle_id
    if fused_appearance_id is not None:
        embodiment["appearance"] = fused_appearance_id
    if fused_visual_identity is not None:
        embodiment["visual_identity"] = fused_visual_identity
    config = embodiment.get("appearance_config")
    if isinstance(config, dict) and isinstance(config.get("accessories"), list):
        config["accessories"] = [item for item in config["accessories"] if item != accessory_id]
    for field in ("scene", "appearance_catalog"):
        value = payload.get(field)
        if isinstance(value, str):
            payload[field] = str((source_population_path.parent / value).resolve())
    if appearance_catalog_path is not None:
        payload["appearance_catalog"] = str(appearance_catalog_path.resolve())
    payload["asset_manifest"] = str(manifest_path.resolve())
    return payload


def source_appearance_id(payload: dict[str, object], npc_id: str) -> str:
    """Read the one source appearance whose body atlas a fallback extends."""
    npcs = payload.get("npcs")
    if not isinstance(npcs, dict) or not isinstance(npcs.get(npc_id), dict):
        raise ValueError(f"Population does not define NPC '{npc_id}'")
    embodiment = npcs[npc_id].get("embodiment")
    if not isinstance(embodiment, dict) or not isinstance(embodiment.get("appearance"), str):
        raise ValueError(f"NPC '{npc_id}' has no appearance")
    return embodiment["appearance"]


def source_visual_identity(payload: dict[str, object], npc_id: str) -> str | None:
    npcs = payload.get("npcs")
    if not isinstance(npcs, dict) or not isinstance(npcs.get(npc_id), dict):
        raise ValueError(f"Population does not define NPC '{npc_id}'")
    embodiment = npcs[npc_id].get("embodiment")
    if not isinstance(embodiment, dict):
        raise ValueError(f"NPC '{npc_id}' has no embodiment")
    identity = embodiment.get("visual_identity")
    if identity is not None and not isinstance(identity, str):
        raise ValueError(f"NPC '{npc_id}' visual_identity must be a string")
    return identity


def build_surface_fallback_catalog(
    *,
    source_population: dict[str, object],
    source_population_path: Path,
    source_identity: str | None,
    fallback_appearance_id: str,
    accessory_id: str,
    output_dir: Path,
) -> tuple[Path | None, str | None]:
    """Project a catalog identity for a target-only fallback appearance.

    The source catalog remains authoritative for the base and layers.  This
    projection only maps one cloned identity to the recipe-generated atlas, so
    the runtime population keeps the normal visual-identity validation path.
    """
    raw_catalog = source_population.get("appearance_catalog")
    if raw_catalog is None:
        return None, None
    if not isinstance(raw_catalog, str) or not raw_catalog or source_identity is None:
        raise ValueError("Surface fallback population requires an appearance catalog and identity")
    source_catalog = (source_population_path.parent / raw_catalog).resolve()
    payload = json.loads(source_catalog.read_text(encoding="utf-8"))
    identities = payload.get("identities")
    if not isinstance(identities, dict) or not isinstance(identities.get(source_identity), dict):
        raise ValueError(f"Appearance catalog has no identity '{source_identity}'")
    projected_identity = f"{source_identity}__{accessory_id}__surface_fallback"
    payload = copy.deepcopy(payload)
    payload["base"] = str((source_catalog.parent / payload["base"]).resolve())
    for layer in payload["layers"].values():
        if not isinstance(layer, dict) or not isinstance(layer.get("image"), str):
            raise ValueError("Appearance catalog layer is malformed")
        layer["image"] = str((source_catalog.parent / layer["image"]).resolve())
    if payload.get("semantic_mask_manifest") is not None:
        payload["semantic_mask_manifest"] = str(
            (source_catalog.parent / payload["semantic_mask_manifest"]).resolve()
        )
    cloned = copy.deepcopy(payload["identities"][source_identity])
    cloned["appearance_id"] = fallback_appearance_id
    payload["identities"][projected_identity] = cloned
    catalog_path = output_dir / "appearance" / f"{projected_identity}.catalog.json"
    catalog_path.parent.mkdir(parents=True, exist_ok=True)
    catalog_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return catalog_path, projected_identity


def apply_surface_fallback(
    *,
    manifest: dict[str, object],
    manifest_path: Path,
    source_manifest: Path,
    output_dir: Path,
    source_bundle_id: str,
    fused_bundle_id: str,
    source_appearance_id: str,
    recipe: FusedAccessoryRecipe,
) -> tuple[str, SurfaceFallbackArtifacts] | tuple[None, None]:
    """Create a target-only appearance projection when the recipe requests one."""
    fallback = recipe.render_policy.surface_fallback
    if fallback is None:
        return None, None
    bundles = manifest.get("bundles")
    if not isinstance(bundles, dict) or not isinstance(bundles.get(fused_bundle_id), dict):
        raise ValueError(f"Fused bundle '{fused_bundle_id}' is missing from generated manifest")
    bundle = bundles[fused_bundle_id]
    appearances = bundle.get("appearances")
    if not isinstance(appearances, dict) or not isinstance(
        appearances.get(source_appearance_id), dict
    ):
        raise ValueError(f"Fused bundle has no source appearance '{source_appearance_id}'")
    source_appearance = appearances[source_appearance_id]
    textures = source_appearance.get("textures")
    if not isinstance(textures, dict) or not isinstance(textures.get("body"), str):
        raise ValueError("SMPL-X head surface fallback requires a body texture")
    source_bundle = json.loads(source_manifest.read_text(encoding="utf-8"))["bundles"]
    source_frame = source_bundle[source_bundle_id]["clips"]["idle"]["frames"][0]
    artifact_dir = output_dir / "appearance"
    fused_appearance_id = f"{source_appearance_id}__{recipe.accessory_id}__surface_fallback"
    texture_path = artifact_dir / f"{fused_appearance_id}.png"
    mask_path = artifact_dir / f"{fused_appearance_id}.mask.png"
    artifacts = bake_smplx_head_surface_fallback(
        source_atlas=(source_manifest.parent / textures["body"]).resolve(),
        body_frame=(source_manifest.parent / source_frame).resolve(),
        destination_texture=texture_path,
        destination_mask=mask_path,
        fallback=fallback,
    )
    output_relative = texture_path.relative_to(manifest_path.parent).as_posix()
    appearances[fused_appearance_id] = {
        **source_appearance,
        "textures": {**textures, "body": output_relative},
    }
    hashes = bundle.get("sha256")
    if not isinstance(hashes, dict):
        raise ValueError(f"Fused bundle '{fused_bundle_id}' has no sha256 map")
    hashes[output_relative] = _sha256(texture_path)
    return fused_appearance_id, artifacts


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


def clear_derived_projection(output_dir: Path) -> None:
    """Remove stale generated OBJ projections before rebuilding one recipe.

    ``output_dir`` is a generated runtime projection, never a source asset
    directory.  Clearing only tool-owned projection subdirectories prevents
    old clip frames or fallback textures from surviving a rebuild while leaving
    receipts/manifests in their separate configured locations untouched.
    """
    output_dir = output_dir.resolve()
    if output_dir.name in {"", ".", ".."}:
        raise ValueError("Refusing to clear an unsafe generated output directory")
    for name in ("accessory", "frames", "appearance"):
        child = output_dir / name
        if child.exists():
            if not child.is_dir():
                raise ValueError(f"Generated projection path is not a directory: {child}")
            shutil.rmtree(child)


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
    source_format: str = "nested_zip_obj",
    nested_archive_member: str | None = "source/cap.zip",
    obj_member: str | None = "cap.obj",
    source_unit_scale: float = 0.01,
) -> dict[str, str]:
    """Materialize every derived projection from one immutable pose recipe."""
    recipe = FusedAccessoryRecipe.from_json(recipe_path)
    source_population_payload = json.loads(source_population.read_text(encoding="utf-8"))
    source_appearance = (
        source_appearance_id(source_population_payload, npc_id)
        if recipe.render_policy.surface_fallback is not None
        else None
    )
    source_identity = (
        source_visual_identity(source_population_payload, npc_id)
        if recipe.render_policy.surface_fallback is not None
        else None
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    clear_derived_projection(output_dir)
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    output_population.parent.mkdir(parents=True, exist_ok=True)
    prepare = _tool_module("prepare_obj_accessory.py")
    fuse = _tool_module("fuse_obj_accessory.py")
    accessory_dir = output_dir / "accessory"
    prepared = prepare.prepare(
        source_archive.resolve(),
        source_manifest.resolve(),
        accessory_dir,
        recipe.accessory_id,
        lateral_offset_m=recipe.lateral_offset_m,
        mesh_scale=recipe.mesh_scale,
        head_clearance_m=recipe.head_clearance_m,
        back_offset_m=recipe.back_offset_m,
        back_tilt_degrees=recipe.back_tilt_degrees,
        roll_degrees=recipe.roll_degrees,
        yaw_degrees=recipe.yaw_degrees,
        source_format=source_format,
        nested_archive_member=nested_archive_member,
        obj_member=obj_member,
        source_unit_scale=source_unit_scale,
        source_vertical_anchor=recipe.source_vertical_anchor,
    )
    receipt_path = Path(prepared["receipt"])
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt_fields = {
        "asset_id": recipe.accessory_id,
        "lateral_offset_m": recipe.lateral_offset_m,
        "mesh_scale": recipe.mesh_scale,
        "head_clearance_m": recipe.head_clearance_m,
        "back_offset_m": recipe.back_offset_m,
        "back_tilt_degrees": recipe.back_tilt_degrees,
        "roll_degrees": recipe.roll_degrees,
        "yaw_degrees": recipe.yaw_degrees,
    }
    for field, expected in receipt_fields.items():
        if receipt.get(field) != expected:
            raise ValueError(f"Generated receipt does not match recipe field '{field}'")
    receipt.update(
        {
            "recipe": str(recipe_path.resolve()),
            "recipe_sha256": recipe_sha256(recipe_path),
            "attachment_mode": recipe.attachment_mode,
            "render_policy": recipe.render_policy.as_dict(),
        }
    )
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    fused_bundle_id = f"{bundle_id}__fused__{recipe.accessory_id}"
    fused = fuse.fuse_manifest(
        source_manifest.resolve(),
        Path(prepared["mesh"]),
        Path(prepared["anchors"]),
        output_dir / "frames",
        output_manifest.resolve(),
        bundle_id=bundle_id,
        fused_bundle_id=fused_bundle_id,
        accessory_uv=recipe.accessory_uv,
        double_sided=recipe.render_policy.double_sided,
    )
    manifest = json.loads(output_manifest.read_text(encoding="utf-8"))
    fused_appearance_id: str | None = None
    fallback_artifacts: SurfaceFallbackArtifacts | None = None
    if source_appearance is not None:
        fused_appearance_id, fallback_artifacts = apply_surface_fallback(
            manifest=manifest,
            manifest_path=output_manifest,
            source_manifest=source_manifest,
            output_dir=output_dir,
            source_bundle_id=bundle_id,
            fused_bundle_id=fused_bundle_id,
            source_appearance_id=source_appearance,
            recipe=recipe,
        )
    if fallback_artifacts is not None:
        receipt["surface_fallback"] = {
            "texture": str(fallback_artifacts.texture.resolve()),
            "texture_sha256": _sha256(fallback_artifacts.texture),
            "mask": str(fallback_artifacts.mask.resolve()),
            "mask_sha256": _sha256(fallback_artifacts.mask),
            "pixels": fallback_artifacts.pixels,
        }
    fallback_catalog_path: Path | None = None
    fallback_visual_identity: str | None = None
    if fused_appearance_id is not None:
        fallback_catalog_path, fallback_visual_identity = build_surface_fallback_catalog(
            source_population=source_population_payload,
            source_population_path=source_population.resolve(),
            source_identity=source_identity,
            fallback_appearance_id=fused_appearance_id,
            accessory_id=recipe.accessory_id,
            output_dir=output_dir,
        )
        if fallback_catalog_path is not None:
            receipt["surface_fallback"]["appearance_catalog"] = str(fallback_catalog_path.resolve())
            receipt["surface_fallback"]["appearance_catalog_sha256"] = _sha256(
                fallback_catalog_path
            )
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    bind_recipe_accessory(
        manifest,
        bundle_id=fused_bundle_id,
        recipe=recipe,
        mesh_path=Path(prepared["mesh"]),
        anchors_path=Path(prepared["anchors"]),
        manifest_path=output_manifest,
    )
    manifest["fused_accessories"][fused_bundle_id].update(
        {
            "recipe": str(recipe_path.resolve()),
            "recipe_sha256": recipe_sha256(recipe_path),
            "receipt": str(receipt_path.resolve()),
            "receipt_sha256": _sha256(receipt_path),
            "attachment_mode": recipe.attachment_mode,
            "fused_bundle": fused_bundle_id,
            "render_policy": recipe.render_policy.as_dict(),
            "surface_fallback": receipt.get("surface_fallback"),
        }
    )
    output_manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    population = runtime_population_payload(
        source_population_payload,
        source_population_path=source_population.resolve(),
        npc_id=npc_id,
        accessory_id=recipe.accessory_id,
        fused_bundle_id=fused_bundle_id,
        fused_appearance_id=fused_appearance_id,
        fused_visual_identity=fallback_visual_identity,
        appearance_catalog_path=fallback_catalog_path,
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
    parser.add_argument("--source-format", required=True)
    parser.add_argument("--nested-archive-member")
    parser.add_argument("--obj-member")
    parser.add_argument("--source-unit-scale", type=float, required=True)
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
        source_format=args.source_format,
        nested_archive_member=args.nested_archive_member,
        obj_member=args.obj_member,
        source_unit_scale=args.source_unit_scale,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
