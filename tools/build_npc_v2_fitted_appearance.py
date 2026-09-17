#!/usr/bin/env python3
"""Build the production V2 fitted-clothing, no-hair NPC runtime bundle."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from stretch_mujoco.npc.appearance_pipeline.catalog import AppearanceCatalog
from stretch_mujoco.npc.assets import NpcAssetManifest
from stretch_mujoco.npc.schema import NpcPopulation


@dataclass(frozen=True)
class ObjMesh:
    vertices: np.ndarray
    vertex_lines: tuple[str, ...]
    uv_lines: tuple[str, ...]
    faces: tuple[tuple[tuple[int, int], ...], ...]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _resolve(recipe_path: Path, value: str) -> Path:
    return (recipe_path.parent / value).resolve()


def _update_population(
    path: Path,
    *,
    source_bundle_id: str,
    target_bundle_id: str,
    target_ids: dict[str, str],
    hair_none_id: str,
) -> dict[str, Any]:
    """Update only appearance values while preserving the authored JSON layout."""
    text = path.read_text(encoding="utf-8")
    current = json.loads(text)
    for npc_id, target_id in target_ids.items():
        embodiment = current["npcs"][npc_id]["embodiment"]
        source_id = str(embodiment["appearance"])
        if source_id != target_id:
            text = text.replace(f'"{source_id}"', f'"{target_id}"')
        source_hair = str(embodiment["appearance_config"]["hair"])
        if source_hair != hair_none_id:
            text = text.replace(
                f'"hair": "{source_hair}"', f'"hair": "{hair_none_id}"'
            )
    text = text.replace(
        f'"bundle": "{source_bundle_id}"', f'"bundle": "{target_bundle_id}"'
    )
    updated = json.loads(text)
    for npc_id, target_id in target_ids.items():
        embodiment = updated["npcs"][npc_id]["embodiment"]
        if (
            embodiment["bundle"] != target_bundle_id
            or embodiment["appearance"] != target_id
            or embodiment["visual_identity"] != target_id
            or embodiment["appearance_config"]["hair"] != hair_none_id
        ):
            raise ValueError(f"Population appearance update failed for {npc_id}")
    path.write_text(text, encoding="utf-8")
    return updated


def _load_obj(path: Path) -> ObjMesh:
    vertex_lines: list[str] = []
    uv_lines: list[str] = []
    vertices: list[list[float]] = []
    faces: list[tuple[tuple[int, int], ...]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("v "):
            vertex_lines.append(line)
            vertices.append([float(value) for value in line.split()[1:4]])
        elif line.startswith("vt "):
            uv_lines.append(line)
        elif line.startswith("f "):
            corners: list[tuple[int, int]] = []
            for token in line.split()[1:]:
                fields = token.split("/")
                if len(fields) < 2 or not fields[1]:
                    raise ValueError(f"OBJ face lacks UV coordinates: {path}")
                corners.append((int(fields[0]) - 1, int(fields[1])))
            faces.append(tuple(corners))
    if not vertices or not uv_lines or not faces:
        raise ValueError(f"OBJ needs vertices, UVs, and faces: {path}")
    return ObjMesh(
        vertices=np.asarray(vertices, dtype=np.float64),
        vertex_lines=tuple(vertex_lines),
        uv_lines=tuple(uv_lines),
        faces=tuple(faces),
    )


def _vertex_normals(mesh: ObjMesh) -> np.ndarray:
    normals = np.zeros_like(mesh.vertices)
    for face in mesh.faces:
        indices = [corner[0] for corner in face]
        origin = mesh.vertices[indices[0]]
        for index in range(1, len(indices) - 1):
            normal = np.cross(
                mesh.vertices[indices[index]] - origin,
                mesh.vertices[indices[index + 1]] - origin,
            )
            for vertex_index in (indices[0], indices[index], indices[index + 1]):
                normals[vertex_index] += normal
    lengths = np.linalg.norm(normals, axis=1)
    normals[lengths > 0] /= lengths[lengths > 0, None]
    normals[lengths == 0] = (0.0, 0.0, 1.0)
    return normals


def _selected_faces(reference: ObjMesh, garment: dict[str, Any]) -> tuple[int, ...]:
    selected: list[int] = []
    for index, face in enumerate(reference.faces):
        centre = reference.vertices[[corner[0] for corner in face]].mean(axis=0)
        if not float(garment["z_min"]) <= centre[2] <= float(garment["z_max"]):
            continue
        max_abs_x = garment.get("max_abs_x")
        high_z_bypass = garment.get("high_z_bypass")
        if max_abs_x is not None and abs(centre[0]) >= float(max_abs_x):
            if high_z_bypass is None or centre[2] <= float(high_z_bypass):
                continue
        selected.append(index)
    if not selected:
        raise ValueError(f"Garment selector produced no faces: {garment}")
    return tuple(selected)


def _rigid_follow(reference: np.ndarray, target: np.ndarray, points: np.ndarray) -> np.ndarray:
    reference_centre = reference.mean(axis=0)
    target_centre = target.mean(axis=0)
    covariance = (reference - reference_centre).T @ (target - target_centre)
    left, _, right_t = np.linalg.svd(covariance)
    rotation = right_t.T @ left.T
    if np.linalg.det(rotation) < 0:
        right_t[-1] *= -1
        rotation = right_t.T @ left.T
    return (points - reference_centre) @ rotation.T + target_centre


def _sole_vertices(side: int, sole: dict[str, Any]) -> np.ndarray:
    centre_x = side * float(sole["centre_x"])
    outline = [(float(y), float(width)) for y, width in sole["outline"]]
    ring = [(centre_x + side * width, y) for y, width in outline]
    ring += [(centre_x - side * width, y) for y, width in reversed(outline)]
    bottom_z, top_z = (float(value) for value in sole["z"])
    return np.asarray(
        [(x, y, bottom_z) for x, y in ring] + [(x, y, top_z) for x, y in ring],
        dtype=np.float64,
    )


def _sole_faces(vertex_offset: int, count: int, uv_index: int) -> list[str]:
    faces: list[tuple[int, int, int]] = []
    for index in range(1, count - 1):
        faces.append((vertex_offset, vertex_offset + index + 1, vertex_offset + index))
        faces.append(
            (vertex_offset + count, vertex_offset + count + index, vertex_offset + count + index + 1)
        )
    for index in range(count):
        next_index = (index + 1) % count
        lower = vertex_offset + index
        lower_next = vertex_offset + next_index
        upper = lower + count
        upper_next = lower_next + count
        faces.extend(((lower, lower_next, upper_next), (lower, upper_next, upper)))
    return ["f " + " ".join(f"{value}/{uv_index}" for value in face) for face in faces]


def _write_fitted_obj(
    source: ObjMesh,
    reference: ObjMesh,
    garment_faces: dict[str, tuple[int, ...]],
    garments: list[dict[str, Any]],
    foot_indices: dict[int, np.ndarray],
    soles: dict[int, np.ndarray],
    sole_uv: dict[int, int],
    destination: Path,
) -> None:
    if len(source.vertices) != len(reference.vertices) or source.faces != reference.faces:
        raise ValueError(f"Source frame topology differs from reference: {destination.name}")
    normals = _vertex_normals(source)
    appended_vertices: list[np.ndarray] = []
    appended_faces: list[str] = []
    vertex_count = len(source.vertices)
    for garment in garments:
        selected = garment_faces[str(garment["name"])]
        original_indices = sorted(
            {corner[0] for face_index in selected for corner in source.faces[face_index]}
        )
        remap = {source_index: vertex_count + index + 1 for index, source_index in enumerate(original_indices)}
        offset = float(garment["offset_m"])
        appended_vertices.extend(source.vertices[index] + normals[index] * offset for index in original_indices)
        for face_index in selected:
            appended_faces.append(
                "f "
                + " ".join(
                    f"{remap[vertex_index]}/{uv_index}"
                    for vertex_index, uv_index in source.faces[face_index]
                )
            )
        vertex_count += len(original_indices)

    for side in (-1, 1):
        indices = foot_indices[side]
        followed = _rigid_follow(reference.vertices[indices], source.vertices[indices], soles[side])
        appended_vertices.extend(followed)
        ring_count = len(soles[side]) // 2
        appended_faces.extend(_sole_faces(vertex_count + 1, ring_count, sole_uv[side]))
        vertex_count += len(followed)

    output = [
        "# Generated V2 fitted-clothing NPC frame; body UV topology is preserved.",
        *source.vertex_lines,
        *(f"v {x:.9g} {y:.9g} {z:.9g}" for x, y, z in appended_vertices),
        *source.uv_lines,
        *("f " + " ".join(f"{vertex + 1}/{uv}" for vertex, uv in face) for face in source.faces),
        *appended_faces,
    ]
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(output) + "\n", encoding="utf-8")


def build(recipe_path: str | Path) -> dict[str, object]:
    recipe_source = Path(recipe_path).resolve()
    recipe = json.loads(recipe_source.read_text(encoding="utf-8"))
    if recipe.get("schema_version") != 1:
        raise ValueError("Unsupported V2 appearance recipe schema_version")
    manifest_path = _resolve(recipe_source, recipe["manifest"])
    catalog_path = _resolve(recipe_source, recipe["catalog"])
    population_path = _resolve(recipe_source, recipe["population"])
    roster_path = _resolve(recipe_source, recipe["roster"])
    lock_path = _resolve(recipe_source, recipe["selection_lock"])
    asset_root = manifest_path.parent
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    population = json.loads(population_path.read_text(encoding="utf-8"))
    roster = json.loads(roster_path.read_text(encoding="utf-8"))
    selection = json.loads(lock_path.read_text(encoding="utf-8"))
    if len(selection.get("npcs", {})) != 10:
        raise ValueError("V2 production selection must contain exactly ten NPCs")

    source_bundle_id = str(recipe["source_bundle"])
    target_bundle_id = str(recipe["target_bundle"])
    target_topology_id = str(recipe["target_topology_id"])
    source_bundle = manifest["bundles"][source_bundle_id]
    reference_relative = source_bundle["clips"]["idle"]["frames"][0]
    reference = _load_obj(asset_root / reference_relative)
    garments = list(recipe["garments"])
    garment_faces = {
        str(garment["name"]): _selected_faces(reference, garment) for garment in garments
    }
    shoe_faces = garment_faces["shoes"]
    shoe_indices = sorted(
        {corner[0] for face_index in shoe_faces for corner in reference.faces[face_index]}
    )
    foot_indices = {
        side: np.asarray(
            [index for index in shoe_indices if side * reference.vertices[index, 0] > 0.02],
            dtype=np.int64,
        )
        for side in (-1, 1)
    }
    if any(len(indices) < 16 for indices in foot_indices.values()):
        raise ValueError("Reference frame has too few foot vertices for fitted soles")
    soles = {side: _sole_vertices(side, recipe["sole"]) for side in (-1, 1)}
    sole_uv: dict[int, int] = {}
    for side in (-1, 1):
        for face_index in shoe_faces:
            face = reference.faces[face_index]
            if side * reference.vertices[[corner[0] for corner in face], 0].mean() > 0.02:
                sole_uv[side] = face[0][1]
                break
    if set(sole_uv) != {-1, 1}:
        raise ValueError("Could not locate shoe UVs for fitted soles")

    output_dir = asset_root / str(recipe["frame_output"])
    frame_mapping: dict[str, str] = {}
    frame_hashes: dict[str, str] = {}
    for clip in source_bundle["clips"].values():
        for source_relative in clip["frames"]:
            if source_relative in frame_mapping:
                continue
            destination = output_dir / Path(source_relative).name
            _write_fitted_obj(
                _load_obj(asset_root / source_relative),
                reference,
                garment_faces,
                garments,
                foot_indices,
                soles,
                sole_uv,
                destination,
            )
            target_relative = destination.relative_to(asset_root).as_posix()
            frame_mapping[source_relative] = target_relative
            frame_hashes[target_relative] = _sha256(destination)

    hair_none = recipe["hair_none"]
    hair_none_path = asset_root / str(hair_none["image"])
    with Image.open(asset_root / catalog["base"]) as base_image:
        transparent = Image.new("RGBA", base_image.size, (0, 0, 0, 0))
    hair_none_path.parent.mkdir(parents=True, exist_ok=True)
    transparent.save(hair_none_path, compress_level=9)
    hair_none_id = str(hair_none["id"])
    catalog["layers"][hair_none_id] = {
        "category": "hair",
        "image": hair_none_path.relative_to(asset_root).as_posix(),
        "sha256": _sha256(hair_none_path),
    }

    roster_identities = {item["id"]: item for item in roster["identities"]}
    roster_layers = {item["id"]: item for item in roster["layers"]}
    suffix = str(recipe["appearance_suffix"])
    target_ids: dict[str, str] = {}
    for npc_id, locked in selection["npcs"].items():
        source_id = str(locked["appearance"])
        target_id = source_id + suffix
        identity = roster_identities[source_id]
        layers = [
            hair_none_id if roster_layers[layer_id]["category"] == "hair" else layer_id
            for layer_id in [*identity["slots"], *identity.get("details", [])]
        ]
        catalog["identities"][target_id] = {
            "appearance_id": target_id,
            "layers": layers,
            "seed": int(identity["seed"]),
            "traits": {
                **{str(key): str(value) for key, value in identity.get("traits", {}).items()},
                "hair": "none",
                "glasses": (
                    "painted_2d" if "glasses_thin_round_v4" in layers else "none"
                ),
            },
        }
        target_ids[npc_id] = target_id
    _write_json(catalog_path, catalog)

    baked_catalog = AppearanceCatalog.from_json(catalog_path)
    appearances: dict[str, dict[str, object]] = {}
    appearance_hashes: dict[str, str] = {}
    for target_id in target_ids.values():
        baked = baked_catalog.bake(target_id, asset_root / "appearances" / target_id)
        fragment = baked.manifest_fragment(relative_to=asset_root)
        fragment["texture_topology_id"] = target_topology_id
        appearances[target_id] = fragment
        for image in baked.textures.values():
            appearance_hashes[image.relative_to(asset_root).as_posix()] = _sha256(image)

    target_bundle = copy.deepcopy(source_bundle)
    target_bundle["topology_id"] = target_topology_id
    target_bundle["appearances"] = appearances
    for clip in target_bundle["clips"].values():
        clip["frames"] = [frame_mapping[path] for path in clip["frames"]]
    target_bundle["sha256"].update(frame_hashes)
    target_bundle["sha256"].update(appearance_hashes)
    manifest["bundles"][target_bundle_id] = target_bundle
    _write_json(manifest_path, manifest)

    population = _update_population(
        population_path,
        source_bundle_id=source_bundle_id,
        target_bundle_id=target_bundle_id,
        target_ids=target_ids,
        hair_none_id=hair_none_id,
    )

    NpcAssetManifest.from_json(manifest_path).validate_population(
        NpcPopulation.from_json(population_path)
    )
    receipt = {
        "schema_version": 1,
        "recipe": str(recipe_source),
        "recipe_sha256": _sha256(recipe_source),
        "source_bundle": source_bundle_id,
        "target_bundle": target_bundle_id,
        "target_topology_id": target_topology_id,
        "frames": len(frame_mapping),
        "npcs": len(target_ids),
        "hair_layer": hair_none_id,
        "glasses": "original accessory_2d texture layer",
        "garment_face_counts": {name: len(faces) for name, faces in garment_faces.items()},
        "manifest_sha256": _sha256(manifest_path),
        "catalog_sha256": _sha256(catalog_path),
        "population_sha256": _sha256(population_path),
    }
    receipt_path = asset_root / str(recipe["receipt"])
    _write_json(receipt_path, receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(build(args.recipe), indent=2))


if __name__ == "__main__":
    main()
