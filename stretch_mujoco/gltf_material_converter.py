"""Convert material-split glTF/GLB assets into MuJoCo-compatible OBJ and PNG files."""

from __future__ import annotations

import base64
import json
import math
import os
import re
import shutil
import struct
import subprocess
from collections import defaultdict
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import trimesh
from PIL import Image


# Bumped whenever the coordinate/material conversion changes. This prevents a
# stale OBJ cache from masking fixes to HSSD stage alignment.
CONVERTER_VERSION = 2
HABITAT_TO_MUJOCO = np.array(((1.0, 0.0, 0.0), (0.0, 0.0, -1.0), (0.0, 1.0, 0.0)))


@dataclass(frozen=True)
class ConvertedMaterialPart:
    obj_path: Path
    texture_path: Path | None
    rgba: tuple[float, float, float, float]
    specular: float
    shininess: float
    reflectance: float


@dataclass(frozen=True)
class ConvertedGltfAsset:
    parts: tuple[ConvertedMaterialPart, ...]
    bounds: np.ndarray


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]", "_", value).strip("_") or "material"


def _read_glb(path: Path) -> tuple[dict[str, Any], bytes]:
    with path.open("rb") as stream:
        magic, version, _ = struct.unpack("<4sII", stream.read(12))
        if magic != b"glTF" or version != 2:
            raise ValueError(f"Expected a glTF 2.0 binary: {path}")
        json_length, json_type = struct.unpack("<II", stream.read(8))
        if json_type != 0x4E4F534A:
            raise ValueError(f"Missing GLB JSON chunk: {path}")
        document = json.loads(stream.read(json_length).decode("utf-8").rstrip("\x00 "))
        binary = b""
        chunk_header = stream.read(8)
        if chunk_header:
            binary_length, binary_type = struct.unpack("<II", chunk_header)
            if binary_type == 0x004E4942:
                binary = stream.read(binary_length)
    return document, binary


def _buffer_view(document: dict[str, Any], binary: bytes, index: int) -> bytes:
    view = document["bufferViews"][index]
    offset = int(view.get("byteOffset", 0))
    return binary[offset : offset + int(view["byteLength"])]


def _image_bytes(
    document: dict[str, Any], binary: bytes, source_path: Path, image_index: int
) -> tuple[bytes, str]:
    image = document["images"][image_index]
    mime_type = image.get("mimeType", "")
    if "bufferView" in image:
        return _buffer_view(document, binary, int(image["bufferView"])), mime_type
    uri = image["uri"]
    if uri.startswith("data:"):
        header, encoded = uri.split(",", 1)
        return base64.b64decode(encoded), header.split(";", 1)[0].removeprefix("data:")
    data = (source_path.parent / uri).read_bytes()
    return data, mime_type


def find_ktx_command(explicit: Path | None = None) -> tuple[Path, Path | None]:
    candidates = []
    if explicit is not None:
        candidates.append(explicit)
    if os.environ.get("KTX_COMMAND"):
        candidates.append(Path(os.environ["KTX_COMMAND"]))
    discovered = shutil.which("ktx")
    if discovered:
        candidates.append(Path(discovered))
    candidates.append(
        Path.home() / ".cache" / "stretch_mujoco" / "ktx-4.4.2" / "usr" / "bin" / "ktx"
    )
    candidates.append(Path("/tmp/ktx-software/usr/bin/ktx"))
    for candidate in candidates:
        if candidate.is_file():
            sibling_lib = candidate.parent.parent / "lib"
            return candidate, sibling_lib if sibling_lib.is_dir() else None
    raise FileNotFoundError(
        "Khronos 'ktx' is required for HSSD BasisU textures. Install KTX-Software or set "
        "KTX_COMMAND to the executable."
    )


def _decode_image(
    data: bytes,
    mime_type: str,
    output: Path,
    ktx_command: Path | None,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if mime_type == "image/ktx2" or data.startswith(b"\xabKTX 20\xbb\r\n\x1a\n"):
        command, lib_dir = find_ktx_command(ktx_command)
        source = output.with_suffix(".ktx2")
        source.write_bytes(data)
        environment = os.environ.copy()
        if lib_dir is not None:
            existing = environment.get("LD_LIBRARY_PATH", "")
            environment["LD_LIBRARY_PATH"] = f"{lib_dir}:{existing}" if existing else str(lib_dir)
        result = subprocess.run(
            [str(command), "extract", "--transcode", "rgba8", str(source), str(output)],
            check=False,
            capture_output=True,
            text=True,
            env=environment,
        )
        source.unlink(missing_ok=True)
        if result.returncode:
            raise RuntimeError(f"KTX decode failed for {output}: {result.stderr.strip()}")
    else:
        with Image.open(BytesIO(data)) as image:
            image.convert("RGBA").save(output)
    # Normalize palette PNGs emitted by some KTX builds to regular RGBA PNGs.
    with Image.open(output) as image:
        image.convert("RGBA").save(output)


def _texture_source(document: dict[str, Any], texture_index: int) -> int:
    texture = document["textures"][texture_index]
    basis = texture.get("extensions", {}).get("KHR_texture_basisu")
    return int(basis["source"] if basis is not None else texture["source"])


def _material_values(material: dict[str, Any]) -> tuple[tuple[float, ...], float, float, float]:
    pbr = material.get("pbrMetallicRoughness", {})
    rgba = tuple(float(value) for value in pbr.get("baseColorFactor", (1, 1, 1, 1)))
    roughness = float(pbr.get("roughnessFactor", 1.0))
    metallic = float(pbr.get("metallicFactor", 0.0))
    specular_extension = material.get("extensions", {}).get("KHR_materials_specular", {})
    specular_factor = float(specular_extension.get("specularFactor", 0.35))
    specular = min(1.0, max(0.0, 0.08 + 0.55 * metallic + 0.25 * specular_factor))
    shininess = min(1.0, max(0.0, 1.0 - roughness))
    reflectance = min(1.0, max(0.0, metallic * 0.35))
    return rgba, specular, shininess, reflectance


def _transform_uv(uv: np.ndarray, texture_info: dict[str, Any]) -> np.ndarray:
    transform = texture_info.get("extensions", {}).get("KHR_texture_transform", {})
    scale = np.asarray(transform.get("scale", (1.0, 1.0)), dtype=float)
    offset = np.asarray(transform.get("offset", (0.0, 0.0)), dtype=float)
    rotation = float(transform.get("rotation", 0.0))
    result = np.asarray(uv, dtype=float) * scale
    if rotation:
        matrix = np.array(
            ((math.cos(rotation), -math.sin(rotation)), (math.sin(rotation), math.cos(rotation)))
        )
        result = result @ matrix.T
    return result + offset


def _write_obj(path: Path, vertices: np.ndarray, faces: np.ndarray, uv: np.ndarray | None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"v {x:.9g} {y:.9g} {z:.9g}" for x, y, z in vertices]
    if uv is not None and len(uv) == len(vertices):
        lines.extend(f"vt {u:.9g} {v:.9g}" for u, v in uv)
        lines.extend(
            "f " + " ".join(f"{int(index) + 1}/{int(index) + 1}" for index in face)
            for face in faces
        )
    else:
        lines.extend("f " + " ".join(str(int(index) + 1) for index in face) for face in faces)
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def _material_lookup(document: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    lookup: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for index, material in enumerate(document.get("materials", [])):
        lookup[material.get("name", f"material_{index}")].append(material)
    return lookup


def _load_cached(output_dir: Path, source: Path) -> ConvertedGltfAsset | None:
    manifest_path = output_dir / "manifest.json"
    if not manifest_path.is_file():
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source_stat = source.stat()
    if (
        manifest.get("converter_version") != CONVERTER_VERSION
        or manifest.get("source_size") != source_stat.st_size
        or manifest.get("source_mtime_ns") != source_stat.st_mtime_ns
    ):
        return None
    parts = tuple(
        ConvertedMaterialPart(
            obj_path=output_dir / part["obj"],
            texture_path=output_dir / part["texture"] if part["texture"] else None,
            rgba=tuple(part["rgba"]),
            specular=part["specular"],
            shininess=part["shininess"],
            reflectance=part["reflectance"],
        )
        for part in manifest["parts"]
    )
    if not all(part.obj_path.is_file() for part in parts):
        return None
    if not all(part.texture_path is None or part.texture_path.is_file() for part in parts):
        return None
    return ConvertedGltfAsset(parts, np.asarray(manifest["bounds"], dtype=float))


def convert_glb_with_materials(
    source: Path,
    output_dir: Path,
    *,
    ktx_command: Path | None = None,
) -> ConvertedGltfAsset:
    cached = _load_cached(output_dir, source)
    if cached is not None:
        return cached
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    document, binary = _read_glb(source)
    material_lookup = _material_lookup(document)
    loaded = trimesh.load(source, force="scene", process=False)
    scene = loaded if isinstance(loaded, trimesh.Scene) else trimesh.Scene(loaded)
    parts = []
    all_vertices = []
    decoded_images: dict[int, Path] = {}

    for part_index, node in enumerate(scene.graph.nodes_geometry):
        transform, geometry_name = scene.graph[node]
        mesh = scene.geometry[geometry_name].copy()
        mesh.apply_transform(transform)
        vertices = np.asarray(mesh.vertices, dtype=float) @ HABITAT_TO_MUJOCO.T
        faces = np.asarray(mesh.faces, dtype=np.int64)
        all_vertices.append(vertices)
        material_name = getattr(getattr(mesh.visual, "material", None), "name", None)
        matches = material_lookup.get(material_name or "", [])
        material = matches[0] if matches else {}
        rgba, specular, shininess, reflectance = _material_values(material)
        pbr = material.get("pbrMetallicRoughness", {})
        texture_info = pbr.get("baseColorTexture")
        texture_path = None
        uv = getattr(mesh.visual, "uv", None)
        if texture_info is not None and uv is not None:
            image_index = _texture_source(document, int(texture_info["index"]))
            if image_index not in decoded_images:
                texture_path = output_dir / "textures" / f"image_{image_index:03d}.png"
                data, mime_type = _image_bytes(document, binary, source, image_index)
                _decode_image(data, mime_type, texture_path, ktx_command)
                decoded_images[image_index] = texture_path
            texture_path = decoded_images[image_index]
            uv = _transform_uv(np.asarray(uv, dtype=float), texture_info)
        else:
            uv = None
        obj_name = f"part_{part_index:03d}_{_safe_name(material_name or geometry_name)}.obj"
        obj_path = output_dir / "meshes" / obj_name
        _write_obj(obj_path, vertices, faces, uv)
        parts.append(
            ConvertedMaterialPart(
                obj_path=obj_path,
                texture_path=texture_path,
                rgba=rgba,
                specular=specular,
                shininess=shininess,
                reflectance=reflectance,
            )
        )

    vertices = np.vstack(all_vertices)
    bounds = np.stack((vertices.min(axis=0), vertices.max(axis=0)))
    source_stat = source.stat()
    manifest = {
        "converter_version": CONVERTER_VERSION,
        "source": str(source),
        "source_size": source_stat.st_size,
        "source_mtime_ns": source_stat.st_mtime_ns,
        "bounds": bounds.tolist(),
        "parts": [
            {
                "obj": str(part.obj_path.relative_to(output_dir)),
                "texture": (
                    str(part.texture_path.relative_to(output_dir)) if part.texture_path else None
                ),
                "rgba": list(part.rgba),
                "specular": part.specular,
                "shininess": part.shininess,
                "reflectance": part.reflectance,
            }
            for part in parts
        ],
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return ConvertedGltfAsset(tuple(parts), bounds)
