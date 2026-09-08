import importlib.util
import io
import json
import struct
import zipfile
from pathlib import Path

import numpy as np
import pytest


def _module():
    path = Path(__file__).parents[1] / "tools" / "prepare_obj_accessory.py"
    spec = importlib.util.spec_from_file_location("prepare_obj_accessory", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _nested_cap(path: Path) -> None:
    obj = b"mtllib cap.mtl\nv 0 10 0\nv 20 10 0\nv 0 20 10\nf 1 2 3\n"
    nested = io.BytesIO()
    with zipfile.ZipFile(nested, "w") as archive:
        archive.writestr("cap.obj", obj)
    with zipfile.ZipFile(path, "w") as outer:
        outer.writestr("source/cap.zip", nested.getvalue())


def _triangle_glb(path: Path) -> None:
    binary = struct.pack("<9f3H", 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 2)
    document = {
        "asset": {"version": "2.0"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0, "translation": [1, 2, 3]}],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0}, "indices": 1}]}],
        "buffers": [{"byteLength": len(binary)}],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": 36},
            {"buffer": 0, "byteOffset": 36, "byteLength": 6},
        ],
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3"},
            {"bufferView": 1, "componentType": 5123, "count": 3, "type": "SCALAR"},
        ],
    }
    json_chunk = json.dumps(document, separators=(",", ":")).encode("utf-8")
    json_chunk += b" " * (-len(json_chunk) % 4)
    binary += b"\0" * (-len(binary) % 4)
    payload = (
        struct.pack("<4sII", b"glTF", 2, 12 + 8 + len(json_chunk) + 8 + len(binary))
        + struct.pack("<II", len(json_chunk), 0x4E4F534A)
        + json_chunk
        + struct.pack("<II", len(binary), 0x004E4942)
        + binary
    )
    path.write_bytes(payload)


def test_normalized_source_cap_is_meter_z_up_and_records_nested_provenance(tmp_path: Path) -> None:
    module = _module()
    archive = tmp_path / "cap.zip"
    _nested_cap(archive)

    source, provenance = module._source_obj(archive)
    normalized = module.normalized_obj(source, mesh_scale=1.0, back_tilt_degrees=0.0).decode(
        "utf-8"
    )
    vertices = np.array(
        [
            [float(value) for value in line.split()[1:4]]
            for line in normalized.splitlines()
            if line.startswith("v ")
        ]
    )

    assert provenance["nested_archive_member"] == "source/cap.zip"
    assert provenance["source_obj_member"] == "cap.obj"
    assert np.isclose(vertices[:, 2].min(), 0.0)
    assert np.isclose(vertices[:, 2].max(), 0.1)
    assert np.isclose(vertices[:, :2].ptp(axis=0)[0], 0.2)
    assert "mtllib" not in normalized
    scaled = module.normalized_obj(source, mesh_scale=1.3, back_tilt_degrees=0.0).decode("utf-8")
    scaled_vertices = np.array(
        [
            [float(value) for value in line.split()[1:4]]
            for line in scaled.splitlines()
            if line.startswith("v ")
        ]
    )
    assert np.isclose(scaled_vertices[:, :2].ptp(axis=0)[0], 0.26)
    tilted = module.normalized_obj(source, mesh_scale=1.3, back_tilt_degrees=13.0).decode("utf-8")
    tilted_vertices = np.array(
        [
            [float(value) for value in line.split()[1:4]]
            for line in tilted.splitlines()
            if line.startswith("v ")
        ]
    )
    assert not np.allclose(tilted_vertices, scaled_vertices)


def test_glb_source_is_converted_to_obj_with_node_transform(tmp_path: Path) -> None:
    module = _module()
    source_asset = tmp_path / "cap.glb"
    _triangle_glb(source_asset)

    source, provenance = module._source_obj(
        source_asset,
        source_format="glb",
        nested_archive_member=None,
        obj_member=None,
    )

    converted = source.decode("utf-8")
    assert "v 1 2 3" in converted
    assert "f 1 2 3" in converted
    assert provenance["source_glb_sha256"] == provenance["source_asset_sha256"]


def test_prepare_writes_production_receipt_and_all_clip_anchors(
    tmp_path: Path, monkeypatch
) -> None:
    module = _module()
    archive = tmp_path / "cap.zip"
    _nested_cap(archive)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"bundles": {"smplx_office_neutral_v1": {"clips": {}}}}))
    captured: dict[str, float] = {}

    def anchors(
        _manifest: Path, *, head_clearance_m: float, back_offset_m: float
    ) -> dict[str, list[dict[str, list[int]]]]:
        captured["head_clearance_m"] = head_clearance_m
        captured["back_offset_m"] = back_offset_m
        return {"idle": [{"position": [0, 0, 1]}]}

    monkeypatch.setattr(module, "head_top_anchors", anchors)

    paths = module.prepare(
        archive,
        manifest,
        tmp_path / "output",
        "baseball_cap_v1",
        head_clearance_m=-0.026,
        back_offset_m=0.098,
        mesh_scale=1.3,
        back_tilt_degrees=13.0,
    )
    receipt = json.loads(Path(paths["receipt"]).read_text())

    assert receipt["asset_quality"] == "production"
    assert receipt["head_clearance_m"] == -0.026
    assert receipt["back_offset_m"] == 0.098
    assert receipt["mesh_scale"] == 1.3
    assert receipt["back_tilt_degrees"] == 13.0
    assert captured == {"head_clearance_m": -0.026, "back_offset_m": 0.098}
    assert receipt["outputs"]["mesh"] == "baseball_cap_v1.obj"
    assert Path(paths["mesh"]).is_file()
    assert Path(paths["anchors"]).is_file()


def test_head_top_anchors_rejects_invalid_offsets(tmp_path: Path) -> None:
    module = _module()
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"bundles": {"smplx_office_neutral_v1": {"clips": {}}}}))

    with pytest.raises(ValueError, match="must be finite"):
        module.head_top_anchors(manifest, head_clearance_m=float("nan"))
    with pytest.raises(ValueError, match="must not be negative"):
        module.head_top_anchors(manifest, back_offset_m=-0.001)
    with pytest.raises(ValueError, match="positive finite"):
        module.normalized_obj(b"v 0 0 0\nf 1 1 1\n", mesh_scale=0)
    with pytest.raises(ValueError, match="must be finite"):
        module.normalized_obj(b"v 0 0 0\nf 1 1 1\n", back_tilt_degrees=float("nan"))
