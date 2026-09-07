import importlib.util
import io
import json
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


def test_normalized_source_cap_is_meter_z_up_and_records_nested_provenance(tmp_path: Path) -> None:
    module = _module()
    archive = tmp_path / "cap.zip"
    _nested_cap(archive)

    source, provenance = module._source_obj(archive)
    normalized = module.normalized_obj(source, mesh_scale=1.0).decode("utf-8")
    vertices = np.array(
        [[float(value) for value in line.split()[1:4]] for line in normalized.splitlines() if line.startswith("v ")]
    )

    assert provenance["nested_archive_member"] == "source/cap.zip"
    assert provenance["source_obj_member"] == "cap.obj"
    assert np.isclose(vertices[:, 2].min(), 0.0)
    assert np.isclose(vertices[:, 2].max(), 0.1)
    assert np.isclose(vertices[:, :2].ptp(axis=0)[0], 0.2)
    assert "mtllib" not in normalized
    scaled = module.normalized_obj(source, mesh_scale=1.3).decode("utf-8")
    scaled_vertices = np.array(
        [[float(value) for value in line.split()[1:4]] for line in scaled.splitlines() if line.startswith("v ")]
    )
    assert np.isclose(scaled_vertices[:, :2].ptp(axis=0)[0], 0.26)


def test_prepare_writes_preview_receipt_and_all_clip_anchors(tmp_path: Path, monkeypatch) -> None:
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
        "cap_source_v1",
        head_clearance_m=-0.003,
        back_offset_m=0.08,
        mesh_scale=1.3,
    )
    receipt = json.loads(Path(paths["receipt"]).read_text())

    assert receipt["asset_quality"] == "preview"
    assert receipt["head_clearance_m"] == -0.003
    assert receipt["back_offset_m"] == 0.08
    assert receipt["mesh_scale"] == 1.3
    assert captured == {"head_clearance_m": -0.003, "back_offset_m": 0.08}
    assert receipt["outputs"]["mesh"] == "cap_source_v1.obj"
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
