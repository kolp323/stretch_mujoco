import importlib.util
import io
import json
import zipfile
from pathlib import Path

import numpy as np


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
    normalized = module.normalized_obj(source).decode("utf-8")
    vertices = np.array(
        [[float(value) for value in line.split()[1:4]] for line in normalized.splitlines() if line.startswith("v ")]
    )

    assert provenance["nested_archive_member"] == "source/cap.zip"
    assert provenance["source_obj_member"] == "cap.obj"
    assert np.isclose(vertices[:, 2].min(), 0.0)
    assert np.isclose(vertices[:, 2].max(), 0.1)
    assert np.isclose(vertices[:, :2].ptp(axis=0)[0], 0.2)
    assert "mtllib" not in normalized


def test_prepare_writes_preview_receipt_and_all_clip_anchors(tmp_path: Path, monkeypatch) -> None:
    module = _module()
    archive = tmp_path / "cap.zip"
    _nested_cap(archive)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"bundles": {"smplx_office_neutral_v1": {"clips": {}}}}))
    monkeypatch.setattr(module, "head_top_anchors", lambda _manifest: {"idle": [{"position": [0, 0, 1]}]})

    paths = module.prepare(archive, manifest, tmp_path / "output", "cap_source_v1")
    receipt = json.loads(Path(paths["receipt"]).read_text())

    assert receipt["asset_quality"] == "preview"
    assert receipt["outputs"]["mesh"] == "cap_source_v1.obj"
    assert Path(paths["mesh"]).is_file()
    assert Path(paths["anchors"]).is_file()
