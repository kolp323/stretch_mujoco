import importlib.util
import json
from pathlib import Path


def _module():
    path = Path(__file__).parents[1] / "tools" / "fuse_obj_accessory.py"
    spec = importlib.util.spec_from_file_location("fuse_obj_accessory", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_fused_obj_keeps_body_and_positions_accessory_in_one_mesh(tmp_path: Path) -> None:
    module = _module()
    body = tmp_path / "body.obj"
    body.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nvt 0.5 0.5\nf 1/1 2/1 3/1\n")
    cap = tmp_path / "cap.obj"
    cap.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n")

    fused = module.fused_obj(body, cap, [2.0, 3.0, 4.0]).decode("utf-8")

    assert "v 2 3 4" in fused
    assert "f 4/2 5/2 6/2" in fused
    assert "f 1/1 2/1 3/1" in fused


def test_fuse_manifest_repoints_every_clip_frame(tmp_path: Path) -> None:
    module = _module()
    body = tmp_path / "body.obj"
    body.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nvt 0 0\nf 1/1 2/1 3/1\n")
    cap = tmp_path / "cap.obj"
    cap.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n")
    anchors = tmp_path / "anchors.json"
    anchors.write_text(json.dumps({"idle": [{"position": [1, 2, 3]}]}))
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "bundles": {
                    "bundle": {
                        "clips": {"idle": {"frames": ["body.obj"]}},
                        "sha256": {},
                    }
                }
            }
        )
    )
    output_manifest = tmp_path / "preview_manifest.json"

    result = module.fuse_manifest(
        manifest, cap, anchors, tmp_path / "fused", output_manifest, bundle_id="bundle"
    )
    generated = json.loads(output_manifest.read_text())
    relative = generated["bundles"]["bundle"]["clips"]["idle"]["frames"][0]

    assert result["clips"]["idle"] == [relative]
    assert (tmp_path / relative).is_file()
    assert relative in generated["bundles"]["bundle"]["sha256"]
