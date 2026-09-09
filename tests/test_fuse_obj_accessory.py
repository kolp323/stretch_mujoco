import importlib.util
import json
from pathlib import Path

import numpy as np


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


def test_head_follow_applies_reference_head_rotation_and_translation() -> None:
    module = _module()
    reference = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    target = np.array([[2.0, 3.0, 0.0], [2.0, 4.0, 0.0], [1.0, 3.0, 0.0]])
    accessory = np.array([[0.0, 0.0, 1.0]])

    followed = module.head_follow_vertices(reference, target, accessory)

    assert np.allclose(followed, [[2.0, 3.0, 1.0]])


def test_fuse_manifest_repoints_every_clip_frame(tmp_path: Path) -> None:
    module = _module()
    body = tmp_path / "body.obj"
    body.write_text(
        "\n".join(
            ["v 0 0 2" for _ in range(16)]
            + ["v 1 0 0", "v 0 1 0", "v 0 0 0", "vt 0 0", "f 17/1 18/1 19/1"]
        )
        + "\n"
    )
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
                        "accessories": {"cap": {"mesh": "cap.obj", "anchors": "anchors.json"}},
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
    assert generated["bundles"]["bundle"]["sha256"]["cap.obj"] == module._sha256(cap)
    assert generated["bundles"]["bundle"]["sha256"]["anchors.json"] == module._sha256(anchors)


def test_fuse_manifest_clones_a_target_specific_bundle(tmp_path: Path) -> None:
    module = _module()
    body = tmp_path / "body.obj"
    body.write_text(
        "\n".join(
            ["v 0 0 2" for _ in range(16)]
            + ["v 1 0 0", "v 0 1 0", "v 0 0 0", "vt 0 0", "f 17/1 18/1 19/1"]
        )
        + "\n"
    )
    accessory = tmp_path / "accessory.obj"
    accessory.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n")
    anchors = tmp_path / "anchors.json"
    anchors.write_text(json.dumps({"idle": [{"position": [1, 2, 3]}]}))
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "bundles": {
                    "base": {
                        "clips": {"idle": {"frames": ["body.obj"]}},
                        "accessories": {},
                        "sha256": {},
                    }
                }
            }
        )
    )
    output_manifest = tmp_path / "fused_manifest.json"

    result = module.fuse_manifest(
        manifest,
        accessory,
        anchors,
        tmp_path / "fused",
        output_manifest,
        bundle_id="base",
        fused_bundle_id="base__fused__accessory",
        accessory_uv=(0.25, 0.75),
    )

    generated = json.loads(output_manifest.read_text())
    assert result["bundle"] == "base__fused__accessory"
    assert generated["bundles"]["base"]["clips"]["idle"]["frames"] == ["body.obj"]
    assert generated["bundles"]["base__fused__accessory"]["clips"]["idle"]["frames"] != ["body.obj"]
    assert (
        "vt 0.25 0.75"
        in (
            tmp_path / generated["bundles"]["base__fused__accessory"]["clips"]["idle"]["frames"][0]
        ).read_text()
    )
