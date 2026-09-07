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


def test_fused_obj_culls_only_head_faces_covered_by_the_cap(tmp_path: Path) -> None:
    module = _module()
    body = tmp_path / "body.obj"
    body.write_text(
        "\n".join(
            [
                "v 0 0 2",
                "v 0.1 0 2",
                "v 0 0.1 2",
                "v 1 0 0",
                "v 1.1 0 0",
                "v 1 0.1 0",
                "vt 0 0",
                "f 1/1 2/1 3/1",
                "f 4/1 5/1 6/1",
            ]
        )
        + "\n"
    )
    cap = tmp_path / "cap.obj"
    cap.write_text("v -0.1 -0.1 2\nv 0.2 -0.1 2\nv -0.1 0.2 2\nf 1 2 3\n")

    fused = module._fused_obj_with_vertices(
        body,
        cap,
        module._vertices_and_faces(cap)[0],
        body_occlusion_mode="cull_covered_head_faces",
    ).decode("utf-8")

    assert "f 1/1 2/1 3/1" not in fused
    assert "f 4/1 5/1 6/1" in fused
    assert "f 7/1 8/1 9/1" in fused
    assert "f 10/2 11/2 12/2" in fused


def test_culling_rejects_an_open_accessory_shell(tmp_path: Path) -> None:
    module = _module()
    body = tmp_path / "body.obj"
    body.write_text("\n".join(["v 0 0 2" for _ in range(16)] + ["vt 0 0", "f 1/1 2/1 3/1"]) + "\n")
    cap = tmp_path / "cap.obj"
    cap.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n")
    anchors = tmp_path / "anchors.json"
    anchors.write_text(json.dumps({"idle": [{"position": [0, 0, 0]}]}))
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "bundles": {
                    "bundle": {
                        "clips": {"idle": {"frames": ["body.obj"]}},
                        "accessories": {},
                        "sha256": {},
                    }
                }
            }
        )
    )

    with np.testing.assert_raises_regex(ValueError, "watertight accessory occlusion mesh"):
        module.fuse_manifest(
            manifest,
            cap,
            anchors,
            tmp_path / "fused",
            tmp_path / "preview_manifest.json",
            bundle_id="bundle",
            body_occlusion_mode="cull_covered_head_faces",
        )


def test_maskable_faces_keep_one_topology_when_frame_coverage_changes(tmp_path: Path) -> None:
    module = _module()
    body = tmp_path / "body.obj"
    body.write_text(
        "\n".join(["v 0 0 2", "v 0.1 0 2", "v 0 0.1 2", "vt 0 0", "f 1/1 2/1 3/1"]) + "\n"
    )
    cap = tmp_path / "cap.obj"
    cap.write_text("v -0.1 -0.1 2\nv 0.2 -0.1 2\nv -0.1 0.2 2\nf 1 2 3\n")
    vertices, _ = module._vertices_and_faces(cap)

    hidden = module._fused_obj_with_vertices(
        body,
        cap,
        vertices,
        body_occlusion_mode="cull_covered_head_faces",
        covered_body_face_indices={0},
        maskable_body_face_indices={0},
    ).decode("utf-8")
    visible = module._fused_obj_with_vertices(
        body,
        cap,
        vertices,
        body_occlusion_mode="cull_covered_head_faces",
        covered_body_face_indices=set(),
        maskable_body_face_indices={0},
    ).decode("utf-8")

    assert [line for line in hidden.splitlines() if line.startswith("f ")] == [
        line for line in visible.splitlines() if line.startswith("f ")
    ]
    assert sum(line.startswith("v ") for line in hidden.splitlines()) == sum(
        line.startswith("v ") for line in visible.splitlines()
    )


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
