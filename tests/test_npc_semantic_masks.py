import json
from pathlib import Path

import cv2
import numpy as np

from stretch_mujoco.npc.appearance_pipeline.semantic_masks import generate_semantic_masks


def test_semantic_masks_write_flat_and_mesh_based_masks(tmp_path: Path) -> None:
    base = np.array([[[30, 20, 10], [3, 2, 1]]], dtype=np.uint8)
    assert cv2.imwrite(str(tmp_path / "base.png"), base)
    (tmp_path / "mesh.obj").write_text(
        "v 0 0.1 1.6\nv 0.1 0.1 1.6\nv 0 0.2 1.6\n" "vt 0 0\nvt 1 0\nvt 0 1\nf 1/1 2/2 3/3\n"
    )
    spec_path = tmp_path / "masks.json"
    spec_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "texture_topology_id": "topology-v1",
                "base": "base.png",
                "reference_mesh": "mesh.obj",
                "flat_regions": {"skin": {"source_rgb": [10, 20, 30]}},
                "head": {"min_height_m": 1.5, "hair_min_height_m": 1.55},
            }
        )
    )

    generated = generate_semantic_masks(spec_path, tmp_path / "output")

    assert set(generated.masks) == {"skin", "hair_cap", "face"}
    assert json.loads(generated.manifest_path.read_text())["masks"]["skin"]["sha256"]
    skin = cv2.imread(str(generated.masks["skin"]), cv2.IMREAD_GRAYSCALE)
    assert skin is not None and tuple(skin[0]) == (255, 0)


def test_semantic_masks_can_select_front_face_and_upward_hair_surfaces(tmp_path: Path) -> None:
    base = np.zeros((2, 2, 3), dtype=np.uint8)
    assert cv2.imwrite(str(tmp_path / "base.png"), base)
    (tmp_path / "mesh.obj").write_text(
        # Face triangle has an outward -Y normal at the left side of the atlas.
        "v 0 -0.08 1.6\nv 0.1 -0.08 1.6\nv 0 -0.08 1.7\n"
        # Top triangle has an outward +Z normal at the right side of the atlas.
        "v 0 0 1.7\nv 0.1 0 1.7\nv 0 0.1 1.7\n"
        "vt 0 0\nvt 0.4 0\nvt 0 0.4\nvt 0.6 0.6\nvt 1 0.6\nvt 0.6 1\n"
        "f 1/1 2/2 3/3\nf 4/4 5/5 6/6\n"
    )
    spec_path = tmp_path / "masks.json"
    spec_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "texture_topology_id": "topology-v1",
                "base": "base.png",
                "reference_mesh": "mesh.obj",
                "flat_regions": {"skin": {"source_rgb": [0, 0, 0]}},
                "head": {
                    "min_height_m": 1.5,
                    "hair_min_height_m": 1.55,
                    "horizontal_radius_m": 1.0,
                    "face_max_y_m": -0.02,
                    "face_normal_y_max": -0.5,
                    "hair_normal_z_min": 0.5,
                },
            }
        )
    )

    generated = generate_semantic_masks(spec_path, tmp_path / "output")
    face = cv2.imread(str(generated.masks["face"]), cv2.IMREAD_GRAYSCALE)
    hair = cv2.imread(str(generated.masks["hair_cap"]), cv2.IMREAD_GRAYSCALE)

    assert face is not None and face[900, 100] == 255 and face[300, 700] == 0
    assert hair is not None and hair[900, 100] == 0 and hair[300, 700] == 255


def test_semantic_masks_can_write_a_bounded_neck_exclusion(tmp_path: Path) -> None:
    assert cv2.imwrite(str(tmp_path / "base.png"), np.zeros((2, 2, 3), dtype=np.uint8))
    (tmp_path / "mesh.obj").write_text(
        "v 0 0 1.3\nv 0.1 0 1.3\nv 0 0.1 1.3\n"
        "v 0 0 1.6\nv 0.1 0 1.6\nv 0 0.1 1.6\n"
        "vt 0 0\nvt 0.4 0\nvt 0 0.4\nvt 0.6 0.6\nvt 1 0.6\nvt 0.6 1\n"
        "f 1/1 2/2 3/3\nf 4/4 5/5 6/6\n"
    )
    spec_path = tmp_path / "masks.json"
    spec_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "texture_topology_id": "topology-v1",
                "base": "base.png",
                "reference_mesh": "mesh.obj",
                "flat_regions": {"skin": {"source_rgb": [0, 0, 0]}},
                "head": {"min_height_m": 1.5, "hair_min_height_m": 1.55},
                "neck": {"min_height_m": 1.2, "max_height_m": 1.4, "horizontal_radius_m": 1.0},
            }
        )
    )

    neck = cv2.imread(
        str(generate_semantic_masks(spec_path, tmp_path / "output").masks["neck"]),
        cv2.IMREAD_GRAYSCALE,
    )

    assert neck is not None and neck[900, 100] == 255 and neck[300, 700] == 0


def test_semantic_masks_write_sideburns_only_on_outer_head_surfaces(tmp_path: Path) -> None:
    assert cv2.imwrite(str(tmp_path / "base.png"), np.zeros((2, 2, 3), dtype=np.uint8))
    (tmp_path / "mesh.obj").write_text(
        # The first triangle is adjacent to an ear; the second is in the side-face centre.
        "v 0.08 0 1.55\nv 0.09 0 1.55\nv 0.08 0.02 1.55\n"
        "v 0.01 0 1.55\nv 0.02 0 1.55\nv 0.01 0.02 1.55\n"
        "vt 0 0\nvt 0.4 0\nvt 0 0.4\nvt 0.6 0.6\nvt 1 0.6\nvt 0.6 1\n"
        "f 1/1 2/2 3/3\nf 4/4 5/5 6/6\n"
    )
    spec_path = tmp_path / "masks.json"
    spec_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "texture_topology_id": "topology-v1",
                "base": "base.png",
                "reference_mesh": "mesh.obj",
                "flat_regions": {"skin": {"source_rgb": [0, 0, 0]}},
                "head": {"min_height_m": 1.45, "hair_min_height_m": 1.55},
                "sideburns": {
                    "min_height_m": 1.47,
                    "max_height_m": 1.60,
                    "horizontal_radius_m": 1.0,
                    "min_abs_x_m": 0.07,
                    "max_abs_x_m": 0.10,
                    "max_y_m": 0.01,
                },
            }
        )
    )

    sideburns = cv2.imread(
        str(generate_semantic_masks(spec_path, tmp_path / "output").masks["sideburns"]),
        cv2.IMREAD_GRAYSCALE,
    )

    assert sideburns is not None and sideburns[900, 100] == 255 and sideburns[300, 700] == 0


def test_semantic_neck_can_tighten_only_the_front_surface(tmp_path: Path) -> None:
    assert cv2.imwrite(str(tmp_path / "base.png"), np.zeros((2, 2, 3), dtype=np.uint8))
    (tmp_path / "mesh.obj").write_text(
        # Front and rear neck triangles share height/radius, but occupy separate UV regions.
        "v 0.08 -0.08 1.48\nv 0.10 -0.08 1.48\nv 0.08 -0.06 1.48\n"
        "v 0.08 0.08 1.48\nv 0.10 0.08 1.48\nv 0.08 0.10 1.48\n"
        "vt 0 0\nvt 0.4 0\nvt 0 0.4\nvt 0.6 0.6\nvt 1 0.6\nvt 0.6 1\n"
        "f 1/1 2/2 3/3\nf 4/4 5/5 6/6\n"
    )
    spec_path = tmp_path / "masks.json"
    spec_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "texture_topology_id": "topology-v1",
                "base": "base.png",
                "reference_mesh": "mesh.obj",
                "flat_regions": {"skin": {"source_rgb": [0, 0, 0]}},
                "head": {"min_height_m": 1.45, "hair_min_height_m": 1.55},
                "neck": {
                    "min_height_m": 1.42,
                    "max_height_m": 1.55,
                    "horizontal_radius_m": 0.14,
                    "front_max_y_m": -0.02,
                    "front_horizontal_radius_m": 0.07,
                },
            }
        )
    )

    neck = cv2.imread(
        str(generate_semantic_masks(spec_path, tmp_path / "output").masks["neck"]),
        cv2.IMREAD_GRAYSCALE,
    )

    assert neck is not None and neck[900, 100] == 0 and neck[300, 700] == 255


def test_semantic_neck_can_tighten_front_facing_trapezius_surfaces(tmp_path: Path) -> None:
    assert cv2.imwrite(str(tmp_path / "base.png"), np.zeros((2, 2, 3), dtype=np.uint8))
    (tmp_path / "mesh.obj").write_text(
        # Both triangles are behind the positional front threshold.  Their normals
        # differ, so only the forward-facing trapezius triangle is tightened.
        "v 0.08 0.08 1.48\nv 0.10 0.08 1.48\nv 0.08 0.08 1.50\n"
        "v 0.08 0.08 1.48\nv 0.10 0.08 1.48\nv 0.08 0.08 1.50\n"
        "vt 0 0\nvt 0.4 0\nvt 0 0.4\nvt 0.6 0.6\nvt 1 0.6\nvt 0.6 1\n"
        "f 1/1 2/2 3/3\nf 4/4 6/6 5/5\n"
    )
    spec_path = tmp_path / "masks.json"
    spec_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "texture_topology_id": "topology-v1",
                "base": "base.png",
                "reference_mesh": "mesh.obj",
                "flat_regions": {"skin": {"source_rgb": [0, 0, 0]}},
                "head": {"min_height_m": 1.45, "hair_min_height_m": 1.55},
                "neck": {
                    "min_height_m": 1.42,
                    "max_height_m": 1.55,
                    "horizontal_radius_m": 0.14,
                    "front_max_y_m": -0.02,
                    "front_horizontal_radius_m": 0.07,
                    "front_normal_y_max": -0.05,
                },
            }
        )
    )

    neck = cv2.imread(
        str(generate_semantic_masks(spec_path, tmp_path / "output").masks["neck"]),
        cv2.IMREAD_GRAYSCALE,
    )

    assert neck is not None and neck[900, 100] == 0 and neck[300, 700] == 255
