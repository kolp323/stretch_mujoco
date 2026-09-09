import json
from pathlib import Path

import cv2

from stretch_mujoco.npc.appearance_pipeline.hair_layers import generate_short_hair_layers


def test_short_hair_layers_use_high_uv_mapped_faces(tmp_path: Path) -> None:
    (tmp_path / "mesh.obj").write_text(
        "v 0 0 1.6\nv 0.1 0 1.6\nv 0 0.1 1.6\n" "v 0 0 1.0\nvt 0 0\nvt 1 0\nvt 0 1\nf 1/1 2/2 3/3\n"
    )
    spec = {
        "reference_mesh": "mesh.obj",
        "min_height_m": 1.5,
        "horizontal_radius_m": 0.2,
        "styles": {"black": {"rgb": [10, 20, 30]}},
    }
    spec_path = tmp_path / "hair.json"
    spec_path.write_text(json.dumps(spec))

    paths = generate_short_hair_layers(spec_path, tmp_path / "layers")

    image = cv2.imread(str(paths["black"]), cv2.IMREAD_UNCHANGED)
    assert image is not None
    assert tuple(image[0, 0]) == (30, 20, 10, 255)
    assert tuple(image[100, 900]) == (30, 20, 10, 0)


def test_short_hair_can_exclude_non_upward_face_triangles(tmp_path: Path) -> None:
    (tmp_path / "mesh.obj").write_text(
        "v 0 0 1.7\nv 0.1 0 1.7\nv 0 0.1 1.7\n"
        "v 0 -0.1 1.6\nv 0.1 -0.1 1.6\nv 0 -0.1 1.7\n"
        "vt 0 0\nvt 0.4 0\nvt 0 0.4\nvt 0.6 0.6\nvt 1 0.6\nvt 0.6 1\n"
        "f 1/1 2/2 3/3\nf 4/4 5/5 6/6\n"
    )
    spec_path = tmp_path / "hair.json"
    spec_path.write_text(
        json.dumps(
            {
                "reference_mesh": "mesh.obj",
                "min_height_m": 1.55,
                "horizontal_radius_m": 1.0,
                "normal_z_min": 0.5,
                "styles": {"hair": {"rgb": [1, 2, 3]}},
            }
        )
    )

    image = cv2.imread(
        str(generate_short_hair_layers(spec_path, tmp_path / "layers")["hair"]),
        cv2.IMREAD_UNCHANGED,
    )

    assert image is not None
    assert image[800, 100, 3] == 255
    assert image[300, 700, 3] == 0
