import json
from pathlib import Path

import cv2
import numpy as np

from stretch_mujoco.npc.appearance_pipeline.face_details import generate_face_detail_layers


def test_face_detail_layers_stay_within_formal_face_mask(tmp_path: Path) -> None:
    mask = np.zeros((64, 64), dtype=np.uint8)
    cv2.rectangle(mask, (12, 8), (52, 56), 255, -1)
    assert cv2.imwrite(str(tmp_path / "face.png"), mask)
    spec_path = tmp_path / "details.json"
    spec_path.write_text(
        json.dumps(
            {
                "face_mask": "face.png",
                "styles": {
                    "freckles": {"kind": "freckles", "rgb": [80, 40, 20], "seed": 4},
                    "brows": {"kind": "brows", "rgb": [30, 20, 10]},
                    "beard": {"kind": "beard", "rgb": [20, 15, 10]},
                    "glasses": {"kind": "glasses_2d", "rgb": [10, 10, 10]},
                },
            }
        )
    )

    paths = generate_face_detail_layers(spec_path, tmp_path / "layers")

    assert set(paths) == {"freckles", "brows", "beard", "glasses"}
    for path in paths.values():
        layer = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        assert layer is not None
        assert np.all(layer[..., 3][mask == 0] == 0)


def test_face_detail_layers_cover_each_disconnected_uv_island(tmp_path: Path) -> None:
    mask = np.zeros((80, 120), dtype=np.uint8)
    cv2.rectangle(mask, (8, 12), (48, 68), 255, -1)
    cv2.rectangle(mask, (70, 12), (110, 68), 255, -1)
    assert cv2.imwrite(str(tmp_path / "face.png"), mask)
    spec_path = tmp_path / "details.json"
    spec_path.write_text(
        json.dumps(
            {
                "face_mask": "face.png",
                "styles": {"glasses": {"kind": "glasses_2d", "rgb": [10, 10, 10]}},
            }
        )
    )

    path = generate_face_detail_layers(spec_path, tmp_path / "layers")["glasses"]
    layer = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)

    assert layer is not None
    assert np.any(layer[..., 3][:, :60])
    assert np.any(layer[..., 3][:, 60:])


def test_face_detail_layers_ignore_small_neck_and_seam_islands(tmp_path: Path) -> None:
    mask = np.zeros((100, 100), dtype=np.uint8)
    cv2.rectangle(mask, (15, 10), (85, 70), 255, -1)
    cv2.rectangle(mask, (45, 80), (55, 86), 255, -1)
    assert cv2.imwrite(str(tmp_path / "face.png"), mask)
    spec_path = tmp_path / "details.json"
    spec_path.write_text(
        json.dumps(
            {
                "face_mask": "face.png",
                "styles": {"beard": {"kind": "beard", "rgb": [10, 10, 10]}},
            }
        )
    )

    layer = cv2.imread(
        str(generate_face_detail_layers(spec_path, tmp_path / "layers")["beard"]),
        cv2.IMREAD_UNCHANGED,
    )

    assert layer is not None
    assert np.any(layer[..., 3][10:71, 15:86])
    assert not np.any(layer[..., 3][80:87, 45:56])
