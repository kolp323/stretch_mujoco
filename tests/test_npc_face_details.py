import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from stretch_mujoco.npc.appearance_pipeline.face_details import generate_face_detail_layers


def test_face_detail_layers_stay_within_formal_face_mask(tmp_path: Path) -> None:
    mask = np.zeros((64, 64), dtype=np.uint8)
    cv2.rectangle(mask, (12, 8), (52, 56), 255, -1)
    assert cv2.imwrite(str(tmp_path / "face.png"), mask)
    sideburn_mask = np.zeros((64, 64), dtype=np.uint8)
    sideburn_mask[18:46, 12:18] = 255
    sideburn_mask[18:46, 47:53] = 255
    assert cv2.imwrite(str(tmp_path / "sideburns.png"), sideburn_mask)
    spec_path = tmp_path / "details.json"
    spec_path.write_text(
        json.dumps(
            {
                "face_mask": "face.png",
                "styles": {
                    "freckles": {"kind": "freckles", "rgb": [80, 40, 20], "seed": 4},
                    "brows": {"kind": "brows", "rgb": [30, 20, 10]},
                    "beard": {"kind": "beard", "rgb": [20, 15, 10]},
                    "handlebar": {"kind": "moustache_handlebar", "rgb": [20, 15, 10]},
                    "full_beard": {
                        "kind": "beard_full",
                        "rgb": [20, 15, 10],
                        "sideburn_mask": "sideburns.png",
                    },
                    "boxed_beard": {"kind": "beard_boxed", "rgb": [20, 15, 10]},
                    "goatee": {"kind": "goatee", "rgb": [20, 15, 10]},
                    "glasses": {"kind": "glasses_2d", "rgb": [10, 10, 10]},
                },
            }
        )
    )

    paths = generate_face_detail_layers(spec_path, tmp_path / "layers")

    assert set(paths) == {
        "freckles",
        "brows",
        "beard",
        "handlebar",
        "full_beard",
        "boxed_beard",
        "goatee",
        "glasses",
    }
    for path in paths.values():
        layer = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        assert layer is not None
        permitted_mask = np.maximum(mask, sideburn_mask) if path.stem == "full_beard" else mask
        assert np.all(layer[..., 3][permitted_mask == 0] == 0)


def test_facial_hair_styles_have_distinct_coverage(tmp_path: Path) -> None:
    mask = np.zeros((120, 100), dtype=np.uint8)
    cv2.rectangle(mask, (10, 10), (90, 110), 255, -1)
    assert cv2.imwrite(str(tmp_path / "face.png"), mask)
    sideburn_mask = np.zeros((120, 100), dtype=np.uint8)
    sideburn_mask[30:80, :8] = 255
    sideburn_mask[30:80, 92:] = 255
    assert cv2.imwrite(str(tmp_path / "sideburns.png"), sideburn_mask)
    spec_path = tmp_path / "details.json"
    spec_path.write_text(
        json.dumps(
            {
                "face_mask": "face.png",
                "styles": {
                    "handlebar": {"kind": "moustache_handlebar", "rgb": [20, 15, 10]},
                    "full": {
                        "kind": "beard_full",
                        "rgb": [20, 15, 10],
                        "sideburn_mask": "sideburns.png",
                    },
                    "sideburns": {
                        "kind": "sideburns",
                        "rgb": [20, 15, 10],
                        "mask": "sideburns.png",
                    },
                    "boxed": {"kind": "beard_boxed", "rgb": [20, 15, 10]},
                    "goatee": {"kind": "goatee", "rgb": [20, 15, 10]},
                    "lowered_goatee": {
                        "kind": "goatee",
                        "rgb": [20, 15, 10],
                        "vertical_offset": 0.09,
                    },
                },
            }
        )
    )
    paths = generate_face_detail_layers(spec_path, tmp_path / "layers")
    alpha = {
        style: cv2.imread(str(path), cv2.IMREAD_UNCHANGED)[..., 3] for style, path in paths.items()
    }

    assert all(np.any(layer) for layer in alpha.values())
    assert np.any(alpha["handlebar"][65:88, 10:91])
    assert np.where(alpha["handlebar"] > 0)[0].mean() >= 77
    assert not np.array_equal(alpha["full"], alpha["boxed"])
    assert np.count_nonzero(alpha["full"]) > np.count_nonzero(alpha["goatee"])
    assert np.count_nonzero(alpha["boxed"]) > np.count_nonzero(alpha["goatee"])
    assert np.where(alpha["lowered_goatee"] > 0)[0].mean() > np.where(alpha["goatee"] > 0)[0].mean()
    assert np.all(alpha["full"][sideburn_mask > 0] == 255)
    assert np.array_equal(alpha["sideburns"], sideburn_mask)


def test_full_beard_requires_a_topology_derived_sideburn_mask(tmp_path: Path) -> None:
    mask = np.full((32, 32), 255, dtype=np.uint8)
    assert cv2.imwrite(str(tmp_path / "face.png"), mask)
    spec_path = tmp_path / "details.json"
    spec_path.write_text(
        json.dumps(
            {
                "face_mask": "face.png",
                "styles": {"full": {"kind": "beard_full", "rgb": [20, 15, 10]}},
            }
        )
    )

    with pytest.raises(ValueError, match="sideburn_mask"):
        generate_face_detail_layers(spec_path, tmp_path / "layers")


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
