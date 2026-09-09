import json
from pathlib import Path

import cv2
import numpy as np

from stretch_mujoco.npc.appearance_pipeline.flat_layers import generate_flat_layers


def test_flat_layers_create_transparent_uv_overlays(tmp_path: Path) -> None:
    base = np.array([[[30, 20, 10], [60, 50, 40]]], dtype=np.uint8)
    assert cv2.imwrite(str(tmp_path / "base.png"), base)
    spec_path = tmp_path / "layers.json"
    spec_path.write_text(
        json.dumps(
            {
                "base": "base.png",
                "regions": {"skin": {"source_rgb": [10, 20, 30], "target_rgb": [100, 110, 120]}},
            }
        )
    )

    paths = generate_flat_layers(spec_path, tmp_path / "layers")

    layer = cv2.imread(str(paths["skin"]), cv2.IMREAD_UNCHANGED)
    assert layer is not None
    assert tuple(layer[0, 0]) == (120, 110, 100, 255)
    assert tuple(layer[0, 1]) == (120, 110, 100, 0)


def test_flat_layers_can_exclude_a_neck_mask(tmp_path: Path) -> None:
    base = np.full((3, 3, 3), (30, 20, 10), dtype=np.uint8)
    exclude = np.zeros((3, 3), dtype=np.uint8)
    exclude[1, 1] = 255
    assert cv2.imwrite(str(tmp_path / "base.png"), base)
    assert cv2.imwrite(str(tmp_path / "neck.png"), exclude)
    spec_path = tmp_path / "layers.json"
    spec_path.write_text(
        json.dumps(
            {
                "base": "base.png",
                "regions": {
                    "top": {
                        "source_rgb": [10, 20, 30],
                        "target_rgb": [100, 110, 120],
                        "exclude_mask": "neck.png",
                    }
                },
            }
        )
    )

    layer = cv2.imread(
        str(generate_flat_layers(spec_path, tmp_path / "layers")["top"]), cv2.IMREAD_UNCHANGED
    )

    assert layer is not None
    assert layer[..., 3][1, 1] == 0
    assert np.count_nonzero(layer[..., 3]) == 8


def test_flat_layers_can_include_a_neck_mask_in_skin(tmp_path: Path) -> None:
    base = np.zeros((3, 3, 3), dtype=np.uint8)
    include = np.zeros((3, 3), dtype=np.uint8)
    include[1, 1] = 255
    assert cv2.imwrite(str(tmp_path / "base.png"), base)
    assert cv2.imwrite(str(tmp_path / "neck.png"), include)
    spec_path = tmp_path / "layers.json"
    spec_path.write_text(
        json.dumps(
            {
                "base": "base.png",
                "regions": {
                    "skin": {
                        "source_rgb": [10, 20, 30],
                        "target_rgb": [100, 110, 120],
                        "include_mask": "neck.png",
                    }
                },
            }
        )
    )

    layer = cv2.imread(
        str(generate_flat_layers(spec_path, tmp_path / "layers")["skin"]), cv2.IMREAD_UNCHANGED
    )

    assert layer is not None
    assert tuple(layer[1, 1]) == (120, 110, 100, 255)
    assert np.count_nonzero(layer[..., 3]) == 1
