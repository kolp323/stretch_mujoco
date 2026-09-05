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
