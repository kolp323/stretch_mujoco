import hashlib
import json
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from stretch_mujoco.npc.appearance_pipeline.textile_layers import generate_textile_layers


def _png(path: Path, image: np.ndarray) -> None:
    assert cv2.imwrite(str(path), image)


def _spec(source_sha256: str) -> dict[str, object]:
    return {
        "schema_version": 1,
        "base": "base.png",
        "receipt": "generated/receipt.json",
        "layers": [
            {
                "id": "gingham_top_v1",
                "source_archive": "source.zip",
                "source_sha256": source_sha256,
                "source_url": "https://example.test/gingham",
                "license": "CC0-1.0",
                "archive_member": "textures/diffuse.png",
                "mask": "top.png",
                "output": "generated/gingham_top_v1.png",
                "tile_width_px": 2,
                "seed": 19,
            }
        ],
    }


def test_textile_layers_pin_archive_and_apply_only_semantic_mask(tmp_path: Path) -> None:
    _png(tmp_path / "base.png", np.zeros((4, 4, 3), dtype=np.uint8))
    mask = np.zeros((4, 4), dtype=np.uint8)
    mask[1:3, 1:3] = 255
    _png(tmp_path / "top.png", mask)
    source = np.array([[[10, 20, 30], [40, 50, 60]]], dtype=np.uint8)
    encoded = cv2.imencode(".png", source)[1].tobytes()
    archive = tmp_path / "source.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("textures/diffuse.png", encoded)
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    spec_path = tmp_path / "textiles.json"
    spec_path.write_text(json.dumps(_spec(digest)))

    generated = generate_textile_layers(spec_path, tmp_path)

    layer = cv2.imread(str(generated["gingham_top_v1"]), cv2.IMREAD_UNCHANGED)
    assert layer is not None
    assert layer.shape == (4, 4, 4)
    assert layer[..., 3].tolist() == mask.tolist()
    assert np.any(layer[1:3, 1:3, :3])
    assert (
        json.loads((tmp_path / "generated/receipt.json").read_text())["layers"]["gingham_top_v1"][
            "source_archive_sha256"
        ]
        == digest
    )


def test_textile_layers_reject_changed_source_archive(tmp_path: Path) -> None:
    _png(tmp_path / "base.png", np.zeros((2, 2, 3), dtype=np.uint8))
    _png(tmp_path / "top.png", np.full((2, 2), 255, dtype=np.uint8))
    archive = tmp_path / "source.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("textures/diffuse.png", b"not the expected source")
    spec_path = tmp_path / "textiles.json"
    spec_path.write_text(json.dumps(_spec("0" * 64)))

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        generate_textile_layers(spec_path, tmp_path)
