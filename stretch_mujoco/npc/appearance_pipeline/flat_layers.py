"""Generate transparent UV layers from the flat-color SMPL-X reference atlas."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np


def generate_flat_layers(spec_path: str | Path, output_dir: str | Path) -> dict[str, Path]:
    """Create one RGBA layer per recolored region in a flat-color base atlas.

    This is intentionally limited to the locally generated SMPL-X reference
    atlas. It converts its known flat regions into editable, UV-aligned PNG
    layers; it is not a general semantic texture segmentation tool.
    """
    source = Path(spec_path).resolve()
    spec = _mapping(json.loads(source.read_text(encoding="utf-8")), "Layer spec")
    base_name = _required_string(spec, "base", "Layer spec")
    regions = _mapping(spec.get("regions"), "Layer spec regions")
    base = cv2.imread(str(source.parent / base_name), cv2.IMREAD_COLOR)
    if base is None:
        raise ValueError(f"Could not decode base atlas '{source.parent / base_name}'")
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    for name, raw_region in regions.items():
        region = _mapping(raw_region, f"Region '{name}'")
        source_rgb = _rgb(region.get("source_rgb"), f"Region '{name}' source_rgb")
        target_rgb = _rgb(region.get("target_rgb"), f"Region '{name}' target_rgb")
        tolerance = int(region.get("tolerance", 8))
        if not 0 <= tolerance <= 255:
            raise ValueError(f"Region '{name}' tolerance must be in [0, 255]")
        source_bgr = np.array(source_rgb[::-1], dtype=np.int16)
        distance = np.max(np.abs(base.astype(np.int16) - source_bgr), axis=2)
        mask = (distance <= tolerance).astype(np.uint8) * 255
        layer = np.zeros((*base.shape[:2], 4), dtype=np.uint8)
        layer[..., :3] = target_rgb[::-1]
        layer[..., 3] = mask
        path = destination / f"{name}.png"
        if not cv2.imwrite(str(path), layer, [cv2.IMWRITE_PNG_COMPRESSION, 9]):
            raise OSError(f"Could not write layer '{path}'")
        written[str(name)] = path
    return written


def _mapping(value: object, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be an object")
    return value


def _required_string(payload: Mapping[str, Any], key: str, context: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{context} must define non-empty string '{key}'")
    return value


def _rgb(value: object, context: str) -> tuple[int, int, int]:
    if (
        not isinstance(value, list)
        or len(value) != 3
        or not all(isinstance(item, int) for item in value)
    ):
        raise ValueError(f"{context} must be a three-item integer RGB array")
    if not all(0 <= item <= 255 for item in value):
        raise ValueError(f"{context} components must be in [0, 255]")
    return value[0], value[1], value[2]


def main() -> None:
    """Console entry point for creating editable SMPL-X UV overlay layers."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    for path in generate_flat_layers(args.spec, args.output_dir).values():
        print(path)
