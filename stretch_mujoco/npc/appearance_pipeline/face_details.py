"""Generate deterministic, mask-constrained 2D facial identity detail layers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np


def generate_face_detail_layers(spec_path: str | Path, output_dir: str | Path) -> dict[str, Path]:
    """Generate freckles, brows, beard, or explicitly 2D glasses overlays.

    All drawing is clipped by a formal ``face`` mask. The glasses output is a
    texture decal, not a replacement for a geometric glasses asset.
    """
    source = Path(spec_path).resolve()
    spec = _mapping(json.loads(source.read_text(encoding="utf-8")), "Face detail spec")
    face_mask_path = source.parent / _string(spec.get("face_mask"), "Face detail face_mask")
    face_mask = cv2.imread(str(face_mask_path), cv2.IMREAD_GRAYSCALE)
    if face_mask is None:
        raise ValueError(f"Could not decode face mask '{face_mask_path}'")
    styles = _mapping(spec.get("styles"), "Face detail styles")
    if not styles:
        raise ValueError("Face detail spec must define at least one style")
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    for style_id, raw_style in styles.items():
        style = _mapping(raw_style, f"Face detail style '{style_id}'")
        kind = _string(style.get("kind"), f"Face detail style '{style_id}' kind")
        color = _rgb(style.get("rgb"), f"Face detail style '{style_id}' rgb")
        opacity = int(style.get("opacity", 255))
        if not 0 <= opacity <= 255:
            raise ValueError(f"Face detail style '{style_id}' opacity must be in [0, 255]")
        artwork = np.zeros(face_mask.shape, dtype=np.uint8)
        if kind == "freckles":
            _draw_freckles(
                artwork, face_mask, int(style.get("seed", 0)), int(style.get("count", 28))
            )
        elif kind == "brows":
            _draw_brows(artwork, face_mask)
        elif kind == "beard":
            _draw_beard(artwork, face_mask)
        elif kind == "glasses_2d":
            _draw_glasses(artwork, face_mask)
        else:
            raise ValueError(f"Face detail style '{style_id}' has unsupported kind '{kind}'")
        alpha = np.minimum(artwork, face_mask)
        alpha = np.round(alpha.astype(np.float32) * opacity / 255).astype(np.uint8)
        layer = np.zeros((*face_mask.shape, 4), dtype=np.uint8)
        layer[..., :3] = color[::-1]
        layer[..., 3] = alpha
        destination = output / f"{style_id}.png"
        if not cv2.imwrite(str(destination), layer, [cv2.IMWRITE_PNG_COMPRESSION, 9]):
            raise OSError(f"Could not write face detail layer '{destination}'")
        written[str(style_id)] = destination
    return written


def _draw_freckles(image: np.ndarray, mask: np.ndarray, seed: int, count: int) -> None:
    if count < 1:
        raise ValueError("Freckle count must be positive")
    generator = np.random.default_rng(seed)
    regions = _decal_regions(mask)
    for index, region in enumerate(regions):
        points = np.argwhere(region > 0)
        regional_count = count // len(regions) + (index < count % len(regions))
        for row, column in points[generator.integers(len(points), size=regional_count)]:
            cv2.circle(image, (int(column), int(row)), int(generator.integers(1, 3)), 255, -1)


def _draw_brows(image: np.ndarray, mask: np.ndarray) -> None:
    for region in _decal_regions(mask):
        x, y, width, height = _bounds(region)
        thickness = max(1, round(height * 0.035))
        for center_x in (x + round(width * 0.32), x + round(width * 0.68)):
            cv2.ellipse(
                image,
                (center_x, y + round(height * 0.34)),
                (max(2, round(width * 0.11)), max(1, round(height * 0.04))),
                0,
                195,
                345,
                255,
                thickness,
                lineType=cv2.LINE_AA,
            )


def _draw_beard(image: np.ndarray, mask: np.ndarray) -> None:
    for region in _decal_regions(mask):
        x, y, width, height = _bounds(region)
        cv2.ellipse(
            image,
            (x + width // 2, y + round(height * 0.84)),
            (max(2, round(width * 0.13)), max(2, round(height * 0.10))),
            0,
            10,
            170,
            255,
            -1,
            lineType=cv2.LINE_AA,
        )


def _draw_glasses(image: np.ndarray, mask: np.ndarray) -> None:
    for region in _decal_regions(mask):
        x, y, width, height = _bounds(region)
        radius = max(2, round(min(width, height) * 0.085))
        center_y = y + round(height * 0.48)
        left = x + round(width * 0.34)
        right = x + round(width * 0.66)
        thickness = max(1, round(radius * 0.08))
        cv2.circle(image, (left, center_y), radius, 255, thickness, lineType=cv2.LINE_AA)
        cv2.circle(image, (right, center_y), radius, 255, thickness, lineType=cv2.LINE_AA)
        cv2.line(image, (left + radius, center_y), (right - radius, center_y), 255, thickness)
        temple_y = y + round(height * 0.45)
        cv2.line(
            image, (left - radius, center_y), (x + round(width * 0.08), temple_y), 255, thickness
        )
        cv2.line(
            image,
            (right + radius, center_y),
            (x + round(width * 0.92), temple_y),
            255,
            thickness,
        )


def _face_regions(mask: np.ndarray) -> tuple[np.ndarray, ...]:
    """Return the substantial disconnected UV islands of a face mask.

    A single SMPL-X face is commonly split over multiple texture islands.  Each
    island gets a complete decal, rather than treating their combined bounding
    box as one flat portrait and drawing into the transparent seam.
    """
    count, labels, statistics, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8))
    regions = tuple(
        np.where(labels == index, 255, 0).astype(np.uint8)
        for index in range(1, count)
        if statistics[index, cv2.CC_STAT_AREA] >= 16
    )
    if not regions:
        raise ValueError("Face mask has no drawable texture islands")
    return regions


def _decal_regions(mask: np.ndarray) -> tuple[np.ndarray, ...]:
    """Return face islands large enough to carry a complete facial decal.

    The formal mask also contains neck and small seam islands.  Repeating a
    beard or glasses on those islands creates visible duplicates, so retain
    islands no smaller than half the primary face island.  Equal-sized UV
    splits remain supported.
    """
    regions = _face_regions(mask)
    largest_area = max(int(np.count_nonzero(region)) for region in regions)
    return tuple(region for region in regions if np.count_nonzero(region) * 2 >= largest_area)


def _bounds(mask: np.ndarray) -> tuple[int, int, int, int]:
    points = cv2.findNonZero(mask)
    if points is None:
        raise ValueError("Face mask has no drawable pixels")
    return cv2.boundingRect(points)


def _mapping(value: object, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be an object")
    return value


def _string(value: object, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{context} must be a non-empty string")
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
    """Console entry point for 2D identity detail layer generation."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    for path in generate_face_detail_layers(args.spec, args.output_dir).values():
        print(path)
