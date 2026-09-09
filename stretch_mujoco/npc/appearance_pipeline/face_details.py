"""Generate deterministic, mask-constrained 2D facial identity detail layers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np


# The reference UV face places the lip line below the naïve geometric centre
# used by the first facial-hair pass.  Keep every beard style on the lower
# face, rather than allowing its moustache component to paint over the lips.
FACIAL_HAIR_VERTICAL_OFFSET = 0.055


def generate_face_detail_layers(spec_path: str | Path, output_dir: str | Path) -> dict[str, Path]:
    """Generate freckles, brows, facial hair, or 2D glasses overlays.

    Facial decals are clipped by a formal ``face`` mask. Full beards additionally
    require the topology-derived ``sideburn_mask`` so their sideburns stay by the
    ears on every persona rather than being guessed from the 2D face atlas.
    The glasses output is a texture decal, not a replacement for a geometric
    glasses asset.
    """
    source = Path(spec_path).resolve()
    spec = _mapping(json.loads(source.read_text(encoding="utf-8")), "Face detail spec")
    face_mask_path = source.parent / _string(spec.get("face_mask"), "Face detail face_mask")
    face_mask = _read_mask(face_mask_path, "Face detail face_mask")
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
        mask_name = style.get("mask")
        style_mask = (
            _read_mask(
                source.parent / _string(mask_name, f"Face detail style '{style_id}' mask"),
                f"Face detail style '{style_id}' mask",
            )
            if mask_name is not None
            else face_mask
        )
        if style_mask.shape != face_mask.shape:
            raise ValueError(
                f"Face detail style '{style_id}' mask dimensions do not match face_mask"
            )
        artwork = np.zeros(style_mask.shape, dtype=np.uint8)
        alpha_mask = style_mask
        if kind == "freckles":
            _draw_freckles(
                artwork, style_mask, int(style.get("seed", 0)), int(style.get("count", 28))
            )
        elif kind == "brows":
            _draw_brows(artwork, style_mask)
        elif kind == "beard":
            _draw_beard(artwork, style_mask)
        elif kind == "moustache_handlebar":
            _draw_handlebar_moustache(artwork, style_mask)
        elif kind == "beard_full":
            _draw_full_beard(artwork, style_mask, include_sideburns=False)
            sideburn_mask = _sideburn_mask(source, style, style_id, face_mask.shape)
            artwork[sideburn_mask > 0] = 255
            alpha_mask = np.maximum(style_mask, sideburn_mask)
        elif kind == "beard_full_core":
            _draw_full_beard(artwork, style_mask, include_sideburns=False)
        elif kind == "beard_boxed":
            _draw_boxed_beard(artwork, style_mask)
        elif kind == "goatee":
            _draw_goatee(
                artwork,
                style_mask,
                vertical_offset=_facial_hair_vertical_offset(style, style_id),
            )
        elif kind == "sideburns":
            artwork[style_mask > 0] = 255
        elif kind == "glasses_2d":
            _draw_glasses(artwork, style_mask)
        else:
            raise ValueError(f"Face detail style '{style_id}' has unsupported kind '{kind}'")
        alpha = np.minimum(artwork, alpha_mask)
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
            (x + width // 2, _facial_hair_y(y, height, 0.84)),
            (max(2, round(width * 0.13)), max(2, round(height * 0.10))),
            0,
            10,
            170,
            255,
            -1,
            lineType=cv2.LINE_AA,
        )


def _draw_handlebar_moustache(image: np.ndarray, mask: np.ndarray) -> None:
    """Draw a classic 八字胡 whose tips sweep upward from the philtrum."""
    for region in _decal_regions(mask):
        x, y, width, height = _bounds(region)
        center_x = x + round(width * 0.5)
        center_y = _facial_hair_y(y, height, 0.64)
        thickness = max(2, round(height * 0.042))
        for direction in (-1, 1):
            points = np.array(
                [
                    (center_x, center_y),
                    (center_x + round(direction * width * 0.13), center_y + round(height * 0.018)),
                    (center_x + round(direction * width * 0.24), center_y - round(height * 0.010)),
                    (center_x + round(direction * width * 0.28), center_y - round(height * 0.075)),
                ],
                dtype=np.int32,
            )
            cv2.polylines(image, [points], False, 255, thickness, lineType=cv2.LINE_AA)
            tip = tuple(int(value) for value in points[-1])
            cv2.circle(image, tip, max(1, thickness // 2), 255, -1, lineType=cv2.LINE_AA)


def _draw_full_beard(
    image: np.ndarray,
    mask: np.ndarray,
    *,
    include_sideburns: bool = True,
) -> None:
    """Draw jaw coverage, chin, moustache, and optional legacy sideburns."""
    for region in _decal_regions(mask):
        x, y, width, height = _bounds(region)
        center_x = x + round(width * 0.5)
        # The lower oval gives dense chin coverage; the side polygons connect it
        # to the cheek line without extending into the eye area.
        cv2.ellipse(
            image,
            (center_x, _facial_hair_y(y, height, 0.79)),
            (max(3, round(width * 0.285)), max(3, round(height * 0.225))),
            0,
            0,
            180,
            255,
            -1,
            lineType=cv2.LINE_AA,
        )
        if not include_sideburns:
            _draw_moustache_bar(image, x, y, width, height)
            continue
        for direction in (-1, 1):
            cheek = np.array(
                [
                    (
                        center_x + round(direction * width * 0.18),
                        _facial_hair_y(y, height, 0.54),
                    ),
                    (
                        center_x + round(direction * width * 0.29),
                        _facial_hair_y(y, height, 0.50),
                    ),
                    (
                        center_x + round(direction * width * 0.31),
                        _facial_hair_y(y, height, 0.73),
                    ),
                    (
                        center_x + round(direction * width * 0.20),
                        _facial_hair_y(y, height, 0.90),
                    ),
                ],
                dtype=np.int32,
            )
            cv2.fillConvexPoly(image, cheek, 255, lineType=cv2.LINE_AA)
        _draw_moustache_bar(image, x, y, width, height)


def _draw_boxed_beard(image: np.ndarray, mask: np.ndarray) -> None:
    """Draw a neat short beard with defined, office-friendly edges."""
    for region in _decal_regions(mask):
        x, y, width, height = _bounds(region)
        center_x = x + round(width * 0.5)
        cv2.rectangle(
            image,
            (x + round(width * 0.31), _facial_hair_y(y, height, 0.70)),
            (x + round(width * 0.69), _facial_hair_y(y, height, 0.89)),
            255,
            -1,
            lineType=cv2.LINE_AA,
        )
        cv2.ellipse(
            image,
            (center_x, _facial_hair_y(y, height, 0.88)),
            (max(2, round(width * 0.19)), max(2, round(height * 0.06))),
            0,
            0,
            180,
            255,
            -1,
            lineType=cv2.LINE_AA,
        )
        _draw_moustache_bar(image, x, y, width, height)


def _draw_goatee(image: np.ndarray, mask: np.ndarray, *, vertical_offset: float) -> None:
    """Draw a separated moustache and pointed chin goatee."""
    for region in _decal_regions(mask):
        x, y, width, height = _bounds(region)
        center_x = x + round(width * 0.5)
        goatee = np.array(
            [
                (center_x - round(width * 0.10), _facial_hair_y(y, height, 0.74, vertical_offset)),
                (center_x + round(width * 0.10), _facial_hair_y(y, height, 0.74, vertical_offset)),
                (center_x + round(width * 0.13), _facial_hair_y(y, height, 0.87, vertical_offset)),
                (center_x, _facial_hair_y(y, height, 0.94, vertical_offset)),
                (center_x - round(width * 0.13), _facial_hair_y(y, height, 0.87, vertical_offset)),
            ],
            dtype=np.int32,
        )
        cv2.fillConvexPoly(image, goatee, 255, lineType=cv2.LINE_AA)
        _draw_moustache_bar(image, x, y, width, height, vertical_offset=vertical_offset)


def _draw_moustache_bar(
    image: np.ndarray,
    x: int,
    y: int,
    width: int,
    height: int,
    *,
    vertical_offset: float = FACIAL_HAIR_VERTICAL_OFFSET,
) -> None:
    """Add a short, gently curved moustache centered just above the mouth."""
    center_x = x + round(width * 0.5)
    center_y = _facial_hair_y(y, height, 0.64, vertical_offset)
    axes = (max(2, round(width * 0.145)), max(1, round(height * 0.037)))
    cv2.ellipse(
        image,
        (center_x, center_y),
        axes,
        0,
        190,
        350,
        255,
        -1,
        lineType=cv2.LINE_AA,
    )


def _facial_hair_y(
    y: int, height: int, normalized_y: float, vertical_offset: float = FACIAL_HAIR_VERTICAL_OFFSET
) -> int:
    """Map a facial-hair landmark into the lower, lip-safe UV position."""
    return y + round(height * (normalized_y + vertical_offset))


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


def _read_mask(path: Path, context: str) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Could not decode {context} '{path}'")
    return mask


def _sideburn_mask(
    source: Path,
    style: Mapping[str, Any],
    style_id: object,
    expected_shape: tuple[int, int],
) -> np.ndarray:
    """Read the mandatory mesh-derived sideburn region for a full beard."""
    mask_name = _string(style.get("sideburn_mask"), f"Face detail style '{style_id}' sideburn_mask")
    sideburn_mask = _read_mask(
        source.parent / mask_name, f"Face detail style '{style_id}' sideburn_mask"
    )
    if sideburn_mask.shape != expected_shape:
        raise ValueError(
            f"Face detail style '{style_id}' sideburn_mask dimensions do not match face_mask"
        )
    return sideburn_mask


def _facial_hair_vertical_offset(style: Mapping[str, Any], style_id: object) -> float:
    """Return a small, style-local downward adjustment for a facial-hair layer."""
    value = style.get("vertical_offset", FACIAL_HAIR_VERTICAL_OFFSET)
    if not isinstance(value, int | float):
        raise ValueError(f"Face detail style '{style_id}' vertical_offset must be a number")
    offset = float(value)
    if not 0.0 <= offset <= 0.12:
        raise ValueError(f"Face detail style '{style_id}' vertical_offset must be in [0.0, 0.12]")
    return offset


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
