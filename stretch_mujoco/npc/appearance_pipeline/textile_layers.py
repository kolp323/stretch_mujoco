"""Create deterministic, UV-masked textile layers from approved PBR archives."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import zipfile
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np


TEXTILE_LAYER_SCHEMA_VERSION = 1


def generate_textile_layers(spec_path: str | Path, asset_root: str | Path) -> dict[str, Path]:
    """Tile approved archive diffuse maps into transparent semantic-region layers.

    The versioned spec pins each local source archive's hash, its archive member,
    licence and origin URL.  It therefore never reads an unreviewed raw intake
    path and fails before generating a layer if a promoted source changed.
    """
    spec_source = Path(spec_path).resolve()
    root = Path(asset_root).resolve()
    spec = _mapping(json.loads(spec_source.read_text(encoding="utf-8")), "Textile layer spec")
    _validate_spec(spec)
    base = _read_color(_rooted(root, _string(spec["base"], "Textile layer spec base")))
    outputs: dict[str, Path] = {}
    receipt_layers: dict[str, dict[str, object]] = {}
    for raw_layer in spec["layers"]:
        layer = _mapping(raw_layer, "Textile layer entry")
        _validate_layer(layer)
        layer_id = _string(layer["id"], "Textile layer id")
        archive_relative = _string(layer["source_archive"], f"Textile layer '{layer_id}' archive")
        archive = (root / archive_relative).resolve()
        if not archive.is_file():
            raise ValueError(f"Textile layer '{layer_id}' source archive is missing: {archive}")
        archive_sha = _sha256(archive.read_bytes())
        expected_sha = _sha256_value(
            layer["source_sha256"], f"Textile layer '{layer_id}' source_sha256"
        )
        if archive_sha != expected_sha:
            raise ValueError(f"Textile layer '{layer_id}' source archive SHA-256 mismatch")
        source = _archive_image(
            archive, _string(layer["archive_member"], f"Textile layer '{layer_id}' archive_member")
        )
        mask_path = _rooted(root, _string(layer["mask"], f"Textile layer '{layer_id}' mask"))
        mask = _read_mask(mask_path)
        if mask.shape != base.shape[:2]:
            raise ValueError(
                f"Textile layer '{layer_id}' semantic mask dimensions do not match base"
            )
        exclude_mask = layer.get("exclude_mask")
        if exclude_mask is not None:
            exclude_path = _rooted(
                root, _string(exclude_mask, f"Textile layer '{layer_id}' exclude_mask")
            )
            exclusion = _read_mask(exclude_path)
            if exclusion.shape != mask.shape:
                raise ValueError(
                    f"Textile layer '{layer_id}' exclude_mask dimensions do not match base"
                )
            mask = mask.copy()
            mask[exclusion > 0] = 0
        tiled = _tile_image(
            source,
            base.shape[:2],
            int(layer["tile_width_px"]),
            int(layer["seed"]),
        )
        output = _rooted(root, _string(layer["output"], f"Textile layer '{layer_id}' output"))
        rgba = np.empty((*base.shape[:2], 4), dtype=np.uint8)
        rgba[..., :3] = tiled
        rgba[..., 3] = mask
        _write_png(output, rgba)
        outputs[layer_id] = output
        receipt_layers[layer_id] = {
            "source_archive": archive_relative,
            "source_archive_sha256": archive_sha,
            "source_url": _string(layer["source_url"], f"Textile layer '{layer_id}' source_url"),
            "license": _string(layer["license"], f"Textile layer '{layer_id}' license"),
            "archive_member": _string(
                layer["archive_member"], f"Textile layer '{layer_id}' archive_member"
            ),
            "mask": _string(layer["mask"], f"Textile layer '{layer_id}' mask"),
            "mask_sha256": _sha256(mask_path.read_bytes()),
            "seed": int(layer["seed"]),
            "tile_width_px": int(layer["tile_width_px"]),
            "output": str(output.relative_to(root)),
            "output_sha256": _sha256(output.read_bytes()),
        }
        if exclude_mask is not None:
            receipt_layers[layer_id]["exclude_mask"] = _string(
                exclude_mask, f"Textile layer '{layer_id}' exclude_mask"
            )
            receipt_layers[layer_id]["exclude_mask_sha256"] = _sha256(exclude_path.read_bytes())
    receipt = _rooted(root, _string(spec["receipt"], "Textile layer spec receipt"))
    _write_json(receipt, {"schema_version": TEXTILE_LAYER_SCHEMA_VERSION, "layers": receipt_layers})
    return outputs


def _validate_spec(spec: Mapping[str, Any]) -> None:
    required = {"schema_version", "base", "receipt", "layers"}
    _exact_fields(spec, required, "Textile layer spec")
    if spec["schema_version"] != TEXTILE_LAYER_SCHEMA_VERSION:
        raise ValueError("Unsupported textile layer spec schema_version")
    if not isinstance(spec["layers"], list) or not spec["layers"]:
        raise ValueError("Textile layer spec layers must be a non-empty array")


def _validate_layer(layer: Mapping[str, Any]) -> None:
    required = {
        "id",
        "source_archive",
        "source_sha256",
        "source_url",
        "license",
        "archive_member",
        "mask",
        "output",
        "tile_width_px",
        "seed",
    }
    allowed = required | {"exclude_mask"}
    unknown, missing = set(layer) - allowed, required - set(layer)
    if unknown or missing:
        raise ValueError(
            f"Textile layer entry fields mismatch: missing={sorted(missing)}, unknown={sorted(unknown)}"
        )
    if not isinstance(layer["tile_width_px"], int) or layer["tile_width_px"] <= 0:
        raise ValueError("Textile layer tile_width_px must be a positive integer")
    if (
        not isinstance(layer["seed"], int)
        or isinstance(layer["seed"], bool)
        or not 0 <= layer["seed"] < 2**32
    ):
        raise ValueError("Textile layer seed must be an unsigned 32-bit integer")


def _archive_image(archive: Path, member: str) -> np.ndarray:
    try:
        with zipfile.ZipFile(archive) as source:
            encoded = source.read(member)
    except (KeyError, zipfile.BadZipFile) as error:
        raise ValueError(
            f"Could not read textile archive member '{member}' from '{archive}'"
        ) from error
    image = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Textile archive member '{member}' is not a decodable colour image")
    return image


def _tile_image(
    source: np.ndarray, shape: tuple[int, int], tile_width: int, seed: int
) -> np.ndarray:
    height, width = shape
    tile_height = max(1, round(tile_width * source.shape[0] / source.shape[1]))
    tile = cv2.resize(source, (tile_width, tile_height), interpolation=cv2.INTER_AREA)
    generator = np.random.default_rng(seed)
    offset_x = int(generator.integers(0, tile_width))
    offset_y = int(generator.integers(0, tile_height))
    repetitions_x = (width + offset_x + tile_width - 1) // tile_width
    repetitions_y = (height + offset_y + tile_height - 1) // tile_height
    repeated = np.tile(tile, (repetitions_y, repetitions_x, 1))
    return repeated[offset_y : offset_y + height, offset_x : offset_x + width]


def _read_color(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not decode textile base '{path}'")
    return image


def _read_mask(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Could not decode textile semantic mask '{path}'")
    if image.ndim == 2:
        return image
    if image.ndim == 3 and image.shape[2] == 4:
        return image[..., 3]
    if image.ndim == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    raise ValueError(f"Textile semantic mask has unsupported channels: {path}")


def _rooted(root: Path, relative: str) -> Path:
    destination = (root / relative).resolve()
    if root != destination and root not in destination.parents:
        raise ValueError(f"Textile output must remain below asset root: {relative}")
    return destination


def _write_png(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp.png")
    if not cv2.imwrite(str(temporary), image, [cv2.IMWRITE_PNG_COMPRESSION, 9]):
        raise OSError(f"Could not write textile layer '{path}'")
    os.replace(temporary, path)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_value(value: object, context: str) -> str:
    result = _string(value, context)
    if len(result) != 64 or any(character not in "0123456789abcdef" for character in result):
        raise ValueError(f"{context} must be a lowercase SHA-256 hex digest")
    return result


def _string(value: object, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{context} must be a non-empty string")
    return value


def _mapping(value: object, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be an object")
    return value


def _exact_fields(payload: Mapping[str, Any], required: set[str], context: str) -> None:
    unknown, missing = set(payload) - required, required - set(payload)
    if unknown or missing:
        raise ValueError(
            f"{context} fields mismatch: missing={sorted(missing)}, unknown={sorted(unknown)}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--asset-root", required=True, type=Path)
    args = parser.parse_args()
    for layer_id, path in generate_textile_layers(args.spec, args.asset_root).items():
        print(f"{layer_id}: {path}")


if __name__ == "__main__":
    main()
