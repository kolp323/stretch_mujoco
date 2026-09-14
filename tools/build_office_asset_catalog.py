"""Build a small, reviewable catalog of HSSD assets useful in an office scene.

The catalog contains source metadata only. It deliberately does not copy or
convert GLB files, so running it does not make the repository substantially
larger.
"""

from __future__ import annotations

import argparse
import csv
import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from stretch_mujoco.paths import configured_path, require_external_directory

DEFAULT_HSSD_ROOT = configured_path("STRETCH_MUJOCO_HSSD_ROOT")
DEFAULT_OUTPUT = Path("stretch_mujoco/models/assets/office_assets/catalog/hssd_candidates.json")


@dataclass(frozen=True)
class CategoryRule:
    output_category: str
    main_categories: tuple[str, ...]
    name_terms: tuple[str, ...] = ()
    prefer_office: bool = False
    limit: int = 12


RULES = (
    CategoryRule("furniture/chairs", ("chair",), ("office", "swivel", "task"), True, 20),
    CategoryRule("furniture/desks", ("table",), ("desk", "writing", "computer"), True, 20),
    CategoryRule("furniture/sofas", ("couch",), ("sofa", "settee"), False, 10),
    CategoryRule(
        "furniture/storage",
        ("cabinet", "shelves", "chest_of_drawers", "wardrobe"),
        ("office", "bookcase", "storage", "filing"),
        True,
        16,
    ),
    CategoryRule(
        "electronics",
        ("monitor", "keyboard", "laptop", "printer"),
        ("monitor", "keyboard", "laptop", "printer", "computer"),
        True,
        16,
    ),
    CategoryRule("lighting", ("table_lamp", "floor_lamp"), ("lamp", "light"), False, 12),
    CategoryRule("props/plants", ("potted_plant",), ("indoor", "plant", "orchid"), False, 10),
    CategoryRule("props/trash_bins", ("trashcan",), ("waste", "paper", "bin"), True, 10),
    CategoryRule("props/books", ("book",), ("book", "notebook"), True, 12),
    CategoryRule(
        "props/drinkware",
        ("cup", "mug", "bottle", "drinkware"),
        ("cup", "mug", "bottle", "coffee"),
        False,
        12,
    ),
    CategoryRule("decoration/rugs", ("carpet",), ("rug", "runner"), False, 8),
    CategoryRule("decoration/curtains", ("curtain",), ("curtain", "blind"), False, 8),
    CategoryRule("decoration/wall_art", ("picture",), ("poster", "print", "art"), False, 10),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hssd-root",
        type=Path,
        default=DEFAULT_HSSD_ROOT,
        help="HSSD dataset root (or set STRETCH_MUJOCO_HSSD_ROOT)",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _text(row: dict[str, str]) -> str:
    fields = ("name", "foundIn", "floorplanner-category-tags", "main_category")
    return " ".join(row.get(field, "") or "" for field in fields).lower()


def _score(row: dict[str, str], rule: CategoryRule) -> tuple[int, str]:
    text = _text(row)
    score = 0
    if "office" in (row.get("foundIn") or "").lower():
        score += 100 if rule.prefer_office else 20
    score += sum(12 for term in rule.name_terms if term in text)
    if row.get("main_category") in rule.main_categories:
        score += 10
    if (row.get("isArticulatable") or "").lower() == "true":
        score += 2
    return score, row.get("id", "")


def _matches(row: dict[str, str], rule: CategoryRule) -> bool:
    main_category = row.get("main_category", "")
    text = _text(row)
    if main_category in rule.main_categories:
        if rule.output_category == "furniture/desks":
            return any(term in text for term in rule.name_terms) or "office" in text
        return True
    return bool(rule.name_terms) and any(term in text for term in rule.name_terms)


def _find_object_config(root: Path, asset_id: str) -> Path | None:
    candidates = (
        root / "objects" / asset_id[0] / f"{asset_id}.object_config.json",
        root / "objects" / "decomposed" / asset_id[0] / f"{asset_id}.object_config.json",
    )
    return next((path for path in candidates if path.is_file()), None)


def _read_glb_summary(path: Path) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "exists": path.is_file(),
        "size_bytes": path.stat().st_size if path.is_file() else None,
    }
    if not path.is_file():
        return summary
    try:
        with path.open("rb") as stream:
            magic, version, _ = struct.unpack("<4sII", stream.read(12))
            chunk_length, chunk_type = struct.unpack("<II", stream.read(8))
            if magic != b"glTF" or chunk_type != 0x4E4F534A:
                return summary
            document = json.loads(stream.read(chunk_length).decode("utf-8").rstrip("\x00 "))
        summary.update(
            {
                "gltf_version": version,
                "mesh_count": len(document.get("meshes", [])),
                "material_count": len(document.get("materials", [])),
                "texture_count": len(document.get("textures", [])),
                "image_count": len(document.get("images", [])),
                "embedded_image_count": sum(
                    "bufferView" in image for image in document.get("images", [])
                ),
            }
        )
    except (OSError, ValueError, json.JSONDecodeError, struct.error):
        summary["inspection_error"] = True
    return summary


def _candidate(root: Path, row: dict[str, str], rule: CategoryRule) -> dict[str, Any] | None:
    asset_id = row["id"]
    config_path = _find_object_config(root, asset_id)
    if config_path is None:
        return None
    config = json.loads(config_path.read_text(encoding="utf-8"))
    render_path = config_path.parent / config["render_asset"]
    collision_asset = config.get("collision_asset")
    collision_path = config_path.parent / collision_asset if collision_asset else None
    return {
        "asset_id": asset_id,
        "display_name": row.get("name") or asset_id,
        "office_asset_category": rule.output_category,
        "hssd": {
            "main_category": row.get("main_category") or None,
            "super_category": row.get("super_category") or None,
            "found_in": [item.strip() for item in (row.get("foundIn") or "").split(",") if item],
            "dimensions_y_up_m": [
                float(value) for value in (row.get("aligned.dims") or "").split(",") if value
            ],
            "semantic_id": config.get("semantic_id"),
            "is_articulatable": (row.get("isArticulatable") or "").lower() == "true",
        },
        "source": {
            "object_config": str(config_path),
            "render_asset": str(render_path),
            "collision_asset": str(collision_path) if collision_path else None,
            "license": "CC-BY-NC-4.0",
        },
        "render_glb": _read_glb_summary(render_path),
        "conversion_status": "not_converted",
        "review_status": "unreviewed",
    }


def build_catalog(root: Path) -> dict[str, Any]:
    semantics_path = root / "semantics" / "objects.csv"
    with semantics_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))

    candidates: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    for rule in RULES:
        matches = [row for row in rows if _matches(row, rule)]
        matches.sort(key=lambda row: _score(row, rule), reverse=True)
        selected = []
        for row in matches:
            item = _candidate(root, row, rule)
            if item is not None:
                selected.append(item)
            if len(selected) >= rule.limit:
                break
        candidates.extend(selected)
        counts[rule.output_category] = len(selected)

    return {
        "schema_version": 1,
        "description": "Reviewable HSSD candidates for the custom MuJoCo office asset library.",
        "source_dataset": {
            "name": "Habitat Synthetic Scenes Dataset (HSSD)",
            "root": str(root),
            "license": "CC-BY-NC-4.0",
        },
        "coordinate_note": "HSSD assets are Y-up/front -Z and require conversion to MuJoCo Z-up.",
        "category_counts": counts,
        "candidate_count": len(candidates),
        "candidates": candidates,
    }


def main() -> None:
    args = parse_args()
    try:
        hssd_root = require_external_directory(
            args.hssd_root,
            environment_variable="STRETCH_MUJOCO_HSSD_ROOT",
            description="HSSD dataset root",
        )
    except (ValueError, FileNotFoundError) as exc:
        raise SystemExit(f"error: {exc}") from exc
    catalog = build_catalog(hssd_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(catalog, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Wrote {catalog['candidate_count']} candidates to {args.output}")


if __name__ == "__main__":
    main()
