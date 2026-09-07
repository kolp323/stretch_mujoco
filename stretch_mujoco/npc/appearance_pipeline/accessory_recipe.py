"""Versioned, single-source configuration for fused NPC OBJ accessories."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path


ACCESSORY_RECIPE_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class FusedAccessoryRecipe:
    accessory_id: str
    attachment_mode: str
    mesh_scale: float
    head_clearance_m: float
    back_offset_m: float
    back_tilt_degrees: float

    @classmethod
    def from_json(cls, path: str | Path) -> "FusedAccessoryRecipe":
        source = Path(path)
        payload = json.loads(source.read_text(encoding="utf-8"))
        required = {
            "schema_version",
            "accessory_id",
            "attachment_mode",
            "mesh_scale",
            "head_clearance_m",
            "back_offset_m",
            "back_tilt_degrees",
        }
        unknown = set(payload) - required
        missing = required - set(payload)
        if unknown or missing:
            raise ValueError(
                "Accessory recipe fields mismatch: "
                f"missing={sorted(missing)}, unknown={sorted(unknown)}"
            )
        if payload["schema_version"] != ACCESSORY_RECIPE_SCHEMA_VERSION:
            raise ValueError("Unsupported accessory recipe schema_version")
        if not isinstance(payload["accessory_id"], str) or not payload["accessory_id"]:
            raise ValueError("Accessory recipe accessory_id must be a non-empty string")
        if payload["attachment_mode"] != "head_follow_fused":
            raise ValueError("Accessory recipe attachment_mode must be 'head_follow_fused'")
        values = {
            name: float(payload[name])
            for name in ("mesh_scale", "head_clearance_m", "back_offset_m", "back_tilt_degrees")
        }
        if values["mesh_scale"] <= 0 or values["back_offset_m"] < 0:
            raise ValueError("Accessory recipe has invalid scale or back offset")
        return cls(str(payload["accessory_id"]), str(payload["attachment_mode"]), **values)

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": ACCESSORY_RECIPE_SCHEMA_VERSION,
            "accessory_id": self.accessory_id,
            "attachment_mode": self.attachment_mode,
            "mesh_scale": self.mesh_scale,
            "head_clearance_m": self.head_clearance_m,
            "back_offset_m": self.back_offset_m,
            "back_tilt_degrees": self.back_tilt_degrees,
        }


def recipe_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@dataclass(frozen=True)
class FusedAccessoryRuntimeConfig:
    """Declarative inputs and outputs for one recipe-backed runtime build.

    This is deliberately separate from the recipe: the recipe owns pose facts,
    while this configuration owns machine-local source and generated paths.
    """

    recipe: str
    source_archive: str
    source_manifest: str
    source_population: str
    npc_id: str
    bundle: str
    output_dir: str
    output_manifest: str
    output_population: str

    @classmethod
    def from_json(cls, path: str | Path) -> "FusedAccessoryRuntimeConfig":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        required = {
            "schema_version",
            "recipe",
            "source_archive",
            "source_manifest",
            "source_population",
            "npc_id",
            "bundle",
            "output_dir",
            "output_manifest",
            "output_population",
        }
        unknown = set(payload) - required
        missing = required - set(payload)
        if unknown or missing:
            raise ValueError(
                "Accessory runtime config fields mismatch: "
                f"missing={sorted(missing)}, unknown={sorted(unknown)}"
            )
        if payload["schema_version"] != ACCESSORY_RECIPE_SCHEMA_VERSION:
            raise ValueError("Unsupported accessory runtime config schema_version")
        values = {field: payload[field] for field in required - {"schema_version"}}
        if not all(isinstance(value, str) and value for value in values.values()):
            raise ValueError("Accessory runtime config fields must be non-empty strings")
        return cls(**values)

    def resolve_path(self, config_path: str | Path, field: str) -> Path:
        """Resolve a configured local path relative to its config file."""
        return (Path(config_path).resolve().parent / getattr(self, field)).resolve()
