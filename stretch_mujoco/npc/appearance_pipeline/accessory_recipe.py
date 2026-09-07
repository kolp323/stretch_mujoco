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
