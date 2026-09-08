"""Versioned, single-source configuration for fused NPC OBJ accessories."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path


ACCESSORY_RECIPE_SCHEMA_VERSION = 2


@dataclass(frozen=True)
class FusedAccessoryRecipe:
    accessory_id: str
    attachment_mode: str
    mesh_scale: float
    head_clearance_m: float
    back_offset_m: float
    back_tilt_degrees: float
    source_vertical_anchor: float | None
    accessory_uv: tuple[float, float]

    @classmethod
    def from_json(cls, path: str | Path) -> "FusedAccessoryRecipe":
        source = Path(path)
        payload = json.loads(source.read_text(encoding="utf-8"))
        legacy_required = {
            "schema_version",
            "accessory_id",
            "attachment_mode",
            "mesh_scale",
            "head_clearance_m",
            "back_offset_m",
            "back_tilt_degrees",
        }
        required = {
            *legacy_required,
            "source_vertical_anchor",
            "accessory_uv",
        }
        if payload.get("schema_version") == 1:
            required = legacy_required
        unknown = set(payload) - required
        missing = required - set(payload)
        if unknown or missing:
            raise ValueError(
                "Accessory recipe fields mismatch: "
                f"missing={sorted(missing)}, unknown={sorted(unknown)}"
            )
        if payload["schema_version"] not in {1, ACCESSORY_RECIPE_SCHEMA_VERSION}:
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
        anchor = payload.get("source_vertical_anchor")
        if anchor is not None:
            if not isinstance(anchor, (int, float)) or isinstance(anchor, bool):
                raise ValueError("Accessory recipe source_vertical_anchor must be a number or null")
            anchor = float(anchor)
        uv = payload.get("accessory_uv", [0.0, 0.0])
        if (
            not isinstance(uv, list)
            or len(uv) != 2
            or not all(
                isinstance(value, (int, float)) and not isinstance(value, bool) for value in uv
            )
            or not all(0 <= float(value) <= 1 for value in uv)
        ):
            raise ValueError("Accessory recipe accessory_uv must be two values in [0, 1]")
        return cls(
            str(payload["accessory_id"]),
            str(payload["attachment_mode"]),
            **values,
            source_vertical_anchor=anchor,
            accessory_uv=(float(uv[0]), float(uv[1])),
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": ACCESSORY_RECIPE_SCHEMA_VERSION,
            "accessory_id": self.accessory_id,
            "attachment_mode": self.attachment_mode,
            "mesh_scale": self.mesh_scale,
            "head_clearance_m": self.head_clearance_m,
            "back_offset_m": self.back_offset_m,
            "back_tilt_degrees": self.back_tilt_degrees,
            "source_vertical_anchor": self.source_vertical_anchor,
            "accessory_uv": list(self.accessory_uv),
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
    source_format: str
    nested_archive_member: str | None
    obj_member: str | None
    source_unit_scale: float
    source_up_axis: str

    @classmethod
    def from_json(cls, path: str | Path) -> "FusedAccessoryRuntimeConfig":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        legacy_required = {
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
        required = {
            *legacy_required,
            "source_format",
            "nested_archive_member",
            "obj_member",
            "source_unit_scale",
            "source_up_axis",
        }
        if payload.get("schema_version") == 1:
            required = legacy_required
        unknown = set(payload) - required
        missing = required - set(payload)
        if unknown or missing:
            raise ValueError(
                "Accessory runtime config fields mismatch: "
                f"missing={sorted(missing)}, unknown={sorted(unknown)}"
            )
        if payload["schema_version"] not in {1, ACCESSORY_RECIPE_SCHEMA_VERSION}:
            raise ValueError("Unsupported accessory runtime config schema_version")
        values = {field: payload[field] for field in required - {"schema_version"}}
        if payload["schema_version"] == 1:
            values.update(
                {
                    "source_format": "nested_zip_obj",
                    "nested_archive_member": "source/cap.zip",
                    "obj_member": "cap.obj",
                    "source_unit_scale": 0.01,
                    "source_up_axis": "y",
                }
            )
        string_fields = required - {
            "schema_version",
            "nested_archive_member",
            "obj_member",
            "source_unit_scale",
        }
        if not all(isinstance(values[field], str) and values[field] for field in string_fields):
            raise ValueError("Accessory runtime config string fields must be non-empty")
        if values["source_format"] == "nested_zip_obj":
            if not all(
                isinstance(values[field], str) and values[field]
                for field in ("nested_archive_member", "obj_member")
            ):
                raise ValueError(
                    "nested_zip_obj runtime config needs nested_archive_member and obj_member"
                )
        elif values["source_format"] == "glb":
            if values["nested_archive_member"] is not None or values["obj_member"] is not None:
                raise ValueError("glb runtime config must not define nested archive or OBJ members")
        else:
            raise ValueError("Accessory runtime config source_format must be nested_zip_obj or glb")
        if (
            not isinstance(values["source_unit_scale"], (int, float))
            or isinstance(values["source_unit_scale"], bool)
            or float(values["source_unit_scale"]) <= 0
        ):
            raise ValueError("Accessory runtime config source_unit_scale must be positive")
        if values["source_up_axis"] != "y":
            raise ValueError("Accessory runtime config source_up_axis must be y")
        values["source_unit_scale"] = float(values["source_unit_scale"])
        return cls(**values)

    def resolve_path(self, config_path: str | Path, field: str) -> Path:
        """Resolve a configured local path relative to its config file."""
        return (Path(config_path).resolve().parent / getattr(self, field)).resolve()
