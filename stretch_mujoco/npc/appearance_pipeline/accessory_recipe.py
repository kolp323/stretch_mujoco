"""Versioned, single-source configuration for fused NPC OBJ accessories."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path


ACCESSORY_RECIPE_SCHEMA_VERSION = 5
ACCESSORY_RUNTIME_CONFIG_SCHEMA_VERSION = 2


@dataclass(frozen=True)
class SmplxHeadSurfaceFallback:
    """Recipe-owned opaque hair underlay for sparse OBJ hair cards."""

    color_bgr: tuple[int, int, int]
    front_hairline_z_m: float
    temple_min_z_m: float
    temple_min_y_m: float
    rear_min_z_m: float
    rear_min_y_m: float
    head_min_z_m: float
    head_radius_m: float

    def as_dict(self) -> dict[str, object]:
        return {
            "mode": "smplx_head_uv_v1",
            "color_bgr": list(self.color_bgr),
            "front_hairline_z_m": self.front_hairline_z_m,
            "temple_min_z_m": self.temple_min_z_m,
            "temple_min_y_m": self.temple_min_y_m,
            "rear_min_z_m": self.rear_min_z_m,
            "rear_min_y_m": self.rear_min_y_m,
            "head_min_z_m": self.head_min_z_m,
            "head_radius_m": self.head_radius_m,
        }


@dataclass(frozen=True)
class AccessoryRenderPolicy:
    """Rendering facts that belong to an OBJ accessory recipe, not a receipt."""

    double_sided: bool
    surface_fallback: SmplxHeadSurfaceFallback | None

    def as_dict(self) -> dict[str, object]:
        return {
            "double_sided": self.double_sided,
            "surface_fallback": (
                None if self.surface_fallback is None else self.surface_fallback.as_dict()
            ),
        }


@dataclass(frozen=True)
class FusedAccessoryRecipe:
    accessory_id: str
    attachment_mode: str
    mesh_scale: float
    lateral_offset_m: float
    head_clearance_m: float
    back_offset_m: float
    back_tilt_degrees: float
    roll_degrees: float
    yaw_degrees: float
    source_vertical_anchor: float | None
    accessory_uv: tuple[float, float]
    render_policy: AccessoryRenderPolicy

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
        version_2_required = {
            *legacy_required,
            "source_vertical_anchor",
            "accessory_uv",
        }
        version_3_required = {*version_2_required, "yaw_degrees"}
        version_4_required = {*version_3_required, "lateral_offset_m", "roll_degrees"}
        current_required = {*version_4_required, "render_policy"}
        schema_version = payload.get("schema_version")
        if schema_version == 1:
            required = legacy_required
        elif schema_version == 2:
            required = version_2_required
        elif schema_version == 3:
            required = version_3_required
        elif schema_version == 4:
            required = version_4_required
        else:
            required = current_required
        unknown = set(payload) - required
        missing = required - set(payload)
        if unknown or missing:
            raise ValueError(
                "Accessory recipe fields mismatch: "
                f"missing={sorted(missing)}, unknown={sorted(unknown)}"
            )
        if payload["schema_version"] not in {1, 2, 3, 4, ACCESSORY_RECIPE_SCHEMA_VERSION}:
            raise ValueError("Unsupported accessory recipe schema_version")
        if not isinstance(payload["accessory_id"], str) or not payload["accessory_id"]:
            raise ValueError("Accessory recipe accessory_id must be a non-empty string")
        if payload["attachment_mode"] != "head_follow_fused":
            raise ValueError("Accessory recipe attachment_mode must be 'head_follow_fused'")
        numeric_fields = (
            "mesh_scale",
            "head_clearance_m",
            "back_offset_m",
            "back_tilt_degrees",
        )
        if not all(
            isinstance(payload[name], (int, float)) and not isinstance(payload[name], bool)
            for name in numeric_fields
        ):
            raise ValueError("Accessory recipe pose fields must be numbers")
        values = {name: float(payload[name]) for name in numeric_fields}
        if not all(math.isfinite(value) for value in values.values()):
            raise ValueError("Accessory recipe pose fields must be finite")
        if values["mesh_scale"] <= 0:
            raise ValueError("Accessory recipe mesh_scale must be positive")
        raw_lateral_offset_m = payload.get("lateral_offset_m", 0.0)
        raw_roll_degrees = payload.get("roll_degrees", 0.0)
        raw_yaw_degrees = payload.get("yaw_degrees", 0.0)
        if not all(
            isinstance(value, (int, float)) and not isinstance(value, bool)
            for value in (raw_lateral_offset_m, raw_roll_degrees, raw_yaw_degrees)
        ):
            raise ValueError("Accessory recipe offsets and rotations must be numbers")
        lateral_offset_m = float(raw_lateral_offset_m)
        roll_degrees = float(raw_roll_degrees)
        yaw_degrees = float(raw_yaw_degrees)
        if not all(math.isfinite(value) for value in (lateral_offset_m, roll_degrees, yaw_degrees)):
            raise ValueError("Accessory recipe offsets and rotations must be finite")
        anchor = payload.get("source_vertical_anchor")
        if anchor is not None:
            if not isinstance(anchor, (int, float)) or isinstance(anchor, bool):
                raise ValueError("Accessory recipe source_vertical_anchor must be a number or null")
            anchor = float(anchor)
            if not math.isfinite(anchor):
                raise ValueError("Accessory recipe source_vertical_anchor must be finite")
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
        render_policy = _parse_render_policy(payload.get("render_policy"))
        return cls(
            str(payload["accessory_id"]),
            str(payload["attachment_mode"]),
            **values,
            lateral_offset_m=lateral_offset_m,
            roll_degrees=roll_degrees,
            yaw_degrees=yaw_degrees,
            source_vertical_anchor=anchor,
            accessory_uv=(float(uv[0]), float(uv[1])),
            render_policy=render_policy,
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": ACCESSORY_RECIPE_SCHEMA_VERSION,
            "accessory_id": self.accessory_id,
            "attachment_mode": self.attachment_mode,
            "mesh_scale": self.mesh_scale,
            "lateral_offset_m": self.lateral_offset_m,
            "head_clearance_m": self.head_clearance_m,
            "back_offset_m": self.back_offset_m,
            "back_tilt_degrees": self.back_tilt_degrees,
            "roll_degrees": self.roll_degrees,
            "yaw_degrees": self.yaw_degrees,
            "source_vertical_anchor": self.source_vertical_anchor,
            "accessory_uv": list(self.accessory_uv),
            "render_policy": self.render_policy.as_dict(),
        }


def _parse_render_policy(value: object) -> AccessoryRenderPolicy:
    """Parse the versioned render contract, defaulting legacy recipes safely."""
    if value is None:
        return AccessoryRenderPolicy(double_sided=False, surface_fallback=None)
    if not isinstance(value, dict) or set(value) != {"double_sided", "surface_fallback"}:
        raise ValueError("Accessory recipe render_policy fields mismatch")
    double_sided = value["double_sided"]
    if not isinstance(double_sided, bool):
        raise ValueError("Accessory recipe render_policy.double_sided must be boolean")
    raw_fallback = value["surface_fallback"]
    if raw_fallback is None:
        return AccessoryRenderPolicy(double_sided=double_sided, surface_fallback=None)
    required = {
        "mode",
        "color_bgr",
        "front_hairline_z_m",
        "temple_min_z_m",
        "temple_min_y_m",
        "rear_min_z_m",
        "rear_min_y_m",
        "head_min_z_m",
        "head_radius_m",
    }
    if not isinstance(raw_fallback, dict) or set(raw_fallback) != required:
        raise ValueError("Accessory recipe surface_fallback fields mismatch")
    if raw_fallback["mode"] != "smplx_head_uv_v1":
        raise ValueError("Accessory recipe surface_fallback mode is unsupported")
    color = raw_fallback["color_bgr"]
    if (
        not isinstance(color, list)
        or len(color) != 3
        or not all(isinstance(item, int) and not isinstance(item, bool) for item in color)
        or not all(0 <= item <= 255 for item in color)
    ):
        raise ValueError("Accessory recipe surface_fallback.color_bgr must be three bytes")
    numeric_names = required - {"mode", "color_bgr"}
    if not all(
        isinstance(raw_fallback[name], (int, float)) and not isinstance(raw_fallback[name], bool)
        for name in numeric_names
    ):
        raise ValueError("Accessory recipe surface_fallback thresholds must be numbers")
    numeric = {name: float(raw_fallback[name]) for name in numeric_names}
    if not all(math.isfinite(item) for item in numeric.values()) or numeric["head_radius_m"] <= 0:
        raise ValueError("Accessory recipe surface_fallback thresholds are invalid")
    return AccessoryRenderPolicy(
        double_sided=double_sided,
        surface_fallback=SmplxHeadSurfaceFallback(
            color_bgr=tuple(color),
            front_hairline_z_m=numeric["front_hairline_z_m"],
            temple_min_z_m=numeric["temple_min_z_m"],
            temple_min_y_m=numeric["temple_min_y_m"],
            rear_min_z_m=numeric["rear_min_z_m"],
            rear_min_y_m=numeric["rear_min_y_m"],
            head_min_z_m=numeric["head_min_z_m"],
            head_radius_m=numeric["head_radius_m"],
        ),
    )


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
        if payload["schema_version"] not in {1, ACCESSORY_RUNTIME_CONFIG_SCHEMA_VERSION}:
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
