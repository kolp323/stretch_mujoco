"""NPC asset manifest parsing and preflight validation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping

import cv2

ASSET_SCHEMA_VERSION = 1


def _mapping(value: object, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be an object")
    return value


class AssetFormat(str, Enum):
    MESH_SEQUENCE = "mesh_sequence"
    ARTICULATED = "articulated"


@dataclass(frozen=True)
class ClipManifest:
    fps: float
    loop: bool
    root_motion: str
    frames: tuple[str, ...]
    markers: tuple[dict[str, object], ...] = ()


@dataclass(frozen=True)
class AppearanceManifest:
    textures: dict[str, str]
    texture_topology_id: str


@dataclass(frozen=True)
class AccessoryManifest:
    mesh: str
    anchors: str


@dataclass(frozen=True)
class AssetBundle:
    bundle_id: str
    format: AssetFormat
    topology_id: str
    coordinate_system: str
    unit: str
    height_m: float
    material_slots: tuple[str, ...]
    appearances: dict[str, AppearanceManifest]
    accessories: dict[str, AccessoryManifest]
    clips: dict[str, ClipManifest]
    sha256: dict[str, str]
    asset_quality: str = "production"


@dataclass(frozen=True)
class NpcAssetManifest:
    bundles: dict[str, AssetBundle]
    source_path: Path

    @classmethod
    def from_json(cls, path: str | Path) -> "NpcAssetManifest":
        source = Path(path).resolve()
        payload = json.loads(source.read_text(encoding="utf-8"))
        if payload.get("schema_version") != ASSET_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported NPC asset schema_version {payload.get('schema_version')!r}; "
                f"expected {ASSET_SCHEMA_VERSION}"
            )
        bundles_data = payload.get("bundles")
        if not isinstance(bundles_data, Mapping) or not bundles_data:
            raise ValueError("NPC asset manifest must define at least one bundle")
        bundles: dict[str, AssetBundle] = {}
        for bundle_id, raw_bundle in bundles_data.items():
            if not isinstance(raw_bundle, Mapping):
                raise ValueError(f"Asset bundle '{bundle_id}' must be an object")
            required = {
                "format",
                "topology_id",
                "coordinate_system",
                "unit",
                "height_m",
                "material_slots",
                "appearances",
                "clips",
                "sha256",
            }
            missing = required - raw_bundle.keys()
            if missing:
                raise ValueError(
                    f"Asset bundle '{bundle_id}' missing fields: {', '.join(sorted(missing))}"
                )
            raw_slots = raw_bundle["material_slots"]
            if not isinstance(raw_slots, list) or not all(
                isinstance(item, str) for item in raw_slots
            ):
                raise ValueError(f"Asset bundle '{bundle_id}' material_slots must be strings")
            slots = tuple(raw_slots)
            appearances_data = _mapping(
                raw_bundle["appearances"], f"Asset bundle '{bundle_id}' appearances"
            )
            appearances: dict[str, AppearanceManifest] = {}
            for appearance_id, raw_appearance in appearances_data.items():
                item = _mapping(raw_appearance, f"Appearance '{appearance_id}'")
                if "textures" not in item or "texture_topology_id" not in item:
                    raise ValueError(f"Appearance '{appearance_id}' is incomplete")
                textures = _mapping(item["textures"], f"Appearance '{appearance_id}' textures")
                appearances[str(appearance_id)] = AppearanceManifest(
                    textures={str(key): str(value) for key, value in textures.items()},
                    texture_topology_id=str(item["texture_topology_id"]),
                )
            accessories: dict[str, AccessoryManifest] = {}
            for accessory_id, raw_accessory in _mapping(
                raw_bundle.get("accessories", {}), f"Asset bundle '{bundle_id}' accessories"
            ).items():
                item = _mapping(raw_accessory, f"Accessory '{accessory_id}'")
                if not isinstance(item.get("mesh"), str) or not isinstance(
                    item.get("anchors"), str
                ):
                    raise ValueError(f"Accessory '{accessory_id}' requires mesh and anchors")
                accessories[str(accessory_id)] = AccessoryManifest(item["mesh"], item["anchors"])
            clips_data = _mapping(raw_bundle["clips"], f"Asset bundle '{bundle_id}' clips")
            clips: dict[str, ClipManifest] = {}
            for clip_id, raw_clip in clips_data.items():
                item = _mapping(raw_clip, f"Clip '{clip_id}'")
                clip_required = {"fps", "loop", "root_motion", "frames"}
                clip_missing = clip_required - item.keys()
                if clip_missing:
                    raise ValueError(
                        f"Clip '{clip_id}' missing fields: {', '.join(sorted(clip_missing))}"
                    )
                frames = item["frames"]
                markers = item.get("markers", [])
                if not isinstance(frames, list) or not all(
                    isinstance(frame, str) for frame in frames
                ):
                    raise ValueError(f"Clip '{clip_id}' frames must be a list of paths")
                if not isinstance(markers, list) or not all(
                    isinstance(marker, Mapping) for marker in markers
                ):
                    raise ValueError(f"Clip '{clip_id}' markers must be a list of objects")
                clips[str(clip_id)] = ClipManifest(
                    fps=float(item["fps"]),
                    loop=bool(item["loop"]),
                    root_motion=str(item["root_motion"]),
                    frames=tuple(frames),
                    markers=tuple(dict(marker) for marker in markers),
                )
            hashes = _mapping(raw_bundle["sha256"], f"Asset bundle '{bundle_id}' sha256")
            bundles[str(bundle_id)] = AssetBundle(
                bundle_id=str(bundle_id),
                format=AssetFormat(str(raw_bundle["format"])),
                topology_id=str(raw_bundle["topology_id"]),
                coordinate_system=str(raw_bundle["coordinate_system"]),
                unit=str(raw_bundle["unit"]),
                height_m=float(raw_bundle["height_m"]),
                material_slots=slots,
                appearances=appearances,
                accessories=accessories,
                clips=clips,
                sha256={str(key): str(value) for key, value in hashes.items()},
                asset_quality=str(raw_bundle.get("asset_quality", "production")),
            )
        manifest = cls(bundles, source)
        manifest.validate()
        return manifest

    def validate(self) -> None:
        errors: list[str] = []
        for bundle in self.bundles.values():
            self._validate_bundle(bundle, errors)
        if errors:
            raise ValueError("Invalid NPC asset manifest:\n- " + "\n- ".join(errors))

    def validate_population(self, population: Any) -> None:
        errors: list[str] = []
        for npc_id, definition in population.npcs.items():
            bundle = self.bundles.get(definition.embodiment.bundle)
            if bundle is None:
                errors.append(
                    f"NPC '{npc_id}' references unknown bundle '{definition.embodiment.bundle}'"
                )
                continue
            if definition.embodiment.appearance not in bundle.appearances:
                errors.append(
                    f"NPC '{npc_id}' references unknown appearance "
                    f"'{definition.embodiment.appearance}' in bundle '{bundle.bundle_id}'"
                )
            unknown_accessories = set(definition.embodiment.accessories) - bundle.accessories.keys()
            if unknown_accessories:
                errors.append(
                    f"NPC '{npc_id}' references unknown accessories: {', '.join(sorted(unknown_accessories))}"
                )
            if not definition.embodiment.animation_graph:
                errors.append(f"NPC '{npc_id}' must declare an animation graph")
        if errors:
            raise ValueError("Invalid NPC population assets:\n- " + "\n- ".join(errors))

    def animation_graph(self, bundle_id: str, graph_id: str):
        """Build the runtime graph from the bundle's validated clip metadata."""
        from .animation.graph import AnimationGraph

        bundle = self.bundles.get(bundle_id)
        if bundle is None:
            raise ValueError(f"Unknown NPC asset bundle '{bundle_id}'")
        return AnimationGraph.from_bundle(graph_id, bundle)

    def _validate_bundle(self, bundle: AssetBundle, errors: list[str]) -> None:
        if bundle.height_m <= 0:
            errors.append(f"Bundle '{bundle.bundle_id}' height_m must be positive")
        if bundle.coordinate_system != "mujoco_z_up" or bundle.unit != "meter":
            errors.append(f"Bundle '{bundle.bundle_id}' must use mujoco_z_up coordinates in meters")
        if bundle.asset_quality not in {"preview", "production", "restricted"}:
            errors.append(f"Bundle '{bundle.bundle_id}' has unknown asset_quality")
        if not bundle.material_slots or len(set(bundle.material_slots)) != len(
            bundle.material_slots
        ):
            errors.append(
                f"Bundle '{bundle.bundle_id}' material_slots must be unique and non-empty"
            )
        if "idle" not in bundle.clips:
            errors.append(f"Bundle '{bundle.bundle_id}' is missing mandatory clip 'idle'")
        if bundle.asset_quality in {"production", "restricted"}:
            from .animation.graph import OFFICE_CLIPS

            missing_clips = set(OFFICE_CLIPS) - bundle.clips.keys()
            if missing_clips:
                errors.append(
                    f"Bundle '{bundle.bundle_id}' restricted production clips are incomplete: "
                    f"{', '.join(sorted(missing_clips))}"
                )
        sit_clip = bundle.clips.get("sit") or bundle.clips.get("sit_down")
        stand_up_clip = bundle.clips.get("stand_up")
        if sit_clip is not None and stand_up_clip is not None:
            if stand_up_clip.frames != tuple(reversed(sit_clip.frames)):
                errors.append(
                    f"Bundle '{bundle.bundle_id}' stand_up frames must be the reverse of "
                    "its registered sit sequence"
                )
        for hash_path, digest in bundle.sha256.items():
            if len(digest) != 64 or any(
                character not in "0123456789abcdefABCDEF" for character in digest
            ):
                errors.append(f"Bundle '{bundle.bundle_id}' has invalid SHA-256 for '{hash_path}'")

        referenced_paths = [frame for clip in bundle.clips.values() for frame in clip.frames]
        contains_cesium = "cesium" in bundle.bundle_id.lower() or any(
            "cesium" in path.lower() for path in referenced_paths
        )
        if contains_cesium and bundle.asset_quality != "preview":
            errors.append(
                f"Bundle '{bundle.bundle_id}' contains CesiumMan assets but is not marked preview"
            )
        if "smplx" in bundle.bundle_id.lower() and any(
            "cesium" in path.lower() for path in referenced_paths
        ):
            errors.append(f"Bundle '{bundle.bundle_id}' cannot label CesiumMan frames as SMPL-X")

        referenced: list[str] = []
        for appearance_id, appearance in bundle.appearances.items():
            unknown_slots = set(appearance.textures) - set(bundle.material_slots)
            if unknown_slots:
                errors.append(
                    f"Appearance '{appearance_id}' uses unknown material slots: "
                    f"{', '.join(sorted(unknown_slots))}"
                )
            if appearance.texture_topology_id != bundle.topology_id:
                errors.append(
                    f"Appearance '{appearance_id}' texture topology "
                    f"'{appearance.texture_topology_id}' does not match bundle topology "
                    f"'{bundle.topology_id}'"
                )
            referenced.extend(appearance.textures.values())
        for accessory in bundle.accessories.values():
            referenced.extend((accessory.mesh, accessory.anchors))

        topology: tuple[int, tuple[tuple[str, ...], ...]] | None = None
        for clip_id, clip in bundle.clips.items():
            if clip.fps <= 0:
                errors.append(f"Bundle '{bundle.bundle_id}' clip '{clip_id}' fps must be positive")
            if not clip.frames:
                errors.append(f"Bundle '{bundle.bundle_id}' clip '{clip_id}' has no frames")
            if clip.root_motion not in {"in_place", "authored"}:
                errors.append(
                    f"Bundle '{bundle.bundle_id}' clip '{clip_id}' has invalid root_motion"
                )
            marker_names: set[str] = set()
            for marker in clip.markers:
                name = marker.get("name")
                phase = marker.get("phase")
                if not isinstance(name, str) or not name or name in marker_names:
                    errors.append(
                        f"Bundle '{bundle.bundle_id}' clip '{clip_id}' has invalid marker name"
                    )
                else:
                    marker_names.add(name)
                if not isinstance(phase, (int, float)) or not 0 <= float(phase) <= 1:
                    errors.append(
                        f"Bundle '{bundle.bundle_id}' clip '{clip_id}' has invalid marker phase"
                    )
            referenced.extend(clip.frames)
            for frame in clip.frames:
                frame_path = self.source_path.parent / frame
                if not frame_path.exists():
                    continue
                current = _obj_topology(frame_path, errors)
                if current is not None and topology is None:
                    topology = current
                elif current is not None and current != topology:
                    errors.append(
                        f"Bundle '{bundle.bundle_id}' frame '{frame}' has mismatched OBJ topology"
                    )

        for relative_path in sorted(set(referenced)):
            path = self.source_path.parent / relative_path
            if not path.is_file():
                errors.append(f"Bundle '{bundle.bundle_id}' asset is missing: {relative_path}")
                continue
            expected_hash = bundle.sha256.get(relative_path)
            if expected_hash is None:
                errors.append(f"Bundle '{bundle.bundle_id}' has no SHA-256 for '{relative_path}'")
            else:
                actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
                if actual_hash != expected_hash.lower():
                    errors.append(
                        f"Bundle '{bundle.bundle_id}' SHA-256 mismatch for '{relative_path}'"
                    )
            if path.suffix.lower() == ".png":
                image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
                if image is None or image.ndim not in {2, 3}:
                    errors.append(
                        f"Bundle '{bundle.bundle_id}' PNG cannot be decoded: {relative_path}"
                    )
                elif image.ndim == 3 and image.shape[2] not in {3, 4}:
                    errors.append(
                        f"Bundle '{bundle.bundle_id}' PNG has unsupported channels: {relative_path}"
                    )


def _obj_topology(path: Path, errors: list[str]) -> tuple[int, tuple[tuple[str, ...], ...]] | None:
    vertex_count = 0
    texture_count = 0
    faces: list[tuple[str, ...]] = []
    try:
        with path.open(encoding="utf-8", errors="strict") as file:
            for line in file:
                if line.startswith("v "):
                    vertex_count += 1
                elif line.startswith("vt "):
                    texture_count += 1
                elif line.startswith("f "):
                    # Preserve vertex/UV indices while ignoring normals.
                    faces.append(
                        tuple("/".join(token.split("/")[:2]) for token in line.split()[1:])
                    )
    except (OSError, UnicodeDecodeError) as error:
        errors.append(f"Could not read OBJ '{path}': {error}")
        return None
    if vertex_count == 0 or not faces:
        errors.append(f"OBJ '{path}' has no vertices or faces")
        return None
    if texture_count == 0 or any("/" not in token for face in faces for token in face):
        errors.append(f"OBJ '{path}' has no complete UV topology")
    return vertex_count, tuple(faces)


def main() -> None:
    """Console entry point for population asset preflight."""
    import argparse

    from .schema import NpcPopulation

    parser = argparse.ArgumentParser()
    parser.add_argument("--population", required=True)
    args = parser.parse_args()
    population = NpcPopulation.from_json(args.population)
    manifest = NpcAssetManifest.from_json(population.resolve_path(population.asset_manifest))
    manifest.validate_population(population)
    print(f"Validated {len(population.npcs)} NPC(s) and {len(manifest.bundles)} bundle(s)")
