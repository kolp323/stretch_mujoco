#!/usr/bin/env python3
"""Validate the pinned ten-NPC no-hair appearance and acceptance artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from stretch_mujoco.npc.schema import NpcPopulation


LOCK_SCHEMA_VERSION = 1
REQUIRED_ACCEPTANCE_SHOTS = (
    ("front", "walk"),
    ("rear", "walk"),
    ("left", "walk"),
    ("right", "walk"),
    ("top", "walk"),
    ("seated_front", "sit"),
    ("seated_rear", "sit"),
    ("seated_left", "sit"),
    ("seated_right", "sit"),
    ("seated_top", "sit"),
)


def validate_no_hair_npc_lock(
    lock_path: str | Path, workspace_root: str | Path
) -> dict[str, object]:
    """Reject any drift from a pinned no-hair appearance review baseline."""
    root = Path(workspace_root).resolve()
    lock_source = Path(lock_path).resolve()
    lock = _mapping(json.loads(lock_source.read_text(encoding="utf-8")), "No-hair lock")
    if lock.get("schema_version") != LOCK_SCHEMA_VERSION:
        raise ValueError("Unsupported no-hair lock schema_version")

    source_hashes = _mapping(lock.get("source_hashes"), "No-hair lock source_hashes")
    for relative, expected_hash in source_hashes.items():
        _validate_hash(root, str(relative), expected_hash, "No-hair lock source")

    build = _mapping(lock.get("build"), "No-hair lock build")
    build_tool = _string(build.get("tool"), "No-hair lock build tool")
    if not _resolve(root, build_tool).is_file() or build.get("include_base_scene") is not True:
        raise ValueError("No-hair lock must use the checked-in base-scene builder")
    runtime_configs = _string_list(
        build.get("accessory_runtime_configs"), "No-hair lock runtime configs"
    )
    for runtime_config in runtime_configs:
        if not _resolve(root, runtime_config).is_file():
            raise ValueError(f"No-hair lock runtime config is missing: {runtime_config}")

    population_relative = _string(lock.get("population"), "No-hair lock population")
    population_path = _resolve(root, population_relative)
    population = NpcPopulation.from_json(population_path)
    population_raw = _mapping(
        json.loads(population_path.read_text(encoding="utf-8")), "NPC population"
    )
    population_npcs = _mapping(population_raw.get("npcs"), "NPC population npcs")

    npcs = _mapping(lock.get("npcs"), "No-hair lock npcs")
    if set(npcs) != set(population.npcs):
        raise ValueError("No-hair lock NPC IDs do not match the production population")
    for npc_id, raw_expected in sorted(npcs.items()):
        expected = _mapping(raw_expected, f"No-hair lock NPC '{npc_id}'")
        actual = _mapping(population_npcs.get(npc_id), f"NPC population '{npc_id}'")
        embodiment = _mapping(actual.get("embodiment"), f"NPC '{npc_id}' embodiment")
        for field in ("appearance", "visual_identity", "appearance_config"):
            if embodiment.get(field) != expected.get(field):
                raise ValueError(f"No-hair lock NPC '{npc_id}' {field} does not match population")
        _validate_hash(
            root,
            _string(expected.get("body_path"), f"No-hair lock NPC '{npc_id}' body_path"),
            expected.get("body_sha256"),
            f"No-hair lock NPC '{npc_id}' body texture",
        )
        _validate_hash(
            root,
            _string(expected.get("video_path"), f"No-hair lock NPC '{npc_id}' video_path"),
            expected.get("video_sha256"),
            f"No-hair lock NPC '{npc_id}' acceptance video",
        )
        report_relative = _string(
            expected.get("report_path"), f"No-hair lock NPC '{npc_id}' report_path"
        )
        _validate_hash(
            root,
            report_relative,
            expected.get("report_sha256"),
            f"No-hair lock NPC '{npc_id}' acceptance report",
        )
        report = _mapping(
            json.loads(_resolve(root, report_relative).read_text(encoding="utf-8")),
            f"No-hair lock NPC '{npc_id}' acceptance report",
        )
        if report.get("npc_id") != npc_id or report.get("passed") is not True:
            raise ValueError(f"No-hair lock NPC '{npc_id}' acceptance report is not passing")
        reported_shots = tuple(
            (shot.get("name"), shot.get("clip"))
            for shot in _list_of_mappings(
                report.get("shots"), f"No-hair lock NPC '{npc_id}' acceptance shots"
            )
        )
        if reported_shots != REQUIRED_ACCEPTANCE_SHOTS:
            raise ValueError(
                f"No-hair lock NPC '{npc_id}' acceptance shots do not include the required "
                "walk and seated sit views"
            )

    scene = _mapping(lock.get("scene"), "No-hair lock scene")
    scene_relative = _string(scene.get("path"), "No-hair lock scene path")
    _validate_hash(root, scene_relative, scene.get("sha256"), "No-hair lock scene")
    scene_text = _resolve(root, scene_relative).read_text(encoding="utf-8")
    for excluded in _string_list(scene.get("excluded_accessory_ids"), "No-hair lock exclusions"):
        if excluded in scene_text:
            raise ValueError(f"No-hair scene unexpectedly includes '{excluded}'")
    for required in _string_list(scene.get("required_accessory_ids"), "No-hair lock requirements"):
        if required not in scene_text:
            raise ValueError(f"No-hair scene is missing required '{required}'")
    if tuple(build.get("excluded_runtime_accessory_ids", ())) != tuple(
        scene.get("excluded_accessory_ids", ())
    ):
        raise ValueError("No-hair lock build and scene exclusions do not match")

    return {"npc_count": len(npcs), "lock": str(lock_source), "passed": True}


def _validate_hash(root: Path, relative: str, expected_hash: object, context: str) -> None:
    if not isinstance(expected_hash, str) or len(expected_hash) != 64:
        raise ValueError(f"{context} sha256 must be a 64-character string")
    path = _resolve(root, relative)
    if not path.is_file():
        raise ValueError(f"{context} is missing: {path}")
    actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual_hash != expected_hash:
        raise ValueError(f"{context} sha256 does not match: {path}")


def _resolve(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if root != path and root not in path.parents:
        raise ValueError(f"No-hair lock path escapes workspace root: {relative}")
    return path


def _mapping(value: object, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be an object")
    return value


def _string(value: object, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{context} must be a non-empty string")
    return value


def _string_list(value: object, context: str) -> tuple[str, ...]:
    if (
        not isinstance(value, list)
        or not value
        or not all(isinstance(item, str) and item for item in value)
    ):
        raise ValueError(f"{context} must be a non-empty list of strings")
    return tuple(value)


def _list_of_mappings(value: object, context: str) -> tuple[Mapping[str, Any], ...]:
    if not isinstance(value, list) or not all(isinstance(item, Mapping) for item in value):
        raise ValueError(f"{context} must be a list of objects")
    return tuple(value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", required=True, type=Path)
    parser.add_argument("--workspace-root", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(validate_no_hair_npc_lock(args.lock, args.workspace_root), sort_keys=True))


if __name__ == "__main__":
    main()
