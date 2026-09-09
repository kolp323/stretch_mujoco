"""Build a validated NPC MJCF wrapper, optionally rebuilding fused accessories."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

from stretch_mujoco.npc.appearance_pipeline.accessory_recipe import FusedAccessoryRuntimeConfig
from stretch_mujoco.npc.scene_builder import build_npc_scene


def _tool_module(filename: str) -> ModuleType:
    path = Path(__file__).with_name(filename)
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def build_scene_from_accessory_runtime_config(
    config_path: str | Path, output_path: str | Path, *, include_base_scene: bool = False
) -> Path:
    """Materialize current recipe projections, then compose their runtime scene.

    There is intentionally no receipt or cache fast path: each invocation calls
    the builder, so an edited recipe is reflected in OBJ, anchors, fused frames,
    manifest and population before MuJoCo sees the scene.
    """
    config_source = Path(config_path).resolve()
    config = FusedAccessoryRuntimeConfig.from_json(config_source)
    builder = _tool_module("build_npc_fused_accessory.py")
    result = builder.build_fused_accessory(
        recipe_path=config.resolve_path(config_source, "recipe"),
        source_archive=config.resolve_path(config_source, "source_archive"),
        source_manifest=config.resolve_path(config_source, "source_manifest"),
        source_population=config.resolve_path(config_source, "source_population"),
        npc_id=config.npc_id,
        output_dir=config.resolve_path(config_source, "output_dir"),
        output_manifest=config.resolve_path(config_source, "output_manifest"),
        output_population=config.resolve_path(config_source, "output_population"),
        bundle_id=config.bundle,
    )
    return build_npc_scene(result["population"], output_path, include_base_scene=include_base_scene)


def main() -> None:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--population")
    source.add_argument("--accessory-runtime-config")
    parser.add_argument("--output", required=True)
    parser.add_argument("--include-base-scene", action="store_true")
    args = parser.parse_args()
    if args.accessory_runtime_config:
        print(
            build_scene_from_accessory_runtime_config(
                args.accessory_runtime_config,
                args.output,
                include_base_scene=args.include_base_scene,
            )
        )
        return
    print(build_npc_scene(args.population, args.output, include_base_scene=args.include_base_scene))


if __name__ == "__main__":
    main()
