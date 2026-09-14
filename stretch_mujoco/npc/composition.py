"""Declarative composition of a base MuJoCo scene and an NPC population.

The population JSON is the single source for the base scene, asset manifest,
and canonical NPC identities.  This module deliberately compiles a new MJCF;
MuJoCo models cannot accept new bodies after compilation.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path

import mujoco

from .assets import NpcAssetManifest
from .naming import body_name, interaction_site_name
from .scene_builder import build_npc_scene
from .schema import NpcPopulation
from .system import NpcSystem

_COMPOSITION_VERSION = "npc_scene_composition/v3"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True)
class NpcSceneComposition:
    """A compiled-scene receipt for a population-driven MJCF composition."""

    scene_path: Path
    receipt_path: Path
    population_path: Path
    base_scene_path: Path
    manifest_path: Path
    npc_ids: tuple[str, ...]
    body_names: tuple[str, ...]
    handover_sites: tuple[str, ...]
    spawn_sites: tuple[str, ...]
    reused: bool = False

    def metadata(self) -> dict[str, object]:
        return json.loads(self.receipt_path.read_text(encoding="utf-8"))


@dataclass(frozen=True)
class ComposedNpcRuntime:
    """Objects that must share one population when running an NPC scene."""

    composition: NpcSceneComposition
    model: mujoco.MjModel
    npc_system: NpcSystem
    semantic_world: object
    office_runtime: object


def _receipt_path(destination: Path, receipt_path: str | Path | None) -> Path:
    return (
        Path(receipt_path).resolve()
        if receipt_path is not None
        else destination.with_suffix(".composition.json")
    )


def _expected_metadata(
    population: NpcPopulation, population_path: Path, manifest_path: Path
) -> dict[str, object]:
    base_scene_path = population.resolve_path(population.scene)
    return {
        "schema_version": 1,
        "composition_version": _COMPOSITION_VERSION,
        "population": str(population_path),
        "base_scene": str(base_scene_path),
        "asset_manifest": str(manifest_path),
        "sha256": {
            "population": _sha256(population_path),
            "base_scene": _sha256(base_scene_path),
            "asset_manifest": _sha256(manifest_path),
        },
        "npc_ids": list(population.npcs),
        "body_names": [body_name(npc_id) for npc_id in population.npcs],
        "handover_sites": [interaction_site_name(npc_id, "handover") for npc_id in population.npcs],
        "spawn_sites": [definition.spawn.site for definition in population.npcs.values()],
        "interaction_template_sites": sorted(
            {
                station.site
                for template in (population.interaction_templates or {}).values()
                for station in template.roles.values()
            }
        ),
    }


def _model_id(model: mujoco.MjModel, object_type: mujoco.mjtObj, name: str, label: str) -> None:
    if mujoco.mj_name2id(model, object_type, name) < 0:
        raise ValueError(f"composed_scene_missing_{label}:{name}")


def _validate_model(model: mujoco.MjModel, metadata: dict[str, object]) -> None:
    for name in metadata["body_names"]:
        _model_id(model, mujoco.mjtObj.mjOBJ_BODY, str(name), "body")
    for name in metadata["handover_sites"]:
        _model_id(model, mujoco.mjtObj.mjOBJ_SITE, str(name), "handover_site")
    for name in metadata["spawn_sites"]:
        _model_id(model, mujoco.mjtObj.mjOBJ_SITE, str(name), "spawn_site")
    for name in metadata.get("interaction_template_sites", []):
        _model_id(model, mujoco.mjtObj.mjOBJ_SITE, str(name), "interaction_template_site")


def _portable_base_wrapper(base_scene_path: Path, destination: Path) -> Path:
    """Write only generated include wrappers; never rewrite the base scene.

    MuJoCo resolves a nested ``compiler assetdir`` against the final composed
    file rather than the included base file.  A copied Stretch include with an
    absolute assetdir keeps a combined scene loadable from ``outputs/`` or
    ``/tmp`` while retaining the original office XML byte-for-byte.
    """
    import xml.etree.ElementTree as ET

    wrapper = destination.with_name(f"{destination.stem}.base.xml")
    stretch_wrapper = destination.with_name(f"{destination.stem}.stretch.xml")
    destination.parent.mkdir(parents=True, exist_ok=True)
    tree = ET.parse(base_scene_path)
    for include in tree.getroot().findall("include"):
        source = (base_scene_path.parent / include.attrib["file"]).resolve()
        if source.name == "stretch.xml":
            stretch_tree = ET.parse(source)
            compiler = stretch_tree.getroot().find("compiler")
            if compiler is None:
                raise ValueError(f"base_scene_stretch_compiler_missing:{source}")
            assetdir = compiler.get("assetdir", "")
            compiler.set("assetdir", str((source.parent / assetdir).resolve()))
            ET.indent(stretch_tree, space="  ")
            stretch_tree.write(stretch_wrapper, encoding="unicode", xml_declaration=False)
            include.set("file", str(stretch_wrapper))
        else:
            include.set("file", str(source))
    ET.indent(tree, space="  ")
    tree.write(wrapper, encoding="unicode", xml_declaration=False)
    return wrapper


def compose_npc_scene(
    population_path: str | Path,
    output_path: str | Path,
    *,
    receipt_path: str | Path | None = None,
    reuse: bool = True,
) -> NpcSceneComposition:
    """Compose, compile, validate, and receipt a complete population scene.

    ``population.scene`` is authoritative; this API intentionally has no scene
    override, preventing a physical scene and semantic population from drifting.
    Existing output is reused only when its receipt pins all current source and
    generated-scene hashes and its MuJoCo model still validates.
    """
    source = Path(population_path).resolve()
    destination = Path(output_path).resolve()
    population = NpcPopulation.from_json(source)
    manifest_path = population.resolve_path(population.asset_manifest)
    manifest = NpcAssetManifest.from_json(manifest_path)
    manifest.validate_population(population)
    expected = _expected_metadata(population, source, manifest_path)
    receipt = _receipt_path(destination, receipt_path)

    if reuse and destination.is_file() and receipt.is_file():
        try:
            cached = json.loads(receipt.read_text(encoding="utf-8"))
            source_hashes = dict(cached.get("sha256", {}))
            source_hashes.pop("generated_scene", None)
            cached_sources = {key: cached.get(key) for key in expected if key != "sha256"}
            expected_sources = {key: expected.get(key) for key in expected if key != "sha256"}
            if (
                cached_sources == expected_sources
                and source_hashes == expected["sha256"]
                and cached.get("sha256", {}).get("generated_scene") == _sha256(destination)
            ):
                _validate_model(mujoco.MjModel.from_xml_path(str(destination)), expected)
                return NpcSceneComposition(
                    destination,
                    receipt,
                    source,
                    population.resolve_path(population.scene),
                    manifest_path,
                    tuple(population.npcs),
                    tuple(expected["body_names"]),
                    tuple(expected["handover_sites"]),
                    tuple(expected["spawn_sites"]),
                    reused=True,
                )
        except (OSError, ValueError, json.JSONDecodeError, mujoco.FatalError):
            # A stale/corrupt cache is rebuilt below; source validation remains strict.
            pass

    portable_base = _portable_base_wrapper(population.resolve_path(population.scene), destination)
    build_npc_scene(source, destination, include_base_scene=True, _base_scene_path=portable_base)
    model = mujoco.MjModel.from_xml_path(str(destination))
    _validate_model(model, expected)
    expected["sha256"]["generated_scene"] = _sha256(destination)
    receipt.parent.mkdir(parents=True, exist_ok=True)
    receipt.write_text(json.dumps(expected, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return NpcSceneComposition(
        destination,
        receipt,
        source,
        population.resolve_path(population.scene),
        manifest_path,
        tuple(population.npcs),
        tuple(expected["body_names"]),
        tuple(expected["handover_sites"]),
        tuple(expected["spawn_sites"]),
    )


def _bind_population_semantics(world: object, population: NpcPopulation) -> None:
    """Migrate base-scene preview bindings to the generated canonical contract."""
    from stretch_mujoco.semantics import (
        BindingKind,
        InteractionPoint,
        InteractionRole,
        SemanticBinding,
    )

    for npc_id in population.npcs:
        existing = world.objects.get(npc_id)
        if existing is not None:
            world.objects[npc_id] = replace(
                existing, binding=SemanticBinding(BindingKind.BODY, body_name(npc_id))
            )
        point_id = f"{npc_id}_handover"
        existing_point = world.interaction_points.get(point_id)
        if existing_point is not None:
            world.interaction_points[point_id] = InteractionPoint(
                point_id=point_id,
                role=InteractionRole.HANDOVER,
                owner=npc_id,
                site=interaction_site_name(npc_id, "handover"),
                attributes=existing_point.attributes,
            )


def load_composed_npc_runtime(
    population_path: str | Path,
    output_path: str | Path,
    *,
    receipt_path: str | Path | None = None,
    semantic_world_path: str | Path | None = None,
    simulation_seed: int = 0,
) -> ComposedNpcRuntime:
    """Load one composed scene into matching NPC, semantic, and agent runtimes."""
    composition = compose_npc_scene(population_path, output_path, receipt_path=receipt_path)
    population = NpcPopulation.from_json(composition.population_path)
    manifest = NpcAssetManifest.from_json(composition.manifest_path)
    model = mujoco.MjModel.from_xml_path(str(composition.scene_path))
    npc_system = NpcSystem.from_population(
        model,
        population,
        manifest,
        simulation_seed=simulation_seed,
        scene_path=composition.base_scene_path,
    )
    from stretch_mujoco.agents import OfficeAgentRuntime
    from stretch_mujoco.semantics import SemanticWorld

    world = (
        SemanticWorld.from_json(semantic_world_path)
        if semantic_world_path is not None
        else SemanticWorld.for_scene(composition.base_scene_path)
    )
    if world is None:
        raise ValueError(f"composed_scene_missing_semantics:{composition.base_scene_path}")
    _bind_population_semantics(world, population)
    runtime = OfficeAgentRuntime.from_json(world, composition.population_path, auto_plan=False)
    world.validate_model(model)
    return ComposedNpcRuntime(composition, model, npc_system, world, runtime)


def main() -> None:
    """Console entry point for population-driven combined MJCF generation."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--population", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--receipt")
    parser.add_argument("--no-reuse", action="store_true")
    args = parser.parse_args()
    result = compose_npc_scene(
        args.population, args.output, receipt_path=args.receipt, reuse=not args.no_reuse
    )
    print(
        json.dumps(
            {
                "scene": str(result.scene_path),
                "receipt": str(result.receipt_path),
                "reused": result.reused,
            }
        )
    )


if __name__ == "__main__":
    main()
