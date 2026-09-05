"""Validate an NPC population and all referenced local assets."""

from __future__ import annotations

import argparse

from stretch_mujoco.npc.assets import NpcAssetManifest
from stretch_mujoco.npc.schema import NpcPopulation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--population", required=True)
    args = parser.parse_args()
    population = NpcPopulation.from_json(args.population)
    manifest = NpcAssetManifest.from_json(population.resolve_path(population.asset_manifest))
    manifest.validate_population(population)
    print(f"Validated {len(population.npcs)} NPC(s) and {len(manifest.bundles)} bundle(s)")


if __name__ == "__main__":
    main()
