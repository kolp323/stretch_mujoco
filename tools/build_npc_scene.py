"""Build a validated NPC MJCF wrapper."""

from __future__ import annotations

import argparse

from stretch_mujoco.npc.scene_builder import build_npc_scene


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--population", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(build_npc_scene(args.population, args.output))


if __name__ == "__main__":
    main()
