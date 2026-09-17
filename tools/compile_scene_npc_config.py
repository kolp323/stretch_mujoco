"""Compile or config-only audit scene_npc_config/v1 resources."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from stretch_mujoco.npc.scene_compiler import compile_scene_npc_config, config_only_inventory


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--config-only", action="store_true")
    args = parser.parse_args()
    if args.config_only:
        report = config_only_inventory(args.config)
        print(json.dumps(report, indent=2, sort_keys=True))
        if report["status"] != "closed":
            raise SystemExit(2)
        return
    if args.output is None:
        parser.error("--output is required unless --config-only is used")
    result = compile_scene_npc_config(args.config, args.output)
    print(json.dumps({key: str(value) for key, value in result.items()}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
