"""Author all twenty candidate interaction plans from each scene's own resources."""

from __future__ import annotations

import json
from pathlib import Path

try:
    from tools.propose_interaction_sites import _write, propose_plan
except ModuleNotFoundError:
    from propose_interaction_sites import _write, propose_plan


ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "stretch_mujoco" / "models"


def main() -> None:
    catalog = json.loads((MODELS / "scene_npc_configs/catalog.json").read_text(encoding="utf-8"))
    for relative in catalog["configs"]:
        config_path = MODELS / "scene_npc_configs" / relative
        config = json.loads(config_path.read_text(encoding="utf-8"))
        scene = config["scene"]
        kind = scene["kind"]
        xml_path = (config_path.parent / scene["source_mjcf"]).resolve()
        manifest_path = (config_path.parent / scene["source_manifest"]).resolve()
        output = MODELS / "scene_interaction_plans" / kind / f"{scene['id']}.json"
        _write(output, propose_plan(xml_path, manifest_path, kind))
        print(json.dumps({"scene_id": scene["id"], "plan": str(output)}, sort_keys=True))


if __name__ == "__main__":
    main()
