"""Calibrate authored seat ingress sites against compiled MuJoCo navigation geometry.

This tool is candidate-only: it changes only interaction_site_plan/v1 ingress
coordinates and never publishes an active catalog.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile

import mujoco
import numpy as np

from stretch_mujoco.humanoid.navigation import NavigationPathError, OfficeNavigationMesh
from stretch_mujoco.npc.scene_compiler import compile_scene_npc_config
from stretch_mujoco.npc.scene_config import load_scene_npc_config

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "stretch_mujoco/models"


def calibrate(config_path: Path, *, write: bool) -> dict[str, object]:
    config = load_scene_npc_config(config_path)
    if config.interaction_plan is None:
        raise ValueError(f"interaction_plan_missing:{config.scene_id}")
    payload = json.loads(config.interaction_plan.path.read_text(encoding="utf-8"))
    with tempfile.TemporaryDirectory(prefix=f"seat-calibration-{config.scene_id}-") as raw:
        outputs = compile_scene_npc_config(config_path, Path(raw) / "scene.xml", check_navigation=False)
        model = mujoco.MjModel.from_xml_path(str(outputs["scene"]))
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        mesh = OfficeNavigationMesh.from_model(
            model, data,
            resolution=float(config.navigation["resolution"]),
            agent_radius=float(config.navigation["agent_radius"]) + float(config.navigation["clearance"]),
            floor_geom_name=str(config.navigation["surface"]),
            exclude_body_roots=("base_link",),
        )
        changes: list[dict[str, object]] = []
        failures: list[dict[str, object]] = []
        for slot in payload["seat_slots"]:
            original = np.asarray(slot["ingress"]["position"][:2], dtype=float)
            sit = np.asarray(slot["sit"]["position"][:2], dtype=float)
            try:
                projected = mesh.nearest_free_world(
                    original, toward=sit, max_distance=1.25, component_id=mesh.primary_component_id
                )
                mesh.plan(projected, sit if mesh.is_world_free(sit) else projected)
            except NavigationPathError as error:
                failures.append({"seat_slot": slot["id"], "error": str(error)})
                continue
            movement = float(np.linalg.norm(projected - original))
            if movement > 1e-9:
                slot["ingress"]["position"][0] = round(float(projected[0]), 6)
                slot["ingress"]["position"][1] = round(float(projected[1]), 6)
                changes.append({
                    "seat_slot": slot["id"],
                    "from": [round(float(value), 6) for value in original],
                    "to": [round(float(value), 6) for value in projected],
                    "movement_m": round(movement, 6),
                })
    if failures:
        raise ValueError(
            f"seat_ingress_calibration_failed:{config.scene_id}:{failures[0]['seat_slot']}:{failures[0]['error']}"
        )
    if write and changes:
        config.interaction_plan.path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"scene_id": config.scene_id, "seat_slots": len(payload["seat_slots"]), "changed": len(changes), "changes": changes}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if not args.all:
        parser.error("--all is required")
    catalog = json.loads((MODELS / "scene_npc_configs/catalog.json").read_text(encoding="utf-8"))
    reports = [calibrate(MODELS / "scene_npc_configs" / item, write=args.write) for item in catalog["configs"]]
    result = {
        "schema": "seat_ingress_calibration/v1",
        "write": args.write,
        "scene_count": len(reports),
        "seat_slot_count": sum(int(item["seat_slots"]) for item in reports),
        "changed_slot_count": sum(int(item["changed"]) for item in reports),
        "scenes": reports,
    }
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(encoded, encoding="utf-8")
    print(encoded)


if __name__ == "__main__":
    main()
