"""Validate authored interaction plans and emit candidate-only evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from stretch_mujoco.npc.interaction_station import load_interaction_site_plan
from stretch_mujoco.npc.scene_config import load_scene_npc_config

try:
    from tools.propose_interaction_sites import _mesh
except ModuleNotFoundError:
    from propose_interaction_sites import _mesh


def validate_config(config_path: Path) -> dict:
    config = load_scene_npc_config(config_path)
    if config.interaction_plan is None:
        raise ValueError(f"interaction_plan_missing:{config.scene_id}")
    plan = load_interaction_site_plan(
        config.interaction_plan.path,
        source_manifest=config.source_manifest,
        source_mjcf=config.source_mjcf,
        expected_scene_id=config.scene_id,
    )
    manifest = json.loads(config.source_manifest.read_text(encoding="utf-8"))
    navigation = _mesh(config.source_mjcf, manifest, config.kind)
    targets: list[tuple[str, tuple[float, float, float]]] = []
    for station in plan.conversation_stations:
        targets.extend((f"{station.station_id}:{role}", pose.position) for role, pose in station.roles.items())
    for station in plan.handover_stations:
        targets.extend((f"{station.station_id}:{role}", pose.position) for role, pose in station.roles.items() if role in {"giver", "receiver"})
    targets.extend((f"{slot.slot_id}:ingress", slot.ingress.position) for slot in plan.seat_slots)
    unavailable = []
    for target_id, position in targets:
        cell = navigation.world_to_cell(np.asarray(position[:2], dtype=float))
        if navigation.component_labels[cell] != navigation.primary_component_id:
            unavailable.append(target_id)
    if unavailable:
        raise ValueError(f"npc_navigation_target_unavailable:{config.scene_id}:{','.join(unavailable)}")
    covered = {slot.owner_entity for slot in plan.seat_slots} | {
        exemption["owner_entity"] for exemption in plan.seat_exemptions
    }
    return {
        "scene_id": config.scene_id,
        "kind": config.kind,
        "plan_sha256": hashlib.sha256(config.interaction_plan.path.read_bytes()).hexdigest(),
        "counts": {
            "conversation": len(plan.conversation_stations),
            "handover": len(plan.handover_stations),
            "seat_slots": len(plan.seat_slots),
            "seat_exemptions": len(plan.seat_exemptions),
            "seat_entities_closed": len(covered),
            "workstations": len(plan.workstations),
        },
        "validators": {
            "schema": "passed",
            "source_bindings": "passed",
            "npc_navigation": "passed",
            "robot_navigation": "deferred_out_of_scope",
            "static_collision": "not_run",
            "action_acceptance": "phase_6",
        },
        "candidate_validated": True,
        "production_evidence": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("configs", nargs="+", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    records = [validate_config(path) for path in sorted(args.configs)]
    payload = {"schema_version": 1, "scene_count": len(records), "scenes": records}
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
