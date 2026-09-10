#!/usr/bin/env python3
"""Preflight a declarative NPC trajectory profile against an MJCF scene."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco

from stretch_mujoco.npc.trajectory_profile import NpcTrajectoryProfile, TrajectoryProfileError


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--npc-id", default="employee_01")
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    model = mujoco.MjModel.from_xml_path(str(args.scene))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    try:
        profile = NpcTrajectoryProfile.from_json(args.profile)
        profile.validate_scene(args.scene)
        routes = profile.preflight(model, data)
        clearance = profile.audit_npc_clearance(model, data, args.npc_id)
    except TrajectoryProfileError as error:
        parser.error(str(error))
    receipt = {
        "profile_id": profile.profile_id,
        "scene": str(args.scene),
        "npc_id": args.npc_id,
        "routes": [
            {
                "route_id": route.route_id,
                "waypoints": [list(point) for point in route.waypoints],
                "clearance": {
                    "sample_period_s": 0.01,
                    "sampled_poses": audit.sampled_poses,
                    "collision_free": True,
                },
            }
            for route, audit in zip(routes, clearance, strict=True)
        ],
    }
    if args.receipt is not None:
        args.receipt.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(f"Preflighted {len(routes)} NPC trajectory route(s) for '{profile.profile_id}'")


if __name__ == "__main__":
    main()
