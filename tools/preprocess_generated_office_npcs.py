"""Generate NPC-ready derivatives for every generated Office scene.

The source Office catalog remains immutable.  This tool copies each scene next
to its source, adds only deterministic navigation sites, and writes the
population, semantic graph, trajectory contract, and provenance separately
under ``models/generated_office_npc``.  It is intentionally idempotent.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from stretch_mujoco.humanoid.navigation import NavigationPathError, OfficeNavigationMesh


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS = PROJECT_ROOT / "stretch_mujoco" / "models"
CATALOG = MODELS / "assets" / "office_scenes"
DEFAULT_OUTPUT = MODELS / "generated_office_npc"
PRODUCTION_POPULATION = MODELS / "office_population.production.example.json"
NPC_IDS = ("npc_alex_chen", "npc_morgan_lee", "npc_jordan_patell")
SITE_Z = "0.025"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _scene_paths() -> tuple[Path, ...]:
    return tuple(
        path
        for path in sorted(CATALOG.glob("office_[0-9][0-9]_*.xml"))
        if "_robot" not in path.stem and not path.stem.endswith("_npc")
    )


def _zone_bounds(manifest: dict[str, Any]) -> dict[str, tuple[float, float, float, float]]:
    return {
        item["type"]: tuple(float(value) for value in item["bounds"]) for item in manifest["zones"]
    }


def _free_candidates(
    navigation: OfficeNavigationMesh,
    bounds: tuple[float, float, float, float],
    target: tuple[float, float],
) -> list[tuple[float, float]]:
    """Return collision-free grid positions, closest to the requested target."""
    x0, x1, y0, y1 = bounds
    # Keep the capsule proxy and its navigation inflation away from zone edges.
    margin = 0.45
    candidates: list[tuple[float, float]] = []
    for x in np.arange(x0 + margin, x1 - margin + 1e-9, 0.20):
        for y in np.arange(y0 + margin, y1 - margin + 1e-9, 0.20):
            point = (round(float(x), 4), round(float(y), 4))
            if navigation.is_world_free(point):
                candidates.append(point)
    if not candidates:
        raise ValueError(f"No collision-free NPC site candidate in zone {bounds}")
    return sorted(candidates, key=lambda point: math.dist(point, target))


def _choose(
    candidates: list[tuple[float, float]],
    occupied: list[tuple[float, float]],
    *,
    separation: float = 0.6,
) -> tuple[float, float]:
    for point in candidates:
        if all(math.dist(point, other) >= separation for other in occupied):
            return point
    raise ValueError("No collision-free candidate satisfies NPC separation")


def _pair(
    candidates: list[tuple[float, float]],
    occupied: list[tuple[float, float]],
) -> tuple[tuple[float, float], tuple[float, float]]:
    for first in candidates:
        if any(math.dist(first, other) < 0.6 for other in occupied):
            continue
        for second in candidates:
            distance = math.dist(first, second)
            if 0.75 <= distance <= 1.10 and all(
                math.dist(second, other) >= 0.6 for other in occupied
            ):
                return first, second
    raise ValueError("No collision-free facing NPC interaction pair")


def _reachable(
    navigation: OfficeNavigationMesh,
    source: tuple[float, float],
    candidates: list[tuple[float, float]],
) -> tuple[float, float]:
    """Choose the nearest candidate in the same navigation component."""
    for candidate in candidates:
        try:
            navigation.plan(np.asarray(source), np.asarray(candidate))
        except NavigationPathError:
            continue
        return candidate
    raise ValueError("No candidate is reachable from the preceding NPC route anchor")


def _route_chain(
    navigation: OfficeNavigationMesh,
    candidates: dict[str, list[tuple[float, float]]],
) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float], tuple[float, float]]:
    """Find a work→meeting→lounge→snack chain in one nav component."""
    for work in candidates["work"]:
        try:
            meeting = _reachable(navigation, work, candidates["meeting"])
            lounge = _reachable(navigation, meeting, candidates["lounge"])
            snack = _reachable(navigation, lounge, candidates["snack"])
        except ValueError:
            continue
        return work, meeting, lounge, snack
    raise ValueError("Generated office zones do not share a traversable NPC route chain")


def _body_for_asset(model: mujoco.MjModel, data: mujoco.MjData, asset: dict[str, Any]) -> str:
    if asset.get("body"):
        return str(asset["body"])
    position = np.asarray(asset["position"][:2], dtype=float)
    candidates: list[tuple[float, str]] = []
    for body_id in range(1, model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        if not name or name.startswith(("asset_component", "collision_component")):
            continue
        candidates.append((float(np.linalg.norm(data.xpos[body_id][:2] - position)), name))
    distance, name = min(candidates)
    if distance > 0.02:
        raise ValueError(f"Could not bind asset at {position.tolist()} to a model body")
    return name


def _asset_by_category(manifest: dict[str, Any], category: str) -> dict[str, Any]:
    for asset in manifest["assets"]:
        if asset.get("category") == category:
            return asset
    raise ValueError(f"Manifest has no {category} asset")


def _append_sites(
    source: Path, destination: Path, sites: dict[str, tuple[float, float, float]]
) -> None:
    tree = ET.parse(source)
    root = tree.getroot()
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError(f"Scene has no worldbody: {source}")
    existing = {node.get("name") for node in worldbody.findall("site")}
    for name, (x, y, yaw) in sites.items():
        if name in existing:
            raise ValueError(f"Generated site conflicts with source site: {name}")
        ET.SubElement(
            worldbody,
            "site",
            name=name,
            pos=f"{x:.4f} {y:.4f} {SITE_Z}",
            euler=f"0 0 {yaw:.9g}",
            size="0.015",
            rgba="0 0 0 0",
        )
    root.insert(0, ET.Comment(" Generated by preprocess_generated_office_npcs.py; do not edit. "))
    destination.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(tree, space="  ")
    tree.write(destination, encoding="unicode", xml_declaration=False)


def _semantic_world(
    scene_name: str,
    bodies: dict[str, str],
    graspable: dict[str, Any],
    sites: dict[str, tuple[float, float, float]],
) -> dict[str, Any]:
    objects = {
        "stretch_3": {
            "type": "StretchRobot",
            "binding": {"kind": "body", "name": "base_link"},
            "attributes": {"mobile": True, "manipulator": True},
        },
        "workstation": {
            "type": "Workstation",
            "binding": {"kind": "body", "name": bodies["workstation"]},
            "attributes": {"zone": "work"},
        },
        "meeting_table": {
            "type": "MeetingTable",
            "binding": {"kind": "body", "name": bodies["meeting"]},
            "attributes": {"zone": "meeting"},
        },
        "lounge": {
            "type": "Counter",
            "binding": {"kind": "body", "name": bodies["lounge"]},
            "attributes": {"zone": "lounge"},
        },
        "snack_counter": {
            "type": "Counter",
            "binding": {"kind": "body", "name": bodies["snack"]},
            "attributes": {"zone": "snack"},
        },
        # ObjectType intentionally has no generic interactive-object member.
        # Document is the existing graspable-task carrier; preserve the actual
        # catalog identity so consumers do not mistake a calculator or drink
        # for a literal document.
        "shared_object": {
            "type": "Document",
            "binding": {"kind": "body", "name": graspable["body"]},
            "attributes": {
                "location": "snack_counter",
                "owner": NPC_IDS[0],
                "graspable": True,
                "grasp_site": graspable["grasp_site"],
                "source_asset_id": graspable["asset_id"],
                "source_category": graspable["category"],
                "source_name": graspable["name"],
            },
        },
    }
    for npc_id in NPC_IDS:
        objects[npc_id] = {
            "type": "Employee",
            "binding": {"kind": "body", "name": f"npc__{npc_id}"},
            "attributes": {},
        }
    points = {
        "work": {
            "role": "desk_work_site",
            "owner": "workstation",
            "site": "npc_work_site",
            "attributes": {"yaw": sites["npc_work_site"][2]},
        },
        "meeting": {
            "role": "meeting_place_site",
            "owner": "meeting_table",
            "site": "npc_meeting_site",
            "attributes": {"yaw": sites["npc_meeting_site"][2]},
        },
        "lounge": {"role": "human_stand_site", "owner": "lounge", "site": "npc_lounge_site"},
        "snack": {"role": "human_stand_site", "owner": "snack_counter", "site": "npc_snack_site"},
        "object_grasp": {
            "role": "document_grasp_site",
            "owner": "shared_object",
            "site": graspable["grasp_site"],
        },
        "conversation_speaker": {
            "role": "conversation_site",
            "owner": "meeting_table",
            "site": "npc_conversation_speaker_site",
            "attributes": {"yaw": sites["npc_conversation_speaker_site"][2]},
        },
        "conversation_listener": {
            "role": "conversation_site",
            "owner": "meeting_table",
            "site": "npc_conversation_listener_site",
            "attributes": {"yaw": sites["npc_conversation_listener_site"][2]},
        },
        "handover_giver": {
            "role": "handover_site",
            "owner": NPC_IDS[0],
            "site": "npc_handover_giver_site",
            "attributes": {"yaw": sites["npc_handover_giver_site"][2]},
        },
        "handover_receiver": {
            "role": "handover_site",
            "owner": NPC_IDS[1],
            "site": "npc_handover_receiver_site",
            "attributes": {"yaw": sites["npc_handover_receiver_site"][2]},
        },
        "stretch_request": {
            "role": "robot_request_site",
            "owner": "stretch_3",
            "site": "npc_stretch_request_site",
        },
        "stretch_delivery": {
            "role": "robot_delivery_site",
            "owner": "workstation",
            "site": "npc_stretch_delivery_site",
        },
        "stretch_rendezvous": {
            "role": "human_stand_site",
            "owner": "stretch_3",
            "site": "npc_stretch_rendezvous_site",
        },
    }
    for index, npc_id in enumerate(NPC_IDS, 1):
        points[f"spawn_{index}"] = {
            "role": "human_stand_site",
            "owner": "workstation",
            "site": f"npc_spawn_{index:02d}_site",
        }
    return {
        "schema_version": 1,
        "scene": scene_name,
        "objects": objects,
        "relations": [
            {"subject": "shared_object", "relation": "ON", "object": "snack_counter"},
            {"subject": "shared_object", "relation": "BELONGS_TO", "object": NPC_IDS[0]},
            {"subject": "workstation", "relation": "NEAR", "object": "meeting_table"},
        ],
        "interaction_points": points,
    }


def _population(
    scene_name: str, output: Path, sites: dict[str, tuple[float, float, float]]
) -> dict[str, Any]:
    source = json.loads(PRODUCTION_POPULATION.read_text(encoding="utf-8"))
    population_dir = output / "populations"

    def relative_to_population(path: Path) -> str:
        return os.path.relpath(path.resolve(), population_dir.resolve())

    npcs = {npc_id: copy.deepcopy(source["npcs"][npc_id]) for npc_id in NPC_IDS}
    for index, npc_id in enumerate(NPC_IDS, 1):
        npcs[npc_id]["spawn"] = {
            "location": (
                "workstation" if index == 1 else ("lounge" if index == 2 else "snack_counter")
            ),
            "site": f"npc_spawn_{index:02d}_site",
            "yaw": sites[f"npc_spawn_{index:02d}_site"][2],
        }
    return {
        "schema_version": 2,
        "scene": relative_to_population(CATALOG / scene_name),
        "asset_manifest": relative_to_population(
            MODELS / "assets" / "humanoid" / "generated" / "animations" / "manifest.json"
        ),
        "appearance_catalog": relative_to_population(
            MODELS / "assets" / "humanoid" / "generated" / "animations" / "appearance_catalog.json"
        ),
        "trajectory_profile": relative_to_population(
            output / "trajectory_profiles" / f"{scene_name.removesuffix('.xml')}.json"
        ),
        "interaction_templates": {
            "conversation": {
                "speaker": {
                    "site": "npc_conversation_speaker_site",
                    "yaw": sites["npc_conversation_speaker_site"][2],
                },
                "listener": {
                    "site": "npc_conversation_listener_site",
                    "yaw": sites["npc_conversation_listener_site"][2],
                },
            },
            "handover": {
                "giver": {
                    "site": "npc_handover_giver_site",
                    "yaw": sites["npc_handover_giver_site"][2],
                },
                "receiver": {
                    "site": "npc_handover_receiver_site",
                    "yaw": sites["npc_handover_receiver_site"][2],
                },
            },
        },
        "clock": source["clock"],
        "npcs": npcs,
    }


def _trajectory(scene_name: str, scene_hash: str) -> dict[str, Any]:
    anchors = {
        "work": {"site": "npc_work_site", "role": "workstation"},
        "meeting": {"site": "npc_meeting_site", "role": "meeting"},
        "lounge": {"site": "npc_lounge_site", "role": "lounge"},
        "snack": {"site": "npc_snack_site", "role": "snack"},
        "stretch_rendezvous": {"site": "npc_stretch_rendezvous_site", "role": "stretch_rendezvous"},
    }
    return {
        "schema_version": 1,
        "profile_id": scene_name.removesuffix(".xml"),
        "scene": scene_name,
        "scene_sha256": scene_hash,
        "anchors": anchors,
        "routes": [
            {
                "route_id": "work_to_meeting",
                "from": "work",
                "to": "meeting",
                "actions": ["move_to", "attend_meeting"],
            },
            {
                "route_id": "meeting_to_lounge",
                "from": "meeting",
                "to": "lounge",
                "actions": ["move_to", "rest"],
            },
            {
                "route_id": "lounge_to_snack",
                "from": "lounge",
                "to": "snack",
                "actions": ["move_to", "pick_up"],
            },
            {
                "route_id": "snack_to_stretch",
                "from": "snack",
                "to": "stretch_rendezvous",
                "actions": ["move_to", "put_down"],
            },
        ],
    }


def _preprocess(scene: Path, output: Path) -> None:
    manifest_path = scene.with_suffix(".json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    navigation = OfficeNavigationMesh.from_model(model, data)
    zones = _zone_bounds(manifest)
    candidates = {
        name: _free_candidates(
            navigation, bounds, ((bounds[0] + bounds[1]) / 2, (bounds[2] + bounds[3]) / 2)
        )
        for name, bounds in zones.items()
    }
    work, meeting, lounge, snack = _route_chain(navigation, candidates)
    occupied: list[tuple[float, float]] = [work, meeting, lounge, snack]
    spawn_1 = _choose(candidates["work"], occupied, separation=1.0)
    occupied.append(spawn_1)
    spawn_2 = _choose(candidates["lounge"], occupied, separation=1.0)
    occupied.append(spawn_2)
    spawn_3 = _choose(candidates["snack"], occupied, separation=1.0)
    occupied.append(spawn_3)
    # Interaction templates are alternative activities, not simultaneous NPC
    # spawns.  They may share a clear meeting-zone footprint with each other
    # and with a navigation anchor, while each pair is independently separated.
    conversation = _pair(candidates["meeting"], [])
    handover = _pair(candidates["meeting"], [])
    request = _choose(candidates["work"], occupied, separation=0.6)
    delivery = _choose(candidates["work"], occupied + [request], separation=0.6)
    # The rendezvous remains a separately named station, while sharing the
    # reachable meeting anchor guarantees the final profile route is valid.
    rendezvous = meeting

    def facing(first: tuple[float, float], second: tuple[float, float]) -> tuple[float, float]:
        yaw = math.atan2(second[1] - first[1], second[0] - first[0])
        return yaw, math.atan2(first[1] - second[1], first[0] - second[0])

    conversation_yaws = facing(*conversation)
    handover_yaws = facing(*handover)
    sites = {
        "npc_work_site": (*work, 0.0),
        "npc_meeting_site": (*meeting, 0.0),
        "npc_lounge_site": (*lounge, 0.0),
        "npc_snack_site": (*snack, 0.0),
        "npc_spawn_01_site": (*spawn_1, 0.0),
        "npc_spawn_02_site": (*spawn_2, 0.0),
        "npc_spawn_03_site": (*spawn_3, 0.0),
        "npc_conversation_speaker_site": (*conversation[0], conversation_yaws[0]),
        "npc_conversation_listener_site": (*conversation[1], conversation_yaws[1]),
        "npc_handover_giver_site": (*handover[0], handover_yaws[0]),
        "npc_handover_receiver_site": (*handover[1], handover_yaws[1]),
        "npc_stretch_request_site": (*request, 0.0),
        "npc_stretch_delivery_site": (*delivery, 0.0),
        "npc_stretch_rendezvous_site": (*rendezvous, 0.0),
    }
    generated_scene = scene.with_name(f"{scene.stem}_npc.xml")
    _append_sites(scene, generated_scene, sites)
    generated_model = mujoco.MjModel.from_xml_path(str(generated_scene))
    generated_data = mujoco.MjData(generated_model)
    mujoco.mj_forward(generated_model, generated_data)
    workstation = _body_for_asset(
        generated_model, generated_data, _asset_by_category(manifest, "workstation_pods")
    )
    meeting_body = _body_for_asset(
        generated_model, generated_data, _asset_by_category(manifest, "meeting_tables")
    )
    lounge_body = _body_for_asset(
        generated_model, generated_data, _asset_by_category(manifest, "legacy_sofas")
    )
    snack_body = _body_for_asset(
        generated_model, generated_data, _asset_by_category(manifest, "furniture/snack_counter")
    )
    graspable = next(asset for asset in manifest["assets"] if asset.get("graspable"))
    semantic = _semantic_world(
        generated_scene.name,
        {
            "workstation": workstation,
            "meeting": meeting_body,
            "lounge": lounge_body,
            "snack": snack_body,
        },
        graspable,
        sites,
    )
    scene_id = scene.stem
    _json_write(output / "semantics" / f"{scene_id}.semantic.json", semantic)
    _json_write(
        output / "populations" / f"{scene_id}.population.json",
        _population(generated_scene.name, output, sites),
    )
    _json_write(
        output / "trajectory_profiles" / f"{scene_id}_npc.json",
        _trajectory(generated_scene.name, _sha256(generated_scene)),
    )
    _json_write(
        output / "receipts" / f"{scene_id}.json",
        {
            "schema_version": 1,
            "generator": "tools/preprocess_generated_office_npcs.py",
            "scene_id": scene_id,
            "sha256": {
                "source_scene": _sha256(scene),
                "manifest": _sha256(manifest_path),
                "generated_scene": _sha256(generated_scene),
                "generator": _sha256(Path(__file__)),
            },
            "sites": {name: [round(value, 4) for value in point] for name, point in sites.items()},
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    for scene in _scene_paths():
        _preprocess(scene, args.output.resolve())
    print(
        json.dumps({"generated_scenes": len(_scene_paths()), "output": str(args.output.resolve())})
    )


if __name__ == "__main__":
    main()
