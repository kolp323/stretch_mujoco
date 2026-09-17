"""Propose scene-local interaction plans without publishing production support."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from stretch_mujoco.humanoid.navigation import OfficeNavigationMesh


NATIVE_HOME_COMPUTERS = {
    "home_03_107734119_175999932",
    "home_05_102344094",
    "home_06_104348082_171512994",
}
SEAT_CATEGORIES = {
    "meeting_chairs", "task_chairs", "lounge_seating", "chairs", "sofas",
    "benches", "stools", "ottomans", "seating_furniture",
}


def _write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _yaw_toward(source: np.ndarray, target: np.ndarray) -> float:
    return float(math.atan2(float(target[0] - source[0]), float(target[1] - source[1])))


def _asset_body(asset: dict) -> str | None:
    value = asset.get("body") or asset.get("object_id")
    return value if isinstance(value, str) and value else None


def _resolve_body(asset: dict, bodies: set[str]) -> str | None:
    body = _asset_body(asset)
    if body in bodies:
        return body
    instance = asset.get("instance")
    if isinstance(instance, int):
        matches = sorted(name for name in bodies if name.startswith(f"asset_{instance:03d}_"))
        if len(matches) == 1:
            return matches[0]
    return None


def _radius(asset: dict) -> float:
    value = asset.get("collision_radius")
    if isinstance(value, (int, float)) and not isinstance(value, bool) and value >= 0:
        return float(value)
    bounds = asset.get("bounds_mujoco")
    if isinstance(bounds, list) and len(bounds) == 2:
        return max(
            0.1,
            math.hypot(float(bounds[1][0]) - float(bounds[0][0]),
                       float(bounds[1][1]) - float(bounds[0][1])) / 2,
        )
    return 0.5


def _freejoint_bodies(xml_path: Path) -> set[str]:
    root = ET.parse(xml_path).getroot()
    return {
        str(body.get("name"))
        for body in root.findall(".//body")
        if body.get("name") and body.find("freejoint") is not None
    }


def _mesh(xml_path: Path, manifest: dict, kind: str) -> OfficeNavigationMesh:
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    robot = manifest.get("robot", {})
    robot_body = robot.get("body") if isinstance(robot, dict) else None
    return OfficeNavigationMesh.from_model(
        model,
        data,
        resolution=0.08 if kind == "office" else 0.06,
        agent_radius=0.22 if kind == "office" else 0.16,
        floor_geom_name="office_floor" if kind == "office" else "hssd_floor_collision",
        exclude_body_roots=(robot_body,) if isinstance(robot_body, str) and robot_body else (),
    )


def _primary_points(mesh: OfficeNavigationMesh) -> list[np.ndarray]:
    cells = np.argwhere(mesh.component_labels == mesh.primary_component_id)
    return [np.asarray(mesh.cell_to_world(tuple(cell)), dtype=float) for cell in cells]


def _region(manifest: dict, point: np.ndarray) -> str:
    entries = manifest.get("zones", []) or manifest.get("regions", [])
    for entry in entries:
        bounds = entry.get("bounds")
        if isinstance(bounds, list) and len(bounds) == 4 and (
            bounds[0] <= point[0] <= bounds[1] and bounds[2] <= point[1] <= bounds[3]
        ):
            name = str(entry.get("id") or entry.get("type"))
            return name if name.startswith(("zone.", "room.")) else f"zone.{name}"
    first = entries[0]
    name = str(first.get("id") or first.get("type"))
    return name if name.startswith(("zone.", "room.")) else f"zone.{name}"


def _station_pairs(
    points: list[np.ndarray], manifest: dict, count: int = 6
) -> list[tuple[np.ndarray, np.ndarray]]:
    pairs: list[tuple[np.ndarray, np.ndarray]] = []
    centers: list[np.ndarray] = []
    ordered = sorted(points, key=lambda p: (float(p[0]), float(p[1])))
    while len(pairs) < count:
        ranked = sorted(
            ordered,
            key=lambda p: (
                min((float(np.linalg.norm(p - c)) for c in centers), default=1e9),
                float(p[0]), float(p[1]),
            ),
            reverse=True,
        )
        found = None
        for first in ranked:
            if centers and min(float(np.linalg.norm(first - c)) for c in centers) < 1.5:
                continue
            candidates = [
                second for second in ordered
                if 0.72 <= float(np.linalg.norm(second - first)) <= 0.95
                and _region(manifest, second) == _region(manifest, first)
            ]
            if candidates:
                second = min(candidates, key=lambda p: (abs(float(np.linalg.norm(p-first))-0.82), float(p[0]), float(p[1])))
                found = (first, second)
                break
        if found is None:
            raise ValueError("scene_cannot_provide_six_distributed_station_pairs")
        pairs.append(found)
        centers.append((found[0] + found[1]) / 2)
    return pairs


def _nearest_ingress(points: list[np.ndarray], center: np.ndarray, clearance: float) -> np.ndarray | None:
    candidates = [
        point for point in points
        if clearance + 0.05 < float(np.linalg.norm(point - center)) <= 1.25
    ]
    return min(candidates, key=lambda p: (float(np.linalg.norm(p-center)), float(p[0]), float(p[1]))) if candidates else None


def ensure_home_primitive_kit(xml_path: Path, manifest_path: Path) -> bool:
    """Idempotently add a redistributable primitive workstation and handover cup."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    scene_id = str(manifest["scene_id"])
    if scene_id in NATIVE_HOME_COMPUTERS:
        return False
    if any(asset.get("provenance") == "primitive_workstation_kit/v1" for asset in manifest["assets"]):
        return False
    mesh = _mesh(xml_path, manifest, "home")
    point = max(_primary_points(mesh), key=lambda p: (float(p[0]), float(p[1])))
    x, y = round(float(point[0]), 6), round(float(point[1]), 6)
    root = ET.parse(xml_path)
    worldbody = root.getroot().find("worldbody")
    if worldbody is None:
        raise ValueError(f"missing_worldbody:{scene_id}")
    table = ET.SubElement(worldbody, "body", name="primitive_workstation_table", pos=f"{x} {y} 0")
    ET.SubElement(table, "geom", name="primitive_workstation_table_geom", type="box", size="0.6 0.35 0.36", pos="0 0 0.36", rgba="0.45 0.45 0.48 1")
    ET.SubElement(table, "site", name="primitive_workstation_work_site", pos="0 -0.58 0.73", size="0.02", rgba="0 0 0 0")
    computer = ET.SubElement(worldbody, "body", name="primitive_workstation_computer", pos=f"{x} {y} 0.78")
    ET.SubElement(computer, "geom", name="primitive_workstation_computer_geom", type="box", size="0.22 0.06 0.16", rgba="0.12 0.14 0.16 1")
    chair = ET.SubElement(worldbody, "body", name="primitive_workstation_chair", pos=f"{x} {y-0.9} 0")
    ET.SubElement(chair, "geom", name="primitive_workstation_chair_geom", type="box", size="0.28 0.28 0.22", pos="0 0 0.22", rgba="0.25 0.35 0.45 1")
    cup = ET.SubElement(worldbody, "body", name="primitive_handover_cup", pos=f"{x+0.35} {y} 0.82")
    ET.SubElement(cup, "freejoint")
    ET.SubElement(cup, "geom", name="primitive_handover_cup_geom", type="cylinder", size="0.04 0.07", mass="0.12", rgba="0.2 0.55 0.8 1")
    ET.SubElement(cup, "site", name="primitive_handover_cup_grasp_site", pos="0 0 0", size="0.015")
    ET.indent(root, space="  ")
    root.write(xml_path, encoding="unicode")
    assets = manifest["assets"]
    assets.extend([
        {"object_id":"primitive_workstation_table","body":"primitive_workstation_table","name":"Primitive workstation desk","category":"desks","position":[x,y,0.0],"collision_radius":0.7,"provenance":"primitive_workstation_kit/v1","license":"CC0-1.0"},
        {"object_id":"primitive_workstation_computer","body":"primitive_workstation_computer","name":"Primitive computer","category":"computers","position":[x,y,0.78],"collision_radius":0.0,"support":"primitive_workstation_table","provenance":"primitive_workstation_kit/v1","license":"CC0-1.0"},
        {"object_id":"primitive_workstation_chair","body":"primitive_workstation_chair","name":"Primitive workstation chair","category":"chairs","position":[x,y-0.9,0.0],"collision_radius":0.4,"provenance":"primitive_workstation_kit/v1","license":"CC0-1.0"},
        {"object_id":"primitive_handover_cup","body":"primitive_handover_cup","name":"Primitive handover cup","category":"interactive_objects","position":[x+0.35,y,0.82],"collision_radius":0.05,"dynamic":True,"graspable":True,"support":"primitive_workstation_table","grasp_site":"primitive_handover_cup_grasp_site","provenance":"primitive_workstation_kit/v1","license":"CC0-1.0"},
    ])
    _write(manifest_path, manifest)
    return True


def _normalize_native_workstation(manifest_path: Path) -> tuple[str, str]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assets = manifest["assets"]
    computers = [a for a in assets if any(token in str(a.get("semantic_name") or a.get("name") or "").lower() for token in ("computer", "imac", "laptop"))]
    if not computers:
        computers = [a for a in assets if a.get("category") == "computers"]
    computer = computers[0]
    computer["category"] = "computers"
    computer["body"] = _asset_body(computer)
    position = np.asarray(computer["position"][:2], dtype=float)
    candidates = [a for a in assets if any(token in str(a.get("semantic_name") or a.get("name") or "").lower() for token in ("desk", "table"))]
    workstation = min(candidates, key=lambda a: float(np.linalg.norm(np.asarray(a["position"][:2], dtype=float)-position)))
    workstation["category"] = "desks"
    workstation["body"] = _asset_body(workstation)
    workstation["collision_radius"] = _radius(workstation)
    computer["support"] = str(workstation["body"])
    _write(manifest_path, manifest)
    return str(workstation["body"]), str(computer["body"])


def propose_plan(xml_path: Path, manifest_path: Path, kind: str) -> dict:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    scene_id = str(manifest["scene_id"])
    xml_bodies = {
        str(body.get("name"))
        for body in ET.parse(xml_path).getroot().findall(".//body")
        if body.get("name")
    }
    if kind == "home":
        ensure_home_primitive_kit(xml_path, manifest_path)
        workstation_id, computer_id = _normalize_native_workstation(manifest_path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        xml_bodies = {
            str(body.get("name"))
            for body in ET.parse(xml_path).getroot().findall(".//body")
            if body.get("name")
        }
    else:
        assets = manifest["assets"]
        computer = next(a for a in assets if a.get("category") == "displays")
        bodies = xml_bodies
        def resolve(asset: dict) -> str:
            body = _asset_body(asset)
            if body in bodies:
                return str(body)
            prefix = f"asset_{int(asset['instance']):03d}_"
            matches = sorted(name for name in bodies if name.startswith(prefix))
            if len(matches) != 1:
                raise ValueError(f"asset_body_unbound:{scene_id}:{asset.get('instance')}")
            return matches[0]
        computer_id = resolve(computer)
        position = np.asarray(computer["position"][:2], dtype=float)
        workstation = min((a for a in assets if a.get("category") in {"workstation_pods","meeting_tables"}), key=lambda a: float(np.linalg.norm(np.asarray(a["position"][:2], dtype=float)-position)))
        workstation_id = resolve(workstation)
        computer["body"] = computer_id
        workstation["body"] = workstation_id
        computer["support"] = workstation_id
        _write(manifest_path, manifest)
    mesh = _mesh(xml_path, manifest, kind)
    points = _primary_points(mesh)
    pairs = _station_pairs(points, manifest)
    dynamic = [a for a in manifest["assets"] if a.get("dynamic") is True and a.get("graspable") is True and _resolve_body(a, xml_bodies) in _freejoint_bodies(xml_path)]
    if not dynamic:
        raise ValueError(f"no_freejoint_handover_object:{scene_id}")
    object_id = str(_resolve_body(dynamic[0], xml_bodies))

    conversations = []
    handovers = []
    for index, (first, second) in enumerate(pairs[:3], 1):
        conversations.append({"id":f"conversation.{scene_id}.{index:02d}","region":_region(manifest,(first+second)/2),"allowed_actor_pairs":["npc_npc"],"roles":{"speaker":{"position":[round(float(first[0]),6),round(float(first[1]),6),0.025],"yaw":_yaw_toward(first,second)},"listener":{"position":[round(float(second[0]),6),round(float(second[1]),6),0.025],"yaw":_yaw_toward(second,first)}},"distance_m":{"min":0.55,"max":1.05},"yaw_tolerance_rad":0.3})
    for index, (first, second) in enumerate(pairs[3:], 1):
        midpoint=(first+second)/2
        handovers.append({"id":f"handover.{scene_id}.{index:02d}","region":_region(manifest,midpoint),"modes":["npc_to_npc"],"roles":{"giver":{"position":[round(float(first[0]),6),round(float(first[1]),6),0.025],"yaw":_yaw_toward(first,second)},"receiver":{"position":[round(float(second[0]),6),round(float(second[1]),6),0.025],"yaw":_yaw_toward(second,first)},"robot":{"position":[round(float(first[0]),6),round(float(first[1]),6),0.025],"yaw":_yaw_toward(first,second)}},"transfer_position":[round(float(midpoint[0]),6),round(float(midpoint[1]),6),0.92],"object_ids":[object_id],"distance_m":{"min":0.55,"max":1.05},"yaw_tolerance_rad":0.3})

    slots=[]; exemptions=[]
    seat_assets=[a for a in manifest["assets"] if a.get("category") in SEAT_CATEGORIES and _resolve_body(a, xml_bodies)]
    for asset in sorted(seat_assets,key=lambda a:str(_resolve_body(a, xml_bodies))):
        owner=str(_resolve_body(asset, xml_bodies)); center=np.asarray(asset["position"][:2],dtype=float); clearance=min(0.4,max(0.2,_radius(asset)*0.45)); ingress=_nearest_ingress(points,center,clearance)
        if ingress is None:
            exemptions.append({"owner_entity":owner,"reason":"no_local_primary_component_ingress","evidence":"navmesh_search_within_1.25m"}); continue
        yaw=_yaw_toward(ingress,center); bounds=asset.get("bounds_mujoco"); z=0.45
        if isinstance(bounds,list) and len(bounds)==2: z=max(0.3,min(0.55,float(bounds[0][2])+(float(bounds[1][2])-float(bounds[0][2]))*0.45))
        slots.append({"id":f"seat.{owner}.01","owner_entity":owner,"seat_type":"sofa" if "sofa" in str(asset.get("semantic_name") or asset.get("name") or "").lower() else "chair","slot_index":1,"ingress":{"position":[round(float(ingress[0]),6),round(float(ingress[1]),6),0.025],"yaw":yaw},"sit":{"position":[round(float(center[0]),6),round(float(center[1]),6),round(z,6)],"yaw":yaw},"clearance_radius_m":round(clearance,6)})
    if not slots:
        raise ValueError(f"no_usable_seat_slot:{scene_id}")
    workstation_asset=next(a for a in manifest["assets"] if _asset_body(a)==workstation_id)
    workstation_xy=np.asarray(workstation_asset["position"][:2],dtype=float)
    slot=min(slots,key=lambda s:float(np.linalg.norm(np.asarray(s["sit"]["position"][:2])-workstation_xy)))
    workstations=[{"id":f"workstation.{scene_id}.01","workstation_entity":workstation_id,"computer_entity":computer_id,"seat_slot":slot["id"],"work_site":{"position":[*slot["sit"]["position"][:2],round(float(slot["sit"]["position"][2])+0.27,6)],"yaw":slot["sit"]["yaw"]}}]
    return {"schema":"interaction_site_plan/v1","scene_id":scene_id,"conversation_stations":conversations,"handover_stations":handovers,"seat_slots":slots,"workstations":workstations,"seat_exemptions":exemptions}


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mjcf",type=Path,required=True); parser.add_argument("--manifest",type=Path,required=True); parser.add_argument("--kind",choices=("office","home"),required=True); parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args(); _write(args.output,propose_plan(args.mjcf,args.manifest,args.kind))


if __name__ == "__main__": main()
