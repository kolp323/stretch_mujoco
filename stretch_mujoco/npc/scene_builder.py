"""Generate deterministic MJCF from a validated NPC population."""

from __future__ import annotations

import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco

from .assets import NpcAssetManifest
from .naming import (
    accessory_frame_geom_name,
    attachment_anchor_site_name,
    body_name,
    collision_geom_name,
    frame_geom_name,
    interaction_site_name,
)
from .schema import NpcPopulation


def build_npc_scene(
    population_path: str | Path,
    output_path: str | Path,
    *,
    include_base_scene: bool = False,
    _base_scene_path: str | Path | None = None,
) -> Path:
    """Build a standalone, includable MJCF containing canonical per-NPC bodies.

    ``include_base_scene`` is available for callers that write beside a portable
    base scene. The default artifact is independently compilable and can be
    included by the owning scene without copying its relative asset rules.
    """
    population = NpcPopulation.from_json(population_path)
    manifest_path = population.resolve_path(population.asset_manifest)
    manifest = NpcAssetManifest.from_json(manifest_path)
    manifest.validate_population(population)
    destination = Path(output_path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)

    # ``_base_scene_path`` is intentionally private: public callers must take
    # the scene from the population.  The composition module uses it only for
    # a generated portable wrapper of that same source scene.
    scene_path = (
        Path(_base_scene_path).resolve()
        if _base_scene_path is not None
        else population.resolve_path(population.scene)
    )
    source_model = mujoco.MjModel.from_xml_path(str(scene_path))
    source_data = mujoco.MjData(source_model)
    mujoco.mj_forward(source_model, source_data)

    root = ET.Element("mujoco", {"model": "generated_npc_population"})
    source_hash = hashlib.sha256(
        Path(population_path).read_bytes() + manifest_path.read_bytes()
    ).hexdigest()
    root.append(ET.Comment(f" source_hash={source_hash} "))
    if include_base_scene:
        ET.SubElement(root, "include", {"file": str(scene_path)})
    asset = ET.SubElement(root, "asset")
    worldbody = ET.SubElement(root, "worldbody")

    mesh_names: dict[tuple[str, float, str, int], str] = {}
    material_names: dict[tuple[str, str], str] = {}
    accessory_meshes: dict[tuple[str, str], str] = {}
    attachment_anchor_payloads: dict[tuple[str, str], dict[str, object]] = {}
    for npc_id, definition in population.npcs.items():
        bundle = manifest.bundles[definition.embodiment.bundle]
        appearance = bundle.appearances[definition.embodiment.appearance]
        material_key = (bundle.bundle_id, definition.embodiment.appearance)
        if material_key not in material_names:
            texture_path = appearance.textures.get(bundle.material_slots[0])
            material_name = f"npc_material__{bundle.bundle_id}__{definition.embodiment.appearance}"
            if texture_path is not None:
                texture_name = (
                    f"npc_texture__{bundle.bundle_id}__{definition.embodiment.appearance}"
                )
                ET.SubElement(
                    asset,
                    "texture",
                    {
                        "name": texture_name,
                        "type": "2d",
                        "file": str((manifest_path.parent / texture_path).resolve()),
                    },
                )
                ET.SubElement(
                    asset,
                    "material",
                    {"name": material_name, "texture": texture_name},
                )
            else:
                ET.SubElement(asset, "material", {"name": material_name, "rgba": "1 1 1 1"})
            material_names[material_key] = material_name

        site_id = mujoco.mj_name2id(source_model, mujoco.mjtObj.mjOBJ_SITE, definition.spawn.site)
        if site_id < 0:
            raise ValueError(
                f"NPC '{npc_id}' spawn site '{definition.spawn.site}' is missing from "
                f"'{scene_path}'"
            )
        position = source_data.site_xpos[site_id]
        root_position = (float(position[0]), float(position[1]), 0.0)
        body = ET.SubElement(
            worldbody,
            "body",
            {
                "name": body_name(npc_id),
                "mocap": "true",
                "pos": " ".join(f"{value:.9g}" for value in root_position),
                "euler": f"0 0 {definition.spawn.yaw:.9g}",
            },
        )
        first_geom = True
        accessory_anchors = {
            accessory_id: json.loads(
                (manifest_path.parent / bundle.accessories[accessory_id].anchors).read_text()
            )
            for accessory_id in definition.embodiment.accessories
        }
        attachment_anchors: dict[str, dict[str, object]] = {}
        for role, relative_path in bundle.attachment_anchors.items():
            key = (bundle.bundle_id, role)
            payload = attachment_anchor_payloads.get(key)
            if payload is None:
                payload = json.loads((manifest_path.parent / relative_path).read_text())
                if payload.get("coordinate_system") != bundle.coordinate_system:
                    raise ValueError(
                        f"Attachment anchors '{relative_path}' use an incompatible coordinate system"
                    )
                clips = payload.get("clips")
                if not isinstance(clips, dict):
                    raise ValueError(f"Attachment anchors '{relative_path}' must define clips")
                attachment_anchor_payloads[key] = payload
            attachment_anchors[role] = payload
        for clip_id, clip in bundle.clips.items():
            for frame_index, frame_path in enumerate(clip.frames):
                mesh_key = (bundle.bundle_id, definition.embodiment.scale, clip_id, frame_index)
                mesh_name = mesh_names.get(mesh_key)
                if mesh_name is None:
                    mesh_name = (
                        f"npc_mesh__{bundle.bundle_id}__scale__{definition.embodiment.scale:g}"
                        f"__clip__{clip_id}__frame__{frame_index:03d}"
                    )
                    ET.SubElement(
                        asset,
                        "mesh",
                        {
                            "name": mesh_name,
                            "file": str((manifest_path.parent / frame_path).resolve()),
                            "scale": " ".join([f"{definition.embodiment.scale:g}"] * 3),
                        },
                    )
                    mesh_names[mesh_key] = mesh_name
                for slot in bundle.material_slots:
                    ET.SubElement(
                        body,
                        "geom",
                        {
                            "name": frame_geom_name(npc_id, clip_id, frame_index, slot),
                            "type": "mesh",
                            "mesh": mesh_name,
                            "material": material_names[material_key],
                            "mass": "0",
                            "contype": "0",
                            "conaffinity": "0",
                            "group": "2",
                            "rgba": "1 1 1 1" if first_geom else "1 1 1 0",
                        },
                    )
                    first_geom = False
                for role, anchor_payload in attachment_anchors.items():
                    try:
                        anchor = anchor_payload["clips"][clip_id][frame_index]
                    except (KeyError, IndexError, TypeError) as error:
                        raise ValueError(
                            f"Attachment anchor '{role}' lacks {clip_id}/{frame_index}"
                        ) from error
                    if not isinstance(anchor, list) or len(anchor) != 3:
                        raise ValueError(
                            f"Attachment anchor '{role}' has invalid {clip_id}/{frame_index}"
                        )
                    ET.SubElement(
                        body,
                        "site",
                        {
                            "name": attachment_anchor_site_name(
                                npc_id, role, clip_id, frame_index
                            ),
                            "pos": " ".join(
                                f"{float(value) * definition.embodiment.scale:.9g}"
                                for value in anchor
                            ),
                            "size": "0.001",
                            "rgba": "0 0 0 0",
                        },
                    )
                for accessory_id in definition.embodiment.accessories:
                    definition_asset = bundle.accessories[accessory_id]
                    accessory_mesh_key = (bundle.bundle_id, accessory_id)
                    accessory_mesh_name = accessory_meshes.get(accessory_mesh_key)
                    if accessory_mesh_name is None:
                        accessory_mesh_name = f"npc_accessory__{bundle.bundle_id}__{accessory_id}"
                        ET.SubElement(
                            asset,
                            "mesh",
                            {
                                "name": accessory_mesh_name,
                                "file": str(
                                    (manifest_path.parent / definition_asset.mesh).resolve()
                                ),
                            },
                        )
                        accessory_meshes[accessory_mesh_key] = accessory_mesh_name
                    try:
                        anchor = accessory_anchors[accessory_id][clip_id][frame_index]
                        position = anchor["position"]
                    except (KeyError, IndexError, TypeError) as error:
                        raise ValueError(
                            f"Accessory '{accessory_id}' lacks anchor for {clip_id}/{frame_index}"
                        ) from error
                    ET.SubElement(
                        body,
                        "geom",
                        {
                            "name": accessory_frame_geom_name(
                                npc_id, clip_id, frame_index, accessory_id
                            ),
                            "type": "mesh",
                            "mesh": accessory_mesh_name,
                            "pos": " ".join(str(float(value)) for value in position),
                            "mass": "0",
                            "contype": "0",
                            "conaffinity": "0",
                            "group": "2",
                            "rgba": "0.04 0.04 0.05 1" if first_geom else "0.04 0.04 0.05 0",
                        },
                    )
                    first_geom = False
        ET.SubElement(
            body,
            "geom",
            {
                "name": collision_geom_name(npc_id, "torso"),
                "type": "capsule",
                "fromto": "0 0 0.72 0 0 1.38",
                "size": "0.16",
                "rgba": "0 0 0 0",
            },
        )
        handover_payload = attachment_anchors.get("handover")
        if handover_payload is None:
            hand_anchor = tuple(
                coordinate * definition.embodiment.scale
                for coordinate in (-0.25, 0.06, 0.90)
            )
        else:
            hand_anchor = tuple(
                float(value) * definition.embodiment.scale
                for value in handover_payload["clips"]["idle"][0]
            )
        ET.SubElement(
            body,
            "site",
            {
                "name": interaction_site_name(npc_id, "handover"),
                # MeshSequenceBackend replaces this local pose with the anchor
                # belonging to every visible animation frame.
                "pos": " ".join(f"{coordinate:.9g}" for coordinate in hand_anchor),
                "size": "0.025",
                "rgba": "0 0 0 0",
            },
        )

    ET.indent(root, space="  ")
    tree = ET.ElementTree(root)
    tree.write(destination, encoding="unicode", xml_declaration=False)
    with destination.open("a", encoding="utf-8") as file:
        file.write("\n")
    return destination


def main() -> None:
    """Console entry point for deterministic scene generation."""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--population", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--include-base-scene",
        action="store_true",
        help="Write a combined base-scene and NPC MJCF (legacy standalone output is the default).",
    )
    args = parser.parse_args()
    print(build_npc_scene(args.population, args.output, include_base_scene=args.include_base_scene))
