#!/usr/bin/env python3
"""Render the interactive objects used by the generated office scenes.

The script uses the same importer as ``generate_office_scenes.py``. By default
it reads the generated office-scene manifests and renders every distinct
asset/yaw combination that actually occurs in those scenes. It writes PNG files
and a labelled contact sheet. Use ``--viewer`` to inspect one object
interactively.

Examples
--------
    python tools/visualize_interactive_assets.py
    python tools/visualize_interactive_assets.py --viewer --asset 017_calculator
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
import os
import re
import time
import xml.etree.ElementTree as ET
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import cv2
import mujoco
import numpy as np

from office_interactive_assets import (
    INTERACTIVE_ROTATION_CORRECTIONS,
    InteractiveAsset,
    corrected_interactive_bounds,
    interactive_euler,
    load_interactive_assets,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "stretch_mujoco" / "models" / "assets" / "interactive_previews"
OFFICE_SCENES = ROOT / "stretch_mujoco" / "models" / "assets" / "office_scenes"


@dataclass(frozen=True)
class PreviewRequest:
    asset: InteractiveAsset
    yaw: float
    source_scenes: tuple[str, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--asset", help="Asset id/name to render (substring match is allowed).")
    parser.add_argument("--viewer", action="store_true", help="Open an interactive viewer instead of writing PNGs.")
    return parser.parse_args()


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "asset"


PREVIEW_WIDTH = 512
PREVIEW_HEIGHT = 512
PREVIEW_COLUMNS = 4


def make_model(asset: InteractiveAsset, yaw: float) -> mujoco.MjModel:
    """Create a small floor-and-light scene containing one imported object."""
    root = ET.Element("mujoco", model=f"preview_{safe_name(asset.asset_id)}")
    ET.SubElement(root, "compiler", angle="radian", balanceinertia="true")
    ET.SubElement(root, "option", gravity="0 0 -9.81")
    visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "headlight", ambient="0.35 0.35 0.35", diffuse="0.7 0.7 0.7", specular="0.15 0.15 0.15")
    assets = ET.SubElement(root, "asset")
    for definition in asset.definitions:
        ET.SubElement(assets, definition.tag, **definition.attributes)

    world = ET.SubElement(root, "worldbody")
    ET.SubElement(world, "light", pos="1 -2 4", dir="-0.2 0.3 -1", diffuse="0.8 0.8 0.8", castshadow="true")
    ET.SubElement(world, "light", pos="-2 1 2", dir="0.4 -0.2 -1", diffuse="0.35 0.4 0.5", castshadow="false")
    ET.SubElement(world, "geom", name="floor", type="plane", size="2 2 0.01", rgba="0.13 0.16 0.19 1", friction="1 0.01 0.001")

    # Imported assets are usually normalised so their lowest point is z=0.
    # Keep a small, visible clearance because a mesh exactly coplanar with the
    # floor can look clipped by depth-buffer precision (and a few source meshes
    # contain tiny negative vertex coordinates).
    floor_clearance = 0.02
    corrected_bounds = corrected_interactive_bounds(asset, yaw)
    object_z = -float(corrected_bounds[0, 2]) + floor_clearance
    # A fixed body is intentional here: this tool previews appearance, so
    # gravity/contact dynamics should not move the object between frames.
    body = ET.SubElement(
        world,
        "body",
        name="object",
        pos=f"0 0 {object_z:.9g}",
        euler=f"0 0 {yaw:.9g}",
    )
    orientation = ET.SubElement(
        body,
        "body",
        name="object_orientation",
        euler=" ".join(f"{value:.9g}" for value in interactive_euler(asset.asset_id)),
    )
    visual_attrs = {
        "name": "object_visual",
        "type": "mesh",
        "mesh": asset.visual_mesh,
        "mass": str(asset.mass_kg),
        "contype": "0",
        "conaffinity": "0",
    }
    if asset.material:
        visual_attrs["material"] = asset.material
    ET.SubElement(orientation, "geom", **visual_attrs)
    # Keep the collision mesh available for visual parity/debugging, but make it
    # non-colliding so it does not obscure the visual mesh.
    ET.SubElement(orientation, "geom", name="object_collision", type="mesh", mesh=asset.collision_mesh, contype="0", conaffinity="0", rgba="0 0 0 0")

    extent = np.asarray(corrected_bounds[1] - corrected_bounds[0], dtype=float)
    radius = max(float(extent.max()), 0.1)
    center = np.asarray((0.0, 0.0, object_z + max(float(extent[2]) * 0.5, 0.05)))
    target = ET.SubElement(world, "body", name="camera_target", pos=f"0 0 {center[2]:.9g}")
    ET.SubElement(target, "site", name="target_site", pos="0 0 0", size="0.001", rgba="0 0 0 0")
    distance = radius * 2.8
    ET.SubElement(world, "camera", name="preview", mode="targetbody", target="camera_target", pos=f"{distance:.9g} {-distance:.9g} {distance * 0.75:.9g}", fovy="42")
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, PREVIEW_WIDTH)
    model.vis.global_.offheight = max(model.vis.global_.offheight, PREVIEW_HEIGHT)
    return model


def render(asset: InteractiveAsset, yaw: float) -> np.ndarray:
    model = make_model(asset, yaw)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    renderer = mujoco.Renderer(model, height=PREVIEW_HEIGHT, width=PREVIEW_WIDTH)
    renderer.update_scene(data, camera="preview")
    image = renderer.render()
    renderer.close()
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


def run_viewer(asset: InteractiveAsset, yaw: float) -> None:
    import mujoco.viewer

    model = make_model(asset, yaw)
    data = mujoco.MjData(model)
    print(
        f"Asset: {asset.asset_id} ({asset.name}), yaw={math.degrees(yaw):.0f}° "
        "— Esc or close the window to quit"
    )
    with mujoco.viewer.launch_passive(model, data, show_left_ui=False, show_right_ui=False) as viewer:
        corrected_bounds = corrected_interactive_bounds(asset, yaw)
        extent = np.asarray(corrected_bounds[1] - corrected_bounds[0], dtype=float)
        radius = max(float(extent.max()), 0.1)
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        viewer.cam.lookat[:] = (
            0.0,
            0.0,
            -float(corrected_bounds[0, 2])
            + 0.02
            + max(float(extent[2]) * 0.5, 0.05),
        )
        viewer.cam.distance = radius * 2.8
        viewer.cam.azimuth = 135
        viewer.cam.elevation = -25
        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(0.01)


def actual_placement_requests(assets: list[InteractiveAsset]) -> list[PreviewRequest]:
    """Return one request per distinct (asset, yaw) in generated scenes."""
    manifest_paths = sorted(OFFICE_SCENES.glob("office_*.json"))
    if not manifest_paths:
        raise FileNotFoundError(f"No generated scene manifests found in {OFFICE_SCENES}")
    asset_by_id = {asset.asset_id: asset for asset in assets}
    usages: dict[tuple[str, float], set[str]] = {}
    for manifest_path in manifest_paths:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        scene_id = str(manifest.get("scene_id", manifest_path.stem))
        for placement in manifest.get("assets", []):
            if placement.get("category") != "interactive_objects":
                continue
            asset_id = placement.get("asset_id")
            if asset_id not in asset_by_id:
                continue
            # Round only for use as a dictionary key: the original angle is
            # passed through to MuJoCo unchanged below.
            yaw = float(placement.get("yaw", 0.0))
            key = (asset_id, round(yaw, 8))
            usages.setdefault(key, set()).add(scene_id)
    if not usages:
        raise ValueError(
            "The selected manifests contain no interactive objects. Regenerate "
            "the office scenes with tools/generate_office_scenes.py first."
        )
    return [
        PreviewRequest(asset_by_id[asset_id], yaw, tuple(sorted(source_scenes)))
        for (asset_id, yaw), source_scenes in sorted(usages.items())
    ]


def main() -> None:
    args = parse_args()
    assets = list(load_interactive_assets())
    if args.asset:
        query = args.asset.lower()
        assets = [a for a in assets if query in a.asset_id.lower() or query in a.name.lower()]
        if not assets:
            available = ", ".join(a.asset_id for a in load_interactive_assets())
            raise SystemExit(f"No asset matched {args.asset!r}. Available: {available}")
    requests = actual_placement_requests(assets)
    if args.asset:
        selected_ids = {asset.asset_id for asset in assets}
        requests = [request for request in requests if request.asset.asset_id in selected_ids]
    if not requests:
        raise SystemExit("No selected asset occurs in the requested office-scene manifests.")
    if args.viewer:
        run_viewer(requests[0].asset, requests[0].yaw)
        return

    args.output.mkdir(parents=True, exist_ok=True)
    rendered: list[tuple[PreviewRequest, np.ndarray, str | None]] = []
    for index, request in enumerate(requests, 1):
        asset = request.asset
        correction = INTERACTIVE_ROTATION_CORRECTIONS.get(asset.asset_id, (0.0, 0.0, 0.0))
        print(
            f"[{index}/{len(requests)}] {asset.asset_id}, "
            f"scene_yaw={math.degrees(request.yaw):.0f}°, "
            f"correction_xyz={tuple(round(value, 4) for value in correction)}"
        )
        try:
            image = render(asset, request.yaw)
            yaw_label = f"yaw_{math.degrees(request.yaw):+.0f}".replace("+", "plus")
            filename = f"{safe_name(asset.asset_id)}__{yaw_label}.png"
            cv2.imwrite(str(args.output / filename), image)
            rendered.append((request, image, None))
        except Exception as exc:
            print(f"  failed: {exc}")
            rendered.append(
                (request, np.zeros((PREVIEW_HEIGHT, PREVIEW_WIDTH, 3), np.uint8), str(exc))
            )

    columns = PREVIEW_COLUMNS
    rows = (len(rendered) + columns - 1) // columns
    sheet = np.zeros(
        (rows * (PREVIEW_HEIGHT + 32), columns * PREVIEW_WIDTH, 3), dtype=np.uint8
    )
    sheet[:] = (25, 30, 35)
    manifest = []
    for index, (request, image, error) in enumerate(rendered):
        asset = request.asset
        row, col = divmod(index, columns)
        x, y = col * PREVIEW_WIDTH, row * (PREVIEW_HEIGHT + 32)
        sheet[y : y + PREVIEW_HEIGHT, x : x + PREVIEW_WIDTH] = image
        label = f"{asset.asset_id} @ {math.degrees(request.yaw):.0f} deg"
        cv2.putText(
            sheet,
            label,
            (x + 8, y + PREVIEW_HEIGHT + 23),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (235, 240, 245),
            1,
            cv2.LINE_AA,
        )
        yaw_label = f"yaw_{math.degrees(request.yaw):+.0f}".replace("+", "plus")
        manifest.append(
            {
                "asset_id": asset.asset_id,
                "name": asset.name,
                "yaw_rad": request.yaw,
                "yaw_deg": math.degrees(request.yaw),
                "source_scenes": list(request.source_scenes),
                "preview": f"{safe_name(asset.asset_id)}__{yaw_label}.png",
                "error": error,
            }
        )
    cv2.imwrite(str(args.output / "contact_sheet.png"), sheet)
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {sum(item[2] is None for item in rendered)}/{len(rendered)} previews to {args.output}")


if __name__ == "__main__":
    main()
