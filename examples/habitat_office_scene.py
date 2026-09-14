"""Visualize an HSSD Habitat scene after runtime conversion to MuJoCo."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import click
import mujoco
import mujoco.viewer
import numpy as np

from stretch_mujoco.habitat_scene_gallery import (
    DEFAULT_CACHE_ROOT,
    DEFAULT_HSSD_ROOT,
    DEFAULT_SCENE_ID,
    load_habitat_scene_model,
    prepare_habitat_scene,
)


def _camera_presets(bounds: np.ndarray) -> dict[str, dict[str, Any]]:
    center = bounds.mean(axis=0)
    extent = bounds[1] - bounds[0]
    distance = max(float(extent[0]), float(extent[1])) * 1.2
    lookat = np.array((center[0], center[1], max(0.8, center[2])), dtype=float)
    return {
        "overview": {"lookat": lookat, "distance": distance, "azimuth": 135, "elevation": -32},
        "top": {"lookat": lookat, "distance": distance * 1.05, "azimuth": 90, "elevation": -89},
        "north": {"lookat": lookat, "distance": distance, "azimuth": 180, "elevation": -24},
        "east": {"lookat": lookat, "distance": distance, "azimuth": 90, "elevation": -24},
        "south": {"lookat": lookat, "distance": distance, "azimuth": 0, "elevation": -24},
        "west": {"lookat": lookat, "distance": distance, "azimuth": -90, "elevation": -24},
    }


def _apply_camera_preset(viewer: mujoco.viewer.Handle, preset: dict[str, Any]) -> None:
    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    viewer.cam.lookat[:] = preset["lookat"]
    viewer.cam.distance = preset["distance"]
    viewer.cam.azimuth = preset["azimuth"]
    viewer.cam.elevation = preset["elevation"]


@click.command()
@click.option("--scene-id", default=DEFAULT_SCENE_ID, show_default=True)
@click.option(
    "--hssd-root",
    type=click.Path(path_type=Path, exists=True, file_okay=False),
    default=DEFAULT_HSSD_ROOT,
    help="HSSD dataset root (or set STRETCH_MUJOCO_HSSD_ROOT).",
)
@click.option(
    "--cache-root", type=click.Path(path_type=Path), default=DEFAULT_CACHE_ROOT, show_default=True
)
@click.option(
    "--ktx-command",
    type=click.Path(path_type=Path),
    default=None,
    help="Path to the Khronos ktx executable used to decode BasisU textures.",
)
@click.option("--uncluttered", is_flag=True, help="Use scenes-uncluttered instead of scenes.")
@click.option("--objects-only", is_flag=True, help="Hide the stage shell and show only objects.")
@click.option(
    "--opaque-stage", is_flag=True, help="Render the stage opaque instead of translucent."
)
@click.option("--top-view", is_flag=True, help="Start with the top-down layout camera.")
@click.option(
    "--view",
    type=click.Choice(("overview", "top", "north", "east", "south", "west")),
    default="overview",
    show_default=True,
    help="Initial free-camera preset.",
)
@click.option("--auto-orbit", is_flag=True, help="Continuously orbit around the complete scene.")
@click.option(
    "--orbit-speed",
    type=click.FloatRange(min=1.0),
    default=12.0,
    show_default=True,
    help="Auto-orbit speed in degrees per second.",
)
@click.option("--rebuild", is_flag=True, help="Discard cached OBJ files and reconvert all GLBs.")
@click.option("--headless", is_flag=True, help="Build and compile without opening a window.")
def main(
    scene_id: str,
    hssd_root: Path | None,
    cache_root: Path,
    ktx_command: Path | None,
    uncluttered: bool,
    objects_only: bool,
    opaque_stage: bool,
    top_view: bool,
    view: str,
    auto_orbit: bool,
    orbit_speed: float,
    rebuild: bool,
    headless: bool,
) -> None:
    """Show scene 108294417_176709879 as a static MuJoCo visual scene."""
    scene = prepare_habitat_scene(
        scene_id=scene_id,
        hssd_root=hssd_root,
        cache_root=cache_root,
        uncluttered=uncluttered,
        stage_alpha=1.0 if opaque_stage else 0.32,
        include_stage=not objects_only,
        rebuild=rebuild,
        ktx_command=ktx_command,
    )
    model = load_habitat_scene_model(scene)
    data = mujoco.MjData(model)
    click.echo(f"Generated MJCF: {scene.xml_path}")
    click.echo(
        f"Objects: {scene.object_count}; unique templates: {scene.unique_template_count}; "
        f"meshes: {model.nmesh}; geoms: {model.ngeom}"
    )
    click.echo(f"Bounds: {scene.bounds.tolist()}")
    click.echo(f"Categories: {json.dumps(scene.category_counts, sort_keys=True)}")
    if headless:
        return

    initial_view = "top" if top_view else view
    presets = _camera_presets(scene.bounds)
    state: dict[str, Any] = {
        "pending_preset": initial_view,
        "free_camera": False,
        "auto_orbit": auto_orbit,
        "toggle_stage": False,
    }

    preset_keys = {
        ord("1"): "overview",
        ord("2"): "top",
        ord("3"): "north",
        ord("4"): "east",
        ord("5"): "south",
        ord("6"): "west",
    }

    def key_callback(keycode: int) -> None:
        if keycode in preset_keys:
            state["pending_preset"] = preset_keys[keycode]
            state["free_camera"] = False
        elif keycode == ord("F"):
            state["free_camera"] = True
            state["auto_orbit"] = False
            click.echo("Free camera enabled")
        elif keycode == ord("O"):
            state["auto_orbit"] = not state["auto_orbit"]
            state["free_camera"] = False
            click.echo(f"Auto orbit: {'on' if state['auto_orbit'] else 'off'}")
        elif keycode == ord("T"):
            state["toggle_stage"] = True

    click.echo("\nViewer controls:")
    click.echo("  Mouse drag: rotate/pan    Wheel: zoom")
    click.echo("  1 overview  2 top  3 north  4 east  5 south  6 west")
    click.echo("  O toggle auto-orbit      F free camera")
    click.echo("  T toggle stage opaque/translucent")
    stage_material_ids = [
        index
        for index in range(model.nmat)
        if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MATERIAL, index) or "").startswith(
            "habitat_stage_"
        )
    ]
    stage_is_opaque = opaque_stage
    with mujoco.viewer.launch_passive(
        model,
        data,
        key_callback=key_callback,
        show_left_ui=False,
        show_right_ui=False,
    ) as viewer:
        last_time = time.monotonic()
        while viewer.is_running():
            current_time = time.monotonic()
            elapsed = current_time - last_time
            last_time = current_time
            pending_preset = state.pop("pending_preset", None)
            if pending_preset is not None:
                _apply_camera_preset(viewer, presets[pending_preset])
                click.echo(f"Camera preset: {pending_preset}")
            elif state["free_camera"]:
                viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
                state["free_camera"] = False
            if state["auto_orbit"]:
                viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
                viewer.cam.azimuth = (viewer.cam.azimuth + orbit_speed * elapsed) % 360
            if state.pop("toggle_stage", False):
                stage_is_opaque = not stage_is_opaque
                for material_id in stage_material_ids:
                    model.mat_rgba[material_id, 3] = 1.0 if stage_is_opaque else 0.32
                click.echo(f"Stage: {'opaque' if stage_is_opaque else 'translucent'}")
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(0.01)


if __name__ == "__main__":
    main()
