"""Open one of the generated simplified HSSD home scenes."""

from __future__ import annotations

import json
import time
from pathlib import Path

import click
import mujoco
import mujoco.viewer


SCENE_ROOT = Path(__file__).resolve().parents[1] / "stretch_mujoco" / "models" / "assets" / "home_scenes"


@click.command()
@click.option("--scene", type=click.IntRange(1, 10), default=1, show_default=True)
@click.option("--top-view", is_flag=True, help="Start with the top-down camera.")
@click.option("--auto-orbit", is_flag=True, help="Continuously orbit around the home.")
@click.option("--orbit-speed", type=float, default=12.0, show_default=True)
@click.option("--headless", is_flag=True, help="Compile the scene without opening a window.")
def main(scene: int, top_view: bool, auto_orbit: bool, orbit_speed: float, headless: bool) -> None:
    """Interactively inspect a generated home scene."""
    catalog = json.loads((SCENE_ROOT / "catalog.json").read_text(encoding="utf-8"))
    entries = catalog.get("scenes", [])
    if len(entries) != 10:
        raise click.ClickException(f"Expected 10 generated scenes in {SCENE_ROOT}, found {len(entries)}")
    path = SCENE_ROOT / entries[scene - 1]["mjcf"]
    model = mujoco.MjModel.from_xml_path(str(path))
    data = mujoco.MjData(model)
    home_key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if home_key >= 0:
        mujoco.mj_resetDataKeyframe(model, data, home_key)
    mujoco.mj_forward(model, data)
    click.echo(f"Loaded {path.name}: {model.ntex} textures, {model.nmesh} meshes, {model.ngeom} geoms")
    if headless:
        return

    state = {"preset": "top" if top_view else "overview", "auto_orbit": auto_orbit}

    def key_callback(keycode: int) -> None:
        if keycode == ord("1"):
            state["preset"] = "overview"
        elif keycode == ord("2"):
            state["preset"] = "top"
        elif keycode == ord("O"):
            state["auto_orbit"] = not state["auto_orbit"]
            click.echo(f"Auto orbit: {'on' if state['auto_orbit'] else 'off'}")

    click.echo("\nCamera controls:")
    click.echo("  Left mouse drag: rotate")
    click.echo("  Right mouse drag: pan")
    click.echo("  Mouse wheel: zoom")
    click.echo("  1: overview   2: top view   O: toggle auto-orbit")
    with mujoco.viewer.launch_passive(
        model,
        data,
        key_callback=key_callback,
        show_left_ui=False,
        show_right_ui=False,
    ) as viewer:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        # Stretch contains rangefinder sensors. MuJoCo draws their yellow rays
        # by default, which makes a normal scene look like a yellow light
        # explosion; office viewer follows the same suppression policy.
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_RANGEFINDER] = False
        last_time = time.monotonic()
        while viewer.is_running():
            current_time = time.monotonic()
            elapsed = current_time - last_time
            last_time = current_time
            preset = state.pop("preset", None)
            if preset is not None:
                viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
                viewer.cam.lookat[:] = model.stat.center
                viewer.cam.distance = model.stat.extent * (1.05 if preset == "top" else 1.25)
                viewer.cam.azimuth = 90 if preset == "top" else 135
                viewer.cam.elevation = -89 if preset == "top" else -32
            if state["auto_orbit"]:
                viewer.cam.azimuth = (viewer.cam.azimuth + orbit_speed * elapsed) % 360
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(0.01)


if __name__ == "__main__":
    main()
