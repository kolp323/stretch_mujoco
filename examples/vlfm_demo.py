#!/usr/bin/env python3
r"""VLFM (Vision-Language Frontier-based Method) demo.

Stop-and-think exploration guided by CLIP / SigLIP / GPT-4o.
The robot pauses at each decision point, scores candidate frontier
directions against a natural-language instruction, then navigates with A\*.

Usage
-----
  # CLIP (local model — default)
  .venv/bin/python examples/vlfm_demo.py --headless --steps 15

  # SigLIP (stronger local model)
  .venv/bin/python examples/vlfm_demo.py --vlm siglip --headless --steps 15

  # OpenAI GPT-4o (cloud API)
  .venv/bin/python examples/vlfm_demo.py --vlm openai --api-key sk-xxx
"""

from __future__ import annotations

import io
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import click
import matplotlib.pyplot as plt
import mujoco
import numpy as np
from PIL import Image

from stretch_mujoco.habitat_scene_gallery import (
    DEFAULT_CACHE_ROOT,
    DEFAULT_SCENE_ID,
    load_habitat_scene_model,
    prepare_habitat_scene,
)
from stretch_mujoco.navigations import NavigationController
from stretch_mujoco.navigations.FBE import FBEState
from stretch_mujoco.navigations.VLFM import (
    VLFMPlanner,
    BLIP2ITMVLMClient,
    CLIPVLMClient,
    SigLIPVLMClient,
    OpenAIVLMClient,
    ImageCapture,
)


# ---------------------------------------------------------------------------
def _render_grid(ax, data, bounds, title, cmap="Greys", alpha=1.0):
    xmin, xmax, ymin, ymax = bounds
    image = ax.imshow(
        data,
        extent=[xmin, xmax, ymin, ymax],
        origin="lower",
        cmap=cmap,
        alpha=alpha,
        aspect="equal",
        interpolation="none",
    )
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_title(title, fontsize=10, fontweight="bold")
    return image


# ---------------------------------------------------------------------------
@click.command()
@click.option("--scene-id", default=DEFAULT_SCENE_ID, show_default=True)
@click.option(
    "--cache-root",
    type=click.Path(path_type=Path),
    default=DEFAULT_CACHE_ROOT,
    show_default=True,
)
@click.option("--resolution", default=0.05, show_default=True)
@click.option("--agent-radius", default=0.3, show_default=True)
@click.option("--max-range", default=5.0, show_default=True)
@click.option("--steps", default=30, show_default=True, help="Number of VLFM decision cycles.")
@click.option(
    "--vlm-query-interval",
    type=click.IntRange(min=1),
    default=5,
    show_default=True,
    help="Run the VLM once every N control cycles.",
)
@click.option(
    "--vlm",
    "vlm_type",
    type=click.Choice(["blip2", "clip", "siglip", "openai"]),
    default="clip",
    show_default=True,
)
@click.option(
    "--device", default="cpu", show_default=True, help="Torch device for local models (cpu / cuda)."
)
@click.option(
    "--blip2-url",
    default="http://127.0.0.1:12182",
    show_default=True,
    help="URL of the isolated BLIP2 ITM service.",
)
@click.option("--api-key", default="", help="OpenAI API key (for --vlm openai).")
@click.option("--model", default="gpt-4o", show_default=True, help="OpenAI model name.")
@click.option("--base-url", default="", help="Custom OpenAI-compatible base URL.")
@click.option(
    "--instruction",
    default="Seems like there is a chair ahead.",
    show_default=True,
    help="Natural-language exploration goal.",
)
@click.option("--headless", is_flag=True, help="Run without viewer (text + plot).")
@click.option("--save/--no-save", default=True, help="Save result plot.")
@click.option(
    "--allow-geometric-fallback",
    is_flag=True,
    help="Continue with geometric FBE selection when VLM inference fails.",
)
def main(
    scene_id,
    cache_root,
    resolution,
    agent_radius,
    max_range,
    steps,
    vlm_query_interval,
    vlm_type,
    device,
    blip2_url,
    api_key,
    model,
    base_url,
    instruction,
    headless,
    save,
    allow_geometric_fallback,
):
    """VLFM demo — VLM-guided frontier exploration."""
    # --- Scene ---
    scene = prepare_habitat_scene(
        scene_id=scene_id,
        cache_root=cache_root,
        stage_alpha=1.0,
    )
    mujoco_model = load_habitat_scene_model(scene)
    data = mujoco.MjData(mujoco_model)
    mujoco.mj_step(mujoco_model, data)

    mx = float(scene.bounds[0, 0] + 0.8)
    Mx = float(scene.bounds[1, 0] - 0.8)
    my = float(scene.bounds[0, 1] + 0.8)
    My = float(scene.bounds[1, 1] - 0.8)
    gbounds = (mx, Mx, my, My)

    nav = NavigationController(
        mujoco_model,
        data,
        bounds=(mx, Mx, my, My),
        resolution=resolution,
        agent_radius=agent_radius,
        minimum_obstacle_height=0.1,
        maximum_obstacle_height=3.0,
        require_collision=False,
        exclude_prefixes=("habitat_stage_",),
    )
    god_grid = nav.grid

    # --- Find reachable start ---
    rng = np.random.default_rng(42)
    free_pts = [
        (float(x), float(y))
        for x in np.linspace(mx + 0.5, Mx - 0.5, 10)
        for y in np.linspace(my + 0.5, My - 0.5, 10)
        if god_grid.is_world_free(np.array([x, y]))
    ]
    rng.shuffle(free_pts)

    start_pt = None
    for s in free_pts[:20]:
        for g in free_pts[1:]:
            if g == s:
                continue
            d = float(np.linalg.norm(np.array(s) - np.array(g)))
            if 3 < d < 15:
                try:
                    nav.plan(start=s, goal=g)
                    start_pt = s
                    break
                except Exception:
                    continue
        if start_pt is not None:
            break
    if start_pt is None:
        start_pt = free_pts[0]

    # --- VLM client ---
    if vlm_type == "blip2":
        vlm = BLIP2ITMVLMClient(base_url=blip2_url)
    elif vlm_type == "clip":
        vlm = CLIPVLMClient(device=device)
    elif vlm_type == "siglip":
        vlm = SigLIPVLMClient(device=device)
    elif vlm_type == "openai":
        if not api_key:
            api_key = click.prompt("OpenAI API key", hide_input=True)
        vlm = OpenAIVLMClient(model=model, api_key=api_key, base_url=base_url or None)
    else:
        raise ValueError(f"Unknown VLM type: {vlm_type}")

    try:
        vlm.validate_environment()
    except Exception as exc:
        raise click.ClickException(f"VLM environment validation failed: {exc}") from exc

    # Load local model weights before creating an EGL/OpenGL renderer. Some
    # native ML runtimes tear down an active GL context unsafely when model
    # loading fails, so fail early with a normal Python error instead.
    vlm_ready = True
    if steps > 0:
        try:
            vlm.prepare()
        except Exception as exc:
            if not allow_geometric_fallback:
                raise click.ClickException(f"VLM model preparation failed: {exc}") from exc
            vlm_ready = False
            print(f"[warn] VLM unavailable; geometric fallback enabled: {exc}")

    # --- Image capture (try; may fail in headless envs) ---
    capture: Optional[ImageCapture] = None
    if vlm_ready:
        try:
            capture = ImageCapture(mujoco_model, data, width=320, height=240)
        except Exception as exc:
            if not allow_geometric_fallback:
                raise click.ClickException(f"Image capture initialization failed: {exc}") from exc
            print(f"[warn] Image capture unavailable; geometric fallback enabled: {exc}")

    # --- VLFM planner ---
    vlfm = VLFMPlanner(
        god_grid,
        vlm,
        instruction=instruction,
        robot_xy=start_pt,
        robot_yaw=0.0,
        num_rays=120,
        max_range_m=max_range,
        fov_degrees=270,
        min_cluster_size=3,
        explore_threshold=0.85,
        allow_geometric_fallback=allow_geometric_fallback,
    )

    robot = np.array(start_pt, dtype=float)
    yaw = 0.0

    print(f"\n{'='*60}")
    print(f"  VLFM Exploration")
    print(f"{'='*60}")
    print(f"VLM: {vlm_type} ({type(vlm).__name__})")
    print(f'Instruction: "{instruction}"')
    print(f"Start: ({start_pt[0]:.1f}, {start_pt[1]:.1f})")
    print(
        f"Grid: {god_grid.occupancy.shape} | "
        f"God-free: {100*(~god_grid.occupancy).sum()/god_grid.occupancy.size:.0f}%"
    )
    print(f"Laser: {max_range}m × 270° | Steps: {steps}\n")

    try:
        _run_visual_demo(
            vlfm,
            capture,
            robot,
            yaw,
            god_grid,
            gbounds,
            start_pt,
            steps,
            save,
            display=not headless,
            vlm_query_interval=vlm_query_interval,
        )
    finally:
        if capture is not None:
            capture.close()


# ---------------------------------------------------------------------------
def _run_visual_demo(
    vlfm,
    capture,
    robot,
    yaw,
    god_grid,
    gbounds,
    start_pt,
    steps,
    save,
    *,
    display,
    vlm_query_interval,
):
    """Run VLFM with synchronized map, value-map, and RGB visualisation."""
    plt.ion()
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    ax_god, ax_local, ax_value, ax_camera = axes.flat
    fig.suptitle("VLFM Exploration", fontsize=13, fontweight="bold")

    god_img = np.where(god_grid.occupancy, 0.2, 0.9)
    local_img = np.full(vlfm.local_map.data.shape, 0.35)
    value_img = vlfm.value_map_image()
    camera_img = np.zeros((capture.height, capture.width, 3), dtype=np.uint8) if capture else np.zeros((2, 2, 3), dtype=np.uint8)
    _render_grid(ax_god, god_img, gbounds, "God Grid")
    local_artist = _render_grid(
        ax_local,
        local_img,
        gbounds,
        "Local Map (0%)",
        cmap="gray",
    )
    # The initial local map is uniformly unknown (0.5). Fix the colour range
    # explicitly so the first frame cannot collapse Matplotlib's normalization
    # to vmin == vmax and hide all later free/obstacle updates.
    local_artist.set_clim(0.0, 1.0)
    xmin, xmax, ymin, ymax = gbounds
    value_artist = ax_value.imshow(
        value_img,
        extent=[xmin, xmax, ymin, ymax],
        origin="upper",
        aspect="equal",
        interpolation="none",
    )
    ax_value.set_xlim(xmin, xmax)
    ax_value.set_ylim(ymin, ymax)
    ax_value.set_title("Value Map", fontsize=10, fontweight="bold")
    camera_artist = ax_camera.imshow(camera_img)
    ax_camera.set_title("Current RGB View", fontsize=10, fontweight="bold")
    ax_camera.axis("off")
    god_robot, = ax_god.plot([], [], "r*", markersize=12, markeredgecolor="black")
    local_robot, = ax_local.plot([], [], "r*", markersize=12, markeredgecolor="black")
    ax_god.plot(start_pt[0], start_pt[1], "go", markersize=6)
    ax_local.plot(start_pt[0], start_pt[1], "go", markersize=6)
    target_artist, = ax_local.plot([], [], "rX", markersize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    t_start = time.perf_counter()
    last_rgb = None
    executor = ThreadPoolExecutor(max_workers=1) if display and capture is not None else None
    pending_query = None
    semantic_enabled = capture is not None

    try:
        for cycle in range(steps):
            if vlfm.is_finished():
                print(f"[{cycle:3d}] FINISHED — explored {vlfm.local_map.explored_ratio():.0%}")
                break

            # Apply a completed background VLM result on the main thread.
            if pending_query is not None and pending_query[0].done():
                future, observation = pending_query
                try:
                    score = future.result()
                    vlfm.inject_scored_observation(score, *observation)
                except Exception as exc:
                    if not vlfm.allow_geometric_fallback:
                        raise
                    print(f"[warn] VLM observation failed; using geometry: {exc}")
                    semantic_enabled = False
                pending_query = None

            camera_fovy = 70.0
            query_due = semantic_enabled and cycle % vlm_query_interval == 0
            query_due = query_due and pending_query is None

            if capture is not None and display:
                # Render RGB every GUI frame. Only depth rendering and VLM
                # inference remain at the lower semantic-query frequency.
                last_rgb = capture.capture_rgb_array(
                    (float(robot[0]), float(robot[1]), 0.5),
                    yaw,
                    pitch=-0.3,
                    fovy=camera_fovy,
                )
                if query_due:
                    depth = capture.capture_depth(
                        (float(robot[0]), float(robot[1]), 0.5),
                        yaw,
                        pitch=-0.3,
                        fovy=camera_fovy,
                    )
                    camera_xy, camera_yaw = capture.camera_pose_xy_yaw(tuple(robot), yaw)
                    hfov = capture.horizontal_fov_rad(camera_fovy)
                    rgb = _encode_rgb_png(last_rgb)
                    future = executor.submit(vlfm.vlm.score_image, rgb, vlfm.instruction)
                    pending_query = (
                        future,
                        (depth, camera_xy, camera_yaw, hfov),
                    )
            elif capture is not None and query_due:
                # Headless runs keep deterministic synchronous inference.
                try:
                    rgb, depth, camera_xy, camera_yaw, hfov = (
                        capture.capture_robot_observation(
                            tuple(robot),
                            yaw,
                            camera_height=0.5,
                            pitch=-0.3,
                            fovy=camera_fovy,
                        )
                    )
                    vlfm.inject_observation(
                        rgb,
                        depth,
                        camera_xy,
                        camera_yaw,
                        hfov,
                        min_depth=0.1,
                        max_depth=5.0,
                    )
                    last_rgb = np.asarray(Image.open(io.BytesIO(rgb)).convert("RGB"))
                except Exception as exc:
                    if not vlfm.allow_geometric_fallback:
                        raise
                    print(f"[warn] RGB-D/VLM observation failed; using geometry: {exc}")
                    semantic_enabled = False

            # Scan → value-guided frontier selection → plan → move.
            vlfm.step(tuple(robot), yaw)

            # Move robot along path
            if vlfm.has_path():
                wp = vlfm.pop_command()
                d = wp - robot
                dist = float(np.linalg.norm(d))
                if dist > 0.005:
                    yaw = np.arctan2(d[1], d[0])
                    robot += d / dist * min(dist, 0.3)

            # Log
            if cycle % 5 == 0 or vlfm.state == FBEState.FINISHED:
                score_str = (
                    f"  score={vlfm.diag.last_vlm_score:.3f}"
                    if vlfm.diag.vlm_query_count
                    else ""
                )
                print(
                    f"[{cycle:3d}] {vlfm.state.value:9s}  "
                    f"expl={vlfm.local_map.explored_ratio():.0%}  "
                    f"f={vlfm.diag.frontier_count:3d}  "
                    f"vlm_q={vlfm.diag.vlm_query_count}"
                    f"{score_str}"
                )

            render_interval = 1 if display else 3
            if cycle % render_interval == 0 or cycle == steps - 1:
                local_img = np.full(vlfm.local_map.data.shape, 0.35)
                local_img[vlfm.local_map.data == 0] = 0.9
                local_img[vlfm.local_map.data == 1] = 0.15
                local_artist.set_data(local_img)
                ax_local.set_title(
                    f"Local Map ({vlfm.local_map.explored_ratio():.0%})",
                    fontsize=10,
                    fontweight="bold",
                )
                god_robot.set_data([robot[0]], [robot[1]])
                local_robot.set_data([robot[0]], [robot[1]])
                if vlfm._target_cluster:
                    tc = vlfm._target_cluster
                    target_artist.set_data([tc.centroid_xy[0]], [tc.centroid_xy[1]])
                else:
                    target_artist.set_data([], [])

                value_artist.set_data(vlfm.value_map_image())
                ax_value.set_title(
                    f"Value Map (score {vlfm.diag.last_vlm_score:.3f})",
                    fontsize=10,
                    fontweight="bold",
                )
                if last_rgb is not None:
                    camera_artist.set_data(last_rgb)
                fig.canvas.draw()
                fig.canvas.flush_events()
                plt.pause(0.03 if display else 0.001)
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)

    elapsed = time.perf_counter() - t_start
    print(
        f"\nDone in {elapsed:.1f}s | VLM queries: {vlfm.diag.vlm_query_count} | "
        f"Explored: {vlfm.local_map.explored_ratio():.1%}"
    )

    if save:
        out = Path(__file__).resolve().parent / "vlfm_demo.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved → {out}")
    plt.ioff()
    if display:
        plt.show()
    else:
        plt.close(fig)


def _encode_rgb_png(rgb: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(rgb, mode="RGB").save(buffer, format="PNG")
    return buffer.getvalue()


if __name__ == "__main__":
    main()
