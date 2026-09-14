#!/usr/bin/env python3
"""Navigation demo: A* vs FMM in a Habitat scene.

Loads a converted HSSD Habitat scene, builds an occupancy grid, and plans
paths with both algorithms, visualising the results with matplotlib.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import click
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import mujoco
import numpy as np

from stretch_mujoco.habitat_scene_gallery import (
    DEFAULT_CACHE_ROOT,
    load_habitat_scene_model,
    prepare_habitat_scene,
)
from stretch_mujoco.navigations import (
    Algorithm,
    NavigationController,
    NavigationPathError,
)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_SCENE_ID = "108294417_176709879"
DEFAULT_CACHE = DEFAULT_CACHE_ROOT
GRID_RESOLUTION = 0.2  # metres / cell
AGENT_RADIUS = 0.3       # inflation radius (metres)
MIN_OBSTACLE_H = 0.1     # ignore geoms entirely below this Z
MAX_OBSTACLE_H = 3.0     # ignore geoms entirely above this Z


# ---------------------------------------------------------------------------
def find_free_pair(
    nav: NavigationController,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    min_dist: float = 2.0,
    max_dist: float = 15.0,
) -> Optional[tuple[tuple[float, float], tuple[float, float]]]:
    """Return a pair of free points within the given distance range."""
    free = []
    for x in np.linspace(*x_range, 12):
        for y in np.linspace(*y_range, 12):
            if nav.is_free((x, y)):
                free.append((float(x), float(y)))
    rng = np.random.default_rng(42)
    rng.shuffle(free)
    for i, s in enumerate(free[:30]):
        for g in free[i + 1 :]:
            d = float(np.linalg.norm(np.array(s) - np.array(g)))
            if min_dist <= d <= max_dist:
                try:
                    nav.plan(start=s, goal=g)
                    return s, g
                except NavigationPathError:
                    continue
    return None


def build_grid_image(controller: NavigationController) -> np.ndarray:
    occ = controller.grid.occupancy
    img = np.full((*occ.shape, 3), 255, dtype=np.uint8)
    img[occ] = [40, 40, 40]
    return img


def plot_demo(controller, test_cases, algorithm, ax, colours):
    grid_img = build_grid_image(controller)
    extent = [
        controller.grid.x_min,
        controller.grid.x_max,
        controller.grid.y_min,
        controller.grid.y_max,
    ]
    ax.imshow(grid_img, extent=extent, origin="lower", aspect="equal",
              interpolation="none")

    for idx, ((sx, sy), (gx, gy), label) in enumerate(test_cases):
        color = colours[idx % len(colours)]
        t0 = time.perf_counter()
        try:
            path = controller.plan(start=(sx, sy), goal=(gx, gy))
            elapsed = (time.perf_counter() - t0) * 1000
        except NavigationPathError as e:
            print(f"  [{algorithm.value.upper()}] {label}: FAILED — {e}")
            continue
        pts = np.array(path)
        dist = sum(float(np.linalg.norm(pts[i+1]-pts[i])) for i in range(len(pts)-1))
        ax.plot(pts[:, 0], pts[:, 1], "-o", color=color, markersize=3, linewidth=2,
                label=f"{label}  ({len(pts)} wp, {dist:.2f} m, {elapsed:.1f} ms)")
        ax.plot(sx, sy, "o", color=color, markersize=10, markeredgecolor="black")
        ax.plot(gx, gy, "X", color=color, markersize=12, markeredgecolor="black")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(f"{algorithm.value.upper()} — Habitat Scene")
    ax.legend(loc="upper right", fontsize=7, framealpha=0.9)
    ax.grid(True, alpha=0.15)


# ---------------------------------------------------------------------------
@click.command()
@click.option("--scene-id", default=DEFAULT_SCENE_ID, show_default=True)
@click.option("--cache-root", type=click.Path(path_type=Path),
              default=DEFAULT_CACHE, show_default=True)
@click.option("--resolution", default=GRID_RESOLUTION, show_default=True)
@click.option("--agent-radius", default=AGENT_RADIUS, show_default=True)
@click.option("--save/--no-save", default=True, help="Save figure to PNG")
def main(scene_id, cache_root, resolution, agent_radius, save):
    """Run A* and FMM navigation on a Habitat scene."""
    print("=" * 60)
    print("  Navigation Demo — Habitat Scene")
    print("=" * 60)

    # --- Load scene ---
    scene = prepare_habitat_scene(
        scene_id=scene_id,
        cache_root=cache_root,
    )
    model = load_habitat_scene_model(scene)
    data = mujoco.MjData(model)
    mujoco.mj_step(model, data)

    bounds = scene.bounds
    mx, Mx = float(bounds[0, 0] + 0.8), float(bounds[1, 0] - 0.8)
    my, My = float(bounds[0, 1] + 0.8), float(bounds[1, 1] - 0.8)

    print(f"Scene: {scene_id}")
    print(f"Bounds: X ∈ [{mx:.1f}, {Mx:.1f}]  Y ∈ [{my:.1f}, {My:.1f}]  "
          f"Z ∈ [{bounds[0,2]:.1f}, {bounds[1,2]:.1f}]")
    print(f"Objects: {scene.object_count}  |  Geoms: {model.ngeom}\n")

    # --- Build controllers ---
    grid_kw = dict(
        bounds=(mx, Mx, my, My),
        resolution=resolution,
        agent_radius=agent_radius,
        minimum_obstacle_height=MIN_OBSTACLE_H,
        maximum_obstacle_height=MAX_OBSTACLE_H,
        require_collision=False,
        exclude_prefixes=("habitat_stage_",),
    )

    nav_a = NavigationController(model, data, algorithm=Algorithm.ASTAR, **grid_kw)
    nav_f = NavigationController(model, data, algorithm=Algorithm.FMM, **grid_kw)

    occ = nav_a.grid.occupancy
    free = int((~occ).sum())
    print(f"Grid: {occ.shape[0]}×{occ.shape[1]} cells  "
          f"|  Free: {free}/{occ.size} ({100*free/occ.size:.1f}%)\n")

    # --- Find test cases ---
    print("Finding test point pairs …")
    test_cases: list[tuple[tuple[float, float], tuple[float, float], str]] = []

    # Auto-discover reachable pairs
    xr = (mx + 2, Mx - 2)
    yr = (my + 2, My - 2)
    for dist_range, label in [((2.0, 5.0), "Short hop"),
                               ((5.0, 10.0), "Medium crossing"),
                               ((10.0, 18.0), "Long traverse")]:
        pair = find_free_pair(nav_a, xr, yr, *dist_range)
        if pair:
            s, g = pair
            test_cases.append((s, g, label))
            print(f"  {label}: ({s[0]:.1f}, {s[1]:.1f}) → ({g[0]:.1f}, {g[1]:.1f})")

    if len(test_cases) < 2:
        # Fallback: find any two free points
        free_pts = []
        for x in np.linspace(mx, Mx, 8):
            for y in np.linspace(my, My, 8):
                if nav_a.is_free((x, y)):
                    free_pts.append((float(x), float(y)))
        if len(free_pts) >= 2:
            test_cases = [(free_pts[0], free_pts[-1], "Free-space path")]

    if not test_cases:
        print("ERROR: No valid test cases found.")
        return

    # --- Run planners ---
    print("\n--- A* ---")
    for (sx, sy), (gx, gy), label in test_cases:
        t0 = time.perf_counter()
        try:
            path = nav_a.plan(start=(sx, sy), goal=(gx, gy))
            elapsed = (time.perf_counter() - t0) * 1000
            pts = np.array(path)
            dist = sum(float(np.linalg.norm(pts[i+1]-pts[i])) for i in range(len(pts)-1))
            print(f"  {label}: {len(pts)} wp, {dist:.2f}m, {elapsed:.1f}ms")
        except NavigationPathError as e:
            print(f"  {label}: FAILED — {e}")

    print("\n--- FMM ---")
    for (sx, sy), (gx, gy), label in test_cases:
        t0 = time.perf_counter()
        try:
            path = nav_f.plan(start=(sx, sy), goal=(gx, gy))
            elapsed = (time.perf_counter() - t0) * 1000
            pts = np.array(path)
            dist = sum(float(np.linalg.norm(pts[i+1]-pts[i])) for i in range(len(pts)-1))
            print(f"  {label}: {len(pts)} wp, {dist:.2f}m, {elapsed:.1f}ms")
        except NavigationPathError as e:
            print(f"  {label}: FAILED — {e}")

    # --- Plot ---
    fig, (ax_a, ax_f) = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle(f"Navigation Demo — Habitat Scene {scene_id}", fontsize=14,
                 fontweight="bold")
    colours = plt.cm.tab10.colors

    plot_demo(nav_a, test_cases, Algorithm.ASTAR, ax_a, colours)
    plot_demo(nav_f, test_cases, Algorithm.FMM, ax_f, colours)

    plt.tight_layout()
    if save:
        out_path = Path(__file__).resolve().parent / "navigation_demo.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"\nFigure saved → {out_path}")
    plt.show()


if __name__ == "__main__":
    main()
