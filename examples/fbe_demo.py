#!/usr/bin/env python3
"""Frontier-Based Exploration (FBE) demo — matplotlib visualisation.

Shows the exploration progress in three panels:
  left:  God grid (ground truth)
  mid:   Local map (robot's current knowledge)
  right: Frontier map (frontier cells highlighted)

Usage:
  .venv/bin/python examples/fbe_demo.py
  .venv/bin/python examples/fbe_demo.py --steps 300
"""

from __future__ import annotations

import time
from pathlib import Path

import click
import matplotlib.pyplot as plt
import mujoco
import numpy as np

from stretch_mujoco.habitat_scene_gallery import (
    DEFAULT_CACHE_ROOT,
    DEFAULT_SCENE_ID,
    load_habitat_scene_model,
    prepare_habitat_scene,
)
from stretch_mujoco.navigations import NavigationController
from stretch_mujoco.navigations.FBE import (
    FBEPlanner,
    FBEState,
    detect_frontier_cells,
    cluster_frontiers,
)


# ---------------------------------------------------------------------------
def _render_grid(ax, data, bounds, title, cmap="Greys", alpha=1.0):
    """Render a 2-D grid on *ax*."""
    xmin, xmax, ymin, ymax = bounds
    ax.imshow(
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


# ---------------------------------------------------------------------------
@click.command()
@click.option("--scene-id", default=DEFAULT_SCENE_ID, show_default=True)
@click.option(
    "--cache-root",
    type=click.Path(path_type=Path),
    default=DEFAULT_CACHE_ROOT,
    show_default=True,
)
@click.option("--resolution", default=0.2, show_default=True)
@click.option("--agent-radius", default=0.3, show_default=True)
@click.option("--max-range", default=4.0, show_default=True, help="Laser max range (metres).")
@click.option("--steps", default=250, show_default=True, help="Number of exploration steps.")
@click.option("--save/--no-save", default=True, help="Save animation frames.")
def main(scene_id, cache_root, resolution, agent_radius, max_range, steps, save):
    """FBE exploration demo with matplotlib."""
    # --- Build god grid ---
    scene = prepare_habitat_scene(scene_id=scene_id, cache_root=cache_root)
    model = load_habitat_scene_model(scene)
    data = mujoco.MjData(model)
    mujoco.mj_step(model, data)

    mx, Mx = float(scene.bounds[0, 0] + 0.8), float(scene.bounds[1, 0] - 0.8)
    my, My = float(scene.bounds[0, 1] + 0.8), float(scene.bounds[1, 1] - 0.8)

    nav = NavigationController(
        model,
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
    gbounds = (god_grid.x_min, god_grid.x_max, god_grid.y_min, god_grid.y_max)

    # --- Find free start ---
    free_pts = []
    for x in np.linspace(mx + 1, Mx - 1, 10):
        for y in np.linspace(my + 1, My - 1, 10):
            if god_grid.is_world_free(np.array([x, y])):
                free_pts.append((float(x), float(y)))
    start = free_pts[len(free_pts) // 2]

    # --- FBE ---
    fbe = FBEPlanner(
        god_grid,
        robot_xy=start,
        robot_yaw=0.0,
        num_rays=120,
        max_range_m=max_range,
        fov_degrees=270,
        min_cluster_size=3,
        explore_threshold=0.85,
    )
    robot = np.array(start, dtype=float)
    yaw = 0.0

    print(f"Scene: {scene_id}  |  Start: ({start[0]:.1f}, {start[1]:.1f})")
    print(
        f"Grid: {god_grid.occupancy.shape}  |  God-free: "
        f"{100*(~god_grid.occupancy).sum()/god_grid.occupancy.size:.0f}%"
    )
    print(f"Laser: {fbe.num_rays} rays × {max_range}m range, {fbe.fov_degrees}° FOV")
    print(f"Running {steps} steps …\n")

    # --- Matplotlib setup ---
    plt.ion()
    fig, (ax_god, ax_local, ax_frontier) = plt.subplots(1, 3, figsize=(18, 5.5))
    fig.suptitle("Frontier-Based Exploration (FBE)", fontsize=13, fontweight="bold")

    god_img = np.where(god_grid.occupancy, 0.2, 0.9)  # dark = obstacle

    history: list[dict] = []
    t_start = time.perf_counter()

    for step in range(steps):
        fbe.step(tuple(robot), yaw)
        if fbe.state == FBEState.FINISHED:
            print(f"Finished at step {step}")
            break

        if fbe.has_path():
            wp = fbe.current_target()
            d = wp - robot
            dist = float(np.linalg.norm(d))
            if dist > 0.005:
                yaw = np.arctan2(d[1], d[0])
                robot += d / dist * min(dist, 0.3)

        # Log every N steps
        if step % 20 == 0 or step == steps - 1:
            fmask = detect_frontier_cells(fbe.local_map)
            clusters = cluster_frontiers(fbe.local_map, fmask, min_cluster_size=1)
            history.append(
                {
                    "step": step,
                    "explored": fbe.local_map.explored_ratio(),
                    "frontiers": int(fmask.sum()),
                    "clusters": len(clusters),
                    "state": fbe.state.value,
                    "pos": robot.copy(),
                }
            )
            status = (
                f"step {step:3d} | {fbe.state.value:9s} | "
                f"explored {fbe.local_map.explored_ratio():.0%} | "
                f"frontiers {int(fmask.sum()):4d} | "
                f"clusters {len(clusters)}"
            )
            print(status)

        # Update plots every 10 steps
        if step % 10 == 0:
            for ax in (ax_god, ax_local, ax_frontier):
                ax.clear()

            # God grid (ground truth)
            _render_grid(ax_god, god_img, gbounds, "God Grid (ground truth)")

            # Local map
            local_img = np.full(fbe.local_map.data.shape, 0.5)  # grey = unknown
            local_img[fbe.local_map.data == 0] = 0.9  # white = free
            local_img[fbe.local_map.data == 1] = 0.15  # dark = obstacle
            _render_grid(
                ax_local,
                local_img,
                gbounds,
                f"Local Map ({fbe.local_map.explored_ratio():.0%} explored)",
            )

            # Frontiers
            f_img = np.full(fbe.local_map.data.shape, 0.95)  # white bg
            fmask = detect_frontier_cells(fbe.local_map)
            f_img[fmask] = 0.3  # dark = frontier
            _render_grid(ax_frontier, f_img, gbounds, f"Frontiers ({int(fmask.sum())} cells)")

            # Plot robot position on all axes
            for ax in (ax_god, ax_local, ax_frontier):
                ax.plot(robot[0], robot[1], "r*", markersize=12, markeredgecolor="black")
                ax.plot(start[0], start[1], "go", markersize=6, alpha=0.6, label="Start")

            # Plot current path if any
            if fbe.diag.current_path:
                path_pts = np.array(fbe.diag.current_path)
                if len(path_pts) > 1:
                    ax_local.plot(path_pts[:, 0], path_pts[:, 1], "g-", linewidth=2, alpha=0.8)

            # Plot target frontier
            if fbe._target_cluster:
                tc = fbe._target_cluster
                ax_frontier.plot(
                    tc.centroid_xy[0], tc.centroid_xy[1], "rX", markersize=14, markeredgecolor="red"
                )

            fig.canvas.draw()
            fig.canvas.flush_events()
            plt.pause(0.01)

    elapsed = time.perf_counter() - t_start
    print(
        f"\nElapsed: {elapsed:.1f}s  |  Final explored: "
        f"{fbe.local_map.explored_ratio():.1%}  |  State: {fbe.state.value}"
    )
    if fbe.is_finished():
        print(f"Finish reason: {fbe.diag.finish_reason}")

    if save:
        out = Path(__file__).resolve().parent / "fbe_demo.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved → {out}")

    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()
