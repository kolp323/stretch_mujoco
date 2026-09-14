#!/usr/bin/env python3
"""Batch navigation regression test for the generated office scenes.

The script deliberately tests the *static* navigation model first.  The
Stretch body is kept in the XML for compatibility, but its own collision geoms
are disabled while rasterising the navigation grid; otherwise the robot's
initial pose is (correctly) seen as an obstacle.

Examples
--------
  conda run -n habitat_310 python tools/test_office_navigation.py --scene 1
  conda run -n habitat_310 python tools/test_office_navigation.py --scene all
  conda run -n habitat_310 python tools/test_office_navigation.py \
      --algorithm astar --resolution 0.10 --agent-radius 0.30

Each scene writes ``<output>/<scene>.png`` and ``<output>/<scene>.json``.
The PNG shows the occupancy grid, requested/selected targets, and planned
paths.  The JSON is intended to be consumed by later physical-base tests.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

os.environ.setdefault("MPLCONFIGDIR", "/tmp/stretch_mujoco_matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np

from stretch_mujoco.navigations import Algorithm, NavigationController, NavigationPathError


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SCENE_ROOT = ROOT / "stretch_mujoco" / "models" / "assets" / "office_scenes"

def _floor_geom_name(model: mujoco.MjModel) -> str:
    for name in ("office_floor", "hssd_floor_collision"):
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name) >= 0:
            return name
    return "office_floor"


def _scene_paths(scene: str) -> list[tuple[Path, Path]]:
    scene_root = Path(os.environ.get("STRETCH_SCENE_ROOT", str(DEFAULT_SCENE_ROOT)))
    catalog = json.loads((scene_root / "catalog.json").read_text(encoding="utf-8"))
    entries = catalog["scenes"]
    if scene.lower() != "all":
        if scene.isdigit():
            index = int(scene) - 1
            if not 0 <= index < len(entries):
                raise ValueError(f"scene number must be in 1..{len(entries)}")
            entries = [entries[index]]
        else:
            entries = [entry for entry in entries if entry["scene_id"] == scene]
            if not entries:
                raise ValueError(f"unknown scene id: {scene}")
    return [(scene_root / e["mjcf"], scene_root / e["manifest"]) for e in entries]


def _is_descendant_of(model: mujoco.MjModel, body_id: int, ancestor_id: int) -> bool:
    current = body_id
    while current >= 0:
        if current == ancestor_id:
            return True
        parent = int(model.body_parentid[current])
        if parent == current:
            break
        current = parent
    return False


def _disable_robot_collision(model: mujoco.MjModel) -> int:
    """Disable only robot collision geoms in the planning copy of a model."""
    base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    if base_id < 0:
        return 0
    changed = 0
    for geom_id in range(model.ngeom):
        body_id = int(model.geom_bodyid[geom_id])
        if _is_descendant_of(model, body_id, base_id):
            if model.geom_contype[geom_id] or model.geom_conaffinity[geom_id]:
                model.geom_contype[geom_id] = 0
                model.geom_conaffinity[geom_id] = 0
                changed += 1
    return changed


def _free_near(nav: NavigationController, point: np.ndarray) -> np.ndarray | None:
    """Return the closest free grid-cell center to a requested world point."""
    row, col = nav.grid.world_to_cell(np.asarray(point, dtype=float))
    max_radius = max(nav.grid.occupancy.shape)
    for radius in range(max_radius):
        candidates: list[tuple[float, np.ndarray]] = []
        r0, r1 = max(0, row - radius), min(nav.grid.height - 1, row + radius)
        c0, c1 = max(0, col - radius), min(nav.grid.width - 1, col + radius)
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                if max(abs(r - row), abs(c - col)) != radius:
                    continue
                if nav.grid.is_cell_free((r, c)):
                    candidate = nav.grid.cell_to_world((r, c))
                    # The planner also performs a continuous AABB check on
                    # endpoints; reject points that happen to fall inside an
                    # inflated obstacle despite their raster cell being free.
                    if not nav.grid.point_inside_obstacle(candidate):
                        candidates.append((float(np.linalg.norm(candidate - point[:2])), candidate))
        if candidates:
            return min(candidates, key=lambda item: item[0])[1]
    return None


def _approach_point(nav: NavigationController, object_xy: np.ndarray) -> np.ndarray | None:
    """Choose a free floor point around a tabletop object."""
    angles = np.linspace(0.0, 2.0 * np.pi, 24, endpoint=False)
    for radius in (0.55, 0.70, 0.90, 1.10):
        points = [object_xy[:2] + radius * np.array([np.cos(a), np.sin(a)]) for a in angles]
        free = [p for p in points if nav.is_free(p) and not nav.grid.point_inside_obstacle(p)]
        if free:
            return np.asarray(free[0], dtype=float)
    return _free_near(nav, object_xy)


def _best_zone_point(nav: NavigationController, bounds: list[float], center: np.ndarray) -> np.ndarray | None:
    xmin, xmax, ymin, ymax = map(float, bounds)
    margin = max(nav.grid.agent_radius, 0.35)
    candidates = []
    for x in np.arange(xmin + margin, xmax - margin + nav.grid.resolution * 0.5, nav.grid.resolution):
        for y in np.arange(ymin + margin, ymax - margin + nav.grid.resolution * 0.5, nav.grid.resolution):
            point = np.array([x, y])
            if not nav.is_free(point) or nav.grid.point_inside_obstacle(point):
                continue
            clearance = min(
                np.hypot(max(o.minimum[0] - x, 0.0, x - o.maximum[0]),
                         max(o.minimum[1] - y, 0.0, y - o.maximum[1]))
                for o in nav.grid.obstacles
            )
            candidates.append((float(clearance), float(np.linalg.norm(point - center[:2])), point))
    if not candidates:
        return _free_near(nav, center)
    safe = [item for item in candidates if item[0] >= max(0.20, 0.5 * nav.grid.agent_radius)]
    return min(safe or candidates, key=lambda item: item[1])[2]


def _targets(model: mujoco.MjModel, data: mujoco.MjData, nav: NavigationController,
             include_grasps: bool, zones: list[dict] | None = None) -> list[dict]:
    zone_bounds = {str(z.get("type")): z.get("bounds") for z in (zones or [])}
    targets: list[dict] = []
    for site_id in range(model.nsite):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, site_id) or ""
        if name.startswith("zone_") and name.endswith("_center"):
            requested = data.site(name).xpos[:2].copy()
            zone_type = name.removeprefix("zone_").removesuffix("_center")
            selected = (_best_zone_point(nav, zone_bounds[zone_type], requested)
                        if zone_type in zone_bounds else _free_near(nav, requested))
            if selected is not None:
                targets.append({"name": name, "kind": "zone", "requested": requested, "goal": selected})
        elif include_grasps and name.endswith("_grasp_site"):
            requested = data.site(name).xpos[:2].copy()
            selected = _approach_point(nav, requested)
            if selected is not None:
                targets.append({"name": name, "kind": "approach", "requested": requested, "goal": selected})
    return targets


def _path_length(path: list[np.ndarray]) -> float:
    return float(sum(np.linalg.norm(path[i + 1] - path[i]) for i in range(len(path) - 1)))


def _plot(scene_id: str, nav: NavigationController, start: np.ndarray,
          targets: list[dict], results: dict[str, list[dict]], output: Path) -> None:
    algorithms = list(results)
    fig, axes = plt.subplots(1, len(algorithms), figsize=(8 * len(algorithms), 7), squeeze=False)
    for axis, algorithm in zip(axes[0], algorithms):
        occupancy = nav.grid.occupancy
        extent = [nav.grid.x_min, nav.grid.x_max, nav.grid.y_min, nav.grid.y_max]
        axis.imshow(occupancy, origin="lower", extent=extent, cmap="Greys", alpha=0.85,
                    interpolation="none", aspect="equal")
        axis.plot(start[0], start[1], "r*", markersize=14, label="robot start")
        colours = plt.cm.tab20(np.linspace(0, 1, max(len(targets), 1)))
        by_name = {item["name"]: item for item in results[algorithm]}
        for colour, target in zip(colours, targets):
            goal = target["goal"]
            item = by_name[target["name"]]
            axis.plot(goal[0], goal[1], "o", color=colour, markersize=5)
            if item.get("success"):
                path = np.asarray(item["path"])
                axis.plot(path[:, 0], path[:, 1], color=colour, linewidth=1.5, alpha=0.9)
        axis.set_title(f"{scene_id} — {algorithm.upper()}")
        axis.set_xlabel("X (m)")
        axis.set_ylabel("Y (m)")
        axis.grid(alpha=0.15)
    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        axes[0][0].legend(handles, labels, loc="upper right")
    fig.suptitle("Office navigation occupancy grid and planned paths")
    fig.tight_layout()
    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)


def run_scene(xml_path: Path, manifest_path: Path, args: argparse.Namespace, output_dir: Path) -> bool:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    scene_id = manifest["scene_id"]
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    disabled = _disable_robot_collision(model)
    mujoco.mj_forward(model, data)

    start = np.asarray(manifest["robot"]["initial_pose"][:2], dtype=float)
    algorithms = ["astar", "fmm"] if args.algorithm == "both" else [args.algorithm]
    # Build once to select targets using the requested algorithm's grid.
    floor_geom_name = _floor_geom_name(model)
    target_nav = NavigationController(model, data, algorithm=Algorithm.ASTAR,
                                      resolution=args.resolution, agent_radius=args.agent_radius,
                                      require_collision=True, floor_geom_name=floor_geom_name)
    targets = _targets(model, data, target_nav, args.include_grasps, manifest.get("zones"))
    results: dict[str, list[dict]] = {}
    scene_ok = bool(target_nav.is_free(start))
    for algorithm_name in algorithms:
        nav = NavigationController(model, data, algorithm=Algorithm(algorithm_name),
                                   resolution=args.resolution, agent_radius=args.agent_radius,
                                   require_collision=True, floor_geom_name=floor_geom_name)
        entries: list[dict] = []
        for target in targets:
            t0 = time.perf_counter()
            item = {"name": target["name"], "kind": target["kind"],
                    "requested": target["requested"].tolist(), "goal": target["goal"].tolist()}
            try:
                path = nav.plan(start, target["goal"])
                item.update(success=True, path=[point.tolist() for point in path],
                            waypoints=len(path), path_length_m=_path_length(path),
                            planning_ms=(time.perf_counter() - t0) * 1000.0)
            except (NavigationPathError, ValueError, RuntimeError) as error:
                item.update(success=False, error=str(error), planning_ms=(time.perf_counter() - t0) * 1000.0)
                scene_ok = False
            entries.append(item)
        results[algorithm_name] = entries

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{scene_id}.json"
    report = {
        "scene_id": scene_id,
        "xml": str(xml_path),
        "resolution_m": args.resolution,
        "agent_radius_m": args.agent_radius,
        "robot_collision_geoms_disabled": disabled,
        "start": start.tolist(),
        "start_free": bool(target_nav.is_free(start)),
        "grid": {"width": target_nav.grid.width, "height": target_nav.grid.height,
                 "bounds": [target_nav.grid.x_min, target_nav.grid.x_max,
                             target_nav.grid.y_min, target_nav.grid.y_max],
                 "free_cells": int((~target_nav.grid.occupancy).sum()),
                 "total_cells": int(target_nav.grid.occupancy.size)},
        "targets": [{"name": t["name"], "kind": t["kind"],
                     "requested": t["requested"].tolist(), "goal": t["goal"].tolist()} for t in targets],
        "results": results,
        "success": scene_ok,
    }
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    # Reuse the last controller's grid; all controllers share the same grid parameters.
    _plot(scene_id, target_nav, start, targets, results, output_dir / f"{scene_id}.png")
    print(f"{scene_id}: {'PASS' if scene_ok else 'FAIL'} | targets={len(targets)} "
          f"start_free={report['start_free']} | report={json_path}")
    return scene_ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", default="all", help="1..10, scene id, or all")
    parser.add_argument("--scene-root", type=Path, default=None,
                        help="Generated scene directory; defaults to office_scenes")
    parser.add_argument("--algorithm", choices=("astar", "fmm", "both"), default="both")
    parser.add_argument("--resolution", type=float, default=0.10, help="Grid resolution in metres")
    parser.add_argument("--agent-radius", type=float, default=0.30, help="Inflation radius in metres")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Output directory (defaults to output/<scene family>_navigation)")
    parser.add_argument("--include-grasps", dest="include_grasps", action="store_true", default=True,
                        help="Include approach points for grasp sites (default: enabled)")
    parser.add_argument("--zones-only", dest="include_grasps", action="store_false",
                        help="Only test the four zone targets")
    args = parser.parse_args()
    if args.scene_root is not None:
        os.environ["STRETCH_SCENE_ROOT"] = str(args.scene_root.resolve())
    if args.output_dir is None:
        root_name = Path(os.environ.get("STRETCH_SCENE_ROOT", str(DEFAULT_SCENE_ROOT))).name
        family = "home" if "home" in root_name.lower() else "office"
        args.output_dir = ROOT / "output" / f"{family}_navigation"
    if args.resolution <= 0 or args.agent_radius < 0:
        parser.error("--resolution must be > 0 and --agent-radius must be >= 0")
    failures = 0
    for xml_path, manifest_path in _scene_paths(args.scene):
        if not run_scene(xml_path, manifest_path, args, args.output_dir):
            failures += 1
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
