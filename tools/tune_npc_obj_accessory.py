#!/usr/bin/env python3
"""Interactively tune a recipe-backed NPC OBJ/GLB accessory against a body frame."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from types import ModuleType
from typing import cast

import numpy as np

from stretch_mujoco.npc.appearance_pipeline.accessory_recipe import (
    FusedAccessoryRecipe,
    FusedAccessoryRuntimeConfig,
)


@dataclass(frozen=True)
class ObjGeometry:
    vertices: np.ndarray
    triangles: np.ndarray


@dataclass(frozen=True)
class TuningContext:
    runtime_path: Path
    recipe_path: Path
    recipe: FusedAccessoryRecipe
    runtime: FusedAccessoryRuntimeConfig
    body: ObjGeometry
    source_accessory: ObjGeometry
    reference_clip: str
    reference_frame: int


@lru_cache(maxsize=None)
def _tool_module(filename: str) -> ModuleType:
    path = Path(__file__).with_name(filename)
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _obj_geometry(source: bytes, *, source_name: str) -> ObjGeometry:
    vertices: list[list[float]] = []
    triangles: list[tuple[int, int, int]] = []
    for line in source.decode("utf-8").splitlines():
        values = line.split()
        if not values:
            continue
        if values[0] == "v" and len(values) >= 4:
            vertices.append([float(value) for value in values[1:4]])
        elif values[0] == "f" and len(values) >= 4:
            face: list[int] = []
            for token in values[1:]:
                raw_index = int(token.split("/")[0])
                face.append(raw_index - 1 if raw_index > 0 else len(vertices) + raw_index)
            triangles.extend(
                (face[0], face[index], face[index + 1]) for index in range(1, len(face) - 1)
            )
    if not vertices or not triangles:
        raise ValueError(f"OBJ '{source_name}' needs vertices and faces")
    vertex_array = np.asarray(vertices, dtype=np.float64)
    triangle_array = np.asarray(triangles, dtype=np.int64)
    if triangle_array.min() < 0 or triangle_array.max() >= len(vertex_array):
        raise ValueError(f"OBJ '{source_name}' has an invalid face index")
    return ObjGeometry(vertex_array, triangle_array)


def load_tuning_context(
    runtime_path: str | Path, *, reference_clip: str = "idle", reference_frame: int = 0
) -> TuningContext:
    """Load the exact recipe, source accessory and body frame used by production."""
    runtime_source = Path(runtime_path).resolve()
    runtime = FusedAccessoryRuntimeConfig.from_json(runtime_source)
    recipe_path = runtime.resolve_path(runtime_source, "recipe")
    recipe = FusedAccessoryRecipe.from_json(recipe_path)
    manifest_path = runtime.resolve_path(runtime_source, "source_manifest")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    try:
        frames = manifest["bundles"][runtime.bundle]["clips"][reference_clip]["frames"]
    except (KeyError, TypeError) as error:
        raise ValueError(
            f"Reference clip does not exist: {runtime.bundle}/{reference_clip}"
        ) from error
    if not isinstance(frames, list) or not 0 <= reference_frame < len(frames):
        maximum = len(frames) - 1 if isinstance(frames, list) else "unknown"
        raise ValueError(
            f"Reference frame {reference_frame} is out of range for "
            f"{runtime.bundle}/{reference_clip}; valid range is 0..{maximum}"
        )
    frame_relative = frames[reference_frame]
    body_path = (manifest_path.parent / frame_relative).resolve()
    prepare = _tool_module("prepare_obj_accessory.py")
    source_obj, _ = prepare._source_obj(
        runtime.resolve_path(runtime_source, "source_archive"),
        source_format=runtime.source_format,
        nested_archive_member=runtime.nested_archive_member,
        obj_member=runtime.obj_member,
    )
    return TuningContext(
        runtime_path=runtime_source,
        recipe_path=recipe_path,
        recipe=recipe,
        runtime=runtime,
        body=_obj_geometry(body_path.read_bytes(), source_name=str(body_path)),
        source_accessory=_obj_geometry(source_obj, source_name=runtime.source_archive),
        reference_clip=reference_clip,
        reference_frame=reference_frame,
    )


def transformed_accessory(context: TuningContext, recipe: FusedAccessoryRecipe) -> np.ndarray:
    """Project source accessory vertices into the selected body frame."""
    prepare = _tool_module("prepare_obj_accessory.py")
    local_vertices = prepare.transform_accessory_vertices(
        context.source_accessory.vertices,
        mesh_scale=recipe.mesh_scale,
        back_tilt_degrees=recipe.back_tilt_degrees,
        roll_degrees=recipe.roll_degrees,
        yaw_degrees=recipe.yaw_degrees,
        source_unit_scale=context.runtime.source_unit_scale,
        source_vertical_anchor=recipe.source_vertical_anchor,
    )
    position = prepare.head_top_position(
        context.body.vertices,
        lateral_offset_m=recipe.lateral_offset_m,
        head_clearance_m=recipe.head_clearance_m,
        back_offset_m=recipe.back_offset_m,
    )
    return local_vertices + np.asarray(position, dtype=np.float64)


def save_recipe(path: str | Path, recipe: FusedAccessoryRecipe) -> None:
    """Atomically replace a recipe after validating its canonical projection."""
    destination = Path(path).resolve()
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(recipe.as_dict(), indent=2) + "\n", encoding="utf-8")
    try:
        FusedAccessoryRecipe.from_json(temporary)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def _sample_triangles(geometry: ObjGeometry, maximum: int, *, head_only: bool) -> np.ndarray:
    triangles = geometry.triangles
    if head_only:
        centroids = geometry.vertices[triangles].mean(axis=1)
        selected = (centroids[:, 2] > 1.35) & (np.hypot(centroids[:, 0], centroids[:, 1]) < 0.3)
        if np.any(selected):
            triangles = triangles[selected]
    if len(triangles) > maximum:
        indices = np.linspace(0, len(triangles) - 1, maximum, dtype=np.int64)
        triangles = triangles[indices]
    return triangles


def run_tuner(
    context: TuningContext,
    *,
    snapshot: str | Path | None = None,
    full_body: bool = False,
    maximum_faces: int = 6000,
) -> None:
    """Open the interactive tuner, or render its initial state for headless QA."""
    if snapshot is not None:
        import matplotlib

        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Button, RadioButtons, Slider, TextBox
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from mpl_toolkits.mplot3d.axes3d import Axes3D

    figure = plt.figure(figsize=(13.5, 8.5))
    axis = cast(Axes3D, figure.add_axes((0.04, 0.08, 0.64, 0.86), projection="3d"))
    body_triangles = _sample_triangles(context.body, maximum_faces, head_only=not full_body)
    accessory_triangles = _sample_triangles(
        context.source_accessory, maximum_faces, head_only=False
    )
    # Both meshes are deliberately opaque.  Semi-transparent preview meshes make
    # a bad fit look acceptable by letting the body show through the accessory.
    body_collection = Poly3DCollection(
        context.body.vertices[body_triangles],
        facecolor="#c29a78",
        edgecolor="#8a684f",
        linewidth=0.02,
        alpha=1.0,
    )
    axis.add_collection3d(body_collection)
    initial_accessory = transformed_accessory(context, context.recipe)
    accessory_collection = Poly3DCollection(
        initial_accessory[accessory_triangles],
        facecolor="#2e86c1",
        edgecolor="#154360",
        linewidth=0.08,
        alpha=1.0,
    )
    axis.add_collection3d(accessory_collection)
    axis.set_xlabel("X lateral")
    axis.set_ylabel("Y back (+)")
    axis.set_zlabel("Z up")
    axis.set_box_aspect((1, 1, 1))
    axis.view_init(elev=8, azim=-90)

    if full_body:
        minimum = context.body.vertices.min(axis=0)
        maximum = context.body.vertices.max(axis=0)
        centre = (minimum + maximum) / 2
        half_extent = max(float((maximum - minimum).max()) * 0.55, 0.5)
    else:
        centre = np.array([0.0, 0.0, 1.65])
        half_extent = max(
            0.33,
            float(np.ptp(initial_accessory, axis=0).max()) * 0.75,
        )
    axis.set_xlim(centre[0] - half_extent, centre[0] + half_extent)
    axis.set_ylim(centre[1] - half_extent, centre[1] + half_extent)
    axis.set_zlim(centre[2] - half_extent, centre[2] + half_extent)

    status = figure.text(
        0.72,
        0.04,
        f"Loaded {context.recipe.accessory_id} | unsaved changes: no",
        fontsize=9,
    )
    slider_specs = (
        ("Scale", "mesh_scale", 0.01, max(3.0, context.recipe.mesh_scale * 2.5)),
        ("X lateral (m)", "lateral_offset_m", -0.30, 0.30),
        ("Y back (m)", "back_offset_m", -0.30, 0.30),
        ("Z clearance (m)", "head_clearance_m", -0.30, 0.30),
        ("Pitch/back (deg)", "back_tilt_degrees", -180.0, 180.0),
        ("Roll (deg)", "roll_degrees", -180.0, 180.0),
        ("Yaw (deg)", "yaw_degrees", -180.0, 180.0),
    )
    sliders: dict[str, Slider] = {}
    value_boxes: dict[str, TextBox] = {}
    updating_controls = False
    for index, (label, field, lower, upper) in enumerate(slider_specs):
        value = float(getattr(context.recipe, field))
        lower, upper = min(lower, value), max(upper, value)
        y = 0.82 - index * 0.085
        slider_axis = figure.add_axes((0.75, y, 0.14, 0.032))
        sliders[field] = Slider(
            slider_axis,
            label,
            lower,
            upper,
            valinit=value,
            valfmt="%1.4f",
        )
        value_axis = figure.add_axes((0.90, y, 0.07, 0.032))
        value_boxes[field] = TextBox(value_axis, "", initial=f"{value:.6g}")

    def selected_recipe() -> FusedAccessoryRecipe:
        return replace(
            context.recipe,
            mesh_scale=float(sliders["mesh_scale"].val),
            lateral_offset_m=float(sliders["lateral_offset_m"].val),
            back_offset_m=float(sliders["back_offset_m"].val),
            head_clearance_m=float(sliders["head_clearance_m"].val),
            back_tilt_degrees=float(sliders["back_tilt_degrees"].val),
            roll_degrees=float(sliders["roll_degrees"].val),
            yaw_degrees=float(sliders["yaw_degrees"].val),
        )

    def update(_: object = None) -> None:
        nonlocal updating_controls
        recipe = selected_recipe()
        vertices = transformed_accessory(context, recipe)
        accessory_collection.set_verts(vertices[accessory_triangles])
        if not updating_controls:
            updating_controls = True
            try:
                for field, box in value_boxes.items():
                    box.set_val(f"{float(getattr(recipe, field)):.6g}")
            finally:
                updating_controls = False
        status.set_text(f"Loaded {recipe.accessory_id} | unsaved changes: yes")
        figure.canvas.draw_idle()

    for slider in sliders.values():
        slider.on_changed(update)

    def edit_value(field: str, raw: str) -> None:
        nonlocal updating_controls
        if updating_controls:
            return
        try:
            value = float(raw)
            if not np.isfinite(value):
                raise ValueError
            updating_controls = True
            sliders[field].set_val(value)
        except (TypeError, ValueError):
            status.set_text(f"Invalid numeric value for {field}: {raw!r}")
            value_boxes[field].set_val(f"{float(sliders[field].val):.6g}")
        finally:
            updating_controls = False

    for field, box in value_boxes.items():
        box.on_submit(lambda raw, field=field: edit_value(field, raw))

    save_axis = figure.add_axes((0.75, 0.10, 0.10, 0.05))
    save_button = Button(save_axis, "Save recipe")

    def save(_: object = None) -> None:
        recipe = selected_recipe()
        save_recipe(context.recipe_path, recipe)
        status.set_text(f"Saved {context.recipe_path.name}; run production build to fuse all clips")
        figure.canvas.draw_idle()
        print(f"Saved recipe: {context.recipe_path}")

    save_button.on_clicked(save)
    reset_axis = figure.add_axes((0.87, 0.10, 0.09, 0.05))
    reset_button = Button(reset_axis, "Reset")

    def reset(_: object = None) -> None:
        nonlocal updating_controls
        updating_controls = True
        for field, slider in sliders.items():
            slider.set_val(float(getattr(context.recipe, field)))
        updating_controls = False
        for field, slider in sliders.items():
            value_boxes[field].set_val(f"{float(slider.val):.6g}")
        status.set_text(f"Loaded {context.recipe.accessory_id} | unsaved changes: no")
        figure.canvas.draw_idle()

    reset_button.on_clicked(reset)
    view_axis = figure.add_axes((0.75, 0.16, 0.20, 0.13))
    views = RadioButtons(view_axis, ("Front", "Left", "Back", "Right", "Top"), active=0)
    view_angles = {
        "Front": (8, -90),
        "Left": (8, 0),
        "Back": (8, 90),
        "Right": (8, 180),
        "Top": (90, -90),
    }

    def change_view(label: str | None) -> None:
        if label is None:
            return
        elevation, azimuth = view_angles[label]
        axis.view_init(elev=elevation, azim=azimuth)
        figure.canvas.draw_idle()

    views.on_clicked(change_view)
    figure.suptitle(
        f"NPC accessory tuner | {context.recipe.accessory_id} | "
        f"{context.reference_clip}/{context.reference_frame}\n"
        "Blue: accessory  Tan: body  Mouse: orbit/zoom  Ctrl+S: save  R: reset",
        fontsize=11,
    )

    def key_press(event: object) -> None:
        key = getattr(event, "key", None)
        if key in {"ctrl+s", "cmd+s"}:
            save()
        elif key == "r":
            reset()

    figure.canvas.mpl_connect("key_press_event", key_press)
    if snapshot is not None:
        destination = Path(snapshot).resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(destination, dpi=120)
        plt.close(figure)
        return
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-config", type=Path, required=True)
    parser.add_argument("--reference-clip", default="idle")
    parser.add_argument("--reference-frame", type=int, default=0)
    parser.add_argument("--full-body", action="store_true")
    parser.add_argument("--maximum-faces", type=int, default=6000)
    parser.add_argument(
        "--snapshot", type=Path, help="Render current settings without opening a GUI"
    )
    args = parser.parse_args()
    if args.reference_frame < 0:
        parser.error("--reference-frame must not be negative")
    if args.maximum_faces <= 0:
        parser.error("--maximum-faces must be positive")
    context = load_tuning_context(
        args.runtime_config,
        reference_clip=args.reference_clip,
        reference_frame=args.reference_frame,
    )
    run_tuner(
        context,
        snapshot=args.snapshot,
        full_body=args.full_body,
        maximum_faces=args.maximum_faces,
    )


if __name__ == "__main__":
    main()
