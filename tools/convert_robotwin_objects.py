#!/usr/bin/env python3
"""Convert RoboTwin GLB objects into MuJoCo graspable object assets.

RoboTwin stores each object as ``visual/base0.glb`` + ``collision/base0.glb``
plus a ``model_data0.json`` giving the uniform scale to real-world size.
MuJoCo 3.2.6 only loads .stl/.obj/.ply meshes (no glTF), so each object is
converted to an OBJ visual mesh + STL collision mesh + optional texture, wrapped
in an MJCF that defines a free-jointed body ready to be placed on a table.

Two quirks handled here:
  * ``trimesh`` converts glTF texture coordinates to OBJ/OpenGL coordinates
    while loading. The OBJ export must retain those ``vt`` values or MuJoCo
    falls back to generated coordinates and the textures appear misplaced.
  * Objects without a texture image (solid PBR color) get the color baked into
    the MJCF material instead.

Output layout:
  stretch_mujoco/models/assets/grasp_objects/<name>/
      <name>.xml            # <include>-able MJCF (freejoint body)
      <name>_visual.obj     # scaled visual mesh with UV coordinates
      <name>_collision.stl  # scaled collision mesh
      <name>.png            # texture (only for textured objects)
      metadata.json         # min_z/height/extents/mass for table placement

Usage:
  uv run python tools/convert_robotwin_objects.py                      # curated 12
  uv run python tools/convert_robotwin_objects.py --objects 071_can,035_apple
  uv run python tools/convert_robotwin_objects.py --objects 001_bottle --model-id 1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh

from stretch_mujoco.paths import configured_path, require_external_directory
from trimesh.exchange.obj import export_obj
from trimesh.util import concatenate

# Curated set: drinks / fruit+snacks / office supplies.
DEFAULT_OBJECTS = [
    # drinks
    "001_bottle",
    "071_can",
    "038_milk-box",
    "101_milk-tea",
    # fruit & snacks
    "035_apple",
    "green_apple",
    "075_bread",
    "025_chips-tub",
    "069_vagetable",
    # office supplies
    "043_book",
    "092_notebook",
    "116_keyboard",
    "017_calculator",
]

# RoboTwin's model_data["scale"] is unreliable (missing or wildly off for many
# objects), so each curated object is scaled to a canonical real-world size:
#   name -> (largest bounding-box dimension in meters, stand upright on table)
# ``upright`` rotates containers so their long axis points up (+z); flat objects
# keep the GLB orientation.
CANONICAL_SIZE: dict[str, tuple[float, bool]] = {
    "001_bottle": (0.25, True),
    "071_can": (0.12, True),
    "038_milk-box": (0.20, True),
    "101_milk-tea": (0.18, True),
    "025_chips-tub": (0.18, True),
    "035_apple": (0.08, False),
    "green_apple": (0.08, False),
    "075_bread": (0.20, False),
    "069_vagetable": (0.15, False),
    "043_book": (0.24, False),
    "092_notebook": (0.21, False),
    "116_keyboard": (0.45, False),
    "017_calculator": (0.15, False),
}

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROBOTWIN_DIR = configured_path("STRETCH_MUJOCO_ROBOTWIN_ROOT")
DEFAULT_OUT_DIR = PROJECT_ROOT / "stretch_mujoco" / "models" / "assets" / "grasp_objects"

# RoboTwin ships several variants per object; the default (base0) texture is
# often low-detail (nearly flat). These overrides pick a sharper variant where
# one exists, so objects don't look blurry in the cameras/viewer.
MODEL_VARIANTS: dict[str, int] = {
    "101_milk-tea": 2,  # texture sharpness 16 -> 118
    "025_chips-tub": 1,  # 115 -> 290
    "001_bottle": 10,  # 142 -> 561
}

# Friendly object ids backed by a specific RoboTwin source variant.
SOURCE_VARIANTS: dict[str, tuple[str, int]] = {
    "green_apple": ("103_fruit", 0),
}

# Mass estimation: volume (m^3) x bulk density. Food/plastic containers are
# mostly ~0.5-1 g/cm^3; keep the result in a graspable band.
DENSITY_KG_M3 = 500.0
MIN_MASS_KG = 0.1
MAX_MASS_KG = 2.0
# Realistic tabletop contact friction: objects sit on wood, not rubber.
# Sliding 0.5 is a typical can/bottle-on-table value; rolling 0.01 is a small
# but non-zero rolling resistance (MuJoCo's default 0.0001 lets round objects
# roll almost freely off the table, while larger values make them look glued).
FRICTION = 0.5
ROLLING_FRICTION = 0.01


def _load_mesh(path: Path) -> trimesh.Trimesh:
    """Load a GLB and return a single merged Trimesh (handles multi-geometry scenes)."""
    loaded = trimesh.load(path)
    if isinstance(loaded, trimesh.Scene):
        return concatenate(loaded.dump())
    if isinstance(loaded, trimesh.Trimesh):
        return loaded.copy()
    return next(iter(loaded.geometry.values())).copy()


def _texture_image(visual: trimesh.Trimesh) -> object | None:
    material = getattr(visual.visual, "material", None)
    if material is None:
        return None
    return getattr(material, "baseColorTexture", None)


def _solid_rgba(visual: trimesh.Trimesh) -> list[float] | None:
    """Return an MJCF rgba for objects without a texture (PBR base color).

    RoboTwin GLBs store the base color both as standard 0-1 factors and as
    0-255 ints, so values above 1 are normalized to the [0, 1] MJCF range.
    """
    material = getattr(visual.visual, "material", None)
    if material is None:
        return None
    for attr in ("baseColorFactor", "main_color"):
        value = getattr(material, attr, None)
        if value is None:
            continue
        value = np.asarray(value, dtype=float)
        if value.ndim != 1 or value.size < 3:
            continue
        if value.max() > 1.5:
            value = value / 255.0  # stored as 0-255 ints
        rgb = value[:3]
        alpha = float(value[3]) if value.size > 3 else 1.0
        return [float(rgb[0]), float(rgb[1]), float(rgb[2]), alpha]
    return None


def _upright_rotation(mesh: trimesh.Trimesh) -> np.ndarray:
    """Return a 4x4 rotation aligning *mesh*'s longest principal axis to +z.

    The visual and collision GLBs are aligned independently, so the SAME matrix
    must be applied to both to keep them consistent (separate calls would leave
    one of them upside down when the two GLBs point their long axes opposite).
    """
    cov = np.cov(mesh.vertices.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    axis = eigvecs[:, int(np.argmax(eigvals))]
    if axis[2] < 0:
        axis = -axis
    return trimesh.geometry.align_vectors(axis, [0, 0, 1])


def convert_object(
    object_name: str,
    robotwin_dir: Path,
    out_dir: Path,
    *,
    model_id: int = 0,
) -> dict:
    """Convert one RoboTwin object; return its placement metadata."""
    source_name, model_id = SOURCE_VARIANTS.get(object_name, (object_name, model_id))
    source = robotwin_dir / source_name
    if not source.is_dir():
        raise FileNotFoundError(f"RoboTwin object not found: {source}")

    model_id = MODEL_VARIANTS.get(object_name, model_id)
    visual = _load_mesh(source / "visual" / f"base{model_id}.glb")
    collision = _load_mesh(source / "collision" / f"base{model_id}.glb")

    # Stand containers upright, then scale the largest dimension to its
    # canonical real-world size. Falls back to model_data["scale"] for objects
    # not in CANONICAL_SIZE, sanity-clamped so nothing is absurd.
    canonical = CANONICAL_SIZE.get(object_name)
    if canonical is not None:
        target, upright = canonical
        if upright:
            rotation = _upright_rotation(visual)
            visual.apply_transform(rotation)
            collision.apply_transform(rotation)
        largest = float(visual.extents.max())
        scale = target / largest if largest > 0 else 1.0
        if not 0.01 <= scale <= 20.0:
            scale = 1.0
    else:
        try:
            with open(source / f"model_data{model_id}.json") as fh:
                scale = float(np.asarray(json.load(fh).get("scale", [1.0])).ravel()[0])
        except Exception:
            scale = 1.0

    # Bake real-world scale into the vertices (uniform for all RoboTwin objects).
    visual.apply_scale(scale)
    collision.apply_scale(scale)

    dest = out_dir / object_name
    dest.mkdir(parents=True, exist_ok=True)
    prefix = object_name.replace("-", "_")
    visual_obj = dest / f"{object_name}_visual.obj"
    collision_stl = dest / f"{object_name}_collision.stl"

    # Keep the GLB's UVs, but discard its material references: MJCF owns the
    # texture and material, and no companion MTL file is needed.
    obj_text = export_obj(visual, include_texture=True)
    obj_text = (
        "\n".join(
            line for line in obj_text.splitlines() if not line.startswith(("mtllib ", "usemtl "))
        )
        + "\n"
    )
    visual_obj.write_text(obj_text)

    collision.export(str(collision_stl))

    # Texture or solid color.
    image = _texture_image(visual)
    texture_png = dest / f"{object_name}.png"
    if image is not None:
        image.save(str(texture_png))
        material_xml = f'<material name="{prefix}_mat" texture="{prefix}_tex"/>'
        texture_xml = f'<texture name="{prefix}_tex" type="2d" file="{texture_png.name}"/>'
    else:
        texture_png = None
        rgba = _solid_rgba(visual) or [0.5, 0.5, 0.5, 1.0]
        material_xml = (
            f'<material name="{prefix}_mat" rgba="{" ".join(f"{v:.4g}" for v in rgba)}"/>'
        )
        texture_xml = ""

    # Mass from scaled volume, clamped to a graspable band.
    try:
        volume = float(visual.volume)
        mass = volume * DENSITY_KG_M3 if volume > 0 else 0.3
    except Exception:
        mass = 0.3
    mass = float(np.clip(mass, MIN_MASS_KG, MAX_MASS_KG))

    # Placement is driven by the COLLISION mesh (the physical contact surface);
    # the visual bottom can sit a fraction of a mm off and would otherwise get
    # pushed up by the solver, tipping round objects.
    phys = collision.bounds  # (2, 3) min/max after scale
    min_z = float(phys[0, 2])
    height = float(phys[1, 2] - phys[0, 2])
    bounds = visual.bounds  # visual bounds for footprint metadata

    xml = f"""<mujoco model="{object_name}">
  <asset>
    <mesh name="{prefix}_visual" file="{visual_obj.name}"/>
    <mesh name="{prefix}_collision" file="{collision_stl.name}"/>
    {texture_xml}
    {material_xml}
  </asset>
  <worldbody>
    <body name="{object_name}" pos="0 0 0">
      <freejoint/>
      <geom type="mesh" mesh="{prefix}_visual" material="{prefix}_mat"
            mass="{mass:.4g}" contype="0" conaffinity="0"/>
      <geom type="mesh" mesh="{prefix}_collision" mass="0"
            contype="1" conaffinity="1"
            friction="{FRICTION:.3g} .005 {ROLLING_FRICTION:.4g}"
            rgba="0 0 0 0"/>
    </body>
  </worldbody>
</mujoco>
"""
    (dest / f"{object_name}.xml").write_text(xml + "\n")

    metadata = {
        "name": object_name,
        "min_z": min_z,
        "height": height,
        "extents": [[float(bounds[0, i]), float(bounds[1, i])] for i in range(3)],
        "mass_kg": mass,
        "textured": texture_png is not None,
    }
    (dest / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--objects",
        default=",".join(DEFAULT_OBJECTS),
        help="comma-separated RoboTwin object ids (default: curated 12)",
    )
    parser.add_argument(
        "--robotwin-dir",
        type=Path,
        default=DEFAULT_ROBOTWIN_DIR,
        help="RoboTwin assets/objects directory (or set STRETCH_MUJOCO_ROBOTWIN_ROOT)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="output assets directory",
    )
    parser.add_argument("--model-id", type=int, default=0, help="variant index (baseN)")
    args = parser.parse_args()

    try:
        robotwin_dir = require_external_directory(
            args.robotwin_dir,
            environment_variable="STRETCH_MUJOCO_ROBOTWIN_ROOT",
            description="RoboTwin assets/objects root",
        )
    except (ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))

    object_names = [name.strip() for name in args.objects.split(",") if name.strip()]
    for i, name in enumerate(object_names, 1):
        meta = convert_object(name, robotwin_dir, args.out_dir, model_id=args.model_id)
        print(
            f"[{i}/{len(object_names)}] {name}: "
            f"h={meta['height'] * 100:.1f}cm mass={meta['mass_kg']:.2f}kg "
            f"textured={meta['textured']}"
        )
    print(f"done -> {args.out_dir}")


if __name__ == "__main__":
    main()
