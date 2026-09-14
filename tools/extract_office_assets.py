#!/usr/bin/env python3
"""Extract selected furniture assets from a Habitat scene into standalone XML.

Usage:
  .venv/bin/python tools/extract_office_assets.py

Reads the converted Habitat scene and copies requested objects (meeting
tables, chairs, whiteboards, displays, desks) into
stretch_mujoco/models/assets/office_assets/.
"""

from __future__ import annotations

import re
import shutil
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from typing import Optional

import mujoco

from stretch_mujoco.habitat_scene_gallery import (
    DEFAULT_SCENE_ID,
    load_habitat_scene_model,
    prepare_habitat_scene,
)
from stretch_mujoco.paths import cache_root

# ---------------------------------------------------------------------------
# Target objects — human label → (template_id, output_name, category_dir)
# ---------------------------------------------------------------------------
TARGETS: dict[str, tuple[str, str, str]] = {
    # --- chairs ----------------------------------------------------------
    "Cornell Swivel Office Chair": (
        "0070649630e7b2262ec0427c0e2035232240fb4d",
        "office_chair",
        "chairs",
    ),
    "Hooker Furniture Katherine Home Office Chair": (
        "17a04e30401165d49c85421b1997650d943ac75b",
        "home_office_chair",
        "chairs",
    ),
    "Stance Chair": (
        "2502dd408e62b2aa751080d4555d9b126f5a8d22",
        "stance_chair",
        "chairs",
    ),
    "Bucket Seat Dining Chair Grey Velvet": (
        "0a7074a49b0a1174916e5fdb66045c9b9de21ff6",
        "dining_chair",
        "chairs",
    ),
    # --- desks -----------------------------------------------------------
    "CB Desk 2400 Dressed": (
        "2e70722a552add06c4c800c85ee90a174fa05ec3",
        "cb_desk_2400",
        "desks",
    ),
    "CB Reception Desk Curved": (
        "7379d8877fb6d9f4f83e0b0207b44746d23a1860",
        "reception_desk",
        "desks",
    ),
    "Smartstudy Modular Adjustable ELEM": (
        "ad5387354cfc817d0731b12e2c354ab6e578be53",
        "adjustable_desk",
        "desks",
    ),
    "Custom Oak Sq Table Cranbrook Leg": (
        "2be31ee230714129948eda5af7d3b29777601dba",
        "oak_square_table",
        "desks",
    ),
    # --- meeting table ---------------------------------------------------
    "SmartStudy Sit-Stand Teaming Table": (
        "abbf8a4b081a056da7258544be89aabb37fd7a10",
        "teaming_table",
        "meeting_table",
    ),
    # --- whiteboard ------------------------------------------------------
    "TW Dry Wipe Module": (
        "b579dad917d549c04323882cc1c55f888b0253d2",
        "dry_wipe_module",
        "whiteboard",
    ),
    # --- displays --------------------------------------------------------
    "APPLE iMac 5K 27": (
        "2efebdcc1ba9514b0deb0dfb05951341432bd26a",
        "imac_27",
        "displays",
    ),
    "Apple iMAC Core 2 Duo 24": (
        "367fe59968558d7ca74557aacdc5add33d8c43d5",
        "imac_24",
        "displays",
    ),
}

# Destination root
DEST = Path(__file__).resolve().parent.parent / "stretch_mujoco" / "models" / "assets" / "office_assets"


# ---------------------------------------------------------------------------
def _body_geoms(model: mujoco.MjModel, body_id: int) -> list[int]:
    """Return geom ids belonging to *body_id* or its descendants."""
    geoms: list[int] = []
    for gid in range(model.ngeom):
        bid = int(model.geom_bodyid[gid])
        # walk up to see if *body_id* is an ancestor
        cur = bid
        while cur >= 0:
            if cur == body_id:
                geoms.append(gid)
                break
            cur = int(model.body_parentid[cur])
    return geoms


def _collect_asset_refs(xml_str: str, body_elem: ET.Element) -> dict[str, set[str]]:
    """Walk *body_elem* and collect mesh, material, texture refs."""
    refs: dict[str, set[str]] = {"mesh": set(), "material": set(), "texture": set()}

    ns = {"mesh", "material", "texture"}
    for elem in body_elem.iter():
        for attr, value in elem.attrib.items():
            if attr in ns:
                refs[attr].add(value)
            elif attr in {"file", "texture"}:
                refs["texture"].add(value)
    return refs


def _extract_assets_from_xml(
    root: ET.Element, ref_names: set[str], asset_type: str
) -> list[ET.Element]:
    """Return <asset> children whose ``name`` (mesh/material) or ``file``
    (texture) matches *ref_names*."""
    results: list[ET.Element] = []
    asset_elem = root.find("asset")
    if asset_elem is None:
        return results
    for child in asset_elem:
        name = child.get("name", "")
        file_attr = child.get("file", "")
        if asset_type == "texture":
            # texture refs are file paths, match by basename
            if Path(file_attr).name in ref_names or file_attr in ref_names:
                results.append(child)
        else:
            if name in ref_names:
                results.append(child)
                # Also collect textures referenced by materials
                tex = child.get("texture")
                if tex:
                    ref_names.add(tex)
    return results


# ---------------------------------------------------------------------------
def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)

    # Load scene
    scene = prepare_habitat_scene(
        scene_id=DEFAULT_SCENE_ID,
        cache_root=cache_root() / "habitat_scenes",
    )
    model = load_habitat_scene_model(scene)
    scene_xml = Path(scene.xml_path).read_text(encoding="utf-8")
    root = ET.fromstring(scene_xml)
    converted_dir = Path(scene.xml_path).parent / "converted"

    # Build asset name → file path mapping
    asset_map = _build_asset_map(root)

    # Build template_id → body_name mapping
    tid_to_bodies: dict[str, list[str]] = defaultdict(list)
    for i in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
        if name and name.startswith("object_"):
            parts = name.split("_", 2)
            if len(parts) >= 3:
                tid = re.sub(r"_part_\d+$", "", parts[2])
                tid_to_bodies[tid].append(name)

    # Process each target
    for label, (tid, out_name, category) in TARGETS.items():
        body_names = tid_to_bodies.get(tid, [])
        if not body_names:
            print(f"  SKIP {label}: template {tid} not found in scene")
            continue

        cat_dir = DEST / category
        cat_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"  {label}")
        print(f"  → {category}/{out_name}  ({len(body_names)} body parts)")
        print(f"{'='*60}")

        # Collect all body elements from XML
        body_elems: list[ET.Element] = []
        all_refs: dict[str, set[str]] = {"mesh": set(), "material": set(), "texture": set()}
        for body_name in body_names:
            body_elem = root.find(f".//body[@name='{body_name}']")
            if body_elem is not None:
                body_elems.append(body_elem)
                refs = _collect_asset_refs(scene_xml, body_elem)
                for k in all_refs:
                    all_refs[k] |= refs[k]

        if not body_elems:
            print("  WARNING: no body elements found in XML")
            continue

        # --- Collect referenced asset elements ---
        asset_elems: list[ET.Element] = []
        asset_elem = root.find("asset")
        name_to_asset: dict[str, ET.Element] = {}
        if asset_elem is not None:
            for child in asset_elem:
                n = child.get("name", "")
                if n:
                    name_to_asset[n] = child

        # 1. Meshes
        for mesh_name in all_refs["mesh"]:
            if mesh_name in name_to_asset:
                asset_elems.append(name_to_asset[mesh_name])

        # 2. Materials + collect texture refs
        tex_names: set[str] = set()
        for mat_name in all_refs["material"]:
            if mat_name in name_to_asset:
                mat_elem = name_to_asset[mat_name]
                asset_elems.append(mat_elem)
                tex_ref = mat_elem.get("texture", "")
                if tex_ref:
                    tex_names.add(tex_ref)

        # 3. Textures (by name)
        for tex_name in tex_names:
            if tex_name in name_to_asset:
                asset_elems.append(name_to_asset[tex_name])
                # Also track file copy via the texture's file attr
                all_refs["texture"].add(name_to_asset[tex_name].get("file", tex_name))
            else:
                all_refs["texture"].add(tex_name)

        # Copy mesh & texture files
        for ref_set, subdir in [
            (all_refs["mesh"], "meshes"),
            (all_refs["texture"], "textures"),
        ]:
            if not ref_set:
                continue
            (cat_dir / subdir).mkdir(parents=True, exist_ok=True)
            for ref in ref_set:
                # Resolve file path
                src = _resolve_file(ref, converted_dir, asset_map)
                if src and src.exists():
                    dst = cat_dir / subdir / src.name
                    if not dst.exists():
                        shutil.copy2(src, dst)
                        print(f"  copied: {subdir}/{src.name}")
                else:
                    print(f"  MISSING: {ref}")

        # ---- Build standalone XML ----
        out_xml_path = cat_dir / f"{out_name}.xml"
        _write_standalone_xml(out_xml_path, body_elems, asset_elems, out_name)

        print(f"  wrote: {out_xml_path}")

    # Write catalog
    _write_catalog(DEST)
    print(f"\nDone! Assets extracted to {DEST}")


def _resolve_file(ref: str, converted_dir: Path, asset_map: dict[str, str]) -> Optional[Path]:
    """Resolve a mesh/texture/material ref to an actual file path.

    *ref* may be a mesh/material/texture *name* (looked up in
    *asset_map*) or a direct file path.
    """
    # 1. Look up in asset_map (name → file path)
    if ref in asset_map:
        file_path = asset_map[ref]
        p = Path(file_path)
        if p.exists():
            return p
        # Try relative to converted_dir
        p2 = converted_dir / file_path.lstrip("/")
        if p2.exists():
            return p2

    # 2. Try as a direct file path
    p = Path(ref)
    if p.exists():
        return p
    p = converted_dir / ref.lstrip("/")
    if p.exists():
        return p

    # 3. Search in converted_dir by filename
    fname = Path(ref).name
    for candidate in converted_dir.rglob(fname):
        return candidate

    return None


def _build_asset_map(root: ET.Element) -> dict[str, str]:
    """Build a mapping from asset ``name`` → ``file`` path."""
    mapping: dict[str, str] = {}
    asset_elem = root.find("asset")
    if asset_elem is None:
        return mapping
    for child in asset_elem:
        name = child.get("name")
        file_attr = child.get("file")
        if name and file_attr:
            mapping[name] = file_attr
        # textures: file IS the path
        if child.tag == "texture" and file_attr:
            mapping[file_attr] = file_attr
            # Also map by basename
            mapping[Path(file_attr).name] = file_attr
    return mapping


def _write_standalone_xml(
    path: Path,
    body_elems: list[ET.Element],
    asset_elems: list[ET.Element],
    model_name: str,
) -> None:
    """Write a minimal MuJoCo XML containing only these bodies + assets.

    File paths in mesh/texture elements are rewritten to be relative to
    the XML file's parent directory.
    """
    def _fix_path(file_attr: str, *, subfolder: str = "") -> str:
        """Convert an absolute path to be relative to *xml_dir*.

        If *subfolder* is given (e.g. ``meshes``, ``textures``), the path
        is rewritten as ``subfolder/filename``.
        """
        src = Path(file_attr)
        fname = src.name
        if subfolder:
            return f"{subfolder}/{fname}"
        return fname

    lines = [
        f'<mujoco model="{model_name}">',
        '  <compiler angle="radian" balanceinertia="true"/>',
        "",
        "  <asset>",
    ]
    seen = set()
    for e in asset_elems:
        # Deduplicate by name or file
        key = e.get("name") or e.get("file") or ET.tostring(e, encoding="unicode")
        if key in seen:
            continue
        seen.add(key)
        # Clone and fix paths
        clone = ET.fromstring(ET.tostring(e, encoding="unicode"))
        # Determine subfolder based on element type
        sub = "textures" if clone.tag == "texture" else "meshes"
        for attr in ("file",):
            val = clone.get(attr)
            if val:
                clone.set(attr, _fix_path(val, subfolder=sub))
        raw = ET.tostring(clone, encoding="unicode")
        lines.append(f"    {raw}")
    lines.append("  </asset>")
    lines.append("")
    lines.append("  <worldbody>")

    for body_elem in body_elems:
        raw = ET.tostring(body_elem, encoding="unicode")
        lines.append(f"    {raw}")

    lines.append("  </worldbody>")
    lines.append("</mujoco>")

    path.write_text("\n".join(lines), encoding="utf-8")


def _write_catalog(dest: Path) -> None:
    """Write a README catalog listing all extracted assets."""
    lines = [
        "# Office Assets Catalog",
        "",
        "Extracted from HSSD Habitat scene `108294417_176709879`.",
        "",
        "| Category | Asset | File |",
        "|----------|-------|------|",
    ]
    for label, (tid, out_name, category) in sorted(TARGETS.items(), key=lambda x: (x[1][2], x[1][1])):
        fname = f"{category}/{out_name}.xml"
        lines.append(f"| {category} | {label} | [{out_name}.xml]({fname}) |")
    (dest / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
