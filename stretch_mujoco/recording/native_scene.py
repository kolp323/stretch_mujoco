"""Build a multi-NPC variant of the project-native office scene for recording."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from copy import deepcopy
from pathlib import Path

MODELS_PATH = Path(__file__).resolve().parents[1] / "models"
NATIVE_OFFICE_SCENE = MODELS_PATH / "office_scene.xml"


def _rename_employee(body: ET.Element, employee_number: int) -> None:
    """Give a cloned ``humanoid_preview`` body stable multi-NPC identifiers."""
    employee_id = f"employee_{employee_number:02d}"
    frame_prefix = f"em{employee_number:02d}_frame_"
    body.set("name", f"{employee_id}_body")
    for element in body.iter():
        name = element.get("name")
        if name:
            name = name.replace("humanoid_preview_frame_", frame_prefix)
            name = name.replace("humanoid_preview", employee_id)
            name = name.replace("employee_01_handover", f"{employee_id}_handover")
            element.set("name", name)


def build_native_multi_npc_scene(
    destination: str | Path, *, employee_numbers: tuple[int, ...] = (1, 2)
) -> Path:
    """Write a real-office scene containing independently poseable NPCs.

    The source office remains authoritative for furniture, lighting, collision
    geometry, and Stretch.  Only its existing animated humanoid is cloned;
    this avoids maintaining a second, simplified office for video playback.
    """
    if not employee_numbers or any(number <= 0 for number in employee_numbers):
        raise ValueError("employee_numbers must contain positive employee numbers")
    if len(set(employee_numbers)) != len(employee_numbers):
        raise ValueError("employee_numbers must not contain duplicates")

    output = Path(destination).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    stretch_tree = ET.parse(MODELS_PATH / "stretch.xml")
    stretch_compiler = stretch_tree.getroot().find("compiler")
    if stretch_compiler is None:
        raise ValueError(
            f"Stretch model has no compiler declaration: {MODELS_PATH / 'stretch.xml'}"
        )
    stretch_compiler.set("assetdir", str(MODELS_PATH / "assets"))
    stretch_path = output.parent / "native_office_stretch.xml"
    ET.indent(stretch_tree, space="  ")
    stretch_tree.write(stretch_path, encoding="utf-8", xml_declaration=True)

    tree = ET.parse(NATIVE_OFFICE_SCENE)
    root = tree.getroot()
    # The generated file lives beside a recording, not beside office_scene.xml.
    # Resolve both includes and the asset directory before writing it so a
    # recording remains usable from any output directory.
    root.insert(0, ET.Element("compiler", {"assetdir": str(MODELS_PATH / "assets")}))
    for include in root.findall("include"):
        include_file = include.get("file")
        if include_file:
            include.set(
                "file",
                (
                    str(stretch_path)
                    if include_file == "stretch.xml"
                    else str(MODELS_PATH / include_file)
                ),
            )
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError(f"Native office scene is missing worldbody: {NATIVE_OFFICE_SCENE}")

    template = next(
        (body for body in worldbody.findall("body") if body.get("name") == "humanoid_preview"),
        None,
    )
    if template is None:
        raise ValueError(f"Native office scene has no humanoid_preview body: {NATIVE_OFFICE_SCENE}")

    insertion_index = list(worldbody).index(template)
    worldbody.remove(template)
    for offset, number in enumerate(employee_numbers):
        employee = deepcopy(template)
        _rename_employee(employee, number)
        worldbody.insert(insertion_index + offset, employee)

    ET.indent(tree, space="  ")
    tree.write(output, encoding="utf-8", xml_declaration=True)
    return output
