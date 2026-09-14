from __future__ import annotations

from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import pytest


SCENE_ROOT = (
    Path(__file__).resolve().parents[1] / "stretch_mujoco" / "models" / "assets" / "office_scenes"
)
SCENES = tuple(
    path for path in sorted(SCENE_ROOT.glob("office_[0-9][0-9]_*.xml")) if "_robot" not in path.stem
)


@pytest.mark.parametrize("scene_path", SCENES, ids=lambda path: path.stem)
def test_generated_office_scene_is_relocatable_and_loadable(scene_path: Path) -> None:
    root = ET.parse(scene_path).getroot()
    path_attributes = [
        value
        for node in root.iter()
        for attribute, value in node.attrib.items()
        if attribute in {"file", "assetdir"}
    ]

    assert path_attributes
    assert all(not Path(value).is_absolute() for value in path_attributes)
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    assert model.nbody > 1
