import json
import xml.etree.ElementTree as ET
from pathlib import Path


SCENE_ROOT = (
    Path(__file__).resolve().parents[1]
    / "stretch_mujoco"
    / "models"
    / "assets"
    / "home_scenes"
)


def test_generated_home_catalog_has_ten_distinct_scenes() -> None:
    catalog = json.loads((SCENE_ROOT / "catalog.json").read_text(encoding="utf-8"))

    assert catalog["scene_count"] == 10
    assert len(catalog["scenes"]) == 10
    assert len({scene["hssd_scene_id"] for scene in catalog["scenes"]}) == 10


def test_generated_home_metadata_does_not_capture_source_machine_paths() -> None:
    for manifest_path in sorted(SCENE_ROOT.glob("home_*.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert "hssd_root" not in manifest


def test_generated_home_xml_uses_portable_asset_paths() -> None:
    for xml_path in sorted(SCENE_ROOT.glob("home_*.xml")):
        root = ET.parse(xml_path).getroot()
        for node in root.findall(".//*[@file]"):
            assert not Path(node.attrib["file"]).is_absolute(), (
                f"{xml_path.name} contains absolute asset path {node.attrib['file']}"
            )
        for compiler in root.findall("compiler"):
            assetdir = compiler.get("assetdir")
            assert assetdir is None or not Path(assetdir).is_absolute(), (
                f"{xml_path.name} contains absolute assetdir {assetdir}"
            )
