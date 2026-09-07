import importlib.util
from pathlib import Path

import pytest


def _link_module():
    path = Path(__file__).parents[1] / "tools" / "link_npc_shared_assets.py"
    spec = importlib.util.spec_from_file_location("link_npc_shared_assets", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _asset_root(root: Path) -> Path:
    return root / "stretch_mujoco/models/assets/humanoid"


def test_links_shared_sources_and_generated_once(tmp_path: Path) -> None:
    module = _link_module()
    shared, worktree = tmp_path / "main", tmp_path / "appearance"
    for directory in ("sources/npc", "generated/animations"):
        (_asset_root(shared) / directory).mkdir(parents=True)

    linked = module.link_shared_npc_assets(shared, worktree)

    assert {path.relative_to(_asset_root(worktree)).as_posix() for path in linked} == {
        "sources/npc",
        "generated/animations",
    }
    assert (_asset_root(worktree) / "sources/npc").resolve() == _asset_root(shared) / "sources/npc"
    assert module.link_shared_npc_assets(shared, worktree) == []


def test_links_shared_generated_registration_manifest(tmp_path: Path) -> None:
    module = _link_module()
    shared, worktree = tmp_path / "main", tmp_path / "appearance"
    for directory in ("sources/npc", "generated/animations"):
        (_asset_root(shared) / directory).mkdir(parents=True)
    registration = _asset_root(shared) / "generated/animations/manifest.json"
    registration.write_text('{"schema_version": 1}\n')

    module.link_shared_npc_assets(shared, worktree)

    linked_registration = _asset_root(worktree) / "generated/animations/manifest.json"
    assert linked_registration.resolve() == registration
    assert linked_registration.read_text() == registration.read_text()


def test_refuses_to_replace_existing_asset_directory(tmp_path: Path) -> None:
    module = _link_module()
    shared, worktree = tmp_path / "main", tmp_path / "appearance"
    for directory in ("sources/npc", "generated/animations"):
        (_asset_root(shared) / directory).mkdir(parents=True)
    (_asset_root(worktree) / "sources/npc").mkdir(parents=True)

    with pytest.raises(ValueError, match="Refusing to replace"):
        module.link_shared_npc_assets(shared, worktree)


def test_links_shared_office_payloads_without_linking_xml(tmp_path: Path) -> None:
    module = _link_module()
    shared, worktree = tmp_path / "main", tmp_path / "appearance"
    for directory in ("sources/npc", "generated/animations"):
        (_asset_root(shared) / directory).mkdir(parents=True)
    models = shared / "stretch_mujoco/models/stretch"
    models.mkdir(parents=True)
    (models / "hand_crush.png").write_bytes(b"png")
    (models / "robot.xml").write_text("<mujoco/>")

    linked = module.link_shared_npc_assets(shared, worktree)

    assert (worktree / "stretch_mujoco/models/stretch/hand_crush.png").is_symlink()
    assert not (worktree / "stretch_mujoco/models/stretch/robot.xml").exists()
    assert (worktree / "stretch_mujoco/models/stretch/hand_crush.png") in linked
