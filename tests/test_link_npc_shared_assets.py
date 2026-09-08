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


def _prepare_shared_root(root: Path) -> None:
    for directory in (
        "sources/npc",
        "generated/animations",
        "private/amass_intake",
        "private/smplx",
        "private/uv",
    ):
        (_asset_root(root) / directory).mkdir(parents=True)
    (root / "aaa_workspace/raw_resources").mkdir(parents=True)


def test_links_shared_sources_and_generated_once(tmp_path: Path) -> None:
    module = _link_module()
    shared, worktree = tmp_path / "main", tmp_path / "appearance"
    _prepare_shared_root(shared)

    linked = module.link_shared_npc_assets(shared, worktree)

    assert {path.relative_to(_asset_root(worktree)).as_posix() for path in linked} == {
        "sources/npc",
        "generated/animations",
        "private/amass_intake",
        "private/smplx",
        "private/uv",
    }
    assert (_asset_root(worktree) / "sources/npc").resolve() == _asset_root(shared) / "sources/npc"
    assert module.link_shared_npc_assets(shared, worktree) == []


def test_links_shared_generated_registration_manifest(tmp_path: Path) -> None:
    module = _link_module()
    shared, worktree = tmp_path / "main", tmp_path / "appearance"
    _prepare_shared_root(shared)
    registration = _asset_root(shared) / "generated/animations/manifest.json"
    registration.write_text('{"schema_version": 1}\n')

    module.link_shared_npc_assets(shared, worktree)

    linked_registration = _asset_root(worktree) / "generated/animations/manifest.json"
    assert linked_registration.resolve() == registration
    assert linked_registration.read_text() == registration.read_text()


def test_refuses_to_replace_existing_asset_directory(tmp_path: Path) -> None:
    module = _link_module()
    shared, worktree = tmp_path / "main", tmp_path / "appearance"
    _prepare_shared_root(shared)
    (_asset_root(worktree) / "sources/npc").mkdir(parents=True)

    with pytest.raises(ValueError, match="Refusing to replace"):
        module.link_shared_npc_assets(shared, worktree)


def test_links_shared_office_payloads_without_linking_xml(tmp_path: Path) -> None:
    module = _link_module()
    shared, worktree = tmp_path / "main", tmp_path / "appearance"
    _prepare_shared_root(shared)
    models = shared / "stretch_mujoco/models/stretch"
    models.mkdir(parents=True)
    (models / "hand_crush.png").write_bytes(b"png")
    (models / "robot.xml").write_text("<mujoco/>")

    linked = module.link_shared_npc_assets(shared, worktree)

    assert (worktree / "stretch_mujoco/models/stretch/hand_crush.png").is_symlink()
    assert not (worktree / "stretch_mujoco/models/stretch/robot.xml").exists()
    assert (worktree / "stretch_mujoco/models/stretch/hand_crush.png") in linked


def test_root_workspace_symlink_is_idempotent_and_does_not_block_models(
    tmp_path: Path,
) -> None:
    module = _link_module()
    shared, worktree = tmp_path / "main", tmp_path / "conversation"
    _prepare_shared_root(shared)
    raw_file = shared / "aaa_workspace/raw_resources/source.jsonl"
    raw_file.write_text("candidate\n")
    model_file = shared / "stretch_mujoco/models/stretch/hand_crush.png"
    model_file.parent.mkdir(parents=True)
    model_file.write_bytes(b"png")
    worktree.mkdir()
    (worktree / "aaa_workspace").symlink_to(shared / "aaa_workspace", target_is_directory=True)

    linked = module.link_shared_npc_assets(shared, worktree)

    assert (worktree / "aaa_workspace/raw_resources/source.jsonl").resolve() == raw_file
    assert (worktree / "stretch_mujoco/models/stretch/hand_crush.png").is_symlink()
    assert (worktree / "stretch_mujoco/models/stretch/hand_crush.png") in linked
    assert module.link_shared_npc_assets(shared, worktree) == []
