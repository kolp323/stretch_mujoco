#!/usr/bin/env python3
"""Link an NPC worktree to the main project's shared local asset store."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


_HUMANOID_ASSET_RELATIVE = Path("stretch_mujoco/models/assets/humanoid")
# Keep versioned README files in the parent directories local to each worktree.
# These child trees contain the large, ignored asset payloads.
_SHARED_DIRECTORIES = (Path("sources/npc"), Path("generated/animations"))
_MODEL_PAYLOAD_SUFFIXES = frozenset(
    {".glb", ".jpg", ".jpeg", ".msh", ".mtl", ".obj", ".png", ".stl"}
)


def link_shared_npc_assets(shared_project_root: Path, worktree_root: Path) -> list[Path]:
    """Link local source and generated assets without copying or overwriting data.

    Existing matching links are retained. A real directory or an unrelated link
    is rejected so a setup command can never silently replace local assets.
    """
    shared_assets = (shared_project_root.resolve() / _HUMANOID_ASSET_RELATIVE).resolve()
    worktree_assets = (worktree_root.resolve() / _HUMANOID_ASSET_RELATIVE).resolve()
    linked: list[Path] = []
    for relative_directory in _SHARED_DIRECTORIES:
        source = shared_assets / relative_directory
        target = worktree_assets / relative_directory
        if not source.is_dir():
            raise ValueError(f"Shared NPC asset directory is missing: {source}")
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.is_symlink():
            if target.resolve() != source.resolve():
                raise ValueError(f"NPC asset link points elsewhere: {target}")
            continue
        if target.exists():
            raise ValueError(f"Refusing to replace existing NPC asset directory: {target}")
        target.symlink_to(source, target_is_directory=True)
        linked.append(target)
    return linked + _link_shared_model_payloads(shared_project_root, worktree_root)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _link_shared_model_payloads(shared_project_root: Path, worktree_root: Path) -> list[Path]:
    """Link ignored office mesh/texture payloads while keeping XML local."""
    shared_models = (shared_project_root.resolve() / "stretch_mujoco/models").resolve()
    worktree_models = (worktree_root.resolve() / "stretch_mujoco/models").resolve()
    linked: list[Path] = []
    for source in sorted(shared_models.rglob("*")):
        if not source.is_file() or source.suffix.lower() not in _MODEL_PAYLOAD_SUFFIXES:
            continue
        relative = source.relative_to(shared_models)
        if relative.parts[:3] in {
            ("assets", "humanoid", "generated"),
            ("assets", "humanoid", "sources"),
        }:
            continue
        target = worktree_models / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.is_symlink():
            if target.resolve() != source.resolve():
                raise ValueError(f"Model payload link points elsewhere: {target}")
            continue
        if target.exists():
            if _sha256(target) != _sha256(source):
                raise ValueError(f"Model payload differs from shared source: {target}")
            continue
        target.symlink_to(source)
        linked.append(target)
    return linked


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shared-project-root", type=Path, required=True)
    parser.add_argument("--worktree-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    for path in link_shared_npc_assets(args.shared_project_root, args.worktree_root):
        print(path)


if __name__ == "__main__":
    main()
