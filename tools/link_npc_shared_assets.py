#!/usr/bin/env python3
"""Link an NPC worktree to the main project's shared local asset store."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


_HUMANOID_ASSET_RELATIVE = Path("stretch_mujoco/models/assets/humanoid")
# Keep versioned README files in the parent directories local to each worktree.
# These child trees contain the large, ignored asset payloads.
_SHARED_DIRECTORIES = (
    Path("sources/npc"),
    Path("generated/animations"),
    Path("private/amass_intake"),
    Path("private/smplx"),
    Path("private/uv"),
)
_MODEL_PAYLOAD_SUFFIXES = frozenset(
    {".glb", ".jpg", ".jpeg", ".msh", ".mtl", ".obj", ".png", ".stl"}
)


def link_shared_npc_assets(shared_project_root: Path, worktree_root: Path) -> list[Path]:
    """Link local source and generated assets without copying divergent data.

    Existing matching links are retained. Matching ignored generated-root files
    are adopted as links; a real directory, divergent file, or unrelated link is
    rejected so setup can never silently discard a distinct local asset.
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
    linked.extend(_link_shared_generated_files(shared_project_root, worktree_root))
    linked.extend(_link_shared_raw_inputs(shared_project_root, worktree_root))
    linked.extend(_link_shared_model_payloads(shared_project_root, worktree_root))
    return linked


def _link_shared_generated_files(shared_project_root: Path, worktree_root: Path) -> list[Path]:
    """Adopt matching ignored files at the generated-root as shared projections."""
    shared = shared_project_root.resolve() / _HUMANOID_ASSET_RELATIVE / "generated"
    target_root = worktree_root.resolve() / _HUMANOID_ASSET_RELATIVE / "generated"
    target_root.mkdir(parents=True, exist_ok=True)
    linked: list[Path] = []
    for source in sorted(path for path in shared.iterdir() if path.is_file()):
        if source.name == "README.md":
            continue
        target = target_root / source.name
        if target.is_symlink():
            if target.resolve() != source.resolve():
                raise ValueError(f"Generated-asset link points elsewhere: {target}")
            continue
        if target.exists():
            if not target.is_file() or _sha256(target) != _sha256(source):
                raise ValueError(f"Generated asset differs from shared source: {target}")
            target.unlink()
        target.symlink_to(source)
        linked.append(target)
    return linked


def _link_shared_raw_inputs(shared_project_root: Path, worktree_root: Path) -> list[Path]:
    """Expose raw staging unless the root workspace is already shared."""
    shared_raw = shared_project_root.resolve() / "aaa_workspace/raw_resources"
    worktree_raw = worktree_root.resolve() / "aaa_workspace/raw_resources"
    if not shared_raw.is_dir():
        raise ValueError(f"Shared raw-resource directory is missing: {shared_raw}")
    if worktree_raw.resolve() == shared_raw.resolve():
        return []
    worktree_raw.mkdir(parents=True, exist_ok=True)
    linked: list[Path] = []
    for source in sorted(shared_raw.iterdir()):
        if source.name == "README.md":
            continue
        target = worktree_raw / source.name
        if target.is_symlink():
            if target.resolve() != source.resolve():
                raise ValueError(f"Raw-resource link points elsewhere: {target}")
            continue
        if target.exists():
            raise ValueError(f"Refusing to replace existing raw-resource entry: {target}")
        target.symlink_to(source, target_is_directory=source.is_dir())
        linked.append(target)
    return linked


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
