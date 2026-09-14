"""Repository-local runtime paths and external data configuration.

Runtime artifacts belong outside the source package.  Callers may override the
defaults with environment variables; paths are resolved when requested so tests
and long-running processes can change their environment deliberately.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent
MODELS_ROOT = PACKAGE_ROOT / "models"


def configured_path(environment_variable: str, default: Path | None = None) -> Path | None:
    """Return an expanded absolute path from the environment or ``default``."""
    value = os.environ.get(environment_variable)
    path = Path(value).expanduser() if value else default
    return path.resolve() if path is not None else None


def output_root() -> Path:
    """Return the root for logs, recordings, evaluations, and generated reports."""
    path = configured_path("STRETCH_MUJOCO_OUTPUT_DIR", PROJECT_ROOT / "outputs")
    assert path is not None
    return path


def cache_root() -> Path:
    """Return the disposable conversion/cache root."""
    default = Path(tempfile.gettempdir()) / "stretch_mujoco"
    path = configured_path("STRETCH_MUJOCO_CACHE_DIR", default)
    assert path is not None
    return path


def require_external_directory(
    value: Path | None,
    *,
    environment_variable: str,
    description: str,
) -> Path:
    """Resolve and validate a CLI/environment-provided external data directory."""
    path = value or configured_path(environment_variable)
    if path is None:
        raise ValueError(
            f"{description} is required; pass its CLI option or set {environment_variable}"
        )
    path = path.expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"{description} directory does not exist: {path}")
    return path
