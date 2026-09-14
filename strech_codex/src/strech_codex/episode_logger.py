"""Persist one Codex live task as a JSON episode."""

from __future__ import annotations

import json
import re
import time
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from stretch_mujoco.paths import output_root

DEFAULT_LOG_DIR = output_root() / "strech_codex" / "episodes"
_SECRET_WORDS = ("api_key", "apikey", "password", "secret", "token")


class EpisodeLogger:
    def __init__(self, task: str, log_dir: Path | None = None) -> None:
        self._started = time.perf_counter()
        self._steps: dict[str, dict[str, Any]] = {}
        started_at = _now()
        timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
        name = re.sub(r"[^\w.-]+", "_", task, flags=re.UNICODE).strip("._")[:80] or "task"
        episode_name = f"{name}_{timestamp}"
        directory = (log_dir or DEFAULT_LOG_DIR) / episode_name
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / f"{episode_name}.json"
        self.video_path = self.path.with_suffix(".mp4")
        self.data: dict[str, Any] = {
            "task": task,
            "started_at": started_at,
            "completed_at": None,
            "duration_ms": None,
            "status": "running",
            "skills": [],
            "steps": [],
            "artifacts": [],
            "final_response": None,
            "error": None,
        }
        self._write()

    def record(self, event: Any) -> None:
        method = getattr(event, "method", "")
        payload = getattr(event, "payload", None)
        if method not in {"item/started", "item/completed"}:
            return
        item = getattr(payload, "item", None)
        root = getattr(item, "root", None)
        if getattr(root, "type", None) != "mcpToolCall":
            return

        item_id = str(getattr(root, "id", "unknown"))
        if method == "item/started":
            step = {
                "id": item_id,
                "type": "mcp_tool",
                "server": getattr(root, "server", None),
                "tool": getattr(root, "tool", None),
                "arguments": _jsonable(getattr(root, "arguments", None)),
                "started_at": _timestamp(payload, "started_at_ms"),
                "completed_at": None,
                "duration_ms": None,
                "status": "inProgress",
                "result": None,
                "error": None,
            }
            self._steps[item_id] = step
            self.data["steps"].append(step)
        else:
            step = self._steps.get(item_id)
            if step is None:
                step = {"id": item_id, "type": "mcp_tool"}
                self._steps[item_id] = step
                self.data["steps"].append(step)
            step.update(
                {
                    "server": getattr(root, "server", None),
                    "tool": getattr(root, "tool", None),
                    "arguments": _jsonable(getattr(root, "arguments", None)),
                    "completed_at": _timestamp(payload, "completed_at_ms"),
                    "duration_ms": getattr(root, "duration_ms", None),
                    "status": _enum_value(getattr(root, "status", None)),
                    "result": _jsonable(getattr(root, "result", None)),
                    "error": _jsonable(getattr(root, "error", None)),
                }
            )
        self._write()

    def finish(self, final_response: str = "", error: BaseException | None = None) -> None:
        artifacts = []
        if self.video_path.is_file() and self.video_path.stat().st_size > 0:
            artifacts.append(
                {
                    "type": "video/mp4",
                    "path": str(self.video_path),
                    "size_bytes": self.video_path.stat().st_size,
                }
            )
        evidence_root = self.path.parent / "evidence"
        for path in sorted(evidence_root.rglob("*")) if evidence_root.is_dir() else []:
            if path.is_file():
                artifacts.append(
                    {
                        "type": _artifact_type(path),
                        "path": str(path),
                        "size_bytes": path.stat().st_size,
                    }
                )
        tool_failed = any(step.get("status") == "failed" for step in self.data["steps"])
        self.data.update(
            {
                "completed_at": _now(),
                "duration_ms": round((time.perf_counter() - self._started) * 1000, 3),
                "status": "failed" if error or tool_failed else "completed",
                "artifacts": artifacts,
                "final_response": final_response,
                "error": str(error) if error else None,
            }
        )
        self._write()

    def _write(self) -> None:
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(self.data, ensure_ascii=False, indent=2, default=str) + "\n",
            encoding="utf-8",
        )
        temporary.replace(self.path)


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="milliseconds")


def _artifact_type(path: Path) -> str:
    return {".png": "image/png", ".npy": "application/x-npy", ".json": "application/json"}.get(
        path.suffix.lower(), "application/octet-stream"
    )


def _timestamp(value: Any, field: str) -> str | None:
    milliseconds = getattr(value, field, None)
    if milliseconds is None:
        return _now()
    return (
        datetime.fromtimestamp(milliseconds / 1000).astimezone().isoformat(timespec="milliseconds")
    )


def _enum_value(value: Any) -> Any:
    return value.value if isinstance(value, Enum) else value


def _jsonable(value: Any, key: str = "") -> Any:
    if any(word in key.lower() for word in _SECRET_WORDS):
        return "[REDACTED]"
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json", by_alias=True)
    elif isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(k): _jsonable(v, str(k)) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, str) and len(value) > 10_000:
        return value[:10_000] + "... [truncated]"
    return value
