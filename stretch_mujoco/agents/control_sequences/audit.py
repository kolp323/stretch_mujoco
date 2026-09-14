"""Append-only JSONL audit projection for control-sequence execution."""

from __future__ import annotations

import json
from dataclasses import is_dataclass
from pathlib import Path
from typing import Any


class SequenceAudit:
    def __init__(self, path: str | Path | None = None) -> None:
        self.path = None if path is None else Path(path)
        self.records: list[dict[str, Any]] = []

    def record(self, event: str, **details: Any) -> dict[str, Any]:
        value = {"event": event, **details}
        self.records.append(value)
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(value, default=self._json_default, sort_keys=True) + "\n")
        return value

    @staticmethod
    def _json_default(value: Any) -> Any:
        if is_dataclass(value) and not isinstance(value, type):
            return value.__dict__
        if hasattr(value, "value"):
            return value.value
        return str(value)
