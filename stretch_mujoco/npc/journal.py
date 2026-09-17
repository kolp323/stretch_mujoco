"""Durable command journal for NPC command replay and recovery."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from threading import RLock
from typing import Any, Iterable

@dataclass(frozen=True)
class JournalRecord:
    event: str
    command_id: str
    npc_id: str
    sequence: int
    payload_hash: str
    execution_epoch: int
    receipt: dict[str, Any] | None = None

class CommandJournal:
    def append(self, record: JournalRecord) -> None: raise NotImplementedError
    def records(self) -> tuple[JournalRecord, ...]: raise NotImplementedError
    def replay(self) -> dict[str, JournalRecord]:
        return {record.command_id: record for record in self.records()}

class JsonlCommandJournal(CommandJournal):
    def __init__(self, path: str | Path, *, execution_epoch: int = 0) -> None:
        self.path = Path(path)
        self.execution_epoch = int(execution_epoch)
        self._lock = RLock()
        self.path.parent.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def payload_hash(payload: object) -> str:
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
        return hashlib.sha256(encoded).hexdigest()

    def append(self, record: JournalRecord) -> None:
        with self._lock:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record.__dict__, sort_keys=True) + "\n")

    def records(self) -> tuple[JournalRecord, ...]:
        if not self.path.is_file(): return ()
        result: list[JournalRecord] = []
        with self._lock, self.path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip(): continue
                payload = json.loads(line)
                result.append(JournalRecord(**payload))
        return tuple(result)

    def append_command(self, command: object, *, event: str = "accepted", receipt: dict[str, Any] | None = None) -> JournalRecord:
        record = JournalRecord(event, command.command_id, command.npc_id, int(command.sequence), self.payload_hash(command.payload), self.execution_epoch, receipt)
        self.append(record)
        return record
