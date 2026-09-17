"""Small transactional boundary for semantic state updates."""
from __future__ import annotations
from dataclasses import dataclass
from copy import deepcopy
import hashlib, json

@dataclass(frozen=True)
class SemanticCommit:
    execution_id: str
    before_hash: str
    after_hash: str
    event_sequence: int

class SemanticTransaction:
    def __init__(self, world, execution_id: str, event_sequence: int = 0) -> None:
        self.world, self.execution_id, self.event_sequence = world, execution_id, int(event_sequence)
        self._before = self._snapshot(); self._working = deepcopy(self._before); self._closed = False
    def _snapshot(self):
        return deepcopy(getattr(self.world, "_objects", getattr(self.world, "objects", {}))), deepcopy(getattr(self.world, "_relations", getattr(self.world, "relations", {})))
    @staticmethod
    def _hash(value) -> str:
        return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()
    def validate(self) -> None:
        if self._closed: raise RuntimeError("semantic_transaction_closed")
        if hasattr(self.world, "validate_invariants"): self.world.validate_invariants()
    def commit(self) -> SemanticCommit:
        self.validate(); self._closed = True
        after = self._snapshot()
        return SemanticCommit(self.execution_id, self._hash(self._before), self._hash(after), self.event_sequence)
    def abort(self) -> None:
        self._closed = True
