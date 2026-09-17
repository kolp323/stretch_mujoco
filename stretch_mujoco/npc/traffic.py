from __future__ import annotations
from dataclasses import dataclass
from threading import RLock
@dataclass(frozen=True)
class RouteReservation:
    scene_id: str
    route_id: str
    npc_id: str
    route_revision: int = 0
class TrafficManager:
    def __init__(self, scene_id: str = "") -> None:
        self.scene_id=scene_id; self._owners={}; self._held={}; self._lock=RLock()
    def acquire(self, route_id: str, npc_id: str, route_revision: int=0):
        key=(route_id,int(route_revision))
        with self._lock:
            owner=self._owners.get(key)
            if owner not in (None,npc_id): return None
            r=RouteReservation(self.scene_id,route_id,npc_id,int(route_revision)); self._owners[key]=npc_id; self._held[npc_id]=r; return r
    def release(self, npc_id: str, route_id: str|None=None, route_revision: int=0):
        with self._lock:
            r=self._held.get(npc_id)
            if r is None or (route_id is not None and (r.route_id,r.route_revision)!=(route_id,int(route_revision))): return False
            self._owners.pop((r.route_id,r.route_revision),None); self._held.pop(npc_id,None); return True
    def snapshot(self):
        with self._lock: return tuple(self._held.values())