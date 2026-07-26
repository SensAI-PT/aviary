"""In-memory cluster node registry with heartbeat lease."""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any


DEFAULT_HEARTBEAT_SEC = float(os.environ.get("AVIARY_HEARTBEAT_SEC", "2"))
DEFAULT_HEARTBEAT_MISS = int(os.environ.get("AVIARY_HEARTBEAT_MISS", "3"))


@dataclass
class NodeRecord:
    node_id: str
    host: str
    http_port: int
    model_id: str
    endpoint: str
    registered_at: float = field(default_factory=time.time)
    last_heartbeat: float = field(default_factory=time.time)
    inflight: int = 0
    hwinfo: dict[str, Any] | None = None
    tiers: dict[str, Any] | None = None
    emap: dict[str, Any] | None = None
    hits: str = ""
    hits_seq: int = 0
    model_path: str = ""
    status: str = "healthy"
    missed_heartbeats: int = 0

    def snapshot(self) -> dict[str, Any]:
        now = time.time()
        return {
            "node_id": self.node_id,
            "endpoint": self.endpoint,
            "host": self.host,
            "http_port": self.http_port,
            "model_id": self.model_id,
            "model_path": self.model_path,
            "status": self.status,
            "inflight": self.inflight,
            "uptime_sec": now - self.registered_at,
            "last_heartbeat_age_sec": now - self.last_heartbeat,
            "hwinfo": self.hwinfo,
            "tiers": self.tiers,
            "emap": self.emap,
            "hits": self.hits,
            "hits_seq": self.hits_seq,
        }


class NodeRegistry:
    def __init__(self, heartbeat_miss: int = DEFAULT_HEARTBEAT_MISS):
        self._lock = threading.Lock()
        self._nodes: dict[str, NodeRecord] = {}
        self.heartbeat_miss = heartbeat_miss

    def register(self, node_id: str, host: str, http_port: int, model_id: str,
                 payload: dict[str, Any]) -> NodeRecord:
        host = payload.get("host") or host
        endpoint = f"http://{host}:{http_port}"
        with self._lock:
            if node_id in self._nodes and self._nodes[node_id].status != "dead":
                raise ValueError("DUPLICATE")
            record = NodeRecord(
                node_id=node_id,
                host=host,
                http_port=http_port,
                model_id=model_id,
                endpoint=endpoint,
                hwinfo=payload.get("hwinfo"),
                tiers=payload.get("tiers"),
                emap=payload.get("emap"),
                hits=payload.get("hits") or "",
                hits_seq=int(payload.get("hits_seq") or 0),
                model_path=str(payload.get("model_path") or ""),
            )
            self._nodes[node_id] = record
            return record

    def heartbeat(self, node_id: str, inflight: int, payload: dict[str, Any]) -> NodeRecord | None:
        with self._lock:
            record = self._nodes.get(node_id)
            if record is None or record.status == "dead":
                return None
            record.inflight = inflight
            record.last_heartbeat = time.time()
            record.missed_heartbeats = 0
            record.status = "healthy"
            if payload.get("hwinfo"):
                record.hwinfo = payload["hwinfo"]
            if payload.get("tiers"):
                record.tiers = payload["tiers"]
            if payload.get("emap"):
                record.emap = payload["emap"]
            if "hits" in payload:
                record.hits = payload.get("hits") or ""
            if "hits_seq" in payload:
                record.hits_seq = int(payload["hits_seq"] or 0)
            return record

    def deregister(self, node_id: str) -> None:
        with self._lock:
            self._nodes.pop(node_id, None)

    def mark_dead(self, node_id: str) -> None:
        with self._lock:
            record = self._nodes.get(node_id)
            if record:
                record.status = "dead"

    def tick_leases(self) -> list[str]:
        evicted = []
        with self._lock:
            for node_id, record in list(self._nodes.items()):
                if record.status == "dead":
                    continue
                age = time.time() - record.last_heartbeat
                if age > DEFAULT_HEARTBEAT_SEC * self.heartbeat_miss:
                    record.missed_heartbeats += 1
                    if record.missed_heartbeats >= self.heartbeat_miss:
                        record.status = "dead"
                        evicted.append(node_id)
                elif age > DEFAULT_HEARTBEAT_SEC:
                    record.status = "stale"
        return evicted

    def healthy_nodes(self) -> list[NodeRecord]:
        with self._lock:
            return [n for n in self._nodes.values() if n.status == "healthy"]

    def pick_least_loaded(self) -> NodeRecord | None:
        nodes = self.healthy_nodes()
        if not nodes:
            return None
        return min(nodes, key=lambda n: (n.inflight, n.last_heartbeat))

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            nodes = [record.snapshot() for record in self._nodes.values()]
        healthy = sum(1 for n in nodes if n["status"] == "healthy")
        return {"nodes": nodes, "healthy": healthy, "total": len(nodes)}

    def increment_inflight(self, node_id: str, delta: int = 1) -> None:
        with self._lock:
            record = self._nodes.get(node_id)
            if record and record.status == "healthy":
                record.inflight = max(0, record.inflight + delta)
