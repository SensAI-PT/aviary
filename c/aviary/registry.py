"""In-memory cluster node registry with heartbeat lease and Phase 2 cohort state."""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any


DEFAULT_HEARTBEAT_SEC = float(os.environ.get("AVIARY_HEARTBEAT_SEC", "2"))
DEFAULT_HEARTBEAT_MISS = int(os.environ.get("AVIARY_HEARTBEAT_MISS", "3"))

CONTROL_IDLE_TIMEOUT_MS = int(DEFAULT_HEARTBEAT_SEC * 1000 * (DEFAULT_HEARTBEAT_MISS + 2))


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
    arch: str = ""
    engine_id: int | None = None
    usage: list[dict[str, Any]] = field(default_factory=list)
    costs: list[dict[str, Any]] = field(default_factory=list)
    profile: list[dict[str, Any]] = field(default_factory=list)
    expert_port: int = int(os.environ.get("AVIARY_EXPERT_PORT", "9003"))
    control_conn: Any = None
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
            "arch": self.arch,
            "engine_id": self.engine_id,
            "status": self.status,
            "inflight": self.inflight,
            "uptime_sec": now - self.registered_at,
            "last_heartbeat_age_sec": now - self.last_heartbeat,
            "hwinfo": self.hwinfo,
            "tiers": self.tiers,
            "emap": self.emap,
            "hits": self.hits,
            "hits_seq": self.hits_seq,
            "usage": self.usage,
            "costs": self.costs,
            "profile": self.profile,
            "expert_port": self.expert_port,
        }


@dataclass
class ClusterCohort:
    arch: str = ""
    model_id: str = ""
    engine_id: int | None = None
    emap_rows: int = 0
    emap_cols: int = 0


class NodeRegistry:
    def __init__(self, heartbeat_miss: int = DEFAULT_HEARTBEAT_MISS):
        self._lock = threading.Lock()
        self._nodes: dict[str, NodeRecord] = {}
        self._cohort: ClusterCohort | None = None
        self.heartbeat_miss = heartbeat_miss
        self._merged_usage: dict[tuple[int, int], int] = {}

    def _validate_cohort(self, model_id: str, payload: dict[str, Any]) -> None:
        arch = str(payload.get("arch") or "")
        engine_id = payload.get("engine_id")
        emap = payload.get("emap") or {}
        rows, cols = int(emap.get("rows") or 0), int(emap.get("cols") or 0)
        with self._lock:
            if self._cohort is None:
                self._cohort = ClusterCohort(arch=arch, model_id=model_id,
                                             engine_id=int(engine_id) if engine_id is not None else None,
                                             emap_rows=rows, emap_cols=cols)
                return
            if self._cohort.model_id and model_id != self._cohort.model_id:
                raise ValueError("COHORT_MODEL")
            if arch and self._cohort.arch and arch != self._cohort.arch:
                raise ValueError("COHORT_ARCH")
            if engine_id is not None and self._cohort.engine_id is not None:
                if int(engine_id) != self._cohort.engine_id:
                    raise ValueError("COHORT_ENGINE")
            if rows and self._cohort.emap_rows and rows != self._cohort.emap_rows:
                raise ValueError("COHORT_EMAP")
            if cols and self._cohort.emap_cols and cols != self._cohort.emap_cols:
                raise ValueError("COHORT_EMAP")

    def register(self, node_id: str, host: str, http_port: int, model_id: str,
                 payload: dict[str, Any], control_conn: Any = None) -> NodeRecord:
        self._validate_cohort(model_id, payload)
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
                arch=str(payload.get("arch") or ""),
                engine_id=int(payload["engine_id"]) if payload.get("engine_id") is not None else None,
                usage=list(payload.get("usage") or []),
                costs=list(payload.get("costs") or []),
                profile=list(payload.get("profile") or []),
                expert_port=int(payload.get("expert_port") or os.environ.get("AVIARY_EXPERT_PORT", "9003")),
                control_conn=control_conn,
            )
            self._nodes[node_id] = record
            return record

    def set_control_conn(self, node_id: str, conn: Any) -> None:
        with self._lock:
            record = self._nodes.get(node_id)
            if record:
                record.control_conn = conn

    def heartbeat(self, node_id: str, inflight: int, payload: dict[str, Any]) -> NodeRecord | None:
        with self._lock:
            record = self._nodes.get(node_id)
            if record is None or record.status == "dead":
                return None
            record.inflight = inflight
            record.last_heartbeat = time.time()
            record.missed_heartbeats = 0
            record.status = "healthy"
            for key in ("hwinfo", "tiers", "emap"):
                if payload.get(key):
                    setattr(record, key, payload[key])
            if "hits" in payload:
                record.hits = payload.get("hits") or ""
            if "hits_seq" in payload:
                record.hits_seq = int(payload["hits_seq"] or 0)
            if payload.get("usage"):
                record.usage = list(payload["usage"])
                for r in record.usage:
                    k = (int(r["layer"]), int(r["expert"]))
                    self._merged_usage[k] = self._merged_usage.get(k, 0) + int(r.get("count", 0))
            if payload.get("costs"):
                record.costs = list(payload["costs"])
            if payload.get("profile"):
                record.profile = list(payload["profile"])
            if payload.get("arch"):
                record.arch = str(payload["arch"])
            if payload.get("engine_id") is not None:
                record.engine_id = int(payload["engine_id"])
            return record

    def deregister(self, node_id: str) -> None:
        with self._lock:
            self._nodes.pop(node_id, None)

    def mark_dead(self, node_id: str) -> None:
        with self._lock:
            record = self._nodes.get(node_id)
            if record:
                record.status = "dead"
                record.control_conn = None

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
                        record.control_conn = None
                        evicted.append(node_id)
                elif age > DEFAULT_HEARTBEAT_SEC:
                    record.status = "stale"
        return evicted

    def healthy_nodes(self) -> list[NodeRecord]:
        with self._lock:
            return [n for n in self._nodes.values() if n.status == "healthy"]

    def pick_least_loaded(self, affinity_node: str | None = None) -> NodeRecord | None:
        nodes = self.healthy_nodes()
        if not nodes:
            return None
        if affinity_node:
            match = next((n for n in nodes if n.node_id == affinity_node), None)
            if match and match.inflight <= min(n.inflight for n in nodes) + 1:
                return match
        return min(nodes, key=lambda n: (n.inflight, n.last_heartbeat))

    def pick_with_affinity(self, hot_experts: set[tuple[int, int]] | None = None) -> NodeRecord | None:
        nodes = self.healthy_nodes()
        if not nodes:
            return None
        if not hot_experts:
            return self.pick_least_loaded()
        from aviary.placement import decode_emap

        def hot_hits(n: NodeRecord) -> int:
            inv = decode_emap(n.emap)
            return sum(1 for k in hot_experts if inv.get(k, {}).get("tier", 0) >= 1)

        hits = {n.node_id: hot_hits(n) for n in nodes}
        max_hits = max(hits.values())
        min_hits = min(hits.values())

        # Bootstrap cold executors: when one node has warmed hot experts and another
        # has almost none, route chat there so it can collect usage/ECOST and pin
        # experts. Otherwise affinity permanently locks all traffic on the first
        # warmed node (205 vs 0 resident hot experts → 100% on one machine).
        bootstrap_ratio = float(os.environ.get("AVIARY_ROUTE_BOOTSTRAP_RATIO", "0.1"))
        if max_hits > 0 and min_hits < max_hits * bootstrap_ratio:
            cold = [n for n in nodes if hits[n.node_id] <= min_hits + 1]
            if cold:
                return min(cold, key=lambda n: (n.inflight, n.last_heartbeat))

        def score(n: NodeRecord) -> tuple[int, float, int]:
            return (-hits[n.node_id], n.inflight, n.last_heartbeat)

        return min(nodes, key=score)

    def cluster_state(self) -> dict[str, Any]:
        with self._lock:
            nodes = [record.snapshot() for record in self._nodes.values()]
            cohort = self._cohort
        healthy = sum(1 for n in nodes if n["status"] == "healthy")
        return {
            "nodes": nodes,
            "healthy": healthy,
            "total": len(nodes),
            "cohort": {
                "arch": cohort.arch if cohort else "",
                "model_id": cohort.model_id if cohort else "",
                "engine_id": cohort.engine_id if cohort else None,
            },
            "merged_usage": [{"layer": l, "expert": e, "count": c}
                             for (l, e), c in sorted(self._merged_usage.items(),
                                                     key=lambda kv: -kv[1])[:256]],
        }

    def snapshot(self) -> dict[str, Any]:
        return self.cluster_state()

    def increment_inflight(self, node_id: str, delta: int = 1) -> None:
        with self._lock:
            record = self._nodes.get(node_id)
            if record and record.status == "healthy":
                record.inflight = max(0, record.inflight + delta)

    def control_connections(self) -> dict[str, Any]:
        with self._lock:
            return {nid: rec.control_conn for nid, rec in self._nodes.items()
                    if rec.control_conn and rec.status == "healthy"}
