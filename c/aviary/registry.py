"""In-memory cluster node registry with heartbeat lease and Phase 2 cohort state."""

from __future__ import annotations

import os
import statistics
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from aviary.tiers_util import sanitize_tiers


DEFAULT_HEARTBEAT_SEC = float(os.environ.get("AVIARY_HEARTBEAT_SEC", "2"))
DEFAULT_HEARTBEAT_MISS = int(os.environ.get("AVIARY_HEARTBEAT_MISS", "3"))
COORD_SLOW_RATIO = float(os.environ.get("AVIARY_COORD_SLOW_RATIO", "1.5"))

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
    agent_inflight: int = 0
    proxy_inflight: int = 0
    hwinfo: dict[str, Any] | None = None
    tiers: dict[str, Any] | None = None
    tiers_config: dict[str, Any] | None = None
    emap: dict[str, Any] | None = None
    hits: str = ""
    hits_seq: int = 0
    model_path: str = ""
    arch: str = ""
    engine_id: int | None = None
    usage: list[dict[str, Any]] = field(default_factory=list)
    costs: list[dict[str, Any]] = field(default_factory=list)
    profile: list[dict[str, Any]] = field(default_factory=list)
    gpu_tier: dict[str, Any] | None = None
    expert_port: int = int(os.environ.get("AVIARY_EXPERT_PORT", "9003"))
    control_conn: Any = None
    control_lock: threading.Lock = field(default_factory=threading.Lock)
    status: str = "healthy"
    missed_heartbeats: int = 0

    def snapshot(self, roles: dict[str, tuple[bool, str]] | None = None) -> dict[str, Any]:
        now = time.time()
        eligible, reason = (roles or {}).get(self.node_id, (True, ""))
        tc = self.tiers_config or {}
        t = self.tiers or {}
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
            "tiers_config": self.tiers_config,
            "ram_config_gb": tc.get("ram_gb"),
            "vram_config_gb": tc.get("vram_gb"),
            "ram_occ_gb": t.get("ram_gb"),
            "vram_occ_gb": t.get("vram_gb"),
            "gpu_tier": self.gpu_tier,
            "coordinator_eligible": eligible,
            "donor_only_reason": reason,
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
                tiers=sanitize_tiers(payload.get("tiers")) or payload.get("tiers"),
                tiers_config=dict(payload.get("tiers_config") or {}),
                emap=payload.get("emap"),
                hits=payload.get("hits") or "",
                hits_seq=int(payload.get("hits_seq") or 0),
                model_path=str(payload.get("model_path") or ""),
                arch=str(payload.get("arch") or ""),
                engine_id=int(payload["engine_id"]) if payload.get("engine_id") is not None else None,
                usage=list(payload.get("usage") or []),
                costs=list(payload.get("costs") or []),
                profile=list(payload.get("profile") or []),
                gpu_tier=dict(payload["gpu_tier"]) if payload.get("gpu_tier") else None,
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
            record.agent_inflight = inflight
            record.inflight = record.agent_inflight + record.proxy_inflight
            record.last_heartbeat = time.time()
            record.missed_heartbeats = 0
            record.status = "healthy"
            for key in ("hwinfo", "tiers", "emap", "tiers_config", "gpu_tier"):
                if payload.get(key):
                    value = payload[key]
                    if key == "tiers":
                        value = sanitize_tiers(value) or value
                    setattr(record, key, value)
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

    @staticmethod
    def _load_key(n: NodeRecord) -> tuple[float, float, float, str]:
        # Prefer faster hardware when load/penalty ties (avoid UUID lottery).
        return (n.inflight + NodeRegistry._slow_penalty(n),
                -NodeRegistry._speed_hint(n), -n.last_heartbeat, n.node_id)

    @staticmethod
    def _slow_penalty(n: NodeRecord) -> float:
        penalty = 0.0
        for turn in (n.profile or [])[-4:]:
            penalty += float(turn.get("expert_wait_s", 0)) * 2.0
            penalty += float(turn.get("wall_s", 0)) * 0.5
        for cost in (n.costs or [])[-32:]:
            penalty += float(cost.get("exec_us", 0)) / 1_000_000
        return penalty

    @staticmethod
    def _emap_resident_count(n: NodeRecord) -> int:
        from aviary.placement import decode_emap
        inv = decode_emap(n.emap)
        return sum(1 for v in inv.values() if v.get("tier", 0) >= 1)

    @staticmethod
    def _speed_hint(n: NodeRecord) -> float:
        hw = n.hwinfo or {}
        cores = float(hw.get("cores") or 0)
        ram = float(hw.get("ram_avail_gb") or hw.get("ram_total_gb") or 0)
        tiers = n.tiers or {}
        ram_budget = float(tiers.get("ram_gb") or 0)
        if ram_budget > 1024:
            ram_budget = 0.0
        return cores * 10.0 + ram + ram_budget - NodeRegistry._slow_penalty(n)

    @staticmethod
    def _median_ram_exec_us(n: NodeRecord) -> float | None:
        vals = [float(c.get("exec_us", 0)) for c in (n.costs or [])
                if int(c.get("tier", 1)) == 1 and float(c.get("exec_us", 0)) > 0]
        return statistics.median(vals) if vals else None

    @staticmethod
    def _has_tier_signal(n: NodeRecord) -> bool:
        tc = n.tiers_config or {}
        t = n.tiers or {}
        return bool(float(tc.get("ram_gb") or 0) or float(tc.get("vram_gb") or 0)
                    or float(t.get("ram_gb") or 0) or float(t.get("vram_gb") or 0))

    @staticmethod
    def _coord_score(n: NodeRecord) -> float:
        exec_us = NodeRegistry._median_ram_exec_us(n)
        exec_bonus = (1_000_000.0 / exec_us) if exec_us and exec_us > 0 else 0.0
        return NodeRegistry._speed_hint(n) + exec_bonus

    def node_roles(self, nodes: list[NodeRecord] | None = None) -> dict[str, tuple[bool, str]]:
        """Return coordinator_eligible and donor_only reason per node."""
        healthy = nodes if nodes is not None else self.healthy_nodes()
        execs = {n.node_id: self._median_ram_exec_us(n) for n in healthy}
        leader_exec = min((v for v in execs.values() if v), default=None)
        hw = {n.node_id: self._speed_hint(n) for n in healthy if (n.hwinfo or {}).get("cores")}
        leader_hw = max(hw.values(), default=0.0)
        roles: dict[str, tuple[bool, str]] = {}
        for n in healthy:
            if not self._has_tier_signal(n) and not (n.costs or []):
                roles[n.node_id] = (False, "missing_tiers")
                continue
            if leader_hw and n.node_id in hw and hw[n.node_id] * COORD_SLOW_RATIO < leader_hw:
                roles[n.node_id] = (False, "slow_hw")
                continue
            exec_us = execs.get(n.node_id)
            if leader_exec and exec_us and exec_us > leader_exec * COORD_SLOW_RATIO:
                roles[n.node_id] = (False, "slow_ram_exec")
                continue
            wall = max((float(t.get("wall_s", 0)) for t in (n.profile or [])[-2:]), default=0.0)
            if wall > 5.0 and leader_exec and exec_us and exec_us > leader_exec:
                roles[n.node_id] = (False, "slow_profile")
                continue
            roles[n.node_id] = (True, "")
        return roles

    def coordinator_nodes(self) -> list[NodeRecord]:
        healthy = self.healthy_nodes()
        roles = self.node_roles(healthy)
        coords = [n for n in healthy if roles.get(n.node_id, (True, ""))[0]]
        return coords or healthy

    def _pick_coordinator_from(self, nodes: list[NodeRecord],
                               affinity_node: str | None = None) -> NodeRecord | None:
        if not nodes:
            return None
        if affinity_node:
            match = next((n for n in nodes if n.node_id == affinity_node), None)
            if match and match.inflight <= min(n.inflight for n in nodes) + 1:
                return match
        return min(nodes, key=lambda n: (n.inflight + self._slow_penalty(n),
                                          -self._coord_score(n), -n.last_heartbeat, n.node_id))

    def pick_coordinator(self, affinity_node: str | None = None) -> NodeRecord | None:
        """Pick chat coordinator: fastest eligible node, never EMAP-warmth."""
        healthy = self.healthy_nodes()
        roles = self.node_roles(healthy)
        coords = [n for n in healthy if roles.get(n.node_id, (True, ""))[0]]
        return self._pick_coordinator_from(coords or healthy, affinity_node)

    def pick_least_loaded(self, affinity_node: str | None = None) -> NodeRecord | None:
        return self.pick_coordinator(affinity_node)

    def pick_with_affinity(self, hot_experts: set[tuple[int, int]] | None = None) -> NodeRecord | None:
        """Chat routing: coordinator only. hot_experts does not move the conversation."""
        del hot_experts
        return self.pick_coordinator()

    def cluster_state(self) -> dict[str, Any]:
        with self._lock:
            healthy = [n for n in self._nodes.values() if n.status == "healthy"]
            roles = self.node_roles(healthy)
            nodes = [record.snapshot(roles) for record in self._nodes.values()]
            cohort = self._cohort
            coords = [n for n in healthy if roles.get(n.node_id, (True, ""))[0]]
            coord = self._pick_coordinator_from(coords or healthy)
        healthy_count = sum(1 for n in nodes if n["status"] == "healthy")
        return {
            "nodes": nodes,
            "healthy": healthy_count,
            "total": len(nodes),
            "coordinator_id": coord.node_id if coord else "",
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
                record.proxy_inflight = max(0, record.proxy_inflight + delta)
                record.inflight = record.agent_inflight + record.proxy_inflight

    def clear_usage(self) -> None:
        with self._lock:
            self._merged_usage.clear()
            for record in self._nodes.values():
                record.usage = []

    def control_connections(self) -> dict[str, Any]:
        with self._lock:
            return {nid: rec.control_conn for nid, rec in self._nodes.items()
                    if rec.control_conn and rec.status == "healthy"}

    def control_send(self, node_id: str, data: bytes) -> None:
        """Serialize writes on the node's control socket (handler + placement ticker)."""
        with self._lock:
            record = self._nodes.get(node_id)
            if not record or not record.control_conn or record.status != "healthy":
                raise OSError("control connection unavailable")
            conn, lock = record.control_conn, record.control_lock
        with lock:
            conn.sendall(data)

    def control_write_line(self, node_id: str, line: str) -> None:
        self.control_send(node_id, (line.rstrip("\n") + "\n").encode("utf-8"))
