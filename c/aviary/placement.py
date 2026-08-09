"""Cluster-wide expert placement scheduler (Aviary Phase 2)."""

from __future__ import annotations

import os
import random
import statistics
import time
from dataclasses import dataclass, field
from typing import Any

from aviary.rpc_hist import RpcHistogram


DEFAULT_RECOMPUTE_SEC = float(os.environ.get("AVIARY_PLACEMENT_SEC", "4"))
EXPLORE_RATE = float(os.environ.get("AVIARY_EXPLORE_RATE", "0.01"))
MAX_PIN_PER_TICK = int(os.environ.get("AVIARY_PIN_BATCH", "32"))


def blocks_from_experts(experts: dict[str, str], node_ids: list[str]) -> dict[str, list[dict[str, int]]]:
    """Aggregate expert assignments into contiguous layer ranges per node."""
    by_node: dict[str, set[int]] = {nid: set() for nid in node_ids}
    for key, nid in experts.items():
        if ":" not in key:
            continue
        layer_s, _ = key.split(":", 1)
        try:
            by_node.setdefault(nid, set()).add(int(layer_s))
        except ValueError:
            continue
    out: dict[str, list[dict[str, int]]] = {}
    for nid, layers in by_node.items():
        if not layers:
            out[nid] = []
            continue
        sorted_layers = sorted(layers)
        ranges: list[dict[str, int]] = []
        start = prev = sorted_layers[0]
        for layer in sorted_layers[1:]:
            if layer == prev + 1:
                prev = layer
                continue
            ranges.append({"start": start, "end": prev})
            start = prev = layer
        ranges.append({"start": start, "end": prev})
        out[nid] = ranges
    return out


def decode_emap(emap: dict[str, Any] | None) -> dict[tuple[int, int], dict[str, int]]:
    """Decode EMAP hex into {(layer, expert): {tier, heat}}."""
    if not emap or not emap.get("map"):
        return {}
    rows, cols = int(emap["rows"]), int(emap["cols"])
    raw = emap["map"]
    out: dict[tuple[int, int], dict[str, int]] = {}
    for i in range(rows * cols):
        off = i * 2
        if off + 2 > len(raw):
            break
        byte = int(raw[off:off + 2], 16)
        layer, expert = i // cols, i % cols
        out[(layer, expert)] = {"tier": byte >> 6, "heat": byte & 0x3F}
    return out


@dataclass
class CostSample:
    load_us: float = 0.0
    exec_us: float = 0.0
    n: int = 0

    def update(self, load_us: float, exec_us: float) -> None:
        self.n += 1
        self.load_us = (self.load_us * (self.n - 1) + load_us) / self.n
        self.exec_us = (self.exec_us * (self.n - 1) + exec_us) / self.n


@dataclass
class PlacementPlan:
    """Cluster-wide routing table and per-node pin commands."""
    experts: dict[str, str] = field(default_factory=dict)  # "layer:eid" -> node_id
    expert_tiers: dict[str, int] = field(default_factory=dict)
    pin_commands: dict[str, list[tuple[int, int, int]]] = field(default_factory=dict)
    rpc_matrix_us: dict[str, dict[str, float]] = field(default_factory=dict)
    computed_at: float = 0.0


class PlacementScheduler:
    def __init__(self, recompute_sec: float = DEFAULT_RECOMPUTE_SEC):
        self.recompute_sec = recompute_sec
        self._costs: dict[str, dict[tuple[int, int, int], CostSample]] = {}
        self._rpc_us: dict[str, dict[str, float]] = {}
        self._usage: dict[tuple[int, int], int] = {}
        self._last_plan: PlacementPlan | None = None
        self._prev_experts: dict[str, str] = {}
        self._reassignments: int = 0
        self._median_exec: dict[int, float] = {0: 500_000.0, 1: 50_000.0, 2: 10_000.0}
        self.rpc_histogram = RpcHistogram()

    def ingest_costs(self, node_id: str, samples: list[dict[str, Any]]) -> None:
        bucket = self._costs.setdefault(node_id, {})
        for s in samples or []:
            key = (int(s["layer"]), int(s["expert"]), int(s.get("tier", 1)))
            cs = bucket.get(key) or CostSample()
            cs.update(float(s.get("load_us", 0)), float(s.get("exec_us", 0)))
            bucket[key] = cs
            tier = int(s.get("tier", 1))
            if cs.exec_us > 0:
                vals = [v.exec_us for b in self._costs.values() for (l, e, t), v in b.items() if t == tier]
                if vals:
                    self._median_exec[tier] = statistics.median(vals)

    def ingest_usage(self, records: list[dict[str, int]]) -> None:
        for r in records or []:
            key = (int(r["layer"]), int(r["expert"]))
            self._usage[key] = self._usage.get(key, 0) + int(r.get("count", 0))

    def record_rpc(self, src: str, dst: str, us: float) -> None:
        self._rpc_us.setdefault(src, {})[dst] = us
        self.rpc_histogram.record(us)

    def _exec_cost(self, node_id: str, layer: int, expert: int, tier: int) -> float:
        cs = self._costs.get(node_id, {}).get((layer, expert, tier))
        if cs and cs.exec_us > 0:
            return cs.exec_us
        best = min((v.exec_us for b in self._costs.values()
                    for (l, e, t), v in b.items() if l == layer and e == expert and v.exec_us > 0),
                   default=0.0)
        return best or self._median_exec.get(tier, 50_000.0)

    def _load_cost(self, node_id: str, layer: int, expert: int, tier: int) -> float:
        if tier >= 1:
            return 0.0
        cs = self._costs.get(node_id, {}).get((layer, expert, 0))
        if cs and cs.load_us > 0:
            return cs.load_us
        cs1 = self._costs.get(node_id, {}).get((layer, expert, 1))
        if cs1 and cs1.load_us > 0:
            return cs1.load_us
        return self._median_exec.get(0, 500_000.0)

    def _rpc_cost(self, src: str, dst: str) -> float:
        return self._rpc_us.get(src, {}).get(dst, 150_000.0)

    def recompute(self, nodes: list[dict[str, Any]], primary_hint: str | None = None) -> PlacementPlan:
        healthy = [n for n in nodes if n.get("status") == "healthy"]
        plan = PlacementPlan(computed_at=time.time())
        if not healthy:
            self._last_plan = plan
            return plan

        inventories = {n["node_id"]: decode_emap(n.get("emap")) for n in healthy}
        node_ids = [n["node_id"] for n in healthy]

        for n in healthy:
            self.ingest_costs(n["node_id"], n.get("costs") or [])
            self.ingest_usage(n.get("usage") or [])

        hot = sorted(self._usage.items(), key=lambda kv: -kv[1])[:256]
        primary = primary_hint or (min(healthy, key=lambda n: n.get("inflight", 0))["node_id"]
                                    if healthy else node_ids[0])

        for (layer, expert), freq in hot:
            if freq < 1:
                continue
            key = f"{layer}:{expert}"
            inv_p = inventories.get(primary, {}).get((layer, expert), {"tier": 0})
            tier_p = inv_p["tier"]
            cost_local = self._load_cost(primary, layer, expert, tier_p) + self._exec_cost(
                primary, layer, expert, max(tier_p, 1))

            best_node, best_cost = primary, cost_local
            for n in healthy:
                nid = n["node_id"]
                inv = inventories.get(nid, {}).get((layer, expert), {"tier": 0})
                tier = inv["tier"]
                if nid == primary:
                    c = cost_local
                else:
                    exec_t = max(tier, 1) if tier >= 1 else 1
                    c = self._rpc_cost(primary, nid) + self._exec_cost(nid, layer, expert, exec_t)
                    if tier < 1:
                        c += self._load_cost(nid, layer, expert, 0)
                if random.random() < EXPLORE_RATE and nid != primary:
                    c *= 0.5  # ε-greedy exploration toward unknown nodes
                if c < best_cost:
                    best_cost, best_node = c, nid

            plan.experts[key] = best_node
            plan.expert_tiers[key] = inventories.get(best_node, {}).get(
                (layer, expert), {"tier": 0})["tier"]

            # Pin hot experts on their preferred node when tier < RAM
            if best_node and plan.expert_tiers[key] < 1:
                plan.pin_commands.setdefault(best_node, []).append((layer, expert, 1))

        plan.rpc_matrix_us = {k: dict(v) for k, v in self._rpc_us.items()}
        if self._prev_experts:
            self._reassignments += sum(
                1 for key, nid in plan.experts.items()
                if self._prev_experts.get(key) != nid)
        self._prev_experts = dict(plan.experts)
        self._last_plan = plan
        return plan

    def build_agent_payload(self, node_id: str, nodes: list[dict[str, Any]],
                            expert_port: int = 9003) -> dict[str, Any]:
        """JSON written to AVIARY_PLACEMENT on each agent."""
        plan = self._last_plan or self.recompute(nodes)
        peers = {n["node_id"]: f"{n.get('host', '127.0.0.1')}:{expert_port}"
                 for n in nodes if n.get("status") == "healthy" and n["node_id"] != node_id}
        experts = {k: v for k, v in plan.experts.items() if v != node_id}
        return {"node_id": node_id, "peers": peers, "experts": experts}

    @property
    def last_plan(self) -> PlacementPlan | None:
        return self._last_plan

    def snapshot(self, node_ids: list[str] | None = None) -> dict[str, Any]:
        plan = self._last_plan
        experts = dict(plan.experts) if plan else {}
        ids = node_ids or sorted({nid for nid in experts.values()})
        return {
            "experts": experts,
            "expert_tiers": dict(plan.expert_tiers) if plan else {},
            "rpc_matrix_us": dict(plan.rpc_matrix_us) if plan else {},
            "computed_at": plan.computed_at if plan else 0.0,
            "usage_top": [{"layer": l, "expert": e, "count": c}
                             for (l, e), c in sorted(self._usage.items(),
                                                     key=lambda kv: -kv[1])[:32]],
            "blocks": blocks_from_experts(experts, ids),
            "reassignments": self._reassignments,
            "rpc_histogram": self.rpc_histogram.snapshot(),
        }
