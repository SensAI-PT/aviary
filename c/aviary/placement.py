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
LAYER_COHERENT = os.environ.get("AVIARY_LAYER_COHERENT", "0") not in ("0", "")
LAYER_BLOCKS = os.environ.get("AVIARY_LAYER_BLOCKS", "1") not in ("0", "")
VRAM_SLOW_RATIO = float(os.environ.get("AVIARY_VRAM_SLOW_RATIO", "1.25"))
BLOCK_MOVE_PCT = float(os.environ.get("AVIARY_BLOCK_MOVE_PCT", "0.15"))
MIN_TIER_SAMPLES = int(os.environ.get("AVIARY_MIN_TIER_SAMPLES", "8"))
TIER_PROBE = int(os.environ.get("AVIARY_TIER_PROBE", "8"))


def blocks_from_planned(planned: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Group planned layer blocks by owning node."""
    out: dict[str, list[dict[str, Any]]] = {}
    for block in planned or []:
        nid = block["node_id"]
        out.setdefault(nid, []).append({
            "start": block["start"],
            "end": block["end"],
            "max_tier": block.get("max_tier", 2),
        })
    return out


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


def layer_coherence_stats(experts: dict[str, str]) -> dict[str, Any]:
    """Share of hot expert assignments on each layer's dominant node."""
    by_layer: dict[int, dict[str, int]] = {}
    for key, nid in experts.items():
        if ":" not in key:
            continue
        layer = int(key.split(":", 1)[0])
        counts = by_layer.setdefault(layer, {})
        counts[nid] = counts.get(nid, 0) + 1
    layers = []
    weighted = total = 0
    multi = 0
    for layer in sorted(by_layer):
        counts = by_layer[layer]
        assigned = sum(counts.values())
        dominant = max(counts.items(), key=lambda kv: kv[1])[0] if counts else ""
        share = counts[dominant] / assigned if assigned and dominant else 0.0
        if len(counts) > 1:
            multi += 1
        weighted += share * assigned
        total += assigned
        layers.append({"layer": layer, "dominant": dominant, "share": round(share, 3),
                       "assigned": assigned, "nodes": len(counts)})
    return {
        "score": round(weighted / total, 3) if total else 0.0,
        "multi_node_layers": multi,
        "total_layers": len(layers),
        "layers": layers[:64],
    }


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


def _layer_owners_from_plan(plan: "PlacementPlan | None") -> dict[int, str]:
    if not plan:
        return {}
    owners: dict[int, str] = {}
    for block in plan.planned_blocks or []:
        for layer in range(int(block["start"]), int(block["end"]) + 1):
            owners[layer] = block["node_id"]
    for key, nid in (plan.experts or {}).items():
        if ":" not in key:
            continue
        owners[int(key.split(":", 1)[0])] = nid
    return owners


def _expert_cost(sched: "PlacementScheduler", primary: str, nid: str, layer: int, expert: int,
                 inventories: dict[str, dict], tier_hint: int | None = None) -> float:
    inv = inventories.get(nid, {}).get((layer, expert), {"tier": 0})
    tier = tier_hint if tier_hint is not None else inv["tier"]
    if nid == primary:
        tp = tier
        return sched._load_cost(primary, layer, expert, tp) + sched._exec_cost(primary, layer, expert, max(tp, 1))
    exec_t = max(tier, 1) if tier >= 1 else 1
    c = sched._rpc_cost(primary, nid) + sched._exec_cost(nid, layer, expert, exec_t)
    if tier < 1:
        c += sched._load_cost(nid, layer, expert, 0)
    return c


def _node_median_exec(sched: "PlacementScheduler", node_id: str, tier: int) -> float:
    vals = [v.exec_us for (_, _, t), v in sched._costs.get(node_id, {}).items()
            if t == tier and v.exec_us > 0]
    if vals:
        return statistics.median(vals)
    return sched._median_exec.get(tier, 50_000.0)


def _node_tier_sample_count(sched: "PlacementScheduler", node_id: str, tier: int) -> int:
    return sum(1 for (_, _, t), v in sched._costs.get(node_id, {}).items()
               if t == tier and v.exec_us > 0)


def _cluster_median_exec(sched: "PlacementScheduler", tier: int) -> float:
    vals = [v.exec_us for b in sched._costs.values()
            for (_, _, t), v in b.items() if t == tier and v.exec_us > 0]
    if len(vals) >= MIN_TIER_SAMPLES:
        return statistics.median(vals)
    return sched._median_exec.get(tier, 50_000.0)


def _vram_residency_frac(node: dict[str, Any]) -> float:
    """Fraction of EMAP cells currently marked VRAM (tier 2)."""
    inv = decode_emap(node.get("emap"))
    if not inv:
        tiers = node.get("tiers") or {}
        vram = int(tiers.get("vram", 0) or 0)
        ram = int(tiers.get("ram", 0) or 0)
        disk = int(tiers.get("disk", 0) or 0)
        total = vram + ram + disk
        return (vram / total) if total else 0.0
    n = len(inv)
    if not n:
        return 0.0
    return sum(1 for c in inv.values() if c.get("tier", 0) >= 2) / n


def preferred_max_tier(sched: "PlacementScheduler", node: dict[str, Any]) -> tuple[int, bool]:
    """Return (max_tier, vram_slow) for a node from ECOST + capacity.

    If the node is already fully on GPU, local RAM samples may be missing — compare
    VRAM exec against cluster/prior RAM medians so demotion can still fire.
    """
    nid = node["node_id"]
    tiers = node.get("tiers") or {}
    has_vram = int(tiers.get("vram", 0)) > 0 or float(tiers.get("vram_gb", 0) or 0) > 0
    if not has_vram and _vram_residency_frac(node) < 0.05:
        return 1, False

    n1 = _node_tier_sample_count(sched, nid, 1)
    n2 = _node_tier_sample_count(sched, nid, 2)
    if n2 < MIN_TIER_SAMPLES:
        # No measured GPU cost yet — keep VRAM allowed until we learn.
        return (2 if has_vram or _vram_residency_frac(node) > 0 else 1), False

    exec2 = _node_median_exec(sched, nid, 2)
    if n1 >= MIN_TIER_SAMPLES:
        exec1 = _node_median_exec(sched, nid, 1)
    else:
        exec1 = _cluster_median_exec(sched, 1)

    vram_slow = exec1 > 0 and exec2 > exec1 * VRAM_SLOW_RATIO
    # When the heatmap is almost all VRAM, also compare against the RAM prior so we
    # do not wait forever for local tier-1 samples that cannot appear until we demote.
    if not vram_slow and n1 < MIN_TIER_SAMPLES and _vram_residency_frac(node) >= 0.5:
        prior = sched._median_exec.get(1, 50_000.0)
        vram_slow = prior > 0 and exec2 > prior * VRAM_SLOW_RATIO
    if vram_slow:
        return 1, True
    return 2, False


def _node_median_load(sched: "PlacementScheduler", node_id: str, tier: int = 0) -> float:
    vals = [v.load_us for (_, _, t), v in sched._costs.get(node_id, {}).items()
            if t == tier and v.load_us > 0]
    if vals:
        return statistics.median(vals)
    return sched._median_exec.get(0, 500_000.0)


def _layer_exec_us(sched: "PlacementScheduler", node_id: str, max_tier: int) -> float:
    opts = [_node_median_exec(sched, node_id, t) for t in range(1, min(max_tier, 2) + 1)]
    if max_tier < 1:
        opts.append(_node_median_exec(sched, node_id, 0) + _node_median_load(sched, node_id))
    return min(opts) if opts else _node_median_exec(sched, node_id, 1)


def _layer_cost_on_node(sched: "PlacementScheduler", primary: str, nid: str, layer: int,
                        layer_freq: int, inventories: dict[str, dict], max_tier: int,
                        prev_owner: str | None,
                        layer_experts: list[tuple[int, int]] | None = None) -> float:
    if layer_experts:
        cost = sum(_expert_cost(sched, primary, nid, layer, expert, inventories, None)
                   for expert, _ in layer_experts)
    else:
        cost = layer_freq * _layer_exec_us(sched, nid, max_tier)
    if prev_owner and prev_owner != nid:
        cost += sched._rpc_cost(primary, nid)
    if random.random() < EXPLORE_RATE and nid != primary:
        cost *= 0.5
    return cost


def _merge_blocks(layer_owners: dict[int, str], layer_max_tier: dict[int, int]) -> list[dict[str, Any]]:
    if not layer_owners:
        return []
    layers = sorted(layer_owners)
    blocks: list[dict[str, Any]] = []
    start = prev = layers[0]
    nid = layer_owners[start]
    max_t = layer_max_tier.get(start, 2)
    for layer in layers[1:]:
        if layer == prev + 1 and layer_owners[layer] == nid:
            prev = layer
            max_t = max(max_t, layer_max_tier.get(layer, 2))
            continue
        blocks.append({"start": start, "end": prev, "node_id": nid, "max_tier": max_t})
        start = prev = layer
        nid = layer_owners[layer]
        max_t = layer_max_tier.get(layer, 2)
    blocks.append({"start": start, "end": prev, "node_id": nid, "max_tier": max_t})
    return blocks


def _emit_tier_commands(plan: PlacementPlan, inventories: dict[str, dict],
                        layer_max_tier: dict[int, int], hot: list[tuple[tuple[int, int], int]],
                        probe_demote: dict[str, int] | None = None) -> None:
    plan.pin_commands = {}
    plan.evict_commands = {}
    for (layer, expert), freq in hot:
        if freq < 1:
            continue
        key = f"{layer}:{expert}"
        owner = plan.experts.get(key)
        if not owner:
            continue
        max_tier = layer_max_tier.get(layer, 1)
        inv_tier = inventories.get(owner, {}).get((layer, expert), {"tier": 0})["tier"]
        plan.expert_tiers[key] = inv_tier
        if max_tier <= 0:
            if inv_tier >= 1:
                plan.evict_commands.setdefault(owner, []).append((layer, expert))
            continue
        target = min(max_tier, 2)
        if inv_tier < target:
            plan.pin_commands.setdefault(owner, []).append((layer, expert, target))
        elif inv_tier > target:
            if target >= 1:
                plan.pin_commands.setdefault(owner, []).append((layer, expert, target))
            else:
                plan.evict_commands.setdefault(owner, []).append((layer, expert))

    # Probe: temporarily demote a few VRAM residents to gather RAM ECOST samples.
    for owner, budget in (probe_demote or {}).items():
        if budget <= 0:
            continue
        have = {(l, e) for l, e, _t in plan.pin_commands.get(owner, [])}
        for (layer, expert), freq in hot:
            if budget <= 0:
                break
            if plan.experts.get(f"{layer}:{expert}") != owner:
                continue
            if (layer, expert) in have:
                continue
            inv_tier = inventories.get(owner, {}).get((layer, expert), {"tier": 0})["tier"]
            if inv_tier < 2:
                continue
            if layer_max_tier.get(layer, 2) < 2:
                continue
            plan.pin_commands.setdefault(owner, []).append((layer, expert, 1))
            have.add((layer, expert))
            budget -= 1


def apply_layer_blocks(sched: "PlacementScheduler", plan: PlacementPlan,
                       hot: list[tuple[tuple[int, int], int]], healthy: list[dict[str, Any]],
                       inventories: dict[str, dict], primary: str,
                       prev_owners: dict[int, str]) -> None:
    """Assign contiguous layer blocks to nodes; prefer measured tier per owner."""
    by_layer: dict[int, int] = {}
    by_layer_experts: dict[int, list[tuple[int, int]]] = {}
    for (layer, expert), freq in hot:
        if freq < 1:
            continue
        by_layer[layer] = by_layer.get(layer, 0) + freq
        by_layer_experts.setdefault(layer, []).append((expert, freq))

    if not by_layer:
        return

    node_max: dict[str, tuple[int, bool]] = {n["node_id"]: preferred_max_tier(sched, n) for n in healthy}
    plan.node_tier_prefs = {
        nid: {"max_tier": mt, "vram_slow": slow,
              "median_exec": {str(t): round(_node_median_exec(sched, nid, t))
                              for t in (0, 1, 2)},
              "vram_frac": round(_vram_residency_frac(next(n for n in healthy if n["node_id"] == nid)), 3)}
        for nid, (mt, slow) in node_max.items()
    }

    layer_owners: dict[int, str] = {}
    layer_max_tier: dict[int, int] = {}
    prev_node: str | None = None

    for layer in sorted(by_layer):
        freq = by_layer[layer]
        best_node, best_cost = primary, float("inf")
        for n in healthy:
            nid = n["node_id"]
            max_t, _ = node_max[nid]
            cost = _layer_cost_on_node(sched, primary, nid, layer, freq, inventories, max_t,
                                       prev_node, by_layer_experts.get(layer))
            if cost < best_cost:
                best_cost, best_node = cost, nid

        prev_nid = prev_owners.get(layer)
        if prev_nid and prev_nid != best_node and prev_nid in node_max:
            max_t, _ = node_max[prev_nid]
            prev_cost = _layer_cost_on_node(sched, primary, prev_nid, layer, freq, inventories,
                                            max_t, prev_owners.get(layer - 1),
                                            by_layer_experts.get(layer))
            if prev_cost <= best_cost * (1.0 + BLOCK_MOVE_PCT):
                best_node = prev_nid

        max_t, _ = node_max[best_node]
        layer_owners[layer] = best_node
        layer_max_tier[layer] = max_t
        prev_node = best_node

        for expert, _efreq in by_layer_experts.get(layer, []):
            key = f"{layer}:{expert}"
            plan.experts[key] = best_node

    plan.planned_blocks = _merge_blocks(layer_owners, layer_max_tier)
    plan.layer_caps = {str(layer): layer_max_tier[layer] for layer in layer_max_tier}

    probe: dict[str, int] = {}
    for n in healthy:
        nid = n["node_id"]
        mt, slow = node_max[nid]
        if slow or mt < 2:
            continue
        if TIER_PROBE <= 0:
            continue
        if _node_tier_sample_count(sched, nid, 1) >= MIN_TIER_SAMPLES:
            continue
        if _vram_residency_frac(n) < 0.5:
            continue
        probe[nid] = TIER_PROBE

    _emit_tier_commands(plan, inventories, layer_max_tier, hot, probe_demote=probe)


def apply_layer_coherence(sched: "PlacementScheduler", plan: PlacementPlan,
                          hot: list[tuple[tuple[int, int], int]], healthy: list[dict[str, Any]],
                          inventories: dict[str, dict], primary: str) -> None:
    """Assign all hot experts in a layer to one executor (fewer RPC hops per forward pass)."""
    by_layer: dict[int, list[tuple[int, int]]] = {}
    for (layer, expert), freq in hot:
        if freq < 1:
            continue
        by_layer.setdefault(layer, []).append((expert, freq))
    node_max = {n["node_id"]: preferred_max_tier(sched, n)[0] for n in healthy}
    layer_max_tier: dict[int, int] = {}
    for layer, items in by_layer.items():
        best_node, best_total = primary, float("inf")
        for n in healthy:
            nid = n["node_id"]
            total = sum(_expert_cost(sched, primary, nid, layer, expert, inventories) for expert, _ in items)
            if total < best_total:
                best_total, best_node = total, nid
        layer_max_tier[layer] = node_max.get(best_node, 1)
        for expert, _freq in items:
            plan.experts[f"{layer}:{expert}"] = best_node
    plan.layer_caps = {str(layer): layer_max_tier[layer] for layer in layer_max_tier}
    _emit_tier_commands(plan, inventories, layer_max_tier, hot)


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
    experts: dict[str, str] = field(default_factory=dict)
    expert_tiers: dict[str, int] = field(default_factory=dict)
    pin_commands: dict[str, list[tuple[int, int, int]]] = field(default_factory=dict)
    evict_commands: dict[str, list[tuple[int, int]]] = field(default_factory=dict)
    rpc_matrix_us: dict[str, dict[str, float]] = field(default_factory=dict)
    computed_at: float = 0.0
    layer_coherent: bool = False
    layer_blocks: bool = False
    planned_blocks: list[dict[str, Any]] = field(default_factory=list)
    layer_caps: dict[str, int] = field(default_factory=dict)
    node_tier_prefs: dict[str, dict[str, Any]] = field(default_factory=dict)


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
                vals = [v.exec_us for b in self._costs.values() for (_, _, t), v in b.items() if t == tier]
                if vals:
                    self._median_exec[tier] = statistics.median(vals)

    def clear_usage(self) -> None:
        self._usage.clear()

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
        return best or _node_median_exec(self, node_id, tier)

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
        prev_owners = _layer_owners_from_plan(self._last_plan)

        if LAYER_BLOCKS:
            apply_layer_blocks(self, plan, hot, healthy, inventories, primary, prev_owners)
            plan.layer_blocks = True
            plan.layer_coherent = True
        elif LAYER_COHERENT:
            apply_layer_coherence(self, plan, hot, healthy, inventories, primary)
            plan.layer_coherent = True
        else:
            node_max = {n["node_id"]: preferred_max_tier(self, n)[0] for n in healthy}
            layer_max_tier: dict[int, int] = {}
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
                        c *= 0.5
                    if c < best_cost:
                        best_cost, best_node = c, nid

                plan.experts[key] = best_node
                layer_max_tier[layer] = node_max.get(best_node, 1)
            plan.layer_caps = {str(layer): layer_max_tier[layer] for layer in layer_max_tier}
            _emit_tier_commands(plan, inventories, layer_max_tier, hot)

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
        return {
            "node_id": node_id,
            "peers": peers,
            "experts": experts,
            "layer_caps": plan.layer_caps,
            "blocks": plan.planned_blocks,
        }

    @property
    def last_plan(self) -> PlacementPlan | None:
        return self._last_plan

    def snapshot(self, node_ids: list[str] | None = None) -> dict[str, Any]:
        plan = self._last_plan
        experts = dict(plan.experts) if plan else {}
        ids = node_ids or sorted({nid for nid in experts.values()})
        blocks = blocks_from_planned(plan.planned_blocks) if plan and plan.planned_blocks else blocks_from_experts(experts, ids)
        return {
            "experts": experts,
            "expert_tiers": dict(plan.expert_tiers) if plan else {},
            "rpc_matrix_us": dict(plan.rpc_matrix_us) if plan else {},
            "computed_at": plan.computed_at if plan else 0.0,
            "usage_top": [{"layer": l, "expert": e, "count": c}
                             for (l, e), c in sorted(self._usage.items(),
                                                     key=lambda kv: -kv[1])[:32]],
            "blocks": blocks,
            "planned_blocks": plan.planned_blocks if plan else [],
            "layer_caps": dict(plan.layer_caps) if plan else {},
            "node_tier_prefs": dict(plan.node_tier_prefs) if plan else {},
            "layer_coherence": layer_coherence_stats(experts),
            "layer_coherent": bool(plan and getattr(plan, "layer_coherent", False)),
            "layer_blocks": bool(plan and getattr(plan, "layer_blocks", False)),
            "reassignments": self._reassignments,
            "rpc_histogram": self.rpc_histogram.snapshot(),
        }
