"""Parse and diff `.coli_usage` expert history files (see route_trace.h)."""

from __future__ import annotations

import hashlib
from pathlib import Path

# Must match rt_engine_names[] in route_trace.h
ENGINE_NAMES = ("glm_moe_dsa", "inkling", "olmoe", "kimi_k3", "hy3", "qwen3_moe")


def engine_id_for(arch: str) -> int:
    """FNV-1a 32-bit hash — same as rt_hash() in route_trace.h."""
    h = 2166136261
    for ch in arch.encode("utf-8"):
        h ^= ch
        h = (h * 16777619) & 0xFFFFFFFF
    return h


def read_coli_usage(path: Path) -> tuple[int, int, int | None, list[dict[str, int]]]:
    """Return (n_layers, n_experts, engine_id, sparse usage records)."""
    if not path.is_file():
        return (-1, -1, None, [])
    n_layers, n_experts, engine_id = -1, -1, None
    records: list[dict[str, int]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        fields = line.split()
        if len(fields) != 3:
            continue
        try:
            layer, expert, count = int(fields[0]), int(fields[1]), int(fields[2])
        except ValueError:
            continue
        if layer == -1:
            n_layers, n_experts = expert, count
        elif layer == -2:
            engine_id = count
        elif layer >= 0:
            records.append({"layer": layer, "expert": expert, "count": count})
    return n_layers, n_experts, engine_id, records


def usage_delta(prev: list[dict[str, int]], cur: list[dict[str, int]]) -> list[dict[str, int]]:
    """Sparse delta: entries where count increased."""
    prev_map = {(r["layer"], r["expert"]): r["count"] for r in prev}
    return [{"layer": l, "expert": e, "count": c - prev_map.get((l, e), 0)}
            for r in cur for l, e, c in [(r["layer"], r["expert"], r["count"])]
            if c > prev_map.get((l, e), 0)]


def arch_engine_id(arch: str) -> int:
    return engine_id_for(arch)
