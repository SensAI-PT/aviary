# Qwen3-MoE Aviary bench — 2026-08-14

**Model:** `qwen3-moe-colibri` (engine `qwen3_moe`)
**Harness:** Cluster Bench tab / `POST /cluster/bench` — 8 requests, 32 max tokens, fixed prompt
**EPA:** `p50_cold / p50_warm - 1` (positive = warm faster than cold)
**Nodes:** Linux box `192.168.1.120` (`5a2a179f`, 61.7 GB host RAM, NVMe) and Mac `192.168.1.131` (`4f9668ff`, 17.2 GB host RAM). RAM/VRAM columns are Colibri **tier budgets** after dense weights, not the `--ram` CLI flag.

Raw exports (gitignored): `docs/bench/<utc>-<preset>.{md,json}`.

## Matrix

| # | setup | Linux RAM / VRAM | Mac | files |
|---|---|---|---|---|
| 1 | Solo fat, no GPU | 11.65 / 0 | — | `112430` cold, `112512` warm |
| 2 | Solo fat + CUDA | 10.51–10.54 / 1.08 | — | `112651` cold, `112717` warm |
| 3 | Solo starved | 0.13 / 0 | — | `112751` cold, `112806` warm |
| 4 | Cluster starved + Mac | 0.13 / 0 | 17.2 GB HW, **no tier budget advertised** | `112927` cold, `113005` warm |
| 5 | Cluster fat + Mac | 11.56 / 0 | same Mac | `113120` cold, `113127` warm, `113145` concurrent (W=4) |

## Scoreboard

| setup | p50 cold (s) | p50 warm (s) | EPA | hop mix (local / remote / fallback) | primary mix |
|---|---:|---:|---:|---|---|
| Solo fat, no GPU | 0.344 | 0.351 | **−0.02** | 50 / 0 / 50 | Linux 100% |
| Solo fat + CUDA | 0.436 | 0.447 | **−0.03** | 50 / 0 / 50 | Linux 100% |
| Solo starved | 0.505 | 0.474 | **+0.07** | 50 / 0 / 50 | Linux 100% |
| Cluster starved + Mac | 0.629 | **7.959** | **−0.92** | 44 / 13 / 43 → 41 / 18 / 41 | Linux 88% → **Mac 75%** |
| Cluster fat + Mac | 0.351 | 0.352 | **0.00** | 50 / 0 / 50 | Linux 100% |
| Cluster fat + Mac concurrent | — | p50 **3.28** / p95 **11.8** | RPS@4 **0.48** | 50 / 1 / 49 | Linux 100% |

First request in every cold run is an outlier (1.5–1.7 s solo; **26.6 s** once the Mac became primary). Medians exclude that pattern except where most requests are slow (starved-cluster warm, concurrent).

Median expert exec from placement (`median_exec` µs):

| node | disk (tier 0) | RAM (tier 1) | VRAM (tier 2) |
|---|---:|---:|---:|
| Linux fat / starved | 336–478 | **116–134** | 497–546 (when CUDA on) |
| Mac | 2176 | **1343–1478** | (unused) |

Linux RAM experts are ~10× faster than Mac RAM experts. CUDA VRAM exec is ~4× slower than Linux RAM (546 vs 131 µs) at 1.08 GB VRAM budget.

## Findings

### 1. On this model, SSD page cache already wins — pin budget barely matters

Solo fat (11.65 GB RAM) vs solo starved (0.13 GB) is only **1.47×** (0.344 s vs 0.505 s). Warm vs cold EPA is ~0 on every single-node run. That is not “experts stay on disk and get faster as they pin.” It is “the working set is already in the Linux page cache after the first token.”

Implication: **EPA as “warm RAM vs cold disk” is not measurable on a small Qwen3-MoE sitting on a fast NVMe box with 64 GB of host RAM.** You need a model whose hot experts do not fit in page cache, or you need to drop caches between cold runs (`echo 3 > /proc/sys/vm/drop_caches` on the agent, not just wipe `.coli_usage`).

### 2. 1 GB of CUDA made the fat box slower, not faster

Solo fat + GPU: p50 **0.436 s** vs **0.344 s** RAM-only (**+27%**). Warm p95 also stretched (0.544 s vs 0.367 s). Placement still set `max_tier=2` with `vram_slow=false` even though VRAM median exec was 546 µs vs RAM 131 µs (4.2×). `vram_frac` was only 0.08 — almost nothing actually lived on the card, but the CUDA path still taxed the run.

Implication: **small VRAM budgets on MoE are overhead, not acceleration.** Matches [Why can GPU be slower than RAM?](../AVIARY.md#why-can-gpu-vram-be-slower-than-ram). Detection did not flip `vram_slow` here because the heatmap never filled with VRAM samples.

### 3. Adding a slower peer can make the cluster much worse than solo

Starved Linux + Mac:

- Cold: p50 0.629 s (already worse than solo starved 0.505 s). One of eight chats went to the Mac and took **26.6 s**. Remote hops appeared (13%).
- Warm: routing flipped to **Mac primary on 6/8 chats**. p50 **7.96 s** — **12.7× worse than the same cluster’s cold p50**, **16.8× worse than solo starved warm**.

The Mac advertised 17.2 GB host RAM but **no `ram_gb` / `vram_gb` tiers**. Placement still assigned 256 hot experts and treated the Mac as a viable primary. Once usage accumulated on the Mac, affinity locked traffic there.

Implication: **cluster is not “more nodes = faster.”** Chat latency is the **primary’s** wall time. A slow peer that wins even one warm-lock chat dominates the scoreboard. Expert RPC (13–18% remote) is a rounding error next to “the Mac ran dense + attn + LM head.”

### 4. Fat Linux + Mac is a no-op cluster (and that is the good outcome)

With 11.56 GB on Linux, every sequential chat stayed on Linux (100% primary, 0% remote). p50 0.351 / 0.352 s — **identical to solo fat**. The Mac sat idle. Concurrent (4 workers) still pinned all 8 chats to Linux: p50 3.28 s, p95 11.8 s, **0.48 req/s**. Sequential equivalent is ~2.3 req/s at 0.35 s each; concurrency **queued on one primary** instead of spreading.

Implication: when the fast node has enough RAM, Aviary correctly **does not** send chat to the slower peer. You do **not** get N× throughput from a second replica unless routing actually picks it as primary — and you do not *want* that if it is the Mac. Concurrent RPS here is a **single-node queue**, not a cluster win.

### 5. The 50 / 50 local / fallback mix is not “half the experts missed RAM”

Every solo run (and fat-cluster sequential) reports ~50% local / 50% fallback / 0% remote, often **exact** counts (1024 / 1024). That is instrumentation: a failed remote attempt then a local exec, not a 50% disk miss rate. Real RPC only shows up when a peer is in the placement map (starved cluster 13–18% remote; fat concurrent 1.2%).

Do not put hop-mix percentages in a README table as if they were cache-hit rates.

## What this means for the table you wanted

A useful README row for this hardware is **not** “cluster EPA = X.” It is:

| config | p50 wall (32 tok) | vs solo fat | notes |
|---|---:|---:|---|
| Solo fat, RAM only | **0.34 s** | — | best sequential |
| Solo fat + 1 GB CUDA | 0.44 s | **+27%** | GPU tax |
| Solo starved (0.13 GB) | 0.47–0.51 s | +38–47% | page cache still carries it |
| Cluster fat + Mac | 0.35 s | ~0 | Mac unused; same as solo |
| Cluster starved + Mac (warm) | **8.0 s** | **+23×** | Mac became primary |
| Cluster fat concurrent W=4 | 3.3 s p50 / 0.48 RPS | latency 10× | all 8 chats on Linux |

**Takeaway:** on Qwen3-MoE with a fast NVMe Linux box, **keep the primary fat and RAM-only**. A Mac peer does not help sequential latency and can destroy it if routing treats it as a chat primary. Cluster value on this pair is **capacity only if you pin chat to Linux** (or run two equal-speed boxes).

## Caveats

- 8 × 32-token chats. Good for routing/placement, not tok/s marketing.
- `.coli_usage` wipe does **not** drop the kernel page cache or engine `eheat`.
- Mac never reported Colibri tier budgets — placement compared apples to missing data.
- Concurrent was only run on fat+Mac, not as a same-cluster suite with the sequential pair.
