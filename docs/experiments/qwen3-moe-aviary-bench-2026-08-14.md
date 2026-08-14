# Qwen3-MoE Aviary bench — 2026-08-14

**Model:** `qwen3-moe-colibri` (engine `qwen3_moe`)
**Harness:** Cluster Bench suite — 8 requests × 32 tokens, fixed prompt, cold → warm → concurrent (W=4)
**EPA:** `p50_cold / p50_warm - 1` (positive = warm faster than cold)
**Nodes:** Linux `192.168.1.120` (`5a2a179f`, 61.7 GB host RAM, NVMe, optional CUDA) and Mac `4f9668ff` (17.2 GB host RAM, no Metal; IP `192.168.1.131` then `192.168.1.78`)

| wave | when | raw files |
|---|---|---|
| **0** before coordinator/donor | ~11:24–11:32Z | `docs/bench_backup/` |
| **1** after coordinator/donor (`5373533`) | ~12:30–12:34Z | `docs/bench_backup2/` |
| **2** after Mac telemetry + CUDA fill (`92dfcf0`) | ~14:08–14:13Z | `docs/bench/` |
| **3** honest hop labels + starve claim | code after `92dfcf0` | hop-mix contract; live `drop_caches` numbers pending |

`RAM flag` / `VRAM flag` are agent `--ram` / `--vram` (or inferred from hwinfo). `RAM occ` / `VRAM occ` are Colibri occupied tiers.

## Wave 3 — honest hop labels + true-starve claim

Shipped after wave 2. Wave 0–2 tables below are unchanged (those runs still show the old 100% `fallback` label).

### Hop labels (one event per expert)

`qwen3_moe.c` / `hy3.c` used to call `cluster_rpc_expert` whenever `AVIARY_CLUSTER=1`, tag `fallback` on every lookup miss, then skip the `local` emit. Solo/fat benches therefore printed **100% fallback**. That was not a cache miss.

Contract now (`cluster_has_peer` + `hop_kind()`):

| kind | when |
|---|---|
| `local` | no peer in placement — do not RPC |
| `remote` | assigned peer, RPC succeeded |
| `fallback` | assigned peer, RPC missed/timed out — then local exec, no second event |

Expected on the next solo/fat or local-only suite: **100% local**, 0% fallback, 0% remote. Do not treat fallback% as a cache-miss rate.

### True starve (operator claim, not a remote wipe)

`--ram 2` is pin-store starve. Linux 64 GB page cache still holds the ~18 GB checkpoint. `.coli_usage` wipe is not cold.

Cluster Bench now has **Caches dropped (operator claim)** (`meta.drop_caches_note`). The master does **not** run `drop_caches`. Linux-only procedure between cold steps:

```bash
echo 3 > /proc/sys/vm/drop_caches   # root, on the Linux agent
```

Or use a model that does not fit in page cache.

**Success gate** for starved+Mac after a real drop: Linux **100% primary**, **remote hops > 0** if Mac RAM exec + rpc < Linux disk load, p50 **not** 8 s, no donor-as-primary warning. If Mac RAM+rpc is still worse than Linux SSD, that is a valid measured no-op — write it down, do not force donation.

**Numbers:** hop-mix contract only. Latency/RPS and donor-RPC proof need an operator `drop_caches` re-run; paste `docs/bench/latest.md` here when that exists.

## Wave 2 — after Mac telemetry + CUDA fill (14:08Z)

| # | setup | file | RAM / VRAM flag | occ | p50 warm | RPS@4 | primary / Mac role |
|---|---|---|---|---|---:|---:|---|
| 1 | Solo fat RAM | `140800` | 50 / — | 11.47 / 0 | **0.334** | **2.98** | Linux |
| 2 | Solo fat + CUDA | `140849` | 50 / 14 | 0.54 / **10.87** | 0.532 | 1.82 | Linux; `gpu_tier` clamp `safe_free` |
| 3 | Solo starved | `140924` | 2 / — | 0.13 / 0 | 0.440 | 2.34 | Linux |
| 4 | Fat + Mac | `141034` | 50 + Mac **3.95 inferred** | 11.41 / 0 · Mac occ **0** | **0.335** | **3.12** | Linux 100%, Mac `slow_hw` |
| 5 | Starved + Mac | `141101` | 2 + Mac 3.95 | 0.13 / 0 · Mac occ 0 | 0.442 | 2.24 | Linux 100%, Mac `slow_hw` |
| 6 | Starved + Mac (repeat) | `141127` | same | same | 0.423 | 2.31 | Linux 100%, Mac `slow_hw` |
| 7 | Mac alone | `141155` | Mac **3.7 inferred** | **2.04** / 0 | 1.407 | 0.71 | Mac (only node) |
| 8 | Starved + Mac (Mac rebuilt) | `141330` | 2 + Mac **4.25** | 0.13 / 0 · Mac occ **0.61** | 0.441 | 2.23 | Linux 100%, Mac `slow_hw` |

You were right about Mac advertising: cluster runs 4–6 show inferred **RAM flag ~4 GB** (agent hwinfo synthesis) but **RAM occ 0** — the Mac engine was not printing occupied `TIERS`. Mac-alone and the last starved+Mac show occupied **2.04 GB** then **0.61 GB**. That is the rebuilt agent/engine.

### Is the Mac completely left alone?

**As a chat coordinator: yes.** Every mixed run is Linux 100% primary, zero `donor_primary_warnings`. Reason is `slow_hw` (cores/RAM vs Linux), not missing tiers.

**As an expert donor: also yes, on purpose.** Expert maps are 100% Linux. Remote hop mix is **0%**. Placement keeps a Mac donor score of **~507 ms** (default rpc+load) vs Linux RAM exec **~120 µs**. Fat+Mac cold recorded 55 `rpc_in` on Linux (Mac may have probed Linux), not Linux sending work to the Mac. After the rebuild, Mac RAM exec is **807 µs** (~7× Linux) — still not cheaper than Linux page cache.

So the Mac is registered, advertised, and scored, then correctly ignored. Cluster sequential matches solo Linux. That is the intended coordinator+donor outcome on this pair: the Mac is a legal donor that loses the cost comparison.

### CUDA is finally a real GPU run — and it still loses

`--vram 14` now occupies **10.87 GB** (was 0.3 GB / 1.08 GB). `gpu_tier`: requested 14, occupied 10.87, safe leftover 0.35, clamp `safe_free` (WSL free minus dense minus 512 MB). `vram_frac` 0.66. VRAM median exec **280–283 µs** vs RAM **116 µs** (~2.4×). Placement flipped `vram_slow=true` and capped `max_tier=1` after it measured that.

Wall: GPU warm **0.532 s** vs fat RAM **0.334 s** (**+59%**). Concurrent **1.82 vs 2.98 RPS**. Filling the card made the tax *worse*, not better. Detection now works; the honest conclusion is **keep this Qwen3-MoE on Linux RAM**.

## Wave 1 — after coordinator/donor (`5373533`)

| # | setup | RAM / VRAM flag | occ | file | p50 cold | p50 warm | EPA | RPS@4 | primary |
|---|---|---|---|---|---:|---:|---:|---:|---|
| 1 | Solo fat, no GPU | 50 / — | 11.52 / 0 | `123012` | **0.324** | **0.322** | 0.01 | **3.04** | Linux 100% |
| 2 | Solo fat + CUDA | 50 / 14 | 11.2 / **0.3** | `123058` | 0.356 | 0.462 | **−0.23** | 2.13 | Linux 100% |
| 3 | Solo starved | 2 / — | 0.13 / 0 | `123131` | 0.461 | 0.485 | −0.05 | 2.22 | Linux 100% |
| 4 | Fat + Mac | 50 / — + Mac blank | 11.5 / 0 | `123222` | **0.302** | **0.305** | −0.01 | **3.18** | Linux 100%, Mac `missing_tiers` |
| 5 | Starved + Mac | 2 / — + Mac blank | 0.13 / 0 | `123247` | 0.428 | **0.426** | 0.01 | 2.38 | Linux **95.8%**, Mac 4.2% (warned) |
| 6 | Mac alone | no flags | blank | `123349` | 1.473 | 1.389 | 0.06 | 0.71 | Mac 100% (only node) |

Concurrent p50 (queue on one coordinator): fat 1.31 s · GPU 1.86 s · starved 1.80 s · fat+Mac 1.24 s · starved+Mac 1.68 s · Mac alone **5.57 s**.

First request of each cold step is still an outlier (1.4–2.2 s solo; **33.4 s** on the one Mac-primary chat). Medians hide it except starved+Mac cold p95 **4.98 s**.

Median expert exec (µs) after the fix:

| node | disk (tier 0) | RAM (tier 1) | VRAM (tier 2) |
|---|---:|---:|---:|
| Linux | 340 | **118–145** | 494 (when CUDA on) |
| Mac | 2189 | **876–1092** | unused |

Linux RAM experts stay ~8–10× faster than Mac RAM. CUDA VRAM exec is still ~4× slower than Linux RAM (494 vs 118 µs).

## Before vs after (the bug the plan fixed)

| setup | before warm p50 | after warm p50 | before primary | after primary |
|---|---:|---:|---|---|
| Solo fat | 0.351 | 0.322 | Linux | Linux |
| Solo fat + CUDA | 0.447 | 0.462 | Linux | Linux |
| Solo starved | 0.474 | 0.485 | Linux | Linux |
| Fat + Mac | 0.352 | 0.305 | Linux 100% | Linux 100%, Mac donor-only |
| Starved + Mac | **7.959** | **0.426** | **Mac 75%** | Linux 100% warm / concurrent |
| Fat + Mac concurrent | 3.28 s / **0.48 RPS** | 1.24 s / **3.18 RPS** | Linux (queued badly) | Linux, healthy single-node queue |
| Mac alone | — | 1.389 | — | baseline: **4.3×** Linux fat |

The 8 s starved+Mac disaster was **Mac as chat coordinator**, not expert RPC. After the split, the Mac is `donor_only` (`missing_tiers` when it sends no `TIERS`; `slow_ram_exec` once ECOST arrives). Warm and concurrent never leave Linux. That is the Spark-style outcome the plan specified.

## Findings

### 1. Coordinator vs donor works — with one cold-join leak

Fat+Mac: Mac `coordinator_eligible=no (missing_tiers)`, Linux 100% of 24 chats, 0% remote. Sequential matches (slightly beats) solo fat. Concurrent RPS matches solo fat (~3.1). The Mac is idle as a donor because Linux RAM already wins `load+exec`.

Starved+Mac: Mac `donor_only (slow_ram_exec)`. Warm 8/8 and concurrent 8/8 on Linux at **0.43 s / 2.38 RPS** — vs **7.96 s** before. Bench emitted `donor_only node 4f9668ff was primary 4.2%`. That 4.2% is **one cold chat** (33.4 s, 1443 remote hops) at cluster join, before the role stuck. Cold p50 stayed 0.43 s (Linux); cold p95 4.98 s is that one job.

Implication: **do not treat the Mac as a second chat server.** Pin chat to the fast coordinator. The remaining leak is join-order / first-heartbeat, not warm-lock.

### 2. Mac alone is the missing baseline — ~4× slower

Mac-only suite: warm p50 **1.39 s**, concurrent **0.71 RPS** / 5.57 s p50. Linux fat is 0.32 s / 3.04 RPS. That ratio (about **4.3×**) is why one Mac-primary chat dominates any mixed scoreboard. It also explains the old 8 s warm: six of eight chats ran dense+attn+MTP on the Mac.

The Mac still advertises **no `ram_flag` / `ram_occ`**. Agent synthesis only fires if the process has `RAM_GB` / `--ram`. Start the Mac agent with an explicit `--ram` if you want occupied-vs-configured columns; role assignment already works from missing tiers or slow ECOST.

### 3. Donating experts did not beat Linux page cache

Starved+Mac cold actually used the donor path: first Linux chat had 107 `remote` hops; overall cold hop mix 25% remote / 75% fallback. Placement recorded a Mac donor score of **~504 ms** (rpc+exec). Warm and concurrent then went **0% remote** — the learner decided Mac RAM is not cheaper than Linux SSD/page-cache.

Solo starved vs solo fat is still only **1.4×** (0.46 s vs 0.32 s) with `--ram 2` vs `--ram 50`. Occupied RAM 0.13 GB vs 11.5 GB does not change the fact that the ~18 GB checkpoint fits in 64 GB host RAM. **EPA is not a pin-vs-disk metric on this model** unless you drop the kernel page cache (`echo 3 > /proc/sys/vm/drop_caches` on Linux). Wiping `.coli_usage` is not cold.

Cluster win for this pair is therefore **not** sequential latency. It is (1) not destroying latency by moving the chat, which we now do, and (2) extra coordinator-class machines for concurrent chats, which the Mac is not.

### 4. `--vram 14` is still not 14 GB on GPU

Fat+CUDA: flag **14 GB**, occupied **0.3 GB** (was 1.08 GB in the morning run). Placement now sets coordinator `max_tier=2` (`vram_slow=false`, `vram_frac` 0.023–0.026) — Aviary is no longer silently capping RAM. The engine still only parks a sliver of experts on the card (WSL free-VRAM minus dense minus 2 GB reserve). `gpu_tier` clamp reason is still **not** on `/cluster/nodes`; stderr has the clamp line.

Wall time: warm **0.462 s** vs fat RAM **0.322 s** (**+43%**). Same-run EPA **−0.23** (warm slower than cold as more CUDA path engages). VRAM median exec 494 µs vs RAM 118 µs. Concurrent 2.13 RPS vs 3.04 RAM-only.

Implication: **this is still not a 14 GB GPU result.** Until `vram_occ` is in the 10+ GB range *or* the clamp is explicit in the cluster snapshot, do not attribute the +43% to “14 GB of experts on CUDA.” Small VRAM occupancy on this MoE is overhead. Detection still does not flip `vram_slow` (too few VRAM-resident cells).

### 5. Hop mix is honest about double-counting — and now over-labels fallback

Before: solo reported a fake **50/50 local/fallback** (fallback then local both fired). After: one event per expert, so solo/fat is **100% fallback**. That is `AVIARY_CLUSTER=1` trying RPC, missing (empty or self placement), tagging `fallback`, then suppressing the local emit.

Read hop mix as:

| kind | meaning now |
|---|---|
| `remote` | Activation RPC succeeded (only starved+Mac cold, 16–25%) |
| `fallback` | RPC miss or no donor, then local exec — **includes healthy local work** |
| `local` | Almost never, because cluster mode always attempts RPC first |
| `rpc_in` | This node served an expert for a peer (not in the %-mix table) |

Do not publish fallback% as a cache-miss rate. Wave 3 implements that split: no placement → `local`; assigned remote missed → `fallback`.

## Honest README row (after)

| config | p50 warm (32 tok) | RPS@4 | vs solo fat | notes |
|---|---:|---:|---:|---|
| Solo fat, RAM only | **0.32 s** | **3.04** | — | best single node |
| Fat + Mac | **0.31 s** | **3.18** | ~0 | Mac donor-only, unused |
| Solo starved | 0.49 s | 2.22 | +50% | page cache still carries it |
| Starved + Mac | **0.43 s** | 2.38 | +33% | Linux coordinator; was **8.0 s** |
| Solo fat + CUDA 0.3 GB occ | 0.46 s | 2.13 | +43% | not a 14 GB result |
| Mac alone | 1.39 s | 0.71 | **+4.3×** | why it must not be primary |

**Takeaway:** keep the coordinator on the fast Linux box, RAM-only, for this model. The Mac is a legal expert donor and a disastrous coordinator. Donor RPC only pays if Linux would actually hit disk — it does not, until page cache is dropped or the model is much larger. CUDA needs occupied VRAM in the requested ballpark before we re-bench it as a GPU story.

## Caveats

- 8 × 32-token chats. Good for routing/placement, not tok/s marketing.
- `.coli_usage` wipe does not drop kernel page cache or engine `eheat`.
- Mac agent was not started with `--ram`, so configured/occupied columns stay blank.
- One starved+Mac cold chat still landed on the Mac (join race). Warm/concurrent did not.
- Hop `fallback` in wave 0–2 includes successful local exec whenever cluster mode is on. Wave 3 labels that `local`.
