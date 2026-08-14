# Aviary — run large LLMs on commodity hardware, fast

**Mission:** run large MoE language models on a cluster of commodity machines, as fast as
possible.

Aviary is the cluster control plane for [Colibri](https://github.com/JustVugg/colibri) — a
pure-C MoE inference engine that streams experts from disk/RAM/VRAM based on routing history.
This repository (`SensAI-PT/aviary-hy3`) ships Aviary plus a synced Colibri engine base.

Built on Colibri and the Hy3-enabled fork
[`ErikTromp/colibri-hy3`](https://github.com/ErikTromp/colibri-hy3). See
[`COLIBRI_SYNC.md`](COLIBRI_SYNC.md) for upstream merge procedure and drift tracking.

## Aviary vs Colibri

| | **Colibri** | **Aviary** |
|---|---|---|
| Deployment | Single machine | Cluster (1 master + N agents) |
| Weights | Local disk/RAM/VRAM hot-load | **Full model replica on every node's disk** |
| Network | None | Activations + control RPC only |
| Routing | Per-expert local tiers | Cluster placement + cross-node expert RPC |
| Goal | Fast single-node MoE | Concurrent throughput + placement wins |
| Key question | — | Is activation + RTT faster than disk load on one node? |

Colibri answers "how fast can one machine run this MoE?" Aviary answers "how do we wire several
modest machines so the **next token** is as fast as possible — without shipping checkpoints
over the network?"

## The problem Aviary solves

Frontier MoE models (hundreds of billions of parameters, a handful active per token) do not
fit comfortably on one machine — and datacenter GPU racks are not the only path forward.
Teams already have several boxes with fast NVMe, ample RAM, and optional GPUs.

Aviary wires those machines into one inference cluster and asks a harder question than
"can we distribute?": **given this hardware, where should each expert live so the next
token is as fast as possible?**

That means:

- tracking **per-model** expert usage (`.coli_usage` isolated by engine family)
- measuring **execution cost** per node, tier (disk/RAM/VRAM), and expert
- deciding when **remote RPC** beats **local disk load**
- falling back to local execution whenever the cluster path misses — never blocking on the network

## Architecture

```
Client → master (HTTP :9000)
           ├─ routes chat to primary agent
           ├─ placement scheduler (every ~4s)
           └─ Cluster dashboard (jobs, executors, placement, RPC)

Agent (each node)
  ├─ Colibri engine subprocess (MoE forward, hot-load, .coli_usage)
  ├─ expert RPC server (:9003) — serves individual expert matmuls to peers
  └─ control heartbeat → master (EMAP, ECOST, usage, profile)
```

A single chat request lands on one **primary** agent. Inside `moe()`, the engine may RPC
individual expert forwards to whichever node currently holds that expert hottest. Remote
miss/timeout always falls through to local disk load.

**Phase 1 capacity model:** each agent runs a **full local replica** of the model. Adding
nodes increases **concurrent throughput and availability**, not pooled model memory — N modest
boxes add up to N× throughput, not one virtual GPU with the sum of their VRAM.

## Feature status

| feature | status | description |
|---|---|---|
| `coli master` | ✓ | Node registry, heartbeat lease, OpenAI API proxy, dashboard host |
| `coli agent` | ✓ | Wraps one local `Engine`; relays telemetry to master |
| Cluster dashboard | ✓ | Spark-style: Overview, Jobs, Executors, Placement, RPC tabs |
| Least-loaded + affinity routing | ✓ | Master picks agent by load, hot-expert affinity, and slow-node penalty; probabilistic cold bootstrap |
| Per-request expert trace | ✓ | Jobs tab: layer/expert hops (local, remote, fallback) per chat |
| Cross-node expert RPC | ✓ | `EXEC_EXPERT` mux + TCP expert server; `cluster_rpc.h` in `moe()` |
| Placement scheduler | ✓ | Cost-aware expert placement; `PLACEMENT` pushed to agents |
| Per-model usage isolation | ✓ | Stats keyed by `engine_id`; never merged across architectures |
| Request job tracking | ✓ | `GET /cluster/jobs`, `GET /cluster/overview` |
| MTP / dense layer placement | planned | Phase 2.4 — after expert RPC gates proven |
| Cross-node weight prefetch | ✓ | Phase 4 — best-effort shard fetch from peers (`AVIARY_PREFETCH=1`) |

Enable Phase 2 on agents:

```bash
export AVIARY_CLUSTER=1
COLI_MODEL=/path/to/model ./coli agent --master http://MASTER:9000 ...
```

The master writes placement tables to each agent; agents persist them at
`<model_dir>/.aviary_placement.json` and set `AVIARY_PLACEMENT` for the engine.

## Quick start — 1 master + 2 agents

### 1. Build

```bash
git clone https://github.com/SensAI-PT/aviary-hy3.git && cd aviary-hy3/c
./setup.sh
```

### 2. API key (optional)

```bash
export COLI_API_KEY=your-secret   # set on master and every agent
```

### 3. Model on every node

Place the **same checkpoint directory** on each machine. For cluster testing, use
[Qwen3-30B-A3B int4](https://huggingface.co/UnderstandLing/Qwen3_30B_A3B_i4):

```bash
pip install -U "huggingface_hub[cli]"
hf download UnderstandLing/Qwen3_30B_A3B_i4 --local-dir /path/to/qwen3_i4
cd c && make qwen3_moe
```

DIY convert: [`qwen3_moe.md`](qwen3_moe.md).

### 4. Start master and agents

```bash
# machine A — master (HTTP :9000, control :9002)
./coli master --host 0.0.0.0 --port 9000

# machine A — agent 1
export AVIARY_CLUSTER=1
COLI_MODEL=/path/to/qwen3_i4 ./coli agent --master http://A:9000 --host 0.0.0.0 --port 8001

# machine B — agent 2
export AVIARY_CLUSTER=1
COLI_MODEL=/path/to/qwen3_i4 ./coli agent --master http://A:9000 --host 0.0.0.0 --port 8001 \
  --advertise-host B
```

**Cluster advertise host:** each agent must register a **LAN-routable** address (not
`127.0.0.1`) so peers can reach its expert RPC port. With `AVIARY_CLUSTER=1`, agents
auto-replace loopback `--advertise-host` with the detected LAN IP when possible.

WSL2: if port 9003 is taken by Windows, add `--expert-port 9013`.

### 5. Use the cluster

Open **http://A:9000**. Chat and the **Cluster** tab both go through the master.
Point the web UI server URL at the master, not an individual agent.

## Benchmark hypothesis

The central experiment: **cross-node expert RPC vs single-node disk load**.

| setup | description |
|---|---|
| **Baseline** | One node, `AVIARY_CLUSTER=0`, experts loaded from local NVMe |
| **Cluster** | N nodes with full replicas, `AVIARY_CLUSTER=1`, placement + RPC |
| **Metrics** | tokens/s, p50/p95 latency, `expert_wait_s` (Executors profile), RPC histogram, disk I/O |

```bash
python c/tools/cluster_bench.py --master http://A:9000 --prompt "Hello" --requests 20
```

If activation + RTT across the LAN beats NVMe load latency for your hardware, cluster mode wins.
If not, you still gain **N× concurrent chat capacity** from N full replicas.

## Reading cluster statistics

| UI tab | What it shows | Per chat request? |
|---|---|---|
| **Jobs** | Primary executor, duration, expert path table | **Yes** — pick a job for layer/expert hops |
| **Overview → usage_top** | Cluster-wide hot experts (aggregate) | No |
| **Placement** | Scheduler's expert ownership (~4s refresh) | No — planned, not actual routing |
| **RPC** | PING latency between expert ports | No — infrastructure health |
| **Executors → profile** | Recent turn `wall_s`, `expert_wait_s` per node | Per-node, not job-linked |

**Job trace kinds:** `local` (disk/RAM/VRAM on primary), `remote` (RPC to peer), `fallback`
(remote miss → local load), `rpc_in` (peer served an inbound expert).

### What the heatmap is (and is not)

Aviary does **not** pipeline consecutive layers across machines. Every node keeps a **full
model replica**; one chat runs on a single **primary** agent (dense, attention, LM head).
Cross-node work is **per-hot-expert RPC** only.

| UI label | Meaning |
|---|---|
| **assigned** | Scheduler ownership for a hot expert (~top 256 by usage) |
| **resident** | Assigned experts with EMAP tier ≥ 1 (RAM/VRAM) on that node |
| **layer blocks** | Contiguous layer ranges assigned to one executor; `max_tier` caps disk (0) / RAM (1) / VRAM (2) from measured ECOST |

**Chat latency ≈ primary node speed.** If routing picks a cold or slow primary, the whole
request pays that cost (~seconds to minutes), not LAN RTT alone (expert RPC is typically a
few ms). Prefer warming the fast node (`--advertise-host` + `--ram`) and check
`/cluster/jobs` for which executor was primary.

**How both machines are used together:** one chat still has a single **primary** (dense +
attn + LM head). Peers help only via **per-expert RPC** when placement says they own a hot
expert. That shows up in the Jobs trace as `remote` / `rpc_in` / `fallback`. It is not
pipeline-parallel layers. Cold bootstrap will not send chat to a *slower* cold peer just to
warm it — only to a cold peer that looks faster than the current warm leader.

## Correctness gate

Remote expert RPC is an **optimization**, not a correctness requirement. For every parallel
expert path, output must be **token-exact identical** to the standard local `moe()` kernel —
the same bar Colibri uses for TF oracle gates (see
[Colibri #911](https://github.com/JustVugg/colibri/issues/911)).

Today:

- RPC miss/timeout → automatic local fallback (`cluster_rpc.h`)
- `c/tests/test_cluster_oracle.py` compares local vs cluster token IDs when a model fixture exists

Optional dev flag: `AVIARY_ORACLE=1` (future) dual-runs remote+local and asserts float equality.

### Aviary vs Colibri (placement)

| Layer | Owns |
|---|---|
| **Aviary** | Which **node** owns each layer block; **max tier** (disk/RAM/VRAM) per layer via `layer_caps` in `.aviary_placement.json`; PIN/EVICT to enforce |
| **Colibri** | Which **experts** within a layer occupy RAM/VRAM slots (REPIN, heat) — subject to Aviary caps |

When ECOST shows VRAM slower than RAM on a node (common on some CUDA setups), Aviary sets `max_tier=1` and demotes automatically after enough samples.

### Why can GPU (VRAM) be slower than RAM?

People expect “put experts on the GPU → faster.” For MoE that is often **false**, even when you give the same budget to both (e.g. 10 GB RAM and 10 GB VRAM).

**Short version:** MoE does thousands of **small** expert multiplies per reply. A modern CPU is often better at that than a GPU that has to be fed over the bus for every tiny job. Same gigabytes does **not** mean the same speed.

**Slightly longer:**

1. **Same GB ≠ same job.** RAM budget keeps experts in host memory for CPU kernels. VRAM budget *copies* a hot subset onto the GPU. Those are different code paths, not “the same experts, just faster memory.”

2. **Tiny jobs hurt GPUs.** Chat routing picks a few experts per layer, many times per token. That is a storm of small matmuls. GPUs shine on big batched work; CPU SIMD/OpenMP often wins on small irregular ones. Launching and syncing a GPU kernel can cost more than the multiply.

3. **Shipping data to the GPU costs time.** Activations usually live on the CPU. Each CUDA expert path may pay a host↔device tax. The RAM path skips that.

4. **Your laptop/WSL stack matters.** WSL2 CUDA goes through an extra Windows layer; Metal on a Mac can look great while the same model on a CUDA box looks worse. That is environment, not Aviary “preferring” RAM for fun.

5. **Aviary measures and reacts.** After enough ECOST samples, if VRAM exec is clearly slower than RAM on a node (`AVIARY_VRAM_SLOW_RATIO`, default 1.25×), the Placement UI can show **VRAM slower than RAM** and Aviary caps that node at RAM (`max_tier=1`) so Colibri stops promoting those layers into CUDA.

**Important:** If Colibri already loaded *everything* into VRAM at startup, the engine used to report those runs as “RAM” timings (bug) and Aviary never saw a real RAM vs GPU comparison — so it kept the GPU placement. Fixed engines tag CUDA exec as tier 2. When the heatmap is mostly VRAM, Aviary also **probe-demotes** a few hot experts to RAM (`AVIARY_TIER_PROBE`, default 8) for a couple of placement ticks so it can measure RAM, then demote the rest if GPU loses.

**What to do if CUDA hurts:** rebuild/restart agents after pulling this fix, run 2–3 prompts, watch Placement for **VRAM slower than RAM** and blocks switching to `· RAM`. Or force it with `CUDA_EXPERT_GB=0` / `--gpu none` while keeping a solid `--ram` pin.

## Environment

| variable | default | meaning |
|---|---|---|
| `AVIARY_CLUSTER` | `0` | Enable cross-node expert RPC on agents (`1` to activate) |
| `AVIARY_CONTROL_PORT` | `9002` | Master control-plane TCP port |
| `AVIARY_EXPERT_PORT` | `9003` | Agent expert RPC TCP port (or `coli agent --expert-port`) |
| `AVIARY_HEARTBEAT_SEC` | `2` | Agent heartbeat interval |
| `AVIARY_HEARTBEAT_MISS` | `3` | Missed heartbeats before eviction |
| `AVIARY_RPC_TIMEOUT_MS` | `150` | Expert RPC latency budget (ms) |
| `AVIARY_PLACEMENT_SEC` | `4` | Placement scheduler recompute interval |
| `AVIARY_ROUTE_BOOTSTRAP_RATIO` | `0.1` | When cold nodes have &lt; this fraction of the leader's hot-expert residents, **this fraction** of chat routes to cold (probabilistic, not 100%) |
| `AVIARY_ROUTE_COLD_MIN_RATIO` | `0.35` | Floor for cold routing when one node has zero EMAP residents and another has many (warm-lock escape) |
| `AVIARY_PREFETCH` | `0` | Enable Phase 4 best-effort shard prefetch daemon on agents |
| `AVIARY_PREFETCH_SEC` | `10` | Prefetch poll interval (seconds) |
| `AVIARY_PREFETCH_MAX` | `2` | Max concurrent shard downloads per agent |
| `AVIARY_PIN_BATCH` | `32` | Max PIN/EVICT commands pushed per placement tick |
| `AVIARY_LAYER_BLOCKS` | `1` | Assign hot layers as contiguous blocks across nodes (reduces RPC hops) |
| `AVIARY_LAYER_COHERENT` | `0` | Legacy: one node per layer when `AVIARY_LAYER_BLOCKS=0` |
| `AVIARY_VRAM_SLOW_RATIO` | `1.25` | Cap node at RAM when median VRAM exec exceeds RAM exec by this factor |
| `AVIARY_BLOCK_MOVE_PCT` | `0.15` | Hysteresis: keep a layer block unless moving saves ≥ this fraction |
| `AVIARY_MIN_TIER_SAMPLES` | `8` | ECOST samples required before declaring VRAM-slow on a node |
| `AVIARY_TIER_PROBE` | `8` | When heatmap is mostly VRAM, demote this many hot experts to RAM to measure a baseline |
| `AVIARY_BENCH_DIR` | `<repo>/docs/bench/` | Cluster bench JSON/markdown output directory |
| `COLI_API_KEY` | — | Optional auth on master and agents |

Each agent persists a stable node UUID at `<model_dir>/.aviary_node_id`.

## Master HTTP surface

| route | method | response |
|---|---|---|
| `/cluster/health` | GET | `{ "status": "ok", "nodes": N, "healthy": M }` |
| `/cluster/nodes` | GET | Full node registry snapshot |
| `/cluster/overview` | GET | Combined nodes + placement + jobs (dashboard poll) |
| `/cluster/jobs` | GET | Active and recent proxied requests |
| `/cluster/placement` | GET | Current scheduler output |
| `/cluster/costs` | GET | Cost matrix snapshot |
| `/cluster/bench` | GET | Bench runner status + last result |
| `/cluster/bench` | POST | Start a bench run (409 if already running) |
| `/cluster/bench/latest` | GET | Last saved bench JSON from `AVIARY_BENCH_DIR` |
| `/cluster/shards` | GET | Agent-only: list local safetensors shard files (prefetch peer) |
| `/cluster/shard?name=…` | GET | Agent-only: download one shard file from model dir |
| `/v1/chat/completions` | POST | Proxied to chosen agent (streaming preserved) |
| `/v1/models` | GET | From first healthy agent |
| `/health` | GET | Master liveness + scheduler snapshot |
| static `web/dist` | GET | Dashboard (Chat, Brain, Profiling, Cluster tabs) |

Proxied chat responses include `X-Aviary-Job-Id` and `X-Aviary-Node-Id` headers for tracing
in the Cluster **Jobs** tab.

## Cluster bench (EPA harness)

The Cluster **Bench** tab (or `POST /cluster/bench`) runs repeatable chat workloads through the
same master proxy as the UI, captures job traces and placement, and writes README-ready output
under `AVIARY_BENCH_DIR` (default: `<repo>/docs/bench/`):

- `latest.json` / `latest.md` — most recent run
- `<utc-stamp>-<preset>.json` / `.md` — archived copy

**Presets**

| preset | behavior |
|---|---|
| **Cold sequential** | Wipe `.coli_usage` on all agents + master usage; N chats one-by-one |
| **Warm sequential** | Keep usage; same N chats (multi-turn / warmed experts) |
| **Concurrent** | Keep usage; N chats with W worker threads (throughput) |
| **Run suite** | Cold → warm → concurrent; computes scoreboard |

**Scoreboard (suite)**

- p50 / p95 wall seconds (cold vs warm vs concurrent step)
- Primary `node_id` mix from job routing
- `% local / remote / fallback` from per-request expert traces
- Planned blocks + `node_tier_prefs` after the run
- **EPA** = `p50_cold / p50_warm - 1` (warm vs cold on the same cluster)
- **RPS@W** from the concurrent step

**Toggles**

- **Wipe usage** — default on for Cold / Suite cold step; sends `RESET_USAGE` to agents and clears master scheduler usage. Engine in-memory heat is *not* cleared (restart agents for a fully cold engine).
- **Local-only** — pushes placement with empty `experts` so the primary runs every expert locally (SSD→RAM story without stopping peers).

Paste `docs/bench/latest.md` into the README benchmark table after a run.

| variable | default | meaning |
|---|---|---|
| `AVIARY_BENCH_DIR` | `<repo>/docs/bench/` | Where bench JSON/markdown is written |

## Ownership boundary

| Layer | Owns | Paths |
|---|---|---|
| **Colibri** | Engines, serve protocol, OpenAI HTTP, model detection | `c/*.c`, `c/openai_server.py`, `c/coli` (except master/agent) |
| **Aviary** | Cluster registry, placement, agent/master, cluster protocol, Cluster UI | `c/aviary/**`, `c/cluster_rpc.h`, `docs/cluster_protocol.md`, `web/src/Cluster.tsx` |

**Rule:** Aviary must not permanently fork per-model engines. New Colibri families work
automatically once `coli`/`openai_server` resolve them.

## Synced Colibri base

This repository tracks three lines of development:

| remote | role |
|---|---|
| `SensAI-PT/aviary-hy3` | Aviary product (this repo) |
| `ErikTromp/colibri-hy3` | Hy3/Qwen3 engine fork base |
| `JustVugg/colibri` | Upstream Colibri engines and UI |

See [`COLIBRI_SYNC.md`](COLIBRI_SYNC.md) for merge order, paths to preserve, and validation.
Run `c/tools/sync_drift.sh` to print ahead/behind counts.

## Further reading

- [`COLIBRI_SYNC.md`](COLIBRI_SYNC.md) — upstream merge procedure
- [`cluster_protocol.md`](cluster_protocol.md) — agent⇄master wire format
- [`serve_protocol.md`](serve_protocol.md) — engine⇄server mux protocol (telemetry source)
- [Colibri upstream](https://github.com/JustVugg/colibri) — engines, benchmarks, model roster
