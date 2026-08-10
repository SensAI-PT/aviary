```
     _                         _          _
    / \     _____   _____ _ __(_) ___  __| |
   / _ \   / _ \ \ / / _ \ '__| |/ _ \/ _` |
  / ___ \ |  __/\ V /  __/ |  | |  __/ (_| |
 /_/   \_\ \___| \_/ \___|_|  |_|\___|\__,_|
        distributed MoE inference cluster
```

<p align="center">
  <strong>Run large MoE language models across a cluster of commodity machines — as fast as possible.</strong>
</p>

<p align="center">
  <a href="docs/AVIARY.md"><b>Docs</b></a> ·
  <a href="docs/qwen3_moe.md"><b>Test model</b></a> ·
  <a href="docs/cluster_protocol.md"><b>Protocol</b></a> ·
  <a href="docs/COLIBRI_SYNC.md"><b>Upstream sync</b></a>
</p>

## What is Aviary?

**Aviary is a cluster control plane for sparse MoE inference.**

Wire together the boxes you already have — fast NVMe, ample RAM, optional GPUs — and Aviary
decides **where each expert lives** and **which node serves each token's work**. Point any
OpenAI-compatible client (or the built-in dashboard) at one master URL; it routes chat across
the flock while tracking expert heat, placement, and per-request expert paths.

```
  Client / Web UI
        │
        ▼
  ┌─────────────┐     control heartbeats      ┌──────────────┐
  │   MASTER    │◄────────────────────────────│   Agent B    │
  │  :9000      │                             │  engine+RPC  │
  │  registry   │     chat proxy              └──────▲───────┘
  │  placement  │──────────────────┐                  │
  │  dashboard  │                  │         EXEC_EXPERT (activations)
  └──────┬──────┘                  ▼                  │
         │                  ┌──────────────┐          │
         └─────────────────►│   Agent A    │──────────┘
                            │  engine+RPC  │
                            └──────────────┘
         full model weights on disk at every node — network carries activations, not checkpoints
```

| component | role |
|---|---|
| **master** | Node registry, chat routing, placement scheduler, Cluster dashboard |
| **agent** | One MoE engine per machine; reports usage, costs, expert heat |
| **Cluster UI** | Jobs, executors, placement map, RPC latency — Spark-style observability |

### Design principles

1. **Full replica on every node** — copy the same checkpoint to each agent's disk. Aviary never
   ships weights over the network during inference; the LAN carries activations and control only.
2. **Usage-aware hot-loading** — each agent tracks expert heat (`.coli_usage`, EMAP); experts
   migrate between disk, RAM, and VRAM based on what routing actually needs.
3. **Cluster-wide placement** — the master aggregates costs and pushes decisions: which node
   should own which hot expert, and when remote RPC beats local disk load.
4. **Cross-node expert RPC** — inside a forward pass, matmuls dispatch to whichever node already
   has the expert hot; miss or timeout always falls back to local load.
5. **Correctness never depends on the cluster** — a slow, missing, or disabled peer never
   blocks inference; the local path always works.

**Capacity model:** N nodes ≈ N× concurrent chat throughput, not one virtual GPU with pooled
VRAM. Each agent holds a complete model replica.

### The experiment we're running

| setup | what happens |
|---|---|
| **Baseline** | One node; experts cold-loaded from NVMe per routing decision |
| **Cluster** | Same weights everywhere; primary agent RPCs experts hot on a peer |
| **Question** | Is **activation + RTT** across the LAN faster than **disk load on one box**? |
| **Measure** | tokens/s, p50/p95 latency, expert_wait_s, Cluster RPC histogram, disk I/O |

[`c/tools/cluster_bench.py`](c/tools/cluster_bench.py) runs repeatable load tests. See
[`docs/AVIARY.md`](docs/AVIARY.md) for interpretation.

## Quick start — 1 master + 2 agents

### 1. Build

```bash
git clone https://github.com/SensAI-PT/aviary-hy3.git && cd aviary-hy3/c
./setup.sh
./coli info    # sanity check
```

### 2. API key (optional)

```bash
export COLI_API_KEY=your-secret   # same value on master and all agents
```

### 3. Model on every node

Download **[Qwen3-30B-A3B int4](https://huggingface.co/UnderstandLing/Qwen3_30B_A3B_i4)** to the
**same path on every machine** (or copy the directory after downloading once):

```bash
pip install -U "huggingface_hub[cli]"
hf download UnderstandLing/Qwen3_30B_A3B_i4 --local-dir /path/to/qwen3_i4
cd c && make qwen3_moe
```

DIY convert from the upstream Qwen checkpoint: [`docs/qwen3_moe.md`](docs/qwen3_moe.md).

### 4. Start the cluster

| component | command | ports |
|---|---|---|
| master | `./coli master --host 0.0.0.0 --port 9000` | HTTP `9000`, control `9002` |
| agent | `AVIARY_CLUSTER=1 COLI_MODEL=… ./coli agent --master URL` | HTTP `8001`, expert RPC `9003` |

```bash
# machine A — master
./coli master --host 0.0.0.0 --port 9000

# machine A — agent 1
export AVIARY_CLUSTER=1
COLI_MODEL=/path/to/qwen3_i4 ./coli agent --master http://A:9000 --host 0.0.0.0 --port 8001

# machine B — agent 2
export AVIARY_CLUSTER=1
COLI_MODEL=/path/to/qwen3_i4 ./coli agent --master http://A:9000 --host 0.0.0.0 --port 8001 \
  --advertise-host B
```

**WSL2:** if port 9003 is blocked by Windows, add `--expert-port 9013`.

### 5. Chat and observe

Open **http://A:9000**. Point the web UI at the **master**, not an individual agent.

The **Cluster** tab shows:
- **Jobs** — which node served each chat + layer/expert hop table per request
- **Executors** — per-node heatmaps, profile, owned experts
- **Placement** — scheduler's expert ownership (~4s refresh)
- **RPC** — latency matrix between expert ports

## Status

| phase | focus | status |
|---|---|---|
| **1** | Registry, heartbeat, routing, Cluster dashboard | done |
| **2** | Cross-node expert RPC, placement scheduler, usage isolation | done |
| **3** | Per-request traces, RPC histograms, job timelines | in progress |
| **4** | Cross-node weight prefetch (`AVIARY_PREFETCH=1`) | done |

Enable cluster mode on agents: `export AVIARY_CLUSTER=1`

Key env vars: [`docs/AVIARY.md`](docs/AVIARY.md) · wire format: [`docs/cluster_protocol.md`](docs/cluster_protocol.md)

## Recommended test model

**[Qwen3-30B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507)** — 48 layers,
128 experts, top-8 routing. Large enough for real multi-node placement; small enough to copy to
several nodes and iterate.

| | |
|---|---|
| Total params | ~30.5B |
| Active per token | ~3.3B |
| Ready int4 container | [UnderstandLing/Qwen3_30B_A3B_i4](https://huggingface.co/UnderstandLing/Qwen3_30B_A3B_i4) |

## Documentation

| topic | doc |
|---|---|
| Overview, env vars, benchmarks | [docs/AVIARY.md](docs/AVIARY.md) |
| Upstream engine sync / drift | [docs/COLIBRI_SYNC.md](docs/COLIBRI_SYNC.md) |
| Qwen3 convert + oracle | [docs/qwen3_moe.md](docs/qwen3_moe.md) |
| Master⇄agent wire format | [docs/cluster_protocol.md](docs/cluster_protocol.md) |
| OpenAI API + web dashboard | [docs/api.md](docs/api.md) |

## Repo layout

```
c/
├── aviary/          master, agent, registry, placement, jobs, prefetch
├── cluster_rpc.h    cross-node expert RPC
├── coli             CLI: master, agent, chat, serve
├── openai_server.py OpenAI HTTP gateway
├── qwen3_moe.c      recommended cluster test engine
└── tools/           cluster_bench.py, sync_drift.sh, sync_port.sh
web/src/Cluster.tsx  Cluster dashboard tab
docs/                Aviary docs + engine references
```

Check upstream drift anytime:

```bash
./c/tools/sync_drift.sh
```

## Why "Aviary"?

A hummingbird is tiny, fast, and runs on almost nothing. An **aviary** is where you keep many
of them — one engine per node, one roof over the cluster. **Many modest machines cooperating**
through a shared scheduler, not one giant GPU monolith.

## Acknowledgements

Inference engines ship from the [Colibri](https://github.com/JustVugg/colibri) project (sync
procedure in [`docs/COLIBRI_SYNC.md`](docs/COLIBRI_SYNC.md)). Community discussion:
[Colibri #911](https://github.com/JustVugg/colibri/issues/911).

## License

Apache 2.0. Model weights are subject to each publisher's license.
