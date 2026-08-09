<p align="center">
  <img src="assets/colibri.svg" width="500" alt="Aviary — distributed inference cluster">
</p>

<p align="center">
  <strong>Run large LLMs on a cluster of commodity hardware, as fast as possible.</strong>
</p>

<p align="center">
  <a href="docs/AVIARY.md"><b>Aviary docs</b></a> ·
  <a href="docs/qwen3_moe.md"><b>Qwen3 test model</b></a> ·
  <a href="aviary-cluster-plan.md"><b>Roadmap</b></a> ·
  <a href="https://github.com/JustVugg/colibri"><b>Colibri (engines)</b></a>
</p>

## Mission

**Aviary runs large MoE language models across a cluster of commodity machines — and keeps them fast.**

Frontier models are too big for one box. Datacenter GPUs are expensive. Aviary's bet is simpler:
wire together the hardware you already have (fast NVMe, ample RAM, optional GPUs) and let a
smart control plane decide **where each expert lives** and **which node serves each token's work**.

The goal is not "distributed inference that works." It is **distributed inference that wins on
throughput and latency** against stuffing everything onto a single node.

## What is Aviary?

**Aviary is the cluster layer above [Colibri](https://github.com/JustVugg/colibri).**

Each node runs a local Colibri engine — a pure-C, streaming MoE runtime that hot-loads experts
from disk/RAM/VRAM based on usage. Aviary adds:

| piece | role |
|---|---|
| **master** | Registry, request routing, placement scheduler, dashboard host |
| **agents** | One engine subprocess per node; reports usage, costs, and expert heat |
| **Cluster UI** | Spark-style jobs, executors, placement map, RPC latency — live on the master |

Point any OpenAI-compatible client (or the built-in web UI) at the master. It routes work
across the flock while tracking which experts are hot on which machine.

**Phase 1 adds concurrent throughput and availability, not pooled model capacity.** Each node
holds a full model replica on local disk; N nodes ≈ N× concurrent request capacity, not one
bigger virtual GPU.

This repository (`SensAI-PT/aviary-hy3`) is **Aviary**, not upstream Colibri. It ships a synced
Colibri engine base plus the cluster control plane.

## Why "Aviary"?

A **colibrì** (hummingbird) is tiny, fast, and runs on almost nothing.

An **aviary** is where you keep many of them.

Aviary is the enclosure for a flock of Colibri engines: one bird per node, one roof over the
cluster. The name reflects the mission — **many modest machines cooperating** through a shared
scheduler, not one giant GPU monolith.

## How Aviary makes clusters fast

MoE models activate only a few experts per token. That sparsity is the leverage:

1. **Usage-aware hot-loading** — each agent tracks `.coli_usage` and live EMAP heat; experts
   migrate between disk, RAM, and VRAM based on what routing actually needs.
2. **Cluster-wide placement** — the master aggregates per-node usage and execution costs, then
   pushes placement decisions: which node should own which hot expert, and when remote RPC
   beats local disk load.
3. **Cross-node expert RPC** — a single forward pass can dispatch expert matmuls to the node
   that already has them hot, with automatic fallback to local disk if the remote path misses.
4. **Commodity-first** — full model weights on every node's disk; no weight shipping on the
   critical path. The network carries activations and control, not checkpoints.

Correctness never depends on the cluster. A slow, missing, or disabled peer always falls back
to the same local path single-node Colibri uses today.

## Status

| phase | focus | status |
|---|---|---|
| **1** | Node registry, heartbeat, least-loaded routing, Cluster dashboard | ✓ |
| **2** | Cross-node expert RPC, cost-aware placement scheduler, per-model usage isolation | ✓ |
| **3** | Request timelines, RPC histograms, replication metrics (Spark-style observability) | in progress |
| **4** | Cross-node weight hot-swap prefetch | planned |

Phase 2 is enabled with `AVIARY_CLUSTER=1` on agents. The placement scheduler recomputes every
few seconds from live telemetry — not per token.

Details: [`aviary-cluster-plan.md`](aviary-cluster-plan.md) · [`docs/AVIARY.md`](docs/AVIARY.md)

## Recommended test model: Qwen3-30B-A3B

For cluster development and proof-scale testing, use **[Qwen3-30B-A3B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507)**:

| | |
|---|---|
| Total params | ~30.5B |
| Active per token | ~3.3B |
| Layers / experts | 48L · 128E · top-8 |
| Why here | Large enough for real MoE routing and multi-node placement; small enough to copy to several nodes and iterate |

**Easiest — download the ready int4 container** from
[UnderstandLing/Qwen3_30B_A3B_i4](https://huggingface.co/UnderstandLing/Qwen3_30B_A3B_i4):

```bash
pip install -U "huggingface_hub[cli]"
hf download UnderstandLing/Qwen3_30B_A3B_i4 --local-dir /path/to/qwen3_i4

cd c && make qwen3_moe
export AVIARY_CLUSTER=1
COLI_MODEL=/path/to/qwen3_i4 ./coli agent --master http://MASTER:9000 --ram 10
```

Or convert from the upstream Qwen checkpoint — see [`docs/qwen3_moe.md`](docs/qwen3_moe.md).

## Quick start — Aviary cluster

| component | command | default port | role |
|---|---|---|---|
| **master** | `./c/coli master` | HTTP `9000`, control `9002` | Registry, routing, placement, dashboard |
| **agent** | `./c/coli agent --master URL` | HTTP `8001`, expert RPC `9003` | One local engine + telemetry |

```bash
git clone https://github.com/SensAI-PT/aviary-hy3.git && cd aviary-hy3/c
./setup.sh

# machine A — master
./coli master --host 0.0.0.0 --port 9000

# machine A — first agent
export AVIARY_CLUSTER=1
COLI_MODEL=/path/to/qwen3_i4 ./coli agent --master http://A:9000 --host 0.0.0.0 --port 8001

# machine B — second agent (same weights on disk)
export AVIARY_CLUSTER=1
COLI_MODEL=/path/to/qwen3_i4 ./coli agent --master http://A:9000 --host 0.0.0.0 --port 8001 \
  --advertise-host B
```

Open **http://A:9000**. Chat and the **Cluster** tab go through the master.

The Cluster dashboard shows **Overview**, **Jobs** (running/completed requests), **Executors**
(click-through node detail with heatmaps), **Placement** (which node owns which experts), and
**RPC** latency between nodes.

Environment: [`docs/AVIARY.md`](docs/AVIARY.md) · wire format: [`docs/cluster_protocol.md`](docs/cluster_protocol.md)

## Install

```bash
git clone https://github.com/SensAI-PT/aviary-hy3.git && cd aviary-hy3/c
./setup.sh          # builds engines (incl. qwen3_moe), runs tiny self-tests when fixtures exist
./coli info         # sanity check
```

Prebuilt [Colibri release binaries](https://github.com/JustVugg/colibri/releases) do **not**
include `coli master` / `coli agent` — use this repository for the cluster.

Python 3 is required for the launcher and API gateway; inference engines are pure C.

## What Aviary owns vs Colibri

| Layer | Owns | Where |
|---|---|---|
| **Aviary** | Master, agents, registry, placement, cluster protocol, Cluster UI | `c/aviary/`, `coli master`, `coli agent`, `web/src/Cluster.tsx` |
| **Colibri** (synced base) | Per-model C engines, `coli chat`/`serve`, OpenAI HTTP | `c/*.c`, `c/openai_server.py` — see [Colibri](https://github.com/JustVugg/colibri) |

Aviary does not fork engine architectures. When Colibri adds a model family, agents pick it
up automatically once `coli` resolves `config.json`. Each model keeps its own `.coli_usage`
stats — usage from one architecture never contaminates another.

## Other supported engines

Any Colibri-supported checkpoint works on agents. For day-to-day **Aviary cluster testing**,
prefer **Qwen3-30B-A3B** above.

| Family | Active / total | Aviary notes |
|---|---|---|
| **Qwen3 MoE** | 3.3B / 30B | **Recommended for cluster testing** — [`docs/qwen3_moe.md`](docs/qwen3_moe.md) |
| Hy3 | 21B / 295B | [`docs/hy3.md`](docs/hy3.md) |
| DeepSeek V4 Flash | 13B / 284B | [`docs/deepseek-v4.md`](docs/deepseek-v4.md) |
| GLM-5.2, Inkling, Kimi K3, OLMoE | — | Colibri upstream docs |

Same weights on disk on every agent; same `COLI_MODEL` path (or equivalent copy per node).

## Documentation

| topic | doc |
|---|---|
| Aviary overview, env vars, Phase 2 | [docs/AVIARY.md](docs/AVIARY.md) |
| Qwen3 test model (convert, oracle, cluster) | [docs/qwen3_moe.md](docs/qwen3_moe.md) |
| Agent⇄master wire format | [docs/cluster_protocol.md](docs/cluster_protocol.md) |
| Cluster roadmap | [aviary-cluster-plan.md](aviary-cluster-plan.md) |
| OpenAI API, web dashboard | [docs/api.md](docs/api.md) |
| Colibri engines, tuning, benchmarks | [Colibri repository](https://github.com/JustVugg/colibri) |

## Repo layout (Aviary-relevant)

```
c/
├── aviary/               cluster control plane (master, agent, placement, jobs)
├── cluster_rpc.h         cross-node expert RPC (Phase 2)
├── coli                  CLI: master, agent, chat, serve, …
├── openai_server.py      OpenAI HTTP (used by agents and master proxy)
├── qwen3_moe.c           Qwen3 MoE engine — recommended test target
└── setup.sh              build + tiny self-test
web/                      dashboard — Chat, Brain, Profiling, Cluster tabs
docs/                     Aviary + synced engine docs
aviary-cluster-plan.md    Phase 1–4 plan
```

The Colibri engine tree (`colibri.c`, `hy3.c`, headers, backends, …) lives alongside Aviary
and tracks [upstream Colibri](https://github.com/JustVugg/colibri). Treat it as a dependency,
not as Aviary's product surface.

## Acknowledgements

Aviary builds on [Colibri](https://github.com/JustVugg/colibri) and the Hy3-enabled base
[`ErikTromp/colibri-hy3`](https://github.com/ErikTromp/colibri-hy3). Model weights come from
the open releases behind each engine.

## License

Apache 2.0. Model weights are subject to each publisher's license.
