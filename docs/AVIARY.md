# Aviary — run large LLMs on commodity hardware, fast

**Mission:** run large MoE language models on a cluster of commodity machines, as fast as
possible.

Aviary is the cluster control plane for [Colibri](https://github.com/JustVugg/colibri) — a
pure-C MoE inference engine that streams experts from disk/RAM/VRAM based on routing history.
This repository (`SensAI-PT/aviary-hy3`) ships Aviary plus a synced Colibri engine base.

Built on Colibri and the Hy3-enabled fork
[`ErikTromp/colibri-hy3`](https://github.com/ErikTromp/colibri-hy3).

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
| Least-loaded + affinity routing | ✓ | Master picks agent by load and hot-expert affinity; cold executors bootstrap when peers are warmed |
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

## Quick start

```bash
git clone https://github.com/SensAI-PT/aviary-hy3.git && cd aviary-hy3/c
./setup.sh

# machine A — master (HTTP :9000, control :9002)
./coli master --host 0.0.0.0 --port 9000

# machine A — first agent
export AVIARY_CLUSTER=1
COLI_MODEL=/path/to/model ./coli agent --master http://A:9000 --host 0.0.0.0 --port 8001

# machine B — second agent (same model weights on disk)
export AVIARY_CLUSTER=1
COLI_MODEL=/path/to/model ./coli agent --master http://A:9000 --host 0.0.0.0 --port 8001 \
  --advertise-host B
```

Open **http://A:9000** — chat and the **Cluster** tab both go through the master.
Point the web UI's server URL at the master, not an individual agent.

### Environment

| variable | default | meaning |
|---|---|---|
| `AVIARY_CLUSTER` | `0` | Enable cross-node expert RPC on agents (`1` to activate) |
| `AVIARY_CONTROL_PORT` | `9002` | Master control-plane TCP port |
| `AVIARY_EXPERT_PORT` | `9003` | Agent expert RPC TCP port (or `coli agent --expert-port`) |
| `AVIARY_HEARTBEAT_SEC` | `2` | Agent heartbeat interval |
| `AVIARY_HEARTBEAT_MISS` | `3` | Missed heartbeats before eviction |
| `AVIARY_RPC_TIMEOUT_MS` | `150` | Expert RPC latency budget (ms) |
| `AVIARY_PLACEMENT_SEC` | `4` | Placement scheduler recompute interval |
| `AVIARY_ROUTE_BOOTSTRAP_RATIO` | `0.1` | Route chat to cold executors when their hot-expert residents are below this fraction of the cluster leader (collects usage/ECOST) |
| `AVIARY_PREFETCH` | `0` | Enable Phase 4 best-effort shard prefetch daemon on agents |
| `AVIARY_PREFETCH_SEC` | `10` | Prefetch poll interval (seconds) |
| `AVIARY_PREFETCH_MAX` | `2` | Max concurrent shard downloads per agent |
| `AVIARY_PIN_BATCH` | `32` | Max PIN commands pushed per placement tick |
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
| `/cluster/shards` | GET | Agent-only: list local safetensors shard files (prefetch peer) |
| `/cluster/shard?name=…` | GET | Agent-only: download one shard file from model dir |
| `/v1/chat/completions` | POST | Proxied to chosen agent (streaming preserved) |
| `/v1/models` | GET | From first healthy agent |
| `/health` | GET | Master liveness + scheduler snapshot |
| static `web/dist` | GET | Dashboard (Chat, Brain, Profiling, Cluster tabs) |

Proxied chat responses include `X-Aviary-Job-Id` and `X-Aviary-Node-Id` headers for tracing
in the Cluster **Jobs** tab.

## Ownership boundary

| Layer | Owns | Paths |
|---|---|---|
| **Colibri** | Engines, serve protocol, OpenAI HTTP, model detection | `c/*.c`, `c/openai_server.py`, `c/coli` (except master/agent) |
| **Aviary** | Cluster registry, placement, agent/master, cluster protocol, Cluster UI | `c/aviary/**`, `c/cluster_rpc.h`, `docs/cluster_protocol.md`, `web/src/Cluster.tsx` |

**Rule:** Aviary must not permanently fork per-model engines. New Colibri families work
automatically once `coli`/`openai_server` resolve them.

## Synced Colibri base

This repository's Colibri sources track upstream via periodic merges from
[`ErikTromp/colibri-hy3`](https://github.com/ErikTromp/colibri-hy3) and
[`JustVugg/colibri`](https://github.com/JustVugg/colibri). After refreshing engine sources,
re-apply the Aviary overlay (`c/aviary/`, `coli master`/`agent`, Cluster tab).

## Further reading

- [`cluster_protocol.md`](cluster_protocol.md) — agent⇄master wire format
- [`serve_protocol.md`](serve_protocol.md) — engine⇄server mux protocol (telemetry source)
- [Colibri upstream](https://github.com/JustVugg/colibri) — engines, benchmarks, model roster
