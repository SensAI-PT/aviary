# Aviary — cluster overlay on Colibri

Aviary turns N single-node [Colibri](https://github.com/JustVugg/colibri) engines
into a request-routed cluster with a Spark-style node dashboard. This repository
(`SensAI-PT/aviary-hy3`) is **Aviary**, not upstream Colibri — it ships a synced
Colibri engine base plus the cluster control plane.

Built on Colibri and the Hy3-enabled fork
[`ErikTromp/colibri-hy3`](https://github.com/ErikTromp/colibri-hy3).

## What Aviary adds

| feature | status | description |
|---|---|---|
| `coli master` | Phase 1 ✓ | Node registry, heartbeat lease, OpenAI API proxy, dashboard host |
| `coli agent` | Phase 1 ✓ | Wraps one local `Engine`; relays `HWINFO`/`TIERS`/`EMAP`/`HITS` telemetry |
| Cluster dashboard tab | Phase 1 ✓ | Live node list, hardware, load, per-node expert heatmaps |
| Least-loaded routing | Phase 1 ✓ | Master picks the healthy agent with lowest in-flight count |
| Streaming chat proxy | Phase 1 ✓ | Master correctly close-frames SSE responses to the browser |
| Cross-node expert RPC | Phase 2 | Single request can dispatch expert calls to peer nodes (planned) |
| Placement scheduler | Phase 2 | Cluster-wide EMAP aggregation + `LOAD`/`EVICT`/`PIN` (planned) |

Phase 1 uses **full-replica load balancing**: each agent runs a complete independent
copy of the model. The network is never required for correctness — if a remote node
is slow or gone, agents fall back to local disk exactly like single-node `coli serve`.

See [`aviary-cluster-plan.md`](../aviary-cluster-plan.md) for the full Phase 1–4 roadmap.

## Ownership boundary

| Layer | Owns | Paths |
|---|---|---|
| **Colibri** | Engines, serve protocol, OpenAI HTTP, model detection | `c/*.c`, `c/openai_server.py`, `c/coli` (except master/agent), shared headers |
| **Aviary** | Cluster registry, agent/master, cluster protocol, Cluster UI | `c/aviary/**`, `docs/cluster_protocol.md`, `coli master` / `coli agent`, `web/src/Cluster.tsx` (+ thin tab wiring) |

**Rule:** Aviary must not permanently fork per-model engines (`hy3.c`, `deepseek_v4.c`,
`kimi_k3.c`, `inkling.c`, `colibri.c`, …). New Colibri families work automatically once
`coli`/`openai_server` resolve them — including **DeepSeek V4 Flash** after syncing
upstream Colibri.

Aviary’s runtime contract with Colibri:

1. OpenAI-compatible HTTP (`POST /v1/chat/completions`, …)
2. `GET /health`, `GET /experts`, `GET /profile`
3. Serve-protocol telemetry when the engine emits it (`HWINFO` / `TIERS` / `EMAP` / `HITS`)

## Quick start

```bash
git clone https://github.com/SensAI-PT/aviary-hy3.git && cd aviary-hy3/c
./setup.sh

# machine A — master (HTTP :9000, control :9002)
./coli master --host 0.0.0.0 --port 9000

# machine A — first agent (any Colibri family — Hy3, DeepSeek V4, GLM, …)
COLI_MODEL=/path/to/model ./coli agent --master http://A:9000 --host 0.0.0.0 --port 8001

# machine B — second agent (same model weights on disk)
COLI_MODEL=/path/to/model ./coli agent --master http://A:9000 --host 0.0.0.0 --port 8001 \
  --advertise-host B
```

Open **http://A:9000** — chat and the **Cluster** tab both go through the master.
Point the web UI’s server URL at the master, not an individual agent.

### Environment

| variable | default | meaning |
|---|---|---|
| `AVIARY_CONTROL_PORT` | `9002` | Master control-plane TCP port |
| `AVIARY_HEARTBEAT_SEC` | `2` | Agent heartbeat interval |
| `AVIARY_HEARTBEAT_MISS` | `3` | Missed heartbeats before eviction |
| `AVIARY_RPC_TIMEOUT_MS` | `150` | Control-plane I/O deadline (expert RPC in Phase 2) |
| `COLI_API_KEY` | — | Optional auth on master and agents |

Each agent persists a stable node UUID at `<model_dir>/.aviary_node_id`.

## Master HTTP surface

| route | method | response |
|---|---|---|
| `/cluster/health` | GET | `{ "status": "ok", "nodes": N, "healthy": M }` |
| `/cluster/nodes` | GET | Full node registry snapshot |
| `/v1/chat/completions` | POST | Proxied to least-loaded healthy agent (streaming preserved) |
| `/v1/models` | GET | From first healthy agent |
| `/health` | GET | Master liveness + scheduler snapshot |
| static `web/dist` | GET | Dashboard (Chat, Brain, Profiling, Cluster tabs) |

## Synced Colibri base

This repository’s Colibri sources track upstream via periodic merges from
[`ErikTromp/colibri-hy3`](https://github.com/ErikTromp/colibri-hy3) and
[`JustVugg/colibri`](https://github.com/JustVugg/colibri). After refreshing engine
sources, re-apply the Aviary overlay (`c/aviary/`, `coli master`/`agent`, Cluster tab).

Hy3 mux + dashboard telemetry in `hy3.c` is a **Colibri-base parity** patch (same
capability as `colibri.c` / Inkling serve), not Aviary-specific logic.

## Further reading

- [`cluster_protocol.md`](cluster_protocol.md) — agent⇄master wire format
- [`aviary-cluster-plan.md`](../aviary-cluster-plan.md) — Phase 1–4 roadmap and checklists
- [`serve_protocol.md`](serve_protocol.md) — engine⇄server mux protocol (telemetry source)
- [Colibri upstream](https://github.com/JustVugg/colibri) — engines, benchmarks, model roster
