# Aviary web dashboard

React/Vite interface for an OpenAI-compatible **Aviary master** or single-node
[`coli serve`](https://github.com/JustVugg/colibri) backend.

**Mission context:** Aviary runs large MoE LLMs on a cluster of commodity hardware as fast
as possible. The dashboard is how you see that cluster work — live jobs, executor health,
expert placement, and RPC latency — without leaving the browser.

In cluster mode, point the server URL at the **master** (`http://host:9000/v1`) — not an
individual agent. The master proxies chat completions and exposes cluster routes for the
**Cluster** tab.

```sh
npm install
npm run dev
```

The default endpoint is `http://127.0.0.1:8000/v1` (single-node `coli serve`).
For Aviary, use `http://127.0.0.1:9000/v1` after starting `coli master`.
Use **Probe server** to load models from the connected backend.

Build for production (served by `coli master` and `coli web` from `web/dist`):

```sh
npm run build
```

## Tabs

| tab | single-node | cluster (master) |
|---|---|---|
| **Chat** | OpenAI-compatible chat | Proxied through master to an agent |
| **Brain** | Live expert heatmap | Same, from the routed agent |
| **Profiling** | Per-turn `PROF` breakdown | Same, from the routed agent |
| **Cluster** | N/A (shows “not a master”) | Spark-style cluster observability (see below) |

## Cluster tab (Spark-style)

The Cluster tab polls `GET /cluster/overview` every 2 seconds. Sub-views:

| sub-tab | what it shows |
|---|---|
| **Overview** | Running jobs, executor strip, top expert usage with assigned nodes |
| **Jobs** | All proxied requests (running highlighted); click for detail; drill through to executor |
| **Executors** | Clickable node list → hardware, owned experts, remote targets, live EMAP heatmap |
| **Placement** | Per-node cards: assigned experts, resident tier (disk/RAM/VRAM color-coded) |
| **RPC** | Latency bars and matrix between node pairs |

Click-through flow: job → executor → heatmap. Designed for the same mental model as
Apache Spark's Jobs / Stages / Executors UI — but for MoE expert routing instead of RDD partitions.

## API helpers

Cluster routes are wrapped in `src/lib/api.ts`:

- `getClusterOverview()` — combined snapshot for the dashboard
- `getClusterNodes()` — node registry
- `getClusterJobs()` — active + recent requests
- `getClusterPlacement()` — scheduler output

## Local validation

```sh
npm test
npm run build
```

The test suite stays browser-light: API requests use a mocked `fetch`.

The endpoint and selected model are persisted locally. API keys are intentionally
memory-only.

## Upstream

The UI is shared with [Colibri](https://github.com/JustVugg/colibri). Aviary adds
the Cluster tab and cluster API helpers in `src/lib/api.ts` and `src/Cluster.tsx`.
