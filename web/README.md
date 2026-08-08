# Aviary web dashboard

React/Vite interface for an OpenAI-compatible **Aviary master** or single-node
[`coli serve`](https://github.com/JustVugg/colibri) backend.

In cluster mode, point the server URL at the **master** (`http://host:9000/v1`) —
not an individual agent. The master proxies chat completions and exposes cluster
routes (`/cluster/nodes`) for the **Cluster** tab.

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
| **Cluster** | N/A (shows “not a master”) | Live node list, hardware, load, per-node EMAP heatmaps |

Local validation:

```sh
npm test
npm run build
```

The test suite stays browser-light: API requests use a mocked `fetch`. It checks
that `/health`, `/profile`, and `/cluster/nodes` resolve next to (not below) the
OpenAI `/v1` prefix, supports both boolean and numeric `scheduler.active` responses,
and sends the Colibri-specific `cache_slot` field only when KV-slot support was
advertised.

The endpoint and selected model are persisted locally. API keys are intentionally
memory-only; startup/persistence also removes the legacy `colibri.apiKey` value.

## Upstream

The UI is shared with [Colibri](https://github.com/JustVugg/colibri). Aviary adds
the Cluster tab and cluster API helpers in `src/lib/api.ts` and `src/Cluster.tsx`.
