# Aviary — cluster overlay on Colibri

Aviary is a **control-plane overlay** for [Colibri](https://github.com/JustVugg/colibri)
(and this tree’s Hy3-enabled base from [`ErikTromp/colibri-hy3`](https://github.com/ErikTromp/colibri-hy3)).
It turns N single-node Colibri engines into a request-routed cluster with a Spark-style
node dashboard.

## Ownership boundary

| Layer | Owns | Paths |
|---|---|---|
| **Colibri** | Engines, serve protocol, OpenAI HTTP, model detection | `c/*.c`, `c/openai_server.py`, `c/coli` (except master/agent), shared headers |
| **Aviary** | Cluster registry, agent/master, cluster protocol, Cluster UI | `c/aviary/**`, `docs/cluster_protocol.md`, `coli master` / `coli agent`, `web/src/Cluster.tsx` (+ thin tab wiring) |

**Rule:** Aviary must not permanently fork per-model engines (`hy3.c`, `kimi_k3.c`,
`inkling.c`, `colibri.c`, …). New Colibri families work automatically once
`coli`/`openai_server` resolve them.

Aviary’s only runtime contract with Colibri:

1. OpenAI-compatible HTTP (`POST /v1/chat/completions`, …)
2. `GET /health`, `GET /experts`, `GET /profile`
3. Serve-protocol telemetry when the engine emits it (`HWINFO` / `TIERS` / `EMAP` / `HITS`)

## Synced Colibri base

This repository’s Colibri sources track:

```
ErikTromp/colibri-hy3 main @ a6066aaaa4970e4948f245cde6c9dbc08cfc0903
```

Refresh with:

```bash
git fetch colibri-hy3 main
git checkout colibri-hy3/main -- c/ colibri/ docs/ web/ Makefile pyproject.toml ...
# then re-apply Aviary overlay (master/agent CLI, Cluster tab, c/aviary/)
```

Hy3 mux + dashboard telemetry in `hy3.c` is a **Colibri-base parity** patch (same
capability as `colibri.c` / Inkling serve), not Aviary-specific logic — intended for
upstreaming to ErikTromp / JustVugg.

## Commands

```bash
# master
./coli master --host 0.0.0.0 --port 9000

# worker (any supported model dir)
COLI_MODEL=/path/to/model ./coli agent --master http://master:9000 --port 8001
```

See [`cluster_protocol.md`](cluster_protocol.md) for the agent⇄master wire format.
