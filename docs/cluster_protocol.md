# The cluster protocol — agent ⇄ master control plane

Aviary adds a **control-plane** wire format between worker agents and the cluster
master. It follows the same design as [`serve_protocol.md`](serve_protocol.md):
plain-text line headers, byte-counted JSON payloads, one `fflush` per write, unknown
line kinds ignored for forward compatibility.

The **data plane** (OpenAI HTTP requests) stays on each agent's local HTTP port in
Phase 1. The master proxies `POST /v1/chat/completions` to a chosen agent; activations
and expert RPC are deferred to Phase 2.

## Roles

| role | process | responsibility |
|---|---|---|
| **master** | `coli master` | Node registry, heartbeat lease, request routing, cluster dashboard host |
| **agent** | `coli agent` | Wraps one local `Engine` subprocess; relays telemetry; serves local HTTP API |

## Node identity

Each agent has a stable UUID persisted at `<model_dir>/.aviary_node_id` (same
persistence pattern as `.coli_usage`). Generated on first run with `uuid4`; reused
across restarts so the master can correlate history.

## Transport

- **Control channel:** TCP from agent → master on `--control-port` (default `9002`).
  One persistent connection per agent; length-framed lines as below.
- **Agent HTTP:** each agent exposes `--port` (default `8001`) with the same routes as
  `coli serve` (`/v1/chat/completions`, `/health`, `/experts`, `/profile`).
- **Master HTTP:** `--port` (default `9000`) for proxied API + cluster routes; serves
  `web/dist` like `coli web`.

## Environment

| variable | default | meaning |
|---|---|---|
| `AVIARY_RPC_TIMEOUT_MS` | `150` | Control-plane read/write deadline (ms) |
| `AVIARY_HEARTBEAT_SEC` | `2` | Agent heartbeat interval |
| `AVIARY_HEARTBEAT_MISS` | `3` | Missed heartbeats before master evicts a node |

## Agent → master frames

```
REGISTER <node_id> <agent_http_port> <model_id> <bytes>\n<json_payload>\n
HEARTBEAT <node_id> <inflight> <bytes>\n<json_telemetry>\n
DEREGISTER <node_id>\n
```

### `REGISTER`

Sent once when the control connection opens (before the first `HEARTBEAT`).

- `node_id` — UUID from `.aviary_node_id`
- `agent_http_port` — local HTTP port the master should proxy to
- `model_id` — OpenAI model id string (same as `coli serve --model-id`)
- `bytes` — exact UTF-8 length of the JSON payload
- `json_payload` — object with at least:

```json
{
  "host": "10.0.0.5",
  "hwinfo": { "cores": 32, "ram_total_gb": 128.0, "ram_avail_gb": 96.0, "gpus": 1, "vram_total_gb": 24.0, "cpu": "...", "gpu": "..." },
  "tiers": { "vram": 0, "ram": 12, "disk": 20352, "vram_gb": 0.0, "ram_gb": 4.2 },
  "model_path": "/path/to/hy3_i4"
}
```

The master replies:

```
REGISTERED <node_id>\n
```

or on rejection:

```
ERROR <node_id> <CODE>\n
```

Codes: `DUPLICATE`, `BAD_FRAME`, `MASTER_FULL`.

### `HEARTBEAT`

Sent every `AVIARY_HEARTBEAT_SEC` (and optionally when `hits_seq` changes).

- `inflight` — active generation count on this agent
- `json_payload` — telemetry relay (not re-derived):

```json
{
  "hwinfo": { ... },
  "tiers": { ... },
  "emap": { "rows": 79, "cols": 256, "map": "..." },
  "hits": "<hex>",
  "hits_seq": 42,
  "uptime_sec": 3600.5,
  "arch": "hy3",
  "engine_id": 1234567890,
  "usage": [{"layer": 12, "expert": 45, "count": 3}],
  "costs": [{"layer": 12, "expert": 45, "tier": 1, "load_us": 0, "exec_us": 42000}],
  "profile": [{"wall_s": 0.5, "expert_disk_s": 0.01, "expert_matmul_s": 0.2}]
}
```

Phase 2 fields (`arch`, `engine_id`, `usage`, `costs`, `profile`) are owned by agents;
the master aggregates them for placement but does not write `.coli_usage` itself.
Stats are never merged across different `engine_id` values (one LLM family per cohort).

Master reply (optional, may be omitted under load):

```
HEARTBEAT_ACK <node_id>\n
```

### `DEREGISTER`

Graceful shutdown. Master removes the node immediately and replies:

```
DEREGISTERED <node_id>\n
```

## Master → agent frames

Phase 1 control commands:

```
PING\n
DRAIN\n
```

| frame | meaning |
|---|---|
| `PING` | Agent must reply `PONG\n` within `AVIARY_RPC_TIMEOUT_MS` |
| `DRAIN` | Stop accepting new HTTP work; finish in-flight; agent may reconnect after drain |

Phase 2 control commands:

```
PLACEMENT <bytes>\n<json_payload>\n
PIN <layer> <eid> <tier>\n
LOAD <layer> <eid> <tier>\n
EVICT <layer> <eid>\n
```

| frame | meaning |
|---|---|
| `PLACEMENT` | Full routing table for this agent (`node_id`, `peers`, `experts`) — written to `AVIARY_PLACEMENT` |
| `PIN` | Pin expert to RAM/VRAM tier on this node |
| `LOAD` | Prefetch expert into tier |
| `EVICT` | Evict expert from hot store |

## Failure semantics

- Any control read/write past `AVIARY_RPC_TIMEOUT_MS` is a miss; the agent reconnects
  with a fresh `REGISTER`.
- Master evicts a node after `AVIARY_HEARTBEAT_MISS` consecutive missed heartbeats.
- Evicted nodes are not routed new requests. In-flight proxied streams fail with HTTP
  502.
- Phase 1 never requires the network for correctness: each agent serves from local
  disk/RAM/VRAM tiers exactly like single-node `coli serve`.

## Master HTTP surface (Phase 1)

| route | method | response |
|---|---|---|
| `/cluster/health` | GET | `{ "status": "ok", "nodes": N, "healthy": M }` |
| `/cluster/nodes` | GET | `{ "nodes": [ { "node_id", "endpoint", "status", "hwinfo", "tiers", "emap", "inflight", "uptime_sec", "last_heartbeat_age_sec" }, ... ] }` |
| `/v1/chat/completions` | POST | Proxied to least-loaded healthy agent (streaming preserved) |
| `/v1/models` | GET | From first healthy agent |
| `/health` | GET | Master liveness + scheduler snapshot |
| `/cluster/placement` | GET | Current scheduler output (`experts`, `rpc_matrix_us`, …) |
| `/cluster/costs` | GET | Cost matrix snapshot for debugging |
| `/cluster/jobs` | GET | Active and recent proxied requests (`active`, `history`) |
| `/cluster/overview` | GET | Combined snapshot for the Spark-style dashboard UI |

Auth follows `coli serve`: `COLI_API_KEY` / `--api-key` when set.

## Routing policy (v1)

Pick the healthy node with the lowest `inflight` count. Tie-break: earliest
`last_heartbeat`. No cross-node expert placement in Phase 1.

## Example session

```
agent → master: REGISTER 550e8400-e29b-41d4-a716-446655440000 8001 hy3-colibri 256\n{"host":"127.0.0.1",...}\n
master → agent: REGISTERED 550e8400-e29b-41d4-a716-446655440000\n
agent → master: HEARTBEAT 550e8400-e29b-41d4-a716-446655440000 0 180\n{"hwinfo":{...},"tiers":{...},"emap":{...},"hits":"","hits_seq":0,"uptime_sec":1.2}\n
master → agent: HEARTBEAT_ACK 550e8400-e29b-41d4-a716-446655440000\n
...
agent → master: DEREGISTER 550e8400-e29b-41d4-a716-446655440000\n
master → agent: DEREGISTERED 550e8400-e29b-41d4-a716-446655440000\n
```

## Phase 2 — expert RPC and placement (implemented)

**Design note:** the shipped implementation uses **expert-level** cost-aware placement
(`EXEC_EXPERT`, per-expert routing table), not depth-wise block pipeline placement from the
original roadmap. This keeps hop count bounded by hot-expert count rather than layer count.

Expert execution RPC between agents uses the same line+byte-count philosophy:

```
EXEC_EXPERT <req_id> <layer> <eid> <bytes> [<job_id>]\n<hidden_dim floats>\n
EXPERT_RESULT <req_id> <bytes>\n<hidden_dim floats>\n
EXPERT_MISS <req_id>\n
```

Optional `<job_id>` correlates inbound expert serves with the master's chat job trace.
The engine mux also accepts `CLUSTER_JOB <job_id>\n` before each generation turn so
`TRACE layer eid kind peer rpc_us` lines on stdout attach to that job.

The master pushes `PIN`/`LOAD`/`EVICT` control lines after each `PLACEMENT` update. Agents
forward these to the engine mux as:

```
CLUSTER_PIN <req_id> <layer> <eid> <tier>\n
CLUSTER_LOAD <req_id> <layer> <eid> <tier>\n
CLUSTER_EVICT <req_id> <layer> <eid>\n
CLUSTER_OK <req_id>\n
CLUSTER_MISS <req_id>\n
```

Agents expose peer shard fetch for Phase 4 prefetch:

```
GET /cluster/shards
GET /cluster/shard?name=<safetensors_filename>
```

Heartbeats may include `trace_events` (per-request RPC trace) and `rpc_samples` (latency
measurements) for the Cluster dashboard.

See [`AVIARY.md`](AVIARY.md) for the cluster roadmap and feature status.
