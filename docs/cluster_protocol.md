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
  "uptime_sec": 3600.5
}
```

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

Phase 2 reserves (ignored by Phase 1 agents):

```
LOAD <layer> <eid> <tier>\n
EVICT <layer> <eid>\n
PIN <layer> <eid> <tier>\n
```

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
| `/experts` | GET | Aggregated cluster EMAP (first node or merged in dashboard) |

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

## Phase 2 preview (not implemented in Phase 1)

Expert execution RPC between agents uses the same line+byte-count philosophy:

```
EXEC_EXPERT <req_id> <layer> <eid> <bytes>\n<hidden_dim floats>\n
EXPERT_RESULT <req_id> <bytes>\n<hidden_dim floats>\n
EXPERT_MISS <req_id>\n
```

See [`aviary-cluster-plan.md`](../aviary-cluster-plan.md) for the full roadmap.
