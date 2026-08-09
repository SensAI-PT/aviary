"""Aviary worker agent — local Engine + control-plane heartbeat.

Model-agnostic: uses Colibri's model_arch / argv_for_arch / Engine resolution.
"""

from __future__ import annotations

import json
import os
import signal
import socket
import sys
import threading
import time
from pathlib import Path
from urllib.parse import urlparse

from aviary.expert_rpc import DEFAULT_EXPERT_PORT, start_expert_rpc_server
from aviary.identity import load_or_create_node_id
from aviary.prefetch import PrefetchDaemon
from aviary.protocol import (
    ProtocolError,
    heartbeat_frame,
    read_frame,
    register_frame,
    write_line,
)
from aviary.registry import CONTROL_IDLE_TIMEOUT_MS, DEFAULT_HEARTBEAT_SEC
from aviary.trace import TraceBuffer
from aviary.usage import arch_engine_id, read_coli_usage, usage_delta

try:
    from openai_server import (
        APIServer,
        DEFAULT_CORS_ORIGINS,
        Engine,
        default_engine,
        model_arch,
        model_id_for_arch,
    )
    from resource_plan import analyze_model, discover_gpus, memory_available
except ImportError:
    from openai_server import (  # type: ignore
        APIServer,
        DEFAULT_CORS_ORIGINS,
        Engine,
        default_engine,
        model_arch,
        model_id_for_arch,
    )
    discover_gpus = memory_available = analyze_model = None  # type: ignore


def _local_ip_for(peer: tuple[str, int]) -> str:
    try:
        probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        probe.connect(peer)
        host = probe.getsockname()[0]
        probe.close()
        return host
    except OSError:
        return "127.0.0.1"


class ControlConnection:
    def __init__(self, node_id, master_host, control_port, http_port, model_id, model_path,
                 advertise_host, engine, scheduler, arch, expert_port, placement_path,
                 trace_buffer: TraceBuffer | None = None):
        self.node_id = node_id
        self.master_host = master_host
        self.control_port = control_port
        self.http_port = http_port
        self.model_id = model_id
        self.model_path = model_path
        self.advertise_host = advertise_host
        self.engine = engine
        self.scheduler = scheduler
        self.arch = arch
        self.expert_port = expert_port
        self.placement_path = placement_path
        self.trace_buffer = trace_buffer
        self._usage_snapshot: list[dict] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="aviary-control", daemon=True)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=5)

    def _write_placement(self, payload: dict) -> None:
        self.placement_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.placement_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
        tmp.replace(self.placement_path)

    def _handle_control(self, kind: str, fields: list[str], payload: dict | None) -> None:
        if kind == "PLACEMENT" and payload is not None:
            self._write_placement(payload)
        elif kind == "PIN" and len(fields) >= 4:
            exec_fn = getattr(self.engine, "exec_cluster_cmd", None)
            if exec_fn:
                try:
                    exec_fn("PIN", int(fields[1]), int(fields[2]), int(fields[3]))
                except (ValueError, TypeError) as error:
                    print(f"[aviary-agent] PIN failed: {error}", file=sys.stderr)
        elif kind == "LOAD" and len(fields) >= 4:
            exec_fn = getattr(self.engine, "exec_cluster_cmd", None)
            if exec_fn:
                try:
                    exec_fn("LOAD", int(fields[1]), int(fields[2]), int(fields[3]))
                except (ValueError, TypeError) as error:
                    print(f"[aviary-agent] LOAD failed: {error}", file=sys.stderr)
        elif kind == "EVICT" and len(fields) >= 3:
            exec_fn = getattr(self.engine, "exec_cluster_cmd", None)
            if exec_fn:
                try:
                    exec_fn("EVICT", int(fields[1]), int(fields[2]))
                except (ValueError, TypeError) as error:
                    print(f"[aviary-agent] EVICT failed: {error}", file=sys.stderr)

    def _register_payload(self) -> dict:
        payload = {
            "host": self.advertise_host,
            "model_path": self.model_path,
            "arch": self.arch,
            "engine_id": arch_engine_id(self.arch),
            "expert_port": self.expert_port,
        }
        eng = self.engine
        if getattr(eng, "hwinfo", None):
            payload["hwinfo"] = eng.hwinfo
        if getattr(eng, "tiers", None):
            payload["tiers"] = eng.tiers
        if discover_gpus and not payload.get("hwinfo"):
            try:
                gpus = discover_gpus()
                ram_total, ram_avail = memory_available()
                payload["hwinfo"] = {
                    "cores": os.cpu_count() or 0,
                    "ram_total_gb": ram_total,
                    "ram_avail_gb": ram_avail,
                    "gpus": len(gpus),
                    "vram_total_gb": sum(g.get("memory_total_gb", 0) for g in gpus),
                    "cpu": "",
                    "gpu": gpus[0].get("name", "") if gpus else "",
                }
            except Exception:
                pass
        if analyze_model:
            try:
                plan = analyze_model(self.model_path)
                payload["tiers"] = payload.get("tiers") or {
                    "vram": plan.get("vram_experts", 0),
                    "ram": plan.get("ram_experts", 0),
                    "disk": plan.get("disk_experts", 0),
                    "vram_gb": plan.get("vram_gb", 0),
                    "ram_gb": plan.get("ram_gb", 0),
                }
            except Exception:
                pass
        return payload

    def _telemetry_payload(self) -> dict:
        eng = self.engine
        snap = self.scheduler.snapshot()
        usage_path = Path(self.model_path) / ".coli_usage"
        _, _, engine_id, records = read_coli_usage(usage_path)
        delta = usage_delta(self._usage_snapshot, records)
        self._usage_snapshot = records
        payload = {
            "uptime_sec": time.time() - self._started,
            "hits": getattr(eng, "hits", None) or "",
            "hits_seq": getattr(eng, "hits_seq", 0),
            "arch": self.arch,
            "engine_id": engine_id or arch_engine_id(self.arch),
            "usage": delta or records[-64:],
            "costs": list(getattr(eng, "ecost", []) or []),
            "profile": list(getattr(eng, "profile", ()) or ())[-8:],
        }
        if getattr(eng, "hwinfo", None):
            payload["hwinfo"] = eng.hwinfo
        if getattr(eng, "tiers", None):
            payload["tiers"] = eng.tiers
        if getattr(eng, "emap", None):
            payload["emap"] = eng.emap
        payload["scheduler"] = snap
        if self.trace_buffer:
            payload["trace_events"] = self.trace_buffer.drain()
        server = getattr(self.scheduler, "server", None)
        if server and hasattr(server, "rpc_samples"):
            with server._rpc_samples_lock:
                payload["rpc_samples"] = list(server.rpc_samples)
                server.rpc_samples.clear()
        return payload

    def _run(self):
        self._started = time.time()
        while not self._stop.is_set():
            sock = None
            try:
                sock = socket.create_connection((self.master_host, self.control_port), timeout=5)
                if not self.advertise_host or self.advertise_host == "0.0.0.0":
                    self.advertise_host = _local_ip_for(sock.getpeername())
                reg = register_frame(self.node_id, self.http_port, self.model_id,
                                     self._register_payload())
                sock.sendall(reg.encode("utf-8"))
                kind, fields, _ = read_frame(sock, CONTROL_IDLE_TIMEOUT_MS)
                if kind != "REGISTERED" or len(fields) < 2 or fields[1] != self.node_id:
                    time.sleep(2)
                    continue
                next_beat = time.time()
                while not self._stop.is_set():
                    now = time.time()
                    if now >= next_beat:
                        inflight = self.scheduler.snapshot().get("active", 0)
                        frame = heartbeat_frame(self.node_id, inflight, self._telemetry_payload())
                        sock.sendall(frame.encode("utf-8"))
                        try:
                            kind, fields, payload = read_frame(sock, CONTROL_IDLE_TIMEOUT_MS)
                            if kind == "PING":
                                write_line(sock, "PONG")
                            elif kind == "DRAIN":
                                break
                            else:
                                self._handle_control(kind, fields, payload)
                        except ProtocolError:
                            break
                        next_beat = now + DEFAULT_HEARTBEAT_SEC
                    try:
                        kind, fields, payload = read_frame(
                            sock, int(max(100, (next_beat - time.time()) * 1000)))
                        if kind == "EOF":
                            break
                        if kind == "PING":
                            write_line(sock, "PONG")
                        elif kind == "DRAIN":
                            break
                        else:
                            self._handle_control(kind, fields, payload)
                    except (ProtocolError, socket.timeout):
                        pass
            except OSError as error:
                print(f"[aviary-agent] control connection failed: {error}", file=sys.stderr)
            finally:
                if sock:
                    try:
                        write_line(sock, f"DEREGISTER {self.node_id}")
                    except OSError:
                        pass
                    sock.close()
            if not self._stop.is_set():
                time.sleep(2)


def _agent_allowed_hosts(advertise, allowed_hosts):
    """DNS-rebinding guard extras: master proxies with Host=<advertise>:port."""
    hosts = [h.strip().lower() for h in (allowed_hosts or []) if h and h.strip()]
    for h in os.environ.get("COLI_ALLOWED_HOSTS", "").split(","):
        h = h.strip().lower()
        if h and h not in hosts:
            hosts.append(h)
    if advertise:
        ah = advertise.strip().lower()
        if ah not in hosts:
            hosts.append(ah)
    return tuple(hosts)


def run_agent(model, master_url, host="127.0.0.1", port=8001, model_id=None, api_key=None,
              control_port=None, advertise_host=None, cap=None, max_tokens=1024, env=None,
              max_queue=8, queue_timeout=300, kv_slots=1, engine_path=None, allowed_hosts=None):
    import openai_server

    parsed = urlparse(master_url if "://" in master_url else f"http://{master_url}")
    master_host = parsed.hostname or "127.0.0.1"
    control_port = control_port or int(os.environ.get("AVIARY_CONTROL_PORT", "9002"))
    expert_port = int(os.environ.get("AVIARY_EXPERT_PORT", str(DEFAULT_EXPERT_PORT)))
    node_id = load_or_create_node_id(model)
    arch = model_arch(model)
    openai_server.ARCH = arch
    model_id = model_id or os.environ.get("COLI_MODEL_ID") or model_id_for_arch(arch)
    advertise = advertise_host or (host if host not in ("0.0.0.0", "::") else None)
    if not advertise and host in ("0.0.0.0", "::"):
        advertise = _local_ip_for((master_host, control_port))
    engine_bin = engine_path or str(default_engine())
    placement_path = Path(model) / ".aviary_placement.json"

    child_env = dict(env or os.environ)
    if os.environ.get("AVIARY_CLUSTER", "0") not in ("0", ""):
        child_env["AVIARY_CLUSTER"] = "1"
        child_env["AVIARY_PLACEMENT"] = str(placement_path)

    server = APIServer((host, port), None, model_id, api_key, max_tokens,
                       cors_origins=DEFAULT_CORS_ORIGINS,
                       max_queue=max_queue, queue_timeout=queue_timeout,
                       kv_slots=kv_slots,
                       allowed_hosts=_agent_allowed_hosts(advertise, allowed_hosts))
    runtime = Engine(engine_bin, model, cap, max_tokens, child_env, kv_slots, arch)
    server.engine = runtime
    server.model_path = model
    server.node_id = node_id
    server.trace_buffer = TraceBuffer()
    server.scheduler.server = server

    rpc_host = advertise or host
    if rpc_host in ("0.0.0.0", "::"):
        rpc_host = "0.0.0.0"
    rpc_server, _, _ = start_expert_rpc_server(rpc_host, expert_port, runtime,
                                              trace_buffer=server.trace_buffer,
                                              node_id=node_id)

    control = ControlConnection(node_id, master_host, control_port, port, model_id, model,
                                advertise, runtime, server.scheduler, arch, expert_port,
                                placement_path, server.trace_buffer)
    control.start()

    prefetch = None
    if os.environ.get("AVIARY_PREFETCH", "0") not in ("0", ""):
        master_http = master_url if "://" in master_url else f"http://{master_url}"
        prefetch = PrefetchDaemon(model, placement_path, master_http, node_id, api_key or "")
        prefetch.start()

    print(f"Aviary agent {node_id} arch={arch} listening on http://{host}:{port}/v1 "
          f"(master control {master_host}:{control_port}, expert RPC :{expert_port})",
          file=sys.stderr)

    previous = signal.getsignal(signal.SIGTERM)

    def _shutdown(*_):
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, _shutdown)
    try:
        server.serve_forever()
    finally:
        signal.signal(signal.SIGTERM, previous)
        control.stop()
        if prefetch:
            prefetch.stop()
        rpc_server.shutdown()
        server.scheduler.close()
        server.server_close()
        runtime.close()
