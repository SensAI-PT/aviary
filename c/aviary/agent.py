"""Aviary worker agent — local Engine + control-plane heartbeat.

Model-agnostic: uses Colibri's model_arch / argv_for_arch / Engine resolution.
"""

from __future__ import annotations

import errno
import json
import os
import queue
import signal
import socket
import sys
import threading
import time
from pathlib import Path
from typing import Any
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
from aviary.tiers_util import sanitize_tiers
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


def _is_loopback_host(host: str | None) -> bool:
    return (host or "").strip().lower() in ("127.0.0.1", "localhost", "::1")


def _resolve_cluster_advertise_host(advertise: str | None, bind_host: str,
                                    master_host: str, control_port: int) -> str:
    """Peers and master must reach this agent on a LAN-routable address.

    Replaces loopback advertise when binding on all interfaces, even if the
    operator forgot AVIARY_CLUSTER=1 (otherwise the registry shows 127.0.0.1).
    """
    cluster = os.environ.get("AVIARY_CLUSTER", "0") not in ("0", "")
    chosen = advertise or (bind_host if bind_host not in ("0.0.0.0", "::") else None)
    if chosen and not _is_loopback_host(chosen):
        return chosen
    # Loopback / unbound: resolve a LAN IP whenever we bind publicly or cluster is on.
    if not cluster and bind_host not in ("0.0.0.0", "::") and not _is_loopback_host(bind_host):
        return chosen or bind_host
    peer = master_host if not _is_loopback_host(master_host) else ("8.8.8.8", 53)
    if isinstance(peer, str):
        peer = (peer, control_port)
    resolved = _local_ip_for(peer)
    if _is_loopback_host(resolved):
        print("[aviary-agent] WARNING: advertise host is loopback; "
              "pass --advertise-host <LAN-IP> so peers can reach this agent",
              file=sys.stderr)
        return resolved
    if chosen and _is_loopback_host(chosen):
        print(f"[aviary-agent] replacing loopback advertise {chosen!r} "
              f"with {resolved!r}", file=sys.stderr)
    return resolved

def infer_ram_budget_gb(env: dict | None = None, hwinfo: dict | None = None) -> float:
    """Explicit --ram, else a conservative slice of advertised host RAM."""
    e = env or os.environ
    ram = float(e.get("RAM_GB") or 0)
    if ram > 0:
        return ram
    hw = hwinfo or {}
    avail = float(hw.get("ram_avail_gb") or 0)
    total = float(hw.get("ram_total_gb") or 0)
    if avail > 1:
        return round(min(avail * 0.5, total * 0.5 if total > 1 else avail * 0.5), 2)
    if total > 1:
        return round(total * 0.4, 2)
    return 0.0


def tiers_config_from_env(env: dict | None = None, hwinfo: dict | None = None) -> dict[str, float]:
    """Configured RAM/VRAM budgets from --ram/--vram, else inferred from hwinfo."""
    e = env or os.environ
    ram = infer_ram_budget_gb(e, hwinfo)
    vram = float(e.get("CUDA_EXPERT_GB") or e.get("VRAM_GB") or 0)
    return {k: v for k, v in (("ram_gb", ram), ("vram_gb", vram)) if v > 0}


def _probe_hwinfo() -> dict[str, Any] | None:
    if not (discover_gpus and memory_available):
        return None
    try:
        gpus = discover_gpus()
        ram_total, ram_avail = memory_available()
        return {
            "cores": os.cpu_count() or 0,
            "ram_total_gb": ram_total,
            "ram_avail_gb": ram_avail,
            "gpus": len(gpus),
            "vram_total_gb": sum(g.get("memory_total_gb", 0) for g in gpus),
            "cpu": "",
            "gpu": gpus[0].get("name", "") if gpus else "",
        }
    except Exception:
        return None


def attach_tier_telemetry(payload: dict, engine, tiers_config: dict[str, float] | None,
                          hwinfo: dict | None = None) -> dict:
    """Always publish configured + occupied RAM/VRAM (synthesize if engine is silent)."""
    hw = getattr(engine, "hwinfo", None) or hwinfo or payload.get("hwinfo")
    if hw:
        payload["hwinfo"] = hw
    elif not payload.get("hwinfo"):
        probed = _probe_hwinfo()
        if probed:
            payload["hwinfo"] = probed
            hw = probed
    cfg = dict(tiers_config or {})
    if "ram_gb" not in cfg or not cfg.get("ram_gb"):
        cfg.update(tiers_config_from_env(hwinfo=hw))
    if cfg:
        payload["tiers_config"] = cfg
    tiers = sanitize_tiers(getattr(engine, "tiers", None))
    if not tiers and cfg:
        tiers = {
            "ram_gb": cfg.get("ram_gb", 0),
            "vram_gb": cfg.get("vram_gb", 0),
            "ram": 0, "vram": 0, "disk": 0,
        }
    if tiers:
        payload["tiers"] = tiers
    gpu_tier = getattr(engine, "gpu_tier", None)
    if gpu_tier:
        payload["gpu_tier"] = dict(gpu_tier)
    return payload


class ControlConnection:
    def __init__(self, node_id, master_host, control_port, http_port, model_id, model_path,
                 advertise_host, engine, scheduler, arch, expert_port, placement_path,
                 trace_buffer: TraceBuffer | None = None,
                 tiers_config: dict[str, float] | None = None):
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
        self.tiers_config = dict(tiers_config or tiers_config_from_env())
        self._usage_snapshot: list[dict] = []
        self._stop = threading.Event()
        self._sock = None
        self._sock_lock = threading.Lock()
        # Apply RELOAD/PIN/LOAD/EVICT off the control thread so heartbeats keep flowing
        # while the mux drains CLUSTER_* between tokens (can take > CONTROL_IDLE_TIMEOUT).
        self._cmd_q: queue.SimpleQueue = queue.SimpleQueue()
        self._thread = threading.Thread(target=self._run, name="aviary-control", daemon=True)
        self._cmd_thread = threading.Thread(target=self._cmd_worker, name="aviary-cluster-cmd",
                                            daemon=True)

    def start(self):
        self._cmd_thread.start()
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=5)
        self._cmd_thread.join(timeout=5)

    def _send(self, data: bytes | str) -> None:
        raw = data.encode("utf-8") if isinstance(data, str) else data
        with self._sock_lock:
            sock = self._sock
            if sock is None:
                raise OSError("control socket closed")
            sock.sendall(raw)

    def _write_placement(self, payload: dict) -> None:
        self.placement_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.placement_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
        tmp.replace(self.placement_path)

    def _enqueue_cluster_cmd(self, cmd: str, layer: int = 0, eid: int = 0, tier: int = 0) -> None:
        self._cmd_q.put((cmd, layer, eid, tier))

    def _cmd_worker(self) -> None:
        while not self._stop.is_set():
            try:
                cmd, layer, eid, tier = self._cmd_q.get(timeout=0.5)
            except queue.Empty:
                continue
            exec_fn = getattr(self.engine, "exec_cluster_cmd", None)
            if not exec_fn:
                continue
            try:
                exec_fn(cmd, layer, eid, tier)
            except (ValueError, TypeError) as error:
                print(f"[aviary-agent] {cmd} failed: {error}", file=sys.stderr)

    def _handle_control(self, kind: str, fields: list[str], payload: dict | None) -> None:
        if kind == "PLACEMENT" and payload is not None:
            self._write_placement(payload)
            self._enqueue_cluster_cmd("RELOAD")
        elif kind == "PIN" and len(fields) >= 4:
            self._enqueue_cluster_cmd("PIN", int(fields[1]), int(fields[2]), int(fields[3]))
        elif kind == "LOAD" and len(fields) >= 4:
            self._enqueue_cluster_cmd("LOAD", int(fields[1]), int(fields[2]), int(fields[3]))
        elif kind == "EVICT" and len(fields) >= 3:
            self._enqueue_cluster_cmd("EVICT", int(fields[1]), int(fields[2]))
        elif kind == "RESET_USAGE":
            self._reset_usage()

    def _reset_usage(self) -> None:
        model_dir = Path(self.model_path)
        for name in (".coli_usage", "stats.txt"):
            path = model_dir / name
            if path.is_file():
                path.unlink()
        self._usage_snapshot = []

    def _register_payload(self) -> dict:
        payload = {
            "host": self.advertise_host,
            "model_path": self.model_path,
            "arch": self.arch,
            "engine_id": arch_engine_id(self.arch),
            "expert_port": self.expert_port,
        }
        return attach_tier_telemetry(payload, self.engine, self.tiers_config)

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
        attach_tier_telemetry(payload, eng, self.tiers_config)
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

    def push_trace_now(self) -> None:
        """Flush pending TRACE events to master without waiting for the next heartbeat."""
        if not self.trace_buffer:
            return
        events = self.trace_buffer.drain()
        if not events:
            return
        try:
            snap = self.scheduler.snapshot()
            inflight = int(snap.get("active", 0)) + int(snap.get("queued", 0))
            frame = heartbeat_frame(self.node_id, inflight, {"trace_events": events})
            self._send(frame)
        except OSError:
            pass

    def _run(self):
        self._started = time.time()
        while not self._stop.is_set():
            sock = None
            try:
                sock = socket.create_connection((self.master_host, self.control_port), timeout=5)
                with self._sock_lock:
                    self._sock = sock
                if not self.advertise_host or self.advertise_host == "0.0.0.0":
                    self.advertise_host = _local_ip_for(sock.getpeername())
                reg = register_frame(self.node_id, self.http_port, self.model_id,
                                     self._register_payload())
                self._send(reg)
                kind, fields, _ = read_frame(sock, CONTROL_IDLE_TIMEOUT_MS)
                if kind != "REGISTERED" or len(fields) < 2 or fields[1] != self.node_id:
                    time.sleep(2)
                    continue
                next_beat = time.time()
                while not self._stop.is_set():
                    now = time.time()
                    if now >= next_beat:
                        snap = self.scheduler.snapshot()
                        inflight = int(snap.get("active", 0)) + int(snap.get("queued", 0))
                        frame = heartbeat_frame(self.node_id, inflight, self._telemetry_payload())
                        self._send(frame)
                        try:
                            kind, fields, payload = read_frame(sock, CONTROL_IDLE_TIMEOUT_MS)
                            if kind == "PING":
                                self._send("PONG\n")
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
                            self._send("PONG\n")
                        elif kind == "DRAIN":
                            break
                        else:
                            self._handle_control(kind, fields, payload)
                    except (ProtocolError, socket.timeout):
                        pass
            except OSError as error:
                # Control plane is a raw TCP socket to master --control-port (default 9002),
                # not the master HTTP UI (--port, default 9000) and not local expert RPC.
                target = f"{self.master_host}:{self.control_port}"
                hint = ""
                err = str(error).lower()
                if "timed out" in err or getattr(error, "errno", None) in (
                        getattr(errno, "ETIMEDOUT", -1), getattr(errno, "EHOSTUNREACH", -1)):
                    hint = (f" — nothing answered TCP {target}. "
                            f"Master HTTP UI can still work on another port; "
                            f"confirm the master's --control-port (default 9002) "
                            f"matches this agent's --control-port, and that the "
                            f"firewall allows it.")
                elif getattr(error, "errno", None) in (
                        getattr(errno, "ECONNREFUSED", -1),):
                    hint = (f" — connection refused at {target}. "
                            f"Is the master running with --control-port "
                            f"{self.control_port}?")
                print(f"[aviary-agent] control connection to {target} failed: "
                      f"{error}{hint}", file=sys.stderr)
            finally:
                with self._sock_lock:
                    self._sock = None
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
              max_queue=8, queue_timeout=300, kv_slots=1, engine_path=None, allowed_hosts=None,
              expert_port=None):
    import openai_server

    parsed = urlparse(master_url if "://" in master_url else f"http://{master_url}")
    master_host = parsed.hostname or "127.0.0.1"
    control_port = control_port or int(os.environ.get("AVIARY_CONTROL_PORT", "9002"))
    expert_port = int(expert_port if expert_port is not None
                      else os.environ.get("AVIARY_EXPERT_PORT", str(DEFAULT_EXPERT_PORT)))
    node_id = load_or_create_node_id(model)
    arch = model_arch(model)
    openai_server.ARCH = arch
    model_id = model_id or os.environ.get("COLI_MODEL_ID") or model_id_for_arch(arch)
    advertise = advertise_host or (host if host not in ("0.0.0.0", "::") else None)
    if not advertise and host in ("0.0.0.0", "::"):
        advertise = _local_ip_for((master_host, control_port))
    advertise = _resolve_cluster_advertise_host(advertise, host, master_host, control_port)
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
    runtime.trace_buffer = server.trace_buffer
    runtime.node_id = node_id
    server.scheduler.server = server

    rpc_host = advertise or host
    if rpc_host in ("0.0.0.0", "::"):
        rpc_host = "0.0.0.0"
    rpc_server, _, _ = start_expert_rpc_server(rpc_host, expert_port, runtime,
                                              trace_buffer=server.trace_buffer,
                                              node_id=node_id)

    control = ControlConnection(node_id, master_host, control_port, port, model_id, model,
                                advertise, runtime, server.scheduler, arch, expert_port,
                                placement_path, server.trace_buffer,
                                tiers_config=tiers_config_from_env(child_env))
    server.control = control
    control.start()

    prefetch = None
    if os.environ.get("AVIARY_PREFETCH", "0") not in ("0", ""):
        master_http = master_url if "://" in master_url else f"http://{master_url}"
        prefetch = PrefetchDaemon(model, placement_path, master_http, node_id, api_key or "")
        prefetch.start()

    print(f"Aviary agent {node_id} arch={arch} listening on http://{host}:{port}/v1 "
          f"(master control tcp://{master_host}:{control_port}, "
          f"local expert RPC on {rpc_host}:{expert_port})",
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
