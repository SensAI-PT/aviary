"""Aviary worker agent — local Engine + control-plane heartbeat."""

from __future__ import annotations

import json
import os
import signal
import socket
import sys
import threading
import time
from urllib.parse import urlparse

from aviary.identity import load_or_create_node_id
from aviary.protocol import (
    DEFAULT_RPC_TIMEOUT_MS,
    ProtocolError,
    heartbeat_frame,
    read_frame,
    register_frame,
    write_line,
)
from aviary.registry import DEFAULT_HEARTBEAT_SEC

try:
    from openai_server import APIServer, Engine, default_engine, default_model_id, model_family, DEFAULT_CORS_ORIGINS
    from resource_plan import analyze_model, discover_gpus, memory_available
except ImportError:
    from openai_server import APIServer, Engine, default_engine, default_model_id, model_family, DEFAULT_CORS_ORIGINS  # type: ignore
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
                 advertise_host, engine, scheduler):
        self.node_id = node_id
        self.master_host = master_host
        self.control_port = control_port
        self.http_port = http_port
        self.model_id = model_id
        self.model_path = model_path
        self.advertise_host = advertise_host
        self.engine = engine
        self.scheduler = scheduler
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="aviary-control", daemon=True)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=5)

    def _register_payload(self) -> dict:
        payload = {
            "host": self.advertise_host,
            "model_path": self.model_path,
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
        payload = {
            "uptime_sec": time.time() - self._started,
            "hits": getattr(eng, "hits", None) or "",
            "hits_seq": getattr(eng, "hits_seq", 0),
        }
        if getattr(eng, "hwinfo", None):
            payload["hwinfo"] = eng.hwinfo
        if getattr(eng, "tiers", None):
            payload["tiers"] = eng.tiers
        if getattr(eng, "emap", None):
            payload["emap"] = eng.emap
        payload["scheduler"] = snap
        return payload

    def _run(self):
        self._started = time.time()
        timeout = DEFAULT_RPC_TIMEOUT_MS / 1000.0
        while not self._stop.is_set():
            sock = None
            try:
                sock = socket.create_connection((self.master_host, self.control_port), timeout=5)
                if not self.advertise_host or self.advertise_host == "0.0.0.0":
                    self.advertise_host = _local_ip_for(sock.getpeername())
                reg = register_frame(self.node_id, self.http_port, self.model_id,
                                     self._register_payload())
                sock.sendall(reg.encode("utf-8"))
                kind, fields, _ = read_frame(sock, DEFAULT_RPC_TIMEOUT_MS)
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
                            kind, _, _ = read_frame(sock, DEFAULT_RPC_TIMEOUT_MS)
                            if kind == "PING":
                                write_line(sock, "PONG")
                            elif kind == "DRAIN":
                                break
                        except ProtocolError:
                            break
                        next_beat = now + DEFAULT_HEARTBEAT_SEC
                    sock.settimeout(max(0.1, next_beat - time.time()))
                    try:
                        kind, fields, _ = read_frame(sock, int(max(100, (next_beat - time.time()) * 1000)))
                        if kind == "EOF":
                            break
                        if kind == "PING":
                            write_line(sock, "PONG")
                        elif kind == "DRAIN":
                            break
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


def run_agent(model, master_url, host="127.0.0.1", port=8001, model_id=None, api_key=None,
              control_port=None, advertise_host=None, cap=8, max_tokens=1024, env=None,
              max_queue=8, queue_timeout=300, kv_slots=1, engine_path=None):
    parsed = urlparse(master_url if "://" in master_url else f"http://{master_url}")
    master_host = parsed.hostname or "127.0.0.1"
    control_port = control_port or int(os.environ.get("AVIARY_CONTROL_PORT", "9002"))
    node_id = load_or_create_node_id(model)
    model_id = model_id or os.environ.get("COLI_MODEL_ID") or default_model_id(model)
    advertise = advertise_host or (host if host not in ("0.0.0.0", "::") else None)
    family = model_family(model)
    engine_bin = engine_path or str(default_engine())

    server = APIServer((host, port), None, model_id, api_key, max_tokens,
                       cors_origins=DEFAULT_CORS_ORIGINS,
                       max_queue=max_queue, queue_timeout=queue_timeout,
                       kv_slots=kv_slots, family=family)
    runtime = Engine(engine_bin, model, cap, max_tokens, env, kv_slots, family)
    server.engine = runtime

    control = ControlConnection(node_id, master_host, control_port, port, model_id, model,
                                advertise, runtime, server.scheduler)
    control.start()

    print(f"Aviary agent {node_id} listening on http://{host}:{port}/v1 "
          f"(master control {master_host}:{control_port})", file=sys.stderr)

    previous = signal.getsignal(signal.SIGTERM)

    def _shutdown(*_):
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, _shutdown)
    try:
        server.serve_forever()
    finally:
        signal.signal(signal.SIGTERM, previous)
        control.stop()
        server.scheduler.close()
        server.server_close()
        runtime.close()
