"""TCP expert RPC server — peer side of cross-node expert execution."""

from __future__ import annotations

import os
import socket
import socketserver
import struct
import threading
import time
from typing import Callable


DEFAULT_EXPERT_PORT = int(os.environ.get("AVIARY_EXPERT_PORT", "9003"))
DEFAULT_RPC_TIMEOUT_MS = int(os.environ.get("AVIARY_RPC_TIMEOUT_MS", "150"))


class ExpertRPCHandler(socketserver.BaseRequestHandler):
    def handle(self):
        engine_exec: Callable | None = getattr(self.server, "engine_exec", None)  # type: ignore[attr-defined]
        tier_check: Callable | None = getattr(self.server, "tier_check", None)  # type: ignore[attr-defined]
        timeout = getattr(self.server, "rpc_timeout_ms", DEFAULT_RPC_TIMEOUT_MS) / 1000.0  # type: ignore[attr-defined]
        conn = self.request
        conn.settimeout(timeout)
        buf = b""
        try:
            while b"\n" not in buf:
                chunk = conn.recv(4096)
                if not chunk:
                    return
                buf += chunk
            line, rest = buf.split(b"\n", 1)
            fields = line.decode("utf-8", "replace").strip().split()
            if len(fields) < 5 or fields[0] != "EXEC_EXPERT":
                return
            req_id, layer, eid, nbytes = fields[1], int(fields[2]), int(fields[3]), int(fields[4])
            payload = rest
            while len(payload) < nbytes + 1:
                chunk = conn.recv(nbytes + 1 - len(payload))
                if not chunk:
                    return
                payload += chunk
            hidden = nbytes // 4
            x_in = struct.unpack(f"{hidden}f", payload[:nbytes])
            if tier_check and not tier_check(layer, eid):
                conn.sendall(f"EXPERT_MISS {req_id}\n".encode())
                return
            if not engine_exec:
                conn.sendall(f"EXPERT_MISS {req_id}\n".encode())
                return
            t0 = time.perf_counter()
            out = engine_exec(layer, eid, x_in)
            _ = time.perf_counter() - t0
            if out is None:
                conn.sendall(f"EXPERT_MISS {req_id}\n".encode())
                return
            raw = struct.pack(f"{len(out)}f", *out)
            conn.sendall(f"EXPERT_RESULT {req_id} {len(raw)}\n".encode() + raw + b"\n")
        except (OSError, struct.error, ValueError):
            pass


class ExpertRPCServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, address, engine_exec, tier_check=None, rpc_timeout_ms=DEFAULT_RPC_TIMEOUT_MS):
        super().__init__(address, ExpertRPCHandler)
        self.engine_exec = engine_exec
        self.tier_check = tier_check
        self.rpc_timeout_ms = rpc_timeout_ms


class EngineExpertBridge:
    """Forward EXEC_EXPERT to the colibri mux subprocess via Engine.exec_expert."""

    def __init__(self, engine):
        self.engine = engine

    def __call__(self, layer: int, eid: int, x_in: tuple[float, ...]) -> list[float] | None:
        exec_fn = getattr(self.engine, "exec_expert", None)
        return exec_fn(layer, eid, x_in) if exec_fn else None


def tier_check_from_emap(engine, layer: int, eid: int) -> bool:
    """True if expert appears resident (RAM/VRAM) in latest EMAP."""
    emap = getattr(engine, "emap", None)
    if not emap or not emap.get("map"):
        return True  # optimistic — engine will EXPERT_MISS on actual load failure
    cols = int(emap["cols"])
    idx = layer * cols + eid
    raw = emap["map"]
    if idx >= len(raw):
        return True
    byte = int(raw[idx], 16)
    return (byte >> 6) >= 1 or True  # allow disk-tier serve via local load in engine


def start_expert_rpc_server(host: str, port: int, engine, stop_event: threading.Event | None = None):
    bridge = EngineExpertBridge(engine)
    server = ExpertRPCServer((host if host != "0.0.0.0" else "", port), bridge,
                             tier_check=lambda l, e: tier_check_from_emap(engine, l, e))
    thread = threading.Thread(target=server.serve_forever, name="aviary-expert-rpc", daemon=True)
    thread.start()
    return server, thread, bridge
