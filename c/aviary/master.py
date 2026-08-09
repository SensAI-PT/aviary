"""Aviary cluster master — registry, routing, placement scheduler, cluster HTTP API."""

from __future__ import annotations

import http.client
import json
import os
import signal
import socket
import socketserver
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

from aviary.jobs import JobTracker
from aviary.placement import PlacementScheduler
from aviary.protocol import ProtocolError, placement_frame, read_frame, write_line
from aviary.registry import CONTROL_IDLE_TIMEOUT_MS, DEFAULT_HEARTBEAT_SEC, NodeRegistry

try:
    from openai_server import APIHandler, DEFAULT_CORS_ORIGINS, model_object
except ImportError:
    from openai_server import APIHandler, DEFAULT_CORS_ORIGINS, model_object  # type: ignore

WEB_DIST = Path(__file__).resolve().parent.parent.parent / "web" / "dist"
DEFAULT_PLACEMENT_SEC = float(os.environ.get("AVIARY_PLACEMENT_SEC", "4"))


class ControlPlaneServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, address, registry: NodeRegistry):
        super().__init__(address, ControlPlaneHandler)
        self.registry = registry


class ControlPlaneHandler(socketserver.BaseRequestHandler):
    def handle(self):
        registry: NodeRegistry = self.server.registry  # type: ignore[attr-defined]
        conn = self.request
        peer_host = self.client_address[0]
        timeout_ms = CONTROL_IDLE_TIMEOUT_MS
        node_id = None
        try:
            while True:
                kind, fields, payload = read_frame(conn, timeout_ms)
                if kind == "EOF":
                    break
                if kind == "REGISTER" and len(fields) >= 5 and payload is not None:
                    node_id = fields[1]
                    http_port = int(fields[2])
                    model_id = fields[3]
                    payload.setdefault("host", peer_host)
                    try:
                        registry.register(node_id, peer_host, http_port, model_id, payload,
                                          control_conn=conn)
                        write_line(conn, f"REGISTERED {node_id}")
                    except ValueError as err:
                        code = str(err) if str(err) not in ("DUPLICATE",) else "DUPLICATE"
                        write_line(conn, f"ERROR {node_id} {code}")
                elif kind == "HEARTBEAT" and len(fields) >= 3 and payload is not None:
                    node_id = fields[1]
                    inflight = int(fields[2])
                    registry.set_control_conn(node_id, conn)
                    if registry.heartbeat(node_id, inflight, payload):
                        write_line(conn, f"HEARTBEAT_ACK {node_id}")
                elif kind == "DEREGISTER" and len(fields) >= 2:
                    node_id = fields[1]
                    registry.deregister(node_id)
                    write_line(conn, f"DEREGISTERED {node_id}")
                    break
                elif kind == "PONG":
                    pass
        except (ProtocolError, OSError, ValueError) as error:
            print(f"[aviary-master] control error: {error}", file=sys.stderr)
        finally:
            if node_id:
                registry.mark_dead(node_id)
            conn.close()


class MasterHTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, registry: NodeRegistry, scheduler: PlacementScheduler,
                 jobs: JobTracker, api_key=None, cors_origins=DEFAULT_CORS_ORIGINS):
        super().__init__(address, MasterHTTPHandler)
        self.registry = registry
        self.scheduler = scheduler
        self.jobs = jobs
        self.api_key = api_key
        self.cors_origins = tuple(cors_origins)
        self.created = int(time.time())

    @property
    def model_id(self):
        nodes = self.registry.snapshot().get("nodes") or []
        return next((n["model_id"] for n in nodes if n.get("model_id")),
                    os.environ.get("COLI_MODEL_ID", "colibri"))


class MasterHTTPHandler(APIHandler):
    server_version = "aviary-master"

    def _pick_node(self):
        registry = self.server.registry  # type: ignore[attr-defined]
        scheduler = self.server.scheduler  # type: ignore[attr-defined]
        plan = scheduler.last_plan
        hot: set[tuple[int, int]] = set()
        if plan and plan.experts:
            hot = {tuple(map(int, k.split(":"))) for k in plan.experts}
        node = registry.pick_with_affinity(hot or None)
        if node is None:
            raise RuntimeError("no healthy agents registered")
        return node

    def _proxy_raw(self, node, method, path, body=None, headers=None):
        headers = dict(headers or {})
        headers.pop("Host", None)
        if self.server.api_key:  # type: ignore[attr-defined]
            headers["Authorization"] = f"Bearer {self.server.api_key}"
        parsed = urlsplit(node.endpoint)
        conn = http.client.HTTPConnection(parsed.hostname, parsed.port or 80, timeout=600)
        try:
            conn.request(method, path, body=body, headers=headers)
            response = conn.getresponse()
            return response.status, dict(response.getheaders()), response
        except OSError as error:
            conn.close()
            raise RuntimeError(str(error)) from error

    def do_GET(self):
        path = urlsplit(self.path).path
        if path == "/cluster/health":
            snap = self.server.registry.snapshot()  # type: ignore[attr-defined]
            self.send_json(200, {"status": "ok", "nodes": snap["total"], "healthy": snap["healthy"]})
            return
        if path == "/cluster/nodes":
            self.send_json(200, self.server.registry.snapshot())  # type: ignore[attr-defined]
            return
        if path == "/cluster/placement":
            scheduler = self.server.scheduler  # type: ignore[attr-defined]
            self.send_json(200, scheduler.snapshot())
            return
        if path == "/cluster/costs":
            scheduler = self.server.scheduler  # type: ignore[attr-defined]
            snap = scheduler.snapshot()
            self.send_json(200, {"costs": snap, "nodes": self.server.registry.snapshot()})  # type: ignore[attr-defined]
            return
        if path == "/cluster/jobs":
            jobs = self.server.jobs.snapshot()  # type: ignore[attr-defined]
            self.send_json(200, jobs)
            return
        if path == "/cluster/overview":
            registry = self.server.registry  # type: ignore[attr-defined]
            scheduler = self.server.scheduler  # type: ignore[attr-defined]
            jobs = self.server.jobs  # type: ignore[attr-defined]
            self.send_json(200, {
                "nodes": registry.snapshot(),
                "placement": scheduler.snapshot(),
                "jobs": jobs.snapshot(),
                "updated_at": time.time(),
            })
            return
        if path == "/health":
            snap = self.server.registry.snapshot()  # type: ignore[attr-defined]
            self.send_json(200, {"status": "ok", "scheduler": snap, "cluster": True,
                                 "placement": self.server.scheduler.snapshot()})  # type: ignore[attr-defined]
            return
        if self.serve_static(path):
            return
        if path == "/v1/models":
            try:
                node = self._pick_node()
                status, resp_headers, resp = self._proxy_raw(node, "GET", "/v1/models")
                data = resp.read()
                resp.close()
                self.send_response(status)
                for key, value in resp_headers.items():
                    if key.lower() not in ("transfer-encoding", "connection"):
                        self.send_header(key, value)
                self.end_headers()
                self.wfile.write(data)
            except RuntimeError as error:
                self.send_json(503, {"error": {"message": str(error)}})
            return
        self.send_json(404, {"error": {"message": "Not found."}})

    def do_POST(self):
        path = urlsplit(self.path).path
        if path not in ("/v1/chat/completions", "/v1/completions", "/v1/messages"):
            self.send_json(404, {"error": {"message": "Not found."}})
            return
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length) if length else b""
        node = None
        job = None
        try:
            node = self._pick_node()
            job = self.server.jobs.start(node.node_id, node.endpoint, path)  # type: ignore[attr-defined]
            self.server.registry.increment_inflight(node.node_id, 1)  # type: ignore[attr-defined]
            status, resp_headers, resp = self._proxy_raw(
                node, "POST", path, body=body,
                headers={"Content-Type": self.headers.get("Content-Type", "application/json"),
                         "Accept": self.headers.get("Accept", "*/*")})
            self.send_response(status)
            if job:
                self.send_header("X-Aviary-Job-Id", job.job_id)
                self.send_header("X-Aviary-Node-Id", node.node_id)
            has_length = False
            for key, value in resp_headers.items():
                lower = key.lower()
                if lower in ("transfer-encoding", "connection"):
                    continue
                if lower == "content-length":
                    has_length = True
                self.send_header(key, value)
            if not has_length:
                self.send_header("Connection", "close")
                self.close_connection = True
            self.end_headers()
            while True:
                chunk = resp.read(65536)
                if not chunk:
                    break
                self.wfile.write(chunk)
                self.wfile.flush()
            resp.close()
            if job:
                self.server.jobs.finish(job.job_id, http_status=status)  # type: ignore[attr-defined]
        except RuntimeError as error:
            if job:
                self.server.jobs.finish(job.job_id, error=str(error))  # type: ignore[attr-defined]
            self.send_json(502, {"error": {"message": str(error)}})
        finally:
            if node:
                self.server.registry.increment_inflight(node.node_id, -1)  # type: ignore[attr-defined]

    def serve_static(self, path):
        if path.startswith("/v1/") or path.startswith("/cluster/") or path in (
                "/health", "/experts", "/profile"):
            return False
        if not WEB_DIST.is_dir():
            return False
        if path in ("", "/"):
            path = "/index.html"
        target = (WEB_DIST / path.lstrip("/")).resolve()
        if not str(target).startswith(str(WEB_DIST.resolve())):
            return False
        if target.is_dir():
            target = target / "index.html"
        if not target.is_file():
            if path.startswith("/assets/") or "." in Path(path).name:
                return False
            target = WEB_DIST / "index.html"
            if not target.is_file():
                return False
        data = target.read_bytes()
        ctype = "text/html"
        if target.suffix == ".js":
            ctype = "application/javascript"
        elif target.suffix == ".css":
            ctype = "text/css"
        elif target.suffix == ".json":
            ctype = "application/json"
        elif target.suffix == ".svg":
            ctype = "image/svg+xml"
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_cors_headers()
        self.end_headers()
        self.wfile.write(data)
        return True


def _push_placement(registry: NodeRegistry, scheduler: PlacementScheduler) -> None:
    snap = registry.snapshot()
    nodes = snap.get("nodes") or []
    if len(nodes) < 1:
        return
    scheduler.recompute(nodes)
    conns = registry.control_connections()
    for node in nodes:
        nid = node["node_id"]
        conn = conns.get(nid)
        if not conn:
            continue
        try:
            payload = scheduler.build_agent_payload(nid, nodes, int(node.get("expert_port", 9003)))
            conn.sendall(placement_frame(payload).encode("utf-8"))
        except OSError as error:
            print(f"[aviary-master] placement push to {nid} failed: {error}", file=sys.stderr)


def _placement_ticker(registry: NodeRegistry, scheduler: PlacementScheduler, stop: threading.Event):
    while not stop.wait(DEFAULT_PLACEMENT_SEC):
        try:
            _push_placement(registry, scheduler)
        except Exception as error:
            print(f"[aviary-master] placement error: {error}", file=sys.stderr)


def _lease_ticker(registry: NodeRegistry, stop: threading.Event):
    while not stop.wait(DEFAULT_HEARTBEAT_SEC):
        evicted = registry.tick_leases()
        for node_id in evicted:
            print(f"[aviary-master] evicted node {node_id}", file=sys.stderr)


def run_master(host="0.0.0.0", port=9000, control_port=None, api_key=None, cors_origins=None):
    control_port = control_port or int(os.environ.get("AVIARY_CONTROL_PORT", "9002"))
    if host not in ("127.0.0.1", "localhost", "::1") and not api_key:
        if os.environ.get("COLI_ALLOW_INSECURE_BIND") != "1":
            print("refusing to bind beyond localhost without COLI_API_KEY "
                  "(set COLI_ALLOW_INSECURE_BIND=1 to override)", file=sys.stderr)
            sys.exit(1)
    registry = NodeRegistry()
    scheduler = PlacementScheduler()
    jobs = JobTracker()
    stop = threading.Event()
    ticker = threading.Thread(target=_lease_ticker, args=(registry, stop), daemon=True)
    ticker.start()
    placement_thread = threading.Thread(
        target=_placement_ticker, args=(registry, scheduler, stop), daemon=True)
    placement_thread.start()

    control = ControlPlaneServer((host if host != "0.0.0.0" else "", control_port), registry)
    control_thread = threading.Thread(
        target=control.serve_forever, name="aviary-control-plane", daemon=True)
    control_thread.start()

    origins = DEFAULT_CORS_ORIGINS if cors_origins is None else tuple(cors_origins)
    http = MasterHTTPServer((host, port), registry, scheduler, jobs, api_key, origins)
    print(f"Aviary master HTTP http://{host}:{port}  control :{control_port}", file=sys.stderr)

    previous = signal.getsignal(signal.SIGTERM)

    def _shutdown(*_):
        threading.Thread(target=http.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, _shutdown)
    try:
        http.serve_forever()
    finally:
        signal.signal(signal.SIGTERM, previous)
        stop.set()
        control.shutdown()
        http.server_close()
