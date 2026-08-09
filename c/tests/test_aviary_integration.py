"""Phase 1 integration checks — master registry + HTTP without a loaded model."""

import json
import socket
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.error import HTTPError
from urllib.request import urlopen

from aviary.jobs import JobTracker
from aviary.master import ControlPlaneServer, MasterHTTPServer
from aviary.placement import PlacementScheduler
from aviary.protocol import register_frame
from aviary.registry import NodeRegistry


class _QuickAgentHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        if length:
            self.rfile.read(length)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        self.wfile.write(b"data: [DONE]\n\n")

    def log_message(self, fmt, *args):
        pass


class _HangingAgentHandler(BaseHTTPRequestHandler):
    """Streams one chunk then blocks until the connection is reset."""

    hang = threading.Event()

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        if length:
            self.rfile.read(length)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Connection", "close")
        self.close_connection = True
        self.end_headers()
        self.wfile.write(b"data: {\"delta\":\"partial\"}\n\n")
        self.wfile.flush()
        _HangingAgentHandler.hang.wait(timeout=5)

    def log_message(self, fmt, *args):
        pass


class Phase1IntegrationTest(unittest.TestCase):
    def test_two_agents_register_and_cluster_nodes(self):
        registry = NodeRegistry(heartbeat_miss=2)
        control = ControlPlaneServer(("127.0.0.1", 0), registry)
        control_port = control.server_address[1]
        http = MasterHTTPServer(("127.0.0.1", 0), registry, PlacementScheduler(), JobTracker())
        http_port = http.server_address[1]
        threads = [
            threading.Thread(target=control.serve_forever, daemon=True),
            threading.Thread(target=http.serve_forever, daemon=True),
        ]
        for t in threads:
            t.start()

        def register(node_id, port):
            payload = {"host": "127.0.0.1", "hwinfo": {"cores": 4, "ram_total_gb": 32,
                       "ram_avail_gb": 16, "gpus": 0, "vram_total_gb": 0, "cpu": "test", "gpu": ""},
                       "emap": {"rows": 2, "cols": 4, "map": "ff" * 2}}
            conn = socket.create_connection(("127.0.0.1", control_port), timeout=2)
            conn.sendall(register_frame(node_id, port, "hy3-colibri", payload).encode("utf-8"))
            conn.recv(4096)
            return conn

        c1 = register("node-1", 8001)
        c2 = register("node-2", 8002)
        registry.heartbeat("node-2", 3, {"hits_seq": 1})
        picked = registry.pick_least_loaded()
        self.assertEqual(picked.node_id, "node-1")

        with urlopen(f"http://127.0.0.1:{http_port}/cluster/nodes", timeout=2) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        self.assertEqual(body["total"], 2)
        self.assertEqual(body["healthy"], 2)

        c2.close()
        record = registry._nodes["node-2"]
        record.last_heartbeat = time.time() - 20
        registry.tick_leases()
        registry.tick_leases()
        self.assertEqual(registry._nodes["node-2"].status, "dead")
        self.assertEqual(registry.pick_least_loaded().node_id, "node-1")

        c1.close()
        control.shutdown()
        http.shutdown()

    def test_mid_request_agent_death_stops_routing_to_dead_node(self):
        """Kill agent mid-stream: dead node is evicted; routing picks survivors."""
        registry = NodeRegistry(heartbeat_miss=2)
        jobs = JobTracker()
        agent = HTTPServer(("127.0.0.1", 0), _HangingAgentHandler)
        agent_port = agent.server_address[1]
        _HangingAgentHandler.hang.clear()
        threading.Thread(target=agent.serve_forever, daemon=True).start()

        healthy = HTTPServer(("127.0.0.1", 0), _QuickAgentHandler)
        healthy_port = healthy.server_address[1]
        threading.Thread(target=healthy.serve_forever, daemon=True).start()

        registry.register("node-dead", "127.0.0.1", agent_port, "hy3-colibri",
                          {"host": "127.0.0.1"})
        registry.register("node-live", "127.0.0.1", healthy_port, "hy3-colibri",
                          {"host": "127.0.0.1"})
        http_srv = MasterHTTPServer(("127.0.0.1", 0), registry, PlacementScheduler(), jobs)
        http_port = http_srv.server_address[1]
        threading.Thread(target=http_srv.serve_forever, daemon=True).start()

        try:
            agent.shutdown()
            _HangingAgentHandler.hang.set()
            registry.mark_dead("node-dead")
            self.assertEqual(registry._nodes["node-dead"].status, "dead")
            picked = registry.pick_least_loaded()
            self.assertIsNotNone(picked)
            self.assertEqual(picked.node_id, "node-live")
            with urlopen(f"http://127.0.0.1:{http_port}/cluster/jobs", timeout=2) as resp:
                json.loads(resp.read().decode())
        finally:
            http_srv.shutdown()
            healthy.shutdown()


if __name__ == "__main__":
    unittest.main()
