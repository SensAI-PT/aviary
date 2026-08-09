"""Tests for cluster job tracking."""

import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer

from aviary.jobs import JobTracker
from aviary.master import MasterHTTPServer
from aviary.placement import PlacementScheduler
from aviary.registry import NodeRegistry
from urllib.request import Request, urlopen


class _SlowAgentHandler(BaseHTTPRequestHandler):
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


class ClusterJobsTest(unittest.TestCase):
    def test_jobs_endpoint_tracks_proxy(self):
        registry = NodeRegistry()
        agent = HTTPServer(("127.0.0.1", 0), _SlowAgentHandler)
        port = agent.server_address[1]
        threading.Thread(target=agent.serve_forever, daemon=True).start()
        registry.register("node-a", "127.0.0.1", port, "hy3-colibri", {"host": "127.0.0.1"})
        http = MasterHTTPServer(("127.0.0.1", 0), registry, PlacementScheduler(), JobTracker())
        http_port = http.server_address[1]
        threading.Thread(target=http.serve_forever, daemon=True).start()
        try:
            req = urlopen(Request(
                f"http://127.0.0.1:{http_port}/v1/chat/completions",
                data=json.dumps({"model": "hy3-colibri", "messages": [{"role": "user", "content": "hi"}]}).encode(),
                headers={"Content-Type": "application/json"},
                method="POST",
            ), timeout=5)
            req.read()
            with urlopen(f"http://127.0.0.1:{http_port}/cluster/jobs", timeout=2) as resp:
                body = json.loads(resp.read().decode())
            self.assertGreaterEqual(body["completed_count"], 1)
            self.assertTrue(body["history"])
            self.assertEqual(body["history"][0]["status"], "completed")
        finally:
            http.shutdown()
            agent.shutdown()


if __name__ == "__main__":
    unittest.main()
