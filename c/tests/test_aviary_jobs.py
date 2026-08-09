"""Tests for cluster job tracking."""

import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.request import Request, urlopen

from aviary.jobs import JobTracker
from aviary.master import MasterHTTPServer
from aviary.placement import PlacementScheduler
from aviary.registry import NodeRegistry


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


class ClusterJobTraceTest(unittest.TestCase):
    def test_append_trace_to_active_job(self):
        jobs = JobTracker()
        job = jobs.start("node-a", "http://127.0.0.1:8001", "/v1/chat/completions")
        jobs.append_trace(job.job_id, [{
            "ts": 1.0, "job_id": job.job_id, "kind": "rpc_in", "node_id": "node-a",
            "layer": 3, "expert": 7, "local": True, "rpc_us": 1200.0,
        }])
        jobs.finish(job.job_id, http_status=200)
        snap = jobs.snapshot()
        self.assertEqual(len(snap["history"][0]["trace"]), 1)
        self.assertEqual(snap["history"][0]["trace"][0]["layer"], 3)


if __name__ == "__main__":
    unittest.main()
