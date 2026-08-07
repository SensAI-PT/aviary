import json
import socket
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.error import HTTPError
from urllib.request import urlopen

from aviary.master import ControlPlaneServer, MasterHTTPServer
from aviary.protocol import heartbeat_frame, read_frame, register_frame
from aviary.registry import NodeRegistry


class MasterControlTest(unittest.TestCase):
    def test_agent_register_flow(self):
        registry = NodeRegistry(heartbeat_miss=3)
        server = ControlPlaneServer(("127.0.0.1", 0), registry)
        port = server.server_address[1]
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            payload = {"host": "127.0.0.1", "hwinfo": {"cores": 8}}
            frame = register_frame("node-a", 8001, "hy3-colibri", payload)
            conn = socket.create_connection(("127.0.0.1", port), timeout=2)
            conn.sendall(frame.encode("utf-8"))
            kind, fields, _ = read_frame(conn, 500)
            self.assertEqual(kind, "REGISTERED")
            self.assertEqual(fields[1], "node-a")
            picked = registry.pick_least_loaded()
            self.assertIsNotNone(picked)
            self.assertEqual(picked.node_id, "node-a")
            conn.close()
        finally:
            server.shutdown()

    def test_control_stays_alive_between_heartbeats(self):
        """Master must not drop the control socket during normal heartbeat silence."""
        registry = NodeRegistry(heartbeat_miss=3)
        server = ControlPlaneServer(("127.0.0.1", 0), registry)
        port = server.server_address[1]
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            conn = socket.create_connection(("127.0.0.1", port), timeout=2)
            conn.sendall(register_frame(
                "node-b", 8001, "hy3-colibri", {"host": "127.0.0.1"}).encode("utf-8"))
            kind, _, _ = read_frame(conn, 500)
            self.assertEqual(kind, "REGISTERED")
            time.sleep(0.4)  # longer than old 150ms RPC timeout, shorter than lease
            conn.sendall(heartbeat_frame("node-b", 0, {"hits_seq": 1}).encode("utf-8"))
            kind, fields, _ = read_frame(conn, 500)
            self.assertEqual(kind, "HEARTBEAT_ACK")
            self.assertEqual(fields[1], "node-b")
            self.assertEqual(registry.pick_least_loaded().node_id, "node-b")
            conn.close()
        finally:
            server.shutdown()


class _FakeAgentHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        body = b'{"object":"list","data":[{"id":"hy3-colibri","object":"model"}]}'
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        pass


class MasterHttpRoutingTest(unittest.TestCase):
    def test_v1_models_is_not_spa_fallback(self):
        registry = NodeRegistry()
        agent = HTTPServer(("127.0.0.1", 0), _FakeAgentHandler)
        agent_port = agent.server_address[1]
        agent_thread = threading.Thread(target=agent.serve_forever, daemon=True)
        agent_thread.start()
        registry.register("node-a", "127.0.0.1", agent_port, "hy3-colibri",
                          {"host": "127.0.0.1"})
        http = MasterHTTPServer(("127.0.0.1", 0), registry)
        http_port = http.server_address[1]
        http_thread = threading.Thread(target=http.serve_forever, daemon=True)
        http_thread.start()
        try:
            with urlopen(f"http://127.0.0.1:{http_port}/v1/models", timeout=2) as resp:
                body = json.loads(resp.read().decode("utf-8"))
            self.assertEqual(body["data"][0]["id"], "hy3-colibri")
            with self.assertRaises(HTTPError) as ctx:
                urlopen(f"http://127.0.0.1:{http_port}/v1/missing", timeout=2)
            self.assertEqual(ctx.exception.code, 404)
            err = json.loads(ctx.exception.read().decode("utf-8"))
            self.assertIn("error", err)
        finally:
            http.shutdown()
            agent.shutdown()


if __name__ == "__main__":
    unittest.main()
