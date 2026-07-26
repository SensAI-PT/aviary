"""Phase 1 integration checks — master registry + HTTP without a loaded model."""

import json
import socket
import threading
import time
import unittest
from urllib.request import urlopen

from aviary.master import ControlPlaneServer, MasterHTTPServer
from aviary.protocol import register_frame
from aviary.registry import NodeRegistry


class Phase1IntegrationTest(unittest.TestCase):
    def test_two_agents_register_and_cluster_nodes(self):
        registry = NodeRegistry(heartbeat_miss=2)
        control = ControlPlaneServer(("127.0.0.1", 0), registry)
        control_port = control.server_address[1]
        http = MasterHTTPServer(("127.0.0.1", 0), registry)
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


if __name__ == "__main__":
    unittest.main()
