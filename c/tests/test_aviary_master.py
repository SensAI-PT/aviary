import socket
import threading
import time
import unittest

from aviary.master import ControlPlaneServer
from aviary.protocol import read_frame, register_frame, write_line
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


if __name__ == "__main__":
    unittest.main()
