import json
import socket
import threading
import time
import unittest

from aviary.identity import load_or_create_node_id, node_id_path
from aviary.protocol import (
    evict_frame,
    heartbeat_frame,
    load_frame,
    pin_frame,
    read_frame,
    register_frame,
    send_frame,
    write_line,
)
from aviary.registry import NodeRegistry


class ProtocolTest(unittest.TestCase):
    def test_register_round_trip(self):
        payload = {"host": "127.0.0.1", "hwinfo": {"cores": 4}}
        frame = register_frame("abc", 8001, "hy3", payload)
        self.assertIn("REGISTER abc 8001 hy3", frame)
        self.assertIn('"host"', frame)

    def test_pin_frames(self):
        self.assertEqual(pin_frame(3, 7, 1), "PIN 3 7 1\n")
        self.assertEqual(load_frame(3, 7, 0), "LOAD 3 7 0\n")
        self.assertEqual(evict_frame(3, 7), "EVICT 3 7\n")

    def test_read_write_frames(self):
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.bind(("127.0.0.1", 0))
        server.listen(1)
        port = server.getsockname()[1]
        payload = {"emap": {"rows": 1, "cols": 2, "map": "ff"}}

        def accept():
            conn, _ = server.accept()
            kind, fields, body = read_frame(conn, 500)
            self.assertEqual(kind, "HEARTBEAT")
            self.assertEqual(fields[1], "node-1")
            self.assertEqual(body["emap"]["rows"], 1)
            write_line(conn, "HEARTBEAT_ACK node-1")
            conn.close()

        thread = threading.Thread(target=accept)
        thread.start()
        client = socket.create_connection(("127.0.0.1", port))
        client.sendall(heartbeat_frame("node-1", 2, payload).encode("utf-8"))
        kind, fields, _ = read_frame(client, 500)
        self.assertEqual(kind, "HEARTBEAT_ACK")
        client.close()
        thread.join(timeout=2)
        server.close()


class RegistryTest(unittest.TestCase):
    def test_register_and_pick_least_loaded(self):
        reg = NodeRegistry(heartbeat_miss=3)
        reg.register("a", "127.0.0.1", 8001, "m", {"host": "127.0.0.1"})
        reg.register("b", "127.0.0.1", 8002, "m", {"host": "127.0.0.1"})
        reg.heartbeat("b", 5, {"hits_seq": 1})
        picked = reg.pick_least_loaded()
        self.assertIsNotNone(picked)
        self.assertEqual(picked.node_id, "a")

    def test_lease_eviction(self):
        reg = NodeRegistry(heartbeat_miss=1)
        reg.register("a", "127.0.0.1", 8001, "m", {"host": "127.0.0.1"})
        record = reg._nodes["a"]
        record.last_heartbeat = time.time() - 10
        evicted = reg.tick_leases()
        self.assertEqual(evicted, ["a"])
        self.assertEqual(reg.pick_least_loaded(), None)


class IdentityTest(unittest.TestCase):
    def test_persists_node_id(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            first = load_or_create_node_id(tmp)
            second = load_or_create_node_id(tmp)
            self.assertEqual(first, second)
            self.assertTrue(node_id_path(tmp).is_file())


if __name__ == "__main__":
    unittest.main()
