"""RPC latency and expert server smoke tests."""

import socket
import struct
import threading
import time
import unittest

from aviary.expert_rpc import ExpertRPCServer, DEFAULT_RPC_TIMEOUT_MS


class _FakeEngine:
    def exec_expert(self, layer, eid, x_in, timeout=0.15):
        return [v * 2.0 for v in x_in]


class ExpertRPCLatencyTest(unittest.TestCase):
    def test_loopback_rpc_latency(self):
        engine = _FakeEngine()
        server = ExpertRPCServer(("127.0.0.1", 0), engine.exec_expert)
        port = server.server_address[1]
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            latencies = []
            for _ in range(20):
                conn = socket.create_connection(("127.0.0.1", port), timeout=1)
                conn.settimeout(DEFAULT_RPC_TIMEOUT_MS / 1000.0)
                hidden = 8
                x = tuple(0.1 * i for i in range(hidden))
                raw = struct.pack(f"{hidden}f", *x)
                t0 = time.perf_counter()
                conn.sendall(f"EXEC_EXPERT 1 0 0 {len(raw)}\n".encode() + raw + b"\n")
                buf = b""
                while b"\n" not in buf:
                    buf += conn.recv(4096)
                elapsed_us = (time.perf_counter() - t0) * 1e6
                latencies.append(elapsed_us)
                self.assertIn(b"EXPERT_RESULT", buf)
                conn.close()
            p50 = sorted(latencies)[len(latencies) // 2]
            self.assertLess(p50, 500_000)  # loopback p50 under 500ms (generous for CI)
        finally:
            server.shutdown()


if __name__ == "__main__":
    unittest.main()
