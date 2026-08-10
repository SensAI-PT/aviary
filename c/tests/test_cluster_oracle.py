"""Worker-vs-local token-exact correctness gate for Aviary cluster RPC.

When COLI_MODEL points at a Qwen3 MoE checkpoint and qwen3_moe is built, compares
completion text with AVIARY_CLUSTER=0 (local-only oracle) vs AVIARY_CLUSTER=1.
Skips in CI without a model fixture.
"""

from __future__ import annotations

import os
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
BINARY = HERE / "qwen3_moe"
MODEL = os.environ.get("COLI_MODEL", "")


def _model_available() -> bool:
    return bool(MODEL) and Path(MODEL).is_dir() and BINARY.is_file()


def _completion(cluster: bool) -> str:
    from openai_server import Engine

    env = dict(os.environ, SNAP=MODEL, SERVE="1", SERVE_BATCH="1", NGEN="16", KV_SLOTS="1")
    env["AVIARY_CLUSTER"] = "1" if cluster else "0"
    if cluster:
        placement = Path(MODEL) / ".aviary_placement.json"
        env["AVIARY_PLACEMENT"] = str(placement)
    eng = Engine(str(BINARY), MODEL, max_tokens=16, env=env, kv_slots=1, arch="qwen3_moe")
    parts: list[str] = []
    try:
        eng.generate("Say hi in one word.", 8, 0.0, 1.0, parts.append)
    finally:
        eng.close()
    return "".join(parts).strip()


@unittest.skipUnless(_model_available(), "COLI_MODEL fixture and qwen3_moe binary required")
class ClusterOracleIntegrationTest(unittest.TestCase):
    def test_cluster_matches_local_tokens(self):
        """Remote expert path must produce the same text as local-only moe()."""
        local = _completion(cluster=False)
        cluster = _completion(cluster=True)
        self.assertTrue(local, "local oracle produced empty output")
        self.assertEqual(cluster, local,
                         msg=f"cluster output diverged from local oracle\nlocal: {local!r}\ncluster: {cluster!r}")


class ClusterOracleContractTest(unittest.TestCase):
    def test_trace_kinds_documented(self):
        kinds = {"local", "remote", "fallback", "rpc_in"}
        from aviary.trace import TraceBuffer

        buf = TraceBuffer()
        buf.set_job("job-1")
        for kind in kinds:
            buf.record(kind, "node-a", 0, 1, rpc_us=100.0, local=kind in ("local", "fallback", "rpc_in"))
        events = buf.drain()
        self.assertEqual(len(events), len(kinds))
        self.assertEqual({e["kind"] for e in events}, kinds)


if __name__ == "__main__":
    unittest.main()
