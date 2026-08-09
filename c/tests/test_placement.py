"""Tests for Aviary placement scheduler."""

import unittest

from aviary.placement import PlacementScheduler, blocks_from_experts, decode_emap


class DecodeEmapTest(unittest.TestCase):
    def test_two_nibble_hex(self):
        emap = {"rows": 1, "cols": 2, "map": "4080"}
        inv = decode_emap(emap)
        self.assertEqual(inv[(0, 0)]["tier"], 1)
        self.assertEqual(inv[(0, 0)]["heat"], 0)
        self.assertEqual(inv[(0, 1)]["tier"], 2)
        self.assertEqual(inv[(0, 1)]["heat"], 0)


class PlacementSchedulerTest(unittest.TestCase):
    def test_recompute_picks_remote_when_cheaper(self):
        sched = PlacementScheduler(recompute_sec=1)
        nodes = [
            {
                "node_id": "a", "status": "healthy", "inflight": 0,
                "emap": {"rows": 1, "cols": 1, "map": "00"},
                "costs": [{"layer": 0, "expert": 0, "tier": 0, "load_us": 500000, "exec_us": 100000}],
                "usage": [{"layer": 0, "expert": 0, "count": 100}],
            },
            {
                "node_id": "b", "status": "healthy", "inflight": 0,
                "emap": {"rows": 1, "cols": 1, "map": "40"},
                "costs": [{"layer": 0, "expert": 0, "tier": 1, "load_us": 0, "exec_us": 10000}],
                "usage": [{"layer": 0, "expert": 0, "count": 50}],
            },
        ]
        sched.record_rpc("a", "b", 5000)
        plan = sched.recompute(nodes, primary_hint="a")
        self.assertEqual(plan.experts.get("0:0"), "b")

    def test_build_agent_payload(self):
        sched = PlacementScheduler()
        nodes = [
            {"node_id": "a", "status": "healthy", "host": "10.0.0.1", "inflight": 0,
             "usage": [{"layer": 0, "expert": 1, "count": 10}]},
            {"node_id": "b", "status": "healthy", "host": "10.0.0.2", "inflight": 0,
             "usage": [{"layer": 0, "expert": 1, "count": 5}]},
        ]
        sched.recompute(nodes)
        payload = sched.build_agent_payload("a", nodes, 9003)
        self.assertEqual(payload["node_id"], "a")
        self.assertIn("b", payload["peers"])

    def test_blocks_from_experts(self):
        blocks = blocks_from_experts({"0:1": "a", "1:2": "a", "3:4": "b"}, ["a", "b"])
        self.assertEqual(blocks["a"], [{"start": 0, "end": 1}])
        self.assertEqual(blocks["b"], [{"start": 3, "end": 3}])


if __name__ == "__main__":
    unittest.main()
