"""Tests for Aviary placement scheduler."""

import unittest
from unittest import mock

import aviary.placement as p
from aviary.placement import PlacementScheduler, blocks_from_experts, decode_emap, layer_coherence_stats


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
        self.assertTrue(plan.layer_blocks)

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
        self.assertIn("layer_caps", payload)

    def test_blocks_from_experts(self):
        blocks = blocks_from_experts({"0:1": "a", "1:2": "a", "3:4": "b"}, ["a", "b"])
        self.assertEqual(blocks["a"], [{"start": 0, "end": 1}])
        self.assertEqual(blocks["b"], [{"start": 3, "end": 3}])

    def test_layer_coherence_stats(self):
        stats = layer_coherence_stats({"0:1": "a", "0:2": "a", "1:1": "b", "1:2": "c"})
        self.assertEqual(stats["multi_node_layers"], 1)
        self.assertLess(stats["score"], 1.0)

    def test_layer_coherent_mode(self):
        nodes = [
            {"node_id": "a", "status": "healthy", "inflight": 0,
             "emap": {"rows": 2, "cols": 2, "map": "0000"},
             "usage": [{"layer": 0, "expert": 0, "count": 10}, {"layer": 0, "expert": 1, "count": 10}]},
            {"node_id": "b", "status": "healthy", "inflight": 0,
             "emap": {"rows": 2, "cols": 2, "map": "0000"}, "usage": []},
        ]
        with mock.patch.object(p, "LAYER_BLOCKS", False), mock.patch.object(p, "LAYER_COHERENT", True):
            sched = p.PlacementScheduler(recompute_sec=1)
            plan = sched.recompute(nodes, primary_hint="a")
            self.assertEqual(plan.experts.get("0:0"), plan.experts.get("0:1"))
            self.assertTrue(plan.layer_coherent)

    def test_layer_blocks_contiguous(self):
        nodes = [
            {"node_id": "a", "status": "healthy", "inflight": 0,
             "emap": {"rows": 2, "cols": 1, "map": "00"},
             "usage": [{"layer": 0, "expert": 0, "count": 50}, {"layer": 1, "expert": 0, "count": 40}]},
            {"node_id": "b", "status": "healthy", "inflight": 0,
             "emap": {"rows": 2, "cols": 1, "map": "00"}, "usage": []},
        ]
        sched = PlacementScheduler()
        plan = sched.recompute(nodes, primary_hint="a")
        self.assertTrue(plan.planned_blocks)
        self.assertEqual(plan.experts.get("0:0"), plan.experts.get("1:0"))
        self.assertEqual(plan.planned_blocks[0]["start"], 0)
        self.assertEqual(plan.planned_blocks[0]["end"], 1)

    def test_vram_slow_caps_tier(self):
        costs_t1 = [{"layer": i, "expert": 0, "tier": 1, "load_us": 0, "exec_us": 20_000} for i in range(10)]
        costs_t2 = [{"layer": i, "expert": 0, "tier": 2, "load_us": 0, "exec_us": 50_000} for i in range(10)]
        nodes = [{
            "node_id": "cuda", "status": "healthy", "inflight": 0,
            "tiers": {"vram": 8, "vram_gb": 12, "ram": 32},
            "emap": {"rows": 2, "cols": 2, "map": "8000" * 2},
            "costs": costs_t1 + costs_t2,
            "usage": [{"layer": 0, "expert": 0, "count": 100}],
        }]
        sched = PlacementScheduler()
        plan = sched.recompute(nodes, primary_hint="cuda")
        pref = plan.node_tier_prefs.get("cuda", {})
        self.assertTrue(pref.get("vram_slow"))
        self.assertEqual(pref.get("max_tier"), 1)

    def test_vram_slow_without_local_ram_samples(self):
        """Full-GPU node: only tier-2 ECOST — still demote when slower than RAM prior."""
        costs_t2 = [{"layer": i, "expert": 0, "tier": 2, "load_us": 0, "exec_us": 80_000} for i in range(12)]
        # EMAP all VRAM (tier bits = 2 => 0x80)
        nodes = [{
            "node_id": "cuda", "status": "healthy", "inflight": 0,
            "tiers": {"vram": 64, "vram_gb": 10, "ram": 0, "disk": 0},
            "emap": {"rows": 2, "cols": 4, "map": "80" * 8},
            "costs": costs_t2,
            "usage": [{"layer": 0, "expert": 0, "count": 100}],
        }]
        sched = PlacementScheduler()
        plan = sched.recompute(nodes, primary_hint="cuda")
        pref = plan.node_tier_prefs.get("cuda", {})
        self.assertTrue(pref.get("vram_slow"), pref)
        self.assertEqual(pref.get("max_tier"), 1)
        pins = plan.pin_commands.get("cuda") or []
        self.assertTrue(any(t == 1 for _, _, t in pins))

    def test_vram_probe_demote_when_unmeasured(self):
        """Heavy VRAM residency with fast GPU vs prior — probe demotes a few to learn RAM cost."""
        costs_t2 = [{"layer": i, "expert": 0, "tier": 2, "load_us": 0, "exec_us": 5_000} for i in range(12)]
        nodes = [{
            "node_id": "cuda", "status": "healthy", "inflight": 0,
            "tiers": {"vram": 64, "vram_gb": 10, "ram": 0},
            "emap": {"rows": 1, "cols": 4, "map": "80" * 4},
            "costs": costs_t2,
            "usage": [{"layer": 0, "expert": e, "count": 50} for e in range(4)],
        }]
        with mock.patch.object(p, "TIER_PROBE", 2):
            sched = PlacementScheduler()
            plan = sched.recompute(nodes, primary_hint="cuda")
            pref = plan.node_tier_prefs.get("cuda", {})
            self.assertFalse(pref.get("vram_slow"))
            pins = plan.pin_commands.get("cuda") or []
            self.assertGreaterEqual(sum(1 for _, _, t in pins if t == 1), 1)

    def test_hysteresis_keeps_block(self):
        nodes = [
            {"node_id": "a", "status": "healthy", "inflight": 0,
             "emap": {"rows": 2, "cols": 1, "map": "0000"},
             "usage": [{"layer": 0, "expert": 0, "count": 100}]},
            {"node_id": "b", "status": "healthy", "inflight": 0,
             "emap": {"rows": 2, "cols": 1, "map": "4000"},
             "costs": [{"layer": 0, "expert": 0, "tier": 1, "load_us": 0, "exec_us": 9_000}],
             "usage": []},
        ]
        sched = PlacementScheduler()
        sched.record_rpc("a", "b", 1_000)
        plan1 = sched.recompute(nodes, primary_hint="a")
        owner1 = plan1.experts.get("0:0")
        plan2 = sched.recompute(nodes, primary_hint="a")
        self.assertEqual(plan2.experts.get("0:0"), owner1)


if __name__ == "__main__":
    unittest.main()
