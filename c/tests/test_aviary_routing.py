"""Tests for Aviary coordinator vs donor chat routing."""

import os
import unittest

from aviary.registry import NodeRegistry


def _emap(*tiers: int) -> dict:
    return {"rows": 1, "cols": len(tiers), "map": "".join(f"{t << 6:02x}" for t in tiers)}


class CoordinatorRoutingTest(unittest.TestCase):
    def test_emap_warmth_does_not_route_chat(self):
        """Warm Mac EMAP must not steal coordinator role from fast Linux."""
        reg = NodeRegistry()
        reg.register("linux", "10.0.0.1", 8001, "m", {
            "host": "10.0.0.1", "emap": _emap(0, 0, 0, 0),
            "hwinfo": {"cores": 20, "ram_avail_gb": 58.0},
            "tiers_config": {"ram_gb": 50},
            "tiers": {"ram_gb": 11, "vram_gb": 0, "ram": 100, "vram": 0, "disk": 0},
        })
        reg.register("mac", "10.0.0.2", 8001, "m", {
            "host": "10.0.0.2", "emap": _emap(1, 1, 1, 1),
            "hwinfo": {"cores": 8, "ram_avail_gb": 8.0},
            "tiers_config": {"ram_gb": 8},
        })
        hot = {(0, i) for i in range(4)}
        self.assertEqual(reg.pick_with_affinity(hot).node_id, "linux")

    def test_missing_tiers_is_donor_only(self):
        reg = NodeRegistry()
        reg.register("linux", "10.0.0.1", 8001, "m", {
            "host": "10.0.0.1",
            "tiers_config": {"ram_gb": 50},
            "tiers": {"ram_gb": 11, "ram": 1, "vram": 0, "disk": 0},
            "hwinfo": {"cores": 16, "ram_avail_gb": 48.0},
        })
        reg.register("mac", "10.0.0.2", 8001, "m", {
            "host": "10.0.0.2", "hwinfo": {"cores": 8, "ram_avail_gb": 8.0},
        })
        roles = reg.node_roles()
        self.assertTrue(roles["linux"][0])
        self.assertFalse(roles["mac"][0])
        self.assertEqual(roles["mac"][1], "missing_tiers")
        self.assertEqual(reg.pick_coordinator().node_id, "linux")

    def test_slow_ram_exec_is_donor_only(self):
        reg = NodeRegistry()
        reg.register("fast", "10.0.0.1", 8001, "m", {
            "host": "10.0.0.1",
            "tiers_config": {"ram_gb": 50},
            "tiers": {"ram_gb": 11, "ram": 1, "vram": 0, "disk": 0},
            "costs": [{"layer": 0, "expert": 0, "tier": 1, "exec_us": 40_000}],
        })
        reg.register("slow", "10.0.0.2", 8001, "m", {
            "host": "10.0.0.2",
            "tiers_config": {"ram_gb": 8},
            "tiers": {"ram_gb": 6, "ram": 1, "vram": 0, "disk": 0},
            "costs": [{"layer": 0, "expert": 0, "tier": 1, "exec_us": 200_000}],
        })
        roles = reg.node_roles()
        self.assertTrue(roles["fast"][0])
        self.assertFalse(roles["slow"][0])
        self.assertEqual(roles["slow"][1], "slow_ram_exec")

    def test_fast_hardware_wins_when_all_cold(self):
        reg = NodeRegistry()
        cold = _emap(*([0] * 8))
        reg.register("slow", "10.0.0.2", 8001, "m", {
            "host": "10.0.0.2", "emap": cold,
            "hwinfo": {"cores": 8, "ram_avail_gb": 1.0},
            "tiers_config": {"ram_gb": 2},
            "tiers": {"ram_gb": 2, "ram": 0, "vram": 0, "disk": 0},
        })
        reg.register("fast", "10.0.0.1", 8001, "m", {
            "host": "10.0.0.1", "emap": cold,
            "hwinfo": {"cores": 20, "ram_avail_gb": 58.0},
            "tiers_config": {"ram_gb": 50},
            "tiers": {"ram_gb": 50, "ram": 0, "vram": 0, "disk": 0},
        })
        self.assertEqual(reg.pick_with_affinity({(0, i) for i in range(8)}).node_id, "fast")

    def test_proxy_and_agent_inflight_merge(self):
        reg = NodeRegistry()
        reg.register("a", "10.0.0.1", 8001, "m", {"host": "10.0.0.1", "tiers_config": {"ram_gb": 8}})
        reg.register("b", "10.0.0.2", 8001, "m", {"host": "10.0.0.2", "tiers_config": {"ram_gb": 8}})
        reg.increment_inflight("a", 2)
        reg.heartbeat("a", 1, {})
        self.assertEqual(reg._nodes["a"].agent_inflight, 1)
        self.assertEqual(reg._nodes["a"].proxy_inflight, 2)
        self.assertEqual(reg._nodes["a"].inflight, 3)
        reg.increment_inflight("a", -2)
        self.assertEqual(reg._nodes["a"].inflight, 1)

    def test_slow_node_penalty(self):
        reg = NodeRegistry()
        reg.register("fast", "10.0.0.1", 8001, "m", {
            "host": "10.0.0.1", "tiers_config": {"ram_gb": 50},
            "tiers": {"ram_gb": 11, "ram": 1, "vram": 0, "disk": 0},
        })
        reg.register("slow", "10.0.0.2", 8001, "m", {
            "host": "10.0.0.2", "tiers_config": {"ram_gb": 8},
            "tiers": {"ram_gb": 6, "ram": 1, "vram": 0, "disk": 0},
            "profile": [{"expert_wait_s": 5.0, "wall_s": 10.0}],
        })
        self.assertEqual(reg.pick_with_affinity({(0, 0)}).node_id, "fast")

    def test_slow_hw_is_donor_only_even_with_inferred_tiers(self):
        """Mac with synthesized RAM budget must not become coordinator vs Linux."""
        reg = NodeRegistry()
        reg.register("linux", "10.0.0.1", 8001, "m", {
            "host": "10.0.0.1",
            "tiers_config": {"ram_gb": 50},
            "tiers": {"ram_gb": 11, "ram": 1, "vram": 0, "disk": 0},
            "hwinfo": {"cores": 20, "ram_avail_gb": 58.0},
        })
        reg.register("mac", "10.0.0.2", 8001, "m", {
            "host": "10.0.0.2",
            "tiers_config": {"ram_gb": 4},
            "tiers": {"ram_gb": 4, "ram": 0, "vram": 0, "disk": 0},
            "hwinfo": {"cores": 8, "ram_avail_gb": 8.0},
        })
        roles = reg.node_roles()
        self.assertTrue(roles["linux"][0])
        self.assertFalse(roles["mac"][0])
        self.assertEqual(roles["mac"][1], "slow_hw")
        self.assertEqual(reg.pick_coordinator().node_id, "linux")

    def test_cluster_state_includes_coordinator_id(self):
        reg = NodeRegistry()
        reg.register("linux", "10.0.0.1", 8001, "m", {
            "host": "10.0.0.1", "tiers_config": {"ram_gb": 50},
            "tiers": {"ram_gb": 11, "ram": 1, "vram": 0, "disk": 0},
            "hwinfo": {"cores": 16, "ram_avail_gb": 48.0},
        })
        state = reg.cluster_state()
        self.assertEqual(state["coordinator_id"], "linux")
        node = next(n for n in state["nodes"] if n["node_id"] == "linux")
        self.assertTrue(node["coordinator_eligible"])


if __name__ == "__main__":
    unittest.main()
