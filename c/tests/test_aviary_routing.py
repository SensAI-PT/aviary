"""Tests for Aviary master chat routing (affinity + cold-node bootstrap)."""

import os
import unittest
from unittest import mock

from aviary.registry import NodeRegistry


def _emap(*tiers: int) -> dict:
    return {"rows": 1, "cols": len(tiers), "map": "".join(f"{t << 6:02x}" for t in tiers)}


class PickWithAffinityTest(unittest.TestCase):
    def test_warmed_node_wins_when_both_have_residents(self):
        reg = NodeRegistry()
        reg.register("a", "10.0.0.1", 8001, "m", {"host": "10.0.0.1", "emap": _emap(1, 1, 0, 0)})
        reg.register("b", "10.0.0.2", 8001, "m", {"host": "10.0.0.2", "emap": _emap(1, 0, 0, 0)})
        hot = {(0, 0), (0, 1), (0, 2), (0, 3)}
        self.assertEqual(reg.pick_with_affinity(hot).node_id, "a")

    @mock.patch.dict(os.environ, {"AVIARY_ROUTE_BOOTSTRAP_RATIO": "1.0"})
    @mock.patch("aviary.registry.random.random", return_value=0.0)
    def test_cold_node_bootstrapped_when_peer_is_warmed(self, _rand):
        """Cold executor receives traffic only when it is the faster node."""
        reg = NodeRegistry()
        warm_map = _emap(*([1] * 8))
        cold_map = _emap(*([0] * 8))
        # Warm = slow hardware, cold = fast — bootstrap should help the fast box.
        reg.register("warm", "10.0.0.1", 8001, "m", {
            "host": "10.0.0.1", "emap": warm_map,
            "hwinfo": {"cores": 8, "ram_avail_gb": 2.0},
        })
        reg.register("cold", "10.0.0.2", 8001, "m", {
            "host": "10.0.0.2", "emap": cold_map,
            "hwinfo": {"cores": 20, "ram_avail_gb": 58.0},
        })
        hot = {(0, i) for i in range(8)}
        self.assertEqual(reg.pick_with_affinity(hot).node_id, "cold")

    @mock.patch.dict(os.environ, {
        "AVIARY_ROUTE_BOOTSTRAP_RATIO": "1.0",
        "AVIARY_ROUTE_COLD_MIN_RATIO": "1.0",
    })
    @mock.patch("aviary.registry.random.random", return_value=0.0)
    def test_do_not_bootstrap_onto_slower_cold_peer(self, _rand):
        """Warm fast primary must keep traffic; do not lottery to a slow cold peer."""
        reg = NodeRegistry()
        warm_map = _emap(*([1] * 64))
        cold_map = _emap(*([0] * 64))
        reg.register("fast", "10.0.0.1", 8001, "m", {
            "host": "10.0.0.1", "emap": warm_map,
            "hwinfo": {"cores": 20, "ram_avail_gb": 58.0},
        })
        reg.register("slow", "10.0.0.2", 8001, "m", {
            "host": "10.0.0.2", "emap": cold_map,
            "hwinfo": {"cores": 8, "ram_avail_gb": 1.0},
        })
        hot = {(0, i) for i in range(8)}
        self.assertEqual(reg.pick_with_affinity(hot).node_id, "fast")

    @mock.patch.dict(os.environ, {"AVIARY_ROUTE_BOOTSTRAP_RATIO": "0.1"})
    @mock.patch("aviary.registry.random.random")
    def test_bootstrap_is_probabilistic_not_exclusive(self, rand):
        reg = NodeRegistry()
        # Cold must be faster than warm or bootstrap is skipped entirely.
        reg.register("warm", "10.0.0.1", 8001, "m", {
            "host": "10.0.0.1", "emap": _emap(1, 1, 1, 1),
            "hwinfo": {"cores": 8, "ram_avail_gb": 2.0},
        })
        reg.register("cold", "10.0.0.2", 8001, "m", {
            "host": "10.0.0.2", "emap": _emap(0, 0, 0, 0),
            "hwinfo": {"cores": 20, "ram_avail_gb": 58.0},
        })
        hot = {(0, i) for i in range(4)}
        rand.side_effect = [0.05, 0.5, 0.05, 0.5]
        picks = [reg.pick_with_affinity(hot).node_id for _ in range(4)]
        self.assertEqual(picks.count("cold"), 2)
        self.assertEqual(picks.count("warm"), 2)

    @mock.patch.dict(os.environ, {"AVIARY_ROUTE_BOOTSTRAP_RATIO": "1.0"})
    @mock.patch("aviary.registry.random.random", return_value=0.0)
    def test_least_loaded_among_cold_peers(self, _rand):
        reg = NodeRegistry()
        reg.register("a", "10.0.0.1", 8001, "m", {
            "host": "10.0.0.1", "emap": _emap(1, 1),
            "hwinfo": {"cores": 4, "ram_avail_gb": 2.0},
        })
        reg.register("b", "10.0.0.2", 8001, "m", {
            "host": "10.0.0.2", "emap": _emap(0, 0),
            "hwinfo": {"cores": 16, "ram_avail_gb": 40.0},
        })
        reg.register("c", "10.0.0.3", 8001, "m", {
            "host": "10.0.0.3", "emap": _emap(0, 0),
            "hwinfo": {"cores": 32, "ram_avail_gb": 60.0},
        })
        reg.increment_inflight("b", 3)
        hot = {(0, 0), (0, 1)}
        self.assertEqual(reg.pick_with_affinity(hot).node_id, "c")

    def test_proxy_and_agent_inflight_merge(self):
        reg = NodeRegistry()
        reg.register("a", "10.0.0.1", 8001, "m", {"host": "10.0.0.1"})
        reg.register("b", "10.0.0.2", 8001, "m", {"host": "10.0.0.2"})
        reg.increment_inflight("a", 2)
        reg.heartbeat("a", 1, {})
        self.assertEqual(reg._nodes["a"].agent_inflight, 1)
        self.assertEqual(reg._nodes["a"].proxy_inflight, 2)
        self.assertEqual(reg._nodes["a"].inflight, 3)
        reg.increment_inflight("a", -2)
        self.assertEqual(reg._nodes["a"].inflight, 1)

    def test_slow_node_penalty(self):
        reg = NodeRegistry()
        reg.register("fast", "10.0.0.1", 8001, "m", {"host": "10.0.0.1", "emap": _emap(1, 1)})
        reg.register("slow", "10.0.0.2", 8001, "m", {
            "host": "10.0.0.2", "emap": _emap(1, 1),
            "profile": [{"expert_wait_s": 5.0, "wall_s": 10.0}],
        })
        hot = {(0, 0), (0, 1)}
        self.assertEqual(reg.pick_with_affinity(hot).node_id, "fast")

    @mock.patch.dict(os.environ, {"AVIARY_ROUTE_COLD_MIN_RATIO": "1.0"})
    @mock.patch("aviary.registry.random.random", return_value=0.0)
    def test_zero_resident_fast_node_bootstrapped_over_warm_slow(self, _rand):
        """Warm-lock escape: cold fast node gets traffic when peer has many residents."""
        reg = NodeRegistry()
        warm_map = _emap(*([1] * 64))
        cold_map = _emap(*([0] * 64))
        reg.register("fast", "10.0.0.1", 8001, "m", {"host": "10.0.0.1", "emap": cold_map})
        reg.register("slow", "10.0.0.2", 8001, "m", {
            "host": "10.0.0.2", "emap": warm_map,
            "profile": [{"wall_s": 120.0, "expert_wait_s": 30.0}],
        })
        hot = {(0, i) for i in range(8)}
        self.assertEqual(reg.pick_with_affinity(hot).node_id, "fast")

    def test_warm_lock_escape_without_bootstrap_lottery(self):
        """Deterministic escape when warm node profile is much slower than cold."""
        reg = NodeRegistry()
        warm_map = _emap(*([1] * 40))
        cold_map = _emap(*([0] * 40))
        reg.register("fast", "10.0.0.1", 8001, "m", {
            "host": "10.0.0.1", "emap": cold_map,
            "hwinfo": {"cores": 20, "ram_avail_gb": 58.0},
        })
        reg.register("slow", "10.0.0.2", 8001, "m", {
            "host": "10.0.0.2", "emap": warm_map,
            "hwinfo": {"cores": 8, "ram_avail_gb": 1.2},
            "profile": [{"wall_s": 200.0, "expert_wait_s": 50.0}],
        })
        hot = {(0, i) for i in range(8)}
        with mock.patch.dict(os.environ, {"AVIARY_ROUTE_BOOTSTRAP_RATIO": "0"}):
            self.assertEqual(reg.pick_with_affinity(hot).node_id, "fast")

    def test_all_cold_prefers_faster_hardware_not_uuid(self):
        """When nobody has residents, do not fall through to node_id lottery."""
        reg = NodeRegistry()
        cold = _emap(*([0] * 8))
        # Lexicographically smaller UUID is the *slow* node — old code picked it.
        reg.register("4f9668ff-slow", "10.0.0.2", 8001, "m", {
            "host": "10.0.0.2", "emap": cold,
            "hwinfo": {"cores": 8, "ram_avail_gb": 1.0},
        })
        reg.register("5a2a179f-fast", "10.0.0.1", 8001, "m", {
            "host": "10.0.0.1", "emap": cold,
            "hwinfo": {"cores": 20, "ram_avail_gb": 58.0},
        })
        hot = {(0, i) for i in range(8)}
        self.assertEqual(reg.pick_with_affinity(hot).node_id, "5a2a179f-fast")


if __name__ == "__main__":
    unittest.main()
