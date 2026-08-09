"""Tests for Aviary master chat routing (affinity + cold-node bootstrap)."""

import unittest

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

    def test_cold_node_bootstrapped_when_peer_is_warmed(self):
        """205 vs 0 resident hot experts — traffic must not stay 100% on one node."""
        reg = NodeRegistry()
        warm_map = _emap(*([1] * 8))
        cold_map = _emap(*([0] * 8))
        reg.register("warm", "10.0.0.1", 8001, "m", {"host": "10.0.0.1", "emap": warm_map})
        reg.register("cold", "10.0.0.2", 8001, "m", {"host": "10.0.0.2", "emap": cold_map})
        hot = {(0, i) for i in range(8)}
        self.assertEqual(reg.pick_with_affinity(hot).node_id, "cold")

    def test_least_loaded_among_cold_peers(self):
        reg = NodeRegistry()
        reg.register("a", "10.0.0.1", 8001, "m", {"host": "10.0.0.1", "emap": _emap(1, 1)})
        reg.register("b", "10.0.0.2", 8001, "m", {"host": "10.0.0.2", "emap": _emap(0, 0)})
        reg.register("c", "10.0.0.3", 8001, "m", {"host": "10.0.0.3", "emap": _emap(0, 0)})
        reg.increment_inflight("b", 3)
        hot = {(0, 0), (0, 1)}
        self.assertEqual(reg.pick_with_affinity(hot).node_id, "c")


if __name__ == "__main__":
    unittest.main()
