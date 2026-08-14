"""Unit tests for cluster bench scoreboard and markdown (no live cluster)."""

from __future__ import annotations

import unittest

from aviary.bench import (
    PRESET_STEPS,
    build_scoreboard,
    compute_epa,
    hop_mix,
    render_markdown,
    summarize_latencies,
)


class TestBenchMath(unittest.TestCase):
    def test_epa_warm_faster_than_cold(self):
        self.assertAlmostEqual(compute_epa(40.0, 20.0), 1.0)

    def test_epa_missing_inputs(self):
        self.assertIsNone(compute_epa(None, 20.0))
        self.assertIsNone(compute_epa(40.0, 0.0))

    def test_hop_mix_percentages(self):
        jobs = [
            {"trace": [{"kind": "local"}, {"kind": "local"}, {"kind": "remote"}, {"kind": "fallback"}]},
            {"trace": [{"kind": "remote"}]},
        ]
        mix = hop_mix(jobs)
        self.assertEqual(mix["local"], 40.0)
        self.assertEqual(mix["remote"], 40.0)
        self.assertEqual(mix["fallback"], 20.0)

    def test_summarize_latencies(self):
        lat = summarize_latencies([1.0, 2.0, 3.0, 4.0, 100.0])
        self.assertEqual(lat["p50_sec"], 3.0)
        self.assertEqual(lat["min_sec"], 1.0)
        self.assertEqual(lat["max_sec"], 100.0)

    def test_scoreboard_suite_epa_and_rps(self):
        steps = [
            {"step": "cold_sequential", "latency": {"p50_sec": 30.0, "p95_sec": 45.0}, "jobs": []},
            {"step": "warm_sequential", "latency": {"p50_sec": 15.0, "p95_sec": 20.0}, "jobs": []},
            {"step": "concurrent", "rps": 2.5, "jobs": [{"trace": [{"kind": "local"}]}]},
        ]
        board = build_scoreboard(steps)
        self.assertEqual(board["p50_cold_sec"], 30.0)
        self.assertEqual(board["p50_warm_sec"], 15.0)
        self.assertEqual(board["epa"], 1.0)
        self.assertEqual(board["rps_at_w"], 2.5)
        self.assertEqual(board["hop_mix_pct"]["local"], 100.0)

    def test_markdown_contains_table_and_epa(self):
        result = {
            "finished_at": "2026-08-14T12:00:00Z",
            "meta": {"preset": "suite", "wipe_usage": True, "local_only": False, "requests": 8, "workers": 4, "max_tokens": 32},
            "scoreboard": {"p50_cold_sec": 30.0, "p50_warm_sec": 15.0, "epa": 1.0, "rps_at_w": 2.0,
                           "hop_mix_pct": {"local": 50.0, "remote": 40.0, "fallback": 10.0}},
            "steps": [{"step": "cold_sequential", "ok": 8, "requests": 8, "errors": 0,
                       "latency": {"p50_sec": 30.0, "p95_sec": 40.0}, "hop_mix_pct": {"local": 50.0, "remote": 40.0, "fallback": 10.0}}],
        }
        md = render_markdown(result)
        self.assertIn("## Aviary cluster bench", md)
        self.assertIn("| epa | 1.0 |", md)
        self.assertIn("cold_sequential", md)
        self.assertIn("p50=30.0s", md)

    def test_markdown_contains_cluster_config(self):
        result = {
            "finished_at": "2026-08-14T12:00:00Z",
            "cluster": {
                "cohort": {"model_id": "hy3-colibri", "arch": "hy3"},
                "nodes": [{"node_id": "abc-123", "host": "192.168.1.120", "ram_gb": 50, "vram_gb": 0,
                           "ram_total_gb": 64, "gpu": ""}],
            },
            "meta": {"preset": "cold_sequential", "wipe_usage": True, "local_only": False, "requests": 8, "workers": 1, "max_tokens": 32},
            "scoreboard": {},
            "steps": [],
        }
        md = render_markdown(result)
        self.assertIn("### Cluster config", md)
        self.assertIn("50 GB", md)
        self.assertIn("192.168.1.120", md)

    def test_suite_preset_wipe_flags(self):
        suite = PRESET_STEPS["suite"]
        self.assertTrue(suite[0]["wipe_usage"])
        self.assertFalse(suite[1]["wipe_usage"])
        self.assertFalse(suite[2]["wipe_usage"])
        self.assertEqual(suite[0]["workers"], 1)
        self.assertIsNone(suite[2]["workers"])


if __name__ == "__main__":
    unittest.main()
