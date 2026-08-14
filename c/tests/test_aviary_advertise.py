"""Tests for cluster advertise-host resolution on agents."""

import os
import unittest
from unittest import mock

from aviary.agent import (
    _is_loopback_host,
    _resolve_cluster_advertise_host,
    attach_tier_telemetry,
    infer_ram_budget_gb,
    tiers_config_from_env,
)
from aviary.tiers_util import sanitize_tiers


class AdvertiseHostTest(unittest.TestCase):
    def test_loopback_detection(self):
        self.assertTrue(_is_loopback_host("127.0.0.1"))
        self.assertTrue(_is_loopback_host("localhost"))
        self.assertFalse(_is_loopback_host("192.168.1.120"))

    @mock.patch.dict(os.environ, {"AVIARY_CLUSTER": "1"})
    @mock.patch("aviary.agent._local_ip_for", return_value="192.168.1.120")
    def test_loopback_replaced_under_cluster(self, _ip):
        host = _resolve_cluster_advertise_host("127.0.0.1", "0.0.0.0", "127.0.0.1", 9002)
        self.assertEqual(host, "192.168.1.120")

    @mock.patch.dict(os.environ, {"AVIARY_CLUSTER": "0"})
    @mock.patch("aviary.agent._local_ip_for", return_value="192.168.1.120")
    def test_loopback_replaced_when_binding_all_interfaces(self, _ip):
        host = _resolve_cluster_advertise_host("127.0.0.1", "0.0.0.0", "127.0.0.1", 9002)
        self.assertEqual(host, "192.168.1.120")

    def test_sanitize_rejects_absurd_tier_counts(self):
        self.assertIsNone(sanitize_tiers({"vram": 0, "ram": 82373088, "disk": 0}))
        self.assertEqual(sanitize_tiers({"vram": 0, "ram": 12, "disk": 100})["ram"], 12)
        cleaned = sanitize_tiers({"vram": 0, "ram": 12, "disk": 0, "ram_gb": 218635.0})
        self.assertEqual(cleaned["ram_gb"], 0.0)


class TiersTelemetryTest(unittest.TestCase):
    def test_infer_ram_from_hwinfo_when_no_flag(self):
        self.assertEqual(infer_ram_budget_gb({}, {"ram_total_gb": 17.2, "ram_avail_gb": 8.0}), 4.0)
        self.assertEqual(infer_ram_budget_gb({"RAM_GB": "8"}, {"ram_total_gb": 17.2}), 8.0)

    def test_mac_without_engine_tiers_still_reports(self):
        class Eng:
            tiers = None
            hwinfo = {"cores": 8, "ram_total_gb": 17.2, "ram_avail_gb": 8.0}
            gpu_tier = None
        payload = attach_tier_telemetry({}, Eng(), {})
        self.assertEqual(payload["tiers_config"]["ram_gb"], 4.0)
        self.assertEqual(payload["tiers"]["ram_gb"], 4.0)
        self.assertEqual(payload["hwinfo"]["ram_total_gb"], 17.2)

    def test_explicit_vram_flag_in_config(self):
        cfg = tiers_config_from_env({"CUDA_EXPERT_GB": "14", "RAM_GB": "50"})
        self.assertEqual(cfg["vram_gb"], 14.0)
        self.assertEqual(cfg["ram_gb"], 50.0)


if __name__ == "__main__":
    unittest.main()
