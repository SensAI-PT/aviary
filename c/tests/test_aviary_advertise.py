"""Tests for cluster advertise-host resolution on agents."""

import os
import unittest
from unittest import mock

from aviary.agent import _is_loopback_host, _resolve_cluster_advertise_host
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


if __name__ == "__main__":
    unittest.main()
