"""Tests for .coli_usage parsing and engine_id isolation."""

import tempfile
import unittest
from pathlib import Path

from aviary.usage import arch_engine_id, read_coli_usage, usage_delta


class UsageTest(unittest.TestCase):
    def test_read_and_delta(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / ".coli_usage"
            path.write_text("-1 2 4\n-2 1 999\n0 1 5\n1 2 3\n", encoding="utf-8")
            nl, ne, eid, records = read_coli_usage(path)
            self.assertEqual(nl, 2)
            self.assertEqual(ne, 4)
            self.assertEqual(len(records), 2)
            delta = usage_delta(records, [{"layer": 0, "expert": 1, "count": 7}, {"layer": 1, "expert": 2, "count": 3}])
            self.assertEqual(delta, [{"layer": 0, "expert": 1, "count": 2}])

    def test_engine_id_differs_by_arch(self):
        self.assertNotEqual(arch_engine_id("hy3"), arch_engine_id("qwen3_moe"))


if __name__ == "__main__":
    unittest.main()
