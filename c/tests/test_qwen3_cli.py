#!/usr/bin/env python3
"""CLI routing smoke tests for Qwen3 MoE."""
import importlib.util
import json
import sys
import tempfile
import unittest
from importlib.machinery import SourceFileLoader
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
CLI = HERE / "coli"


class Qwen3CliTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        loader = SourceFileLoader("coli_qwen3_test", str(CLI))
        spec = importlib.util.spec_from_loader(loader.name, loader)
        cls.coli = importlib.util.module_from_spec(spec)
        loader.exec_module(cls.coli)

    def test_model_arch_routes_qwen3_moe(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp) / "m"
            model.mkdir()
            (model / "config.json").write_text(json.dumps({"model_type": "qwen3_moe"}))
            self.assertEqual(self.coli.model_arch(str(model)), "qwen3_moe")

    def test_argv_for_arch_includes_int4_tail(self):
        self.assertEqual(self.coli.argv_for_arch("qwen3_moe", 64), ["64", "4", "8"])

    def test_is_qwen3_moe_repo(self):
        self.assertTrue(self.coli.is_qwen3_moe_repo("Qwen/Qwen3-30B-A3B-Instruct-2507"))


if __name__ == "__main__":
    unittest.main()
