"""Tests for Aviary prefetch helpers."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from aviary.prefetch import download_shard, list_local_shards


class PrefetchTest(unittest.TestCase):
    def test_list_local_shards(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "model-00001.safetensors").write_bytes(b"x")
            (root / "readme.txt").write_text("nope")
            self.assertEqual(list_local_shards(root), {"model-00001.safetensors"})

    def test_download_shard_atomic(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "model-00002.safetensors"
            payload = b"shard-bytes"

            class FakeResp:
                def read(self):
                    return payload

                def __enter__(self):
                    return self

                def __exit__(self, *args):
                    return False

            with patch("urllib.request.urlopen", return_value=FakeResp()):
                self.assertTrue(download_shard("http://peer:8001", "model-00002.safetensors", dest))
            self.assertEqual(dest.read_bytes(), payload)


if __name__ == "__main__":
    unittest.main()
