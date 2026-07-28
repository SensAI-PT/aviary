"""int4-rans256-g0 offline-tool e2e: repack_rans.py + rans_verify.py against
synthetic per-row-int4 fixtures.

Covers the tool-level contract the C oracle (test_rans.c) cannot see:
  - end-to-end lossless round trip through real shard files (original weight
    bytes reproduced exactly; .qs sidecars byte-identical pass-through);
  - writer determinism (two runs diff clean, including a C-codec run vs a
    forced pure-Python run — the two implementations must emit identical
    bytes, not just equivalent ones);
  - the MANDATORY stamp + per-shard table metadata shape;
  - the verifier's named-refusal battery: every corruption class produces
    its class, enumerated not sampled;
  - writer-side geometry refusals (g64-signature and missing-scales inputs
    are refused by name, never silently entropy-coded).
"""
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

np = None
rf = None
try:
    import numpy as np

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
    import rans_format as rf
except ImportError:  # pragma: no cover - dependency-free CI skips these
    pass

TOOLS = Path(__file__).resolve().parent.parent / "tools"


def run_tool(script, *args, env_extra=None):
    env = dict(os.environ)
    if env_extra:
        env.update(env_extra)
    return subprocess.run([sys.executable, str(TOOLS / script), *args],
                          capture_output=True, text=True, env=env)


@unittest.skipIf(rf is None, "numpy not available (offline-tooling test)")
class TestRansRepack(unittest.TestCase):
    O, I = 8, 128           # per-row ratio I/2 = 64 (safely above the g64 signature)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.indir = self.root / "int4"
        self.indir.mkdir()
        rng = np.random.default_rng(42)
        self.weights = {}
        tensors_a, tensors_b = [], []
        wb = self.O * self.I // 2
        for e in range(3):
            for proj in ("gate", "up", "down"):
                name = f"model.layers.0.mlp.experts.{e}.{proj}_proj.weight"
                # skewed 15-symbol nibbles (value 0 never used, like the corpus)
                nib = rng.choice(
                    np.arange(1, 16, dtype=np.uint8), size=wb * 2,
                    p=np.array([30, 5, 4, 3, 3, 2, 2, 20, 10, 6, 5, 4, 3, 2, 1],
                               dtype=np.float64) / 100.0)
                raw = rf.pack_nibbles(nib)
                qs = rng.standard_normal(self.O).astype(np.float32).tobytes()
                self.weights[name] = (raw, qs)
                tensors_a.append((name, "U8", [wb], raw))
                # one expert's sidecars live in a SECOND shard: the tool must
                # find them through its corpus-wide index
                (tensors_b if e == 2 else tensors_a).append(
                    (name + ".qs", "F32", [self.O], qs))
        tensors_a.append(("model.norm.weight", "F32", [4],
                          np.ones(4, dtype=np.float32).tobytes()))
        rf.write_shard(str(self.indir / "out-00000.safetensors"), tensors_a, {})
        rf.write_shard(str(self.indir / "out-00001.safetensors"), tensors_b, {})

    def tearDown(self):
        self.tmp.cleanup()

    def repack(self, outdir, **env):
        r = run_tool("repack_rans.py", "--indir", str(self.indir),
                     "--outdir", str(outdir), env_extra=env or None)
        self.assertEqual(r.returncode, 0, msg=r.stdout + r.stderr)
        return sorted(Path(outdir).glob("*.safetensors"))

    def test_roundtrip_stamp_and_determinism(self):
        out1 = self.repack(self.root / "out1")
        self.assertEqual(len(out1), 1)      # only shard A holds weight tensors
        header, data_start = rf.read_header(str(out1[0]))
        meta = header["__metadata__"]

        # MANDATORY stamp + table metadata
        fmt_map = json.loads(meta[rf.METADATA_KEY])
        self.assertEqual(sorted(fmt_map), sorted(self.weights))
        self.assertTrue(all(v == rf.FORMAT_NAME for v in fmt_map.values()))
        table = rf.parse_table_blob(meta[rf.TABLE_KEY])
        self.assertEqual(table["table_id"], rf.TABLE_ID)
        self.assertEqual(int(table["freq"][0]), 0)   # 15-symbol corpus shape

        # byte-exact round trip + raw .qs pass-through
        for name, (raw, qs) in self.weights.items():
            blob, _ = rf.read_tensor_bytes(str(out1[0]), header, data_start, name)
            self.assertEqual(rf.decode_record(blob, table), raw, name)
            qblob, _ = rf.read_tensor_bytes(str(out1[0]), header, data_start,
                                            name + ".qs")
            self.assertEqual(qblob, qs, name + ".qs")

        # determinism: an independent second run is byte-identical
        out2 = self.repack(self.root / "out2")
        self.assertEqual(out1[0].read_bytes(), out2[0].read_bytes())

        # implementation identity: a forced pure-Python run emits the SAME
        # bytes as the (lib-backed, when built) default run
        out3 = self.repack(self.root / "out3", RANS_FORCE_PYTHON="1")
        self.assertEqual(out1[0].read_bytes(), out3[0].read_bytes())

        # the validator trusts it
        r = run_tool("rans_verify.py", str(out1[0]))
        self.assertEqual(r.returncode, 0, msg=r.stdout + r.stderr)
        self.assertIn("RESULT: TRUST", r.stdout)

    def _reshard(self, shard, meta_mutator=None, tensor_mutator=None):
        """Read a shard fully, apply mutators, rewrite it deterministically."""
        header, data_start = rf.read_header(str(shard))
        meta = dict(header["__metadata__"])
        tensors = []
        for name, info in header.items():
            if name == "__metadata__":
                continue
            blob, _ = rf.read_tensor_bytes(str(shard), header, data_start, name)
            if tensor_mutator:
                blob = tensor_mutator(name, bytearray(blob))
            tensors.append((name, info["dtype"], info["shape"], bytes(blob)))
        if meta_mutator:
            meta = meta_mutator(meta)
        rf.write_shard(str(shard), tensors, meta)

    def _corrupt_reshard_and_expect(self, expected_code, meta_mutator=None,
                                    tensor_mutator=None):
        outdir = self.root / f"corrupt_{expected_code}"
        shard = self.repack(outdir)[0]
        self._reshard(shard, meta_mutator, tensor_mutator)
        r = run_tool("rans_verify.py", str(shard))
        self.assertEqual(r.returncode, 1,
                         msg=f"{expected_code}: verifier did not refuse\n"
                             f"{r.stdout}{r.stderr}")
        self.assertIn(expected_code, r.stdout,
                      msg=f"wanted {expected_code} in:\n{r.stdout}")

    def test_refusal_battery(self):
        target = "model.layers.0.mlp.experts.0.gate_proj.weight"

        def flip_payload(name, blob):
            if name == target:
                blob[rf.record_header_bytes() + 5] ^= 0x10   # payload corruption
            return blob

        def flip_count(name, blob):
            if name == target:
                blob[8] ^= 1                                  # packed_bytes field
            return blob

        def break_offsets(name, blob):
            if name == target:
                blob[16 + 8 * 4] ^= 0x80                      # offsets landmine
            return blob

        def truncate(name, blob):
            if name == target:
                del blob[len(blob) - 16:]                     # drop final 16 bytes
            return blob

        def pad_dirt(name, blob):
            if name == target:
                blob[16 + (rf.N_STREAMS + 1) * 4] = 0x77      # header pad byte
            return blob

        def drop_stamp(meta):
            del meta[rf.METADATA_KEY]
            return meta

        def drop_table(meta):
            del meta[rf.TABLE_KEY]
            return meta

        def break_crc(meta):
            blob = json.loads(meta[rf.TABLE_KEY])
            blob["table_crc32"] = "00000000"
            meta[rf.TABLE_KEY] = json.dumps(blob, sort_keys=True)
            return meta

        def break_freq(meta):
            blob = json.loads(meta[rf.TABLE_KEY])
            blob["freq"][15] += 1
            meta[rf.TABLE_KEY] = json.dumps(blob, sort_keys=True)
            return meta

        def dangling_stamp(meta):
            fmt = json.loads(meta[rf.METADATA_KEY])
            fmt["model.layers.0.mlp.experts.99.gate_proj.weight"] = rf.FORMAT_NAME
            meta[rf.METADATA_KEY] = json.dumps(fmt, sort_keys=True)
            return meta

        # payload corruption trips a stream invariant or the re-encode pin;
        # both are refusals — assert on the shared prefix
        self._corrupt_reshard_and_expect("E_", tensor_mutator=flip_payload)
        self._corrupt_reshard_and_expect("E_COUNT_MISMATCH",
                                         tensor_mutator=flip_count)
        self._corrupt_reshard_and_expect("E_", tensor_mutator=break_offsets)
        self._corrupt_reshard_and_expect("E_", tensor_mutator=truncate)
        self._corrupt_reshard_and_expect("E_PAD_NONZERO", tensor_mutator=pad_dirt)
        self._corrupt_reshard_and_expect("E_STAMP_MISSING", meta_mutator=drop_stamp)
        self._corrupt_reshard_and_expect("E_TABLE_MISSING", meta_mutator=drop_table)
        self._corrupt_reshard_and_expect("E_TABLE_CRC", meta_mutator=break_crc)
        self._corrupt_reshard_and_expect("E_TABLE_FREQ_SUM", meta_mutator=break_freq)
        self._corrupt_reshard_and_expect("E_STAMP_DANGLING",
                                         meta_mutator=dangling_stamp)

    def test_geometry_refusals(self):
        # g64 signature: weight/scale ratio exactly 32 -> named refusal
        g64dir = self.root / "g64"
        g64dir.mkdir()
        wb = 2048
        rows = wb // 32
        rf.write_shard(str(g64dir / "out-00000.safetensors"), [
            ("model.layers.0.mlp.experts.0.gate_proj.weight", "U8", [wb],
             bytes(wb)),
            ("model.layers.0.mlp.experts.0.gate_proj.weight.qs", "F32", [rows],
             bytes(rows * 4)),
        ], {})
        r = run_tool("repack_rans.py", "--indir", str(g64dir),
                     "--outdir", str(self.root / "g64out"))
        self.assertEqual(r.returncode, 1)
        self.assertIn("E_GEOMETRY_G64", r.stderr)

        # missing .qs sidecar -> named refusal
        nsdir = self.root / "noscales"
        nsdir.mkdir()
        rf.write_shard(str(nsdir / "out-00000.safetensors"), [
            ("model.layers.0.mlp.experts.0.gate_proj.weight", "U8", [64],
             bytes(64)),
        ], {})
        r = run_tool("repack_rans.py", "--indir", str(nsdir),
                     "--outdir", str(self.root / "nsout"))
        self.assertEqual(r.returncode, 1)
        self.assertIn("E_GEOMETRY_NO_SCALES", r.stderr)

        # nothing selected -> loud error, no silent empty container
        emptydir = self.root / "empty"
        emptydir.mkdir()
        rf.write_shard(str(emptydir / "out-00000.safetensors"), [
            ("model.norm.weight", "F32", [4], bytes(16)),
        ], {})
        r = run_tool("repack_rans.py", "--indir", str(emptydir),
                     "--outdir", str(self.root / "emptyout"))
        self.assertEqual(r.returncode, 1)
        self.assertIn("refusing to emit an empty container", r.stderr)


if __name__ == "__main__":
    unittest.main()
