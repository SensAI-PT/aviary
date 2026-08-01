"""Backend-DLL selection contract for the Windows runtime loader.

``c/backend_loader.c`` picks its DLL name and diagnostic label from
COLI_HIP_DLL: a CUDA_DLL host seeks coli_cuda.dll and says ``[CUDA]``, a
HIP_DLL host seeks coli_hip.dll and says ``[HIP]``. These tests prove that by
building both hosts under distinct EXE names and running each from an empty
directory, so the loader's not-found path is the one exercised.

Nothing here needs a GPU, nvidia-smi, a CUDA or HIP SDK, or a backend DLL: the
contract under test is the *miss*, which is reached before any vendor runtime.

The fixture helpers below are intentionally private copies rather than imports
from ``test_cuda_env``: that module probes the resident ``c/colibri.exe`` at
import time, and this owner must not depend on a binary it never runs.

Prerequisites (tests skip gracefully when unmet):
- Windows (backend_loader.c is compiled only there)
- MinGW make + gcc, to build the two host artifacts
"""

import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent.parent


def _write_shard(path, tensors):
    """Write a minimal safetensors file to *path*."""
    offset = 0
    header = {}
    payload = b""
    for name, size in tensors:
        header[name] = {"dtype": "U8", "shape": [size],
                        "data_offsets": [offset, offset + size]}
        payload += b"\0" * size
        offset += size
    raw = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + payload)


def _minimal_model(parent):
    """Create a minimal model directory that ``model_init()`` can parse.

    Returns the model Path.  The safetensors payload is dummy zeros — the
    binary will reach CUDA init, print its messages, then fail during
    model loading (fake weights).  The CUDA messages are already on stderr
    at that point.
    """
    model = Path(parent) / "model"
    model.mkdir()
    (model / "config.json").write_text(json.dumps({
        "num_hidden_layers": 2,
        "n_routed_experts": 2,
        "kv_lora_rank": 4,
        "qk_rope_head_dim": 2,
        "qk_nope_head_dim": 3,
        "v_head_dim": 5,
        "num_attention_heads": 2,
    }))
    _write_shard(model / "model.safetensors", [
        ("model.embed_tokens.weight", 100),
        ("model.layers.0.self_attn.q_a_proj.weight", 200),
        ("model.layers.1.mlp.experts.0.gate_proj.weight", 30),
        ("model.layers.1.mlp.experts.0.up_proj.weight", 30),
        ("model.layers.1.mlp.experts.1.gate_proj.weight", 30),
        ("model.layers.1.mlp.experts.1.up_proj.weight", 30),
    ])
    return model


class LoaderBackendSelectionTest(unittest.TestCase):
    """Which backend DLL does a Windows host look for, and how does it label its
    own loader messages?

    Deliberately a *miss*-path contract, so it needs no GPU, no vendor SDK and
    no backend DLL: each host is copied into an empty temp directory, and
    ``backend_loader.c`` resolves the DLL next to the executable — so the load
    is guaranteed to fail there and print exactly which file it wanted. That
    makes the assertion behavioural (run the artifact, read its stderr) rather
    than a source-text grep, and keeps it green on a machine with no NVIDIA
    driver, no ROCm and no ``nvidia-smi``.

    The two hosts are built under distinct EXE names so the shared
    ``c/colibri.exe`` other test modules probe is never disturbed.
    """

    _hosts = {}

    @classmethod
    def setUpClass(cls):
        if sys.platform != "win32":
            raise unittest.SkipTest("backend_loader.c is compiled only on Windows")
        if shutil.which("make") is None or shutil.which("gcc") is None:
            raise unittest.SkipTest("MinGW make/gcc needed to build the host matrix")
        # HIP_DLL=1 is a host-only build: it links backend_loader.o and no HIP
        # library, so it must succeed with no HIP SDK, HIP_PATH or HIP_ARCH.
        for key, exe_suffix, flag in (("cuda", "_cudadll.exe", "CUDA_DLL=1"),
                                      ("hip", "_hipdll.exe", "HIP_DLL=1")):
            target = "colibri" + exe_suffix
            # errors="replace": the toolchain speaks the console codepage, which
            # is not always decodable as the Python default on localized hosts.
            proc = subprocess.run(
                ["make", "-C", str(HERE), target, "EXE=" + exe_suffix, flag],
                cwd=str(HERE.parent), text=True, errors="replace",
                capture_output=True, timeout=600,
            )
            built = HERE / target
            if proc.returncode != 0 or not built.exists():
                raise unittest.SkipTest(
                    "could not build the %s host matrix: %s" % (key, proc.stderr[-400:])
                )
            cls._hosts[key] = built

    @classmethod
    def tearDownClass(cls):
        for path in cls._hosts.values():
            try:
                path.unlink()
            except OSError:
                pass
        cls._hosts.clear()

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp.cleanup()

    def _run_isolated(self, key, timeout=60, unset=()):
        """Copy one host into an empty dir and run it with the GPU requested.

        The directory holds the executable and the throwaway model only — no
        backend DLL — so the loader's not-found path is the one exercised.

        ``unset`` names environment variables to drop from the child *after*
        the defaults are applied, which is how a caller opts out of the
        ``COLI_CUDA=1`` request set below and tests the not-requested contract
        instead. The default removes nothing, so existing callers are unchanged.
        """
        # One sandbox per host, so a test may exercise both in a single method.
        sandbox = Path(self.tmp.name) / ("bin_" + key)
        sandbox.mkdir(exist_ok=True)
        exe = sandbox / self._hosts[key].name
        shutil.copy2(self._hosts[key], exe)
        model = _minimal_model(str(sandbox))
        merged = {**os.environ, "SNAP": str(model), "COLI_CUDA": "1"}
        for name in unset:
            merged.pop(name, None)
        return subprocess.run(
            [str(exe), "1"],
            env=merged, cwd=str(sandbox), text=True, errors="replace",
            capture_output=True, timeout=timeout,
        )

    def test_cuda_dll_host_reports_cuda_backend(self):
        """CUDA_DLL host: '[CUDA] coli_cuda.dll not found', never the HIP name."""
        err = self._run_isolated("cuda").stderr or ""
        self.assertIn("[CUDA] coli_cuda.dll not found", err)
        self.assertNotIn("coli_hip.dll", err)
        self.assertNotIn("[HIP]", err)

    def test_hip_dll_host_reports_hip_backend(self):
        """HIP_DLL host: '[HIP] coli_hip.dll not found', never the CUDA name.

        Runs with no AMD GPU, no HIP SDK and no nvidia-smi — the loader has not
        reached any vendor runtime at the point this message is printed.
        """
        err = self._run_isolated("hip").stderr or ""
        self.assertIn("[HIP] coli_hip.dll not found", err)
        self.assertNotIn("coli_cuda.dll", err)

    def test_cuda_dll_host_with_coli_cuda_unset_stays_on_cpu(self):
        """CUDA_DLL host, COLI_CUDA absent: no request, no loader, no message.

        This is the opt-in contract that ``test_cuda_env`` can only assert when
        a CUDA host happens to be the resident binary — here it always runs,
        because the host is built by this class and the sandbox guarantees no
        backend DLL. Nothing consults nvidia-smi, a GPU, a CUDA SDK or
        coli_cuda.dll: with COLI_CUDA removed the engine never reaches the
        loader at all, and continues to model validation on the CPU path.
        """
        result = self._run_isolated(
            "cuda", unset=("COLI_CUDA", "COLI_GPU", "COLI_GPUS"))
        err = result.stderr or ""
        self.assertNotEqual(result.returncode, 2)
        self.assertNotIn("[CUDA]", err)
        self.assertNotIn("coli_cuda.dll not found", err)
        self.assertNotIn("requested backend is unavailable", err)
        # Reachability oracle: the throwaway fixture is rejected by model
        # validation, which proves execution got past GPU setup rather than
        # exiting inside it.
        self.assertIn("this engine requires n_group=1", err)

    def test_backend_miss_is_not_silent_and_does_no_gpu_work(self):
        """Both hosts fail closed the same way: exit 2, no GPU touched."""
        for key in ("cuda", "hip"):
            with self.subTest(host=key):
                result = self._run_isolated(key)
                err = result.stderr or ""
                self.assertIn("not found; GPU tier disabled", err)
                # colibri.c owns this second line and stays vendor-neutral in
                # W1-B1: only the loader's own prefix is discriminated here.
                self.assertIn("[CUDA] requested backend is unavailable", err)
                self.assertEqual(result.returncode, 2)


if __name__ == "__main__":
    unittest.main()
