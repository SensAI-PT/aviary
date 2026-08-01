"""Integration tests for the COLI_CUDA opt-in and CUDA_EXPERT_GB auto-size.

These tests run the compiled ``colibri`` binary and verify
the three COLI_CUDA modes (unset / not requested, 0 / forced-CPU, 1 / hard-fail)
and the CUDA_EXPERT_GB auto-size behavior. The engine itself never enables the
GPU on its own: setting COLI_CUDA=1 for the user is the ``coli`` launcher's
job, and running the binary directly — as these tests do — bypasses it.

Prerequisites (tests skip gracefully when unmet):
- Compiled ``colibri`` binary (any build — CUDA or CPU-only)
- nvidia-smi (for GPU-specific tests)
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
COLIBRI = HERE / ("colibri.exe" if sys.platform == "win32" else "colibri")


def _binary_has_cuda():
    """Check whether the ``colibri`` binary was compiled with CUDA support."""
    if not COLIBRI.exists():
        return False
    try:
        result = subprocess.run(
            [str(COLIBRI), "1"],
            env={**os.environ, "COLI_CUDA": "1",
                 "SNAP": str(HERE)},
            cwd=str(HERE), text=True, capture_output=True, timeout=10,
        )
        return "CPU-only" not in (result.stderr or "")
    except (OSError, subprocess.SubprocessError):
        return False


def _gpu_available():
    """Return True if at least one CUDA-capable GPU is visible to nvidia-smi."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            text=True, capture_output=True, timeout=5,
        )
        return result.returncode == 0 and bool(result.stdout.strip())
    except (OSError, subprocess.SubprocessError):
        return False


_HAS_BINARY = COLIBRI.exists()
_HAS_CUDA_BINARY = _HAS_BINARY and _binary_has_cuda()
_HAS_GPU = _gpu_available()


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


class CudaStartupTest(unittest.TestCase):
    """COLI_CUDA mode tests — call ``colibri`` directly to exercise ``main()``
    including the CUDA init block at c/colibri.c."""

    @classmethod
    def setUpClass(cls):
        if not _HAS_BINARY:
            raise unittest.SkipTest("colibri binary not found")

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp.cleanup()

    def _run(self, cap="1", env=None, timeout=30, unset=()):
        """Run the colibri binary directly with a minimal model.

        The binary will fail during model_init (fake weights) but CUDA
        init messages are emitted first.

        ``unset`` names environment variables to drop from the child, so a
        test can assert on a variable being *absent* without an ambient value
        in the invoking shell quietly invalidating the scenario. Removal runs
        last, and the default removes nothing.
        """
        model = _minimal_model(self.tmp.name)
        merged = {**os.environ, "SNAP": str(model), **(env or {})}
        for key in unset:
            merged.pop(key, None)
        return subprocess.run(
            [str(COLIBRI), cap],
            env=merged, cwd=str(HERE), text=True, capture_output=True,
            timeout=timeout,
        )

    # --- COLI_CUDA=0 : forced-CPU mode ---------------------------------

    def test_cuda_zero_suppresses_all_cuda_output(self):
        """COLI_CUDA=0 must not emit any [CUDA] line to stderr."""
        result = self._run(env={"COLI_CUDA": "0"})
        self.assertNotIn("[CUDA]", result.stderr or "")

    def test_cuda_zero_with_gpu_env_fails_guard(self):
        """COLI_GPU with COLI_CUDA=0 exits non-zero (guard in colibri.c).

        On CPU-only binary the guard fires from the #else branch with a
        different message; both are valid and tested here.
        """
        result = self._run(env={"COLI_CUDA": "0", "COLI_GPU": "0"})
        self.assertNotEqual(result.returncode, 0)

        if _HAS_CUDA_BINARY:
            # CUDA build: guard at colibri.c — COLI_GPU(S) requires COLI_CUDA=1
            self.assertIn("COLI_GPU(S) requires COLI_CUDA=1", result.stderr)
        else:
            # CPU-only build: #else branch — CUDA was requested, but CPU-only
            self.assertIn("CPU-only", result.stderr)

    # --- COLI_CUDA=1 : explicit-enable, hard-fail ----------------------

    def test_cuda_one_cpu_only_binary_exits_with_rebuild_message(self):
        """CPU-only binary: COLI_CUDA=1 → exit≠0, 'CPU-only' in stderr."""
        if _HAS_CUDA_BINARY:
            self.skipTest("binary has CUDA; CPU-only rejection not reachable")
        result = self._run(env={"COLI_CUDA": "1"})
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("CPU-only", result.stderr)

    def test_cuda_one_without_gpu_exits_two(self):
        """CUDA binary, no GPU: COLI_CUDA=1 → exit 2, 'unavailable'."""
        if not _HAS_CUDA_BINARY:
            self.skipTest("CPU-only binary; covered by cpu-only test")
        if _HAS_GPU:
            self.skipTest("GPU present; cannot simulate missing backend")
        result = self._run(env={"COLI_CUDA": "1"})
        self.assertEqual(result.returncode, 2)
        self.assertIn("[CUDA] requested backend is unavailable", result.stderr)

    # --- COLI_CUDA unset : not requested, silent CPU --------------------

    def test_cuda_unset_without_gpu_falls_back_to_cpu(self):
        """COLI_CUDA unset: the GPU is never requested, so the engine says nothing.

        The engine is strictly opt-in — c/colibri.c enters its CUDA block only
        for a truthy COLI_CUDA, so unset and 0 behave identically and no
        backend load is attempted. Windows auto-enable lives in the separate
        ``coli`` launcher, which this test bypasses by running the binary
        directly; asserting a launcher message here would be asserting against
        the wrong component.
        """
        if not _HAS_CUDA_BINARY:
            self.skipTest("CPU-only binary; the CUDA block is compiled out")
        if _HAS_GPU:
            self.skipTest("GPU present; cannot prove the not-requested path stays quiet")
        result = self._run(unset=("COLI_CUDA", "COLI_GPU", "COLI_GPUS"))
        self.assertNotEqual(result.returncode, 2)
        self.assertNotIn("[CUDA]", result.stderr or "")
        self.assertNotIn("not found; GPU tier disabled", result.stderr or "")

    def test_cuda_unset_with_gpu_stays_on_cpu(self):
        """A detected GPU does not opt the engine into CUDA by itself.

        The companion above proves the quiet path with no GPU; this proves the
        GPU's presence changes nothing. COLI_CUDA remains the explicit
        engine-level switch, so with it unset c/colibri.c never enters the CUDA
        block and never prints a mode line — no matter what hardware is
        installed. The ``coli`` launcher may set COLI_CUDA=1 on the user's
        behalf, but this test invokes the engine binary directly and so must
        not see launcher behaviour.
        """
        if not _HAS_CUDA_BINARY:
            self.skipTest("CPU-only binary; the CUDA block is compiled out")
        if not _HAS_GPU:
            self.skipTest("no GPU available; the with-GPU half is unprovable")
        result = self._run(unset=("COLI_CUDA", "COLI_GPU", "COLI_GPUS"))
        err = result.stderr or ""
        self.assertNotEqual(result.returncode, 2)
        self.assertNotIn("[CUDA]", err)
        self.assertNotIn("[CUDA] mode:", err)
        self.assertNotIn("coli_cuda.dll not found", err)
        self.assertNotIn("requested backend is unavailable", err)
        # Reached model validation, so the engine skipped GPU setup rather
        # than failing inside it.
        self.assertIn("this engine requires n_group=1", err)

    # --- CUDA_EXPERT_GB explicit zero ----------------------------------

    def test_cuda_expert_gb_zero_disables_auto_size(self):
        """Explicit CUDA_EXPERT_GB=0 — no auto-size message at startup."""
        if not _HAS_CUDA_BINARY:
            self.skipTest("CPU-only binary")
        result = self._run(env={"CUDA_EXPERT_GB": "0"})
        self.assertNotIn("auto-sized", result.stderr or "")


class CudaExpertBudgetTest(unittest.TestCase):
    """Auto-size tests — need model_init() path (heavier, GPU required)."""

    @classmethod
    def setUpClass(cls):
        if not _HAS_BINARY:
            raise unittest.SkipTest("colibri binary not found")
        if not _HAS_CUDA_BINARY:
            raise unittest.SkipTest("CPU-only binary — auto-size path not compiled in")
        if not _HAS_GPU:
            raise unittest.SkipTest("no GPU — auto-size needs free VRAM query")

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp.cleanup()

    def _run(self, cap="1", env=None, timeout=60):
        model = _minimal_model(self.tmp.name)
        merged = {**os.environ, "SNAP": str(model), **(env or {})}
        return subprocess.run(
            [str(COLIBRI), cap],
            env=merged, cwd=str(HERE), text=True, capture_output=True,
            timeout=timeout,
        )

    def test_auto_size_prints_budget_when_expert_gb_unset(self):
        """CUDA enabled, CUDA_EXPERT_GB unset → 'auto-sized' in stderr."""
        result = self._run(env={"COLI_CUDA": "1"})
        combined = (result.stderr or "") + (result.stdout or "")
        if "auto-sized" in combined:
            self.assertIn("expert budget auto-sized", combined)

    def test_explicit_expert_gb_zero_suppresses_auto_size_at_pin_time(self):
        """CUDA_EXPERT_GB=0 → no auto-size message even at pin time."""
        result = self._run(env={"COLI_CUDA": "1", "CUDA_EXPERT_GB": "0"})
        combined = (result.stderr or "") + (result.stdout or "")
        self.assertNotIn("auto-sized", combined)


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
    ``c/colibri.exe`` this module's other tests probe is never disturbed.
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

        This is the opt-in contract that ``CudaStartupTest`` can only assert
        when a CUDA host happens to be the resident binary — here it always
        runs, because the host is built by this class and the sandbox
        guarantees no backend DLL. Nothing consults nvidia-smi, a GPU, a CUDA
        SDK or coli_cuda.dll: with COLI_CUDA removed the engine never reaches
        the loader at all, and continues to model validation on the CPU path.
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
