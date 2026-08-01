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

The stub fixtures below additionally use ``objdump`` for structural inspection
(PE machine type, imports, exports). That is the objdump bundled with the same
MinGW/MSYS2 binutils as the ``gcc`` these tests already require — it is not a
new dependency, and the fixture class skips honestly if it is missing.

Prerequisites (tests skip gracefully when unmet):
- Windows (backend_loader.c is compiled only there)
- MinGW make + gcc, to build the two host artifacts
- MinGW gcc + objdump, for the stub fixtures
"""

import hashlib
import json
import os
import re
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


class FixtureBuildError(AssertionError):
    """A stub fixture could not be built — distinct from a loader-contract failure.

    Carrying the compiler command and its complete output means a broken
    fixture is diagnosed as such instead of masquerading as a production bug.
    """


def _derive_backend_abi():
    """Parse the loader's own RESOLVE macros for the export set it requires.

    The ABI is read from c/backend_loader.c rather than restated here, so the
    stub cannot silently drift from the contract it is meant to satisfy: add a
    symbol to the loader and the fixture exports it on the next run.
    """
    source = (HERE / "backend_loader.c").read_text(encoding="utf-8", errors="replace")
    mandatory = ["coli_cuda_" + m for m in
                 re.findall(r"^\s+RESOLVE\((\w+),", source, re.M)]
    optional = ["coli_cuda_" + m for m in
                re.findall(r"^\s+RESOLVE_OPT\((\w+),", source, re.M)]
    if not mandatory:
        raise FixtureBuildError("no RESOLVE symbols parsed from backend_loader.c")
    overlap = set(mandatory) & set(optional)
    if overlap:
        raise FixtureBuildError("symbol in both RESOLVE and RESOLVE_OPT: %s"
                                % sorted(overlap))
    names = mandatory + optional
    bad = [n for n in names if not re.fullmatch(r"coli_cuda_[a-z0-9_]+", n)]
    if bad:
        raise FixtureBuildError("invalid generated export name(s): %s" % bad)
    if len(set(names)) != len(names):
        raise FixtureBuildError("duplicate symbol in the derived ABI")
    return mandatory, optional


# Imports every MinGW-linked DLL legitimately carries. Anything outside this
# set, other than the stub's deliberate amdhip64_7.dll, is a real dependency
# the fixture must not have acquired.
_ALLOWED_IMPORT_PREFIXES = ("kernel32", "api-ms-win-crt-", "msvcrt", "ucrtbase")
_FORBIDDEN_IMPORT_MARKERS = ("cudart", "nvcuda", "hiprtc", "rocblas", "rocwmma",
                             "amd_comgr", "hipblas")

_RUNTIME_MARKER_A = 0xA1
_RUNTIME_MARKER_B = 0xB2
_RUNTIME_BASENAME = "amdhip64_7.dll"
_TEST_ACCESSOR = "coli_test_bound_runtime"


class _StubFixture:
    """Two same-named runtime stubs plus one complete fake backend.

    Everything is generated and compiled inside a private temporary root whose
    path deliberately contains a space, so quoting mistakes surface here rather
    than on a user's "C:\\Program Files\\..." install. Nothing is ever written
    into c/.
    """

    def __init__(self):
        self.mandatory, self.optional = _derive_backend_abi()
        self.exports = self.mandatory + self.optional
        # The space is intentional; see the class docstring.
        self._tmp = tempfile.TemporaryDirectory(prefix="coli loader fix ")
        self.root = Path(self._tmp.name)
        self.runtime_a_dir = self.root / "runtime a"
        self.runtime_b_dir = self.root / "runtime-b"
        self.backend_dir = self.root / "backend"
        self.src_dir = self.root / "src"
        for d in (self.runtime_a_dir, self.runtime_b_dir,
                  self.backend_dir, self.src_dir):
            d.mkdir(parents=True)
        self.runtime_a = self.runtime_a_dir / _RUNTIME_BASENAME
        self.runtime_b = self.runtime_b_dir / _RUNTIME_BASENAME
        self.backend = self.backend_dir / "coli_hip.dll"
        self.runtime_a_src = self.src_dir / "runtime_a.c"
        self.runtime_b_src = self.src_dir / "runtime_b.c"
        self.backend_src = self.src_dir / "backend.c"
        self._build()

    # --- construction -------------------------------------------------

    def _gcc(self, args, what):
        cmd = ["gcc"] + args
        # errors="replace": the toolchain speaks the console codepage, which is
        # not always decodable as the Python default on localized hosts.
        proc = subprocess.run(cmd, text=True, errors="replace",
                              capture_output=True, timeout=300)
        if proc.returncode != 0:
            raise FixtureBuildError(
                "%s failed (rc=%d)\ncommand: %s\nstdout:\n%s\nstderr:\n%s"
                % (what, proc.returncode, " ".join(cmd), proc.stdout, proc.stderr))
        return proc

    def _build_runtime(self, source, out_dir, marker, name):
        # Generated sources stay strictly ASCII: they are fed to a toolchain
        # whose input encoding follows the console codepage.
        source.write_text(
            "/* generated stub runtime: no HIP/ROCm/CUDA header, no GPU work,\n"
            " * no DllMain, just a marker the fake backend can read back. */\n"
            "__declspec(dllexport) int coli_test_runtime_marker(void)\n"
            "{ return 0x%02X; }\n" % marker, encoding="ascii")
        implib = out_dir / "libamdhip64_7.a"
        self._gcc(["-O0", "-shared", str(source), "-o", str(out_dir / _RUNTIME_BASENAME),
                   "-Wl,--out-implib," + str(implib)], "building runtime stub " + name)
        return implib

    def _backend_source(self):
        real = {"coli_cuda_init", "coli_cuda_e8_set_grid"}
        lines = [
            "/* generated fake backend: no HIP/ROCm/CUDA header, no GPU work,",
            " * no DllMain. Imports the marker from amdhip64_7.dll by basename",
            " * so the Windows loader binds it exactly as the real DLL would. */",
            "__declspec(dllimport) int coli_test_runtime_marker(void);",
            "",
            "/* Test-only accessor. Not part of the production ABI; the loader",
            " * never resolves it. */",
            "__declspec(dllexport) int %s(void)" % _TEST_ACCESSOR,
            "{ return coli_test_runtime_marker(); }",
            "",
            "/* init returns a plain success signal, never the marker, because",
            " * colibri.c treats this return value as a control signal. */",
            "__declspec(dllexport) int coli_cuda_init(const int *devices, int count)",
            "{ (void)devices; (void)count; (void)coli_test_runtime_marker(); return 1; }",
            "",
            "__declspec(dllexport) int coli_cuda_e8_set_grid(const void *grid)",
            "{ (void)grid; return 1; }",
            "",
            "/* Remaining loader-required exports: never called on the startup",
            " * path under test, so trivial bodies are enough to let symbol",
            " * resolution complete. */",
        ]
        for name in self.exports:
            if name not in real:
                lines.append("__declspec(dllexport) int %s(void) { return 0; }" % name)
        return "\n".join(lines) + "\n"

    def _build(self):
        try:
            implib_a = self._build_runtime(self.runtime_a_src, self.runtime_a_dir,
                                           _RUNTIME_MARKER_A, "A")
            self._build_runtime(self.runtime_b_src, self.runtime_b_dir,
                                _RUNTIME_MARKER_B, "B")
            self.backend_src.write_text(self._backend_source(), encoding="ascii")
            self._gcc(["-O0", "-shared", str(self.backend_src),
                       "-o", str(self.backend),
                       "-L" + str(implib_a.parent), "-lamdhip64_7"],
                      "building fake coli_hip.dll")
        except Exception:
            self.cleanup()
            raise

    # --- inspection (objdump only; nothing is ever loaded) -------------

    @staticmethod
    def objdump(path, flag):
        proc = subprocess.run(["objdump", flag, str(path)], text=True,
                              errors="replace", capture_output=True, timeout=120)
        if proc.returncode != 0:
            raise FixtureBuildError("objdump %s failed on %s:\n%s"
                                    % (flag, path, proc.stderr))
        return proc.stdout

    @classmethod
    def architecture(cls, path):
        for line in cls.objdump(path, "-f").splitlines():
            if "architecture:" in line:
                return line.split("architecture:")[1].split(",")[0].strip()
        return ""

    @classmethod
    def imported_dlls(cls, path):
        return [m.strip().lower() for m in
                re.findall(r"DLL Name:\s*(\S+)", cls.objdump(path, "-p"))]

    @classmethod
    def imported_symbols(cls, path):
        return set(re.findall(r"\bcoli_[a-z0-9_]+", cls.objdump(path, "-p")))

    @classmethod
    def exported_names(cls, path):
        out = cls.objdump(path, "-p")
        block = out.split("Export Address Table")[-1]
        return set(re.findall(r"\b(coli_[a-z0-9_]+)\b", block))

    @staticmethod
    def sha256(path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()

    def generated_paths(self):
        return [self.runtime_a, self.runtime_b, self.backend,
                self.runtime_a_src, self.runtime_b_src, self.backend_src,
                self.runtime_a_dir / "libamdhip64_7.a",
                self.runtime_b_dir / "libamdhip64_7.a"]

    def cleanup(self):
        self._tmp.cleanup()


def _fixture_toolchain_skip():
    """Honest skip reason, or None when the fixture can be built here."""
    if sys.platform != "win32":
        return "stub fixtures target the Windows loader"
    for tool in ("gcc", "objdump"):
        if shutil.which(tool) is None:
            return "MinGW %s (same MSYS2 toolchain as the loader tests) not found" % tool
    return None


class LoaderStubFixtureTest(unittest.TestCase):
    """Structural proof that the stub fixtures are what later slices assume.

    These assertions never load a DLL: everything is read with objdump. A
    failure here means the fixture is broken, not that the production loader
    is — which is exactly why they live apart from the contract tests.
    """

    fixture = None

    @classmethod
    def setUpClass(cls):
        reason = _fixture_toolchain_skip()
        if reason:
            raise unittest.SkipTest(reason)
        # Compiled once per class; individual tests only inspect the result.
        cls.fixture = _StubFixture()

    @classmethod
    def tearDownClass(cls):
        if cls.fixture is not None:
            cls.fixture.cleanup()
            cls.fixture = None

    def test_abi_is_derived_from_the_loader_source(self):
        """46 mandatory + 2 optional, parsed from backend_loader.c."""
        f = self.fixture
        self.assertEqual(len(f.mandatory), 46)
        self.assertEqual(len(f.optional), 2)
        self.assertEqual(len(f.exports), 48)
        self.assertIn("coli_cuda_init", f.mandatory)
        self.assertIn("coli_cuda_e8_set_grid", f.optional)

    def test_both_runtimes_exist_with_the_production_basename(self):
        """Same basename, different directories — the conflict precondition."""
        f = self.fixture
        for path in (f.runtime_a, f.runtime_b):
            self.assertTrue(path.is_file(), path)
            self.assertEqual(path.name, _RUNTIME_BASENAME)
        self.assertNotEqual(f.runtime_a.parent, f.runtime_b.parent)

    def test_runtime_sources_and_binaries_differ(self):
        """Distinguishable by construction and on disk.

        Structural only: proving a *call* returns 0xA1 or 0xB2 needs a process
        that loads them, which is the next slice's harness.
        """
        f = self.fixture
        src_a = f.runtime_a_src.read_text(encoding="ascii")
        src_b = f.runtime_b_src.read_text(encoding="ascii")
        self.assertIn("0x%02X" % _RUNTIME_MARKER_A, src_a)
        self.assertIn("0x%02X" % _RUNTIME_MARKER_B, src_b)
        self.assertNotIn("0x%02X" % _RUNTIME_MARKER_B, src_a)
        self.assertNotEqual(f.sha256(f.runtime_a), f.sha256(f.runtime_b))

    def test_both_runtimes_export_the_marker(self):
        f = self.fixture
        for path in (f.runtime_a, f.runtime_b):
            self.assertIn("coli_test_runtime_marker", f.exported_names(path))

    def test_backend_imports_the_runtime_marker_by_basename(self):
        """The stub must bind amdhip64_7.dll the way the real backend does."""
        f = self.fixture
        self.assertIn(_RUNTIME_BASENAME.lower(), f.imported_dlls(f.backend))
        self.assertIn("coli_test_runtime_marker", f.imported_symbols(f.backend))

    def test_backend_exports_the_complete_derived_abi(self):
        """Every loader-required symbol, plus exactly one test-only accessor."""
        f = self.fixture
        exported = f.exported_names(f.backend)
        missing = sorted(set(f.exports) - exported)
        self.assertEqual(missing, [], "missing production exports: %s" % missing)
        self.assertIn(_TEST_ACCESSOR, exported)
        extra = sorted(exported - set(f.exports) - {_TEST_ACCESSOR})
        self.assertEqual(extra, [], "unexpected extra exports: %s" % extra)

    def test_exported_names_are_undecorated(self):
        f = self.fixture
        for name in f.exported_names(f.backend):
            self.assertNotIn("@", name)
            self.assertFalse(name.startswith("_"), name)

    def test_all_artifacts_are_pe_x86_64(self):
        f = self.fixture
        for path in (f.runtime_a, f.runtime_b, f.backend):
            self.assertEqual(f.architecture(path), "i386:x86-64", path)

    def test_no_real_gpu_runtime_is_imported(self):
        """The deliberate amdhip64_7.dll stub is allowed; real GPU stacks are not."""
        f = self.fixture
        for path in (f.runtime_a, f.runtime_b, f.backend):
            for dll in f.imported_dlls(path):
                if dll == _RUNTIME_BASENAME.lower():
                    continue  # intentional: the stub uses the production basename
                self.assertFalse(
                    any(bad in dll for bad in _FORBIDDEN_IMPORT_MARKERS),
                    "%s imports a real GPU component: %s" % (path.name, dll))
                self.assertTrue(
                    dll.startswith(_ALLOWED_IMPORT_PREFIXES),
                    "%s has an unexpected import: %s" % (path.name, dll))

    def test_fixture_root_contains_a_space_and_avoids_the_repo(self):
        """Quoting bugs must surface here, not on a user's Program Files install."""
        f = self.fixture
        self.assertIn(" ", str(f.root))
        self.assertIn(" ", str(f.runtime_a))
        repo_c = str(HERE.resolve()).lower()
        for path in f.generated_paths():
            self.assertFalse(str(path.resolve()).lower().startswith(repo_c),
                             "generated artifact inside the repo: %s" % path)


class LoaderStubFixtureCleanupTest(unittest.TestCase):
    """Explicit cleanup, not interpreter shutdown, removes every artifact."""

    def test_cleanup_removes_every_generated_artifact(self):
        reason = _fixture_toolchain_skip()
        if reason:
            self.skipTest(reason)
        fixture = _StubFixture()
        paths = fixture.generated_paths()
        root = fixture.root
        for path in paths:
            self.assertTrue(path.exists(), "not built: %s" % path)
        fixture.cleanup()
        self.assertFalse(root.exists(), "temporary root survived: %s" % root)
        for path in paths:
            self.assertFalse(path.exists(), "artifact survived cleanup: %s" % path)
        # The repository must be untouched by a fixture that never targets it.
        self.assertFalse((HERE / "coli_hip.dll").exists())
        self.assertFalse((HERE / _RUNTIME_BASENAME).exists())


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
