"""End-to-end proof that KV prefix reuse changes nothing but the time.

Two turns on one engine (the second reusing the first's state) must produce the
SAME tokens as the second turn alone on a cold engine. That is the whole licence
to ship the optimisation: if reuse alters even one token it is answering from a
state that belongs to a different conversation, and the reply would still look
plausible — nothing else in the tree would catch it.

Runs against the tiny random-init fixture the oracle job already builds
(tools/make_tiny_inkling.py), so it needs no checkpoint and no fast disk. The
real 469 GB model cannot serve this test on a developer machine: measured here,
a single token pulled 79 GB off disk in 9 minutes and had not finished. That is
why this gate lives in CI on a fixture rather than in a benchmark on hardware.
"""
import os
import subprocess
import sys
import time
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
ENGINE = HERE / ("inkling.exe" if sys.platform == "win32" else "inkling")
FIXTURE = Path(os.environ.get("INKLING_TINY", HERE / "tiny_inkling"))
MAXTOK = 4
READY = b"\x01\x01READY\x01\x01\n"


class Engine:
    """A serve-mode inkling, speaking the protocol in docs/serve_protocol.md."""

    def __init__(self, log_prefix=True):
        env = dict(os.environ, SNAP=str(FIXTURE), SERVE="1", NGEN=str(MAXTOK))
        if log_prefix:
            env["INK_PREFIX_LOG"] = "1"
        self.p = subprocess.Popen([str(ENGINE), "8"], env=env,
                                  stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE, bufsize=0)
        deadline = time.time() + 300
        while time.time() < deadline:
            line = self.p.stdout.readline()
            if not line:
                err = self.p.stderr.read().decode(errors="replace")
                raise RuntimeError(f"engine exited before READY:\n{err[-2000:]}")
            if READY.strip() in line:
                return
        raise RuntimeError("engine never reported READY")

    def ask(self, rid, prompt):
        payload = prompt.encode()
        header = f"SUBMIT {rid} 0 {len(payload)} {MAXTOK} 0 1\n".encode()
        self.p.stdin.write(header + payload + b"\n")
        self.p.stdin.flush()
        chunks = []
        while True:
            line = self.p.stdout.readline()
            if not line:
                raise RuntimeError("engine closed mid-request")
            text = line.decode(errors="replace").rstrip("\n")
            kind = text.split(" ", 1)[0]
            if kind == "DATA":
                chunks.append(self.p.stdout.read(int(text.split()[2]))
                              .decode(errors="replace"))
                self.p.stdout.readline()          # the newline after the payload
            elif kind in ("DONE", "END"):
                return "".join(chunks)
            elif kind == "ERROR":
                raise RuntimeError(f"engine error: {text}")

    def close(self):
        try:
            self.p.stdin.close()
            self.p.wait(timeout=60)
        except Exception:
            self.p.kill()
        return self.p.stderr.read().decode(errors="replace")


@unittest.skipUnless(ENGINE.exists(), "inkling is not built")
@unittest.skipUnless((FIXTURE / "config.json").exists(),
                     "tiny inkling fixture is absent (tools/make_tiny_inkling.py)")
class InklingPrefixServeTest(unittest.TestCase):
    def test_reused_prefix_yields_identical_tokens(self):
        warm = Engine()
        first = warm.ask("1", "The capital of France is")
        # Turn 2 EXTENDS what the state already holds — the shape a chat client
        # produces when it resends the transcript with a new question appended.
        second_prompt = "The capital of France is" + first + " and the capital of Spain is"
        reused = warm.ask("2", second_prompt)
        log = warm.close()

        cold = Engine(log_prefix=False)
        fresh = cold.ask("1", second_prompt)
        cold.close()

        self.assertIn("[PREFIX] reusing", log,
                      "the second turn did not reuse the first turn's state; "
                      "this test proves nothing unless it does")
        self.assertEqual(reused, fresh,
                         "reusing the prefix changed the output — the engine "
                         "answered from a state that is not this conversation")

    def test_a_diverging_prompt_is_not_reused(self):
        """The rejection path matters as much as the reuse path: a prompt that
        shares no prefix must start over, not splice itself onto stale state."""
        eng = Engine()
        eng.ask("1", "The capital of France is")
        diverged = eng.ask("2", "Completely different opening text here")
        log = eng.close()

        cold = Engine(log_prefix=False)
        fresh = cold.ask("1", "Completely different opening text here")
        cold.close()

        self.assertEqual(diverged, fresh,
                         "a diverging prompt was contaminated by the previous turn")


if __name__ == "__main__":
    unittest.main()
