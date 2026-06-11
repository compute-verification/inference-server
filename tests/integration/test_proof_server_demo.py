"""End-to-end smoke test for ``demos/proof-server/demo.sh --quick``.

This test is slow (SP1 SDK setup is ~1-2 min per process invocation in
execute mode, even with the precompile patches), so it's parked under
``tests/integration/`` and skipped automatically when the SP1 host binary
isn't built.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
HOST_BIN = REPO_ROOT / "modules/proof_server/sp1/target/release/proof-server-host"
DEMO_SH = REPO_ROOT / "demos/proof-server/demo.sh"


def _have_sp1() -> bool:
    return HOST_BIN.exists() and shutil.which("cargo-prove") is not None


@unittest.skipUnless(_have_sp1(),
                     "SP1 host binary missing; install sp1up + protoc and rebuild")
class TestProofServerDemo(unittest.TestCase):
    def test_quick_demo_all_pass(self):
        # demo.sh is intentionally slow on dev boxes (SP1 SDK setup
        # dominates); allow a generous timeout.
        env = dict(os.environ)
        env.setdefault("PATH", os.environ.get("PATH", ""))
        # Make sure cargo-prove is reachable for any rebuild paths.
        home_sp1_bin = Path.home() / ".sp1/bin"
        if home_sp1_bin.exists():
            env["PATH"] = f"{home_sp1_bin}:{env['PATH']}"
        result = subprocess.run(
            ["bash", str(DEMO_SH), "--quick"],
            capture_output=True,
            timeout=600,
            env=env,
        )
        out = result.stdout.decode("utf-8", errors="replace")
        err = result.stderr.decode("utf-8", errors="replace")
        if result.returncode != 0 or "ALL PASS" not in out:
            self.fail(
                f"demo.sh exited {result.returncode}\n"
                f"--- stdout tail ---\n{out[-2000:]}\n"
                f"--- stderr tail ---\n{err[-2000:]}\n"
            )


if __name__ == "__main__":
    unittest.main()
