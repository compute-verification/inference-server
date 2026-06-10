"""Smoke test for the coding-agent capture harness (mock mode).

Runs the actual scaffold on CPU with the deterministic stand-in model: the
canned codegen output is REAL working code, so the harness's tool path
(extract fenced blocks -> write files -> `python -m unittest discover`)
executes genuinely and must go green. The resulting capture must convert into
a valid canonical graph via trace_coding_real.
"""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
TRACERS = REPO_ROOT / "demos" / "proof-compare" / "tracers"
HARNESS = REPO_ROOT / "demos" / "proof-compare" / "capture" / "run_coding_agent.py"
for p in (REPO_ROOT, TRACERS):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from modules.proof_server.graph import build_graph
import coding


class TestCodingAgentHarnessMock(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        out = Path(tempfile.mkdtemp()) / "capture.json"
        proc = subprocess.run(
            [sys.executable, str(HARNESS), "--mock", "--out", str(out)],
            capture_output=True, text=True, timeout=120)
        assert proc.returncode == 0, proc.stdout + proc.stderr
        cls.stdout = proc.stdout
        cls.cap = json.loads(out.read_text())

    def test_generated_tests_really_ran_and_passed(self):
        # the mock's canned code is correct -> the real unittest subprocess
        # must have gone green with no fix rounds.
        self.assertTrue(self.cap["tests_passed"])
        self.assertEqual(self.cap["fix_rounds"], 0)
        self.assertIn("OK", self.cap["test_output_tail"])

    def test_scaffold_shape(self):
        calls = self.cap["calls"]
        # orient + 4 reads + 3 plans + synth + 2 codegen + verdict = 12 calls
        self.assertEqual(len(calls), 12)
        reads = [c for c in calls if c["role"] == "read"]
        plans = [c for c in calls if c["role"] == "plan" and "candidate" not in c["phase"]]
        self.assertEqual(len(reads), 4)
        # every plan candidate fans in from ALL four reads
        for c in calls:
            if c["phase"].startswith("plan: "):
                self.assertEqual(c["parents"], [r["id"] for r in reads])
        # codegen runs in parallel: both fork from synthesize only
        synth = next(c for c in calls if c["phase"] == "synthesize plan")
        for c in calls:
            if c["role"] == "codegen":
                self.assertEqual(c["parents"], [synth["id"]])

    def test_capture_converts_to_valid_graph(self):
        g = build_graph(coding.trace_coding_real(self.cap))
        self.assertTrue(all(n["flops"] > 0 for n in g.nodes))
        kinds = {n["kind"] for n in g.nodes}
        self.assertEqual(kinds, {"prefill", "decode"})


if __name__ == "__main__":
    unittest.main()
