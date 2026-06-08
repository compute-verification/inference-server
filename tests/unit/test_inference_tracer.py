"""Unit tests for the inference tracer."""
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
TRACERS = REPO_ROOT / "demos" / "proof-compare" / "tracers"
for p in (REPO_ROOT, TRACERS):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from modules.proof_server import flops as F
from modules.proof_server.graph import build_graph
import inference


def _trace(max_tokens=4, prompt_ids=(1, 2, 3)):
    return inference.trace_inference(
        list(prompt_ids), inference.mock_next_token, "Qwen/Qwen3-1.7B",
        F.KNOWN_SHAPES["Qwen/Qwen3-1.7B"], max_tokens)


class TestInferenceTracer(unittest.TestCase):
    def test_one_prefill_then_n_decodes(self):
        evs = _trace(max_tokens=4)["events"]
        kinds = [e["kind"] for e in evs]
        self.assertEqual(kinds, ["prefill", "decode", "decode", "decode", "decode"])

    def test_decode_chain_is_linked(self):
        evs = _trace(max_tokens=3)["events"]
        for i in range(1, len(evs)):
            self.assertEqual(evs[i]["inputs"], [i - 1])
        self.assertEqual(evs[0]["inputs"], [])

    def test_prefill_attended_is_causal_triangle(self):
        evs = _trace(prompt_ids=(1, 2, 3, 4, 5))["events"]  # P = 5
        self.assertEqual(evs[0]["attended"], 5 * 6 // 2)

    def test_decode_attended_grows_each_step(self):
        evs = _trace(max_tokens=3, prompt_ids=(1, 2, 3))["events"]
        decodes = [e for e in evs if e["kind"] == "decode"]
        self.assertEqual([e["attended"] for e in decodes], [4, 5, 6])

    def test_builds_into_a_valid_graph_with_fat_prefill(self):
        g = build_graph(_trace(max_tokens=4))
        flops = [n["flops"] for n in g.nodes]
        self.assertEqual(flops[0], max(flops))  # prefill is the fattest

    def test_zero_max_tokens_is_just_prefill(self):
        evs = _trace(max_tokens=0)["events"]
        self.assertEqual([e["kind"] for e in evs], ["prefill"])


if __name__ == "__main__":
    unittest.main()
