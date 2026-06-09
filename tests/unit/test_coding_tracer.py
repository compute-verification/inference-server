"""Unit tests for the coding-agent tracer (stub).

The defining property (and the whole point of this tracer): every node is one
forward pass — a prefill or a single-token decode — exactly like the inference
tracer. Tool calls are not forward passes, so they are not nodes; their output
shows up as the next prefill's token count.
"""
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
TRACERS = REPO_ROOT / "demos" / "proof-compare" / "tracers"
for p in (REPO_ROOT, TRACERS):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from modules.proof_server.graph import build_graph
import coding

PROMPT = "Summarize a paper that just came out, then implement it"
TURNS = [
    {"role": "reason",  "prefill": 40,   "gen": 4, "label": "read prompt"},
    {"role": "triage",  "prefill": 600,  "gen": 3, "via": "search", "label": "triage"},
    {"role": "plan",    "prefill": 6300, "gen": 5, "via": "fetch", "label": "plan"},
    {"role": "codegen", "prefill": 0,    "gen": 6, "label": "write code"},  # continuation
    {"role": "test",    "prefill": 400,  "gen": 2, "via": "run tests", "label": "tests"},
]


def _trace():
    return coding.trace_coding_stub("agent", PROMPT, TURNS)


class TestCodingTracer(unittest.TestCase):
    def setUp(self):
        self.trace = _trace()
        self.evs = self.trace["events"]
        self.byid = {e["id"]: e for e in self.evs}

    def test_every_node_is_a_forward_pass(self):
        # The core invariant: nothing but prefill/decode (no agent-action nodes).
        self.assertTrue(all(e["kind"] in ("prefill", "decode") for e in self.evs))

    def test_node_count_is_prefills_plus_generated_tokens(self):
        prefills = sum(1 for t in TURNS if t["prefill"] > 0)
        decodes = sum(t["gen"] for t in TURNS)
        self.assertEqual(len(self.evs), prefills + decodes)

    def test_decodes_emit_exactly_one_token(self):
        self.assertTrue(all(e["tokens"] == 1 for e in self.evs if e["kind"] == "decode"))

    def test_continuation_turn_emits_no_prefill(self):
        # the prefill==0 codegen turn adds decodes only.
        prefills = sum(1 for e in self.evs if e["kind"] == "prefill")
        self.assertEqual(prefills, sum(1 for t in TURNS if t["prefill"] > 0))

    def test_run_is_a_single_chain(self):
        roots = [e for e in self.evs if not e["inputs"]]
        self.assertEqual(len(roots), 1)
        self.assertEqual(roots[0]["id"], 0)
        # strictly sequential ids => each non-root chains off its predecessor.
        for e in self.evs[1:]:
            self.assertEqual(e["inputs"], [e["id"] - 1])

    def test_first_prefill_carries_the_prompt(self):
        first = self.evs[0]
        self.assertEqual(first["kind"], "prefill")
        self.assertEqual(first["payload"]["prompt"], PROMPT)

    def test_tool_output_becomes_a_fat_prefill(self):
        # the fetch turn re-prefills 6300 injected tokens (a tool result), tagged.
        fetch = next(e for e in self.evs
                     if e["kind"] == "prefill" and e["payload"].get("via") == "fetch")
        self.assertEqual(fetch["tokens"], 6300)
        self.assertGreater(fetch["tokens"], 1)

    def test_attention_grows_with_context(self):
        decodes = [e for e in self.evs if e["kind"] == "decode"]
        self.assertTrue(all(e["attended"] > 0 for e in decodes))
        # later tokens attend over a longer context than earlier ones.
        self.assertGreater(decodes[-1]["attended"], decodes[0]["attended"])

    def test_builds_into_valid_graph_all_positive(self):
        g = build_graph(self.trace)
        self.assertTrue(all(n["flops"] > 0 for n in g.nodes))

    def test_empty_turns_rejected(self):
        with self.assertRaises(ValueError):
            coding.trace_coding_stub("agent", PROMPT, [])


if __name__ == "__main__":
    unittest.main()
