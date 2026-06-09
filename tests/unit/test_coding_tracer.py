"""Unit tests for the coding-agent tracer (stub).

Two defining properties: (1) every node is exactly one forward pass — a prefill
or a single-token decode — like the inference tracer; (2) the agent runs LLM
calls in parallel, so the trace is a DAG that fans out and fans in (not a single
chain). Tool calls are not forward passes, so they are not nodes; their output
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
# reason -> (read A || read B) -> plan -> (code A || code B) -> test
STAGES = [
    {"role": "reason", "prefill": 40, "gen": 4, "label": "read prompt"},
    {"parallel": [
        {"role": "read", "prefill": 600, "gen": 3, "via": "fetch", "label": "read paper"},
        {"role": "read", "prefill": 800, "gen": 2, "via": "fetch", "label": "read repo"},
    ]},
    {"role": "plan", "prefill": 50, "gen": 5, "via": "merge", "label": "plan"},
    {"parallel": [
        {"role": "codegen", "prefill": 0, "gen": 6, "label": "write src"},
        {"role": "codegen", "prefill": 0, "gen": 7, "label": "write test"},
    ]},
    {"role": "test", "prefill": 400, "gen": 2, "via": "run tests", "label": "tests"},
]


def _flatten(stages):
    for s in stages:
        if "parallel" in s:
            yield from s["parallel"]
        else:
            yield s


def _trace():
    return coding.trace_coding_stub("agent", PROMPT, STAGES)


class TestCodingTracer(unittest.TestCase):
    def setUp(self):
        self.trace = _trace()
        self.evs = self.trace["events"]
        self.byid = {e["id"]: e for e in self.evs}

    def test_every_node_is_a_forward_pass(self):
        # The core invariant: nothing but prefill/decode (no agent-action nodes).
        self.assertTrue(all(e["kind"] in ("prefill", "decode") for e in self.evs))

    def test_node_count_is_prefills_plus_generated_tokens(self):
        turns = list(_flatten(STAGES))
        prefills = sum(1 for t in turns if t["prefill"] > 0)
        decodes = sum(t["gen"] for t in turns)
        self.assertEqual(len(self.evs), prefills + decodes)

    def test_decodes_emit_exactly_one_token(self):
        self.assertTrue(all(e["tokens"] == 1 for e in self.evs if e["kind"] == "decode"))

    def test_is_a_dag_with_inputs_before_id(self):
        for e in self.evs:
            for src in e["inputs"]:
                self.assertLess(src, e["id"])  # acyclic, no forward refs

    def test_has_parallel_fan_out_and_fan_in(self):
        outdeg = {e["id"]: 0 for e in self.evs}
        indeg = {e["id"]: 0 for e in self.evs}
        for e in self.evs:
            for src in e["inputs"]:
                outdeg[src] += 1
                indeg[e["id"]] += 1
        # at least one node fans out to parallel branches, and at least one
        # merge node fans in from multiple branches.
        self.assertTrue(any(v >= 2 for v in outdeg.values()), "expected a fan-out")
        self.assertTrue(any(v >= 2 for v in indeg.values()), "expected a fan-in")

    def test_single_root_is_the_prompt_prefill(self):
        roots = [e for e in self.evs if not e["inputs"]]
        self.assertEqual(len(roots), 1)
        self.assertEqual(roots[0]["kind"], "prefill")
        self.assertEqual(roots[0]["payload"]["prompt"], PROMPT)

    def test_merge_turn_fans_in_from_both_branches(self):
        plan = next(e for e in self.evs
                    if e["kind"] == "prefill" and e["payload"].get("phase") == "plan")
        self.assertEqual(len(plan["inputs"]), 2)  # fed by both parallel reads

    def test_tool_output_becomes_a_fat_prefill(self):
        paper = next(e for e in self.evs
                     if e["kind"] == "prefill" and e["payload"].get("phase") == "read paper")
        self.assertEqual(paper["tokens"], 600)
        self.assertGreater(paper["tokens"], 1)

    def test_builds_into_valid_graph_all_positive(self):
        g = build_graph(self.trace)
        self.assertTrue(all(n["flops"] > 0 for n in g.nodes))

    def test_empty_stages_rejected(self):
        with self.assertRaises(ValueError):
            coding.trace_coding_stub("agent", PROMPT, [])


if __name__ == "__main__":
    unittest.main()
