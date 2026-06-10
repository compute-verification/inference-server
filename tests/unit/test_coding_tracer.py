"""Unit tests for the coding-agent tracer (real-capture converter).

The GPU harness (demos/proof-compare/capture/run_coding_agent.py) records one
entry per real LLM call: prompt/generated token counts and which earlier calls
fed its prompt. ``trace_coding_real`` converts that capture into the canonical
trace. Two defining properties: (1) every node is exactly one forward pass — a
prefill or a single-token decode — like the inference tracer; (2) the agent
issues independent LLM calls in parallel, so the trace is a DAG that fans out
and fans in. Tool calls are not forward passes, so they are not nodes; their
output shows up as the next call's prompt (tagged ``payload.via``).
"""
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
import coding

PROMPT = "Read the paper in this workspace, then implement it"

def _capture(calls=None):
    """A handcrafted capture: orient -> (read A || read B) -> (plan a || plan b) -> synth."""
    return {
        "kind": "coding_agent_capture",
        "model": "Qwen/Qwen3-1.7B",
        "config": dict(F.KNOWN_SHAPES["Qwen/Qwen3-1.7B"]),
        "prompt": PROMPT,
        "tests_passed": True,
        "calls": calls if calls is not None else [
            {"id": 0, "phase": "orient", "role": "reason", "parents": [],
             "prompt_tokens": 40, "gen_tokens": 4, "text": "approach notes"},
            {"id": 1, "phase": "read paper.md", "role": "read", "via": "read file",
             "parents": [0], "prompt_tokens": 600, "gen_tokens": 3, "text": "s1"},
            {"id": 2, "phase": "read reference.py", "role": "read", "via": "read file",
             "parents": [0], "prompt_tokens": 800, "gen_tokens": 2, "text": "s2"},
            {"id": 3, "phase": "plan: correctness", "role": "plan",
             "parents": [1, 2], "prompt_tokens": 900, "gen_tokens": 5, "text": "p1"},
            {"id": 4, "phase": "plan: simplicity", "role": "plan",
             "parents": [1, 2], "prompt_tokens": 905, "gen_tokens": 4, "text": "p2"},
            {"id": 5, "phase": "synthesize plan", "role": "plan",
             "parents": [3, 4], "prompt_tokens": 700, "gen_tokens": 6, "text": "final"},
        ],
    }


class TestCodingTracer(unittest.TestCase):
    def setUp(self):
        self.cap = _capture()
        self.trace = coding.trace_coding_real(self.cap)
        self.evs = self.trace["events"]
        self.byid = {e["id"]: e for e in self.evs}

    def test_every_node_is_a_forward_pass(self):
        self.assertTrue(all(e["kind"] in ("prefill", "decode") for e in self.evs))

    def test_node_count_is_one_prefill_plus_gen_per_call(self):
        want = sum(1 + c["gen_tokens"] for c in self.cap["calls"])
        self.assertEqual(len(self.evs), want)

    def test_prefill_attends_its_own_causal_triangle(self):
        # Each call is an independent context (a freshly assembled prompt), so
        # its prefill attends p*(p+1)/2 — no shared-KV assumption.
        paper = next(e for e in self.evs
                     if e["kind"] == "prefill" and e["payload"]["phase"] == "read paper.md")
        self.assertEqual(paper["tokens"], 600)
        self.assertEqual(paper["attended"], 600 * 601 // 2)

    def test_decodes_emit_one_token_attending_growing_context(self):
        decodes = [e for e in self.evs if e["kind"] == "decode"
                   and e["payload"]["phase"] == "read paper.md"]
        self.assertEqual([d["tokens"] for d in decodes], [1, 1, 1])
        self.assertEqual([d["attended"] for d in decodes], [601, 602, 603])

    def test_is_a_dag_with_inputs_before_id(self):
        for e in self.evs:
            for src in e["inputs"]:
                self.assertLess(src, e["id"])

    def test_has_parallel_fan_out_and_fan_in(self):
        outdeg = {e["id"]: 0 for e in self.evs}
        indeg = {e["id"]: 0 for e in self.evs}
        for e in self.evs:
            for src in e["inputs"]:
                outdeg[src] += 1
                indeg[e["id"]] += 1
        self.assertTrue(any(v >= 2 for v in outdeg.values()), "expected a fan-out")
        self.assertTrue(any(v >= 2 for v in indeg.values()), "expected a fan-in")

    def test_plan_call_fans_in_from_both_reads(self):
        plan = next(e for e in self.evs
                    if e["kind"] == "prefill" and e["payload"]["phase"] == "plan: correctness")
        self.assertEqual(len(plan["inputs"]), 2)

    def test_single_root_is_the_task_prefill_with_prompt(self):
        roots = [e for e in self.evs if not e["inputs"]]
        self.assertEqual(len(roots), 1)
        self.assertEqual(roots[0]["kind"], "prefill")
        self.assertEqual(roots[0]["payload"]["prompt"], PROMPT)

    def test_via_lands_on_the_prefill_payload(self):
        paper = next(e for e in self.evs
                     if e["kind"] == "prefill" and e["payload"]["phase"] == "read paper.md")
        self.assertEqual(paper["payload"]["via"], "read file")

    def test_generated_text_preview_on_last_decode(self):
        tail = max((e for e in self.evs if e["payload"]["phase"] == "orient"),
                   key=lambda e: e["id"])
        self.assertEqual(tail["kind"], "decode")
        self.assertEqual(tail["payload"]["out"], "approach notes")

    def test_builds_into_valid_graph_all_positive(self):
        g = build_graph(self.trace)
        self.assertTrue(all(n["flops"] > 0 for n in g.nodes))

    def test_uses_real_config_not_known_shapes(self):
        self.assertIn("Qwen/Qwen3-1.7B", self.trace["shapes"])
        self.assertEqual(self.trace["shapes"]["Qwen/Qwen3-1.7B"]["hidden_size"], 2048)

    def test_empty_capture_rejected(self):
        with self.assertRaises(ValueError):
            coding.trace_coding_real(_capture(calls=[]))

    def test_forward_parent_reference_rejected(self):
        bad = _capture(calls=[
            {"id": 0, "phase": "a", "role": "reason", "parents": [1],
             "prompt_tokens": 10, "gen_tokens": 1, "text": ""},
            {"id": 1, "phase": "b", "role": "reason", "parents": [],
             "prompt_tokens": 10, "gen_tokens": 1, "text": ""},
        ])
        with self.assertRaises(ValueError):
            coding.trace_coding_real(bad)

    def test_promptless_call_rejected(self):
        bad = _capture(calls=[
            {"id": 0, "phase": "a", "role": "reason", "parents": [],
             "prompt_tokens": 0, "gen_tokens": 5, "text": ""},
        ])
        with self.assertRaises(ValueError):
            coding.trace_coding_real(bad)


if __name__ == "__main__":
    unittest.main()
