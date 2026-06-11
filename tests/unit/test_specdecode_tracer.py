"""Unit tests for the spec-decode tracer."""
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
TRACERS = REPO_ROOT / "demos" / "proof-compare" / "tracers"
for p in (REPO_ROOT, TRACERS):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from modules.proof_server.graph import build_graph
import specdecode

ROUNDS = [
    {"drafts": ["a", "b", "X", "d"], "num_accepted": 2, "correction": "c"},
    {"drafts": ["d", "e", "Y", "g"], "num_accepted": 2, "correction": "f"},
    {"drafts": ["g", "h", "i", "j"], "num_accepted": 4, "correction": "k"},
]
PROMPT_LEN = 3


class TestSpecDecodeTracer(unittest.TestCase):
    def setUp(self):
        self.trace = specdecode.trace_spec_decode(PROMPT_LEN, ROUNDS)
        self.evs = self.trace["events"]
        self.byid = {e["id"]: e for e in self.evs}

    def _verifies(self):
        return [e for e in self.evs if e["kind"] == "verify"]

    def test_every_draft_fans_into_its_round_verify(self):
        for v in self._verifies():
            self.assertEqual(len(v["inputs"]), 4)  # all k drafts feed verify
            for did in v["inputs"]:
                self.assertEqual(self.byid[did]["kind"], "draft")

    def test_accepted_and_rejected_counts(self):
        drafts = [e for e in self.evs if e["kind"] == "draft"]
        acc = [e for e in drafts if e["status"] == "accepted"]
        rej = [e for e in drafts if e["status"] == "rejected"]
        self.assertEqual(len(acc), 8)   # 2 + 2 + 4
        self.assertEqual(len(rej), 4)   # 2 + 2 + 0

    def _first_draft_of_each_round(self):
        # A round's first draft either has no inputs (round 0) or chains off a
        # verify (the prior round's handoff).
        out = []
        for e in self.evs:
            if e["kind"] != "draft":
                continue
            if not e["inputs"] or self.byid[e["inputs"][0]]["kind"] == "verify":
                out.append(e)
        return out

    def test_rounds_are_handed_off(self):
        verify0 = self._verifies()[0]["id"]
        after_v0 = self.byid[verify0 + 1]               # next event after round-0 verify
        self.assertEqual(after_v0["kind"], "draft")
        self.assertEqual(after_v0["inputs"], [verify0])  # round 1's first draft depends on it

    def test_context_grows_across_rounds(self):
        attendeds = [e["attended"] for e in self._first_draft_of_each_round()]
        # round 0 first draft attends to prompt_len=3; round 2 first draft to
        # 3 + (2+1) + (2+1) = 9.
        self.assertEqual(attendeds[0], 3)
        self.assertEqual(attendeds[-1], 9)

    def test_builds_into_valid_graph_verify_fatter_than_draft(self):
        g = build_graph(self.trace)
        verify = next(n for n in g.nodes if n["kind"] == "verify")
        draft = next(n for n in g.nodes if n["kind"] == "draft")
        self.assertGreater(verify["flops"], draft["flops"])

    def test_round_zero_first_draft_has_no_inputs(self):
        self.assertEqual(self.evs[0]["kind"], "draft")
        self.assertEqual(self.evs[0]["inputs"], [])

    def test_all_rejected_round_advances_ctx_by_one(self):
        # a=0 round: ctx must still advance by num_accepted+1 = 1.
        trace = specdecode.trace_spec_decode(5, [
            {"drafts": ["X", "Y"], "num_accepted": 0, "correction": "z"},
            {"drafts": ["a", "b"], "num_accepted": 1, "correction": "c"},
        ])
        evs = trace["events"]
        # round 1's first draft (chains off the round-0 verify) attends to ctx=6.
        r1_first = next(e for e in evs if e["kind"] == "draft"
                        and e["inputs"] and {evs[i]["id"]: evs[i] for i in range(len(evs))}[e["inputs"][0]]["kind"] == "verify")
        self.assertEqual(r1_first["attended"], 6)  # 5 + (0+1)
        build_graph(trace)  # also valid


if __name__ == "__main__":
    unittest.main()
