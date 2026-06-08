"""Unit tests for the Tracer recorder."""
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from modules.proof_server import flops as F
from modules.proof_server.graph import build_graph
from modules.proof_server.tracer import Tracer


class TestTracer(unittest.TestCase):
    def test_event_ids_increment_from_zero(self):
        tr = Tracer()
        self.assertEqual(tr.event("prefill"), 0)
        self.assertEqual(tr.event("decode"), 1)
        self.assertEqual(tr.event("decode"), 2)

    def test_trace_round_trips_into_build_graph(self):
        tr = Tracer()
        tr.add_shape("m", F.KNOWN_SHAPES["Qwen/Qwen3-1.7B"])
        a = tr.event("prefill", model="m", tokens=3, attended=6, logits=1)
        b = tr.event("decode", model="m", tokens=1, attended=4, logits=1, inputs=[a])
        tr.event("decode", model="m", tokens=1, attended=5, logits=1, inputs=[b])
        g = build_graph(tr.trace())
        self.assertEqual(len(g.nodes), 3)
        self.assertEqual(len(g.edges), 2)

    def test_shapes_are_carried_through(self):
        tr = Tracer()
        tr.add_shape("m", {"hello": 1})
        self.assertEqual(tr.trace()["shapes"], {"m": {"hello": 1}})

    def test_no_flops_computed_in_tracer(self):
        # The tracer must not compute cost; flops appears only after build_graph.
        tr = Tracer()
        tr.add_shape("m", F.KNOWN_SHAPES["Qwen/Qwen3-1.7B"])
        tr.event("decode", model="m", tokens=1)
        self.assertNotIn("flops", tr.trace()["events"][0])


if __name__ == "__main__":
    unittest.main()
