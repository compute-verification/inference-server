"""Unit tests for modules.proof_server.task_graph."""
import json
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
import sys
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from modules.proof_server.task_graph import (
    DEFAULT_DIMS,
    MODEL_DIMS,
    build_task_graph,
    dims_for,
    forward_flops,
)


class TestForwardFlops(unittest.TestCase):
    def test_prefill_is_fatter_than_decode(self):
        dims = MODEL_DIMS["hf://Qwen/Qwen3-1.7B"]
        prefill = forward_flops(dims, tokens_in_pass=500, context_len=500)
        decode = forward_flops(dims, tokens_in_pass=1, context_len=500)
        # Prefill processes 500 tokens; a decode step processes 1 -> ~hundreds x.
        self.assertGreater(prefill, decode * 100)

    def test_decode_grows_with_context(self):
        dims = DEFAULT_DIMS
        early = forward_flops(dims, tokens_in_pass=1, context_len=10)
        late = forward_flops(dims, tokens_in_pass=1, context_len=1000)
        self.assertGreater(late, early)

    def test_weight_term_dominates_at_short_context(self):
        dims = MODEL_DIMS["hf://Qwen/Qwen3-1.7B"]
        f = forward_flops(dims, tokens_in_pass=1, context_len=50)
        weight_only = 2 * dims.n_params
        # Attention term is small relative to weights at short context.
        self.assertLess(f - weight_only, weight_only)


class TestDimsFor(unittest.TestCase):
    def test_known_model(self):
        self.assertEqual(dims_for("hf://Qwen/Qwen3-1.7B").n_layers, 28)

    def test_unknown_model_falls_back(self):
        self.assertEqual(dims_for("hf://nope/unknown"), DEFAULT_DIMS)


class TestBuildTaskGraph(unittest.TestCase):
    def setUp(self):
        self.graph = build_task_graph(
            request_id=7,
            prompt="hello there world",
            output="one two three four",
            model_source="hf://Qwen/Qwen3-1.7B",
        )

    def test_one_prefill_then_decode_chain(self):
        kinds = [t.kind for t in self.graph.tasks]
        self.assertEqual(kinds[0], "prefill")
        self.assertTrue(all(k == "decode" for k in kinds[1:]))

    def test_task_count_matches_output_tokens(self):
        # 4 whitespace chunks of output -> 4 forward passes (prefill + 3 decode).
        self.assertEqual(len(self.graph.tasks), 4)

    def test_chain_links_are_consistent(self):
        tasks = self.graph.tasks
        for i, t in enumerate(tasks[:-1]):
            self.assertEqual(t.next, tasks[i + 1].id)
        self.assertIsNone(tasks[-1].next)

    def test_ids_are_sequential(self):
        self.assertEqual([t.id for t in self.graph.tasks], [0, 1, 2, 3])

    def test_prefill_is_the_fattest_task(self):
        flops = [t.flops for t in self.graph.tasks]
        self.assertEqual(flops[0], max(flops))

    def test_output_tokens_reconstruct_via_vocab(self):
        emitted = "".join(self.graph.vocab[t.output_token] for t in self.graph.tasks)
        self.assertEqual(emitted, "one two three four")

    def test_decode_prompt_grows_by_one_each_step(self):
        decode = [t for t in self.graph.tasks if t.kind == "decode"]
        lengths = [len(t.prompt) for t in decode]
        for a, b in zip(lengths, lengths[1:]):
            self.assertEqual(b, a + 1)

    def test_serializes_to_canonical_json(self):
        s = self.graph.to_json()
        self.assertTrue(s.endswith("\n"))
        parsed = json.loads(s)
        self.assertEqual(parsed["request_id"], 7)
        self.assertEqual(len(parsed["tasks"]), 4)

    def test_empty_output_yields_single_prefill(self):
        g = build_task_graph(7, "a prompt", "", "hf://Qwen/Qwen3-1.7B")
        self.assertEqual(len(g.tasks), 1)
        self.assertEqual(g.tasks[0].kind, "prefill")
        self.assertIsNone(g.tasks[0].next)


if __name__ == "__main__":
    unittest.main()
