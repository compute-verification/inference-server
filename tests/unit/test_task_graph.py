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
    EvalPoint,
    build_task_graph,
    build_training_task_graph,
    dims_for,
    forward_flops,
    train_step_flops,
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


class TestTrainingTaskGraph(unittest.TestCase):
    """A small simulated training run: 6 steps, eval every 2 steps -> 3 evals.

    Real LoRA needs an H100 + the manifest weights, so this drives the builder
    with the exact record an eval-augmented train_once would return.
    """

    def setUp(self):
        self.max_steps = 6
        self.loss_traj = [2.0, 1.6, 1.3, 1.1, 0.95, 0.9]
        self.evals = [
            EvalPoint(step=2, metric=1.5, checkpoint_digest="sha256:aa",
                      sample_prompt="2 + 2 =", sample_output="4"),
            EvalPoint(step=4, metric=1.2, checkpoint_digest="sha256:bb",
                      sample_prompt="2 + 2 =", sample_output="4"),
            EvalPoint(step=6, metric=0.8, checkpoint_digest="sha256:cc",
                      sample_prompt="2 + 2 =", sample_output="4"),
        ]
        self.graph = build_training_task_graph(
            request_id=1,
            model_source="hf://Qwen/Qwen3-1.7B",
            max_steps=self.max_steps,
            batch_size=2,
            seq_len=8,
            loss_trajectory=self.loss_traj,
            evals=self.evals,
        )

    def _by_kind(self, kind):
        return [n for n in self.graph.nodes if n.kind == kind]

    def test_node_counts(self):
        self.assertEqual(len(self._by_kind("train_step")), 6)
        self.assertEqual(len(self._by_kind("eval")), 3)
        self.assertEqual(len(self.graph.nodes), 9)

    def test_spine_is_a_chain(self):
        spine = self._by_kind("train_step")
        for i, n in enumerate(spine[:-1]):
            self.assertEqual(n.next, spine[i + 1].id)
        self.assertIsNone(spine[-1].next)

    def test_train_steps_carry_real_losses(self):
        spine = self._by_kind("train_step")
        self.assertEqual([n.loss for n in spine], self.loss_traj)

    def test_evals_branch_off_the_right_checkpoints(self):
        # evals at steps 2, 4, 6 fork off spine indices 1, 3, 5.
        for ev_node, spine_idx, digest in [
            (self._by_kind("eval")[0], 1, "sha256:aa"),
            (self._by_kind("eval")[1], 3, "sha256:bb"),
            (self._by_kind("eval")[2], 5, "sha256:cc"),
        ]:
            spine_node = self.graph.nodes[spine_idx]
            self.assertIn(ev_node.id, spine_node.branches)
            self.assertEqual(spine_node.checkpoint_digest, digest)

    def test_eval_nodes_are_leaves(self):
        for n in self._by_kind("eval"):
            self.assertIsNone(n.next)

    def test_non_eval_steps_have_no_checkpoint(self):
        # only steps 2/4/6 (indices 1/3/5) materialized a checkpoint digest.
        for idx in (0, 2, 4):
            self.assertIsNone(self.graph.nodes[idx].checkpoint_digest)

    def test_eval_node_nests_an_inference_graph(self):
        for n in self._by_kind("eval"):
            self.assertIsNotNone(n.eval_graph)
            self.assertEqual(n.eval_graph["tasks"][0]["kind"], "prefill")
            self.assertGreater(n.flops, 0)  # forward-only cost of the nested graph

    def test_train_step_is_fatter_than_one_eval(self):
        # a 6N step over a 2x8 batch dwarfs a forward-only eval over "4".
        step = self._by_kind("train_step")[0]
        ev = self._by_kind("eval")[0]
        self.assertGreater(step.flops, ev.flops)

    def test_train_step_flops_is_triple_a_forward(self):
        dims = MODEL_DIMS["hf://Qwen/Qwen3-1.7B"]
        fwd = forward_flops(dims, tokens_in_pass=2 * 8, context_len=8)
        self.assertEqual(train_step_flops(dims, 2, 8), 3 * fwd)

    def test_serializes_to_canonical_json(self):
        s = self.graph.to_json()
        self.assertTrue(s.endswith("\n"))
        parsed = json.loads(s)
        self.assertEqual(len(parsed["nodes"]), 9)

    def test_eval_step_out_of_range_raises(self):
        with self.assertRaises(ValueError):
            build_training_task_graph(
                request_id=1, model_source="hf://Qwen/Qwen3-1.7B",
                max_steps=3, batch_size=1, seq_len=4,
                loss_trajectory=[1.0, 0.9, 0.8],
                evals=[EvalPoint(step=9, metric=1.0, checkpoint_digest="sha256:zz")],
            )


if __name__ == "__main__":
    unittest.main()
