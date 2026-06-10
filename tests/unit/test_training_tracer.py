"""Unit tests for the training tracer (real-capture converter).

The GPU harness (demos/proof-compare/capture/run_lora.py) records real per-step
losses and real eval generations from a toy-scale LoRA run.
``trace_training_real`` converts that capture into the canonical trace: a chain
of ``train_step`` events with eval branches flattened into real
``eval_prefill``/``eval_decode`` events.
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
import training


def _capture(steps=None, evals=None):
    return {
        "kind": "lora_training_capture",
        "model": "Qwen/Qwen3-1.7B",
        "config": dict(F.KNOWN_SHAPES["Qwen/Qwen3-1.7B"]),
        "lora": {"r": 8, "alpha": 16, "targets": ["q_proj", "v_proj"]},
        "batch": 2,
        "seq_len": 16,
        "steps": steps if steps is not None else [
            {"step": 0, "loss": 3.2}, {"step": 1, "loss": 2.4},
            {"step": 2, "loss": 1.9}, {"step": 3, "loss": 1.5},
        ],
        "evals": evals if evals is not None else [
            {"after_step": 2, "prompt": "Deterministic inference means",
             "prompt_tokens": 5, "gen_tokens": 3, "text": " the same"},
            {"after_step": 4, "prompt": "Deterministic inference means",
             "prompt_tokens": 5, "gen_tokens": 3, "text": " bitwise"},
        ],
    }


class TestTrainingTracer(unittest.TestCase):
    def setUp(self):
        self.trace = training.trace_training_real(_capture())
        self.evs = self.trace["events"]
        self.byid = {e["id"]: e for e in self.evs}

    def test_train_step_count_matches_capture(self):
        self.assertEqual(len([e for e in self.evs if e["kind"] == "train_step"]), 4)

    def test_each_eval_is_a_flattened_inference(self):
        self.assertEqual(len([e for e in self.evs if e["kind"] == "eval_prefill"]), 2)
        self.assertEqual(len([e for e in self.evs if e["kind"] == "eval_decode"]), 6)

    def test_eval_branches_off_its_train_step(self):
        # eval with after_step=2 hangs off the 2nd train_step (0-indexed step 1).
        steps = [e for e in self.evs if e["kind"] == "train_step"]
        ep = next(e for e in self.evs if e["kind"] == "eval_prefill")
        self.assertEqual(ep["inputs"], [steps[1]["id"]])

    def test_train_steps_are_chained(self):
        steps = [e for e in self.evs if e["kind"] == "train_step"]
        for i in range(1, len(steps)):
            self.assertIn(steps[i - 1]["id"], steps[i]["inputs"])

    def test_step_carries_real_loss_and_lora_mode(self):
        steps = [e for e in self.evs if e["kind"] == "train_step"]
        self.assertEqual([s["payload"]["loss"] for s in steps], [3.2, 2.4, 1.9, 1.5])
        self.assertTrue(all(s["mode"] == "lora_bwd" for s in steps))

    def test_step_token_accounting(self):
        # batch * seq_len tokens, batch causal triangles, logits at every position.
        step = next(e for e in self.evs if e["kind"] == "train_step")
        self.assertEqual(step["tokens"], 2 * 16)
        self.assertEqual(step["attended"], 2 * 16 * 17 // 2)
        self.assertEqual(step["logits"], 2 * 16)

    def test_eval_decode_attends_growing_context(self):
        decs = [e for e in self.evs if e["kind"] == "eval_decode"][:3]
        self.assertEqual([d["attended"] for d in decs], [6, 7, 8])

    def test_eval_text_recorded_on_last_decode(self):
        first_eval_decs = [e for e in self.evs if e["kind"] == "eval_decode"
                           and e["payload"]["after_step"] == 2]
        self.assertEqual(first_eval_decs[-1]["payload"]["out"], " the same")

    def test_lora_step_is_just_over_2x_forward(self):
        g = build_graph(self.trace)
        step = next(n for n in g.nodes if n["kind"] == "train_step")
        shape = F.model_shape_from_config(F.KNOWN_SHAPES["Qwen/Qwen3-1.7B"])
        fwd = F.flops(shape, step["tokens"], step["attended"], "fwd", step["logits"])
        self.assertTrue(2.0 < step["flops"] / fwd < 2.01)

    def test_builds_into_valid_graph(self):
        self.assertTrue(build_graph(self.trace).nodes)

    def test_empty_steps_rejected(self):
        with self.assertRaises(ValueError):
            training.trace_training_real(_capture(steps=[]))

    def test_eval_after_unknown_step_rejected(self):
        with self.assertRaises(ValueError):
            training.trace_training_real(_capture(evals=[
                {"after_step": 99, "prompt": "x", "prompt_tokens": 2,
                 "gen_tokens": 1, "text": "y"}]))


if __name__ == "__main__":
    unittest.main()
