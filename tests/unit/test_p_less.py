"""Unit tests for the agent-implemented p-less sampler (arXiv:2509.23234).

The implementation under test was produced by the coding-agent demo
(demos/coding-agent); these tests are the demo's "verify" node.
"""
import math
import random
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
GEN_DIR = REPO_ROOT / "demos" / "coding-agent" / "generated"
if str(GEN_DIR) not in sys.path:
    sys.path.insert(0, str(GEN_DIR))

import p_less


class TestPLess(unittest.TestCase):
    def test_threshold_is_sum_of_squared_probs(self):
        probs = [0.5, 0.3, 0.2]
        self.assertAlmostEqual(p_less.collision_likelihood(probs),
                               0.25 + 0.09 + 0.04)

    def test_uniform_keeps_everything(self):
        # Uniform: P(v) = 1/n, L = n*(1/n^2) = 1/n, so every token has P(v) == L.
        probs = [0.25] * 4
        idx, renorm, thr = p_less.p_less_filter(probs)
        self.assertEqual(sorted(idx), [0, 1, 2, 3])
        self.assertAlmostEqual(thr, 0.25)

    def test_peaked_keeps_only_the_top(self):
        probs = [0.94, 0.02, 0.02, 0.02]
        idx, renorm, thr = p_less.p_less_filter(probs)
        self.assertEqual(idx, [0])              # L ~ 0.88 > 0.02 prunes the tail
        self.assertAlmostEqual(sum(renorm), 1.0)

    def test_argmax_always_survives(self):
        # max_v P(v) >= Σ P(v)^2, so the kept set is never empty.
        rng = random.Random(0)
        for _ in range(200):
            raw = [rng.random() for _ in range(20)]
            z = sum(raw)
            probs = [x / z for x in raw]
            idx, _, _ = p_less.p_less_filter(probs)
            self.assertIn(max(range(len(probs)), key=lambda j: probs[j]), idx)
            self.assertGreaterEqual(len(idx), 1)

    def test_renormalizes_to_one(self):
        rng = random.Random(1)
        for _ in range(50):
            raw = [rng.random() for _ in range(10)]
            z = sum(raw)
            _, renorm, _ = p_less.p_less_filter([x / z for x in raw])
            self.assertAlmostEqual(sum(renorm), 1.0)

    def test_flatter_keeps_at_least_as_many(self):
        peaked = p_less.softmax([5.0, 1.0, 0.5, 0.0])
        flat = p_less.softmax([1.0, 0.9, 0.8, 0.7])
        n_peaked = len(p_less.p_less_filter(peaked)[0])
        n_flat = len(p_less.p_less_filter(flat)[0])
        self.assertGreaterEqual(n_flat, n_peaked)

    def test_sample_is_deterministic_with_seed(self):
        logits = [2.0, 1.0, 0.5, 0.1, 0.0]
        a = p_less.p_less_sample(logits, rng=random.Random(42))
        b = p_less.p_less_sample(logits, rng=random.Random(42))
        self.assertEqual(a, b)

    def test_sample_only_returns_kept_tokens(self):
        logits = [6.0, 0.1, 0.1, 0.1, 0.1]   # very peaked -> only token 0 kept
        for s in range(30):
            self.assertEqual(p_less.p_less_sample(logits, rng=random.Random(s)), 0)

    def test_softmax_temperature(self):
        self.assertAlmostEqual(sum(p_less.softmax([1.0, 2.0, 3.0], 0.7)), 1.0)
        with self.assertRaises(ValueError):
            p_less.softmax([1.0], 0.0)


if __name__ == "__main__":
    unittest.main()
