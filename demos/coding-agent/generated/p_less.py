"""p-less sampling — a hyperparameter-free truncation sampler.

Implements the method from "p-less Sampling: A Robust Hyperparameter-Free
Approach for LLM Decoding" (Tang et al., arXiv:2509.23234, Sept 2025).

The idea: instead of a tunable cutoff (top-k's k, top-p's p, min-p's p), set the
truncation threshold to the distribution's own **collision likelihood**

    L[P] = Σ_v P(v)^2            (= exp(-H_2(P)), the Renyi-2 entropy)

and keep every token at least as likely as that:

    V_keep = { v : P(v) >= L[P] }

then renormalize over V_keep and sample. This is parameter-free and adapts to
the distribution's shape: a flat (high-entropy) distribution has a small L so it
keeps many tokens; a peaked (confident) distribution has a large L so it keeps
few. The kept set is never empty -- the argmax always survives, because
max_v P(v) >= Σ_v P(v)^2 (as Σ P(v)^2 <= max_v P(v) * Σ_v P(v) = max_v P(v)).

Pure-Python / stdlib only so it unit-tests on CPU with no torch.
"""
from __future__ import annotations

import math
import random
from typing import Optional, Sequence


def softmax(logits: Sequence[float], temperature: float = 1.0) -> list[float]:
    """Numerically-stable softmax with optional temperature."""
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    m = max(logits)
    exps = [math.exp((x - m) / temperature) for x in logits]
    z = sum(exps)
    return [e / z for e in exps]


def collision_likelihood(probs: Sequence[float]) -> float:
    """L[P] = Σ_v P(v)^2 -- the p-less truncation threshold."""
    return sum(p * p for p in probs)


def p_less_filter(probs: Sequence[float]) -> tuple[list[int], list[float], float]:
    """Apply p-less truncation to a probability vector.

    Returns ``(kept_indices, renormalized_probs, threshold)`` where the kept set
    is ``{ v : P(v) >= L[P] }``, renormalized to sum to 1. Never empty.
    """
    threshold = collision_likelihood(probs)
    kept = [(i, p) for i, p in enumerate(probs) if p >= threshold]
    if not kept:  # theoretically impossible; defensive fallback to the argmax
        i = max(range(len(probs)), key=lambda j: probs[j])
        kept = [(i, probs[i])]
    z = sum(p for _, p in kept)
    indices = [i for i, _ in kept]
    renorm = [p / z for _, p in kept]
    return indices, renorm, threshold


def p_less_sample(
    logits: Sequence[float],
    temperature: float = 1.0,
    rng: Optional[random.Random] = None,
) -> int:
    """Sample one token id from ``logits`` using p-less truncation."""
    probs = softmax(logits, temperature)
    indices, renorm, _ = p_less_filter(probs)
    r = (rng or random).random()
    acc = 0.0
    for i, p in zip(indices, renorm):
        acc += p
        if r <= acc:
            return i
    return indices[-1]
