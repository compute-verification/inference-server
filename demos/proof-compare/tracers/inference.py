"""Inference tracer: a decode run -> a canonical trace.

The model is behind ``next_token`` so this is unit-testable on CPU; the real GPU
model is plugged in by demos/proof-compare/capture/run_inference.py (Task 11).
One prefill event (reads the whole prompt) then one decode event per generated
token, chained.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from modules.proof_server.tracer import Tracer


def trace_inference(prompt_ids, next_token, model_key, shape_config, max_tokens):
    """Trace greedy decoding.

    ``next_token(ids) -> int`` is any deterministic next-token function.
    ``shape_config`` is the model's config dict (real or from KNOWN_SHAPES).
    Returns a canonical trace dict.
    """
    tr = Tracer()
    tr.add_shape(model_key, shape_config)

    ctx = list(prompt_ids)
    p = len(ctx)
    # Prefill: read all P prompt tokens in one pass; causal attention triangle.
    prev = tr.event(
        "prefill", model=model_key, tokens=p, attended=p * (p + 1) // 2,
        logits=1, label="prefill", payload={"prompt_len": p},
    )

    for _ in range(max_tokens):
        t = next_token(ctx)
        ctx.append(t)
        # Decode: one new token attending over the whole context so far.
        prev = tr.event(
            "decode", model=model_key, tokens=1, attended=len(ctx),
            logits=1, inputs=[prev], label="decode", payload={"token_id": t},
        )

    return tr.trace()


def mock_next_token(seq):
    """Deterministic stand-in model for tests: emit (last token + 1)."""
    return (seq[-1] + 1) if seq else 0
