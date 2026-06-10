"""Inference tracer: a decode run -> a canonical trace.

The model is behind ``next_token`` so this is unit-testable on CPU; the real GPU
model is plugged in by demos/proof-compare/capture/run_inference.py (Task 11).

One node = one REAL forward pass. HF-style generation runs exactly g forwards
to emit g tokens: the prefill's last position produces the first generated
token, and each decode pass consumes the previous token and produces the next.
No pass ever consumes the final token (generation stops first), so a run of g
tokens is 1 prefill + (g-1) decodes -- each node carries the token it PRODUCED.
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
    payload = {"prompt_len": p}
    if max_tokens > 0:
        t = next_token(ctx)          # produced by the prefill's last position
        ctx.append(t)
        payload["token_id"] = t
    # Prefill: read all P prompt tokens in one pass; causal attention triangle.
    prev = tr.event(
        "prefill", model=model_key, tokens=p, attended=p * (p + 1) // 2,
        logits=1, label="prefill", payload=payload,
    )

    for _ in range(max_tokens - 1):
        # This pass consumes the newest token, attending the whole context.
        attended = len(ctx)
        t = next_token(ctx)
        ctx.append(t)
        prev = tr.event(
            "decode", model=model_key, tokens=1, attended=attended,
            logits=1, inputs=[prev], label="decode", payload={"token_id": t},
        )

    return tr.trace()


def mock_next_token(seq):
    """Deterministic stand-in model for tests: emit (last token + 1)."""
    return (seq[-1] + 1) if seq else 0
