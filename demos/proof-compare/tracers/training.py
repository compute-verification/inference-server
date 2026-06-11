"""Training tracer — converts a REAL captured LoRA run into a canonical trace.

The GPU harness (demos/proof-compare/capture/run_lora.py) runs a toy-scale but
real LoRA fine-tune (frozen base, low-rank adapters on q/v projections) and
records the per-step loss plus a real greedy eval generation every few steps.
This tracer renders that capture as a chain of ``train_step`` events with each
eval FLATTENED into real ``eval_prefill`` + ``eval_decode`` events linked off
its checkpoint step (the canonical model has no nesting; making the eval real
events IS the unification).
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from modules.proof_server.tracer import Tracer

PREVIEW_CHARS = 280


def trace_training_real(capture: dict) -> dict:
    """Canonical trace from a LoRA-training capture.

    ``capture``: ``{model, config, batch, seq_len, steps: [{step, loss}],
    evals: [{after_step, prompt, prompt_tokens, gen_tokens, text}]}``.
    ``after_step`` is 1-based: the eval ran after that many completed steps.
    """
    steps = capture.get("steps") or []
    if not steps:
        raise ValueError("a training capture needs at least one step")

    model_key = capture["model"]
    tr = Tracer()
    tr.add_shape(model_key, capture["config"])

    batch = int(capture["batch"])
    seq_len = int(capture["seq_len"])
    tokens = batch * seq_len
    attended = batch * seq_len * (seq_len + 1) // 2  # one causal triangle per row

    evals_after: dict[int, list[dict]] = {}
    for ev in capture.get("evals", []):
        after = int(ev["after_step"])
        if not 1 <= after <= len(steps):
            raise ValueError(f"eval after_step {after} outside 1..{len(steps)}")
        evals_after.setdefault(after, []).append(ev)

    prev_step = None
    for i, st in enumerate(steps):
        step_id = tr.event(
            "train_step", model=model_key, tokens=tokens, attended=attended,
            mode="lora_bwd", logits=tokens,
            inputs=([prev_step] if prev_step is not None else []),
            label=f"step {st['step']}", payload={"loss": st["loss"]},
        )
        prev_step = step_id

        for ev in evals_after.get(i + 1, []):
            p = int(ev["prompt_tokens"])
            # g generated tokens = g real passes: the eval prefill's last
            # position produces the first token, then g-1 decode passes.
            n_dec = max(int(ev["gen_tokens"]) - 1, 0)
            payload = {"prompt": ev["prompt"], "after_step": i + 1,
                       "loss": st["loss"]}
            if n_dec == 0 and ev.get("text"):
                payload["out"] = ev["text"][:PREVIEW_CHARS]
            prev = tr.event(
                "eval_prefill", model=model_key, tokens=p,
                attended=p * (p + 1) // 2, logits=1, inputs=[step_id],
                label=f"eval after step {st['step']}", payload=payload,
            )
            for j in range(1, n_dec + 1):
                dp = {"phase": "eval", "after_step": i + 1}
                if j == n_dec and ev.get("text"):
                    dp["out"] = ev["text"][:PREVIEW_CHARS]
                prev = tr.event(
                    "eval_decode", model=model_key, tokens=1, attended=p + j,
                    logits=1, inputs=[prev], label="eval_decode", payload=dp,
                )

    return tr.trace()
