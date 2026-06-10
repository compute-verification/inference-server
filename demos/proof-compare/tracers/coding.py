"""Coding-agent tracer — converts a REAL captured agent run into a DAG of
forward passes.

The GPU harness (demos/proof-compare/capture/run_coding_agent.py) runs an
actual agent scaffold against a real model and records one entry per LLM call:
how many prompt tokens it read, how many tokens it generated, and which earlier
calls' outputs fed its prompt. This tracer renders that capture as canonical
events where every node is exactly one forward pass — one ``prefill`` (reads
the call's assembled prompt) or one ``decode`` (emits one token), the same
primitives as inference.

The agent issues independent LLM calls in parallel (concurrent reads of
different files, concurrent plan candidates, concurrent codegen), then merges —
so the graph is a DAG of prefill/decode chains that fan out and fan in:

    orient ─▶ (read × N files) ─▶ (plan × K candidates) ─▶ synthesize
           ─▶ (write src ‖ write test) ─▶ test verdict [─▶ fix ─▶ re-test]

Each call is an independent context (the scaffold assembles a fresh prompt per
call), so its prefill attends its own causal triangle p·(p+1)/2 and decode i
attends p+i — no shared-KV assumption. The dataflow edges (``parents``) carry
the agent structure. Tool calls (file reads, running the tests) execute no
model forward pass, so they are not nodes; their output is the next call's
prompt (tagged ``payload.via``).
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from modules.proof_server.tracer import Tracer

PREVIEW_CHARS = 280


def trace_coding_real(capture: dict) -> dict:
    """Canonical trace from a coding-agent capture (see module docstring).

    ``capture["calls"]`` is ordered; each call: ``{id, phase, role, via?,
    parents, prompt_tokens, gen_tokens, text}``. Parents must be earlier calls.
    """
    calls = capture.get("calls") or []
    if not calls:
        raise ValueError("a coding capture needs at least one call")

    model_key = capture["model"]
    tr = Tracer()
    tr.add_shape(model_key, capture["config"])

    tail: dict[int, int] = {}  # call id -> last node id of that call
    for call in calls:
        cid = int(call["id"])
        for par in call["parents"]:
            if par not in tail:
                raise ValueError(f"call {cid} references unknown/later parent {par}")
        p = int(call["prompt_tokens"])
        if p <= 0:
            raise ValueError(f"call {cid} has no prompt tokens (every real call reads a prompt)")

        phase = call["phase"]
        role = call["role"]
        payload = {"role": role, "phase": phase}
        if call.get("via"):
            payload["via"] = call["via"]
        if cid == 0:
            payload["prompt"] = capture["prompt"]

        prev = tr.event(
            "prefill", model=model_key, tokens=p, attended=p * (p + 1) // 2,
            logits=1, inputs=[tail[par] for par in call["parents"]],
            label=phase, payload=payload,
        )
        g = int(call["gen_tokens"])
        for i in range(1, g + 1):
            dp = {"role": role, "phase": phase}
            if i == g and call.get("text"):
                dp["out"] = call["text"][:PREVIEW_CHARS]
            prev = tr.event(
                "decode", model=model_key, tokens=1, attended=p + i, logits=1,
                inputs=[prev], label=role, payload=dp,
            )
        tail[cid] = prev

    return tr.trace()
