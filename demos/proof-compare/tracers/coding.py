"""Coding-agent tracer (STUB) — a DAG of forward passes.

Every node is exactly one forward pass — one ``prefill`` (reads a batch of
tokens) or one ``decode`` (emits one token), the same primitives as inference.
But a real agent is not a single chain: it issues LLM calls **in parallel**
(concurrent reads of different sources, concurrent sub-agents writing different
files), then merges. So the graph is a DAG of forward-pass chains that fan out
and fan in:

    reason ─▶ (read paper ‖ read repo) ─▶ plan ─▶ (write src ‖ write test) ─▶ test

Each branch is its own chain of prefill/decode; a fan-in node (the merge) takes
all branch tails as inputs. Tool calls (search/fetch/run-tests) run no model
forward pass, so they are not nodes — their output is the next prefill's tokens
(tagged ``payload.via``).

STUB: token counts are estimates from a real "summarize a paper, then implement
it" run; the parallelism models an agent that dispatches concurrent LLM calls
(a real pattern), as opposed to a purely sequential agent (which would be one
chain). The *shape* and FLOPs accounting are correct.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from modules.proof_server import flops as F
from modules.proof_server.tracer import Tracer


def _emit_turn(tr, agent_key, turn, inputs, ctx_in, prompt=None, first=False):
    """Emit one LLM turn as prefill (optional) + one decode per generated token.

    ``inputs`` are the node ids this turn forks from (a fan-in if more than one).
    Returns ``(tail_id, ctx_out)`` so callers can chain or merge.
    """
    role = turn["role"]
    # `phase` (the per-turn label, e.g. "write p_less.py") groups a turn's
    # forward passes together in the viz without merging adjacent same-role turns.
    phase = turn.get("label") or role
    p = int(turn.get("prefill", 0))
    ctx = ctx_in
    cur_inputs = list(inputs)
    prev = None
    if p > 0:
        # New tokens occupy positions [ctx, ctx+p); each attends to all tokens up
        # to and including itself -> a causal-triangle difference.
        attended = (ctx + p) * (ctx + p + 1) // 2 - ctx * (ctx + 1) // 2
        ctx += p
        payload = {"role": role, "phase": phase}
        if first and prompt is not None:
            payload["prompt"] = prompt
        if turn.get("via"):
            payload["via"] = turn["via"]
        prev = tr.event(
            "prefill", model=agent_key, tokens=p, attended=attended, logits=1,
            inputs=cur_inputs, label=phase, payload=payload,
        )
        cur_inputs = [prev]
    for _ in range(int(turn.get("gen", 0))):
        ctx += 1
        prev = tr.event(
            "decode", model=agent_key, tokens=1, attended=ctx, logits=1,
            inputs=cur_inputs, label=role, payload={"role": role, "phase": phase},
        )
        cur_inputs = [prev]
    if prev is None:
        raise ValueError(f"turn {phase!r} emitted no forward pass (need prefill or gen)")
    return prev, ctx


def trace_coding_stub(agent_key, prompt, stages):
    """Render an agent run as a DAG of prefill/decode forward passes.

    ``prompt`` is the user prompt text (stored in the first prefill's payload).
    ``stages`` is a list; each element is either:

      * a turn dict (sequential) -- runs after the current frontier::
            {"role", "prefill", "gen", "via"?, "label"?}
      * a parallel group -- all sub-turns fork from the current frontier and run
        concurrently; the next stage fans in from all their tails::
            {"parallel": [turn, turn, ...]}

    A turn with ``prefill == 0`` is a continuation (forks straight into decodes).
    """
    if not stages:
        raise ValueError("a coding trace needs at least one stage")

    tr = Tracer()
    tr.add_shape(agent_key, F.shape_for(agent_key))

    frontier = []      # node ids the next stage depends on
    ctx = 0            # running context length (shared prefix before a fork)
    first = True
    for stage in stages:
        if isinstance(stage, dict) and "parallel" in stage:
            fork_ctx = ctx
            tails, growth = [], 0
            for turn in stage["parallel"]:
                tail, c = _emit_turn(
                    tr, agent_key, turn, frontier, fork_ctx, prompt=prompt, first=first)
                first = False
                tails.append(tail)
                growth += c - fork_ctx
            frontier = tails
            # the next (merge) turn re-reads every branch: shared prefix once
            # plus each branch's additions.
            ctx = fork_ctx + growth
        else:
            tail, ctx = _emit_turn(
                tr, agent_key, stage, frontier, ctx, prompt=prompt, first=first)
            first = False
            frontier = [tail]

    return tr.trace()
