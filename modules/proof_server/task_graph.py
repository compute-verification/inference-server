"""Task graph: a forward pass -> task chain over one LLM generation.

A generation is autoregressive: each forward pass takes every token so far as
input and emits exactly one new token, which is appended and fed into the next
pass. So a generation of N output tokens is a *chain*:

    [prompt]              --prefill-->  tok_0
    [prompt, tok_0]       --decode_0--> tok_1
    [prompt, tok_0, tok_1]--decode_1--> tok_2
    ...

One ``Task`` == one forward pass == one emitted token. The first pass
(``kind="prefill"``) ingests the whole prompt at once; every subsequent pass
(``kind="decode"``) ingests a single token but attends over the full, growing
context. That asymmetry is real and shows up in ``flops``: prefill is fat
(work proportional to the prompt length, attention quadratic in it) while each
decode step is thin (~``2 * n_params``).

FLOPs model (the "option 2" estimate)
-------------------------------------
A matmul costs 2 FLOPs per parameter per token (one multiply, one add), so the
weight term is ``2 * n_params * tokens_in_pass``. The only term that does NOT
scale with parameter count is attention (activation x activation), which scales
with context length: ``4 * n_layers * d_model * context_len * tokens_in_pass``.
See ``forward_flops``.

Two deliberate approximations (this graph is built but not yet consumed):
  * **Tokenization** is approximate (``_approx_tokenize``). The demo runs in
    mock mode where outputs are canned strings, so a real BPE tokenizer would
    be meaningless; we split on whitespace-delimited chunks instead. Swap in a
    real tokenizer here when the graph gets a consumer.
  * **Model dims** come from a small static lookup keyed by the manifest's
    ``model.source`` (``MODEL_DIMS``), because the manifest pins ``config.json``
    by digest + ``hf://`` source rather than embedding ``n_params/n_layers/
    d_model``. Unknown models fall back to ``DEFAULT_DIMS``.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Model dimensions (for the FLOPs estimate)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelDims:
    n_params: int
    n_layers: int
    d_model: int


# Keyed by manifest ``model.source``. The manifest references config.json by
# digest only, so these are filled in from each model's published HF config.
MODEL_DIMS: dict[str, ModelDims] = {
    "hf://Qwen/Qwen3-1.7B": ModelDims(n_params=1_720_000_000, n_layers=28, d_model=2048),
}

# Used when ``model.source`` is not in the table. Picked so the shapes of the
# graph (fat prefill, thin decode chain) are still sensible.
DEFAULT_DIMS = ModelDims(n_params=1_000_000_000, n_layers=24, d_model=2048)


def dims_for(model_source: str) -> ModelDims:
    """Look up model dims by manifest ``model.source``; fall back to defaults."""
    return MODEL_DIMS.get(model_source, DEFAULT_DIMS)


# ---------------------------------------------------------------------------
# FLOPs
# ---------------------------------------------------------------------------

def forward_flops(dims: ModelDims, tokens_in_pass: int, context_len: int) -> int:
    """FLOPs for one forward pass.

    ``tokens_in_pass`` is how many positions this pass processes (the whole
    prompt for prefill, 1 for a decode step). ``context_len`` is how many
    positions are attended over (== prompt length for prefill; the running
    context length for a decode step).
    """
    weight_term = 2 * dims.n_params * tokens_in_pass
    attention_term = 4 * dims.n_layers * dims.d_model * context_len * tokens_in_pass
    return weight_term + attention_term


# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------

@dataclass
class Task:
    # --- identity ---
    id: int                      # position in the chain: 0 = prefill, 1.. = decode
    # --- the three fields the demo cares about ---
    flops: int                   # cost of THIS forward pass
    prompt: list[int]            # the full (growing) context fed in, as token ids
    next: Optional[int]          # id of the task that consumes this output; None at the tail
    # --- forced on us by the prefill/decode asymmetry ---
    kind: str                    # "prefill" | "decode"
    output_token: int            # the single token id this pass emitted


@dataclass
class TaskGraph:
    request_id: int
    model_source: str
    vocab: dict[int, str]        # token id -> text, so a stored graph is inspectable
    tasks: list[Task] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "request_id": self.request_id,
            "model_source": self.model_source,
            "vocab": {str(k): v for k, v in self.vocab.items()},
            "tasks": [asdict(t) for t in self.tasks],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")) + "\n"


# ---------------------------------------------------------------------------
# Approximate tokenization
# ---------------------------------------------------------------------------

def _approx_tokenize(text: str) -> list[str]:
    """Split ``text`` into whitespace-delimited chunks, keeping trailing space.

    Concatenating the chunks reconstructs the input exactly. This is a stand-in
    for a real tokenizer (see module docstring); good enough to give the graph a
    plausible token count and a per-token chain.
    """
    if not text:
        return []
    return re.findall(r"\S+\s*|\s+", text)


class _Vocab:
    """Assigns a stable int id to each distinct token text, in first-seen order."""

    def __init__(self) -> None:
        self._to_id: dict[str, int] = {}
        self._to_text: dict[int, str] = {}

    def id_of(self, text: str) -> int:
        if text not in self._to_id:
            new_id = len(self._to_id)
            self._to_id[text] = new_id
            self._to_text[new_id] = text
        return self._to_id[text]

    def table(self) -> dict[int, str]:
        return dict(self._to_text)


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

def build_task_graph(
    request_id: int,
    prompt: str,
    output: str,
    model_source: str,
) -> TaskGraph:
    """Transform one (prompt, output) generation into a task-graph chain.

    Task 0 is the prefill (ingests the whole prompt, fat FLOPs). Each subsequent
    task is one decode step emitting one output token, chained via ``next``.
    """
    dims = dims_for(model_source)
    vocab = _Vocab()

    prompt_tokens = [vocab.id_of(t) for t in _approx_tokenize(prompt)]
    output_tokens = [vocab.id_of(t) for t in _approx_tokenize(output)]

    tasks: list[Task] = []

    # --- prefill: one fat pass over the whole prompt, emitting the 1st token ---
    p = len(prompt_tokens)
    first_out = output_tokens[0] if output_tokens else -1
    tasks.append(Task(
        id=0,
        flops=forward_flops(dims, tokens_in_pass=max(p, 1), context_len=max(p, 1)),
        prompt=list(prompt_tokens),
        next=1 if len(output_tokens) > 1 else None,
        kind="prefill",
        output_token=first_out,
    ))

    # --- decode: one thin pass per remaining output token ---
    context = list(prompt_tokens)
    if output_tokens:
        context.append(output_tokens[0])
    for i in range(1, len(output_tokens)):
        ctx_len = len(context)
        has_next = i + 1 < len(output_tokens)
        tasks.append(Task(
            id=i,
            flops=forward_flops(dims, tokens_in_pass=1, context_len=ctx_len),
            prompt=list(context),
            next=(i + 1) if has_next else None,
            kind="decode",
            output_token=output_tokens[i],
        ))
        context.append(output_tokens[i])

    return TaskGraph(
        request_id=request_id,
        model_source=model_source,
        vocab=vocab.table(),
        tasks=tasks,
    )
