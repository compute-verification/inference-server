# Plan: Unified, auto-generated task graphs

**Status:** ready for implementation
**Branch to work on:** `proof-compare-graph` (already checked out at
`/home/jon/projects/dss-proof-compare`). Do **not** start a new branch.
**Audience:** an engineer who is new to this codebase and to the problem domain.
Read Part 0 fully before writing any code.

---

## How to use this document

- Do the tasks **in order**. Each task is sized to be one commit (sometimes two).
- Every code task is **test-first** (write the failing test, then the code). See
  the "Test design primer" in Part 0 — follow it; it is not optional.
- **Commit after every green task.** Small commits. Commit messages: imperative
  mood, one line summary + a short body. Do **not** add Claude/AI as a co-author
  (repo rule).
- When a task says "STOP for review", stop and wait — a reviewer will check your
  work before you continue.
- If something in this plan is wrong or impossible, **stop and say so** — do not
  improvise a different design.

---

# Part 0 — Background (read this first)

## 0.1 What this project is (one paragraph)

This repo is about **bitwise-deterministic LLM inference**: the research claim is
that two independent servers, given the same weights/prompt/config, emit
identical tokens. You do **not** need to understand the determinism machinery.
You are working on a **visualization side-project**: we represent different LLM
workloads (plain inference, LoRA training, speculative decoding, a coding agent)
as **task graphs**, and render them on a web page. Your job is to make those
graphs **auto-generated from real runs** through **one** shared pipeline,
instead of the four hand-built ones that exist today.

## 0.2 Domain primitives you must understand

Learn these five ideas; everything else builds on them.

1. **Forward pass.** An LLM is a stack of `L` transformer layers. Running input
   through it once is a "forward pass". It costs a fixed, computable number of
   floating-point operations (**FLOPs**) that depends only on the model's shape
   and how many tokens you push through. Part 2 gives the exact formula.

2. **Token.** Text is split into tokens (~¾ of a word each). The model's input
   and output are sequences of integer token ids. A **tokenizer** maps text ↔
   ids.

3. **Inference = prefill then decode.**
   - **Prefill:** the model reads the whole prompt (`P` tokens) in *one* forward
     pass. Expensive (work ∝ `P`, attention ∝ `P²`).
   - **Decode:** it then generates one token at a time; each new token is *one*
     forward pass that reads only that token but attends over the whole context
     so far. Cheap per step. So generating `N` tokens = 1 prefill + `N` decode
     passes, forming a **chain**.

4. **LoRA training.** Fine-tuning where the big base weights are **frozen** and
   only small "adapter" matrices are trained. One **training step** = a forward
   pass + a backward pass (gradients) + a weight update, over a batch. We also
   **evaluate** periodically (run the current model on a held-out set) — those
   evals branch off the training chain.

5. **Speculative decoding.** A small **draft** model proposes `k` tokens; the big
   **target** model **verifies** all `k` in one forward pass, keeps the longest
   correct prefix, and emits one correction. Accepted drafts are committed;
   rejected ones are thrown away (but were still *fed into* the verify pass).

6. **The "coding agent".** An LLM agent that, to implement a recent paper, runs
   web **searches**, **fetches** pages, forms a **plan**, writes code
   (**codegen**), and runs tests (**verify**).

**The unifying idea:** every one of these is a graph of **tasks**, where each
task is (essentially) one model forward/backward pass that has: an **input
(prompt/context)**, a **cost (FLOPs)**, and an **output**, plus pointers
(**edges**) to the tasks that consume it. That is the whole abstraction.

## 0.3 Toolset & conventions (this repo is unusual — read carefully)

- **Python, stdlib-first.** No frameworks. Type hints encouraged.
- **Tests use `unittest`, NOT pytest.** Test files live in `tests/unit/` named
  `test_*.py`, classes subclass `unittest.TestCase`.
- **Two Python interpreters:**
  - `python3` (system) — has only stdlib. Use for pure-logic tests.
  - `.venv/bin/python3` — has `pydantic` + `torch`. Use to run the **full**
    suite (some test modules import server code that needs pydantic). If
    `.venv` is missing, create it: `cd <repo root> && uv sync`. Use **`uv`**,
    never pip/apt (repo rule).
- **Run tests:**
  - one module: `python3 -m unittest tests.unit.test_flops -v`
  - whole suite (use the venv): `.venv/bin/python3 -m unittest discover -s tests/unit`
- **Repo-root path inside a test:** `Path(__file__).resolve().parents[2]`.
- **Canonical JSON** (use everywhere we serialize graphs/traces):
  `json.dumps(data, sort_keys=True, separators=(",", ":")) + "\n"`.
- **Digests** are prefixed strings: `sha256:<hex>`.
- **The web page** is a single self-contained file:
  `demos/proof-compare/viz/index.html`. Graph data is **baked inline** into a
  `const DATA = {...};` line by a small Python script (Part 3, Task 8). It is
  published with **surge** (`surge ./ taskgraph-dss.surge.sh`); surge is already
  authenticated. Do not add a build step or CDN dependencies — keep it one file.
- **GPU runs** use **vast.ai** (CLI `vastai`, authenticated). The detailed,
  battle-tested procedure is in Task 11 — follow it exactly; vast has sharp
  edges (the SSH *proxy* is flaky; use the *direct* IP:port).

## 0.4 Test design primer (follow this — your tests will be reviewed)

You said you're rusty on test design. Rules for this codebase:

1. **Test behavior, not implementation.** Assert on what a function *returns or
   guarantees*, never on private internals. Good: "a peaked distribution keeps
   only the top token." Bad: "the function calls `sorted()` once."
2. **One idea per test method.** A test name is a sentence:
   `test_prefill_is_more_expensive_than_one_decode_step`. If you need "and" in
   the name, split it.
3. **Cover the three buckets:** the normal case, the **boundary** (empty input,
   one element, zero tokens), and the **invariant** (a property that must always
   hold — e.g. "renormalized probabilities sum to 1", "every node's cost > 0",
   "edges only point to existing node ids").
4. **Deterministic.** No clocks, no RNG without a fixed seed, **no network in
   unit tests.** If you need randomness, pass `random.Random(0)`.
5. **Hand-compute expected numbers for FLOPs tests.** Do not paste the
   function's own output back in as the "expected" value (that tests nothing).
   Pick a tiny model shape, compute the FLOPs by hand in the test's comment, and
   assert equality.
6. **Use small, explicit fixtures** built inside the test. No giant shared
   fixtures, no loading real models in unit tests (mock the model with a
   deterministic next-token function — see Task 5).
7. **Failure paths matter.** If a function raises on bad input, test that it
   raises (`with self.assertRaises(ValueError):`).

## 0.5 Orientation: what exists today (and is "wrong")

Today there are **four bespoke graph builders** in
`modules/proof_server/task_graph.py` (`build_task_graph`,
`build_training_task_graph`, `build_spec_decode_task_graph`,
`build_coding_agent_task_graph`), four bespoke **renderers** in
`demos/proof-compare/viz/index.html`, and four different node dataclasses. Three
problems we are fixing:

- **Inconsistent cost:** three scenarios use an approximate `2·N + attention`
  formula; the coding one uses weight-only. We are replacing all of it with
  **one exact FLOPs function** (Part 2).
- **Hand-fed data:** only spec-decode is built from a *real run*; the others
  were fed hand-typed / fabricated inputs. We are adding **tracers** so graphs
  come from real (or, for two scenarios, clearly-labelled stub) runs.
- **Four of everything:** we collapse to **one** node type, **one** builder,
  **one** renderer.

Read these files before starting (skim, ~20 min):
`modules/proof_server/task_graph.py`, `demos/spec-decode/spec_decode.py`,
`tests/unit/test_task_graph.py`, `demos/proof-compare/viz/index.html`
(just the `render*` functions and the `card`/`arrow` helpers).

---

# Part 1 — Target architecture

```
   real run ──[TRACER]──▶ trace (events) ──[build_graph]──▶ Graph ──[render]──▶ viz
   per scenario            ONE format        ONE builder      ONE renderer
                                                  │
                                                  ▼
                                          [flops] exact cost
```

- **`flops.py`** — exact FLOPs from a model's real shape (Part 2).
- **`graph.py`** — the canonical `Event` (trace record) + `Graph` (built) +
  `build_graph(trace)`. The single builder.
- **`tracer.py`** — a tiny recorder helper scenarios call to emit events.
- **tracers** — one per scenario, emitting canonical events. Inference is run
  **for real on a GPU** (the forcing function). spec-decode is ported from its
  existing real run. training + coding are **stubs** (canonical-format traces
  from simulated data) until the very end.
- **renderer** — one layered-DAG SVG renderer in the viz, replacing the four.

### The canonical `Event` (this is the contract — everything depends on it)

```python
@dataclass
class Event:
    id: int                # unique within a trace, assigned in creation order
    kind: str              # "prefill"|"decode"|"train_step"|"eval"|"draft"
                           #  |"verify"|"prompt"|"search"|"fetch"|"plan"|"codegen"|"test"
    inputs: list[int]      # ids of events this one depends on (these become edges)
    model: str             # key into the trace's `shapes` table (which model's cost)
    tokens: int            # tokens this event processes
    attended: int          # total (token,key) attention pairs (Part 2 §2.3); 0 = no attention
    mode: str              # "fwd" | "lora_bwd" | "full_bwd"  (cost multiplier set)
    logits: int            # number of positions that take an LM-head (logits) projection
    label: str             # short human title for the node
    payload: dict          # scenario extras: token text, loss, status, file path, etc.
```

A **trace** is `{"shapes": {model_key: <shape dict>}, "events": [Event, ...]}`.
`build_graph(trace)` computes each event's `flops` (via `flops.py`), turns
`inputs` into edges, and returns a `Graph`. **No scenario-specific code lives in
`build_graph`** — that is the whole point.

---

# Part 2 — The exact FLOPs specification

Implement this **exactly**. Convention: one multiply-accumulate = **2 FLOPs**;
a matmul `(m×k)·(k×n)` costs `2·m·k·n`.

### 2.1 Model shape (read from `config.json` / `model.config`)

```
L     = num_hidden_layers
d     = hidden_size
h     = num_attention_heads
d_h   = head_dim            (Qwen3 sets this explicitly; else d // h)
h_kv  = num_key_value_heads (GQA; == h for plain multi-head)
f     = intermediate_size   (MLP)
V     = vocab_size
```

### 2.2 Per-layer weight matmuls (per token)

```
W_layer = 4·d·h·d_h          # Q proj (2·d·h·d_h) + O proj (2·h·d_h·d)
        + 4·d·h_kv·d_h        # K and V projections (GQA -> cheaper)
        + 6·d·f               # gated MLP / SwiGLU: gate + up + down (NOT 4·d·f)
```

### 2.3 Attention (activation×activation)

A single token that attends over `s` keys costs `4·h·d_h·s` per layer
(`QKᵀ` = `2·h·d_h·s`, `softmax·V` = `2·h·d_h·s`). Define **`attended`** =
the sum of `s` over all tokens in the event:
- **decode** of 1 token at sequence length `s`: `attended = s`.
- **causal prefill** of `P` tokens (token `i` attends to `i+1` keys):
  `attended = P·(P+1)/2`.
- **parallel verify** over positions `ctx..ctx+k`: `attended = Σ_{j=0}^{k}(ctx+j)`.
Attention FLOPs `= L · 4·h·d_h · attended`.

### 2.4 LM head

Each position that produces logits costs `2·d·V`. `logits` field = number of such
positions (1 for a normal generated token; `P` for a training step over `P`
positions; `k+1` for a verify).

### 2.5 The single cost function

```
WEIGHT_MULT = {"fwd": 1, "lora_bwd": 2, "full_bwd": 3}   # base weights: fwd(+dX)(+dW)
ATTN_MULT   = {"fwd": 1, "lora_bwd": 3, "full_bwd": 3}   # both operands are activations

flops(shape, tokens, attended, mode, logits) =
      WEIGHT_MULT[mode] · ( tokens · L · W_layer  +  logits · 2·d·V )
    + ATTN_MULT[mode]   · ( L · 4·h·d_h · attended )
```

- Forward inference: `mode="fwd"`.
- LoRA training step: `mode="lora_bwd"` (base weights frozen → no `dW`, so ×2;
  attention activations still need both grads → ×3). **This is why a LoRA step
  is ≈2× a forward, not 3×.**
- Full fine-tune (not used now, but support it): `mode="full_bwd"` (×3 / ×3).

**Explicitly excluded** (document this in a comment): elementwise ops — RMSNorm,
RoPE, softmax, SiLU, residual adds — are `O(tokens·d)`, non-matmul, total <1%.
LoRA adapter matmuls (rank `r ≪ d`) are likewise negligible and omitted. Do
**not** add these unless a future task asks.

### 2.6 Sanity anchor (use as a test)

Qwen3-1.7B: `L=28, d=2048, h=16, d_h=128, h_kv=8, f=6144, V=151936`.
One decode token at `s=500`, `mode="fwd"`, `logits=1`:
- weights/token: `W_layer = 4·2048·2048 + 4·2048·1024 + 6·2048·6144 = 100,663,296`; ×28 = `2.819e9`
- head: `2·2048·151936 = 6.223e8`
- attention: `28·4·16·128·500 = 1.147e8`
- **total ≈ 3.556e9 FLOP.** Put this exact number (recomputed by hand) in a test.

---

# Part 3 — Implementation tasks

> Conventions for every task: write the test first, watch it fail, write code,
> watch it pass, run the **whole** suite with `.venv/bin/python3`, commit.

### Task 1 — Exact FLOPs module  `modules/proof_server/flops.py`

**Goal:** the single cost function from Part 2.

**Files:** create `modules/proof_server/flops.py`; create
`tests/unit/test_flops.py`.

**Write (`flops.py`):**
- `@dataclass(frozen=True) class ModelShape:` fields `L, d, h, d_h, h_kv, f, V`.
- `def model_shape_from_config(cfg: dict) -> ModelShape:` reading the HF config
  keys in §2.1 (with `d_h = cfg.get("head_dim", d // h)`,
  `h_kv = cfg.get("num_key_value_heads", h)`).
- `W_LAYER(shape)`, and `def flops(shape, tokens, attended, mode="fwd", logits=0) -> int`
  implementing §2.5 exactly. Validate `mode in WEIGHT_MULT` (raise `ValueError`).

**Tests (`test_flops.py`) — hand-computed, see §0.4 rule 5:**
- `test_decode_token_matches_hand_count`: the §2.6 anchor, assert `== 3_556_...`
  (compute the exact int yourself and paste it).
- `test_gqa_is_cheaper_than_full_attention`: same shape with `h_kv=h` vs `h_kv=h/2`
  → fewer FLOPs with smaller `h_kv`.
- `test_swiglu_uses_six_d_f`: isolate the MLP contribution (one layer, zero
  attention via `attended=0`, `logits=0`) and assert the `6·d·f` term.
- `test_lora_step_is_double_a_forward_for_weights`: with `attended=0, logits=0`,
  `flops(...,mode="lora_bwd") == 2 * flops(...,mode="fwd")`.
- `test_full_bwd_triples_weights`: same idea, ×3.
- `test_attention_grows_linearly_with_attended`.
- `test_bad_mode_raises`.
- `test_zero_tokens_is_zero` (boundary).

**Run:** `python3 -m unittest tests.unit.test_flops -v` (pure stdlib).
**Commit:** `flops: exact transformer FLOPs from real model shape`.

---

### Task 2 — Canonical graph model + builder  `modules/proof_server/graph.py`

**Goal:** `Event`, `Edge`, `Graph`, and `build_graph(trace)` — the one builder.

**Files:** create `modules/proof_server/graph.py`; create
`tests/unit/test_graph.py`.

**Write:**
- `@dataclass class Event:` exactly the fields in Part 1 (give `attended=0`,
  `mode="fwd"`, `logits=0`, `label=""`, `payload=None→{}` sensible defaults).
- `@dataclass class Edge:` `src, dst`.
- `@dataclass class Graph:` `nodes: list[dict]`, `edges: list[Edge]`,
  `shapes: dict`, plus `to_dict()` / `to_json()` (canonical JSON, §0.3).
- `def build_graph(trace: dict) -> Graph:`
  - look up each event's `ModelShape` from `trace["shapes"][event.model]`
    (via `model_shape_from_config`);
  - compute `flops` with `flops.flops(...)`;
  - each node dict = the event fields + `flops`;
  - for every event, for every `i` in `inputs`, emit `Edge(i, event.id)`;
  - **validate**: raise `ValueError` if an `inputs` id doesn't exist or if ids
    aren't unique.

**Tests:**
- `test_builds_nodes_with_costs`: a 2-event trace (prefill→decode) → 2 nodes,
  flops match `flops.py` for those args.
- `test_inputs_become_edges`: an event with `inputs=[0,1]` → two edges into it.
- `test_rejects_dangling_input` (`assertRaises`).
- `test_rejects_duplicate_ids` (`assertRaises`).
- `test_round_trips_through_canonical_json`: `json.loads(g.to_json())` has the
  right counts; string ends with `\n`.
- `test_empty_trace_is_empty_graph` (boundary).

**Commit:** `graph: canonical Event/Graph + single build_graph()`.

---

### Task 3 — Tracer helper  `modules/proof_server/tracer.py`

**Goal:** a tiny recorder so scenarios emit canonical events without bookkeeping.

**Files:** create `modules/proof_server/tracer.py`; create
`tests/unit/test_tracer.py`.

**Write:**
- `class Tracer:` holds `shapes: dict` and a list of events; auto-increments ids.
  - `add_shape(key, config_dict)`
  - `event(kind, *, inputs=(), model="", tokens=0, attended=0, mode="fwd",
    logits=0, label="", payload=None) -> int` (returns the new id)
  - `trace() -> dict` returns `{"shapes":..., "events":[...]}` (events as dicts).
- It must be a thin recorder: **no FLOPs math here** (that's the builder's job —
  DRY).

**Tests:**
- `test_event_ids_increment_from_zero`.
- `test_trace_round_trips_into_build_graph`: build a 3-event chain with the
  tracer, call `build_graph(tr.trace())`, assert 3 nodes / 2 edges.
- `test_shapes_are_carried_through`.

**Commit:** `tracer: thin canonical-event recorder`.

> **STOP for review #1** — the canon (flops + graph + tracer) is the foundation.
> Do not proceed until reviewed.

---

### Task 4 — Inference tracer (CPU-testable core)  `demos/task-graph/tracers/inference.py`

**Goal:** turn a decode run into a canonical trace. Keep the model behind a
function so it's unit-testable without a GPU; the real GPU model is plugged in in
Task 11.

**Files:** create `demos/task-graph/tracers/inference.py` and
`demos/task-graph/tracers/__init__.py`; create
`tests/unit/test_inference_tracer.py`.

**Write:**
- `def trace_inference(prompt_ids, next_token, model_key, shape_config, max_tokens) -> dict:`
  where `next_token(ids) -> int` is any deterministic next-token function.
  - one `prefill` event: `tokens=len(prompt_ids)`,
    `attended=P*(P+1)//2`, `logits=1`, `inputs=[]`, payload has the prompt length.
  - loop generating tokens; each `decode` event: `tokens=1`,
    `attended=current_seq_len`, `logits=1`, `inputs=[prev_id]`,
    payload `{"token_id": t}` (decode the text in Task 11 when a tokenizer
    exists; for the unit test, ids are fine).
  - returns a trace dict (uses `Tracer`).
- Provide a `mock_next_token(seq)` helper for tests: deterministic, e.g. returns
  `(seq[-1] + 1)`.

**Tests:**
- `test_one_prefill_then_n_decodes`: `max_tokens=4` → 1 prefill + 4 decode events.
- `test_decode_chain_is_linked`: each decode's `inputs` is the previous event id.
- `test_prefill_attended_is_causal_triangle`: `attended == P*(P+1)//2`.
- `test_builds_into_a_valid_graph`: `build_graph(trace)` succeeds, prefill flops
  > a single decode's flops.

**Commit:** `inference tracer: decode run -> canonical trace (model-agnostic)`.

---

### Task 5 — Port spec-decode to canonical events  `demos/task-graph/tracers/specdecode.py`

**Goal:** reuse the **existing real** spec-decode result (it already runs on
GPU); convert its per-round trace to canonical events. This proves the canon
covers the fan-in shape.

**Read first:** `demos/spec-decode/spec_decode.py` — `SpecResult`/`SpecRound`
(fields `drafts`, `num_accepted`, `correction`) and the existing
`build_spec_decode_task_graph` in `task_graph.py` for the intended topology
(draft chain, every draft fans into the verify, round handoff).

**Files:** create `demos/task-graph/tracers/specdecode.py`; create
`tests/unit/test_specdecode_tracer.py`.

**Write:**
- `def trace_spec_decode(prompt_len, rounds, draft_key, draft_cfg, target_key,
  target_cfg) -> dict:` emitting, per round:
  - `draft` events (model=`draft_key`, `tokens=1`, `attended=ctx+i`, chained
    `inputs=[prev draft]`), payload `{"token", "status": accepted|rejected}`;
  - one `verify` event (model=`target_key`, `tokens=k+1`,
    `attended=Σ_{j=0..k}(ctx+j)`, `logits=k+1`, `inputs=[all draft ids this
    round]`), payload `{"correction"}`;
  - round handoff: next round's first draft `inputs` includes the prior verify.
- Accept `rounds` in the exact shape `SpecResult` already produces
  (`[{"drafts":[...], "num_accepted":int, "correction":...}]`).

**Tests** (reuse the fixture pattern from the existing
`tests/unit/test_task_graph.py::TestSpecDecodeTaskGraph`):
- `test_every_draft_fans_into_verify` (incl. rejected ones).
- `test_accepted_and_rejected_counts`.
- `test_rounds_are_handed_off`.
- `test_builds_into_valid_graph` and verify flops > draft flops.

**Commit:** `spec-decode tracer: rounds -> canonical trace`.

---

### Task 6 — Training tracer **(STUB)**  `demos/task-graph/tracers/training.py`

**Goal:** a canonical-format trace from *simulated* training data. Clearly a
stub; real `train_once` instrumentation is Task 12 (deferred — YAGNI for now).

**Files:** create `demos/task-graph/tracers/training.py`; create
`tests/unit/test_training_tracer.py`.

**Write:**
- `def trace_training_stub(model_key, cfg, max_steps, batch, seq_len,
  loss_trajectory, eval_steps, target_key=...) -> dict:`
  - `train_step` events chained, `mode="lora_bwd"`, `tokens=batch*seq_len`,
    `attended=batch*seq_len*(seq_len+1)//2`, `logits=batch*seq_len`,
    payload `{"loss"}`.
  - every `eval_steps`, an `eval` event (`mode="fwd"`) with `inputs=[that step]`,
    payload `{"metric", "checkpoint_digest"}`.
- Add a module docstring: **"STUB — simulated data; replace with Task 12."**

**Tests:** node/edge counts; evals branch off the right steps; a LoRA train_step
costs ≈2× a same-shape forward (compare to `flops.flops(...,mode="fwd")`).

**Commit:** `training tracer (stub): simulated run -> canonical trace`.

---

### Task 7 — Coding-agent tracer **(STUB)**  `demos/task-graph/tracers/coding.py`

**Goal:** canonical trace for the search→plan→codegen→verify diamond from a
captured/typed trace. Stub; a real agent is Task 13 (deferred).

**Files:** create `demos/task-graph/tracers/coding.py`; create
`tests/unit/test_coding_tracer.py`.

**Write:**
- `def trace_coding_stub(agent_key, agent_cfg, prompt, retrievals, plan,
  codegens, verify) -> dict:` — root `prompt` event → each retrieval
  (`search`/`fetch`) → `plan` → each `codegen` → `test`. Every node carries
  `tokens` and a real `context` length so attention is included (per the design
  decision: agent steps have long contexts, so attention is **not** dropped).
  Compute `attended = tokens * context` (estimate — document it as such).
- Docstring: **"STUB — hand-captured trace; replace with Task 13."**

**Tests:** root has no inputs; prompt fans to retrievals; retrievals fan into
plan; plan fans to codegens; codegens fan into test; every node `flops>0`;
includes an attention contribution (a node with `context>0` costs more than the
same node with `context=0`).

**Commit:** `coding tracer (stub): captured trace -> canonical trace`.

> **STOP for review #2** — all four scenarios now emit the canonical format.

---

### Task 8 — One generic renderer + bake script

**Goal:** replace the four bespoke renderers with **one** that draws any
canonical `Graph` as a layered DAG, and a script that builds all four traces into
the data the page embeds.

**Files:**
- create `demos/task-graph/build_all.py` — imports the four tracers, builds each
  trace → `build_graph` → `to_dict()`, writes
  `demos/task-graph/graphs.json` = `{"inference":..., "spec":..., "training":...,
  "coding":...}`, and bakes it into `demos/proof-compare/viz/index.html`
  (replace the `const DATA = ...;` line; use a `lambda` in `re.sub` to avoid
  backslash-escape errors with `\u` in JSON — there is a known footgun here).
- edit `demos/proof-compare/viz/index.html` — delete `renderInference`,
  `renderTraining`, `renderSpec`, `renderCoding`; add **one** `renderGraph(g,
  host, meta)` that:
  - assigns each node a **layer** = longest-path depth from a root (topological
    layers left→right); stacks nodes within a layer vertically;
  - draws edges from `g.edges`; colors nodes by `kind` (keep the existing color
    vars); node card shows `kind`, a short label, and `fmtFlops(flops)` with a
    cost bar (reuse existing `card`/`arrow`/`flopsBar`/`fmtFlops` helpers — do
    not reinvent them, DRY);
  - tooltip shows `payload` + cost.
  - the four tabs each call `renderGraph(DATA[key], ...)`.

**How to verify (no unit test for SVG; do a real check):**
- `python3 demos/task-graph/build_all.py` then
  `node --check` the page's `<script>` (extract it; the repo does this — see the
  bake step pattern). Then open the page or `curl` the local file and confirm all
  four `data-scene` blocks and the four DATA keys are present.
- Eyeball each tab renders without overlap. A layered layout must handle: a pure
  chain (inference), fan-in (spec verify), branch (training eval), diamond
  (coding). If two shapes look wrong, fix the layout, not the data.

**Commit (two commits ok):** `viz: single layered-DAG renderer` and
`task-graph: build_all bakes all four canonical graphs`.

---

### Task 9 — Delete the bespoke builders (DRY cleanup)

**Goal:** remove the now-dead code so there is exactly one of everything.

**Files:** `modules/proof_server/task_graph.py` — delete `build_task_graph`,
`build_training_task_graph`, `build_spec_decode_task_graph`,
`build_coding_agent_task_graph` and their node/edge dataclasses **iff** nothing
imports them anymore. Grep first: `grep -rn "build_task_graph\|build_training_task_graph\|build_spec_decode_task_graph\|build_coding_agent_task_graph" --include=*.py`.
Update/remove `tests/unit/test_task_graph.py` accordingly (the *behavioral*
coverage now lives in the per-tracer tests + `test_graph.py`). Keep
`MODEL_DIMS`/`forward_flops` **only** if still referenced; otherwise delete
(it's superseded by `flops.py`). Keep `demos/*/demo.sh` working — if a demo
imported a deleted builder, point it at the new tracer.

**Verify:** full suite green with `.venv/bin/python3`.
**Commit:** `task-graph: remove the four bespoke builders (superseded by build_graph)`.

> **STOP for review #3** — confirms the migration is complete and nothing dead
> remains.

---

### Task 10 — Wire the proof server to the canon (keep the demos runnable)

**Goal:** the spec-decode proof server currently calls
`build_spec_decode_task_graph`. Point it at `trace_spec_decode` + `build_graph`
so the live `demos/spec-decode/demo.sh` still produces a graph.

**Files:** `demos/spec-decode/servers/proof_server.py` (the `/compare` handler).
Read the model dims it needs from the manifest/config it already has.

**Verify:** `bash demos/spec-decode/demo.sh` (mock mode, no GPU) → `ALL PASS`
and a canonical graph JSON is written.
**Commit:** `spec-decode proof server: build graphs via build_graph`.

---

# Part 4 — The forcing function, publish, and PR

### Task 11 — Real GPU inference run (the forcing function)

**This is the milestone that proves the whole pipeline on real data.** Follow the
vast.ai procedure precisely; it is easy to waste money here.

**A. Add the GPU glue** (`demos/task-graph/capture/run_inference.py`):
- load a real HF model (start with `Qwen/Qwen3-1.7B`) + its tokenizer;
- `next_token(ids)` = greedy argmax of `model(ids).logits[0,-1]` (see
  `hf_models` in `demos/spec-decode/spec_decode.py` for the exact pattern —
  reuse it, do not rewrite);
- `shape_config` = `dict(model.config)` (so `flops.py` reads real dims);
- tokenize a prompt, call `trace_inference(...)`, `build_graph`, write the
  canonical graph JSON; decode token text into each node's `payload` via the
  tokenizer (this is where the real tokenizer replaces the old whitespace fake).

**B. Run it on vast.ai:**
1. Launch: pick the cheapest rentable H100,
   `vastai create instance <id> --image hiyouga/llamafactory:latest --disk 50 --ssh --direct`
   (that image already has torch + transformers).
2. Wait for `actual_status == running` (poll `vastai show instance <id> --raw`).
3. **Use the DIRECT ssh route, not the proxy.** Get it from the raw JSON:
   `public_ipaddr` + `ports["22/tcp"][0]["HostPort"]`. The `sshN.vast.ai` proxy
   frequently refuses connections — do not fight it.
4. `scp` a minimal bundle (the `modules/proof_server/*.py` you wrote +
   `demos/task-graph/`), or `git clone` the branch if the box has auth. **Ship a
   neutralized empty `modules/__init__.py`** so importing `modules.proof_server.*`
   doesn't trigger the heavy real `modules/__init__.py` (it imports a Pipeline
   with deps the box lacks). This bit the previous run.
5. Pre-download the model once before running (avoids a cold first call).
6. Run `run_inference.py`, `scp` the resulting graph JSON back to
   `demos/task-graph/traces/inference.real.json`.
7. **Destroy the instance** (`vastai destroy instance <id>`, confirm with `y`).
   Verify it's gone. Leaving it running costs ~$2/hr.

**C.** Point `build_all.py` at the real inference graph; re-bake; the inference
tab now shows a **real** run (real token ids, real config dims, exact FLOPs).

**Log** everything (instance id, cost, output text, FLOPs totals) in
`demos/task-graph/EXPERIMENT_LOG.md` (append-only; see the repo's experiment-log
convention).

**Commit:** `task-graph: real H100 inference run -> canonical graph (forcing function)`.

> **STOP for review #4** — confirm the real run is honest (real dims, real
> tokenizer, exact FLOPs) and the instance was destroyed.

### Task 12 / 13 — (deferred) make training & coding real

Out of scope for this pass (YAGNI). Leave the stubs, but file follow-ups:
- **12:** instrument `train_once` (`workflows/deterministic_lora_training.py`,
  the `for step in range(cfg["max_steps"])` loop) to emit per-step loss + an
  eval+checkpoint every `eval_steps`, then a real GPU LoRA run.
- **13:** a minimal real agent loop (or a small library) whose tool calls emit
  the coding trace with real token counts.

### Task 14 — Publish + PR

1. `python3 demos/task-graph/build_all.py` (re-bake with real inference data).
2. Extract the `<script>` and `node --check` it; confirm 4 tabs.
3. `cd demos/proof-compare/viz && surge ./ taskgraph-dss.surge.sh`. A `504`
   immediately after publish is a known transient surge edge blip — re-check
   until `200`.
4. Final full suite green (`.venv/bin/python3 -m unittest discover -s tests/unit`).
5. Commit, push, open the PR against base **`worktree-proof-server`** (this
   branch is stacked on it — do not target `main`):
   `gh pr ...`.

**The PR description must include:**
- **How FLOPs are calculated** — reproduce Part 2 (the exact per-matmul formula,
  GQA, SwiGLU `6df`, causal `attended`, LM head, the `mode` multipliers and why
  LoRA is ×2), and the §2.6 worked number.
- **The graph primitives** — the canonical `Event` fields and what each means;
  `inputs`→edges; `build_graph` as the single builder; the tracer→trace→builder→
  renderer pipeline; what's real (inference, spec-decode) vs stubbed (training,
  coding).
- Link the live viz and note the four shapes (chain / fan-in / branch / diamond)
  all come from one renderer.

---

## Appendix A — Command cheat-sheet

```
# tests (pure logic)         python3 -m unittest tests.unit.test_flops -v
# tests (full suite)         .venv/bin/python3 -m unittest discover -s tests/unit
# make the venv              uv sync
# bake the viz               python3 demos/task-graph/build_all.py
# js syntax check            (extract <script> to /tmp/x.js) node --check /tmp/x.js
# publish                    cd demos/proof-compare/viz && surge ./ taskgraph-dss.surge.sh
# spec-decode demo (no GPU)  bash demos/spec-decode/demo.sh
```

## Appendix B — Gotchas (these have already bitten us)

- `re.sub(pat, replacement, s)` treats `\u`/`\g` in *replacement* as escapes →
  baking JSON breaks. Use `re.sub(pat, lambda m: new, s)`.
- `json.tool | head` under `set -o pipefail` SIGPIPEs and aborts a script — put
  any file-copy/important step **before** truncated display, or append `|| true`.
- vast SSH **proxy** is flaky; use the **direct** `public_ipaddr:HostPort`.
- Loading two models on one GPU **concurrently** can abort the CUDA context;
  load sequentially.
- System `python3` lacks `pydantic`/`torch`; use `.venv/bin/python3` for the full
  suite.
- Do not add an AI co-author to commits (repo rule).

## Appendix C — Definition of done

- One `flops.flops(...)`, one `build_graph(...)`, one `renderGraph(...)`. No
  bespoke per-scenario builders or renderers remain.
- All four tabs render from canonical graphs; inference + spec-decode are from
  real runs; training + coding are labelled stubs.
- Full unit suite green. Live viz returns `200`. PR open with the FLOPs + graph
  primitives write-up.
