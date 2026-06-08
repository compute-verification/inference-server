# Plan: Unified, auto-generated task graphs

**Status:** ready for implementation (v2 — incorporates adversarial review)
**Branch to work on:** `proof-compare-graph` (already checked out at
`/home/jon/projects/dss-proof-compare`). Do **not** start a new branch.
**Audience:** an engineer who is new to this codebase and to the problem domain.
Read Part 0 fully before writing any code.

---

## How to use this document

- Do the tasks **in order**. Each task is one (sometimes two) commits.
- Every code task is **test-first** (write the failing test, then the code),
  **except** Task 8 (SVG renderer) which is verified by eyeball — that is called
  out explicitly there.
- **Commit after every green task.** Imperative one-line summary + short body.
  **Do not add Claude/AI as a co-author** (repo rule).
- When a task says **STOP for review**, stop and wait.
- If something here is wrong or impossible, **stop and say so** — do not
  improvise a different design.

### Honesty up front (what "auto-generated" means after this pass)

After this work, **two of four** scenarios are generated from real model runs:
**inference** (a real GPU run — the forcing function) and **spec-decode**
(ported from its existing real GPU run). **Training** and **coding** remain
**stubs** — canonical-format traces from simulated / hand-captured data —
clearly labelled as such, with real capture deferred to Tasks 12/13. Do not
describe training/coding as real. (The coding task was performed for real
once, but its trace was hand-captured, not auto-emitted; that's why it's a stub.)

---

# Part 0 — Background (read this first)

## 0.1 What this project is

Bitwise-deterministic LLM inference research. You do **not** need the
determinism machinery. You work on a **visualization**: we represent LLM
workloads (inference, LoRA training, speculative decoding, a coding agent) as
**task graphs** and render them on a web page. Goal: make those graphs
**auto-generated through one shared pipeline**, replacing four hand-built ones.

## 0.2 Domain primitives

1. **Forward pass:** one run through the `L`-layer transformer. Costs a
   computable number of **FLOPs** depending only on model shape + tokens pushed
   through (Part 2).
2. **Token / tokenizer:** text ↔ integer ids.
3. **Inference = prefill then decode.** Prefill reads the whole prompt (`P`
   tokens) in one pass (work ∝ `P`, attention ∝ `P²`). Decode then emits one
   token per pass, each attending over the whole context so far. `N` output
   tokens = 1 prefill + `N` decode passes — a **chain**.
4. **LoRA training:** base weights **frozen**, only small adapters trained. One
   **step** = forward + backward + update over a batch. We **eval** periodically
   (run the current model on held-out data) — evals branch off the chain, and an
   eval is itself a little inference (prefill + decode).
5. **Speculative decoding:** a small **draft** model proposes `k` tokens; the big
   **target** model **verifies** all `k` in one pass, keeps the longest correct
   prefix, emits one correction. Rejected drafts are discarded **but were still
   fed into the verify pass** (they're real dependencies).
6. **Coding agent:** to implement a paper it does web **search**/**fetch**, forms
   a **plan**, writes code (**codegen**), runs tests (**verify**).

**Unifying idea:** each is a graph of **tasks**; every task has an **input**, a
**cost (FLOPs)**, an **output**, and **edges** to tasks that consume it.

## 0.3 Toolset & conventions (unusual — read carefully)

- Python, stdlib-first. **Tests use `unittest`, NOT pytest** (`tests/unit/`,
  `test_*.py`, `unittest.TestCase`).
- **Two interpreters:** `python3` (stdlib only — pure-logic tests);
  `.venv/bin/python3` (has `pydantic`+`torch` — run the **full** suite). Make the
  venv with `uv sync` if missing. Use **`uv`**, never pip/apt.
- Run: `python3 -m unittest tests.unit.test_flops -v`; full:
  `.venv/bin/python3 -m unittest discover -s tests/unit`.
- Repo-root in a test: `Path(__file__).resolve().parents[2]`.
- **Canonical JSON:** `json.dumps(d, sort_keys=True, separators=(",", ":")) + "\n"`.
  (Note: `payload` floats like loss/metric are **display-only** — not part of any
  determinism claim.)
- The web page is one self-contained file:
  `demos/proof-compare/viz/index.html`. Graph data is **baked inline** into a
  `const DATA = {...};` line by a script you will write (Task 8). Published with
  **surge** (`surge ./ taskgraph-dss.surge.sh`, already authed). No build step,
  no CDN — keep it one file.
- GPU runs: **vast.ai** (`vastai`, authed). Exact procedure in Task 11.

## 0.4 Test design primer (your tests WILL be reviewed)

1. **Test behavior/guarantees, not internals.**
2. **One idea per test**; name it as a sentence.
3. **Cover normal + boundary (empty/zero/one) + invariant (always-true
   property).**
4. **Deterministic:** no clocks, no unseeded RNG, **no network in unit tests**.
5. **Hand-compute expected FLOPs** in a comment; never paste the function's own
   output back as "expected" (circular).
6. **Small explicit fixtures**; never load real models in unit tests — mock the
   model with a deterministic `next_token` function.
7. **Test failure paths** (`assertRaises`).

## 0.5 What exists today (and why it's "wrong")

Four bespoke builders in `modules/proof_server/task_graph.py`
(`build_task_graph`, `build_training_task_graph`,
`build_spec_decode_task_graph`, `build_coding_agent_task_graph`), four bespoke
renderers in `demos/proof-compare/viz/index.html`, four node dataclasses, and
two different cost conventions (three use approx `2N+attn`; coding uses
weight-only). We collapse to **one** cost fn, **one** builder, **one** renderer,
and drive them from tracers.

**Read before starting (~20 min):** `modules/proof_server/task_graph.py` (all
four builders, esp. `forward_flops`, `MODEL_DIMS`, the `eval_graph` nesting in
`build_training_task_graph`, and the `SpecEdge`/`CodingEdge` `kind` fields);
`demos/spec-decode/spec_decode.py` (`greedy_decode`, `hf_models`,
`SpecResult`/`SpecRound`, `to_response`); `demos/proof-compare/viz/index.html`
(the `render*` functions + `card`/`arrow`/`flopsBar`/`fmtFlops` helpers + the
baked `const DATA`); `demos/proof-compare/servers/proof_server.py` and
`demos/spec-decode/servers/proof_server.py` (both call builders today).

---

# Part 1 — Target architecture

```
   real run ──[TRACER]──▶ trace (events) ──[build_graph]──▶ Graph ──[renderGraph]──▶ viz
   per scenario            ONE format        ONE builder      ONE renderer
                                                  │
                                                  ▼  flops.flops(...)  (exact cost)
```

### The canonical `Event` (the contract)

```python
@dataclass
class Event:
    id: int                # unique; assigned in creation order. inputs MUST be < id (DAG, no cycles)
    kind: str              # node type (see KINDS below)
    inputs: list[int]      # ids this depends on -> become edges (src=input, dst=this)
    model: str             # key into the trace's `shapes` table
    tokens: int = 0        # tokens this event processes
    attended: int = 0      # total (token,key) attention pairs (Part 2 §2.3); 0 => no attention term
    mode: str = "fwd"      # "fwd" | "lora_bwd"
    logits: int = 0        # number of positions taking an LM-head projection
    status: str = ""       # optional: e.g. "accepted"|"rejected" (drives node/edge styling)
    label: str = ""        # short human title
    payload: dict = {}     # display extras: token text, loss, metric, file path, digest, ...
```

`KINDS` (a fixed vocabulary, single source of truth — define as a constant):
`prefill, decode, train_step, eval_prefill, eval_decode, draft, verify, prompt,
search, fetch, plan, codegen, test`.

A **trace** = `{"shapes": {model_key: <shape dict>}, "events": [<event dict>...]}`.

**`build_graph(trace)`** computes each event's `flops` via `flops.py`, turns
`inputs` into `Edge(src=input_id, dst=event_id)`, validates the DAG, and returns
a `Graph`. **No scenario-specific logic lives in `build_graph`.** Edges have no
`kind` field — the **renderer** styles edges by looking at the source/target
node `kind`+`status` (presentation concern, kept out of the builder). This is how
we preserve spec-decode's rejected-draft / fan-in styling without scenario code
in the builder.

### Where model shapes come from (resolves a real hole)

Two sources, by context:
- **Real GPU run (inference, Task 11):** read the live `model.config.to_dict()`
  → exact dims. This is the run that *proves* the cost model on real hardware.
- **Everywhere else (spec-decode proof server, stubs, unit tests):** a static
  `KNOWN_SHAPES` table in `flops.py` keyed by model id (`"Qwen/Qwen3-1.7B"`,
  `"Qwen/Qwen3-0.6B"`, `"agent"`), each value a plain config dict. These contexts
  have no GPU/live config; the static table is fine because the *exactness*
  demonstration is the inference forcing function, and the other graphs are about
  *shape*, not a hardware claim. Document this tradeoff in `KNOWN_SHAPES`'
  docstring.

---

# Part 2 — The exact FLOPs specification

One multiply-accumulate = **2 FLOPs**; matmul `(m×k)·(k×n)` = `2mkn`.

### 2.1 Model shape (from `config.json` / `model.config.to_dict()`)

```
L = num_hidden_layers   d = hidden_size      h = num_attention_heads
d_h = head_dim (else d//h)   h_kv = num_key_value_heads (else h)
f = intermediate_size   V = vocab_size
```

### 2.2 Per-layer weight matmuls (per token)

```
W_layer = 4·d·h·d_h        # Q proj (2·d·h·d_h) + O proj (2·h·d_h·d)
        + 4·d·h_kv·d_h      # K and V projections — GQA makes these smaller
        + 6·d·f             # gated MLP / SwiGLU: gate + up + down  (NOT 4·d·f)
```

### 2.3 Attention (activation×activation)

Per token attending over `s` keys: `4·h·d_h·s` per layer (`QKᵀ` + `softmax·V`).
**Important:** this scales with the **query**-head count `h`, *not* `h_kv` — GQA
shrinks the KV projections (§2.2) and KV cache, but **not** the QKᵀ/AV FLOPs.
Define **`attended`** = Σ of `s` over all tokens in the event:
- decode of 1 token at sequence length `s`: `attended = s`.
- causal prefill of `P` tokens: `attended = P·(P+1)//2`.
- parallel verify over positions `ctx..ctx+k`: `attended = Σ_{j=0}^{k}(ctx+j)`.
Attention FLOPs `= L · 4·h·d_h · attended`.

### 2.4 LM head

Each logit position costs `2·d·V`. `logits` = count of such positions (1 for a
generated/draft token; `k+1` for a verify; `batch·seq` for a training step).

### 2.5 The single cost function

```
WEIGHT_MULT = {"fwd": 1, "lora_bwd": 2}   # base weights: fwd; LoRA backward = fwd + dX (frozen -> no dW)
ATTN_MULT   = {"fwd": 1, "lora_bwd": 3}   # attention matmuls have no weights; both operands
                                          # are activations needing grads -> fwd + 2 bwd

flops(shape, tokens, attended, mode, logits) =
      WEIGHT_MULT[mode] · ( tokens · L · W_layer  +  logits · 2·d·V )
    + ATTN_MULT[mode]   · ( L · 4·h·d_h · attended )
```

- Forward inference: `mode="fwd"`.
- LoRA step: `mode="lora_bwd"`. **Why "~2×":** the **weight** term (which
  dominates — attention is <1% at these sizes) is ×2; the small attention term is
  ×3. So a real LoRA step is `2× + ε`, *not exactly* 2× — see the test in Task 1.
- We deliberately **do not** implement `full_bwd` (no caller — YAGNI). Validate
  `mode in WEIGHT_MULT` else `ValueError`.

**Excluded (comment this):** elementwise ops (RMSNorm, RoPE, softmax, SiLU,
residuals) are `O(tokens·d)`, non-matmul, <1%; LoRA adapter matmuls (rank `r≪d`)
are negligible. Do not add unless a later task asks.

### 2.6 Sanity anchor (a required test, hand-computed)

Qwen3-1.7B: `L=28, d=2048, h=16, d_h=128, h_kv=8, f=6144, V=151936`.
One decode token at `s=500`, `mode="fwd"`, `logits=1`:
- `W_layer = 4·2048·2048 + 4·2048·1024 + 6·2048·6144 = 100,663,296`; ×28 →
  `2,818,572,288`
- head: `2·2048·151936 = 622,329,856`
- attention: `28·4·16·128·500 = 114,688,000`
- **total = 3,555,590,144 FLOP** (use this exact integer in the test).

---

# Part 3 — Implementation tasks

> Per task: test first → fail → code → pass → run **full** suite with
> `.venv/bin/python3` → commit.
>
> **File layout (single home):** all new shared code under
> `modules/proof_server/` (`flops.py`, `graph.py`, `tracer.py`); all scenario
> code under `demos/proof-compare/` (the page + its server already live there):
> `demos/proof-compare/tracers/{inference,specdecode,training,coding}.py`,
> `demos/proof-compare/build_all.py`,
> `demos/proof-compare/traces/` (captured trace JSON),
> `demos/proof-compare/EXPERIMENT_LOG.md`. Do **not** invent a separate
> `demos/task-graph/` dir.

### Task 1 — `modules/proof_server/flops.py`  (+ `tests/unit/test_flops.py`)

**Write:**
- `@dataclass(frozen=True) class ModelShape:` `L,d,h,d_h,f,V,h_kv`.
- `model_shape_from_config(cfg: dict) -> ModelShape` reading §2.1 keys with
  fallbacks `d_h = cfg.get("head_dim", d//h)`,
  `h_kv = cfg.get("num_key_value_heads", h)`.
- `KNOWN_SHAPES: dict[str, dict]` with config dicts for `"Qwen/Qwen3-1.7B"`
  (the §2.6 numbers) and `"Qwen/Qwen3-0.6B"` (`L=28,d=1024,h=16,d_h=128,h_kv=8,
  f=3072,V=151936`) and `"agent"` (a documented ~32B stand-in, e.g.
  `L=64,d=5120,h=40,d_h=128,h_kv=8,f=27648,V=151936`). Docstring: explains these
  are for non-GPU contexts; the inference task overrides with live config.
- `W_LAYER(shape) -> int` and `flops(shape, tokens, attended, mode="fwd",
  logits=0) -> int` implementing §2.5 exactly. `ValueError` on bad mode.

**Tests (hand-computed; §0.4 rules 5):**
- `test_decode_token_matches_hand_count`: §2.6 → assert `== 3_555_590_144`.
- `test_mlp_term_uses_six_d_f_not_four`: tiny shape, `attended=0, logits=0,
  tokens=1, L=1` → expected `== W_LAYER` hand-computed; comment shows the `6·d·f`
  sub-term (you assert the full `W_LAYER`, the comment documents the 6·d·f).
- `test_gqa_reduces_kv_projection_only`: with `attended=0` (no attention),
  smaller `h_kv` → fewer FLOPs; **and** a second assert that with `tokens=0,
  logits=0` but `attended>0`, changing `h_kv` does **not** change FLOPs (attention
  uses `h`, not `h_kv`).
- `test_lora_weight_term_is_double_forward`: `attended=0, logits=1` →
  `flops(mode="lora_bwd") == 2*flops(mode="fwd")`.
- `test_lora_full_step_is_just_over_2x`: realistic train shape
  (`tokens=batch·seq`, `attended=batch·seq·(seq+1)//2`, `logits=batch·seq`) →
  `2.0 < lora/fwd < 2.05` (proves attention is the small ×3 correction, not 2×).
- `test_attention_scales_linearly_with_attended`.
- `test_bad_mode_raises` ; `test_zero_tokens_zero_attended_is_zero` (boundary).

**Run:** `python3 -m unittest tests.unit.test_flops -v`.
**Commit:** `flops: exact transformer FLOPs from real model shape`.

---

### Task 2 — `modules/proof_server/graph.py`  (+ `tests/unit/test_graph.py`)

**Write:**
- `KINDS` constant (the list in Part 1).
- `@dataclass class Event:` exactly Part 1 (with the defaults; `payload`
  defaulting to `None` then normalized to `{}` in `__post_init__` — never a
  mutable default arg).
- `@dataclass class Edge:` `src, dst` (no kind).
- `@dataclass class Graph:` `nodes: list[dict]`, `edges: list[Edge]`,
  `shapes: dict`; `to_dict()`, `to_json()` (canonical JSON).
- `build_graph(trace) -> Graph`:
  - resolve each event's shape: `model_shape_from_config(trace["shapes"][ev.model])`;
  - `flops = flops.flops(shape, ev.tokens, ev.attended, ev.mode, ev.logits)`;
  - node dict = event fields + `flops`;
  - edges from `inputs`;
  - **validate (raise `ValueError`):** unique ids; every input id exists; **every
    input id `< event.id`** (acyclic, no forward refs — the renderer's
    longest-path layout assumes a DAG); `ev.model in trace["shapes"]`;
    `ev.kind in KINDS`.

**Tests:** builds-nodes-with-correct-flops; inputs-become-edges;
rejects-dangling-input; rejects-duplicate-ids; **rejects-forward-reference**
(input id ≥ event id → raises); rejects-unknown-kind; round-trips-canonical-json
(ends with `\n`); empty-trace-empty-graph (boundary).

**Commit:** `graph: canonical Event/Graph + single build_graph()`.

---

### Task 3 — `modules/proof_server/tracer.py`  (+ `tests/unit/test_tracer.py`)

**Write:** `class Tracer:` with `add_shape(key, cfg)`, auto-incrementing
`event(kind, *, inputs=(), model="", tokens=0, attended=0, mode="fwd", logits=0,
status="", label="", payload=None) -> int`, and `trace() -> dict`. **No FLOPs
math here** (DRY — that's the builder's job).

**Tests:** ids increment from 0; a 3-event chain `trace()` → `build_graph` → 3
nodes/2 edges; shapes carried through.

**Commit:** `tracer: thin canonical-event recorder`.

> **STOP for review #1** — the canon (flops + graph + tracer) is the foundation.

---

### Task 4 — Inference tracer  `demos/proof-compare/tracers/inference.py`

**Write:** `trace_inference(prompt_ids, next_token, model_key, shape_cfg,
max_tokens) -> dict` where `next_token(ids)->int` is any deterministic fn.
- one `prefill` event: `tokens=len(prompt_ids)`, `attended=P*(P+1)//2`,
  `logits=1`, `inputs=[]`.
- per generated token, a `decode` event: `tokens=1`, `attended=current_seq_len`,
  `logits=1`, `inputs=[prev_id]`, `payload={"token_id": t}`.
- uses `Tracer`; `add_shape(model_key, shape_cfg)`.
- a `mock_next_token(seq)` helper for tests (e.g. `seq[-1]+1`).

**Tests:** 1 prefill + N decode for `max_tokens=N`; decode chain linked
(`inputs`=prev); prefill `attended == P*(P+1)//2`; builds into a valid graph and
prefill flops > one decode's flops.

**Commit:** `inference tracer: decode run -> canonical trace (model-agnostic)`.

---

### Task 5 — Spec-decode tracer  `demos/proof-compare/tracers/specdecode.py`

**Read first:** `build_spec_decode_task_graph` (the `ctx += a+1` recurrence and
fan-in topology) and `to_response` (rounds carry **token text**, not ids).

**Write:** `trace_spec_decode(prompt_len, rounds, draft_key, target_key) -> dict`
where `rounds` is the existing shape `[{"drafts":[str], "num_accepted":int,
"correction":str}]`. Look up shapes via `KNOWN_SHAPES[draft_key]` /
`[target_key]` (no live config here). **Thread context across rounds:**
```
ctx = prompt_len
for each round r with k = len(drafts), a = num_accepted:
    draft i (i=0..k-1): kind="draft", model=draft_key, tokens=1, attended=ctx+i,
        logits=1, status="accepted" if i<a else "rejected",
        inputs=[prev draft this round]  (first draft's inputs = [prior verify] or [])
        payload={"token": drafts[i]}
    verify: kind="verify", model=target_key, tokens=k+1,
        attended=sum(ctx+j for j in 0..k), logits=k+1,
        inputs=[all k draft ids this round], payload={"correction": correction}
    ctx = ctx + a + 1
```

**Tests** (mirror `TestSpecDecodeTaskGraph`): every draft (incl. rejected) has an
edge into its round's verify; accepted/rejected counts via `status`; rounds
handed off (round r+1 first draft depends on round r verify); `ctx` grows so a
later round's draft `attended` > an earlier round's (proves the recurrence);
builds into a valid graph; verify flops > draft flops.

**Commit:** `spec-decode tracer: rounds -> canonical trace`.

---

### Task 6 — Training tracer **(STUB)**  `demos/proof-compare/tracers/training.py`

**Key design point (from review):** an eval is a real little inference, so
**flatten** it into `eval_prefill` + `eval_decode` events linked into the trace
(do **not** embed a sub-graph in a payload — the canonical model has no nesting;
making the eval real events *is* the unification, and the layered renderer draws
it as a branch).

**Write:** `trace_training_stub(model_key, max_steps, batch, seq_len,
loss_trajectory, eval_steps, eval_prompt_len=8, eval_gen=3) -> dict`:
- `train_step` events chained, `mode="lora_bwd"`, `tokens=batch*seq_len`,
  `attended=batch*seq_len*(seq_len+1)//2`, `logits=batch*seq_len`,
  `payload={"loss": loss_trajectory[s]}`.
- every `eval_steps` steps: an `eval_prefill` (`mode="fwd"`,
  `inputs=[that train_step]`, `payload={"checkpoint_digest": "sha256:...",
  "metric": ...}`) followed by `eval_gen` `eval_decode` events chained off it.
- Module docstring: **"STUB — simulated data; real version is Task 12."**
- Shapes from `KNOWN_SHAPES[model_key]`.

**Tests:** train_step count == max_steps; eval branches start at the right steps;
each eval is `1 eval_prefill + eval_gen eval_decode` linked off a train_step; a
`lora_bwd` train_step costs `2.0× < ratio < 2.05×` a same-args `fwd` (reuse the
Task 1 fact); builds into a valid graph.

**Commit:** `training tracer (stub): simulated run -> canonical trace`.

---

### Task 7 — Coding tracer **(STUB)**  `demos/proof-compare/tracers/coding.py`

**Write:** `trace_coding_stub(agent_key, prompt, retrievals, plan, codegens,
verify) -> dict` — root `prompt` event → each retrieval (`search`/`fetch`) →
`plan` → each `codegen` → `test`. Each non-root spec: `{tokens, ...}`; `inputs`
wired per the diamond. **Set `attended=0` for stub nodes** (weight-only, matching
today's coding cost) — do **not** fabricate a `context`/attention number; the
real tracer (Task 13) will supply real context lengths. Shapes from
`KNOWN_SHAPES["agent"]`. Docstring: **"STUB — hand-captured trace; real version
is Task 13; attention omitted until real context is available."**

**Tests:** root `prompt` has no inputs; prompt→retrievals; retrievals→plan;
plan→codegens; codegens→test; every node `flops>0`; builds into a valid graph.

**Commit:** `coding tracer (stub): captured trace -> canonical trace`.

> **STOP for review #2** — all four scenarios emit the canonical format.

---

### Task 8 — One renderer + bake script  (verified by eyeball — no SVG unit test)

Split into **8a** (testable) and **8b** (renderer, eyeball).

**8a — `demos/proof-compare/build_all.py`:**
- import the four tracers; build each trace; `build_graph`; collect
  `{"inference":..., "spec":..., "training":..., "coding":...}` of `to_dict()`s;
  write `demos/proof-compare/traces/graphs.json`.
- **bake** into `index.html`: read it, replace the single line matching
  `r'const DATA = .*?;(?=\n)'` using a **lambda** replacement
  (`re.sub(pat, lambda m: "const DATA = "+json.dumps(data)+";", html, count=1)`)
  — a plain-string replacement breaks on `\u` in JSON (known footgun, Appendix B).
- a tiny test `tests/unit/test_build_all.py` that calls the build function on a
  small fake trace set and asserts: the `const DATA` line is replaced and
  `json.loads` of the embedded object has the 4 keys. (Tests the bake logic, not
  SVG.)
- **Commit:** `build_all: build + bake all four canonical graphs`.

**8b — generic renderer in `index.html`:** delete `renderInference`,
`renderTraining`, `renderSpec`, `renderCoding`; add **one** `renderGraph(g,
host, meta)`:
- **layout:** layer each node by **longest path from a root** (root = node with
  no incoming edge); place layers left→right, stack same-layer nodes vertically.
  This draws a chain (inference), a fan-in (spec verify), a branch (training
  eval), and a diamond (coding) from the same code. Target training layout:
  ```
  step0 ─ step1 ─ step2 ─ step3 ─ ...        (spine, one per layer)
            └ eval_prefill ─ eval_decode ─ … (branch hangs at step's layer+1, stacked below)
  ```
- **nodes:** reuse `card`/`flopsBar`/`fmtFlops` (DRY). Color by `kind` (keep the
  existing color vars); if `node.status=="rejected"` dim it + mark ✗.
- **edges:** reuse `arrow`; style by the **source/target node kind+status** (the
  builder gives plain edges — styling lives here): rejected-draft source → red
  dashed; `draft→verify` → faint thin (fan-in); else solid in the kind's color.
- tooltip shows `payload` + `fmtFlops(flops)` + `tokens`.
- the four tabs each call `renderGraph(DATA[key], …)`.

**Verify (do it, don't skip):** `python3 demos/proof-compare/build_all.py`;
extract the `<script>` to `/tmp/x.js` and `node --check /tmp/x.js` (node is at
`/usr/bin/node`; if absent, skip and rely on browser). Open the page; confirm all
four tabs render without overlapping nodes/edges and the spec-decode tab still
shows rejected drafts distinctly. If a shape looks wrong, fix the **layout**.
- **Commit:** `viz: single layered-DAG renderer for all scenarios`.

---

### Task 9 — Migrate both proof servers to the canon (BEFORE deleting builders)

Both `demos/spec-decode/servers/proof_server.py` (calls
`build_spec_decode_task_graph`) and `demos/proof-compare/servers/proof_server.py`
(calls `build_task_graph`) must move to `tracer → build_graph` first, or Task 10's
delete breaks the demos.
- spec-decode server `/compare`: build the spec trace from its `rounds` via
  `trace_spec_decode(prompt_len, rounds, "Qwen/Qwen3-0.6B", "Qwen/Qwen3-1.7B")`
  then `build_graph`. (Shapes come from `KNOWN_SHAPES`; the server only has
  source strings — this is exactly why `KNOWN_SHAPES` exists.)
- proof-compare server: it builds an *inference* graph from `(prompt, output)`
  text. Tokenize approximately (it has no tokenizer) → ids, then `trace_inference`
  with a precomputed id list... **simpler:** give `trace_inference` an optional
  path that accepts already-known `prompt_ids`/`output_ids`; here pass
  whitespace-split lengths as a coarse stand-in **and label the node payload
  `approx=true`** (this server is the mock-mode demo, not the real run).
- **Verify:** `bash demos/spec-decode/demo.sh` and
  `bash demos/proof-compare/demo.sh` → both `ALL PASS`.
- **Commit:** `proof servers: build graphs via build_graph`.

---

### Task 10 — Delete the four bespoke builders + renderers (DRY)

Now that nothing live calls them. **Grep for EACH symbol** (not just the four
builder names): `build_task_graph`, `build_training_task_graph`,
`build_spec_decode_task_graph`, `build_coding_agent_task_graph`, `forward_flops`,
`train_step_flops`, `dims_for`, `MODEL_DIMS`, `DEFAULT_DIMS`, `ModelDims`,
`EvalPoint`, and the node/edge dataclasses (`Task`, `TaskGraph`, `TrainNode`,
`SpecNode`, `SpecEdge`, `CodingNode`, `CodingEdge`, ...). For each with zero
remaining non-test importers, delete it. Confirm `modules/proof_server/api.py`
does **not** re-export any of them. Update/remove `tests/unit/test_task_graph.py`
(behavioral coverage now lives in `test_flops`, `test_graph`, and the per-tracer
tests). Keep `test_spec_decode.py`, `test_p_less.py`.

**Verify:** full suite green; `grep` shows no remaining references.
**Commit:** `task-graph: remove the four bespoke builders/renderers (superseded)`.

> **STOP for review #3** — migration complete, nothing dead remains.

---

# Part 4 — Forcing function, publish, PR

### Task 11 — Real GPU inference run (the milestone)

**A. GPU glue `demos/proof-compare/capture/run_inference.py`:**
- load `Qwen/Qwen3-1.7B` + tokenizer (reuse the `_argmax` pattern from
  `hf_models` in `demos/spec-decode/spec_decode.py` — do not rewrite);
- `shape_cfg = model.config.to_dict()` (**not** `dict(model.config)` — that loses
  keys);
- `next_token(ids)` = greedy argmax of last-position logits;
- tokenize a prompt → `prompt_ids`; `trace_inference(...)`; `build_graph`; decode
  each `payload.token_id` to text via the tokenizer; write
  `demos/proof-compare/traces/inference.real.json`.

**B. vast.ai (follow exactly):**
1. Pick cheapest rentable H100; launch:
   `vastai create instance <id> --image hiyouga/llamafactory:latest --disk 50 --ssh`
   (**no `--direct` flag** — that's not a `create` flag). The llamafactory image
   has torch+transformers.
2. Poll `vastai show instance <id> --raw` until `actual_status=="running"`.
3. **Use the DIRECT route, not the proxy:** from the raw JSON take
   `public_ipaddr` + `ports["22/tcp"][0]["HostPort"]`; `ssh -p <hostport>
   root@<public_ipaddr>`. The `sshN.vast.ai` proxy often refuses connections.
4. `scp` a minimal bundle: `modules/proof_server/{flops,graph,tracer}.py` +
   `demos/proof-compare/tracers/` + `run_inference.py`, **plus an empty
   `modules/__init__.py`** so importing `modules.proof_server.*` doesn't trigger
   the heavy real `modules/__init__.py` (it imports a Pipeline with deps the box
   lacks — this bit the previous run).
5. Pre-download the model once (`AutoModelForCausalLM.from_pretrained(...)`)
   before the timed run.
6. Run `run_inference.py`; `scp` `inference.real.json` back.
7. **Destroy:** `vastai destroy instance <id>` (confirm `y`); verify it's gone
   (≈$2/hr if left running).

**C.** Point `build_all.py` at `inference.real.json`; re-bake. The inference tab
now shows a real run (real token ids, real config dims, exact FLOPs).
**Log** instance id, cost, output text, total FLOPs in
`demos/proof-compare/EXPERIMENT_LOG.md` (append-only).

**Commit:** `task-graph: real H100 inference run -> canonical graph (forcing function)`.

> **STOP for review #4** — confirm real dims (`model.config.to_dict()`), real
> tokenizer text in payloads, exact FLOPs, and that the instance was destroyed.

### Tasks 12 / 13 — (deferred, file as follow-ups; do NOT do now)
- **12:** instrument `train_once` (`workflows/deterministic_lora_training.py`, the
  `for step in range(cfg["max_steps"])` loop) to emit per-step loss + an
  eval+`hash_adapter_dir` checkpoint every `eval_steps`; real GPU LoRA run.
- **13:** a minimal real agent loop whose tool calls emit the coding trace with
  real token counts + context lengths (so attention is included).

### Task 14 — Publish + PR
1. `python3 demos/proof-compare/build_all.py` (re-bake with real inference data).
2. `node --check` the extracted `<script>`; confirm 4 tabs.
3. `cd demos/proof-compare/viz && surge ./ taskgraph-dss.surge.sh`. A `504`
   right after publish is a transient surge edge blip — re-check until `200`.
4. Full suite green (`.venv/bin/python3 -m unittest discover -s tests/unit`).
5. Push; open PR against base **`worktree-proof-server`** (this branch is stacked
   on it — **not** `main`).

**PR description must include:**
- **How FLOPs are calculated:** Part 2 verbatim-ish — per-matmul derivation, GQA
  (projections only, attention scales with `h`), SwiGLU `6df`, causal `attended`,
  LM head, the `fwd`/`lora_bwd` multipliers and why LoRA ≈2×, plus the §2.6
  worked number `3,555,590,144`.
- **Graph primitives:** the canonical `Event` fields and meaning; `inputs`→edges;
  `build_graph` as the single builder; the tracer→trace→builder→renderer
  pipeline; `KNOWN_SHAPES` vs live config; what's real (inference, spec-decode)
  vs stub (training, coding).
- Live viz link; note all four shapes come from one renderer.

---

## Appendix A — Command cheat-sheet
```
tests (logic)     python3 -m unittest tests.unit.test_flops -v
tests (full)      .venv/bin/python3 -m unittest discover -s tests/unit
make venv         uv sync
bake the viz      python3 demos/proof-compare/build_all.py
js syntax check   node --check /tmp/x.js     (extract <script> first)
publish           cd demos/proof-compare/viz && surge ./ taskgraph-dss.surge.sh
spec demo (CPU)   bash demos/spec-decode/demo.sh
```

## Appendix B — Gotchas (already bit us)
- `re.sub(pat, replacement, s)` treats `\u`/`\g` in *replacement* as escapes →
  baking JSON breaks. Use `re.sub(pat, lambda m: new, s)`.
- `json.tool | head` under `set -o pipefail` SIGPIPEs and aborts a script — do
  important copies **before** truncated display, or append `|| true`.
- vast SSH **proxy** is flaky → use the **direct** `public_ipaddr:HostPort`.
  There is **no `--direct` flag** on `vastai create`.
- `dict(model.config)` drops keys → use `model.config.to_dict()`.
- Loading two models on one GPU concurrently can abort the CUDA context → load
  sequentially.
- System `python3` lacks `pydantic`/`torch` → use `.venv/bin/python3` for the
  full suite.
- Never a mutable default arg (`payload={}`); normalize in `__post_init__`.
- No AI co-author in commits.

## Appendix C — Definition of done
- One `flops.flops`, one `build_graph`, one `renderGraph`. No bespoke
  per-scenario builders/renderers remain (grep-verified).
- All four tabs render from canonical graphs via one renderer; **inference +
  spec-decode are real runs**, **training + coding are labelled stubs**.
- Spec-decode rejected-draft / fan-in styling preserved.
- Full unit suite green. Live viz `200`. PR open with the FLOPs + primitives
  write-up.
