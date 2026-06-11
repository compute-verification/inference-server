# Coding-agent demo — design and implementation plan

A fourth task-graph scenario: a simple coding agent that **summarizes a paper
that just came out and implements it**. Chosen because a paper after the model's
training cutoff is the purest "not one-shottable without search" task — the
implementation literally can't exist without retrieving the paper.

## The graph shape (search → plan → codegen → verify)

```
 [search] ─┐
 [search] ─┤
 [fetch]  ─┼─► [plan: extract algorithm] ─┬─► [codegen: impl]  ─┐
 [fetch]  ─┤                              └─► [codegen: test]  ─┼─► [verify ✓]
 [fetch]  ─┤                                                    │
 [fetch]  ─┘                                                    ┘
```

Retrieval nodes fan **in** to one plan node; the plan fans **out** to codegen
nodes; codegen nodes fan **in** to a verify node. Three edge kinds: `informs`
(retrieval→plan), `plans` (plan→codegen), `verifies` (codegen→verify). Built by
`build_coding_agent_task_graph()` in `modules/proof_server/task_graph.py`.

## The captured run (real)

The agent (me) ran this for real — real `WebSearch`/`WebFetch` — and the trace
is captured in `coding_agent_graph.json`:

- **search** ×2 — for recent LLM sampling/decoding papers
- **fetch** ×4 — arXiv abstract + full HTML + the reference GitHub repo + the
  raw reference code, which surfaced **p-less sampling** (arXiv:2509.23234)
- **plan** — extract the algorithm: threshold = collision likelihood
  `L[P] = Σ P(v)^2` (`= exp(-H_2)`, Rényi-2 entropy); keep tokens with
  `P(v) ≥ L[P]`; renormalize; sample. Hyperparameter-free; the argmax always
  survives (`max P ≥ Σ P^2`), so the kept set is never empty.
- **codegen** ×2 — `generated/p_less.py` (pure-stdlib, CPU) + `tests/unit/test_p_less.py`
- **verify** — `python -m unittest tests.unit.test_p_less` → 9 passed

## Artifacts

- `generated/p_less.py` — the implemented algorithm (real, tested)
- `tests/unit/test_p_less.py` — the verify node (9 cases, CPU)
- `coding_agent_graph.json` — the captured task graph (rendered in the viz)
- `demo.sh` — prints the graph and re-runs the verify node for real

## Verify ≠ replay

The retrieval/plan/codegen nodes are a captured trace of a one-time real run (an
agent loop is non-deterministic, so it isn't re-executed). The **verify** node
is genuinely re-runnable (`demo.sh`), so the green check stays honest.
