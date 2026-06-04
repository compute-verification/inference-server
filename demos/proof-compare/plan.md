# Proof-compare demo — design and implementation plan

A "proof server" (compare + task-graph variant) that sits downstream of the
existing `demos/tap-protocol/` topology. **Distinct from `demos/proof-server/`**
(the SP1/Merkle-ledger proxy): this one does no SP1 and no ledger. It receives
*both* clusters' token responses for each request, bitwise-compares them, logs
MATCH/MISMATCH, and builds a **task graph** from each generation (stored, not
yet consumed).

`./demo.sh → ALL PASS` runs the full path locally in `--mock` mode (no GPU,
no SP1 binary).

## Topology

```
client ──► Gateway (8000)
              │
              ▼
            Tap (8010)          [PATCHED: optional --compare-server-url]
              │
              ├─────────────────► Host Cluster (8020)   [returns host_output]
              │
              └─── _async_verify ─► Recomp Cluster (8030)  [PATCHED: /verify
                       │              now also returns recomp_output]
                       │
                       └─── _async_compare ─► Proof Server (8050)  [NEW]
                              POST /compare {id, prompt,
                                             host_output, recomp_output}
```

The Tap forwards both outputs because the Recomp Cluster already holds the host
output (it recomputes against it) — extending `/verify` to return its own output
is a smaller change than teaching both clusters to fan out independently.

## Components

**Added:**
- `servers/proof_server.py` — `POST /compare` (compare + build graph),
  `GET /health` (counters: compared / matches / mismatches / graphs_built).
- `demo.sh` — host + recomp + tap + gateway + proof server, all `--mock`.
- `modules/proof_server/task_graph.py` — `Task`/`TaskGraph` dataclasses,
  `forward_flops()`, `build_task_graph()`, `MODEL_DIMS` lookup. Unit test:
  `tests/unit/test_task_graph.py`.

**Patched (additive, backward-compatible — existing proof-server demo unaffected):**
- `tap-protocol/servers/recomp_cluster.py` — `/verify` returns `recomp_output`.
- `tap-protocol/servers/tap.py` — `--compare-server-url` + `_async_compare`.

## Task graph

One `Task` = one forward pass = one emitted token. Task 0 is the prefill (whole
prompt at once, fat FLOPs); each later task is a decode step (1 token, thin
FLOPs), chained via `next`. FLOPs use the "option 2" estimate
(`2·n_params·tokens + 4·n_layers·d_model·context·tokens`). See the
`modules/proof_server/task_graph.py` docstring for the two deliberate
approximations (whitespace tokenization in mock mode; static `MODEL_DIMS`
lookup because the manifest pins `config.json` by digest, not dims).

## Out of scope (for now)

Nothing consumes the task graph yet — building and storing it is the deliverable.
Real tokenization, real model dims from `config.json`, prefill chunking, and any
graph scheduler/visualizer are future work.
