# workflows

Runnable compositions of the [modules](../modules/). A workflow is a single
readable file that composes the modules into one end-to-end task — instead of
describing a workload in prose or a bespoke bash script, you hand a colleague
the file and say "this is exactly what I ran."

| Workflow | Composes | Run |
|---|---|---|
| [`deterministic_inference_server.py`](deterministic_inference_server.py) | build + inference + network | `python3 workflows/deterministic_inference_server.py` |
| [`deterministic_lora_training.py`](deterministic_lora_training.py) | build + inference (+ LoRA workload) | `python3 workflows/deterministic_lora_training.py --dry-run` |
| [`verified_inference.py`](verified_inference.py) | inference + attestation (Freivalds) | `python3 workflows/verified_inference.py` |

## Setup

The mock path needs only a few small pure-Python packages (no GPU), pinned in
`uv.lock`:

```bash
uv sync
.venv/bin/python3 workflows/verified_inference.py --mode mock   # wiring check (not a determinism proof)
```

Workflows can be run from any directory (the default manifest resolves relative
to the repo, not your cwd). `--mode vllm` additionally needs `torch` + `vllm` on
an NVIDIA box — see [`scripts/demo.sh`](../scripts/demo.sh).

## How a workflow is built

All workflows lean on two things from `modules/`:

1. **`Pipeline`** — chains the four stages (`manifest.v1 → lockfile.v1 →
   run_bundle.v1 → verify_report.v1`) in a few lines.
2. **Module facades** — `modules.network`, `modules.inference`,
   `modules.attestation`, … for the pieces that aren't just "run the pipeline."

The skeleton every workflow follows:

```python
from modules import Pipeline
from modules.<module> import <verb>

def my_workflow(manifest_path, *, mode="vllm"):
    pipe = Pipeline.from_manifest(manifest_path).resolve().build()
    pipe.run(out_a, mode=mode).run(out_b, mode=mode)
    report = pipe.verify(report_out=..., summary_out=...)
    # ... compose additional modules ...
    return {"status": report["status"], ...}
```

## The workflows

### 1. `deterministic_inference_server.py`

**Composes:** build + inference + network.
**What it proves:** a model served under the deterministic stack produces a
*bitwise-reproducible* run (two independent runs compare `conformant`) **and**
emits *byte-identical egress frames* for the same payload.

```text
$ python3 workflows/deterministic_inference_server.py
verify status : conformant
egress frames : 1 (reproducible: True)
bundles in    : /tmp/det-serve-xxxxxxxx
```

**Use it when:** you want the canonical "deterministic serving" demo, or a
template for deploying a server whose outputs *and* network traffic are
reproducible. `--mode vllm` on a GPU box runs real inference through the same path.

### 2. `deterministic_lora_training.py`

**Composes:** build + inference (+ a LoRA training workload).
**What it does:** deterministically assembles the *training environment* — the c3
env vars + the exact hermetic closure digest the run would execute against +
the LoRA workload descriptor (the same `mixed_lora` workload the
prover-verifier demo uses). The GPU training loop is the clearly-marked
integration point; `--dry-run` prints the plan without touching a GPU.

```text
$ python3 workflows/deterministic_lora_training.py --dry-run
{
  "c3_env": {"CUBLAS_WORKSPACE_CONFIG": ":4096:8", "PYTHONHASHSEED": "0", "VLLM_BATCH_INVARIANT": "1"},
  "lora_workload": {"name": "mixed_lora", "definition": "demos/prover-verifier/scripts/workloads/mixed_lora.py", ...},
  "manifest": "tests/fixtures/positive/manifest.v1.example.json",
  "runtime_closure_digest": "sha256:3bfb14e6…"
}
```

**Use it when:** you want to reproduce or define a deterministic LoRA-training
run. The `--dry-run` plan is itself the shareable artifact — it pins the exact
environment a teammate must match.

### 3. `verified_inference.py`

**Composes:** inference + attestation.
**What it proves:** a reproducible run *plus* an independent correctness proof —
a Freivalds matmul attestation runs alongside the inference verify, so the run
ships with evidence the underlying compute was honest. Uses the pure-Python
stdlib attestation backend, so it needs no GPU.

```text
$ python3 workflows/verified_inference.py
run verify   : conformant
attestation  : passed
bundles in   : /tmp/verified-inf-xxxxxxxx
```

**Use it when:** reproducibility alone isn't enough and you want an attestation
that the matmuls were computed correctly.

## Conventions

- Each workflow is importable (a function you can call) **and** runnable (a CLI
  with `main()`), so it's both a library example and a script.
- Workflows default to `--mode vllm` (the real determinism path). `--mode mock`
  (or `--dry-run`) runs a no-GPU wiring check for CI/laptops — it is **not** a
  determinism proof (mock runs match by construction).
- GPU-only steps are clearly marked (see the `train()` integration point in the
  LoRA workflow).

## Adding a workflow

1. Create `workflows/<name>.py` with a core function (importable) **and** a
   `main(argv)` CLI. Keep it ~100 lines and self-contained.
2. Compose `modules.Pipeline` + the module facades. Default to `--mode vllm`;
   offer `--mode mock` (or a `--dry-run`) so it can also run without a GPU.
3. Mark any GPU-only step explicitly.
4. Add a mock-path smoke check to `tests/modules/` so CI keeps it honest.
5. Add a row to the table above.

All three workflows are smoke-tested in `tests/modules/` (mock / CPU-only) and
run end-to-end without a GPU.
