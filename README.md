# Inference Server

This repository demonstrates a highly reproducible inference server: deterministic builds, tokens, and packets. It is built by the [Compute Verification Project](https://github.com/compute-verification), a research nonprofit designing a protocol by which datacenters can demonstrate they are only running inference, without revealing secrets or requiring auditors to trust their hardware.

Given the same model weights, prompts, and config flags, two independent servers produce bitwise-identical token outputs — and because egress frames are constructed by a simulated userspace TCP/IP stack, the packets on the wire are reproducible too. On top of that determinism sits the verification tooling: a *prover* serves manifest-pinned workloads and commits to every token it emits, an auditor can replay any challenged position, and simulated network taps let a *verifier* check the observed traffic against those commitments — alongside matmul attestation and proof of secure erasure.

Licensed under [Apache-2.0](LICENSE).

> **Status: research prototype.** The determinism results below were produced manually on H100/GH200 instances across millions of tokens; the hosted CI covers the CPU-side surface (unit/integration tests, schema gates, lint), not the GPU determinism gates. This is not a production-hardened serving stack — expect rough edges.

## Capabilities & layout

The stack is organized **by function**. Each capability has a documented
interface ([`modules/`](modules/)). [`workflows/`](workflows/) is the recipe book
that composes them.

| Capability | What it does | Start here |
|---|---|---|
| [build](modules/build/) | Hermetic, reproducible runtime + OCI image | `nix build .#oci` |
| [inference](modules/inference/) | Bitwise-deterministic vLLM (the c3 config) | `modules/inference/` |
| [network](modules/network/) | Deterministic L2 egress frames | `modules.network.egress_frames(...)` |
| [memory](modules/memory/) | PoSE memory wipe + erasure attestation | `modules/memory/` |
| [attestation](modules/attestation/) | Matmul / token / replay verification | `modules/attestation/verifier`, `modules/attestation/freivalds` |
| [utils](modules/utils/) | Provisioning, replay server, helpers | `scripts/deploy/`, `scripts/lambda/lambda_cli.py` |

See the [capability map](modules/README.md). Design and implementation plans
live on the `experiments` branch.

### Repository layout

```
modules/                Capability layer — each module owns its code, plus shared core/ + Pipeline
  build/                Hermetic runtime: builder/ + lockfiles/ + nix/   (flake.nix + flake.lock live at root)
  inference/            Deterministic vLLM — the c3 config
    server/             Proxy server with POST/GET /manifest endpoint
    resolver/           Manifest + HF resolution -> lockfile
    runner/             Manifest + lockfile -> run bundle (mock or vLLM)
    capture/            Server capture log -> run bundle
    manifest/           Pydantic manifest model (typed validation)
    manifests/          Model manifests (Qwen3, Mistral-Large2, DBRX, Llama4-Scout, ... + multinode)
  network/              networkdet/ (sim TCP/IP frame construction) + native/libnetdet/ (DPDK transmit)
  attestation/          freivalds/, e2e/, proverdet/ + verifier/ (+ verifier_cli/server) + prover/
  memory/               PoSE memory wipe + erasure attestation (pose/ sub-package + api.py)
  utils/                Provisioning / replay helpers (re-exports core/common)
  core/                 Shared: common/ (canonical JSON, SHA256, schema validation, HF resolution)
                        + schemas/ (JSON Schema contracts: manifest, lockfile, run_bundle, verify_report, attestation/replay)
workflows/              Recipe book — runnable compositions of the modules
demos/                  End-to-end scenarios: e2e-audit (the scripts/demo.sh path), prover-verifier (the protocol demo). Research experiments live on the `experiments` branch.
scripts/deploy/         Lambda / vast / warden provisioning (utils-owned)
tests/conformance/      Spec conformance catalog + release blockers (read by CI)
flake.nix, flake.lock   Hermetic build entrypoint + pin (at root: src=self packages repo-wide code; callers invoke `.#`)
```

## Results

Reported results from a manual cross-server run on two independent NVIDIA GH200 480GB instances on Lambda Cloud — every cross-server comparison matched bitwise:

| Model | Type | Repeated | Diverse | Tokens |
|-------|------|----------|---------|--------|
| Qwen3-1.7B | Dense transformer | 20/20 match | 34/34 match | 1.6M |
| Qwen3-30B-A3B | Mixture of Experts | 20/20 match | 34/34 match | 2.0M |
| Mistral-7B-Instruct-v0.3 | Dense transformer | 20/20 match | 34/34 match | 2.0M |

Each chunk is 30,000 tokens of greedy decoding (temperature=0). Same container image on both servers, same seed, same config. The scripts used to produce these runs live under `experiments/single-node-determinism/` on the [`experiments` branch](../../tree/experiments).

## Architecture

```
                             Inference Server
 ┌──────────────────────────────────────────────────────────────────────┐
 │                                                                      │
 │  ┌──────────┐    ┌──────────┐    ┌──────────────────────────────┐    │
 │  │ Manifest │───>│ Resolver │───>│ Resolved manifest + Lockfile │    │
 │  │ (author) │    │          │    │ (pinned revisions, digests)  │    │
 │  └──────────┘    └──────────┘    └───────────────┬──────────────┘    │
 │                                                  │                   │
 │                                                  v                   │
 │  ┌────────────────────────────────────────────────────────────────┐  │
 │  │                    Nix Container Image                         │  │
 │  │  ┌──────────────────────────────────────────────────────────┐  │  │
 │  │  │ Proxy Server (modules/inference/server/main.py)          │  │  │
 │  │  │  POST /manifest ── validate schema                       │  │  │
 │  │  │                 ── verify GPU model, count, driver       │  │  │
 │  │  │                 ── verify model file digests             │  │  │
 │  │  │                 ── start vLLM with manifest settings     │  │  │
 │  │  │  GET  /manifest ── return active config + health         │  │  │
 │  │  │  POST /v1/...   ── proxy to vLLM + capture log           │  │  │
 │  │  └──────────────────────────┬───────────────────────────────┘  │  │
 │  │                             │                                  │  │
 │  │                             v                                  │  │
 │  │  ┌──────────────────────────────────────────────────────────┐  │  │
 │  │  │ vLLM 0.17.1 (VLLM_BATCH_INVARIANT=1, --enforce-eager)    │  │  │
 │  │  │  --model, --revision, --seed, --dtype,                   │  │  │
 │  │  │  --attention-backend, --max-model-len, ...               │  │  │
 │  │  │  (every manifest field passed as CLI flag or env var)    │  │  │
 │  │  └──────────────────────────────────────────────────────────┘  │  │
 │  └────────────────────────────────────────────────────────────────┘  │
 │                                                                      │
 │  ┌──────────┐    ┌──────────┐    ┌───────────┐    ┌──────────────┐   │
 │  │  Runner  │───>│ Capture  │───>│ Run Bundle│───>│   Verifier   │   │
 │  │(tokens,  │    │(request/ │    │(observ-   │    │(compare two  │   │
 │  │ logits,  │    │ response │    │ ables,    │    │ bundles via  │   │
 │  │ frames)  │    │ logging) │    │ frames,   │    │ comparison   │   │
 │  │          │    │          │    │ provenance│    │ config)      │   │
 │  └──────────┘    └──────────┘    └───────────┘    └──────────────┘   │
 └──────────────────────────────────────────────────────────────────────┘
```

## Quick start

Bring up an NVIDIA H100 instance with the standard CUDA 12.8 AMI (Lambda Cloud's `gpu_1x_h100_sxm5` and `gpu_1x_h100_pcie` work as-is; GH200 also works), then:

```bash
git clone https://github.com/compute-verification/inference-server
cd inference-server
./scripts/demo.sh
```

`scripts/demo.sh` builds a venv (cu128 torch + vLLM 0.17.1), resolves the audit-enabled smoke manifest at `demos/e2e-audit/scripts/smoke.manifest.json` (declares H100 hardware, Qwen3-1.7B, 2 short prompts), starts the deterministic server, and runs the audit replay loop:

1. `POST /run` — server runs the manifest's requests and returns per-output-token HMAC commitments
2. `POST /replay` at random token positions — server re-runs each request truncated to the challenged position and recomputes the commitment
3. Negative test — a forged commitment must not match

Expected output ends with `ALL PASS`. Total wall time from `git clone` to `ALL PASS`: ~3 minutes (~90s pip install, <5s resolver/builder, ~30s vLLM model load, ~10s audit).

Requirements:
- NVIDIA GPU with compute capability ≥ 9.0 (H100, GH200, etc.) — batch invariance kernels need this
- ~5 GB free GPU memory (Qwen3-1.7B in bf16)
- Outbound internet for the Hugging Face download

### No GPU? Run the mock pipeline (wiring check)

Install the small CPU-only deps, then run the artifact spine on the mock backend —
a wiring check (no model download, no network), **not** a determinism proof:

```bash
uv sync   # installs the pinned CPU/test deps from uv.lock

tmp=$(mktemp -d)
.venv/bin/python3 modules/inference/resolver/main.py --manifest modules/inference/manifests/qwen3-1.7b.manifest.json \
  --lockfile-out $tmp/lock.json
.venv/bin/python3 modules/build/builder/main.py --lockfile $tmp/lock.json --lockfile-out $tmp/built.json
.venv/bin/python3 modules/inference/runner/main.py --manifest modules/inference/manifests/qwen3-1.7b.manifest.json \
  --lockfile $tmp/built.json --out-dir $tmp/run --mode mock
# Produces a run bundle with tokens, logits, and deterministic network frames.
# (Add --resolve-hf to the resolver to re-resolve revisions against live HF; needs network + huggingface_hub.)
```

Or compose the same spine in a few lines via a recipe — see
[`workflows/`](workflows/).

## How It Works

**Manifest** declares the full workload: model (pinned to HF commit SHA), runtime config (seed, dtype, attention backend, batch invariance), hardware requirements, requests, and comparison criteria.

**Resolver** pins everything to immutable references: resolves HF revisions, enumerates model files with per-file SHA256 digests, produces a lockfile.

**Nix container** pins the entire software stack: vLLM, PyTorch, CUDA toolkit, Triton, all Python deps. Same flake = same container = same behavior on any machine.

**Server** validates the manifest against the runtime (GPU model/count, driver version, CUDA version, model file digests), then starts vLLM with every manifest field passed as a CLI flag or env var.

**Runner** generates a run bundle containing tokens, logits, and deterministic L2 network frames (constructed by a simulated TCP/IP stack from the inference output).

**Verifier** compares two run bundles using the manifest's comparison config (exact match for tokens, tolerance for logits, SHA256 for network egress).

## What Makes It Deterministic

| Layer | How |
|-------|-----|
| **Software** | Hermetic Nix container — identical binary on every machine |
| **Model weights** | HF commit SHA pinned, per-file SHA256 verified before serving |
| **CUDA/cuBLAS** | `CUBLAS_WORKSPACE_CONFIG=:4096:8`, `VLLM_BATCH_INVARIANT=1` |
| **Attention** | `--enforce-eager` (no CUDA graphs), fixed attention backend |
| **Scheduling** | Greedy decoding (temperature=0), fixed seed |
| **Network frames** | Simulated TCP/IP stack with fixed MSS segmentation, software checksums, no offloads |

### Workflows & the Pipeline

Every recipe walks the same **artifact spine** — four stages, four named artifacts:

```
manifest.v1 ──resolve──▶ lockfile.v1 ──build──▶ + closure digest
                                                       │
                                                       └──run──▶ run_bundle.v1
                                                                       │
                                                                       └──verify──▶ verify_report.v1
```

Each stage already exists as a standalone CLI (`modules/inference/resolver/main.py`,
`modules/build/builder/main.py`, etc.). [`modules.Pipeline`](modules/pipeline.py)
is a thin fluent wrapper that chains them in-process — each method calls the
same code the per-stage CLI runs, holds the intermediate artifact on the object,
and returns `self`:

```python
from modules import Pipeline
report = (Pipeline.from_manifest("modules/inference/manifests/qwen3-1.7b.manifest.json")
          .resolve()        # -> lockfile.v1        (pins HF revisions + per-file digests)
          .build()           # -> + closure digest   (pins the Nix runtime)
          .run("/tmp/a")    # -> run_bundle.v1      (tokens, logits, network frames)
          .run("/tmp/b")    # -> run_bundle.v1      (independent run)
          .verify())         # -> verify_report.v1   ("conformant" iff identical)
assert report["status"] == "conformant"
```

A **workflow** in [`workflows/`](workflows/) is just a Python file that uses
`Pipeline` (plus any other module helpers it needs) to compose a *named
scenario*, wrapped in a small `argparse` CLI. Examples:

- `deterministic_inference_server.py` — the snippet above + an
  `egress_frames()` check that the network output is also reproducible.
- `verified_inference.py` — adds a matmul attestation pass on top of the run.
- `deterministic_lora_training.py` — the same shape, for LoRA fine-tunes.

So the layering is: spine (the CLIs) ▸ `Pipeline` (the chainer) ▸ workflows
(named recipes that use the chainer). "Workflow" here is much more modest than
in Airflow / GitHub Actions — it's a ~60-line script, not a DAG orchestrator.

**Demo:** [Prover ↔ Verifier protocol](demos/prover-verifier/reports/memo.md) — wire-protocol demo that detects hidden training and exfiltration from external evidence alone. CPU-only; `cd demos/prover-verifier && ./demo.sh --quick`.

## Build & run

Building from this checkout is the canonical, reproducible path. The full
closure compiles vLLM and PyTorch from source, so plan on 30–60 minutes and a
beefy machine for the first build (see
[`.github/workflows/nix-build.yml`](.github/workflows/nix-build.yml) for a
manually-triggerable CI build).

```bash
# Build the hermetic runtime closure
nix build .#closure

# Build the OCI image — produces `inference-server-runtime:<git-rev>`
nix build .#oci
docker load < result

# Run the server in Docker
docker run -d --name vllm-server --gpus all --privileged \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -v "$PWD:/workspace" -p 8000:8000 \
  inference-server-runtime:dev \
  --manifest /workspace/demos/e2e-audit/scripts/smoke.manifest.json \
  --skip-boot-validation
```

The NVIDIA Container Toolkit must be installed and configured as Docker's default runtime:

```bash
sudo nvidia-ctk runtime configure --runtime=docker --set-as-default
sudo systemctl restart docker
```

Troubleshooting:

| Symptom | Fix |
|---------|-----|
| `Failed to infer device type` | Add `--privileged -e NVIDIA_DRIVER_CAPABILITIES=all` |
| `No CUDA GPUs are available` | Add `--privileged` |
| `Can't initialize NVML` | Set `"default-runtime": "nvidia"` in daemon.json |
| `GLIBC_2.38 not found` | Don't set `LD_LIBRARY_PATH` to host system paths |

## CI gates

| Gate | What it runs | Command |
|------|-------------|---------|
| PR | lint + schema + unit/integration | `make ci-pr` |
| Main | + e2e + determinism + nix closure | `make ci-main` |
| Nightly | + chaos + long-run | `make ci-nightly` |
| Release | + release contracts | `make ci-release` |
