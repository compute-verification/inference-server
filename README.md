# Deterministic Inference Server

This repository implements an inference server whose egress traffic is a deterministic function of its ingress traffic. As a result, a verifier can check that the egress traffic is correct by re-executing the inference server's computation. We achieve this primarily through three interventions:
- The image in which the inference server is run is built deterministically using Nix. As a result, when a verifier re-executes the inference server's computation, it can trust that the image is correct.
- Inference is made deterministic using batch-invariant kernels, deterministic cuBLAS kernels, eager execution, and greedy decoding with a fixed seed.
- Tokens are emitted as egress traffic using a custom deterministic userspace TCP/IP stack. 

These interventions guarantee that egress traffic can be bitwise reproduced on a machine _with the same hardware_ as the machine that originally produced the egress traffic. The current implementation does not enable bitwise reproduction on machines with different accelerators because different accelerators handle floating-point operations differently. In the future we will use [Hawkeye](https://arxiv.org/abs/2603.20421) so that requests can be bitwise reproduced on different hardware. We further demonstrate [proof of secure erasure](https://en.wikipedia.org/wiki/Proof_of_secure_erasure), which can be used to erase covert state on the hardware running the inference server.

We have tested the reproducibility of egress traffic when serving several models, including mixture-of-experts models. Across over 5 million tokens and three models, we did not observe a single token or egress bit that could was not bitwise reproduced by the verifier.

Licensed under [Apache-2.0](LICENSE).

### Repository layout

This repository consists of small _modules_ that implement discrete features. Modules are composed to define _workflows_, which are executable programs that determine what the inference server should run, how traffic should captured, etc. Most workflows will center around sending requests to our custom `/manifest` endpoint, which consumes a JSON file that fully determines how the inference server should behave (e.g. it specifies the vLLM config, a hash of the weights and inputs, etc.).

```
modules/                Each module owns its code, plus shared core/ + Pipeline
  build/                Hermetic runtime: builder/ + lockfiles/ + nix/   (flake.nix + flake.lock live at root)
  inference/            Deterministic vLLM
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
workflows/              Runnable compositions of the modules
demos/                  End-to-end scenarios: e2e-audit (the scripts/demo.sh path), prover-verifier (the protocol demo). Research experiments live on the `experiments` branch.
scripts/deploy/         Lambda / vast provisioning (utils-owned)
tests/conformance/      Spec conformance catalog + release blockers (read by CI)
flake.nix, flake.lock   Hermetic build entrypoint + pin (at root: src=self packages repo-wide code; callers invoke `.#`)
```

## Build and run

Demonstrating bitwise reproducible egress traffic takes only a few minutes, but building an OCI image with Nix to run the inference server from can take close to an hour. In practice, we find that using the standard CUDA images provided by compute providers does not affect the reproducibility of egress traffic, though it plausibly could. To run the demo without building the image with Nix, bring up any instance with CUDA 12.8 and an NVIDIA GPU that supports batch invariance (e.g. an H100), then run:

```bash
git clone https://github.com/compute-verification/deterministic-inference-server
cd deterministic-inference-server
./scripts/demo.sh
```

`scripts/demo.sh` builds a venv, starts the inference server, and runs the following loop:

1. `POST /run` — server runs inference and returns per-output token commitments
2. `POST /replay` at random token positions — server re-executes each request truncated to the challenged position and recomputes the commitment
3. Negative test — a forged commitment must not match

To build the OCI image with Nix, run the following:

```bash
# Build the hermetic runtime closure
nix build .#closure

# Build the OCI image — produces `deterministic-inference-server-runtime:<git-rev>`
nix build .#oci
docker load < result

# Run the server in Docker
docker run -d --name vllm-server --gpus all --privileged \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -v "$PWD:/workspace" -p 8000:8000 \
  deterministic-inference-server-runtime:dev \
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
