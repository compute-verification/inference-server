#!/usr/bin/env bash
# proof-server demo entry point.
#
# The proof server is a proxy: it sits between the existing tap-protocol
# topology (Gateway -> Tap -> Host Cluster, plus async Recomp) and a new
# auditor process. The Tap fans out a copy of every verified envelope pair
# to the proof server; the auditor talks only to the proof server.
#
# This demo runs everything on one box in --mock mode (no GPU).
#
# Usage:
#   ./demo.sh             # default: --quick (SP1 execute, no proof bytes)
#   ./demo.sh --quick     # same -- explicit
#   ./demo.sh --prove     # generate a real SP1 proof + verify it (minutes)
#   ./demo.sh --help

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

MODE="execute"
for arg in "$@"; do
  case "$arg" in
    --quick) MODE="execute" ;;
    --prove) MODE="prove" ;;
    -h|--help)
      sed -n '2,15p' "${BASH_SOURCE[0]}"
      exit 0
      ;;
    *)
      echo "Unknown arg: $arg" >&2
      exit 2
      ;;
  esac
done

# --------------------------------------------------------------------------
# Pick a Python interpreter.
# --------------------------------------------------------------------------
PY=""
if [[ -x "${REPO_ROOT}/.venv/bin/python3" ]]; then
  PY="${REPO_ROOT}/.venv/bin/python3"
elif command -v uv >/dev/null 2>&1; then
  echo "demo.sh: provisioning ${REPO_ROOT}/.venv via uv (one-time)" >&2
  ( cd "${REPO_ROOT}" && uv sync --quiet )
  PY="${REPO_ROOT}/.venv/bin/python3"
else
  PY="$(command -v python3 || true)"
  [[ -z "${PY}" ]] && { echo "demo.sh: no python3 available" >&2; exit 1; }
  echo "demo.sh: falling back to system python (needs cryptography + pydantic)" >&2
fi

# --------------------------------------------------------------------------
# SP1 host binary required.
# --------------------------------------------------------------------------
HOST_BIN="${REPO_ROOT}/modules/proof_server/sp1/target/release/proof-server-host"
if [[ ! -x "${HOST_BIN}" ]]; then
  cat >&2 <<EOF
demo.sh: SP1 host binary not found at:
   ${HOST_BIN}

Build it once with:
   curl -L https://sp1.succinct.xyz | bash && \$HOME/.sp1/bin/sp1up
   # protoc is needed by sp1-prover-types' build script:
   PROTOC=\$(which protoc) cargo build --release \\
     --manifest-path modules/proof_server/sp1/host/Cargo.toml

Then re-run this script.
EOF
  exit 3
fi

# --------------------------------------------------------------------------
# Workspace + log directory.
# --------------------------------------------------------------------------
WORK="$(mktemp -d -t proof-server-demo-XXXXXX)"
LOGS="${WORK}/logs"
mkdir -p "${LOGS}"

# Free up the ports the existing tap-protocol demo binds to + the proof
# server's own port.
PORTS=(8000 8010 8020 8021 8022 8030 8031 8032 8040)
for p in "${PORTS[@]}"; do
  pid="$(lsof -ti :"$p" 2>/dev/null || true)"
  [[ -n "$pid" ]] && kill -9 $pid 2>/dev/null || true
done

PIDS=()
cleanup() {
  for pid in "${PIDS[@]}"; do
    kill "$pid" 2>/dev/null || true
  done
  sleep 1
  for pid in "${PIDS[@]}"; do
    kill -9 "$pid" 2>/dev/null || true
  done
  rm -rf "${WORK}"
}
trap cleanup EXIT

start_bg() {
  local name="$1"; shift
  ( "$@" </dev/null >"${LOGS}/${name}.out" 2>&1 ) &
  PIDS+=("$!")
  echo "[demo] started ${name} (pid $!)"
}

wait_health() {
  local url="$1"; local deadline=$(( $(date +%s) + 60 ))
  until [[ $(date +%s) -ge $deadline ]]; do
    if curl -sf "$url" >/dev/null 2>&1; then return 0; fi
    sleep 0.2
  done
  echo "demo.sh: timed out waiting for $url" >&2
  return 1
}

export PYTHONPATH="${REPO_ROOT}"

# --------------------------------------------------------------------------
# Start tap-protocol's host + recomp clusters in --mock mode, then tap,
# then the proof server, then the gateway. Order matters: each cluster's
# /health gates the next.
# --------------------------------------------------------------------------
start_bg host_cluster   "${PY}" "${REPO_ROOT}/demos/tap-protocol/servers/host_cluster.py" \
  --port 8020 --proxy-port 8021 --vllm-port 8022 \
  --manifest "${REPO_ROOT}/demos/tap-protocol/qwen3-1.7b-tap.manifest.json" \
  --out-dir "${WORK}/host-cluster" --mock
wait_health http://127.0.0.1:8020/health

start_bg recomp_cluster "${PY}" "${REPO_ROOT}/demos/tap-protocol/servers/recomp_cluster.py" \
  --port 8030 --proxy-port 8031 --vllm-port 8032 \
  --manifest "${REPO_ROOT}/demos/tap-protocol/qwen3-1.7b-tap.manifest.json" \
  --out-dir "${WORK}/recomp-cluster" --mock
wait_health http://127.0.0.1:8030/health

start_bg proof_server   "${PY}" "${REPO_ROOT}/demos/proof-server/servers/proof_server.py" \
  --port 8040 --host-bin "${HOST_BIN}" --work-dir "${WORK}/proof-server"
wait_health http://127.0.0.1:8040/health

start_bg tap            "${PY}" "${REPO_ROOT}/demos/tap-protocol/servers/tap.py" \
  --port 8010 --host-url http://127.0.0.1:8020 \
  --recomp-url http://127.0.0.1:8030 \
  --proof-server-url http://127.0.0.1:8040
wait_health http://127.0.0.1:8010/health

start_bg gateway        "${PY}" "${REPO_ROOT}/demos/tap-protocol/servers/gateway.py" \
  --port 8000 --tap-url http://127.0.0.1:8010
wait_health http://127.0.0.1:8000/health

# --------------------------------------------------------------------------
# Send two requests through the Gateway. The Tap's existing fan-out pushes
# the verified envelopes to the proof server.
# --------------------------------------------------------------------------
echo "=== sending requests through the gateway ==="
for prompt in "hello" "what is 2+2"; do
  curl -sf -X POST http://127.0.0.1:8000/request \
    -H 'Content-Type: application/json' \
    -d "{\"prompt\": \"${prompt}\", \"max_tokens\": 4}" | head -c 200
  echo
done

# Tap's _async_proof_copy is fire-and-forget. Give it a beat to deliver.
sleep 0.5

ROWS="$(curl -sf http://127.0.0.1:8040/health | "${PY}" -c "import json,sys; print(json.load(sys.stdin)['rows'])")"
echo "[demo] proof server has ${ROWS} ledger row(s)"
if [[ "${ROWS}" -lt 4 ]]; then
  echo "demo.sh: expected 4 rows (2 prompts * 2 envelopes); got ${ROWS}" >&2
  tail -20 "${LOGS}/proof_server.out" >&2 || true
  exit 4
fi

# --------------------------------------------------------------------------
# Auditor: end-to-end audit against the proof server.
# --------------------------------------------------------------------------
echo ""
echo "=== auditor ==="
VERIFY_ARGS=()
[[ "${MODE}" == "prove" ]] && VERIFY_ARGS+=(--verify-proof)
"${PY}" "${REPO_ROOT}/demos/proof-server/scripts/audit.py" \
  --proof-server http://127.0.0.1:8040 \
  --mode "${MODE}" \
  --host-bin "${HOST_BIN}" \
  --work-dir "${WORK}/audit" \
  "${VERIFY_ARGS[@]}"

# Negative paths (tampered signature, mismatched nonce, mismatched ledger
# digest) are covered by `tests/unit/test_proof_server_*.py`. Repeating
# them here would require a second SP1 execute, which is the demo's
# heaviest step.

echo ""
echo "ALL PASS"
