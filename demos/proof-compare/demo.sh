#!/usr/bin/env bash
# proof-compare demo entry point.
#
# A "proof server" (compare + task-graph variant) sits downstream of the
# tap-protocol topology. Both clusters' token responses reach it via the Tap:
# the Tap calls Recomp /verify (which now returns its recomputed output), then
# forwards {host_output, recomp_output} to the proof server's /compare. The
# proof server bitwise-compares the two, logs MATCH/MISMATCH, and builds a task
# graph from each request (written to the work dir; not yet consumed).
#
# Everything runs on one box in --mock mode (no GPU, no SP1).
#
# Usage:  ./demo.sh   [--help]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

for arg in "$@"; do
  case "$arg" in
    -h|--help) sed -n '2,12p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "Unknown arg: $arg" >&2; exit 2 ;;
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
  echo "demo.sh: falling back to system python (needs pydantic)" >&2
fi

WORK="$(mktemp -d -t proof-compare-demo-XXXXXX)"
LOGS="${WORK}/logs"
GRAPHS="${WORK}/graphs"
mkdir -p "${LOGS}" "${GRAPHS}"

PORTS=(8000 8010 8020 8021 8022 8030 8031 8032 8050)
for p in "${PORTS[@]}"; do
  pid="$(lsof -ti :"$p" 2>/dev/null || true)"
  [[ -n "$pid" ]] && kill -9 $pid 2>/dev/null || true
done

PIDS=()
cleanup() {
  for pid in "${PIDS[@]}"; do kill "$pid" 2>/dev/null || true; done
  sleep 1
  for pid in "${PIDS[@]}"; do kill -9 "$pid" 2>/dev/null || true; done
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
TAP_DIR="${REPO_ROOT}/demos/tap-protocol"
MANIFEST="${TAP_DIR}/qwen3-1.7b-tap.manifest.json"

start_bg host_cluster   "${PY}" "${TAP_DIR}/servers/host_cluster.py" \
  --port 8020 --proxy-port 8021 --vllm-port 8022 \
  --manifest "${MANIFEST}" --out-dir "${WORK}/host-cluster" --mock
wait_health http://127.0.0.1:8020/health

start_bg recomp_cluster "${PY}" "${TAP_DIR}/servers/recomp_cluster.py" \
  --port 8030 --proxy-port 8031 --vllm-port 8032 \
  --manifest "${MANIFEST}" --out-dir "${WORK}/recomp-cluster" --mock
wait_health http://127.0.0.1:8030/health

start_bg proof_server   "${PY}" "${REPO_ROOT}/demos/proof-compare/servers/proof_server.py" \
  --port 8050 --work-dir "${GRAPHS}"
wait_health http://127.0.0.1:8050/health

start_bg tap            "${PY}" "${TAP_DIR}/servers/tap.py" \
  --port 8010 --host-url http://127.0.0.1:8020 \
  --recomp-url http://127.0.0.1:8030 \
  --compare-server-url http://127.0.0.1:8050
wait_health http://127.0.0.1:8010/health

start_bg gateway        "${PY}" "${TAP_DIR}/servers/gateway.py" \
  --port 8000 --tap-url http://127.0.0.1:8010
wait_health http://127.0.0.1:8000/health

echo "=== sending requests through the gateway ==="
for prompt in "hello there" "what is two plus two"; do
  curl -sf -X POST http://127.0.0.1:8000/request \
    -H 'Content-Type: application/json' \
    -d "{\"prompt\": \"${prompt}\", \"max_tokens\": 8}" | head -c 200
  echo
done

# The Tap's verify+compare fan-out is fire-and-forget; give it a beat.
sleep 1.0

HEALTH="$(curl -sf http://127.0.0.1:8050/health)"
echo "[demo] proof server health: ${HEALTH}"

COMPARED="$(echo "${HEALTH}" | "${PY}" -c "import json,sys; print(json.load(sys.stdin)['compared'])")"
MATCHES="$(echo "${HEALTH}" | "${PY}" -c "import json,sys; print(json.load(sys.stdin)['matches'])")"
GRAPHS_BUILT="$(echo "${HEALTH}" | "${PY}" -c "import json,sys; print(json.load(sys.stdin)['graphs_built'])")"

if [[ "${COMPARED}" -lt 2 || "${MATCHES}" -lt 2 || "${GRAPHS_BUILT}" -lt 2 ]]; then
  echo "demo.sh: expected >=2 compared/matches/graphs; got compared=${COMPARED} matches=${MATCHES} graphs=${GRAPHS_BUILT}" >&2
  tail -20 "${LOGS}/proof_server.out" >&2 || true
  exit 4
fi

echo ""
echo "=== a built task graph (one request) ==="
GRAPH_FILE="$(ls "${GRAPHS}"/task_graph_*.json | head -1)"
"${PY}" -m json.tool "${GRAPH_FILE}" | head -40

echo ""
echo "ALL PASS"
