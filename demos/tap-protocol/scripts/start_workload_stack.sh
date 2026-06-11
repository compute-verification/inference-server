#!/usr/bin/env bash
# Start the 4-server stack in WORKLOAD mode on the GPU box: no vLLM children
# (--no-vllm) -- /run workload harnesses load their own models, so the GPU
# must stay free for them (the coding scenario loads Qwen3-8B twice in
# sequence). Gateway binds 0.0.0.0:8000 (vast maps it to a public port).
#
# Usage (on the box, from the repo root):  bash demos/tap-protocol/scripts/start_workload_stack.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SERVERS="$REPO_ROOT/demos/tap-protocol/servers"
OUT="${OUT_DIR:-/root/workload-stack}"
mkdir -p "$OUT"

# vast bind-mounts driver libs without the .so.1 symlinks; the symlink fix
# (fix_cuda_symlinks.sh) must have run already. Export for every child.
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu
export CUBLAS_WORKSPACE_CONFIG=:4096:8

start() { # name cmd...
    local name="$1"; shift
    ( nohup "$@" </dev/null >"$OUT/$name.log" 2>&1 & )
    echo "[stack] started $name"
}

wait_healthy() { # url label attempts sleep_s
    local url="$1" label="$2" attempts="$3" sleep_s="$4" i=0
    while [ "$i" -lt "$attempts" ]; do
        if python3 - "$url" <<'EOF'
import sys, urllib.request
try:
    urllib.request.urlopen(sys.argv[1], timeout=5)
except Exception:
    raise SystemExit(1)
EOF
        then echo "[stack] $label healthy"; return 0; fi
        i=$((i+1)); sleep "$sleep_s"
    done
    echo "[stack] $label never became healthy; see $OUT" >&2
    return 1
}

start host   python3 "$SERVERS/host_cluster.py"   --port 8020 --no-vllm \
             --tap-url http://127.0.0.1:8010 --out-dir "$OUT/host"
start recomp python3 "$SERVERS/recomp_cluster.py" --port 8030 --no-vllm \
             --tap-url http://127.0.0.1:8010 --out-dir "$OUT/recomp"
start tap    python3 "$SERVERS/tap.py"            --port 8010 \
             --host-url http://127.0.0.1:8020 --recomp-url http://127.0.0.1:8030
start gw     python3 "$SERVERS/gateway.py"        --port 8000 --host 0.0.0.0 \
             --tap-url http://127.0.0.1:8010

wait_healthy http://127.0.0.1:8020/health host   30 2
wait_healthy http://127.0.0.1:8030/health recomp 30 2
wait_healthy http://127.0.0.1:8010/health tap    15 2
wait_healthy http://127.0.0.1:8000/health gateway 15 2
echo "[stack] workload stack up; gateway on :8000"
