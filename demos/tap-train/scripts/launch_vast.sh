#!/usr/bin/env bash
# Headline launcher for the tap-train demo.
# Runs on the laptop. Provisions an H100 on vast, ships the worktree,
# starts the four servers, prints a ready-to-paste client invocation.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEMO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$DEMO_DIR/../.." && pwd)"

cd "$REPO_ROOT"

# ---- 1. Resolve entrypoint ----
ENTRY=$(bash "$SCRIPT_DIR/resolve_entrypoint.sh")
echo "[launcher] entrypoint: $ENTRY"

# ---- 2. Search offers ----
echo "[launcher] searching H100_SXM offers..."
OFFER_ID=""
while [ -z "$OFFER_ID" ]; do
    OFFER_ID=$(vastai search offers \
        'gpu_name=H100_SXM num_gpus=1 cuda_vers>=12.0 reliability>0.90 inet_down>300 disk_space>80' \
        -o 'dph' --raw | python3 -c "
import sys, json
offers = json.load(sys.stdin)
if offers:
    print(offers[0]['id'])
" || true)
    if [ -z "$OFFER_ID" ]; then
        echo "[launcher] no H100 offers; sleeping 30s..."
        sleep 30
    fi
done
echo "[launcher] picked offer $OFFER_ID"

# ---- 3. Create instance ----
PUBKEY_B64=$(cat ~/.ssh/id_ed25519.pub | base64 -w0)
echo "[launcher] creating instance..."
CREATE_OUT=$(vastai create instance "$OFFER_ID" \
    --image "ghcr.io/compute-verification/inference-server:vast-test" \
    --disk 80 \
    --env "-p 22:22 -p 8000:8000 -e PUBKEY_B64=$PUBKEY_B64 -e SKIP_SERVER=1" \
    --entrypoint "$ENTRY" \
    --args 2>&1)
echo "$CREATE_OUT"
INSTANCE_ID=$(echo "$CREATE_OUT" | python3 -c "
import sys, re
# vastai may print Python dict repr (single quotes) or JSON (double); accept both.
m = re.search(r'[\"\\']new_contract[\"\\']\s*:\s*([0-9]+)', sys.stdin.read())
if m:
    print(m.group(1))
")
if [ -z "$INSTANCE_ID" ]; then
    echo "[launcher] could not parse instance id; aborting" >&2
    exit 1
fi
echo "$INSTANCE_ID" > "$DEMO_DIR/.last_instance"
echo "[launcher] instance id $INSTANCE_ID (recorded in .last_instance)"

# ---- 4. Wait for SSH ----
echo "[launcher] waiting for SSH..."
IP=""
PORT=""
DEADLINE=$(( $(date +%s) + 600 ))
while [ -z "$IP" ] || [ -z "$PORT" ]; do
    [ "$(date +%s)" -gt "$DEADLINE" ] && { echo "ssh wait timeout" >&2; exit 1; }
    RAW=$(vastai show instance "$INSTANCE_ID" --raw 2>/dev/null || true)
    IP=$(echo "$RAW" | python3 -c "
import sys, re
m = re.search(r'\"public_ipaddr\"\s*:\s*\"([^\"]*)\"', sys.stdin.read())
print(m.group(1) if m else '')
")
    PORT=$(echo "$RAW" | python3 -c "
import sys, re
m = re.search(r'\"22/tcp\"\s*:\s*\[\s*\{[^}]*\"HostPort\"\s*:\s*\"([0-9]+)\"', sys.stdin.read())
print(m.group(1) if m else '')
")
    if [ -z "$IP" ] || [ -z "$PORT" ]; then
        sleep 5
    fi
done
echo "[launcher] ssh target: $IP:$PORT"
ssh-keyscan -p "$PORT" "$IP" >> ~/.ssh/known_hosts 2>/dev/null || true
until ssh -o StrictHostKeyChecking=no -i ~/.ssh/id_ed25519 -p "$PORT" root@"$IP" true 2>/dev/null; do
    echo "[launcher] waiting for sshd..."; sleep 5
done

SSH_OPTS="-o StrictHostKeyChecking=no -i $HOME/.ssh/id_ed25519 -p $PORT"
# scp uses -P (uppercase) for port; -p means preserve mtimes and breaks
# the command if $PORT is passed as a value to it. Keep these separate.
SCP_OPTS="-o StrictHostKeyChecking=no -i $HOME/.ssh/id_ed25519 -P $PORT"

# ---- 5. CUDA fixups ----
echo "[launcher] fixing CUDA symlinks..."
ssh $SSH_OPTS root@"$IP" bash -s < "$SCRIPT_DIR/fix_cuda_symlinks.sh"

# ---- 6. Ship code ----
# Nix-only vast-test image has no `rsync` and no `tar` — bundle into a
# tarball locally, scp it, then extract via Python's `tarfile` on the box.
echo "[launcher] shipping worktree (tar + scp)..."
BUNDLE=$(mktemp -t tap-train-bundle.XXXXXX.tar.gz)
trap 'rm -f "$BUNDLE"' EXIT
tar -C "$REPO_ROOT" -czf "$BUNDLE" \
    --exclude='.claude' --exclude='.venv' --exclude='.git' \
    --exclude='__pycache__' --exclude='*.pyc' \
    .
scp $SCP_OPTS "$BUNDLE" "root@$IP:/root/dss.tar.gz"
ssh $SSH_OPTS root@"$IP" 'mkdir -p /root/dss && python3 -m tarfile -e /root/dss.tar.gz /root/dss && rm -f /root/dss.tar.gz && echo "[remote] extracted"'

# ---- 7. Pre-fetch weights ----
echo "[launcher] prefetching weights snapshot..."
ssh $SSH_OPTS root@"$IP" 'bash -se' <<'PREFETCH'
set -euo pipefail
python3 -c "
from huggingface_hub import snapshot_download
p = snapshot_download('Qwen/Qwen3-1.7B', revision='70d244cc86ccca08cf5af4e1e306ecf908b1ad5e')
print(p)
" > /root/snapshot_path
test -s /root/snapshot_path
echo "[remote] snapshot_path=$(cat /root/snapshot_path)"
PREFETCH

# ---- 7b. Ship peft + accelerate (training-only deps not in the vast-test image) ----
# The Nix vast-test image bundles vllm/torch/transformers/huggingface_hub but
# not peft, and Nix python has no pip or system CA, so the box can't fetch
# wheels itself. We `uv pip install --no-deps --target` locally, tarball
# `peft/` + `accelerate/`, scp them, and extract into /root/pylibs. The
# remaining peft transitive deps (huggingface_hub, safetensors, torch,
# transformers, numpy, packaging, psutil, tqdm, etc.) are already on the
# image via vllm.
echo "[launcher] shipping peft + accelerate wheels..."
PEFT_BUILD=$(mktemp -d -t tap-train-peft.XXXXXX)
uv pip install --target "$PEFT_BUILD" --python "$(command -v python3)" --no-deps peft accelerate >/dev/null
PEFT_BUNDLE=$(mktemp -t tap-train-peft.XXXXXX.tar.gz)
tar -C "$PEFT_BUILD" -czf "$PEFT_BUNDLE" peft peft-*.dist-info accelerate accelerate-*.dist-info
scp $SCP_OPTS "$PEFT_BUNDLE" "root@$IP:/root/peft-bundle.tar.gz"
ssh $SSH_OPTS root@"$IP" 'mkdir -p /root/pylibs && python3 -m tarfile -e /root/peft-bundle.tar.gz /root/pylibs && rm -f /root/peft-bundle.tar.gz && PYTHONPATH=/root/pylibs python3 -c "import peft, accelerate; print(\"[remote] peft\", peft.__version__, \"accelerate\", accelerate.__version__)"'
rm -rf "$PEFT_BUILD" "$PEFT_BUNDLE"

# ---- 8. Start servers ----
# Nix image has no `setsid`; the `( ... & )` subshell pattern detaches
# start_servers.sh from this ssh session, and start_servers.sh in turn
# uses the same pattern for each of its four child processes.
# PYTHONPATH carries /root/pylibs so peft+accelerate are importable;
# start_servers.sh preserves it (prepending /root/dss).
echo "[launcher] starting servers..."
ssh $SSH_OPTS root@"$IP" "bash -c '(cd /root/dss && export PYTHONPATH=/root/pylibs && export RUNNER_MODEL_PATH=\$(cat /root/snapshot_path) && nohup bash demos/tap-train/scripts/start_servers.sh > /root/start.out 2>&1 < /dev/null &)'"

# ---- 9. Wait for Gateway externally ----
echo "[launcher] waiting for Gateway /health on $IP:8000 (up to 600s)..."
DEADLINE=$(( $(date +%s) + 600 ))
GW_OK=0
while [ "$(date +%s)" -le "$DEADLINE" ]; do
    if curl -sf "http://$IP:8000/health" >/dev/null 2>&1; then
        GW_OK=1
        break
    fi
    sleep 5
done

if [ "$GW_OK" -ne 1 ]; then
    echo "[launcher] gateway wait timeout; dumping last 60 lines of /root/start.out:" >&2
    ssh $SSH_OPTS root@"$IP" "tail -n 60 /root/start.out 2>/dev/null || echo '(no start.out)'" >&2
    echo "[launcher] last 40 lines of each server log:" >&2
    ssh $SSH_OPTS root@"$IP" "for f in /root/tap-train-logs/*.out; do echo; echo '=== '\$f' ==='; tail -n 40 \$f; done 2>/dev/null || true" >&2
    exit 1
fi

echo
echo "=== Tap-train demo ready ==="
echo "Instance: $INSTANCE_ID  ($IP:$PORT ssh)"
echo "Gateway:  http://$IP:8000"
echo
echo "Try:"
echo "  python3 demos/tap-train/client.py --url http://$IP:8000 --recipe quick"
echo
echo "Teardown:"
echo "  bash demos/tap-train/scripts/teardown.sh"
