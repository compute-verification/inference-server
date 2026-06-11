#!/usr/bin/env bash
# coding-agent demo entry point.
#
# A simple coding agent task captured as a task graph: "summarize a paper that
# just came out, then implement it." The agent ran real web searches + fetches
# to retrieve p-less sampling (arXiv:2509.23234), extracted the algorithm (the
# "plan"), generated an implementation + tests (codegen), and ran the tests
# (verify). The captured graph is demos/coding-agent/coding_agent_graph.json;
# this script re-runs the *verify* node for real so the green check is honest.
#
# Usage: ./demo.sh
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PY="$(command -v python3)"
export PYTHONPATH="${REPO_ROOT}"

echo "=== goal ==="
"${PY}" -c "import json; print(json.load(open('${SCRIPT_DIR}/coding_agent_graph.json'))['goal'])"

echo ""
echo "=== captured task graph (search -> plan -> codegen -> verify) ==="
"${PY}" - "${SCRIPT_DIR}/coding_agent_graph.json" <<'PY'
import json, sys
g = json.load(open(sys.argv[1]))
for n in g["nodes"]:
    print(f"  [{n['kind']:7}] {n['label']}")
kinds = {}
for n in g["nodes"]:
    kinds[n["kind"]] = kinds.get(n["kind"], 0) + 1
print(f"  -> {len(g['nodes'])} nodes, {len(g['edges'])} edges: {kinds}")
PY

echo ""
echo "=== re-running the verify node: tests for the generated implementation ==="
"${PY}" -m unittest tests.unit.test_p_less -v 2>&1 | tail -4

echo ""
echo "implemented: demos/coding-agent/generated/p_less.py  (p-less sampling, arXiv:2509.23234)"
echo "ALL PASS"
