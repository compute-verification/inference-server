#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

log "Running schema gate"

# Prefer the project venv (has jsonschema pinned); fall back to system python3.
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PYTHON="${PYTHON:-${REPO_ROOT}/.venv/bin/python3}"
if [ ! -x "$PYTHON" ]; then
  PYTHON=python3
fi
require_cmd "$PYTHON"

"$PYTHON" scripts/ci/schema_validate.py
"$PYTHON" scripts/ci/schema_compat.py
"$PYTHON" scripts/ci/check_canonical_json.py modules/core/schemas tests/fixtures/positive tests/fixtures/negative
"$PYTHON" scripts/ci/check_conformance_catalog.py
"$PYTHON" scripts/ci/fixture_validate.py
"$PYTHON" scripts/ci/lockfile_validate.py --lockfile tests/fixtures/positive/lockfile.v1.example.json

log "Schema gate passed"
