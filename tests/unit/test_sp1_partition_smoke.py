"""Smoke test for the partition SP1 host binary.

Skipped when the binary isn't compiled (``cargo-prove`` and a usable
``protoc`` aren't on every dev machine). When it IS available, this runs the
partition program in SP1's execute mode and verifies that the program's
committed ``graph_digest`` matches the Python side's byte-stable encoding —
the single point exercising Python<->Rust agreement on the
taskgraph-partition-v1 canonical bytes.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from modules.proof_server.partition import (
    check_partition,
    graph_partition_digest,
    plan_partition,
    sp1_input_json,
)

HOST_BIN = REPO_ROOT / "modules/proof_server/sp1/target/release/partition-host"

GRAPH = {
    "nodes": [
        {"id": 0, "flops": 1000, "tokens": 142, "whitelisted": True},
        {"id": 1, "flops": 400, "tokens": 1},
        {"id": 2, "flops": 400, "tokens": 1},
        {"id": 3, "flops": 900, "tokens": 50},
    ],
    "edges": [{"src": 0, "dst": 1}, {"src": 0, "dst": 2},
              {"src": 1, "dst": 3}, {"src": 2, "dst": 3}],
}
CAP_FLOPS, CAP_INPUT = 1800, 52
NONCE = "11" * 32


def _have_sp1() -> bool:
    return HOST_BIN.exists() and shutil.which("cargo-prove") is not None


def _run(input_json: str) -> subprocess.CompletedProcess:
    return subprocess.run([str(HOST_BIN), "--execute"],
                          input=input_json.encode(), capture_output=True, timeout=600)


@unittest.skipUnless(_have_sp1(),
                     "partition-host binary missing; install sp1up + protoc and rebuild")
class TestSP1PartitionSmoke(unittest.TestCase):
    def test_valid_partition_commits_matching_digest(self):
        parts = plan_partition(GRAPH, CAP_FLOPS, CAP_INPUT)
        stats = check_partition(GRAPH, parts, CAP_FLOPS, CAP_INPUT)

        result = _run(sp1_input_json(GRAPH, parts, CAP_FLOPS, CAP_INPUT, NONCE))
        if result.returncode != 0:
            self.fail(f"partition-host exited {result.returncode}\n"
                      f"stderr: {result.stderr.decode(errors='replace')[-400:]}")
        public = json.loads(result.stdout.decode().strip().splitlines()[-1])

        # The verifier's job: recompute the digest from the published graph
        # and check SP1 committed exactly that, under exactly these caps.
        self.assertEqual(public["graph_digest"], graph_partition_digest(GRAPH))
        self.assertEqual(public["auditor_nonce"], NONCE)
        self.assertEqual(public["cap_flops"], CAP_FLOPS)
        self.assertEqual(public["cap_input"], CAP_INPUT)
        self.assertEqual(public["n_nodes"], 4)
        self.assertEqual(public["n_parts"], stats["n_parts"])

    def test_over_budget_part_aborts_guest(self):
        # All four nodes in one part blows the FLOP cap -> guest assert fires
        # -> zero public-output bytes -> host exits 10.
        result = _run(sp1_input_json(GRAPH, [0, 0, 0, 0], CAP_FLOPS, CAP_INPUT, NONCE))
        self.assertNotEqual(result.returncode, 0,
                            "over-budget partition should have failed the guest")

    def test_backward_edge_between_parts_aborts_guest(self):
        # parts must not decrease along an edge: 0 -> 1 with part 1 -> 0.
        result = _run(sp1_input_json(
            GRAPH, [1, 0, 1, 1], 10**9, 10**9, NONCE))
        self.assertNotEqual(result.returncode, 0,
                            "backward edge between parts should have failed the guest")

    def test_whitelist_is_load_bearing_in_guest(self):
        # CAP_INPUT=52 only fits because node 0's 142 whitelisted tokens are
        # free. Same partition with the flag stripped must abort in-guest.
        stripped = json.loads(json.dumps(GRAPH))
        stripped["nodes"][0].pop("whitelisted")
        parts = plan_partition(GRAPH, CAP_FLOPS, CAP_INPUT)
        result = _run(sp1_input_json(stripped, parts, CAP_FLOPS, CAP_INPUT, NONCE))
        self.assertNotEqual(result.returncode, 0,
                            "without the whitelist the same partition must bust S")
        # ...and the stripped graph hashes differently, so it could not be
        # passed off as the whitelisted one anyway.
        self.assertNotEqual(graph_partition_digest(stripped),
                            graph_partition_digest(GRAPH))


if __name__ == "__main__":
    unittest.main()
