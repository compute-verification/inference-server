"""Unit tests for the build script that emits graphs.json for the viz."""
import json
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
BUILD_DIR = REPO_ROOT / "demos" / "proof-compare"
for p in (REPO_ROOT, BUILD_DIR, BUILD_DIR / "tracers"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import build_all


class TestBuildAll(unittest.TestCase):
    def test_build_all_has_four_scenarios_with_nodes(self):
        data = build_all.build_all()
        self.assertEqual(set(data), {"inference", "spec", "training", "coding"})
        for k, g in data.items():
            self.assertTrue(g["nodes"], f"{k} has no nodes")
            self.assertIn("edges", g)

    def test_coding_nodes_are_all_forward_passes(self):
        # The coding agent must be at forward-pass granularity (prefill/decode),
        # not agent-action nodes — guard against regressing the abstraction.
        data = build_all.build_all()
        kinds = {n["kind"] for n in data["coding"]["nodes"]}
        self.assertTrue(kinds <= {"prefill", "decode"}, f"unexpected coding kinds: {kinds}")

    def test_dump_is_canonical_json_round_trip(self):
        data = build_all.build_all()
        payload = build_all.dump(data)
        self.assertTrue(payload.endswith("\n"))
        self.assertEqual(json.loads(payload), data)

    def test_dump_survives_unicode_in_payloads(self):
        # token text / prompts can contain non-ASCII; dump must not choke.
        build_all.dump({"x": {"nodes": [{"label": "café ⟂"}], "edges": []}})


if __name__ == "__main__":
    unittest.main()
