from __future__ import annotations

import unittest

from pkg.common.contracts import ValidationError, validate_with_schema

GRAPH_SCHEMA = "prover_graph.v1.schema.json"


def _minimal_graph() -> dict:
    return {
        "graph_version": "v1-placeholder",
        "run_id": "demo-001",
        "produced_at": "2026-05-04T12:00:00Z",
        "tasks": [],
        "artifacts": [],
        "transmissions": [],
    }


def _minimal_task() -> dict:
    return {
        "task_id": "task-0",
        "pod_id": "pod-a",
        "operation": "inference",
        "claimed_flops": 1024,
    }


def _minimal_artifact() -> dict:
    return {
        "artifact_id": "art-0",
        "commitment": "sha256:" + "0" * 64,
        "size_bytes": 4096,
    }


def _minimal_transmission() -> dict:
    return {
        "transmission_id": "tx-0",
        "sender_pod_id": "pod-a",
        "receiver_pod_id": "pod-b",
        "artifact_id": "art-0",
        "tap_signature": "deadbeef" * 16,
    }


class TestGraphSchema(unittest.TestCase):
    def test_minimal_graph_validates(self) -> None:
        validate_with_schema(GRAPH_SCHEMA, _minimal_graph())

    def test_graph_with_one_task_artifact_transmission_validates(self) -> None:
        graph = _minimal_graph()
        graph["tasks"] = [_minimal_task()]
        graph["artifacts"] = [_minimal_artifact()]
        graph["transmissions"] = [_minimal_transmission()]
        validate_with_schema(GRAPH_SCHEMA, graph)

    def test_graph_rejects_unknown_field(self) -> None:
        bad = _minimal_graph()
        bad["spurious"] = 1
        with self.assertRaises(ValidationError):
            validate_with_schema(GRAPH_SCHEMA, bad)

    def test_graph_requires_run_id(self) -> None:
        bad = _minimal_graph()
        del bad["run_id"]
        with self.assertRaises(ValidationError):
            validate_with_schema(GRAPH_SCHEMA, bad)

    def test_graph_requires_graph_version_const(self) -> None:
        bad = _minimal_graph()
        bad["graph_version"] = "v2"
        with self.assertRaises(ValidationError):
            validate_with_schema(GRAPH_SCHEMA, bad)

    def test_artifact_commitment_must_be_sha256_prefixed(self) -> None:
        graph = _minimal_graph()
        bad_artifact = _minimal_artifact()
        bad_artifact["commitment"] = "deadbeef"
        graph["artifacts"] = [bad_artifact]
        with self.assertRaises(ValidationError):
            validate_with_schema(GRAPH_SCHEMA, graph)

    def test_task_claimed_flops_rejects_string(self) -> None:
        graph = _minimal_graph()
        bad_task = _minimal_task()
        bad_task["claimed_flops"] = "100"
        graph["tasks"] = [bad_task]
        with self.assertRaises(ValidationError):
            validate_with_schema(GRAPH_SCHEMA, graph)

    def test_task_claimed_flops_rejects_negative(self) -> None:
        graph = _minimal_graph()
        bad_task = _minimal_task()
        bad_task["claimed_flops"] = -1
        graph["tasks"] = [bad_task]
        with self.assertRaises(ValidationError):
            validate_with_schema(GRAPH_SCHEMA, graph)

    def test_transmission_requires_sender_and_receiver(self) -> None:
        graph = _minimal_graph()
        bad_tx = _minimal_transmission()
        del bad_tx["sender_pod_id"]
        graph["transmissions"] = [bad_tx]
        with self.assertRaises(ValidationError):
            validate_with_schema(GRAPH_SCHEMA, graph)

    def test_task_rejects_unknown_field(self) -> None:
        graph = _minimal_graph()
        bad_task = _minimal_task()
        bad_task["mystery"] = "?"
        graph["tasks"] = [bad_task]
        with self.assertRaises(ValidationError):
            validate_with_schema(GRAPH_SCHEMA, graph)


if __name__ == "__main__":
    unittest.main()
