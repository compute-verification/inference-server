from __future__ import annotations

import unittest

from pkg.common.contracts import validate_with_schema
from pkg.proverdet.replay import stub_evidence
from pkg.proverdet.wire import (
    ErasureSpec,
    ProofOfWorkSpec,
    ReplayEvidence,
    ReplayRequest,
    TaskTarget,
)


def _make_request(replay_id: str = "r-1") -> ReplayRequest:
    return ReplayRequest(
        replay_id=replay_id,
        pod_id="pod-a",
        target=TaskTarget(kind="task", task_id="t-0"),
        erasure=ErasureSpec(challenge_seed="deadbeef", deadline_ms=1000, rounds=4),
        proof_of_work=ProofOfWorkSpec(matmul_dim=64, dtype="bf16", rounds=3, report_every_ms=100),
        auxiliary=[],
    )


class TestStubEvidence(unittest.TestCase):
    def test_returns_replay_evidence(self) -> None:
        ev = stub_evidence(_make_request())
        self.assertIsInstance(ev, ReplayEvidence)

    def test_replay_id_is_preserved(self) -> None:
        ev = stub_evidence(_make_request("r-42"))
        self.assertEqual(ev.replay_id, "r-42")

    def test_validates_against_schema(self) -> None:
        ev = stub_evidence(_make_request())
        validate_with_schema("replay_evidence.v1.schema.json", ev.model_dump(exclude_none=True))

    def test_pow_stream_is_empty_in_stub(self) -> None:
        ev = stub_evidence(_make_request())
        self.assertEqual(ev.pow_stream, [])

    def test_erasure_evidence_passed_equals_rounds(self) -> None:
        req = _make_request()
        ev = stub_evidence(req)
        self.assertEqual(ev.erasure_evidence.rounds, req.erasure.rounds)
        self.assertEqual(ev.erasure_evidence.passed, req.erasure.rounds)

    def test_output_commitment_is_sha256_prefixed(self) -> None:
        ev = stub_evidence(_make_request())
        self.assertTrue(ev.output.commitment.startswith("sha256:"))
        self.assertEqual(len(ev.output.commitment), len("sha256:") + 64)

    def test_two_distinct_replay_ids_have_different_commitments(self) -> None:
        a = stub_evidence(_make_request("r-1"))
        b = stub_evidence(_make_request("r-2"))
        self.assertNotEqual(a.output.commitment, b.output.commitment)


if __name__ == "__main__":
    unittest.main()
