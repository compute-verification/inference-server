"""Replay-evidence builders for the prover.

The Phase 2 stub returns a schema-valid evidence object so the prover and
verifier can integrate end-to-end before Phase 6 plumbs in real Freivalds
attestations + PoSE-style erasure proofs. The stub exists *only* to
accelerate integration; it MUST be replaced before the demo is real.
"""

from __future__ import annotations

import base64

from pkg.common.deterministic import sha256_prefixed, utc_now_iso
from pkg.proverdet.wire import (
    ErasureEvidence,
    ReplayEvidence,
    ReplayOutput,
    ReplayRequest,
)

_STUB_PAYLOAD = b"stub-output"


def stub_evidence(req: ReplayRequest) -> ReplayEvidence:
    """Schema-valid placeholder evidence. Replaced in Phase 6."""

    commitment = sha256_prefixed(b"stub:" + req.replay_id.encode("utf-8"))
    return ReplayEvidence(
        replay_id=req.replay_id,
        produced_at=utc_now_iso(),
        output=ReplayOutput(
            commitment=commitment,
            bytes_b64=base64.b64encode(_STUB_PAYLOAD).decode("ascii"),
        ),
        erasure_evidence=ErasureEvidence(
            rounds=req.erasure.rounds,
            passed=req.erasure.rounds,
            log_path=f"erasure-{req.replay_id}.jsonl",
        ),
        pow_stream=[],
    )
