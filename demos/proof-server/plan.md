# Proof-server demo — design and implementation plan

A minimal addition to the existing `demos/tap-protocol/` topology that
demonstrates the "developer-controlled proof server" pattern from the
design doc. The proof server is a **proxy**: it sits between the
datacenter (Tap, Host Cluster, Recomp) and a new auditor process. The
Tap fans out a copy of every verified envelope to the proof server; the
auditor only ever reads from the proof server. The proof server emits a
zero-knowledge proof (via SP1) that the published rows are backed by a
valid signed Merkle attestation it produced over its own ledger.

`./demo.sh --quick → ALL PASS` runs the full path locally in `--mock`
mode (no GPU).


## 1. Topology — minimal diff over PR #19

```
client ──► Gateway (8000)       [unchanged from PR #19]
              │
              ▼
            Tap (8010)          [PATCHED: optional --proof-server-url]
              │
              ├─────────────────► Host Cluster (8020)  [unchanged]
              │
              ├─── _async_verify ─► Recomp Cluster (8030)  [unchanged]
              │
              └─── _async_proof_copy ─► Proof Server (8040)  [NEW]
                                            │
                                            ▼ (POST /commit)
                                       SP1 host binary
                                            │
                                            ▼ (proof + public outputs)
                                         Auditor (CLI)  [NEW]
                                         only reads:
                                         GET /ledger
                                         GET /signer_pubkeys
                                         GET /public_outputs
                                         GET /proof.bin   (--prove only)
```

The proof server is the single egress channel to the auditor; the
auditor never touches Gateway/Tap/Host/Recomp.


## 2. What stays vs. what changes

**Unchanged:**
- `demos/tap-protocol/servers/{gateway,host_cluster,recomp_cluster}.py`
- `demos/tap-protocol/servers/envelope.py` (HMAC `SignedEnvelope` is the
  on-the-wire envelope between the datacenter components)
- `demos/tap-protocol/qwen3-1.7b-tap.manifest.json`

**Patched (~25 added lines):**
- `demos/tap-protocol/servers/tap.py` — adds `--proof-server-url` flag
  and a second fire-and-forget daemon thread (`_async_proof_copy`)
  mirroring the existing `_async_verify` fan-out.

**Added:**
- `demos/proof-server/servers/proof_server.py` — HTTP proxy.
  `POST /tap-copy`, `POST /commit?nonce=...&mode=...`, `GET /ledger`,
  `GET /signer_pubkeys`, `GET /public_outputs`, `GET /proof.bin`.
- `demos/proof-server/scripts/audit.py` — CLI client.
- `demos/proof-server/demo.sh` — orchestrates the four PR-#19 servers
  plus the proof server, sends two requests through the Gateway, runs
  the auditor.
- `modules/proof_server/` — Python module (Merkle tree, Ed25519
  helpers, witness assembler) shared by the proxy and the auditor.
- `modules/proof_server/sp1/` — Cargo workspace with the SP1 program
  (the ~200-line Rust guest) and the host harness CLI.


## 3. Ledger row

The proof server records each envelope's `EnvelopeData` (the inner
`{id, payload}` dict) as a ledger row. *Nothing else.* Both the
request envelope and the response envelope contribute a row; the
canonical-JSON content of each row differs so they de-duplicate
correctly under content hashing. No new wire schema — the
`SignedEnvelope.data` shape from PR #19 *is* the ledger row.

The auditor reads the full row list via `GET /ledger` and is free to
project / scrub it for its own dashboards; the SP1 program binds the
auditor's view by committing `sha256(canonical_json_bytes(rows))`.


## 4. SP1 program

`modules/proof_server/sp1/program/src/main.rs` (~170 Rust lines).
The guest reads `ProofInput`:

```rust
ProofInput {
    auditor_nonce: [u8; 32],
    signer_pubkeys: Vec<[u8; 32]>,   // sorted by hex; v0 has length 1
    ledger_rows_canon: Vec<Vec<u8>>, // per-row canonical JSON, no trailing \n
    witnesses: Vec<RowWitness>,
}

RowWitness {
    signer_idx: u32,
    leaf_index: u64,
    merkle_path: Vec<[u8; 32]>,
    signed_root: [u8; 32],
    signature: Vec<u8>,  // 64-byte Ed25519
}
```

For each row, the guest:
1. Computes `leaf = sha256(row_canon || b"\n")`.
2. Recomputes the Merkle root from `(leaf, leaf_index, merkle_path)`.
3. Asserts the recomputed root equals `witness.signed_root`.
4. Verifies `Ed25519(signer_pubkeys[signer_idx], signed_root, signature)`.

Then it commits 104 bytes of public output, in this exact order:

```
auditor_nonce       (32 B)
ledger_digest       (32 B)  = sha256(canonical_json_bytes(rows))
pubkey_set_digest   (32 B)  = sha256(canonical_json_bytes(sorted_hex_pubkeys))
n_rows              (4 B u32 LE)
n_signers           (4 B u32 LE)
```

The auditor independently recomputes `ledger_digest` and
`pubkey_set_digest` from `GET /ledger` and `GET /signer_pubkeys`, and
asserts equality with the committed bytes. The nonce check binds the
proof to *this* audit.


## 5. Why deterministic Ed25519

RFC 8032 Ed25519 has zero attacker-influenceable bits per signature.
Even before the proof hides the signature, there is no Simmons
subliminal channel to exfiltrate via the bytes — defense in depth.
This is the only place in the v0 demo where Ed25519 matters; the rest
of the on-the-wire envelopes inside the datacenter continue to use the
PR-#19 HMAC envelope. The proof server holds the only Ed25519 keypair
in the system.


## 6. Demo flow (`./demo.sh --quick`)

1. Start PR-#19's `host_cluster`, `recomp_cluster` in `--mock` mode
   (no GPU; deterministic canned output), the patched `tap` (pointing
   at the proof server), and the `gateway`. Plus the new
   `proof_server` (port 8040).
2. Send two requests through the Gateway. The Tap forwards each
   verified envelope pair to the proof server's `POST /tap-copy`.
3. Run `audit.py`: generates a nonce, POSTs `/commit?nonce=...` (which
   shells the SP1 host in execute mode), then fetches `/ledger`,
   `/signer_pubkeys`, `/public_outputs`, recomputes the two digests,
   and verifies all five fields of the public outputs match.
4. Print `ALL PASS`.

`./demo.sh --prove` swaps execute for prove and additionally fetches
`/proof.bin` + runs the SP1 verifier on the bytes.


## 7. Toolchain notes

- Python: `cryptography` dep added in `pyproject.toml`.
- SP1: requires `cargo-prove` on PATH. Install:
  `curl -L https://sp1.succinct.xyz | bash && sp1up`. The
  `sp1-prover-types` crate's `build.rs` calls `prost-build`, so a
  `protoc` binary is also required; passed via the `PROTOC` env var.
  The SP1 toolchain is **not** in the Nix runtime image.
- CI: SP1 build is heavy. The `tests/unit/test_proof_server_*.py`
  suite doesn't need SP1 at all. A separate `tests/integration/test_proof_server_demo.py`
  skips when `cargo-prove` is not on PATH.


## 8. Out of scope for v0

- Multi-signer aggregation. v0 has one signer (the proof server
  itself); the SP1 program is already parameterised over `signer_idx`
  so multi-signer is a witness-assembly change, not a circuit change.
- Payload-hash binding inside the leaf. The leaf is the whole
  `EnvelopeData`, which already includes the payload; v0 does not
  project to a "scrubbed" subset.
- Completeness direction. v0 proves *soundness* only: every published
  row is backed by a valid signed Merkle leaf. Catching the proof
  server hiding rows is a v1 follow-up.
- The other five statements in the design doc (memory erasure,
  bandwidth caps, approximate/exact correctness sampling, graph
  shallowness). All layer on top of the same ledger commitment.
- BBS+ selective disclosure. v0's Ed25519 path is the cheapest SP1
  precompile-friendly choice.
- HTTPS / mTLS between the auditor and the proof server. v0 uses
  plaintext HTTP on localhost.
