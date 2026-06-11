"""Auditor CLI.

Reads only from the proof server. Never talks to the datacenter directly.

  1. Generates a fresh nonce, posts ``/commit?nonce=...&mode=...`` to ask
     the proof server to seal the current ledger and pin it to that nonce.
  2. Fetches ``/ledger?nonce=...``, ``/signer_pubkeys?nonce=...``,
     ``/public_outputs?nonce=...`` — all pinned to the same snapshot the
     SP1 program saw.
  3. Independently recomputes ``ledger_digest`` and ``pubkey_set_digest``;
     asserts the SP1 program committed exactly those values + the auditor's
     nonce.
  4. In ``--verify-proof`` mode, also fetches ``/proof.bin?nonce=...`` and
     runs the SP1 verifier locally.

What this audit DOES NOT establish (v0)
---------------------------------------
- The audit only proves the proof server's view is internally consistent
  for *this* nonce: same rows go into the digest and into the fetch
  surface. It does NOT prove the proof server published every envelope
  the Tap saw. The "completeness direction" is a v1 follow-up; see
  ``demos/proof-server/plan.md`` §8.

- In ``--mode execute`` (the default), the SP1 host runs the program in
  the RISC-V interpreter and surfaces ``assert!`` panics, but does NOT
  generate or check a zero-knowledge proof. ``--mode prove`` is what
  delivers the cryptographic ZK guarantee. ``--quick`` is for the
  architecture check; ``--prove`` is for the security claim.
"""
from __future__ import annotations

import argparse
import json
import secrets
import subprocess
import sys
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from modules.proof_server.ledger import ledger_digest, pubkey_set_digest  # noqa: E402


def _get_json(url: str, timeout: int = 300):
    with urlopen(url, timeout=timeout) as r:
        return json.loads(r.read())


def _get_bytes(url: str, timeout: int = 300) -> bytes:
    with urlopen(url, timeout=timeout) as r:
        return r.read()


def _post(url: str, timeout: int = 1800) -> dict:
    req = Request(url, data=b"", method="POST")
    with urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proof-server", default="http://127.0.0.1:8040",
                        help="Base URL of the proof server")
    parser.add_argument("--mode", choices=["execute", "prove"], default="execute")
    parser.add_argument("--verify-proof", action="store_true",
                        help="Also run the SP1 verifier on proof.bin (--prove mode only)")
    parser.add_argument("--host-bin", type=Path,
                        default=REPO_ROOT / "modules/proof_server/sp1/target/release/proof-server-host")
    parser.add_argument("--work-dir", type=Path, default=Path("/tmp/proof-server-audit"))
    args = parser.parse_args()

    args.work_dir.mkdir(parents=True, exist_ok=True)
    base = args.proof_server.rstrip("/")

    if args.mode == "execute":
        print("[auditor] NOTE: --mode execute runs the SP1 guest in the RISC-V")
        print("[auditor]       interpreter (assert! panics surface as non-zero exit)")
        print("[auditor]       but does NOT generate a ZK proof. Use --mode prove")
        print("[auditor]       for the cryptographic guarantee.")
    if not args.verify_proof:
        print("[auditor] NOTE: not running the SP1 verifier on proof bytes")
        print("[auditor]       (pass --verify-proof with --mode prove to enable).")
    print("[auditor] NOTE: v0 verifies soundness only — that the rows the proof")
    print("[auditor]       server published are bound by a valid signed Merkle")
    print("[auditor]       root for the auditor's nonce. It does NOT verify")
    print("[auditor]       completeness (no check that the proof server saw")
    print("[auditor]       every envelope the Tap forwarded). See plan.md §8.")

    nonce_hex = secrets.token_hex(32)
    print(f"[auditor] nonce: {nonce_hex[:16]}...")
    print(f"[auditor] requesting commit on {base} (mode={args.mode}) ...")

    qs_commit = urlencode({"nonce": nonce_hex, "mode": args.mode})
    try:
        result = _post(f"{base}/commit?{qs_commit}")
    except HTTPError as exc:
        try:
            err = json.loads(exc.read())
        except Exception:
            err = {"error": "<unparseable>"}
        print(f"[auditor] FAIL: /commit returned HTTP {exc.code}: {err}")
        return 1
    print(f"[auditor] commit: {result}")

    qs_get = urlencode({"nonce": nonce_hex})
    ledger_rows = _get_json(f"{base}/ledger?{qs_get}")
    signer_pubkeys = _get_json(f"{base}/signer_pubkeys?{qs_get}")
    public = _get_json(f"{base}/public_outputs?{qs_get}")

    expected_ledger_digest = ledger_digest(ledger_rows)
    expected_pubkey_set_digest = pubkey_set_digest(signer_pubkeys)

    if public["auditor_nonce"] != nonce_hex:
        print(f"[auditor] FAIL: nonce mismatch (expected {nonce_hex}, got {public['auditor_nonce']})")
        return 1
    if public["ledger_digest"] != expected_ledger_digest:
        print(f"[auditor] FAIL: ledger_digest mismatch (expected {expected_ledger_digest}, "
              f"got {public['ledger_digest']})")
        return 1
    if public["pubkey_set_digest"] != expected_pubkey_set_digest:
        print(f"[auditor] FAIL: pubkey_set_digest mismatch (expected {expected_pubkey_set_digest}, "
              f"got {public['pubkey_set_digest']})")
        return 1
    if public["n_rows"] != len(ledger_rows):
        print(f"[auditor] FAIL: n_rows mismatch (expected {len(ledger_rows)}, got {public['n_rows']})")
        return 1
    if public["n_signers"] != len(signer_pubkeys):
        print(f"[auditor] FAIL: n_signers mismatch (expected {len(signer_pubkeys)}, "
              f"got {public['n_signers']})")
        return 1

    print(f"[auditor] nonce               OK")
    print(f"[auditor] ledger_digest       OK  ({expected_ledger_digest})")
    print(f"[auditor] pubkey_set_digest   OK  ({expected_pubkey_set_digest})")
    print(f"[auditor] n_rows={public['n_rows']}, n_signers={public['n_signers']}")

    if args.verify_proof:
        if args.mode != "prove":
            print(f"[auditor] FAIL: --verify-proof requires --mode prove")
            return 1
        if not args.host_bin.exists():
            print(f"[auditor] FAIL: --verify-proof needs SP1 host binary at {args.host_bin}")
            return 1
        proof_bytes = _get_bytes(f"{base}/proof.bin?{qs_get}")
        proof_path = args.work_dir / "proof.bin"
        proof_path.write_bytes(proof_bytes)
        public_path = args.work_dir / "public_outputs.json"
        public_path.write_text(json.dumps(public, indent=2, sort_keys=True))
        completed = subprocess.run(
            [str(args.host_bin), "--verify",
             "--proof", str(proof_path), "--public", str(public_path)],
            capture_output=True,
            timeout=1800,
        )
        if completed.returncode != 0:
            sys.stderr.write(completed.stderr.decode("utf-8", errors="replace"))
            print(f"[auditor] FAIL: SP1 verifier rejected the proof")
            return 1
        print(f"[auditor] SP1 verify          OK")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
