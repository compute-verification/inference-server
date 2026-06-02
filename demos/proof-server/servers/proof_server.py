"""Proof server — the developer-controlled proxy between the datacenter
(``demos/tap-protocol``'s Tap) and the auditor.

The Tap forwards every verified ``(request_env, response_env)`` pair here.
The proof server appends both envelopes' ``EnvelopeData`` to its append-only
ledger. On ``POST /commit?nonce=...`` it (a) builds a Merkle root over the
rows committed so far, (b) signs the root with its own Ed25519 keypair,
(c) shells out to ``proof-server-host`` (execute or prove), (d) pins the
rows + pubkey set + public outputs to that nonce. Subsequent reads at
``GET /ledger?nonce=...`` / ``/signer_pubkeys?nonce=...`` /
``/public_outputs?nonce=...`` serve from the pinned snapshot so the
auditor sees the same bytes the SP1 program committed to.

The auditor never talks to the Tap, the gateways, or anything else inside
the datacenter -- it only reads from this server.

What this proves (and doesn't), v0
----------------------------------
The proof server is the single signer in v0; it holds the only Ed25519
keypair. The SP1 program proves "I (the proof server) know a valid
signature under my own key over a Merkle root containing every leaf I
published to the auditor for this nonce." That's a *self-attestation*
binding the auditor's nonce, the rows the auditor will fetch, and the
two public digests together — it does NOT independently bind those rows
to anything the gateways or Tap observed. A v1 with per-gateway keys
would extend the witness array to verify multiple signers; the SP1
program is already parameterised over ``signer_idx``.

The completeness direction — "did the proof server publish ALL the
envelopes the Tap saw?" — is explicitly out of scope; see
``demos/proof-server/plan.md`` §8.

Tap-copy authentication
-----------------------
The Tap HMAC-signs every envelope it forwards (PR #19's ``SignedEnvelope``).
``_handle_tap_copy`` re-verifies that HMAC before recording anything,
so a process that can reach the localhost ``/tap-copy`` port still
cannot inject rows without the shared HMAC key. This is integrity on the
loopback boundary, not strong authentication; see plan.md §8.

Subliminal-channel defense
--------------------------
The raw Ed25519 signature over the Merkle root stays inside this process.
The SP1 program is the only thing that ever sees it, and the program's
public outputs do not include it. Even before that hiding step,
deterministic Ed25519 leaves zero attacker-influenceable bits in the
signature itself. The residual auditor-visible channel is the
``ledger_digest`` (collision-bounded under SHA-256) plus the proof bytes
themselves in ``--prove`` mode (SP1 proof encoding is deterministic
given the witness + public inputs).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import signal
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from socketserver import ThreadingMixIn
from urllib.parse import urlparse, parse_qs

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from modules.core.common.deterministic import canonical_json_bytes  # noqa: E402
from modules.proof_server.api import (  # noqa: E402
    assemble_witness,
    build_merkle_tree,
    keypair_from_seed,
    leaf_hash,
    pubkey_hex,
    sign_root,
)


# Load demos/tap-protocol/servers/envelope.py by absolute file path so we
# can re-verify the inner HMAC without polluting sys.path with a hyphen-
# named directory (which would collide with our own ``servers`` package).
# Register in sys.modules and rebuild the Pydantic model so the
# ``from __future__ import annotations`` forward references inside the
# tap-protocol module resolve correctly.
_TAP_ENV_PATH = REPO_ROOT / "demos" / "tap-protocol" / "servers" / "envelope.py"
_spec = importlib.util.spec_from_file_location("_tap_protocol_envelope", _TAP_ENV_PATH)
_tap_env = importlib.util.module_from_spec(_spec)
sys.modules["_tap_protocol_envelope"] = _tap_env
_spec.loader.exec_module(_tap_env)  # type: ignore[union-attr]
_tap_env.SignedEnvelope.model_rebuild()
_TapSignedEnvelope = _tap_env.SignedEnvelope
_verify_tap_envelope = _tap_env.verify


# Fixed seed so the demo is reproducible. Real deployments would generate
# the keypair once at provisioning time and hold it in attestable hardware.
DEMO_SEED = b"proof-server-demo-seed----------"
assert len(DEMO_SEED) == 32

# Default cap on how long a single /commit's SP1 host invocation may run.
# Tuned for execute mode on modest hardware; prove mode needs more.
DEFAULT_SP1_TIMEOUT_S = 1200


class _Ledger:
    """Thread-safe append-only ledger of EnvelopeData rows (the published
    scrubbed projection the auditor will see).

    Dedup is keyed by the canonical-JSON leaf hash, NOT by any in-row id
    field — a tap-protocol request and its response share the same Gateway-
    assigned ``id`` but are distinct rows that must both appear in the
    ledger.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._rows: list[dict] = []
        # Map leaf-hash (hex) -> leaf index. Used by assemble_witness later.
        self._leaf_index_of: dict[str, int] = {}

    def append(self, env_data: dict) -> None:
        with self._lock:
            key = leaf_hash(env_data).hex()
            if key in self._leaf_index_of:
                # Idempotent: the Tap may resend the same envelope on retry.
                return
            self._leaf_index_of[key] = len(self._rows)
            self._rows.append(env_data)

    def snapshot(self) -> tuple[list[dict], dict[str, int]]:
        with self._lock:
            return list(self._rows), dict(self._leaf_index_of)

    def __len__(self) -> int:
        with self._lock:
            return len(self._rows)


class _PublicState:
    """Map ``auditor_nonce -> {public_outputs, ledger_snapshot, signer_pubkeys,
    proof_path}``.

    Pinning the snapshot by nonce closes the race the adversarial review
    flagged: an envelope arriving on a `_handle_tap_copy` thread between
    `_Ledger.snapshot()` and the auditor's `GET /ledger` would otherwise
    show up in the published ledger but not in the SP1 program's view,
    breaking digest equality. With the pinned-by-nonce surface, the auditor
    always fetches the exact rows / pubkeys / proof tied to their commit.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._by_nonce: dict[str, dict] = {}

    def set(self, *, nonce_hex: str, public_outputs: dict,
            proof_path: Path | None, ledger_snapshot: list[dict],
            signer_pubkeys: list[str]) -> None:
        with self._lock:
            self._by_nonce[nonce_hex] = {
                "public_outputs": public_outputs,
                "proof_path": proof_path,
                "ledger_snapshot": ledger_snapshot,
                "signer_pubkeys": signer_pubkeys,
            }

    def get(self, nonce_hex: str) -> dict | None:
        with self._lock:
            return self._by_nonce.get(nonce_hex)


def _is_hex_nonce(s: str) -> bool:
    return len(s) == 64 and all(c in "0123456789abcdef" for c in s.lower())


class ProofServerHandler(BaseHTTPRequestHandler):
    """The proxy's HTTP surface. Class-level attributes are mutated by ``main``."""

    ledger: _Ledger = None  # type: ignore[assignment]
    public_state: _PublicState = None  # type: ignore[assignment]
    sk = None
    pk = None
    pk_hex: str = ""
    host_bin: Path = None  # type: ignore[assignment]
    work_dir: Path = None  # type: ignore[assignment]
    sp1_timeout_s: int = DEFAULT_SP1_TIMEOUT_S

    def _send_json(self, code: int, body) -> None:
        payload = json.dumps(body).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _send_bytes(self, code: int, ctype: str, payload: bytes) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _record_for_nonce(self, query: str) -> dict | None:
        params = parse_qs(query)
        nonce_hex = (params.get("nonce") or [""])[0].lower()
        if not _is_hex_nonce(nonce_hex):
            self._send_json(400, {"error": "missing or malformed ?nonce= (need 64 hex chars)"})
            return None
        record = self.public_state.get(nonce_hex)
        if record is None:
            self._send_json(404, {"error": "no commit for that nonce"})
            return None
        return record

    # ----------------------------------------------------------------------
    # GET surface (the auditor's read-only window)
    # ----------------------------------------------------------------------

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        if path == "/health":
            return self._send_json(200, {"status": "ok", "rows": len(self.ledger)})
        if path == "/ledger":
            record = self._record_for_nonce(parsed.query)
            if record is None:
                return
            return self._send_bytes(200, "application/json",
                                    canonical_json_bytes(record["ledger_snapshot"]))
        if path == "/signer_pubkeys":
            record = self._record_for_nonce(parsed.query)
            if record is None:
                return
            return self._send_bytes(200, "application/json",
                                    canonical_json_bytes(record["signer_pubkeys"]))
        if path == "/public_outputs":
            record = self._record_for_nonce(parsed.query)
            if record is None:
                return
            return self._send_json(200, record["public_outputs"])
        if path == "/proof.bin":
            record = self._record_for_nonce(parsed.query)
            if record is None:
                return
            pp = record["proof_path"]
            if pp is None or not pp.exists():
                return self._send_json(404, {"error": "no proof; commit was --execute, not --prove"})
            return self._send_bytes(200, "application/octet-stream", pp.read_bytes())
        return self._send_json(404, {"error": f"not found: {path}"})

    # ----------------------------------------------------------------------
    # POST surface (what the Tap and the auditor write to)
    # ----------------------------------------------------------------------

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        try:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length) if length else b""
        except Exception as exc:  # noqa: BLE001
            return self._send_json(400, {"error": f"bad body: {exc}"})

        if path == "/tap-copy":
            return self._handle_tap_copy(raw)
        if path == "/commit":
            return self._handle_commit(parsed.query)
        return self._send_json(404, {"error": f"not found: {path}"})

    def _handle_tap_copy(self, raw: bytes) -> None:
        try:
            body = json.loads(raw or b"{}")
            req_env_raw = body["request_data"]
            resp_env_raw = body["response_data"]
        except Exception as exc:  # noqa: BLE001
            return self._send_json(400, {"error": f"bad tap-copy: {exc}"})

        # Re-verify the inner HMAC envelopes the Tap signed. This re-check
        # is integrity for the loopback boundary -- any process with the
        # shared HMAC key (which is hardcoded in tap-protocol/envelope.py)
        # can still post here, but a process that lacks the key cannot.
        try:
            req_env = _TapSignedEnvelope.model_validate(req_env_raw)
            resp_env = _TapSignedEnvelope.model_validate(resp_env_raw)
        except Exception as exc:  # noqa: BLE001
            return self._send_json(400, {"error": f"bad envelope shape: {exc}"})
        if not _verify_tap_envelope(req_env):
            return self._send_json(401, {"error": "bad request envelope HMAC"})
        if not _verify_tap_envelope(resp_env):
            return self._send_json(401, {"error": "bad response envelope HMAC"})

        # The published row is the inner EnvelopeData dict; the HMAC signature
        # is consumed here and never crosses to the auditor.
        try:
            self.ledger.append(req_env.data.model_dump())
            self.ledger.append(resp_env.data.model_dump())
        except Exception as exc:  # noqa: BLE001
            return self._send_json(400, {"error": f"could not record: {exc}"})

        return self._send_json(200, {"status": "ok", "rows": len(self.ledger)})

    def _handle_commit(self, query: str) -> None:
        params = parse_qs(query)
        nonce_hex = (params.get("nonce") or [""])[0].lower()
        if not _is_hex_nonce(nonce_hex):
            return self._send_json(400, {"error": "missing or malformed ?nonce= (need 64 hex chars)"})
        mode = (params.get("mode") or ["execute"])[0]
        if mode not in {"execute", "prove"}:
            return self._send_json(400, {"error": f"bad mode {mode!r}; must be execute or prove"})

        # Single snapshot used for both the SP1 witness and the auditor's
        # later GET fetches; pinned to this nonce so a later /tap-copy
        # cannot make the auditor see different bytes than SP1 saw.
        rows, leaf_index_of = self.ledger.snapshot()
        if not rows:
            return self._send_json(409, {"error": "ledger is empty -- nothing to commit"})

        signer_pubkeys = [self.pk_hex]

        try:
            leaves = [leaf_hash(r) for r in rows]
            tree = build_merkle_tree(leaves)
            signature = sign_root(self.sk, tree.root)

            witnesses = assemble_witness(
                rows, signer_pubkeys,
                tree=tree,
                leaf_index_of=leaf_index_of,
                signed_root=tree.root,
                signature=signature,
                signer_pubkey_hex=self.pk_hex,
            )
            stdin_json = {
                "auditor_nonce": nonce_hex,
                "signer_pubkeys": signer_pubkeys,
                "ledger_rows_canon_hex": [
                    w.row_canonical_json.rstrip(b"\n").hex() for w in witnesses
                ],
                "witnesses": [
                    {
                        "signer_idx": w.signer_idx,
                        "leaf_index": w.leaf_index,
                        "merkle_path_hex": [s.hex() for s in w.merkle_path],
                        "signed_root": w.signed_root.hex(),
                        "signature": w.signature.hex(),
                    }
                    for w in witnesses
                ],
            }
        except Exception as exc:  # noqa: BLE001
            return self._send_json(500, {"error": f"witness assembly failed: {exc}"})

        if not self.host_bin.exists():
            return self._send_json(500, {
                "error": f"SP1 host binary not found at {self.host_bin}; "
                         f"build it with PROTOC=... cargo build --release --manifest-path "
                         f"modules/proof_server/sp1/host/Cargo.toml"
            })

        # In --prove mode each nonce gets its own proof file so a later
        # commit doesn't overwrite an earlier audit's bytes.
        proof_out = (self.work_dir / f"proof-{nonce_hex}.bin") if mode == "prove" else None
        args = [str(self.host_bin)]
        if mode == "execute":
            args.append("--execute")
        else:
            args.extend(["--prove", "--proof", str(proof_out)])

        try:
            completed = subprocess.run(
                args,
                input=json.dumps(stdin_json).encode("utf-8"),
                capture_output=True,
                check=False,
                timeout=self.sp1_timeout_s,
            )
        except subprocess.TimeoutExpired:
            return self._send_json(504, {
                "error": f"SP1 host exceeded {self.sp1_timeout_s}s timeout",
            })
        if completed.returncode != 0:
            err = completed.stderr.decode("utf-8", errors="replace")
            return self._send_json(500, {
                "error": f"SP1 host exited {completed.returncode}",
                "stderr_tail": err.splitlines()[-10:],
            })

        try:
            stdout = completed.stdout.decode("utf-8", errors="replace").strip()
            public = json.loads(stdout.splitlines()[-1])
        except Exception as exc:  # noqa: BLE001
            return self._send_json(500, {"error": f"unparseable SP1 output: {exc}"})

        self.public_state.set(
            nonce_hex=nonce_hex,
            public_outputs=public,
            proof_path=proof_out if proof_out and proof_out.exists() else None,
            ledger_snapshot=rows,
            signer_pubkeys=signer_pubkeys,
        )
        return self._send_json(200, {
            "status": "ok",
            "mode": mode,
            "n_rows": public.get("n_rows"),
            "ledger_digest": public.get("ledger_digest"),
        })

    def log_message(self, format, *args):  # noqa: A002
        sys.stderr.write("[proof-server] " + (format % args) + "\n")


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8040)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--host-bin", type=Path,
                        default=REPO_ROOT / "modules/proof_server/sp1/target/release/proof-server-host",
                        help="Path to the compiled proof-server-host binary")
    parser.add_argument("--work-dir", type=Path, default=Path("/tmp/proof-server-state"),
                        help="Where to write per-nonce proof.bin files in --prove mode")
    parser.add_argument("--sp1-timeout", type=int, default=DEFAULT_SP1_TIMEOUT_S,
                        help="Per-commit SP1 host subprocess timeout, seconds")
    args = parser.parse_args()

    args.work_dir.mkdir(parents=True, exist_ok=True)

    sk, pk = keypair_from_seed(DEMO_SEED)
    ProofServerHandler.sk = sk
    ProofServerHandler.pk = pk
    ProofServerHandler.pk_hex = pubkey_hex(pk)
    ProofServerHandler.ledger = _Ledger()
    ProofServerHandler.public_state = _PublicState()
    ProofServerHandler.host_bin = args.host_bin
    ProofServerHandler.work_dir = args.work_dir
    ProofServerHandler.sp1_timeout_s = args.sp1_timeout

    server = ThreadedHTTPServer((args.host, args.port), ProofServerHandler)

    def _shutdown(signum, frame):  # noqa: ARG001
        sys.stderr.write("[proof-server] shutting down\n")
        threading.Thread(target=server.shutdown, daemon=True).start()
    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    print(f"[proof-server] listening on {args.host}:{args.port}; "
          f"pubkey={ProofServerHandler.pk_hex[:16]}...; host_bin={args.host_bin}")
    sys.stdout.flush()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
