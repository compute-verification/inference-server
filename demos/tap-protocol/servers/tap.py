"""Tap: relays SignedEnvelope between Gateway and Host Cluster.

After the response goes back to the Gateway, fires a daemon thread that POSTs
the (request, response) pair to the Recomp Cluster's /verify. Failures are
logged but do not propagate -- verification is async.

Also maintains an in-process ring buffer of protocol events and exposes them
at GET /events (SSE stream) and GET /capture (JSON snapshot). The Tap sees
every protocol step on its own — it receives the signed request from the
Gateway, forwards it to Host, relays the signed response back, and makes
the async /verify call — so it can synthesize the full event stream without
the other servers needing to participate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import signal
import sys
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from socketserver import ThreadingMixIn
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEMO_DIR = Path(__file__).resolve().parent.parent
if str(DEMO_DIR) not in sys.path:
    sys.path.insert(0, str(DEMO_DIR))

from servers.envelope import SignedEnvelope, verify


# ---------------------------------------------------------------------------
# Event ring buffer
# ---------------------------------------------------------------------------

EVENT_BUFFER_SIZE = 4096  # ~4k events; one full request is ~9 events


class EventBus:
    """Thread-safe ring buffer + SSE subscriber fan-out for protocol events.

    `events` is a deque keyed by sequence number; subscribers are notified
    via a Condition variable. Each event is a dict with at minimum
    {seq, ts, type, id}.
    """

    def __init__(self) -> None:
        self.events: deque = deque(maxlen=EVENT_BUFFER_SIZE)
        self._seq = 0
        self._lock = threading.Lock()
        self._cv = threading.Condition(self._lock)

    def emit(self, type_: str, id_: int, **payload) -> None:
        evt = {"ts": time.time(), "type": type_, "id": id_, **payload}
        with self._cv:
            self._seq += 1
            evt["seq"] = self._seq
            self.events.append(evt)
            self._cv.notify_all()
        # Helpful stderr breadcrumb when running interactively.
        sys.stderr.write(f"[tap.event] {type_} id={id_} {json.dumps(payload, default=str)}\n")

    def snapshot(self, since_seq: int = 0) -> list[dict]:
        with self._lock:
            return [e for e in self.events if e["seq"] > since_seq]

    def wait_for_new(self, since_seq: int, timeout: float) -> list[dict]:
        with self._cv:
            deadline = time.monotonic() + timeout
            while True:
                fresh = [e for e in self.events if e["seq"] > since_seq]
                if fresh:
                    return fresh
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return []
                self._cv.wait(timeout=remaining)


BUS = EventBus()


def _output_preview(env_dict: dict, n: int = 140) -> str:
    """Extract a short preview of the inference output from a response envelope dict."""
    try:
        out = env_dict["data"]["payload"]["output"] or ""
    except Exception:
        return ""
    if len(out) <= n:
        return out
    return out[: n - 1] + "…"


def _output_sha256(env_dict: dict) -> str:
    try:
        out = env_dict["data"]["payload"]["output"] or ""
    except Exception:
        return ""
    return "sha256:" + hashlib.sha256(out.encode("utf-8")).hexdigest()


def _signature_prefix(env_dict: dict, n: int = 12) -> str:
    sig = env_dict.get("signature") or ""
    return sig[:n]


# ---------------------------------------------------------------------------
# Async verify (now emits events to the bus)
# ---------------------------------------------------------------------------

def _async_verify(recomp_url: str, request_env: dict, response_env: dict, env_id: int) -> None:
    """Fire-and-forget POST to recomp /verify. Logs verdict; ignores failures."""
    BUS.emit("tap_verify_started", env_id)
    try:
        body = json.dumps({
            "request_data": request_env,
            "response_data": response_env,
        }).encode("utf-8")
        req = Request(
            f"{recomp_url}/verify",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(req, timeout=600) as resp:
            verdict = json.loads(resp.read())
        sys.stderr.write(f"[tap] verify verdict for id={env_id}: {verdict}\n")
        # Surface the verdict on the event bus.
        BUS.emit(
            "recomp_verified",
            env_id,
            is_verified=bool(verdict.get("is_verified", False)),
            expected_sha256=_output_sha256(response_env),
            actual_sha256=verdict.get("actual_sha256") or _output_sha256(response_env),
            reason=verdict.get("reason"),
        )
    except HTTPError as exc:
        sys.stderr.write(f"[tap] verify HTTP {exc.code}: {exc.reason}\n")
        BUS.emit("recomp_verified", env_id, is_verified=False, reason=f"http_{exc.code}")
    except URLError as exc:
        sys.stderr.write(f"[tap] verify unreachable: {exc.reason}\n")
        BUS.emit("recomp_verified", env_id, is_verified=False, reason=f"unreachable")
    except Exception as exc:  # noqa: BLE001
        sys.stderr.write(f"[tap] verify failed: {exc}\n")
        BUS.emit("recomp_verified", env_id, is_verified=False, reason=str(exc))


# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------

class TapHandler(BaseHTTPRequestHandler):
    host_url: str = ""
    recomp_url: str = ""

    def _send_json(self, code: int, body) -> None:
        payload = json.dumps(body).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        # CORS so a Surge-deployed viewer can fetch /capture cross-origin.
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(payload)

    def _send_sse_headers(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("X-Accel-Buffering", "no")  # disable proxy buffering
        self.end_headers()

    def do_GET(self) -> None:
        if self.path == "/health":
            return self._send_json(200, {"status": "ok"})

        if self.path.startswith("/capture"):
            # /capture or /capture?since=N — return all buffered events as
            # one JSON array.
            try:
                qs = self.path.split("?", 1)[1] if "?" in self.path else ""
                since = 0
                for kv in qs.split("&"):
                    if kv.startswith("since="):
                        since = int(kv.split("=", 1)[1])
            except Exception:
                since = 0
            return self._send_json(200, BUS.snapshot(since_seq=since))

        if self.path.startswith("/events"):
            return self._handle_sse()

        return self._send_json(404, {"error": "not found"})

    def _handle_sse(self) -> None:
        """SSE stream: emit existing events then block waiting for new ones."""
        try:
            self._send_sse_headers()
            since = 0
            # Replay any already-buffered events so a late subscriber sees history.
            for evt in BUS.snapshot(since_seq=0):
                self._sse_write(evt)
                since = max(since, evt["seq"])
            # Block waiting for new events; flush each.
            while True:
                fresh = BUS.wait_for_new(since_seq=since, timeout=15.0)
                if not fresh:
                    # heartbeat comment keeps clients alive through proxies
                    try:
                        self.wfile.write(b": heartbeat\n\n")
                        self.wfile.flush()
                    except (BrokenPipeError, ConnectionResetError):
                        return
                    continue
                for evt in fresh:
                    try:
                        self._sse_write(evt)
                    except (BrokenPipeError, ConnectionResetError):
                        return
                    since = max(since, evt["seq"])
        except (BrokenPipeError, ConnectionResetError):
            return

    def _sse_write(self, evt: dict) -> None:
        self.wfile.write(b"data: " + json.dumps(evt).encode("utf-8") + b"\n\n")
        self.wfile.flush()

    def do_POST(self) -> None:
        if self.path != "/request":
            return self._send_json(404, {"error": "not found"})
        try:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length)
            body = json.loads(raw)
            req_env = SignedEnvelope.model_validate(body)
        except Exception as exc:  # noqa: BLE001
            return self._send_json(400, {"error": f"bad envelope: {exc}"})

        if not verify(req_env):
            return self._send_json(401, {"error": "bad request signature"})

        env_id = req_env.data.id
        req_env_dict = req_env.model_dump()
        # The Tap can synthesize the upstream events: by the time we have a
        # validated signed envelope here, the Gateway just sent it (so
        # request_sent + gateway_signed + tap_received all logically
        # happen at "now"). We emit them in sequence so the viewer can
        # show the visible animation, not because they happened apart.
        prompt = ""
        try:
            prompt = req_env_dict["data"]["payload"].get("prompt", "")
        except Exception:
            pass
        BUS.emit("request_sent", env_id, prompt=prompt)
        BUS.emit("gateway_signed", env_id, signature_prefix=_signature_prefix(req_env_dict))
        BUS.emit("tap_received", env_id)
        BUS.emit("tap_relayed_request", env_id)

        # Forward verbatim to host cluster
        try:
            outbound = Request(
                f"{self.host_url}/request",
                data=json.dumps(req_env_dict).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urlopen(outbound, timeout=300) as resp:
                resp_body = json.loads(resp.read())
        except HTTPError as exc:
            try:
                err_body = exc.read().decode("utf-8", errors="replace")
            except Exception:
                err_body = ""
            return self._send_json(502, {"error": f"host returned HTTP {exc.code}", "body": err_body})
        except URLError as exc:
            return self._send_json(502, {"error": f"host unreachable: {exc.reason}"})
        except Exception as exc:  # noqa: BLE001
            return self._send_json(502, {"error": f"host call failed: {exc}"})

        try:
            resp_env = SignedEnvelope.model_validate(resp_body)
        except Exception as exc:  # noqa: BLE001
            return self._send_json(502, {"error": f"bad response envelope: {exc}"})

        if not verify(resp_env):
            return self._send_json(401, {"error": "bad response signature"})

        resp_env_dict = resp_env.model_dump()
        BUS.emit(
            "host_completed",
            env_id,
            output_preview=_output_preview(resp_env_dict),
            output_sha256=_output_sha256(resp_env_dict),
        )
        BUS.emit("tap_relayed_response", env_id)

        # Return the verified response envelope to the Gateway first.
        self._send_json(200, resp_env_dict)
        BUS.emit("client_received", env_id, output_preview=_output_preview(resp_env_dict))

        # Then spawn the async verification tap-copy.
        threading.Thread(
            target=_async_verify,
            args=(self.recomp_url, req_env_dict, resp_env_dict, env_id),
            daemon=True,
        ).start()

    def log_message(self, format, *args):  # noqa: A002
        sys.stderr.write("[tap] " + (format % args) + "\n")


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def main() -> int:
    parser = argparse.ArgumentParser(description="Tap-protocol Tap")
    parser.add_argument("--port", type=int, default=8010)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--host-url", default="http://127.0.0.1:8020")
    parser.add_argument("--recomp-url", default="http://127.0.0.1:8030")
    args = parser.parse_args()

    TapHandler.host_url = args.host_url.rstrip("/")
    TapHandler.recomp_url = args.recomp_url.rstrip("/")

    server = ThreadedHTTPServer((args.host, args.port), TapHandler)

    def _shutdown(signum, frame):  # noqa: ARG001
        sys.stderr.write("[tap] shutting down\n")
        threading.Thread(target=server.shutdown, daemon=True).start()
    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    print(f"[tap] listening on {args.host}:{args.port}; host={TapHandler.host_url}; recomp={TapHandler.recomp_url}")
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
