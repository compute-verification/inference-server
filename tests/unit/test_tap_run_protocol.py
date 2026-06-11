"""End-to-end CPU test of the /run workload protocol across all four servers.

Boots the real gateway/tap/host/recomp processes in --mock mode on test ports,
submits every workload through POST /run, and asserts the full loop: async job
completes, the host's signed capture digest survives, the recomp's independent
re-run bitwise-matches (mock harnesses are deterministic), progress events from
BOTH clusters arrive on the tap's bus, and the gateway serves the run's task
graph. A second fixture with --force-run-divergence asserts the alarm path.
"""
import json
import subprocess
import sys
import time
import unittest
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SERVERS = REPO_ROOT / "demos" / "tap-protocol" / "servers"

GW, TAP, HOST, RECOMP = 28100, 28110, 28120, 28130
GW2, TAP2, HOST2, RECOMP2 = 28200, 28210, 28220, 28230


def _get(url: str):
    with urllib.request.urlopen(url, timeout=30) as r:
        return r.status, json.loads(r.read())


def _post(url: str, body: dict):
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(), method="POST",
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.status, json.loads(r.read())


def _wait_health(port: int, deadline: float = 30.0) -> None:
    end = time.monotonic() + deadline
    while time.monotonic() < end:
        try:
            code, _ = _get(f"http://127.0.0.1:{port}/health")
            if code == 200:
                return
        except Exception:
            pass
        time.sleep(0.2)
    raise AssertionError(f"server on :{port} never became healthy")


def _boot_stack(tmp: Path, gw: int, tap: int, host: int, recomp: int,
                recomp_extra: list[str] = ()) -> list[subprocess.Popen]:
    py = sys.executable
    procs = [
        subprocess.Popen([py, str(SERVERS / "host_cluster.py"), "--port", str(host),
                          "--mock", "--tap-url", f"http://127.0.0.1:{tap}",
                          "--out-dir", str(tmp / "host")],
                         stdout=(tmp / "host.log").open("w"), stderr=subprocess.STDOUT),
        subprocess.Popen([py, str(SERVERS / "recomp_cluster.py"), "--port", str(recomp),
                          "--mock", "--tap-url", f"http://127.0.0.1:{tap}",
                          "--out-dir", str(tmp / "recomp"), *recomp_extra],
                         stdout=(tmp / "recomp.log").open("w"), stderr=subprocess.STDOUT),
        subprocess.Popen([py, str(SERVERS / "tap.py"), "--port", str(tap),
                          "--host-url", f"http://127.0.0.1:{host}",
                          "--recomp-url", f"http://127.0.0.1:{recomp}"],
                         stdout=(tmp / "tap.log").open("w"), stderr=subprocess.STDOUT),
        subprocess.Popen([py, str(SERVERS / "gateway.py"), "--port", str(gw),
                          "--tap-url", f"http://127.0.0.1:{tap}"],
                         stdout=(tmp / "gateway.log").open("w"), stderr=subprocess.STDOUT),
    ]
    for port in (host, recomp, tap, gw):
        _wait_health(port)
    return procs


def _run_to_verdict(gw: int, tap: int, workload: str, params: dict,
                    deadline: float = 90.0) -> tuple[dict, list[dict]]:
    """POST /run, wait for the job AND its recomp_verified event."""
    _, sub = _post(f"http://127.0.0.1:{gw}/run",
                   {"workload": workload, "params": params})
    rid = sub["id"]
    end = time.monotonic() + deadline
    job = None
    while time.monotonic() < end:
        _, job = _get(f"http://127.0.0.1:{gw}/run/{rid}")
        if job["status"] in ("done", "failed"):
            break
        time.sleep(0.3)
    assert job and job["status"] == "done", f"job never finished: {job}"
    verdict_evt = None
    while time.monotonic() < end and verdict_evt is None:
        _, evs = _get(f"http://127.0.0.1:{tap}/capture")
        verdict_evt = next((e for e in evs if e["type"] == "recomp_verified"
                            and e["id"] == rid), None)
        if verdict_evt is None:
            time.sleep(0.3)
    assert verdict_evt is not None, "recomp_verified never arrived"
    _, evs = _get(f"http://127.0.0.1:{tap}/capture")
    return job, [e for e in evs if e["id"] == rid]


class TestRunProtocolAllWorkloads(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import tempfile
        cls.tmp = Path(tempfile.mkdtemp(prefix="tap_run_proto_"))
        cls.procs = _boot_stack(cls.tmp, GW, TAP, HOST, RECOMP)

    @classmethod
    def tearDownClass(cls):
        for p in cls.procs:
            p.terminate()
        for p in cls.procs:
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                p.kill()

    def test_all_four_workloads_verify_green(self):
        for wl, params in [("inference", {"prompt": "hi", "max_tokens": 4}),
                           ("spec", {"prompt": "hi there", "max_tokens": 8, "k": 3}),
                           ("training", {}),
                           ("coding", {})]:
            with self.subTest(workload=wl):
                job, evs = _run_to_verdict(GW, TAP, wl, params)
                self.assertEqual(job["workload"], wl)
                self.assertTrue(job["capture_digest"].startswith("sha256:"))

                verdict = next(e for e in evs if e["type"] == "recomp_verified")
                self.assertTrue(verdict["is_verified"],
                                f"{wl}: {verdict.get('reason')}")
                self.assertEqual(verdict["expected_sha256"], job["capture_digest"])
                self.assertEqual(verdict["actual_sha256"], job["capture_digest"])

                types = [e["type"] for e in evs]
                for t in ("request_sent", "gateway_signed", "tap_received",
                          "tap_relayed_request", "host_completed",
                          "tap_relayed_response", "client_received",
                          "tap_verify_started", "recomp_verified"):
                    self.assertIn(t, types, f"{wl}: missing lifecycle event {t}")
                self.assertIn("host_progress", types, f"{wl}: no host progress")
                self.assertIn("recomp_progress", types, f"{wl}: no recomp progress")
                # progress arrives between relay and completion, both legs tagged
                hp = next(e for e in evs if e["type"] == "host_progress")
                self.assertEqual(hp["workload"], wl)

    def test_gateway_serves_capture_and_graph(self):
        job, _ = _run_to_verdict(GW, TAP, "inference",
                                 {"prompt": "graph me", "max_tokens": 3})
        rid = job["id"]
        _, cap = _get(f"http://127.0.0.1:{GW}/run/{rid}/capture")
        self.assertIn("events", cap)
        _, graphdoc = _get(f"http://127.0.0.1:{GW}/run/{rid}/graph")
        self.assertIn("inference", graphdoc)
        g = graphdoc["inference"]
        self.assertEqual(len(g["nodes"]), 3)
        self.assertTrue(all(n["flops"] > 0 for n in g["nodes"]))
        self.assertIn("shapes", g)

    def test_unknown_workload_and_bad_param_rejected_at_submit(self):
        import urllib.error
        for body in ({"workload": "mining"},
                     {"workload": "inference", "params": {"temperature": 2}},
                     {"workload": "inference", "params": {"max_tokens": "lots"}}):
            with self.assertRaises(urllib.error.HTTPError) as ctx:
                _post(f"http://127.0.0.1:{GW}/run", body)
            self.assertEqual(ctx.exception.code, 400)

    def test_status_endpoint_does_not_leak_full_capture(self):
        job, _ = _run_to_verdict(GW, TAP, "inference",
                                 {"prompt": "small", "max_tokens": 3})
        self.assertNotIn("capture", job)
        self.assertIn("summary", job)

    def test_event_ingest_rejects_lifecycle_types(self):
        # the tap synthesizes lifecycle events itself; /event must only
        # accept the two progress types or verdicts become spoofable
        import urllib.error
        for spoof in ("recomp_verified", "host_completed", "request_sent",
                      "run_failed"):
            with self.assertRaises(urllib.error.HTTPError) as ctx:
                _post(f"http://127.0.0.1:{TAP}/event",
                      {"type": spoof, "id": 1, "is_verified": True})
            self.assertEqual(ctx.exception.code, 400, spoof)
        # the allowed types still work
        code, _ = _post(f"http://127.0.0.1:{TAP}/event",
                        {"type": "host_progress", "id": 999,
                         "workload": "inference", "progress": {"type": "token"}})
        self.assertEqual(code, 200)

    def test_digest_claim_mismatch_alarms(self):
        # a host that ships a capture not hashing to its own claimed digest
        # is internally bogus -- recomp must refuse WITHOUT re-running
        sys.path.insert(0, str(REPO_ROOT / "demos" / "tap-protocol"))
        from servers.envelope import sign
        req_env = sign({"workload": "inference",
                        "params": {"prompt": "x", "max_tokens": 3}}, 7001)
        resp_env = sign({"workload": "inference",
                         "capture_digest": "sha256:" + "0" * 64,
                         "summary": {}, "capture": {"events": [], "shapes": {}}},
                        7001)
        code, verdict = _post(f"http://127.0.0.1:{RECOMP}/verify_run",
                              {"request_data": req_env.model_dump(),
                               "response_data": resp_env.model_dump()})
        self.assertEqual(code, 200)
        self.assertFalse(verdict["is_verified"])
        self.assertEqual(verdict["reason"], "digest_claim_mismatch")
        alarm = self.tmp / "recomp" / "alarm.jsonl"
        self.assertTrue(alarm.exists())
        rec = json.loads(alarm.read_text().splitlines()[-1])
        self.assertEqual(rec["reason"], "digest_claim_mismatch")
        self.assertEqual(rec["id"], 7001)

    def test_host_failure_emits_terminal_run_failed_event(self):
        # invalid params POSTed straight to the tap (skipping the gateway's
        # submit-time validation) make the host's build_argv fail -> the tap
        # must emit a TERMINAL event or a watching client spins forever
        import urllib.error
        sys.path.insert(0, str(REPO_ROOT / "demos" / "tap-protocol"))
        from servers.envelope import sign
        req_env = sign({"workload": "inference", "params": {"bogus": 1}}, 7002)
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            _post(f"http://127.0.0.1:{TAP}/run", req_env.model_dump())
        self.assertEqual(ctx.exception.code, 502)
        _, evs = _get(f"http://127.0.0.1:{TAP}/capture")
        failed = [e for e in evs if e["type"] == "run_failed" and e["id"] == 7002]
        self.assertEqual(len(failed), 1)
        self.assertEqual(failed[0]["reason"], "host_http_400")


class TestForcedDivergenceAlarms(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import tempfile
        cls.tmp = Path(tempfile.mkdtemp(prefix="tap_run_diverge_"))
        cls.procs = _boot_stack(cls.tmp, GW2, TAP2, HOST2, RECOMP2,
                                recomp_extra=["--force-run-divergence"])

    @classmethod
    def tearDownClass(cls):
        for p in cls.procs:
            p.terminate()
        for p in cls.procs:
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                p.kill()

    def test_divergent_rerun_raises_alarm_with_forensics(self):
        job, evs = _run_to_verdict(GW2, TAP2, "inference",
                                   {"prompt": "diverge", "max_tokens": 4})
        verdict = next(e for e in evs if e["type"] == "recomp_verified")
        self.assertFalse(verdict["is_verified"])
        self.assertEqual(verdict["reason"], "capture_digest_mismatch")
        self.assertNotEqual(verdict["actual_sha256"], verdict["expected_sha256"])

        alarm_path = self.tmp / "recomp" / "alarm.jsonl"
        self.assertTrue(alarm_path.exists(), "alarm.jsonl was not written")
        rec = json.loads(alarm_path.read_text().splitlines()[-1])
        self.assertEqual(rec["reason"], "capture_digest_mismatch")
        self.assertEqual(rec["id"], job["id"])
        mismatches = list((self.tmp / "recomp" / "mismatches").glob("*.json"))
        names = {p.name.split(".", 1)[1] for p in mismatches}
        self.assertEqual(names, {"host.json", "recomp.json"})


if __name__ == "__main__":
    unittest.main()
