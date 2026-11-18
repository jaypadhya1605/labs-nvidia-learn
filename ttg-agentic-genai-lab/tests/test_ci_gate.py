"""Release-gate logic (scripts/gate.py) and the two CI helpers (scripts/wait_for.py, scripts/load_api.py).

Offline and fast. The last test runs the REAL CI chain - mock engine -> agent bench -> eval gate - so the gate is
checked against what ttg.bench.agent_bench actually writes, not against a hand-made fixture alone.
"""
from __future__ import annotations

import asyncio
import http.server
import importlib.util
import json
import os
import socket
import subprocess
import sys
import threading
import xml.etree.ElementTree as ET
from pathlib import Path

import httpx
import pytest

REPO = Path(__file__).resolve().parents[1]


def _load(name: str):
    """scripts/ is not a package: import a script file as a module."""
    spec = importlib.util.spec_from_file_location(f"ttg_scripts_{name}", REPO / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod          # dataclasses resolve annotations through sys.modules
    spec.loader.exec_module(mod)
    return mod


gate = _load("gate")
wait_for = _load("wait_for")
load_api = _load("load_api")


def level(c, success=1.0, p95=10.0, usd=0.5, tps=1.0, retries=0.1, n=40):
    return {"concurrency": c, "n": n, "success_rate": success, "p95_latency_s": p95, "p50_latency_s": p95 / 2,
            "usd_per_1k_tasks": usd, "tasks_per_s": tps, "retries_per_task": retries}


def bench(levels, gpu="NVIDIA H100 80GB HBM3", engine="vllm", variant="base", tasks=True, hourly=2.49):
    doc = {"label": f"{engine}-{variant}", "engine": engine, "variant": variant, "model": "ttg-llm", "gpu": gpu,
           "gpus": 1, "gpu_hourly_usd": hourly, "levels": levels}
    if tasks:
        doc["tasks"] = [{"scenario_id": "S0001", "success": True, "concurrency": levels[0]["concurrency"]}]
    return doc


def write(tmp_path, name, doc):
    p = tmp_path / name
    p.write_text(json.dumps(doc))
    return str(p)


def by_name(checks):
    return {c.name: c for c in checks}


# ------------------------------------------------------------------------------------------------ input validation
def test_validate_bench_rejects_bad_shapes():
    for bad, msg in [([], "top level"), ({}, "levels"), ({"levels": []}, "levels"),
                     ({"levels": [{"concurrency": 4, "success_rate": 1.0}]}, "p95_latency_s")]:
        with pytest.raises(gate.InputError, match=msg):
            gate.validate_bench(bad)


def test_load_bench_missing_and_not_json(tmp_path):
    with pytest.raises(gate.InputError, match="no such bench file"):
        gate.load_bench(tmp_path / "nope.json")
    p = tmp_path / "broken.json"
    p.write_text("{not json")
    with pytest.raises(gate.InputError, match="not JSON"):
        gate.load_bench(p)


# ------------------------------------------------------------------------------------------------------ eval gate
def test_eval_gate_passes_at_the_floor_and_fails_below():
    assert all(c.ok for c in gate.eval_gate(bench([level(4, success=0.95)]), min_success=0.95))
    checks = gate.eval_gate(bench([level(1, success=1.0), level(4, success=0.9)]), min_success=0.95)
    assert [c.ok for c in checks] == [True, False]
    assert "0.900 vs floor 0.950" in checks[1].detail


def test_eval_gate_retries_ceiling_and_skip():
    checks = by_name(gate.eval_gate(bench([level(4, retries=0.6)]), max_retries_per_task=0.5))
    assert not checks["eval c=4 retries_per_task"].ok
    lv = level(4)
    del lv["retries_per_task"]
    checks = by_name(gate.eval_gate(bench([lv]), max_retries_per_task=0.5))
    assert checks["eval c=4 retries_per_task"].skipped and checks["eval c=4 retries_per_task"].ok


def test_eval_gate_single_level_and_unknown_level():
    checks = gate.eval_gate(bench([level(1, success=0.5), level(4, success=1.0)]), level=4)
    assert len(checks) == 1 and checks[0].ok
    with pytest.raises(gate.InputError, match="concurrency=16"):
        gate.eval_gate(bench([level(4)]), level=16)


def test_eval_gate_non_numeric_success_is_input_error():
    with pytest.raises(gate.InputError, match="not a number"):
        gate.eval_gate(bench([level(4, success="high")]))


# ------------------------------------------------------------------------------------------------------ perf gate
def test_perf_gate_within_budget_passes():
    base = bench([level(1, p95=10.0, usd=1.0), level(4, p95=12.0, usd=0.5)])
    cand = bench([level(1, p95=10.9, usd=1.09), level(4, p95=11.0, usd=0.4)])
    checks = gate.perf_gate(cand, base)
    assert all(c.ok for c in checks), [c.detail for c in checks if not c.ok]


def test_perf_gate_p95_and_cost_regressions_fail():
    base = bench([level(4, p95=10.0, usd=1.0)])
    checks = by_name(gate.perf_gate(bench([level(4, p95=11.5, usd=1.2)]), base))
    assert not checks["perf c=4 p95_latency_s"].ok and "+15.0%" in checks["perf c=4 p95_latency_s"].detail
    assert not checks["perf c=4 usd_per_1k_tasks"].ok


def test_perf_gate_fast_but_wrong_engine_fails():
    base = bench([level(4, p95=10.0)])
    checks = by_name(gate.perf_gate(bench([level(4, p95=5.0, success=0.8)]), base))
    assert checks["perf c=4 p95_latency_s"].ok
    assert not checks["perf c=4 success_rate"].ok


def test_perf_gate_unpriced_cost_is_skipped_not_failed():
    base = bench([level(4, usd=0)], hourly=0)
    checks = by_name(gate.perf_gate(bench([level(4, usd=9.9)]), base))
    assert checks["perf c=4 usd_per_1k_tasks"].skipped and checks["perf c=4 usd_per_1k_tasks"].ok


def test_perf_gate_throughput_drop_is_opt_in():
    base = bench([level(4, tps=2.0)])
    cand = bench([level(4, tps=1.5)])
    assert "perf c=4 tasks_per_s" not in by_name(gate.perf_gate(cand, base))
    checks = by_name(gate.perf_gate(cand, base, max_throughput_drop=0.10))
    assert not checks["perf c=4 tasks_per_s"].ok


def test_perf_gate_compares_common_levels_only_and_warns_on_config_drift():
    base = bench([level(1), level(4), level(16)])
    cand = bench([level(4), level(32)], engine="sglang")
    checks = gate.perf_gate(cand, base)
    assert {c.name.split()[1] for c in checks} == {"c=4"}
    assert any("engine differs" in w for w in checks[0].warnings)


def test_perf_gate_gpu_mismatch_and_no_common_level_are_input_errors():
    base = bench([level(4)], gpu="NVIDIA H100 80GB HBM3")
    with pytest.raises(gate.InputError, match="GPU differs"):
        gate.perf_gate(bench([level(4)], gpu="NVIDIA A100 80GB PCIe"), base)
    assert gate.perf_gate(bench([level(4)], gpu="NVIDIA A100 80GB PCIe"), base, allow_gpu_mismatch=True)
    with pytest.raises(gate.InputError, match="no concurrency level in common"):
        gate.perf_gate(bench([level(8)]), base)


# ------------------------------------------------------------------------------------------- baselines + JUnit + CLI
def test_make_baseline_needs_a_reason_and_drops_tasks(monkeypatch):
    with pytest.raises(gate.InputError, match="--reason"):
        gate.make_baseline(bench([level(4)]), "short")
    monkeypatch.setenv("CI_COMMIT_SHA", "abc123")
    out = gate.make_baseline(bench([level(4)]), "first H100 baseline for vLLM 0.31.0")
    assert "tasks" not in out and out["levels"][0]["concurrency"] == 4
    assert out["baseline"]["git_sha"] == "abc123" and out["baseline"]["reason"].startswith("first H100")


def test_junit_is_valid_xml_with_counts(tmp_path):
    checks = [gate.Check("a", True, "fine"), gate.Check("b", False, 'p95 "slow" <10%>'),
              gate.Check("c", True, "unpriced", skipped=True)]
    path = tmp_path / "junit" / "gate.xml"
    gate.write_junit(checks, path, "gate-perf")
    suite = ET.parse(path).getroot()
    assert suite.tag == "testsuite" and suite.get("tests") == "3"
    assert suite.get("failures") == "1" and suite.get("skipped") == "1"
    assert suite.find("testcase[@name='b']/failure").get("message") == 'p95 "slow" <10%>'


def test_cli_exit_codes(tmp_path, capsys):
    good = write(tmp_path, "good.json", bench([level(1), level(4)]))
    bad = write(tmp_path, "bad.json", bench([level(4, success=0.5)]))
    assert gate.main(["eval", good, "--junit", str(tmp_path / "eval.xml")]) == 0
    assert (tmp_path / "eval.xml").is_file()
    assert gate.main(["eval", bad]) == 1
    assert gate.main(["eval", str(tmp_path / "missing.json")]) == 2

    base = str(tmp_path / "baselines" / "vllm-base-h100.json")
    assert gate.main(["perf", good, "--baseline", base]) == 2                     # no baseline yet
    assert "gate.py baseline" in capsys.readouterr().err                          # ...and how to record one
    assert gate.main(["baseline", good, "--out", base, "--reason", "first baseline, H100 SXM"]) == 0
    assert "tasks" not in json.loads(Path(base).read_text())
    assert gate.main(["perf", good, "--baseline", base]) == 0
    slow = write(tmp_path, "slow.json", bench([level(1, p95=20.0), level(4, p95=20.0)]))
    assert gate.main(["perf", slow, "--baseline", base, "--junit", str(tmp_path / "perf.xml")]) == 1
    other_gpu = write(tmp_path, "l4.json", bench([level(4)], gpu="NVIDIA L4"))
    assert gate.main(["perf", other_gpu, "--baseline", base]) == 2


# ----------------------------------------------------------------------------------------------------- wait_for.py
def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class _NotFound(http.server.BaseHTTPRequestHandler):
    def do_GET(self):                                  # /mcp answers a GET with 405/406: still "up"
        self.send_response(405)
        self.end_headers()

    def log_message(self, *args):
        pass


def test_wait_for_up_and_down():
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _NotFound)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        up = f"http://127.0.0.1:{srv.server_address[1]}/mcp"
        down = f"http://127.0.0.1:{_free_port()}/mcp"
        assert wait_for.is_up(up)
        assert wait_for.wait([up], timeout=5) == []
        assert wait_for.wait([up, down], timeout=1, interval=0.2) == [down]
        assert wait_for.main([up, "--timeout", "2"]) == 0
        assert wait_for.main([down, "--timeout", "0.5"]) == 1
    finally:
        srv.shutdown()


# ----------------------------------------------------------------------------------------------------- load_api.py
def test_percentile_nearest_rank():
    assert load_api.percentile([], 95) is None
    assert load_api.percentile([3.0], 50) == 3.0
    vals = [float(i) for i in range(1, 101)]
    assert load_api.percentile(vals, 50) == 50.0
    assert load_api.percentile(vals, 95) == 95.0
    assert load_api.percentile(vals, 100) == 100.0


def test_scenario_ids_cycle_the_golden_set():
    ids = load_api.scenario_ids(3, start=199)
    assert ids == ["S0199", "S0200", "S0001"]
    assert len(set(load_api.scenario_ids(400))) == 200


def test_run_and_summarize_with_mock_transport():
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen.append(body["scenario_id"])
        if body["scenario_id"] == "S0003":
            return httpx.Response(503, text="overloaded")
        return httpx.Response(200, json={"scenario_id": body["scenario_id"], "success": body["scenario_id"] != "S0002",
                                         "violations": [] if body["scenario_id"] != "S0002" else ["wrong_amount"]})

    records, wall = asyncio.run(load_api.run("http://agent.test", n=4, concurrency=2,
                                             transport=httpx.MockTransport(handler)))
    assert sorted(seen) == ["S0001", "S0002", "S0003", "S0004"]
    s = load_api.summarize(records, wall, 2, "unit", "http://agent.test")
    assert s["n"] == 4 and s["status_codes"] == {"200": 3, "503": 1}
    assert s["success_rate"] == 0.5 and s["http_error_rate"] == 0.25
    assert s["p95_latency_s"] is not None and s["tasks_per_s"] > 0


# --------------------------------------------------------------------------- the real CI chain: bench -> eval gate
def test_eval_gate_on_a_real_bench_run(mock_engine, tmp_path):
    """What the eval-gate job does, scaled down: the agent runs golden-set tasks against the mock engine, the bench
    writes its JSON, gate.py must accept the schema and pass the policy floor."""
    out = tmp_path / "ci-mock.json"
    env = {**os.environ, "PYTHONPATH": str(REPO), "TTG_BASE_URL": f"{mock_engine}/v1", "TTG_MODEL": "ttg-llm",
           "TTG_MCP_MODE": "inproc", "TTG_TRACE_DIR": str(tmp_path / "traces")}
    run = subprocess.run([sys.executable, "-m", "ttg.bench.agent_bench", "--label", "ci-mock", "--engine", "mock",
                          "--levels", "2", "--n", "4", "--warmup", "0", "--out", str(out)],
                         cwd=REPO, env=env, capture_output=True, text=True, timeout=300)
    assert run.returncode == 0, run.stdout[-2000:] + run.stderr[-2000:]
    data = gate.load_bench(out)                      # the bench's real schema passes validation
    assert data["engine"] == "mock" and data["levels"][0]["concurrency"] == 2
    assert gate.main(["eval", str(out), "--min-success", "0.95", "--junit", str(tmp_path / "eval.xml")]) == 0
