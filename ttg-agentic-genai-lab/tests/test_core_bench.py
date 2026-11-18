"""Task-level bench, statistics, ledger, metrics scraping, AIPerf export and the agent API."""
import json
import os
import subprocess
import sys
import time

import httpx
import pytest

from ttg import world
from ttg.agent.prompts import build_system_prompt
from ttg.bench import agent_bench, export_aiperf, ledger, scrape_metrics
from ttg.bench.stats import pct, summarize_level, usd_per_1k


def test_unreported_cached_tokens_are_unknown_not_zero():
    rs = [{"success": True, "violations": [], "latency_s": 1.0, "first_ttft_s": 0.1, "llm_calls": 6,
           "prompt_tokens": 1000, "cached_tokens": None, "completion_tokens": 50, "retries": 0, "error": None}] * 4
    s = summarize_level(rs, 4, 2.0, engine_cache={"hit_ratio": 0.8})
    assert s["cache_hit_ratio"] is None and s["engine_prefix_hit_ratio"] == 0.8
    row = ledger.build([{"label": "trtllm-base", "gpus": 1, "levels": [s]}], 12, 0.95,
                       {"storm_tasks": 40000, "storm_hours": 6, "peak_factor": 2})[0]
    assert row["cache_hit_ratio"] == 0.8 and row["cache_hit_source"] == "engine-metrics"


def test_stats():
    assert pct([3, 1, 2], 0.5) == 2 and pct([1, 2, 3, 4], 0.95) == pytest.approx(3.85) and pct([], 0.5) is None
    assert usd_per_1k(2.0, 3.0, 1) == pytest.approx(3.0 / 7200 * 1000) and usd_per_1k(0, 3, 1) is None
    rs = [{"success": i % 4 != 0, "violations": ["wrong_amount:meal:1!=2"] if i % 4 == 0 else [], "latency_s": 1 + i,
           "first_ttft_s": 0.1, "llm_calls": 6, "prompt_tokens": 1000, "cached_tokens": 800, "completion_tokens": 50,
           "retries": 0, "error": None} for i in range(8)]
    s = summarize_level(rs, 4, 4.0, gpu_hourly_usd=3.6, gpus=2)
    assert s["tasks_per_s"] == 2.0 and s["success_rate"] == 0.75 and s["cache_hit_ratio"] == 0.8
    assert s["usd_per_1k_tasks"] == pytest.approx(1.0) and s["top_violations"] == {"wrong_amount": 2}
    assert s["tasks_per_gpu_hour"] == 3600.0


def test_agent_bench_writes_the_contract_schema(agent_env, tmp_path):
    out = tmp_path / "b.json"
    assert agent_bench.main(["--label", "mock-base", "--levels", "1,4", "--n", "8", "--gpu-hourly", "2.5",
                             "--gpu-name", "mock", "--metrics-url", f"{agent_env}/metrics",
                             "--reset-url", f"{agent_env}/reset", "--out", str(out)]) == 0
    doc = json.loads(out.read_text())
    assert {"label", "engine", "variant", "model", "gpu", "gpus", "gpu_hourly_usd", "base_url", "created", "levels",
            "tasks"} <= set(doc)
    assert doc["engine"] == "mock" and doc["variant"] == "base" and [lv["concurrency"] for lv in doc["levels"]] == [1, 4]
    lv = doc["levels"][1]
    for k in ("n", "wall_s", "tasks_per_s", "success_rate", "p50_latency_s", "p95_latency_s", "mean_llm_calls",
              "mean_prompt_tokens", "mean_completion_tokens", "cache_hit_ratio", "retries_per_task", "usd_per_1k_tasks"):
        assert lv[k] is not None, k
    assert lv["success_rate"] == 1.0 and 0 < lv["engine_prefix_hit_ratio"] <= 1
    ids = [t["scenario_id"] for t in doc["tasks"]]
    assert len(ids) == len(set(ids)) == 16                     # fresh travelers per level


def _doc(label, levels, gpus=1, hourly=2.0):
    return {"label": label, "engine": label.split("-")[0], "variant": label.split("-")[1], "gpu": "H100", "gpus": gpus,
            "gpu_hourly_usd": hourly, "levels": levels}


def _lv(c, tps, p95, ok=1.0):
    return {"concurrency": c, "tasks_per_s": tps, "p95_latency_s": p95, "success_rate": ok,
            "usd_per_1k_tasks": usd_per_1k(tps, 2.0, 1), "cache_hit_ratio": 0.9, "mean_prompt_tokens": 15000,
            "mean_completion_tokens": 270}


def test_ledger_operating_point_and_outputs(tmp_path):
    docs = [_doc("hf-base", [_lv(1, 0.05, 30)]),
            _doc("vllm-base", [_lv(1, 0.4, 3), _lv(16, 3.0, 8), _lv(64, 5.0, 20)]),
            _doc("vllm-spec", [_lv(16, 3.5, 7), _lv(32, 5.5, 11.5, ok=0.9)])]
    storm = {"storm_tasks": 40000, "storm_hours": 6, "peak_factor": 2}
    rows = ledger.build(docs, 12, 0.95, storm, ["hf-base", "vllm-base", "vllm-spec"])
    by = {r["label"]: r for r in rows}
    assert by["hf-base"]["meets_slo"] is False
    assert by["vllm-base"]["concurrency"] == 16 and by["vllm-spec"]["concurrency"] == 16   # 32 fails success
    assert by["vllm-base"]["x_capacity_vs_first"] == 60.0
    assert by["vllm-base"]["storm_replicas"] == 2 and by["vllm-base"]["storm_gpus_n_plus_1"] == 3
    for d in docs:
        (tmp_path / f"{d['label']}.json").write_text(json.dumps(d))
    assert ledger.main([str(p) for p in sorted(tmp_path.glob("*.json"))] +
                       ["--order", "hf-base,vllm-base,vllm-spec", "--out", str(tmp_path / "out")]) == 0
    md = (tmp_path / "out" / "ledger.md").read_text()
    assert "| vllm-spec |" in md and (tmp_path / "out" / "ledger.csv").exists()
    if pytest.importorskip("matplotlib"):
        assert (tmp_path / "out" / "waterfall.png").stat().st_size > 5000


VLLM = """# HELP vllm:prefix_cache_queries Prefix cache queries
# TYPE vllm:prefix_cache_queries_total counter
vllm:prefix_cache_queries_total{engine="0",model_name="ttg-llm"} 1000.0
vllm:prefix_cache_hits_total{engine="0",model_name="ttg-llm"} 750.0
vllm:num_requests_waiting{engine="0",model_name="ttg-llm"} 3.0
"""
SGLANG = 'sglang:cache_hit_rate{model_name="ttg-llm"} 0.62\nsglang:num_queue_reqs{model_name="ttg-llm"} 5\n'
TRTLLM = '[{"iter": 1, "kvCacheStats": {"reusedBlocks": 30, "missedBlocks": 10, "maxNumBlocks": 900}}]'


def test_metrics_parsing_per_engine():
    v = scrape_metrics.cache_stats(VLLM)
    assert v["engine"] == "vllm" and v["hit_ratio"] == 0.75 and v["waiting"] == 3.0
    s = scrape_metrics.cache_stats(SGLANG)
    assert s["engine"] == "sglang" and s["hit_ratio"] == 0.62 and s["waiting"] == 5
    t = scrape_metrics.cache_stats(TRTLLM)
    assert t["engine"] == "trtllm" and t["hit_ratio"] == 0.75
    # several queued iteration records: the counters are lifetime totals, so the NEWEST record wins (no summing)
    two = json.dumps([{"iter": 7, "kvCacheStats": {"reusedBlocks": 30, "missedBlocks": 10}},
                      {"iter": 6, "kvCacheStats": {"reusedBlocks": 10, "missedBlocks": 10}}])
    t2 = scrape_metrics.cache_stats(two)
    assert (t2["hits"], t2["queries"], t2["hit_ratio"]) == (30, 40, 0.75)
    before = scrape_metrics.cache_stats(json.dumps([{"iter": 6, "kvCacheStats": {"reusedBlocks": 10,
                                                                                  "missedBlocks": 10}}]))
    assert scrape_metrics.delta(before, t2)["hit_ratio"] == 1.0                   # the window: 20 reused of 20
    assert scrape_metrics.cache_stats("[]")["hit_ratio"] is None                  # drained queue: unknown
    later = scrape_metrics.cache_stats(VLLM.replace("1000.0", "1400.0").replace("750.0", "1100.0"))
    assert scrape_metrics.delta(v, later)["hit_ratio"] == pytest.approx(350 / 400)
    assert scrape_metrics.cache_stats("other_metric 1\n")["engine"] == "unknown"


def test_aiperf_export(tmp_path):
    assert export_aiperf.main(["--n", "5", "--out", str(tmp_path)]) == 0
    assert (tmp_path / "system_prompt.txt").read_text() == build_system_prompt()
    sessions = [json.loads(x) for x in (tmp_path / "sessions.jsonl").read_text().splitlines()]
    for s, sc in zip(sessions, world.make_scenarios(5, 7)):
        assert s["session_id"] == sc["scenario_id"]
        assert len(s["turns"]) == len(world.expected_plan(sc))          # traveler + one observation per tool call
        assert s["turns"][1]["text"].startswith("OBSERVATION flight_status")


def _free_port():
    import socket
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture
def api(agent_env, tmp_path):
    port = _free_port()
    env = {k: os.environ[k] for k in ("TTG_BASE_URL", "TTG_MODEL", "TTG_MCP_MODE")}
    proc = _start_api(port, env)
    yield f"http://127.0.0.1:{port}"
    proc.terminate()
    proc.wait(timeout=10)


def _start_api(port, env):
    full = {**os.environ, "PYTHONPATH": str(world.REPO), **env}
    proc = subprocess.Popen([sys.executable, "-m", "uvicorn", "ttg.agent.api:app", "--port", str(port),
                             "--log-level", "warning"], cwd=world.REPO, env=full, stdout=subprocess.DEVNULL,
                            stderr=subprocess.PIPE)
    deadline = time.time() + 30
    while time.time() < deadline:
        try:
            if httpx.get(f"http://127.0.0.1:{port}/healthz", timeout=1).status_code == 200:
                return proc
        except httpx.HTTPError:
            time.sleep(0.2)
    proc.kill()
    raise RuntimeError(proc.stderr.read().decode()[-2000:])


def test_agent_api(api):
    r = httpx.post(f"{api}/v1/disruptions", json={"scenario_id": "S0007"}, timeout=60).json()
    assert r["success"] and r["scenario_id"] == "S0007"
    pub = world.public_view(world.make_scenarios(9, 7)[8])
    assert httpx.post(f"{api}/v1/disruptions", json={"scenario": pub}, timeout=60).json()["success"]
    assert httpx.post(f"{api}/v1/disruptions", json={"scenario_id": "S9999"}, timeout=10).status_code == 404
    assert httpx.post(f"{api}/v1/disruptions", json={"scenario": {"flight_no": "TN1"}}, timeout=10).status_code == 422
    m = httpx.get(f"{api}/metrics", timeout=5).text
    assert "ttg_tasks_total 2" in m and 'ttg_task_latency_seconds_bucket{le="+Inf"} 2' in m
    assert agent_bench.main(["--label", "api-storm", "--api", api, "--levels", "4", "--n", "8",
                             "--out", str(world.REPO / "results" / "bench" / "_test_api.json")]) == 0
    (world.REPO / "results" / "bench" / "_test_api.json").unlink()
