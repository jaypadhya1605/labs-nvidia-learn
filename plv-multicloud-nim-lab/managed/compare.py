#!/usr/bin/env python3
"""M10: one scorecard for self-hosted NIM and managed endpoints - conformance, agentic behaviour, latency, cost.
  cp managed/endpoints.example.yaml managed/endpoints.yaml   # enable entries, export the API key env vars
  python3 managed/compare.py --endpoints managed/endpoints.yaml --out results/managed-scorecard.md
  python3 managed/compare.py --mock          # dry run against the in-process mock (CI)
Cost per 1,000 summaries: pay-per-token = measured tokens x list price; self-hosted = $/hour / measured capacity,
where capacity comes from your M5 sweep (pass --self-hosted-summaries-per-hour) - never from this quick probe.
"""
import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "app"), str(ROOT / "tests")]
from agentic import harness  # noqa: E402
from plover_api import prompts, schema, synth  # noqa: E402


def client(ep, mock):
    if mock:
        from fastapi.testclient import TestClient
        from mock_openai import app
        return TestClient(app)
    import httpx
    key = os.environ.get(ep.get("api_key_env") or "", "")
    return httpx.Client(base_url=ep["base_url"].rstrip("/"), headers={"Authorization": f"Bearer {key}"} if key else {}, timeout=300)


def conformance(ep, mock):
    if mock:
        cmd = [sys.executable, "-m", "pytest", "-q", str(ROOT / "tests" / "conformance"), "--model", ep["model"]]
    else:
        cmd = [sys.executable, "-m", "pytest", "-q", str(ROOT / "tests" / "conformance"), "--endpoint", ep["base_url"],
               "--model", ep["model"], "--extra-body", json.dumps(ep.get("extra_body") or {})]
        if ep.get("api_key_env"):
            cmd += ["--api-key", os.environ.get(ep["api_key_env"], "")]
    out = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT).stdout.strip().splitlines()
    return out[-1] if out else "no result"


def summaries(c, ep, n):
    lat, tin, tout, valid, halluc = [], [], [], 0, 0
    for case in synth.cases(n, 7000):
        t0 = time.perf_counter()
        r = c.post("/v1/chat/completions", json={"model": ep["model"], "messages": prompts.build_messages(case["text"], "long-prompt"),
                                                  "temperature": 0.0, "max_tokens": 500, **(ep.get("extra_body") or {})})
        lat.append(time.perf_counter() - t0)
        if r.status_code != 200:
            continue
        d = r.json()
        u = d.get("usage") or {}
        tin.append(u.get("prompt_tokens", 0))
        tout.append(u.get("completion_tokens", 0))
        obj = schema.extract_json(d["choices"][0]["message"].get("content") or "")
        bad_codes = bool(schema.unfaithful_codes(obj, case["text"]))
        halluc += int(bad_codes)
        valid += int(not schema.validate(obj) and not bad_codes)
    lat.sort()
    return {"n": n, "schema_valid": valid / n, "hallucinated_code_rate": halluc / n, "latency_p50_s": lat[len(lat) // 2], "latency_p95_s": lat[int(0.95 * (len(lat) - 1))],
            "prompt_tokens": statistics.mean(tin) if tin else 0, "completion_tokens": statistics.mean(tout) if tout else 0}


def cost_per_1k(ep, s, sph):
    if ep.get("usd_per_1m_in") is not None and ep.get("usd_per_1m_out") is not None:
        return (s["prompt_tokens"] * ep["usd_per_1m_in"] + s["completion_tokens"] * ep["usd_per_1m_out"]) / 1e6 * 1000
    if ep.get("self_hosted_usd_per_hour") and sph:
        return ep["self_hosted_usd_per_hour"] / sph * 1000
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--endpoints", default=str(ROOT / "managed" / "endpoints.example.yaml"))
    ap.add_argument("--n", type=int, default=20, help="summaries per endpoint for latency/quality/cost")
    ap.add_argument("--self-hosted-summaries-per-hour", type=float, default=None, help="from your M5 sweep at the SLO")
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    eps = yaml.safe_load(Path(a.endpoints).read_text())
    if a.mock:
        eps = [dict(e, enabled=True) for e in eps if e["kind"] == "self-hosted"][:1] + \
              [{"name": "mock-hosted", "kind": "hosted-pay-per-token", "model": "nvidia/nemotron-3.5-lightning-30b-a3b",
                "base_url": "mock", "usd_per_1m_in": 0.025, "usd_per_1m_out": 0.12, "enabled": True}]
    rows = []
    for ep in [e for e in eps if e.get("enabled")]:
        with client(ep, a.mock) as c:
            s = summaries(c, ep, a.n)
            ag, _ = harness.evaluate(c, ep["model"], ep.get("extra_body"), n_per_kind=2)
        rows.append({"name": ep["name"], "kind": ep["kind"], "model": ep["model"], "conformance": conformance(ep, a.mock), **s,
                     "agentic_task_success": ag["task_success"], "injection_success": ag["injection_success"],
                     "usd_per_1k_summaries": cost_per_1k(ep, s, a.self_hosted_summaries_per_hour)})
    md = ["| Endpoint | Kind | Conformance | Schema-valid | p50 / p95 s | Agentic | Injection | $/1k summaries |",
          "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        usd = f"{r['usd_per_1k_summaries']:.3f}" if r["usd_per_1k_summaries"] is not None else "needs M5 capacity"
        md.append(f"| {r['name']} | {r['kind']} | {r['conformance']} | {r['schema_valid']:.0%} | {r['latency_p50_s']:.2f} / "
                  f"{r['latency_p95_s']:.2f} | {r['agentic_task_success']:.0%} | {r['injection_success']:.0%} | {usd} |")
    text = "\n".join(md) + "\n\nSynthetic data only. Hosted latency includes the public Internet; self-hosted is in-region.\n"
    print(text)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(text)
        Path(a.out).with_suffix(".json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
