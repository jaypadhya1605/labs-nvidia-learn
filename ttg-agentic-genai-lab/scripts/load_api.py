#!/usr/bin/env python3
"""Storm-day load test THROUGH the deployed agent API (M9): the path real traffic takes - Service, NetworkPolicy,
agent HPA, MCP servers, engine queue. (`python -m ttg.bench.agent_bench` measures agent + engine from your laptop;
this measures the deployment.)

  kubectl -n ttg port-forward svc/ttg-agent 8080:8080 &
  python scripts/load_api.py --url http://localhost:8080 --concurrency 64 --n 256 --label aks-vllm-base

Posts {"scenario_id": "S0001".."S0200"} (the golden set, cycled) to /v1/disruptions and writes
results/load/<label>.json with p50/p95 latency, success rate (policy-compliant answers), throughput and HTTP errors.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

GOLDEN = 200


def percentile(values: list[float], q: float) -> float | None:
    """Nearest-rank percentile, q in 0..100."""
    if not values:
        return None
    s = sorted(values)
    k = max(1, math.ceil(q / 100.0 * len(s)))
    return s[k - 1]


def scenario_ids(n: int, start: int = 1) -> list[str]:
    return [f"S{((start - 1 + i) % GOLDEN) + 1:04d}" for i in range(n)]


def summarize(records: list[dict], wall_s: float, concurrency: int, label: str, url: str) -> dict:
    ok = [r for r in records if r["status"] == 200]
    lat = [r["latency_s"] for r in ok]
    codes: dict[str, int] = {}
    for r in records:
        codes[str(r["status"])] = codes.get(str(r["status"]), 0) + 1
    n = len(records)
    return {
        "label": label, "url": url, "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "concurrency": concurrency, "n": n, "wall_s": round(wall_s, 3),
        "tasks_per_s": round(n / wall_s, 4) if wall_s > 0 else None,
        "success_rate": round(sum(1 for r in records if r.get("success")) / n, 4) if n else None,
        "http_error_rate": round((n - len(ok)) / n, 4) if n else None,
        "p50_latency_s": percentile(lat, 50), "p95_latency_s": percentile(lat, 95),
        "status_codes": codes, "records": records,
    }


async def run(url: str, n: int, concurrency: int, start: int = 1, timeout: float = 300.0,
              transport: httpx.AsyncBaseTransport | None = None) -> tuple[list[dict], float]:
    sem = asyncio.Semaphore(concurrency)
    records: list[dict] = []

    async with httpx.AsyncClient(base_url=url.rstrip("/"), timeout=timeout, transport=transport) as client:
        async def one(sid: str) -> None:
            async with sem:
                t0 = time.perf_counter()
                try:
                    r = await client.post("/v1/disruptions", json={"scenario_id": sid})
                    body = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
                    records.append({"scenario_id": sid, "status": r.status_code,
                                    "latency_s": round(time.perf_counter() - t0, 4),
                                    "success": bool(body.get("success")) if r.status_code == 200 else False,
                                    "violations": body.get("violations", []) if isinstance(body, dict) else []})
                except httpx.HTTPError as e:
                    records.append({"scenario_id": sid, "status": 0, "latency_s": round(time.perf_counter() - t0, 4),
                                    "success": False, "error": type(e).__name__})

        t0 = time.perf_counter()
        await asyncio.gather(*(one(s) for s in scenario_ids(n, start)))
        wall = time.perf_counter() - t0
    return records, wall


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--url", default="http://localhost:8080")
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--concurrency", type=int, default=64)
    ap.add_argument("--start", type=int, default=1, help="first golden-set scenario number")
    ap.add_argument("--label", default="api-load")
    ap.add_argument("--out", default=None, help="default results/load/<label>.json")
    ap.add_argument("--timeout", type=float, default=300.0)
    args = ap.parse_args(argv)
    records, wall = asyncio.run(run(args.url, args.n, args.concurrency, args.start, args.timeout))
    s = summarize(records, wall, args.concurrency, args.label, args.url)
    out = Path(args.out or f"results/load/{args.label}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(s, indent=2) + "\n")
    p50, p95 = s["p50_latency_s"], s["p95_latency_s"]
    print(f"{args.label}: n={s['n']} c={args.concurrency} success={s['success_rate']} http_errors={s['http_error_rate']} "
          f"p50={p50}s p95={p95}s throughput={s['tasks_per_s']} tasks/s -> {out}")
    return 0 if s["http_error_rate"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
