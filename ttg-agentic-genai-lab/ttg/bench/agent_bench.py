"""Task-level benchmark: run N disruption tasks at several concurrency levels against one engine configuration.

  python -m ttg.bench.agent_bench --label vllm-base --engine vllm --levels 1,4,16,32 --n 48 \
      --gpu-name H100-80GB --gpus 1 --gpu-hourly 2.49 --metrics-url http://localhost:8000/metrics
  python -m ttg.bench.agent_bench --label mock --levels 1,8 --n 16                     # laptop, mock engine
  python -m ttg.bench.agent_bench --label storm --api http://localhost:8080 --levels 64  # M9: through the agent API
Writes results/bench/<label>.json (schema in the lab contract) and prints one row per level.
Why task-level and not just AIPerf: an agent task is 4-12 dependent LLM calls plus tool calls; the customer pays
per handled disruption, and prefix caching / speculative decoding change THAT number in ways a single-request
sweep does not show. AIPerf stays the engine-level microscope (bench/aiperf in the README).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from ttg import world
from ttg.agent.config import Settings
from ttg.bench import scrape_metrics
from ttg.bench.stats import summarize_level


async def _run_inprocess(levels, scenarios_for, settings, warmup, metrics_url, gpu_hourly, gpus, on_level):
    from ttg.agent.graph import Agent
    from ttg.agent.llm import LLMClient
    from ttg.agent.tools import ToolBox
    llm = LLMClient(settings)
    out = []
    async with ToolBox(settings) as tools:
        agent = Agent(settings, llm, tools)
        if warmup:
            await asyncio.gather(*(agent.run(s) for s in world.make_scenarios(warmup, seed=99)))
        for i, c in enumerate(levels):
            out.append(await _level(c, scenarios_for(i), agent.run, metrics_url, gpu_hourly, gpus))
            on_level(out[-1])
    await llm.aclose()
    return out


async def _run_api(levels, scenarios_for, api, warmup, metrics_url, gpu_hourly, gpus, on_level, timeout_s):
    import httpx
    async with httpx.AsyncClient(base_url=api, timeout=timeout_s) as client:
        async def call(sc):
            t0 = time.perf_counter()
            try:
                r = await client.post("/v1/disruptions", json={"scenario": world.public_view(sc)})
                r.raise_for_status()
                res = r.json()
                # the customer waits for the whole HTTP round trip (API queueing included), not the server's view
                res["server_latency_s"] = res.get("latency_s")
                res["latency_s"] = round(time.perf_counter() - t0, 4)
                return res
            except Exception as e:
                return {"scenario_id": sc["scenario_id"], "success": False, "violations": ["error:api"],
                        "steps": 0, "llm_calls": 0, "tool_calls": 0, "retries": 0, "prompt_tokens": None,
                        "completion_tokens": None, "cached_tokens": None,
                        "latency_s": time.perf_counter() - t0, "first_ttft_s": None, "llm_time_s": 0,
                        "tool_time_s": 0, "final": None, "error": f"api: {e}"}
        if warmup:
            await asyncio.gather(*(call(s) for s in world.make_scenarios(warmup, seed=99)))
        out = []
        for i, c in enumerate(levels):
            out.append(await _level(c, scenarios_for(i), call, metrics_url, gpu_hourly, gpus))
            on_level(out[-1])
        return out


RESET_URL = ""


async def _reset_cache():
    """Start every level with a cold prefix cache so earlier runs cannot inflate the hit rate.
    vLLM: POST /reset_prefix_cache (engine started with VLLM_SERVER_DEV_MODE=1); SGLang: /flush_cache; mock: /reset."""
    if not RESET_URL:
        return
    import httpx
    try:
        async with httpx.AsyncClient(timeout=30) as c:
            r = await c.post(RESET_URL)
            r.raise_for_status()
            try:
                body = r.json()
            except ValueError:
                body = None
            if isinstance(body, dict) and body.get("success") is False:     # vLLM: 200 + success=false if blocks held
                print(f"  (cache reset via {RESET_URL} reported success=false: requests still hold KV blocks, so this "
                      "level may start with a warm cache)", flush=True)
    except Exception as e:
        print(f"  (cache reset via {RESET_URL} failed: {e})", flush=True)


async def _level(c, scenarios, runner, metrics_url, gpu_hourly, gpus):
    await _reset_cache()
    before = scrape_metrics.snapshot(metrics_url)
    sem = asyncio.Semaphore(c)

    async def one(sc):
        async with sem:
            return await runner(sc)

    t0 = time.perf_counter()
    results = await asyncio.gather(*(one(s) for s in scenarios))
    wall = time.perf_counter() - t0
    after = scrape_metrics.snapshot(metrics_url)
    eng = scrape_metrics.delta(before, after) if before and after and "error" not in after else None
    summary = summarize_level(results, c, wall, gpu_hourly, gpus, eng)
    return {"summary": summary, "tasks": results}


def _row(s: dict) -> str:
    def f(x, fmt):
        return "-" if x is None else format(x, fmt)
    return (f"c={s['concurrency']:<4} n={s['n']:<4} {s['tasks_per_s']:7.2f} tasks/s  p50 {f(s['p50_latency_s'], '6.2f')}s"
            f"  p95 {f(s['p95_latency_s'], '6.2f')}s  success {s['success_rate']:.0%}  calls {f(s['mean_llm_calls'], '.1f')}"
            f"  prompt {f(s['mean_prompt_tokens'], '.0f')} (cached {f(s['cache_hit_ratio'], '.0%')})"
            f"  out {f(s['mean_completion_tokens'], '.0f')}  retries {f(s['retries_per_task'], '.2f')}"
            f"  $/1k {f(s['usd_per_1k_tasks'], '.3f')}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--label", required=True, help='e.g. "vllm-base" (engine-variant), used as the file name')
    ap.add_argument("--engine", default="", help="vllm | sglang | trtllm | nim | hf | mock (recorded only)")
    ap.add_argument("--variant", default="", help="recorded only; defaults to the part of the label after '-'")
    ap.add_argument("--levels", default="1,4,16", help="comma-separated concurrency levels")
    ap.add_argument("--n", type=int, default=48, help="tasks per level (at least 2x the concurrency is applied)")
    ap.add_argument("--seed", type=int, default=7, help="7 = golden set")
    ap.add_argument("--warmup", type=int, default=2, help="tasks run first and discarded (CUDA graphs, cache)")
    ap.add_argument("--gpu-name", default=os.environ.get("GPU_NAME", ""))
    ap.add_argument("--gpus", type=int, default=int(os.environ.get("GPUS_PER_REPLICA", "1")))
    ap.add_argument("--gpu-hourly", type=float, default=float(os.environ.get("GPU_HOURLY_USD", "0") or 0),
                    help="USD per GPU-hour; 0 = no cost columns")
    ap.add_argument("--metrics-url", default="", help="engine /metrics for server-side prefix-cache hit rate")
    ap.add_argument("--api", default="", help="drive the deployed agent API instead of an in-process agent")
    ap.add_argument("--reset-url", default="", help="POSTed before each level to clear the engine's prefix cache, "
                    "e.g. http://localhost:8000/reset_prefix_cache (vLLM dev mode) or .../flush_cache (SGLang)")
    ap.add_argument("--out", default="")
    a = ap.parse_args(argv)
    levels = [int(x) for x in a.levels.split(",") if x.strip()]
    sizes = [max(a.n, 2 * c) for c in levels]
    pool = world.make_scenarios(max(200, sum(sizes)), a.seed)   # golden set first, then more of the same seed
    starts = [sum(sizes[:i]) for i in range(len(levels))]

    def scenarios_for(i):
        """Fresh travelers per level (by position, so --levels 4,4 gets two different slices): only the shared
        system prompt and each task's own history can hit the prefix cache, as on a real storm day (re-running the
        same tasks would fake a ~100% hit rate)."""
        return pool[starts[i]: starts[i] + sizes[i]]

    global RESET_URL
    RESET_URL = a.reset_url
    settings = Settings().validate()
    print(f"{a.label}: {settings.model} @ {a.api or settings.base_url} | prompt {settings.prompt_variant} | "
          f"MCP {settings.mcp_mode} | levels {levels}")
    levels_out = asyncio.run(
        _run_api(levels, scenarios_for, a.api, a.warmup, a.metrics_url, a.gpu_hourly, a.gpus,
                 lambda r: print(_row(r["summary"]), flush=True), settings.timeout_s * settings.max_steps)
        if a.api else
        _run_inprocess(levels, scenarios_for, settings, a.warmup, a.metrics_url, a.gpu_hourly, a.gpus,
                       lambda r: print(_row(r["summary"]), flush=True)))
    doc = {"label": a.label, "engine": a.engine or a.label.split("-")[0],
           "variant": a.variant or (a.label.split("-", 1)[1] if "-" in a.label else ""),
           "model": settings.model, "gpu": a.gpu_name, "gpus": a.gpus, "gpu_hourly_usd": a.gpu_hourly,
           "base_url": a.api or settings.base_url, "settings": settings.public(),
           "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "levels": [x["summary"] for x in levels_out],
           "tasks": [dict(t, concurrency=x["summary"]["concurrency"]) for x in levels_out for t in x["tasks"]]}
    out = Path(a.out or f"results/bench/{a.label}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=1), encoding="utf-8")
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
