"""Run the Disruption Concierge on golden-set scenarios.

  python -m ttg.agent.run --scenario S0003             # one task, step by step (thought -> action -> observation)
  python -m ttg.agent.run --n 20 --concurrency 4       # a batch: success rate, latency, tokens
  python -m ttg.agent.run --n 5 --json                 # machine-readable results
Endpoint, model and behavior come from TTG_* environment variables (see ttg/agent/config.py).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys

from ttg import world
from ttg.agent.config import Settings
from ttg.agent.graph import Agent
from ttg.agent.llm import LLMClient
from ttg.agent.tools import ToolBox


def _short(x, n=150):
    s = json.dumps(x, ensure_ascii=False) if not isinstance(x, str) else x
    return s if len(s) <= n else s[: n - 3] + "..."


def print_trace(path: str) -> None:
    t = json.load(open(path, encoding="utf-8"))
    for m in t["messages"][2:]:
        role = "LLM " if m["role"] == "assistant" else "TOOL" if m["content"].startswith("OBSERVATION") else "USER"
        print(f"  {role} {_short(m['content'])}")


async def run_many(scenarios: list, concurrency: int, settings: Settings) -> list:
    llm = LLMClient(settings)
    async with ToolBox(settings) as tools:
        agent = Agent(settings, llm, tools)
        sem = asyncio.Semaphore(concurrency)

        async def one(sc):
            async with sem:
                return await agent.run(sc)

        results = await asyncio.gather(*(one(sc) for sc in scenarios))
    await llm.aclose()
    return results


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", help="scenario id from the golden set, e.g. S0003")
    ap.add_argument("--n", type=int, default=1, help="first N golden-set scenarios")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    pool = world.make_scenarios(max(a.n, 200), a.seed)
    if a.scenario:
        pool = [s for s in pool if s["scenario_id"] == a.scenario]
        if not pool:
            print(f"unknown scenario {a.scenario}", file=sys.stderr)
            return 2
        os.environ.setdefault("TTG_TRACE_DIR", "results/traces")
    else:
        pool = pool[: a.n]
    settings = Settings().validate()
    results = asyncio.run(run_many(pool, a.concurrency, settings))
    if a.json:
        print(json.dumps(results, indent=1))
        return 0
    for r in results:
        mark = "PASS" if r["success"] else "FAIL"
        print(f"{r['scenario_id']} {mark} {r['latency_s']:6.2f}s llm_calls={r['llm_calls']} tools={r['tool_calls']} "
              f"prompt={r['prompt_tokens']} cached={r['cached_tokens']} out={r['completion_tokens']} "
              f"retries={r['retries']} {' '.join(r['violations'])}")
        if a.scenario and r.get("trace"):
            print_trace(r["trace"])
            print(f"  FINAL {_short(r['final'], 400)}\n  trace: {r['trace']}")
    ok = sum(r["success"] for r in results)
    print(f"\n{ok}/{len(results)} policy-compliant | model {settings.model} @ {settings.base_url} | "
          f"prompt {settings.prompt_variant} | MCP {settings.mcp_mode}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
