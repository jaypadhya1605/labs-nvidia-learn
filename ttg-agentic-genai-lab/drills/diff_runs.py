#!/usr/bin/env python3
"""What changed, and what moved? Compare a good and a bad agent_bench run (M13 drills).

  python drills/diff_runs.py results/bench/mock-base.json results/bench/mock-drill1.json

Prints the agent settings that differ (recorded by agent_bench), the metrics at every concurrency level both runs
measured, and the most common errors and violations in the second run. Standard library only.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter

METRICS = ["success_rate", "tasks_per_s", "p95_latency_s", "p95_first_ttft_s", "mean_llm_calls", "retries_per_task",
           "cache_hit_ratio", "mean_prompt_tokens", "mean_completion_tokens", "usd_per_1k_tasks"]


def load(path: str) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def diff(a: dict, b: dict) -> dict:
    sa, sb = a.get("settings") or {}, b.get("settings") or {}
    changed = {k: (sa.get(k), sb.get(k)) for k in sorted(set(sa) | set(sb)) if sa.get(k) != sb.get(k)}
    for k in ("engine", "variant", "model", "gpu", "gpus"):
        if a.get(k) != b.get(k):
            changed[k] = (a.get(k), b.get(k))
    la = {lv["concurrency"]: lv for lv in a.get("levels", [])}
    lb = {lv["concurrency"]: lv for lv in b.get("levels", [])}
    levels = {c: {m: (la[c].get(m), lb[c].get(m)) for m in METRICS} for c in sorted(set(la) & set(lb))}
    errors = Counter((t.get("error") or "")[:90] for t in b.get("tasks", []) if t.get("error"))
    violations = Counter(v.split(":")[0] for t in b.get("tasks", []) for v in (t.get("violations") or []))
    return {"settings": changed, "levels": levels, "errors": errors.most_common(5),
            "violations": violations.most_common(5)}


def _fmt(x) -> str:
    if x is None:
        return "-"
    return f"{x:,.3f}" if isinstance(x, float) else str(x)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Compare two agent_bench result files")
    ap.add_argument("good")
    ap.add_argument("bad")
    a = ap.parse_args(argv)
    d = diff(load(a.good), load(a.bad))
    print("Settings that differ (good -> bad):" if d["settings"] else "Settings: identical (look at the engine side)")
    for k, (x, y) in d["settings"].items():
        print(f"  {k}: {x} -> {y}")
    if not d["levels"]:
        print("No common concurrency level: rerun the second file with the same --levels.")
    for c, ms in d["levels"].items():
        print(f"Concurrency {c}:")
        for m, (x, y) in ms.items():
            if x is not None or y is not None:
                print(f"  {m:<24} {_fmt(x):>12} -> {_fmt(y):<12}")
    for title, items in (("Top errors in the bad run:", d["errors"]),
                         ("Top violations in the bad run:", d["violations"])):
        if items:
            print(title)
            for text, n in items:
                print(f"  {n:>4} x {text}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
