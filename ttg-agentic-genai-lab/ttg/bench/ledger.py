"""The acceleration ledger (M10): what each change in the stack did to cost, capacity and latency per disruption.

  python -m ttg.bench.ledger results/bench/*.json --slo-p95-s 12 \
      --order hf-base,vllm-nocache,vllm-base,vllm-fp8,vllm-spec,trtllm-fp8 --out results/ledger
For each benchmark file it picks the OPERATING POINT: the concurrency level with the highest tasks/s whose p95 task
latency meets the SLO and whose policy-compliant success rate meets --min-success (🔁 Lab 02 goodput logic, applied
to tasks instead of tokens). Then it prices it, sizes the storm day, and draws the waterfall of $ per 1,000 tasks.
Outputs: ledger.md, ledger.csv, ledger.json, waterfall.png.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path


def operating_point(levels: list, slo_p95_s: float, min_success: float) -> tuple:
    ok = [lv for lv in levels if lv.get("p95_latency_s") is not None and lv["p95_latency_s"] <= slo_p95_s
          and lv.get("success_rate", 0) >= min_success]
    if ok:
        return max(ok, key=lambda lv: lv["tasks_per_s"]), True
    best = max(levels, key=lambda lv: (lv.get("success_rate", 0), -(lv.get("p95_latency_s") or 1e9)))
    return best, False


def storm_gpus(tasks_per_s: float, gpus_per_replica: int, storm_tasks: int, storm_hours: float,
               peak_factor: float) -> dict:
    peak = storm_tasks / (storm_hours * 3600.0) * peak_factor
    replicas = math.ceil(peak / tasks_per_s) if tasks_per_s > 0 else None
    return {"peak_tasks_per_s": round(peak, 3), "replicas": replicas,
            "gpus_n_plus_1": (replicas + 1) * gpus_per_replica if replicas else None}


def build(docs: list, slo_p95_s: float, min_success: float, storm: dict, order: list | None = None) -> list:
    by_label = {d["label"]: d for d in docs}
    labels = [lb for lb in (order or []) if lb in by_label] + [d["label"] for d in docs
                                                             if d["label"] not in (order or [])]
    rows, base = [], None
    for lb in labels:
        d = by_label[lb]
        lv, meets = operating_point(d["levels"], slo_p95_s, min_success)
        gpus = max(int(d.get("gpus") or 1), 1)
        tph = lv["tasks_per_s"] * 3600 / gpus
        row = {"label": lb, "engine": d.get("engine"), "variant": d.get("variant"), "gpu": d.get("gpu"), "gpus": gpus,
               "meets_slo": meets, "concurrency": lv["concurrency"], "tasks_per_s": lv["tasks_per_s"],
               "tasks_per_gpu_hour": round(tph, 1), "p95_latency_s": lv.get("p95_latency_s"),
               "success_rate": lv.get("success_rate"),
               # usage.cached_tokens when the engine reports it; else the engine's own /metrics counter
               "cache_hit_ratio": lv["cache_hit_ratio"] if lv.get("cache_hit_ratio") is not None
               else lv.get("engine_prefix_hit_ratio"),
               "cache_hit_source": "usage" if lv.get("cache_hit_ratio") is not None
               else ("engine-metrics" if lv.get("engine_prefix_hit_ratio") is not None else None),
               "prompt_tokens_per_task": lv.get("mean_prompt_tokens"),
               "completion_tokens_per_task": lv.get("mean_completion_tokens"),
               "usd_per_1k_tasks": lv.get("usd_per_1k_tasks"), "gpu_hourly_usd": d.get("gpu_hourly_usd"),
               **{f"storm_{k}": v for k, v in storm_gpus(lv["tasks_per_s"], gpus, **storm).items()}}
        if base is None:
            base = row
        row["x_capacity_vs_first"] = round(tph / base["tasks_per_gpu_hour"], 2) if base["tasks_per_gpu_hour"] else None
        if base.get("usd_per_1k_tasks") and row.get("usd_per_1k_tasks"):
            row["cost_change_vs_first"] = round(row["usd_per_1k_tasks"] / base["usd_per_1k_tasks"] - 1, 4)
        rows.append(row)
    return rows


def to_markdown(rows: list, slo: float, min_success: float, storm: dict) -> str:
    def f(x, fmt):
        return "-" if x is None else format(x, fmt)
    out = ["# Acceleration ledger - Tern Disruption Concierge",
           "",
           f"Operating point = highest tasks/s with p95 task latency <= {slo:g} s and policy-compliant success >= "
           f"{min_success:.0%}. Storm day = {storm['storm_tasks']:,} disruptions in {storm['storm_hours']:g} h, peak "
           f"x{storm['peak_factor']:g}; GPUs include one spare replica (N+1).",
           "",
           "| Configuration | GPU | Meets SLO | Concurrency | Tasks/GPU-hour | x capacity | p95 task (s) | Success | "
           "Prefix cache hit | Prompt tok/task | Out tok/task | $ per 1k tasks | Storm-day GPUs |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        out.append(f"| {r['label']} | {r['gpu'] or '-'} x{r['gpus']} | {'yes' if r['meets_slo'] else 'NO'} | "
                   f"{r['concurrency']} | {f(r['tasks_per_gpu_hour'], ',.0f')} | {f(r.get('x_capacity_vs_first'), '.2f')} | "
                   f"{f(r['p95_latency_s'], '.2f')} | {f(r['success_rate'], '.0%')} | "
                   f"{f(r['cache_hit_ratio'], '.0%')}{'*' if r.get('cache_hit_source') == 'engine-metrics' else ''} | "
                   f"{f(r['prompt_tokens_per_task'], ',.0f')} | {f(r['completion_tokens_per_task'], ',.0f')} | "
                   f"{f(r['usd_per_1k_tasks'], '.3f')} | {f(r['storm_gpus_n_plus_1'], 'd')} |")
    out += ["", "Read it left to right: capacity is what the customer buys, $ per 1k tasks is what the CFO reads, and "
            "the storm-day GPU count is what the capacity plan commits to. Numbers come from YOUR runs; a row that "
            "says NO did not meet the SLO at any measured concurrency. Prefix cache hit comes from usage.cached_tokens; "
            "* = the engine's own /metrics counters (engines that do not report cached tokens per request); "
            "- = not reported."]
    return "\n".join(out) + "\n"


def waterfall(rows: list, path: Path) -> bool:
    priced = [r for r in rows if r.get("usd_per_1k_tasks")]
    if len(priced) < 2:
        return False
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    vals = [r["usd_per_1k_tasks"] for r in priced]
    fig, ax = plt.subplots(figsize=(max(6, 1.3 * len(vals) + 2), 4.2))
    prev = None
    for i, (r, v) in enumerate(zip(priced, vals)):
        if prev is None or i == len(vals) - 1:
            ax.bar(i, v, color="#4a6fa5")
            ax.text(i, v, f"${v:.3f}", ha="center", va="bottom", fontsize=8)
        else:
            lo, hi = min(prev, v), max(prev, v)
            ax.bar(i, hi - lo, bottom=lo, color="#2e8b57" if v < prev else "#c0392b")
            ax.text(i, hi, f"{(v / prev - 1):+.0%}", ha="center", va="bottom", fontsize=8)
        prev = v
    ax.set_xticks(range(len(vals)), [r["label"] for r in priced], rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("GPU $ per 1,000 disruptions")
    ax.set_title("What each step of the stack did to the cost of a handled disruption")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+")
    ap.add_argument("--slo-p95-s", type=float, default=12.0)
    ap.add_argument("--min-success", type=float, default=0.95)
    ap.add_argument("--order", default="", help="comma-separated labels: baseline first")
    ap.add_argument("--storm-tasks", type=int, default=40000)
    ap.add_argument("--storm-hours", type=float, default=6.0)
    ap.add_argument("--peak-factor", type=float, default=2.0)
    ap.add_argument("--out", default="results/ledger")
    a = ap.parse_args(argv)
    docs = [json.loads(Path(f).read_text(encoding="utf-8")) for f in a.files]
    storm = {"storm_tasks": a.storm_tasks, "storm_hours": a.storm_hours, "peak_factor": a.peak_factor}
    rows = build(docs, a.slo_p95_s, a.min_success, storm, [x for x in a.order.split(",") if x])
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    md = to_markdown(rows, a.slo_p95_s, a.min_success, storm)
    (out / "ledger.md").write_text(md, encoding="utf-8")
    (out / "ledger.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    with open(out / "ledger.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=sorted({k for r in rows for k in r}))
        w.writeheader()
        w.writerows(rows)
    drew = waterfall(rows, out / "waterfall.png")
    print(md)
    print(f"-> {out}/ledger.md, ledger.csv, ledger.json" + (", waterfall.png" if drew else
          " (waterfall needs >= 2 priced runs: pass --gpu-hourly to agent_bench)"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
