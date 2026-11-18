#!/usr/bin/env python3
"""Energy, tokens per joule, and cost for one AIPerf run, from Prometheus (DCGM + vLLM counters).

Method (see docs/03-sizing-worksheet.md, section 6):
  window  = AIPerf profiling start/end (excludes warmup).
  energy  = increase of DCGM_FI_DEV_TOTAL_ENERGY_CONSUMPTION (mJ) per physical GPU, using max() per UUID
            because MIG repeats device-level fields on every instance series. The WHOLE GPU's energy is
            attributed to the run: tenants are benchmarked one at a time and idle slices still draw power.
  tokens  = output tokens counted by AIPerf; cross-checked with vllm:generation_tokens_total.
  cost    = node $/h x nodes x window, reported for the whole node and for the tenant's slice share.
Needs: kubectl -n monitoring port-forward svc/kps-prometheus 9090:9090
"""
import argparse
import csv
import json
import os
import subprocess
import sys
import urllib.parse
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from aiperf_summary import load  # noqa: E402


def prom_query(base, expr, t):
    url = base.rstrip("/") + "/api/v1/query?" + urllib.parse.urlencode({"query": expr, "time": f"{t:.3f}"})
    with urllib.request.urlopen(url, timeout=30) as r:
        body = json.load(r)
    if body.get("status") != "success":
        raise RuntimeError(body)
    return body["data"]["result"]


def total(result):
    return sum(float(s["value"][1]) for s in result) if result else None


def pool_nodes(pool):
    if not pool:
        return []
    try:
        out = subprocess.run(["kubectl", "get", "nodes", "-l", f"kha.io/pool={pool}", "-o",
                              "jsonpath={.items[*].metadata.name}"], capture_output=True, text=True, timeout=30)
        return out.stdout.split()
    except (OSError, subprocess.SubprocessError):
        return []


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--aiperf", required=True, help="AIPerf artifact dir or profile_export_aiperf.json")
    ap.add_argument("--prom", default=os.environ.get("PROM_URL", "http://localhost:9090"))
    ap.add_argument("--pool", default="mig", help="kha.io/pool label of the metered GPU nodes ('' = all GPUs)")
    ap.add_argument("--namespace", default="", help="namespace for the vLLM token cross-check")
    ap.add_argument("--node-hourly", type=float, default=float(os.environ.get("GPU_HOURLY_USD") or 0), help="VM $/hour")
    ap.add_argument("--slice-share", type=float, default=1.0, help="tenant share of the GPU for chargeback (3g=3/7)")
    ap.add_argument("--label", default="")
    ap.add_argument("--concurrency", default="")
    ap.add_argument("--start", type=float, help="override window start (unix seconds)")
    ap.add_argument("--end", type=float, help="override window end (unix seconds)")
    ap.add_argument("--csv", help="append one row to this CSV")
    a = ap.parse_args()

    s = load(a.aiperf)
    start, end = a.start or s["start_ts"], a.end or s["end_ts"]
    if not start or not end or end <= start:
        sys.exit("No valid time window: pass --start/--end")
    dur = end - start
    if dur < 120:
        print(f"note: {dur:.0f}s window; with 15s scrapes energy error can reach ~{15 / dur:.0%}. Prefer >= 2 min.", file=sys.stderr)

    nodes = pool_nodes(a.pool)
    sel = '{Hostname=~"%s"}' % "|".join(nodes) if nodes else ""
    expr = "sum(max by (UUID) (DCGM_FI_DEV_TOTAL_ENERGY_CONSUMPTION%s))"
    e0, e1 = total(prom_query(a.prom, expr % sel, start)), total(prom_query(a.prom, expr % sel, end))
    if (e0 is None or e1 is None) and sel:
        print("warning: no DCGM series matched the pool's node names; metering all GPUs", file=sys.stderr)
        sel = ""
        e0, e1 = total(prom_query(a.prom, expr % sel, start)), total(prom_query(a.prom, expr % sel, end))
    gpus = len(prom_query(a.prom, "max by (UUID) (DCGM_FI_DEV_POWER_USAGE%s)" % sel, end))
    n_nodes = max(len(nodes), 1)
    if e0 is not None and e1 is not None and e1 > e0:
        energy_j, method = (e1 - e0) / 1000.0, "energy-counter"
    else:
        w = total(prom_query(a.prom, "sum(max by (UUID) (avg_over_time(DCGM_FI_DEV_POWER_USAGE%s[%ds])))" % (sel, max(int(dur), 30)), end))
        energy_j, method = (w or 0.0) * dur, "avg-power"

    ns_sel = '{namespace="%s"}' % a.namespace if a.namespace else ""
    server_tokens = total(prom_query(a.prom, "sum(increase(vllm:generation_tokens_total%s[%ds]))" % (ns_sel, max(int(dur), 30)), end))
    tokens = s["total_output_tokens"] or server_tokens or 0.0
    reqs = s["requests"] or 0.0
    node_cost = a.node_hourly * n_nodes * dur / 3600.0
    row = {
        "label": a.label, "concurrency": a.concurrency, "requests": reqs,
        "req_per_s": s["req_per_s"], "out_tok_per_s": s["out_tok_per_s"],
        "ttft_p50_ms": s["ttft_p50_ms"], "ttft_p95_ms": s["ttft_p95_ms"],
        "itl_p50_ms": s["itl_p50_ms"], "itl_p95_ms": s["itl_p95_ms"], "slo_attainment": s["slo_attainment"],
        "window_s": dur, "gpus_metered": gpus, "avg_gpu_w": energy_j / dur if dur else None, "energy_j": energy_j,
        "tokens": tokens, "server_tokens": server_tokens,
        "tokens_per_j": tokens / energy_j if energy_j else None,
        "kwh_per_1k_req": energy_j / 3.6e6 / reqs * 1000 if reqs else None,
        "usd_per_1k_req_node": node_cost / reqs * 1000 if reqs else None,
        "usd_per_1k_req_slice": node_cost * a.slice_share / reqs * 1000 if reqs else None,
        "usd_per_1m_tok_node": node_cost / tokens * 1e6 if tokens else None,
        "energy_method": method,
    }
    for k, v in row.items():
        if isinstance(v, float):
            row[k] = round(v, 6) if abs(v) < 1 else round(v, 3)
    print(json.dumps(row, indent=2))
    if a.csv:
        new = not Path(a.csv).exists()
        with open(a.csv, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row))
            if new:
                w.writeheader()
            w.writerow(row)


if __name__ == "__main__":
    main()
