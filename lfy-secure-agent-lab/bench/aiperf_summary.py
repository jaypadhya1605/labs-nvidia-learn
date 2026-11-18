#!/usr/bin/env python3
"""Summarize AIPerf 0.13 exports (profile_export_aiperf.json) into the numbers an SA reports.

Usage:
  python3 bench/aiperf_summary.py <artifact-dir-or-json> [more ...]     # one row per run
  python3 bench/aiperf_summary.py --compare <baseline> <candidate>       # % change, e.g. noisy neighbor
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

FIELDS = [
    ("requests", "request_count", "avg"), ("req_per_s", "request_throughput", "avg"),
    ("out_tok_per_s", "output_token_throughput", "avg"), ("tok_per_s_per_user", "output_token_throughput_per_user", "avg"),
    ("ttft_p50_ms", "time_to_first_token", "p50"), ("ttft_p95_ms", "time_to_first_token", "p95"),
    ("itl_p50_ms", "inter_token_latency", "p50"), ("itl_p95_ms", "inter_token_latency", "p95"),
    ("e2e_p95_ms", "request_latency", "p95"), ("isl_avg", "input_sequence_length", "avg"),
    ("osl_avg", "output_sequence_length", "avg"), ("total_output_tokens", "total_output_tokens", "avg"),
    ("goodput_req_per_s", "goodput", "avg"), ("good_requests", "good_request_count", "avg"),
    ("duration_s", "benchmark_duration", "avg"),
]


def _stat(d, key, stat):
    v = d.get(key)
    if isinstance(v, dict):
        x = v.get(stat, v.get("avg"))
        return float(x) if isinstance(x, (int, float)) else None
    return float(v) if isinstance(v, (int, float)) else None


def _ts(s):
    if not s:
        return None
    try:
        dt = datetime.fromisoformat(str(s).replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:  # AIPerf writes container-local time; the toolbox runs in UTC
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def find_export(path):
    p = Path(path)
    if p.is_file():
        return p
    top = p / "profile_export_aiperf.json"
    if top.exists():
        return top
    for c in sorted(p.rglob("profile_export_aiperf.json")):
        if "warmup" not in c.parts:
            return c
    raise FileNotFoundError(f"no profile_export_aiperf.json under {path}")


def load(path):
    f = find_export(path)
    d = json.loads(f.read_text())
    out = {"file": str(f)}
    for name, key, stat in FIELDS:
        out[name] = _stat(d, key, stat)
    if out["total_output_tokens"] is None and out["osl_avg"] and out["requests"]:
        out["total_output_tokens"] = out["osl_avg"] * out["requests"]
    out["slo_attainment"] = (out["good_requests"] / out["requests"]) if out["good_requests"] is not None and out["requests"] else None
    out["start_ts"], out["end_ts"] = _ts(d.get("start_time")), _ts(d.get("end_time"))
    out["errors"] = len(d.get("error_summary") or [])
    return out


def _fmt(v):
    if v is None:
        return "-"
    return f"{v:.3f}" if abs(v) < 10 else f"{v:.1f}"


def main(argv):
    if not argv:
        print(__doc__)
        return 1
    cols = ["requests", "req_per_s", "out_tok_per_s", "ttft_p50_ms", "ttft_p95_ms", "itl_p50_ms", "itl_p95_ms", "slo_attainment", "errors"]
    if argv[0] == "--compare":
        a, b = load(argv[1]), load(argv[2])
        print(f"{'metric':<16}{'baseline':>12}{'candidate':>12}{'change':>10}")
        for c in cols[1:8]:
            x, y = a.get(c), b.get(c)
            ch = f"{(y - x) / x * 100:+.1f}%" if x not in (None, 0) and y is not None else "-"
            print(f"{c:<16}{_fmt(x):>12}{_fmt(y):>12}{ch:>10}")
        return 0
    print("run," + ",".join(cols))
    for p in argv:
        s = load(p)
        print(",".join([Path(p).name] + [_fmt(s.get(c)) for c in cols]))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
