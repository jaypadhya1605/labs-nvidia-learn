#!/usr/bin/env python3
"""CI performance gate: fail when a candidate AIPerf run regresses against a baseline.

Usage: python3 bench/perf_gate.py --baseline bench/baselines/aks-synth-600-200-c32.json --candidate results/aks-synth-600-200-<ts>/c32
Exit code 1 on regression -> blocks the NIM upgrade / config change in CI.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from aiperf_summary import load  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--ttft-tol", type=float, default=0.10, help="allowed TTFT p95 increase (fraction)")
    ap.add_argument("--itl-tol", type=float, default=0.10, help="allowed ITL p95 increase (fraction)")
    ap.add_argument("--tput-tol", type=float, default=0.10, help="allowed throughput decrease (fraction)")
    a = ap.parse_args()
    b, c = load(a.baseline), load(a.candidate)
    checks = [("ttft_p95_ms", "up", a.ttft_tol), ("itl_p95_ms", "up", a.itl_tol), ("out_tok_per_s", "down", a.tput_tol)]
    failed = c["errors"] > 0
    print(f"{'metric':<14}{'baseline':>12}{'candidate':>12}{'change':>9}  verdict")
    for key, bad_dir, tol in checks:
        x, y = b.get(key), c.get(key)
        if not x or y is None:
            print(f"{key:<14}{'-':>12}{'-':>12}{'-':>9}  skipped (missing)")
            continue
        change = (y - x) / x
        bad = change > tol if bad_dir == "up" else change < -tol
        failed |= bad
        print(f"{key:<14}{x:>12.1f}{y:>12.1f}{change:>+9.1%}  {'REGRESSION' if bad else 'ok'}")
    if c["errors"]:
        print(f"candidate had {c['errors']} error types in error_summary")
    print("GATE:", "FAIL" if failed else "PASS")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
