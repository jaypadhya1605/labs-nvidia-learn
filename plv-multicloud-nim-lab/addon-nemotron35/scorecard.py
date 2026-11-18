#!/usr/bin/env python3
"""Add-on A3/A4: Plover-specific scorecard for a model + side-by-side comparison of every model you evaluated.
  scorecard.py run --endpoint http://localhost:8010 --model nvidia/nemotron-3.5-lightning-30b-a3b --out results/addon/x/plover.json
  scorecard.py compare results/addon/*
"run" measures summaries (schema-valid, hallucinated codes, tokens, sequential latency); "compare" joins it with the
agentic scorecards and conformance results written by run-eval.sh. Capacity/cost at load comes from AIPerf (A4).
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "app"), str(ROOT / "tests"), str(ROOT / "managed")]
import compare as managed  # noqa: E402  (managed/compare.py: the same summary probe used for managed endpoints)


def run(a):
    ep = {"base_url": a.endpoint, "model": a.model, "extra_body": json.loads(a.extra_body)}
    with managed.client(ep, a.endpoint == "mock") as c:
        s = managed.summaries(c, ep, a.n)
    s.update(model=a.model, usd_per_hour=a.usd_per_hour)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(s, indent=2))
    print(json.dumps(s, indent=2))


def compare(a):
    print(f"{'label':<22}{'conformance':<26}{'valid':>7}{'halluc':>8}{'p50 s':>7}{'agent':>7}{'inject':>8}{'agent+think':>12}")
    for d in a.dirs:
        d = Path(d)
        p = json.loads((d / "plover.json").read_text()) if (d / "plover.json").exists() else {}
        off = json.loads((d / "agentic-thinking-off.json").read_text())["score"] if (d / "agentic-thinking-off.json").exists() else {}
        on = json.loads((d / "agentic-thinking-on.json").read_text())["score"] if (d / "agentic-thinking-on.json").exists() else {}
        conf = (d / "conformance.txt").read_text().strip() if (d / "conformance.txt").exists() else "-"
        def f(x, fmt):
            return format(x, fmt) if isinstance(x, (int, float)) else "-"
        print(f"{d.name:<22}{conf[:25]:<26}{f(p.get('schema_valid'), '.0%'):>7}{f(p.get('hallucinated_code_rate'), '.0%'):>8}"
              f"{f(p.get('latency_p50_s'), '.2f'):>7}{f(off.get('task_success'), '.0%'):>7}{f(off.get('injection_success'), '.0%'):>8}"
              f"{f(on.get('task_success'), '.0%'):>12}")
    print("\nDecide with the M8 gate mindset: correctness first (valid, no hallucinated codes, injection 0%), then cost.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--endpoint", required=True)
    r.add_argument("--model", required=True)
    r.add_argument("--extra-body", default="{}")
    r.add_argument("--n", type=int, default=30)
    r.add_argument("--usd-per-hour", type=float, default=0.0)
    r.add_argument("--out", required=True)
    c = sub.add_parser("compare")
    c.add_argument("dirs", nargs="+")
    a = ap.parse_args()
    run(a) if a.cmd == "run" else compare(a)


if __name__ == "__main__":
    main()
