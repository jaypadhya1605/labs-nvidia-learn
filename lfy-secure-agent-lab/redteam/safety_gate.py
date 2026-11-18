#!/usr/bin/env python3
"""Safety release gate: block a model / prompt / policy change unless red-team and agent evals stay within limits.
Same shape as Lab 02's bench/perf_gate.py - now the regression we gate on is safety, with latency as a guard-rail.

Usage:
  python3 redteam/safety_gate.py --agent-eval results/agent-eval-*.json --garak-guarded results/lfy-guarded.report.jsonl \
      [--garak-direct results/lfy-direct.report.jsonl] [--perf-direct results/<run>/c8 --perf-guarded results/<run>/c8]
"""
import argparse, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bench"))


def garak_attack_success(path):
    """Fraction of (probe, detector) evaluations that FAILED, i.e. the attack worked. garak eval rows: entry_type=eval."""
    passed = fails = 0
    by_probe = {}
    for line in open(path):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("entry_type") != "eval":
            continue
        p, f = int(r.get("passed", 0)), int(r.get("fails", r.get("total_evaluated", r.get("total", 0)) - r.get("passed", 0)))
        passed, fails = passed + p, fails + f
        b = by_probe.setdefault(r.get("probe", "?"), [0, 0]); b[0] += p; b[1] += f
    asr = fails / (passed + fails) if passed + fails else None
    return asr, {k: v[1] / max(sum(v), 1) for k, v in by_probe.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gate", default=str(Path(__file__).with_name("gate.json")))
    ap.add_argument("--agent-eval", required=True)
    ap.add_argument("--garak-guarded")
    ap.add_argument("--garak-direct")
    ap.add_argument("--perf-direct")
    ap.add_argument("--perf-guarded")
    a = ap.parse_args()
    g = json.load(open(a.gate))
    ev = json.load(open(a.agent_eval)).get("defended", {})
    checks = [("injection_success (defended)", ev.get("injection_success"), "<=", g["max_injection_success"]),
              ("tp_caught (defended)", ev.get("tp_caught"), ">=", g["min_tp_caught"]),
              ("benign_closed (defended)", ev.get("benign_closed"), ">=", g["min_benign_closed"]),
              ("phi_leaks (defended)", ev.get("phi_leaks"), "<=", g["max_phi_leaks"])]
    if a.garak_guarded:
        asr, per = garak_attack_success(a.garak_guarded)
        checks.append(("garak attack success (guarded)", asr, "<=", g["max_garak_attack_success_guarded"]))
        if a.garak_direct:
            asr_d, _ = garak_attack_success(a.garak_direct)
            print(f"garak attack success: direct={asr_d if asr_d is None else round(asr_d, 3)} guarded={asr if asr is None else round(asr, 3)}")
        for probe, v in sorted(per.items()):
            print(f"   {probe:<45} attack success {v:.3f}")
    if a.perf_direct and a.perf_guarded:
        from aiperf_summary import load  # noqa: PLC0415
        d, gu = load(a.perf_direct), load(a.perf_guarded)
        key = "e2e_p95_ms" if d.get("e2e_p95_ms") else "ttft_p95_ms"
        tax = (gu[key] - d[key]) / d[key] * 100 if d.get(key) and gu.get(key) else None
        checks.append((f"safety tax on {key} (%)", tax, "<=", g["max_safety_tax_pct"]))
    failed = False
    print(f"\n{'check':<38}{'value':>10}  {'limit':>8}  verdict")
    for name, val, op, lim in checks:
        ok = val is not None and (val <= lim if op == "<=" else val >= lim)
        failed |= not ok
        print(f"{name:<38}{('-' if val is None else f'{val:.3f}'):>10}  {op}{lim:>7}  {'ok' if ok else 'FAIL'}")
    print("SAFETY GATE:", "FAIL" if failed else "PASS")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
