#!/usr/bin/env python3
"""Release gates over `python -m ttg.bench.agent_bench` result files (🔁 Lab 02/04 perf gate, plus an eval gate).

  # eval gate (every push, CPU mock): is the agent still policy-compliant?
  python scripts/gate.py eval results/bench/ci-mock.json --min-success 0.95

  # perf gate (GPU runner, manual or scheduled): did latency or cost regress against the committed baseline?
  python scripts/gate.py perf results/bench/vllm-base.json --baseline bench/baselines/vllm-base-h100.json \
      --max-p95-regression 0.10 --max-cost-regression 0.10

  # re-record a baseline - deliberately, in its own merge request, with a reason (never automatically)
  python scripts/gate.py baseline results/bench/vllm-base.json --out bench/baselines/vllm-base-h100.json \
      --reason "vLLM 0.31.0 -> 0.32.0 upgrade, accepted 4% p95 gain"

Exit codes: 0 = pass, 1 = a gate failed, 2 = unusable input (missing file, wrong schema, nothing comparable).
Every mode accepts --junit FILE so GitLab / GitHub show each check as a test case.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from xml.sax.saxutils import escape

LEVEL_KEYS = ("concurrency", "success_rate", "p95_latency_s")


class InputError(Exception):
    """The file cannot be gated (exit code 2)."""


@dataclass
class Check:
    name: str
    ok: bool
    detail: str
    skipped: bool = False
    warnings: list[str] = field(default_factory=list)


def load_bench(path: str | Path) -> dict:
    p = Path(path)
    if not p.is_file():
        raise InputError(f"{p}: no such bench file")
    try:
        data = json.loads(p.read_text())
    except json.JSONDecodeError as e:
        raise InputError(f"{p}: not JSON ({e})") from e
    validate_bench(data, str(p))
    return data


def validate_bench(data: dict, where: str = "bench") -> None:
    if not isinstance(data, dict):
        raise InputError(f"{where}: top level must be an object")
    levels = data.get("levels")
    if not isinstance(levels, list) or not levels:
        raise InputError(f"{where}: 'levels' must be a non-empty list (ttg.bench.agent_bench schema)")
    for i, lv in enumerate(levels):
        missing = [k for k in LEVEL_KEYS if k not in lv]
        if missing:
            raise InputError(f"{where}: levels[{i}] lacks {', '.join(missing)}")


def _num(v):
    return v if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def eval_gate(bench: dict, min_success: float = 0.95, max_retries_per_task: float | None = None,
              level: int | None = None) -> list[Check]:
    """Policy compliance: every measured level (or one --level) must meet the success floor."""
    levels = [lv for lv in bench["levels"] if level is None or lv["concurrency"] == level]
    if not levels:
        raise InputError(f"no level with concurrency={level} in {bench.get('label', 'bench')}")
    checks = []
    for lv in levels:
        c, sr = lv["concurrency"], _num(lv["success_rate"])
        n = lv.get("n", "?")
        if sr is None:
            raise InputError(f"concurrency {c}: success_rate is not a number")
        checks.append(Check(f"eval c={c} success_rate", sr >= min_success,
                            f"{sr:.3f} vs floor {min_success:.3f} over {n} tasks"))
        if max_retries_per_task is not None:
            r = _num(lv.get("retries_per_task"))
            if r is None:
                checks.append(Check(f"eval c={c} retries_per_task", True, "not reported", skipped=True))
            else:
                checks.append(Check(f"eval c={c} retries_per_task", r <= max_retries_per_task,
                                    f"{r:.3f} vs ceiling {max_retries_per_task:.3f} (JSON protocol repairs)"))
    return checks


def _regression(cand: float | None, base: float | None) -> float | None:
    if cand is None or base is None or base <= 0:
        return None
    return cand / base - 1.0


def perf_gate(cand: dict, base: dict, max_p95_regression: float = 0.10, max_cost_regression: float = 0.10,
              max_throughput_drop: float | None = None, min_success: float = 0.95,
              allow_gpu_mismatch: bool = False) -> list[Check]:
    """Candidate vs committed baseline, level by level (same concurrency only)."""
    warnings = []
    for key in ("engine", "model", "variant"):
        if cand.get(key) != base.get(key):
            warnings.append(f"{key} differs: candidate={cand.get(key)!r} baseline={base.get(key)!r}")
    if (cand.get("gpu") or base.get("gpu")) and cand.get("gpu") != base.get("gpu") and not allow_gpu_mismatch:
        raise InputError(f"GPU differs (candidate={cand.get('gpu')!r}, baseline={base.get('gpu')!r}): a regression "
                         "check across GPUs is meaningless - record a baseline for this GPU or pass --allow-gpu-mismatch")
    base_by_c = {lv["concurrency"]: lv for lv in base["levels"]}
    common = [lv for lv in cand["levels"] if lv["concurrency"] in base_by_c]
    if not common:
        raise InputError("no concurrency level in common with the baseline - rerun the bench with the baseline's levels "
                         f"({sorted(base_by_c)})")
    checks = []
    for lv in common:
        c, b = lv["concurrency"], base_by_c[lv["concurrency"]]
        sr = _num(lv.get("success_rate"))
        checks.append(Check(f"perf c={c} success_rate", sr is not None and sr >= min_success,
                            f"{sr} vs floor {min_success} (a faster but wrong engine must fail)"))
        reg = _regression(_num(lv.get("p95_latency_s")), _num(b.get("p95_latency_s")))
        if reg is None:
            checks.append(Check(f"perf c={c} p95_latency_s", True, "baseline p95 missing or zero", skipped=True))
        else:
            checks.append(Check(f"perf c={c} p95_latency_s", reg <= max_p95_regression,
                                f"{lv['p95_latency_s']:.3f}s vs {b['p95_latency_s']:.3f}s ({reg:+.1%}, max +{max_p95_regression:.0%})"))
        creg = _regression(_num(lv.get("usd_per_1k_tasks")), _num(b.get("usd_per_1k_tasks")))
        if creg is None:
            checks.append(Check(f"perf c={c} usd_per_1k_tasks", True,
                                "cost not priced (gpu_hourly_usd 0 on one side)", skipped=True))
        else:
            checks.append(Check(f"perf c={c} usd_per_1k_tasks", creg <= max_cost_regression,
                                f"${lv['usd_per_1k_tasks']:.3f} vs ${b['usd_per_1k_tasks']:.3f} ({creg:+.1%}, max +{max_cost_regression:.0%})"))
        if max_throughput_drop is not None:
            treg = _regression(_num(lv.get("tasks_per_s")), _num(b.get("tasks_per_s")))
            if treg is None:
                checks.append(Check(f"perf c={c} tasks_per_s", True, "not reported", skipped=True))
            else:
                checks.append(Check(f"perf c={c} tasks_per_s", -treg <= max_throughput_drop,
                                    f"{lv['tasks_per_s']:.3f} vs {b['tasks_per_s']:.3f} ({treg:+.1%}, max -{max_throughput_drop:.0%})"))
    if checks:
        checks[0].warnings = warnings
    return checks


def _git_sha() -> str | None:
    sha = os.environ.get("CI_COMMIT_SHA") or os.environ.get("GITHUB_SHA")
    if sha:
        return sha
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def make_baseline(bench: dict, reason: str) -> dict:
    """A baseline = the bench summary minus per-task rows, plus who/why/when. Small enough to review in a diff."""
    if not reason or len(reason.strip()) < 10:
        raise InputError("--reason is required (>= 10 characters): baselines move deliberately, with a written why")
    out = {k: v for k, v in bench.items() if k != "tasks"}
    out["baseline"] = {"recorded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                       "git_sha": _git_sha(), "reason": reason.strip()}
    return out


def write_junit(checks: list[Check], path: str | Path, suite: str) -> None:
    failures = sum(1 for c in checks if not c.ok)
    skipped = sum(1 for c in checks if c.skipped)
    cases = []
    for c in checks:
        body = ""
        if not c.ok:
            body = f'<failure message="{escape(c.detail, {chr(34): "&quot;"})}"/>'
        elif c.skipped:
            body = f'<skipped message="{escape(c.detail, {chr(34): "&quot;"})}"/>'
        cases.append(f'  <testcase classname="{escape(suite)}" name="{escape(c.name)}">{body}</testcase>')
    xml = (f'<?xml version="1.0" encoding="UTF-8"?>\n<testsuite name="{escape(suite)}" tests="{len(checks)}" '
           f'failures="{failures}" skipped="{skipped}">\n' + "\n".join(cases) + "\n</testsuite>\n")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(xml)


def report(checks: list[Check], title: str) -> bool:
    print(title)
    for c in checks:
        for w in c.warnings:
            print(f"  [warn] {w}")
    for c in checks:
        tag = "skip" if c.skipped else ("ok" if c.ok else "FAIL")
        print(f"  [{tag:4s}] {c.name:34s} {c.detail}")
    passed = all(c.ok for c in checks)
    print("PASS" if passed else "FAIL")
    return passed


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="mode", required=True)
    e = sub.add_parser("eval", help="policy-compliance floor on a bench file")
    e.add_argument("bench")
    e.add_argument("--min-success", type=float, default=0.95)
    e.add_argument("--max-retries-per-task", type=float, default=None)
    e.add_argument("--level", type=int, default=None, help="gate one concurrency level only")
    e.add_argument("--junit", default=None)
    p = sub.add_parser("perf", help="candidate vs baseline regression gate")
    p.add_argument("bench")
    p.add_argument("--baseline", required=True)
    p.add_argument("--max-p95-regression", type=float, default=0.10)
    p.add_argument("--max-cost-regression", type=float, default=0.10)
    p.add_argument("--max-throughput-drop", type=float, default=None)
    p.add_argument("--min-success", type=float, default=0.95)
    p.add_argument("--allow-gpu-mismatch", action="store_true")
    p.add_argument("--junit", default=None)
    b = sub.add_parser("baseline", help="write a reviewed baseline from a bench file")
    b.add_argument("bench")
    b.add_argument("--out", required=True)
    b.add_argument("--reason", required=True)
    args = ap.parse_args(argv)
    try:
        bench = load_bench(args.bench)
        if args.mode == "baseline":
            out = make_baseline(bench, args.reason)
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out).write_text(json.dumps(out, indent=2) + "\n")
            print(f"baseline written: {args.out}  (commit it in its own merge request; the reason is in the file)")
            return 0
        if args.mode == "eval":
            checks = eval_gate(bench, args.min_success, args.max_retries_per_task, args.level)
            title = f"eval gate: {bench.get('label', args.bench)}"
        else:
            if not Path(args.baseline).is_file():
                raise InputError(f"{args.baseline}: no baseline yet - record one deliberately with "
                                 f"`python scripts/gate.py baseline {args.bench} --out {args.baseline} --reason ...`")
            base = load_bench(args.baseline)
            checks = perf_gate(bench, base, args.max_p95_regression, args.max_cost_regression,
                               args.max_throughput_drop, args.min_success, args.allow_gpu_mismatch)
            title = f"perf gate: {bench.get('label', args.bench)} vs {base.get('label', args.baseline)}"
    except InputError as err:
        print(f"gate: {err}", file=sys.stderr)
        return 2
    if args.junit:
        write_junit(checks, args.junit, f"gate-{args.mode}")
    return 0 if report(checks, title) else 1


if __name__ == "__main__":
    sys.exit(main())
