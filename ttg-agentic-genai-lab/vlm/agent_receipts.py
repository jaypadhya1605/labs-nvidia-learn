#!/usr/bin/env python3
"""M8 - does the VLM change what the agent PAYS? The receipt tasks, run twice: world oracle vs VLM-read receipts.

Only scenarios with a receipt AND a controllable cause call read_receipt (35 of the 200 golden scenarios). Pass 1
answers read_receipt from the world's ground truth; pass 2 sets TTG_VLM_BASE_URL, so the MCP guest server reads
data/receipts/<ref>.png with the VLM (ttg/mcp/guest_server.py -> vlm/extract_receipt.extract_fields).
The difference in policy-compliant tasks is the VLM's business impact: a misread total becomes a wrong voucher.

Usage (from the repo root; text LLM on :8000, VLM on :8001 - see docs/06-multimodal.md):
  python3 -m vlm.make_receipts                                    # the images the guest server will read
  TTG_VLM_BASE_URL=http://localhost:8001/v1 TTG_VLM_MODEL=ttg-vlm python3 -m vlm.agent_receipts --concurrency 8
  python3 -m vlm.agent_receipts --dry-run                         # list the receipt scenarios, run nothing
Needs TTG_MCP_MODE=stdio (the default): the agent spawns the guest server with this process's environment.
"""
import argparse
import asyncio
import json
import os
import statistics
import sys

if __package__ in (None, ""):          # run as a file: put the repo root on sys.path
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

VLM_VARS = ("TTG_VLM_BASE_URL", "TTG_VLM_MODEL", "TTG_VLM_API_KEY")


def receipt_scenarios(n=200, seed=7):
    """Golden-set scenarios whose policy path calls read_receipt (receipt uploaded + controllable cause)."""
    from ttg import world
    return [s for s in world.make_scenarios(n, seed=seed) if s.get("receipt_ref") and s.get("controllable")]


def summarize(results):
    ok = [r for r in results if r.get("success")]
    reimb = [r for r in results if any("reimbursement" in v for v in r.get("violations") or [])]
    lat = [r["latency_s"] for r in results if r.get("latency_s") is not None]
    tool = [r.get("tool_time_s") or 0.0 for r in results]
    return {"tasks": len(results), "success": len(ok), "success_rate": len(ok) / len(results) if results else 0.0,
            "reimbursement_violations": len(reimb), "p50_latency_s": statistics.median(lat) if lat else None,
            "mean_tool_time_s": statistics.mean(tool) if tool else None,
            "errors": sum(1 for r in results if r.get("error"))}


def compare(oracle, vlm):
    """Per-scenario differences between the two passes (results are TaskResult dicts)."""
    by_id = {r["scenario_id"]: r for r in oracle}
    flipped = []
    for r in vlm:
        o = by_id.get(r["scenario_id"])
        if o and o.get("success") and not r.get("success"):
            flipped.append({"scenario_id": r["scenario_id"], "violations": r.get("violations"), "error": r.get("error")})
    return {"oracle": summarize(oracle), "vlm": summarize(vlm), "lost_by_vlm": flipped}


def run_pass(scenarios, concurrency, vlm_env):
    """One agent pass in-process. vlm_env None = oracle receipts; a dict = those TTG_VLM_* variables set."""
    saved = {k: os.environ.get(k) for k in VLM_VARS}
    try:
        for k in VLM_VARS:
            os.environ.pop(k, None)
        for k, v in (vlm_env or {}).items():
            if v:
                os.environ[k] = v
        from ttg.agent.config import Settings          # read the TTG_* environment now
        from ttg.agent.run import run_many
        return asyncio.run(run_many(scenarios, concurrency, Settings().validate()))
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=0, help="first n receipt scenarios (0 = all 35)")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--label", default="vlm")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    scenarios = receipt_scenarios()
    if a.n:
        scenarios = scenarios[:a.n]
    if a.dry_run:
        print(f"{len(scenarios)} receipt scenarios: {', '.join(s['scenario_id'] for s in scenarios)}")
        print("pass 1: read_receipt from the world oracle; pass 2: TTG_VLM_BASE_URL="
              f"{os.environ.get('TTG_VLM_BASE_URL', '<unset>')} TTG_VLM_MODEL={os.environ.get('TTG_VLM_MODEL', 'ttg-vlm')}")
        return 0
    vlm_env = {k: os.environ.get(k, "") for k in VLM_VARS}
    if not vlm_env["TTG_VLM_BASE_URL"]:
        print("Set TTG_VLM_BASE_URL (and TTG_VLM_MODEL) to the VLM endpoint first.", file=sys.stderr)
        return 2
    if os.environ.get("TTG_MCP_MODE", "stdio") != "stdio":
        print("TTG_MCP_MODE must be stdio here (http mode: restart the guest server with TTG_VLM_BASE_URL).",
              file=sys.stderr)
        return 2
    oracle = run_pass(scenarios, a.concurrency, None)
    vlm = run_pass(scenarios, a.concurrency, vlm_env)
    c = compare(oracle, vlm)
    print("| receipts read by | tasks | policy-compliant | reimbursement violations | p50 task s | mean tool s |")
    print("|---|---|---|---|---|---|")
    for name in ("oracle", "vlm"):
        s = c[name]
        print(f"| {name} | {s['tasks']} | {s['success']} ({s['success_rate']:.0%}) | {s['reimbursement_violations']} | "
              f"{s['p50_latency_s'] or 0:.2f} | {s['mean_tool_time_s'] or 0:.2f} |")
    for f in c["lost_by_vlm"][:10]:
        print(f"- lost by the VLM: {f['scenario_id']} {' '.join(f['violations'] or [])} {f['error'] or ''}")
    os.makedirs("results/vlm", exist_ok=True)
    out = f"results/vlm/agent-receipts-{a.label}.json"
    with open(out, "w") as fh:
        json.dump({"label": a.label, **c, "oracle_results": oracle, "vlm_results": vlm}, fh, indent=1, default=str)
    print(f"\n-> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
