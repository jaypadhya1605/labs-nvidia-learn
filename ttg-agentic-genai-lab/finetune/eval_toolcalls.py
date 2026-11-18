#!/usr/bin/env python3
"""Step-level tool-call accuracy on the held-out set (seed 13): did the model pick the right tool, the exact
arguments, and the right final decision - and how many prompt tokens did it need?

Usage (an engine is serving on TTG_BASE_URL, default http://localhost:8000/v1):
  python3 finetune/eval_toolcalls.py --model ttg-short --system-variant short      # LoRA on vLLM (baked prompt)
  python3 finetune/eval_toolcalls.py --model ttg-llm:ttg-short --system-variant short   # LoRA on SGLang
  python3 finetune/eval_toolcalls.py --model ttg-llm --system-variant full         # base model, full prompt (before)
  python3 finetune/eval_toolcalls.py --model ttg-llm --system-variant short        # base model, short prompt (control)
  python3 finetune/eval_toolcalls.py --score-only results/finetune/eval-ttg-short-short.jsonl   # re-score offline
Every prompt is the oracle's conversation up to that step (teacher forcing), so one wrong step cannot derail the
rest - this measures the policy, step by step. Task-level success with real tool calls is the agent bench:
  TTG_PROMPT_VARIANT=short TTG_MODEL=ttg-short python -m ttg.bench.agent_bench --label vllm-lora --engine vllm ...
Metrics: action exact (right tool, or "final" when the plan is done), args exact (right tool AND arguments),
final fields (rebooked_flight, hotel_id, vouchers, needs_human_approval), invalid JSON rate (ProtocolError),
mean prompt / completion tokens (from usage), p50 / p95 latency per call.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from ttg.agent.prompts import build_system_prompt  # noqa: E402
from ttg.agent.protocol import ProtocolError, parse_step  # noqa: E402

FINAL_FIELDS = ("rebooked_flight", "hotel_id", "vouchers", "needs_human_approval")


# --------------------------------------------------------------------------------------------- pure scoring
def _args(a: dict) -> dict:
    return {k: (round(float(v), 2) if isinstance(v, float) else v) for k, v in (a or {}).items()}


def _field(final: dict, name: str):
    if name == "vouchers":
        return sorted((v.get("kind"), round(float(v.get("amount_usd", 0)), 2)) for v in final.get("vouchers") or [])
    if name == "needs_human_approval":
        return bool(final.get(name))
    return final.get(name)


def score_prediction(expected_text: str, predicted_text: str) -> dict:
    """Compare one predicted step with the oracle step. Both go through the agent's own parser (parse_step), so
    types are coerced exactly as at run time."""
    exp = parse_step(expected_text)
    r = {"expected_kind": exp.kind, "expected_action": exp.action or "final", "valid": True, "error": "",
         "predicted_kind": None, "predicted_action": None, "action_ok": False, "args_ok": False, "exact": False,
         "fields": {}}
    try:
        got = parse_step(predicted_text or "")
    except ProtocolError as e:
        r.update(valid=False, error=str(e)[:200])
        if exp.kind == "final":
            r["fields"] = {f: False for f in FINAL_FIELDS}
        return r
    r["predicted_kind"], r["predicted_action"] = got.kind, got.action or "final"
    if exp.kind == "action":
        r["action_ok"] = got.kind == "action" and got.action == exp.action
        r["args_ok"] = r["action_ok"] and _args(got.args) == _args(exp.args)
        r["exact"] = r["args_ok"]
    else:
        r["action_ok"] = got.kind == "final"
        gf = got.final or {}
        r["fields"] = {f: r["action_ok"] and _field(gf, f) == _field(exp.final or {}, f) for f in FINAL_FIELDS}
        r["args_ok"] = r["action_ok"] and all(r["fields"].values())
        r["exact"] = r["args_ok"]
    return r


def _pct(xs: list) -> float | None:
    return round(100.0 * sum(1 for x in xs if x) / len(xs), 2) if xs else None


def _q(xs: list, q: float) -> float | None:
    if not xs:
        return None
    s = sorted(xs)
    return round(s[min(len(s) - 1, int(q * (len(s) - 1) + 0.5))], 3)


def aggregate(rows: list) -> dict:
    """rows: [{"score": score_prediction(...), "usage": {...}, "latency_s": float}, ...]"""
    sc = [r["score"] for r in rows]
    act = [s for s in sc if s["expected_kind"] == "action"]
    fin = [s for s in sc if s["expected_kind"] == "final"]
    per_tool: dict = {}
    for s in act:
        per_tool.setdefault(s["expected_action"], []).append(s["args_ok"])
    usage = [r.get("usage") or {} for r in rows]
    pt = [u["prompt_tokens"] for u in usage if isinstance(u.get("prompt_tokens"), (int, float))]
    ct = [u["completion_tokens"] for u in usage if isinstance(u.get("completion_tokens"), (int, float))]
    lat = [r["latency_s"] for r in rows if isinstance(r.get("latency_s"), (int, float))]
    return {"n": len(sc), "n_action_steps": len(act), "n_final_steps": len(fin),
            "step_exact_pct": _pct([s["exact"] for s in sc]),
            "action_exact_pct": _pct([s["action_ok"] for s in sc]),
            "args_exact_pct": _pct([s["args_ok"] for s in act]),
            "final_exact_pct": _pct([s["exact"] for s in fin]),
            "final_field_pct": {f: _pct([s["fields"].get(f, False) for s in fin]) for f in FINAL_FIELDS},
            "invalid_json_pct": _pct([not s["valid"] for s in sc]),
            "per_tool_args_exact_pct": {k: _pct(v) for k, v in sorted(per_tool.items())},
            "mean_prompt_tokens": round(statistics.fmean(pt), 1) if pt else None,
            "mean_completion_tokens": round(statistics.fmean(ct), 1) if ct else None,
            "p50_latency_s": _q(lat, 0.5), "p95_latency_s": _q(lat, 0.95)}


def with_system_prompt(prompt: list, variant: str) -> list:
    """Swap the system message for the full or short prompt (or keep the one in the data: 'asis')."""
    if variant == "asis":
        return prompt
    sys_msg = {"role": "system", "content": build_system_prompt(variant=variant)}
    return [sys_msg] + [m for m in prompt if m.get("role") != "system"]


# --------------------------------------------------------------------------------------------- live calls
def call(client, base_url: str, model: str, messages: list, max_tokens: int, api_key: str) -> dict:
    t = time.perf_counter()
    r = client.post(f"{base_url.rstrip('/')}/chat/completions",
                    headers={"Authorization": f"Bearer {api_key}"},
                    json={"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": 0})
    dt = time.perf_counter() - t
    if r.status_code != 200:
        return {"text": "", "usage": {}, "latency_s": dt, "http_error": f"{r.status_code}: {r.text[:200]}"}
    d = r.json()
    return {"text": d["choices"][0]["message"].get("content") or "", "usage": d.get("usage") or {}, "latency_s": dt}


def run_live(a) -> list:
    import httpx

    rows = [json.loads(x) for x in Path(a.data).read_text().splitlines() if x.strip()]
    if a.limit:
        rows = rows[: a.limit]
    key = os.environ.get("TTG_API_KEY", "not-used")
    out = []
    with httpx.Client(timeout=a.timeout) as client:
        def one(row):
            msgs = with_system_prompt(row["prompt"], a.system_variant)
            res = call(client, a.base_url, a.model, msgs, a.max_tokens, key)
            return {"meta": row.get("meta", {}), "expected": row["completion"][0]["content"], **res}
        with ThreadPoolExecutor(max_workers=a.concurrency) as pool:
            for i, res in enumerate(pool.map(one, rows), 1):
                out.append(res)
                if i % 200 == 0:
                    print(f"  {i:,}/{len(rows):,} steps", file=sys.stderr)
    errors = [r for r in out if r.get("http_error")]
    if errors:
        print(f"  {len(errors)} HTTP errors, first: {errors[0]['http_error']}", file=sys.stderr)
    return out


def render(label: str, s: dict) -> str:
    ff = s["final_field_pct"]
    lines = [f"{label}: {s['n']:,} steps ({s['n_action_steps']:,} actions, {s['n_final_steps']:,} finals)",
             f"  step exact {s['step_exact_pct']}% | action exact {s['action_exact_pct']}% | args exact "
             f"{s['args_exact_pct']}% | final exact {s['final_exact_pct']}% | invalid JSON {s['invalid_json_pct']}%",
             "  final fields: " + ", ".join(f"{k} {v}%" for k, v in ff.items()),
             "  per tool (args exact): " + ", ".join(f"{k} {v}%" for k, v in s["per_tool_args_exact_pct"].items()),
             f"  tokens per call: prompt {s['mean_prompt_tokens']}, completion {s['mean_completion_tokens']} | "
             f"latency p50 {s['p50_latency_s']} s, p95 {s['p95_latency_s']} s"]
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", default=os.environ.get("TTG_BASE_URL", "http://localhost:8000/v1"))
    ap.add_argument("--model", default=os.environ.get("TTG_MODEL", "ttg-short"))
    ap.add_argument("--data", default=str(REPO / "data" / "sft" / "eval.jsonl"))
    ap.add_argument("--system-variant", choices=["short", "full", "asis"], default="short",
                    help="system prompt to send: short (baked), full, or asis (as in the data)")
    ap.add_argument("--variant", dest="system_variant", choices=["short", "full", "asis"], default=argparse.SUPPRESS,
                    help="same as --system-variant")
    ap.add_argument("--limit", type=int, default=0, help="first N steps only (0 = all)")
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--label", default="")
    ap.add_argument("--out", default="", help="summary JSON (default results/finetune/eval-<label>.json)")
    ap.add_argument("--score-only", help="re-score a predictions .jsonl written by an earlier run (no engine)")
    a = ap.parse_args(argv)

    label = a.label or f"{a.model.replace(':', '_').replace('/', '_')}-{a.system_variant}"
    if a.score_only:
        preds = [json.loads(x) for x in Path(a.score_only).read_text().splitlines() if x.strip()]
        label = a.label or Path(a.score_only).stem.removeprefix("eval-")
    else:
        preds = run_live(a)
        pred_path = Path(a.out).with_suffix(".jsonl") if a.out else REPO / "results" / "finetune" / f"eval-{label}.jsonl"
        pred_path.parent.mkdir(parents=True, exist_ok=True)
        pred_path.write_text("".join(json.dumps(p, ensure_ascii=False) + "\n" for p in preds))
    rows = [{"score": score_prediction(p["expected"], p.get("text", "")), "usage": p.get("usage"),
             "latency_s": p.get("latency_s")} for p in preds]
    summary = {"label": label, "model": a.model, "system_variant": a.system_variant, "data": a.data,
               **aggregate(rows)}
    out = Path(a.out) if a.out else REPO / "results" / "finetune" / f"eval-{label}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2) + "\n")
    print(render(label, summary))
    print(f"  -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
