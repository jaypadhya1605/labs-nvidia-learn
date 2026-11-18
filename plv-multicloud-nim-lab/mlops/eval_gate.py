#!/usr/bin/env python3
"""M8 step 4 - the LoRA release gate: candidate adapter vs today's product (base + long prompt) on held-out cases.
Fails (exit 1) unless the adapter is at least as correct AND cheaper - a Git commit only follows a PASS.
  python3 mlops/eval_gate.py --endpoint http://localhost:8000 --candidate plover-style-v1 --n 120
  python3 mlops/eval_gate.py --endpoint mock --candidate plover-style-v1        # CI dry run
Metrics: schema-valid rate, hallucinated-code rate, recommendation agreement with the baseline, prompt tokens,
latency p95. Thresholds: mlops/gate.json. Optional: --mlflow logs the verdict to the training run.
"""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "app"), str(ROOT / "tests"), str(ROOT / "mlops")]
from plover_api import prompts, schema, synth  # noqa: E402
from make_dataset import client_for  # noqa: E402


def run(c, model, mode, cases, max_tokens=500):
    out = []
    for case in cases:
        t0 = time.perf_counter()
        r = c.post("/v1/chat/completions", json={"model": model, "messages": prompts.build_messages(case["text"], mode),
                                                  "temperature": 0.0, "max_tokens": max_tokens})
        lat = time.perf_counter() - t0
        if r.status_code != 200:
            out.append({"ok": False, "halluc": False, "rec": None, "ptok": 0, "lat": lat, "error": r.status_code})
            continue
        d = r.json()
        obj = schema.extract_json(d["choices"][0]["message"].get("content") or "")
        out.append({"ok": not schema.validate(obj), "halluc": bool(schema.unfaithful_codes(obj, case["text"])),
                    "rec": (obj or {}).get("recommendation") if isinstance(obj, dict) else None,
                    "ptok": (d.get("usage") or {}).get("prompt_tokens", 0), "lat": lat})
    return out


def summarize(rows):
    lat = sorted(r["lat"] for r in rows)
    return {"schema_valid": sum(r["ok"] for r in rows) / len(rows), "hallucinated_code_rate": sum(r["halluc"] for r in rows) / len(rows),
            "prompt_tokens_mean": statistics.mean(r["ptok"] for r in rows), "latency_p95_s": lat[int(0.95 * (len(lat) - 1))]}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--endpoint", required=True)
    ap.add_argument("--baseline", default="meta/llama-3.1-8b-instruct")
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--gate", default=str(ROOT / "mlops" / "gate.json"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--mlflow-run-id", default=None, help="log the verdict to this MLflow run (needs MLFLOW_TRACKING_URI)")
    a = ap.parse_args()
    g = json.load(open(a.gate))
    cases = synth.cases(a.n, 9000)
    with client_for(a.endpoint) as c:
        base = run(c, a.baseline, "long-prompt", cases)
        cand = run(c, a.candidate, "adapter", cases)
    b, k = summarize(base), summarize(cand)
    agree = sum(x["rec"] == y["rec"] for x, y in zip(base, cand) if x["ok"]) / max(sum(x["ok"] for x in base), 1)
    ratio_tok = k["prompt_tokens_mean"] / max(b["prompt_tokens_mean"], 1e-9)
    ratio_lat = k["latency_p95_s"] / max(b["latency_p95_s"], 1e-9)
    checks = [("schema_valid", k["schema_valid"], ">=", g["min_schema_valid"]),
              ("hallucinated_code_rate", k["hallucinated_code_rate"], "<=", g["max_hallucinated_code_rate"]),
              ("recommendation_agreement", agree, ">=", g["min_recommendation_agreement"]),
              ("prompt_token_ratio", ratio_tok, "<=", g["max_prompt_token_ratio"]),
              ("latency_p95_ratio", ratio_lat, "<=", g["max_latency_p95_ratio"])]
    failed = False
    print(f"baseline  {a.baseline:<34} {json.dumps({x: round(y, 4) for x, y in b.items()})}")
    print(f"candidate {a.candidate:<34} {json.dumps({x: round(y, 4) for x, y in k.items()})}")
    for name, val, op, lim in checks:
        ok = val >= lim if op == ">=" else val <= lim
        failed |= not ok
        print(f"  {name:<26}{val:>9.3f} {op} {lim:<6}  {'ok' if ok else 'FAIL'}")
    verdict = "FAIL" if failed else "PASS"
    print("GATE:", verdict)
    report = {"baseline": b, "candidate": k, "agreement": agree, "prompt_token_ratio": ratio_tok,
              "latency_p95_ratio": ratio_lat, "verdict": verdict, "n": a.n, "candidate_model": a.candidate}
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(report, indent=2))
    if a.mlflow_run_id:
        try:   # best effort: a tracking-server hiccup must not flip the release verdict
            import mlflow
            with mlflow.start_run(run_id=a.mlflow_run_id):
                mlflow.log_metrics({"gate_schema_valid": k["schema_valid"], "gate_agreement": agree,
                                    "gate_prompt_token_ratio": ratio_tok, "gate_latency_p95_ratio": ratio_lat})
                mlflow.set_tag("release_gate", verdict)
        except Exception as exc:  # noqa: BLE001
            print(f"warning: verdict not logged to MLflow ({type(exc).__name__}: {exc}); "
                  "pip install mlflow and export MLFLOW_TRACKING_URI=http://localhost:5000 (port-forward svc/mlflow)")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
