#!/usr/bin/env python3
"""Price-performance across clouds: measured (AIPerf sweeps) vs predicted (selector) - the M5 deliverable.

For each run directory (bench/run_sweep.sh output: run.json + summary.csv), take the operating point = the highest
throughput level that still meets the SLO, price it with the instance's $/hour, and put the selector's prediction for
the same GPU/model/shape next to it. A big gap is a finding: wrong precision/profile, a bad node, or a wrong model.
  python3 bench/compare_clouds.py results/aks-synth-600-200-* results/gke-synth-600-200-* --itl-slo-ms 40
"""
import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "selector"))
import instance_selector as sel  # noqa: E402

PRODUCT_TO_GPU = {  # nvidia.com/gpu.product label -> selector/gpus.yaml key
    "NVIDIA-A100-80GB-PCIe": "A100-80GB-PCIe", "NVIDIA-A100-SXM4-80GB": "A100-80GB-SXM", "NVIDIA-A100-SXM4-40GB": "A100-40GB",
    "NVIDIA-RTX-PRO-6000-Blackwell-Server-Edition": "RTX-PRO-6000-BSE", "NVIDIA-L40S": "L40S", "NVIDIA-A10": "A10",
    "NVIDIA-A10G": "A10G", "NVIDIA-L4": "L4", "NVIDIA-H100-NVL": "H100-NVL", "NVIDIA-H100-80GB-HBM3": "H100-SXM",
}


def gpu_key(product):
    if not product:
        return None
    return PRODUCT_TO_GPU.get(product) or next((v for k, v in PRODUCT_TO_GPU.items() if k.split("-", 1)[-1] in product), None)


def operating_point(rows, itl_slo_ms, min_attain):
    ok = [r for r in rows if r.get("out_tok_per_s") and (r.get("itl_p95_ms") is not None and r["itl_p95_ms"] <= itl_slo_ms)
          and (r.get("slo_attainment") is None or r["slo_attainment"] >= min_attain)]
    return max(ok, key=lambda r: r.get("out_tok_per_s") or 0) if ok else None


def load_run(d):
    d = Path(d)
    meta = json.loads((d / "run.json").read_text()) if (d / "run.json").exists() else {"cluster": d.name}
    rows = []
    if (d / "summary.csv").exists():
        for r in csv.DictReader(open(d / "summary.csv")):
            rows.append({k: (None if v in ("", None) else (v if k in ("label", "energy_method") else float(v))) for k, v in r.items()})
    return meta, rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--itl-slo-ms", type=float, default=40)
    ap.add_argument("--min-attainment", type=float, default=0.9)
    ap.add_argument("--precision", default=None, help="precision actually served (default: best supported)")
    ap.add_argument("--models-file", default=None)
    a = ap.parse_args()
    models = sel.load("models.yaml")
    if a.models_file:
        import yaml
        models.update(yaml.safe_load(Path(a.models_file).read_text()))
    gpus = sel.load("gpus.yaml")
    print(f"{'run':<44}{'gpu':<17}{'conc':>5}{'tok/s':>8}{'ITLp95':>8}{'$/h':>7}{'$/1M':>8}{'pred tok/s':>11}{'gap':>7}")
    for d in a.runs:
        meta, rows = load_run(d)
        op = operating_point(rows, a.itl_slo_ms, a.min_attainment)
        gk = gpu_key(meta.get("gpu_product"))
        if not op:
            print(f"{Path(d).name:<44}{(gk or '?'):<17}  no level met ITL p95 <= {a.itl_slo_ms} ms")
            continue
        hourly = meta.get("gpu_hourly_usd") or 0
        usd = hourly / (op["out_tok_per_s"] * 3600) * 1e6 if op.get("out_tok_per_s") and hourly else None
        pred, gap = None, None
        mkey = {"meta/llama-3.1-8b-instruct": "llama-3.1-8b", "nvidia/nemotron-3.5-lightning-30b-a3b": "nemotron-3.5-lightning"}.get(meta.get("model"))
        if gk and mkey in models and meta.get("isl") and meta.get("osl"):
            served = a.precision or meta.get("precision")     # what the run actually served (run.json), never "the best"
            precs = [served] if served else (models[mkey].get("nim_precisions") or ["bf16", "fp8", "nvfp4"])
            best = max((sel.evaluate(models[mkey], gpus[gk], p, meta["isl"], meta["osl"], a.itl_slo_ms, 1e9)
                        for p in precs if sel.supports(gpus[gk], p)), key=lambda r: r.get("tok_s") or 0, default=None)
            pred = best.get("tok_s") if best else None
            gap = (op["out_tok_per_s"] - pred) / pred if pred else None
        print(f"{Path(d).name:<44}{(gk or '?'):<17}{int(op.get('concurrency') or 0):>5}{op['out_tok_per_s']:>8.0f}"
              f"{op['itl_p95_ms']:>8.1f}{hourly:>7.2f}{(f'{usd:.3f}' if usd else '-'):>8}{(pred or '-')!s:>11}"
              f"{(f'{gap:+.0%}' if gap is not None else '-'):>7}")
    print("\nOperating point = highest-throughput level with ITL p95 within the SLO and goodput attainment >= "
          f"{a.min_attainment:.0%}. Predictions need a synthetic preset (fixed ISL/OSL).")


if __name__ == "__main__":
    main()
