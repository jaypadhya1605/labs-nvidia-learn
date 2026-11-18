#!/usr/bin/env python3
"""Instance selector: which cloud instance serves a model at an SLO for the least money? Physics first (🔁 Lab 02):
  - weights must fit:   params_total x bytes/param(precision) + extra_weights + runtime overhead <= GPU memory x util
  - decode is bandwidth-bound: step(b) = [bytes moved per step] / (bandwidth x efficiency) + fixed overhead
        bytes = weights read + b x (KV read for the average context + 2 x SSM state)
        dense: all weights every step.  MoE: shared + experts x P(expert touched by >= 1 of b tokens)  (uniform routing)
        hybrid Mamba: the fixed per-sequence SSM state is read AND written every step (add-on: Nemotron 3.5)
  - roofline: at large batch decode turns compute-bound -> step = max(bytes/bandwidth, 2 x active params x b / FLOPS)
  - inter-token latency (ITL) = step(b); pick the largest batch b that meets the ITL SLO and fits the KV/state budget
  - prefill is compute-bound: TTFT ~= 2 x params_active x prompt_tokens / (tensor TFLOPS x MFU)
  - $/1M output tokens = price_per_hour / (b / step(b) x 3600) x 1e6   (+ optional NVIDIA AI Enterprise $/GPU-hour)
First-order estimates to rank options and to say "no" early. M5 replaces them with AIPerf measurements.
"""
import argparse
import math
import signal
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
# precision: (bytes per parameter, tensor-core TFLOPS key that must exist on the GPU)
PRECISIONS = {
    "bf16": (2.0, "bf16_tflops"),
    "fp16": (2.0, "fp16_tflops"),
    "fp8": (1.0, "fp8_tflops"),
    "nvfp4": (0.5625, "fp4_tflops"),   # 4-bit E2M1 values + one FP8 scale per 16 -> 4.5 bits
    "w4a16": (0.53, "bf16_tflops"),    # 4-bit weights + group scales, BF16 math (runs on Ampere)
    "iq2": (0.258, "bf16_tflops"),     # GGML IQ2_XXS, 2.0625 bits (Bolt experts) - llama.cpp / GGUF runtimes, not NIM
}
OVERHEAD_GB = 2.0          # activations, CUDA graphs, allocator slack
MEM_UTIL = 0.90
BW_EFF = 0.80              # achievable fraction of peak bandwidth (the M3 bandwidth probe measures yours)
STEP_OVERHEAD_S = 0.002    # scheduler + kernel launch per decode step
MFU = 0.45                 # prefill model-FLOPs utilization
MFU_DECODE = 0.50          # batched-decode GEMM efficiency (the compute roof at large batch)
TIE = 0.10                 # within the estimate's error bars, prefer the precision with faster prefill


def load(name):
    return yaml.safe_load((HERE / name).read_text())


def _by_precision(value, precision, default):
    if isinstance(value, dict):
        return value.get(precision, value.get("default", default))
    return default if value is None else value


def kv_bytes_per_token(m, kv_dtype_bytes=2.0):
    return 2 * m["kv_layers"] * m["kv_heads"] * m["head_dim"] * kv_dtype_bytes


def kv_dtype(m, precision):
    return _by_precision(m.get("kv_bytes"), precision, 2)


def extra_gb(m, precision):
    return _by_precision(m.get("extra_weights_gb"), precision, 0.0)


def weights_gb(m, precision):
    return m["params_total_b"] * PRECISIONS[precision][0] + extra_gb(m, precision)


def weights_read_gb(m, precision, batch):
    """Bytes of weights streamed per decode step at a given batch size."""
    bpp = PRECISIONS[precision][0]
    total, active = m["params_total_b"], m["params_active_b"]
    e, k = m.get("moe_experts"), m.get("moe_active_experts")
    if not e or not k or total == active:
        return active * bpp + extra_gb(m, precision)
    frac = k / e
    shared = max((active - total * frac) / (1 - frac), 0.0)
    experts_total = total - shared
    touched = 1 - (1 - frac) ** batch
    return (shared + experts_total * touched) * bpp + extra_gb(m, precision)


def supports(gpu, precision):
    return gpu.get(PRECISIONS[precision][1]) is not None


def verified_note(model, gpu_name):
    v = model.get("nim_verified_gpus")
    if isinstance(v, list):
        return "" if gpu_name in v else "[GPU not NIM-verified for this model]"
    return "[check NIM support matrix]"


def evaluate(model, gpu, precision, isl, osl, itl_slo_ms, ttft_slo_ms, gpus=1):
    mem = gpu["mem_gb"] * gpus
    w = weights_gb(model, precision)
    budget_gb = mem * MEM_UTIL - w - OVERHEAD_GB * gpus
    res = {"precision": precision, "weights_gb": round(w, 1), "kv_budget_gb": round(budget_gb, 1)}
    if not supports(gpu, precision):
        return dict(res, fits=False, note=f"{precision} not supported on {gpu['arch']}")
    if budget_gb <= 0:
        return dict(res, fits=False, note="weights do not fit")
    kvb = kv_bytes_per_token(model, kv_dtype(model, precision))
    state_gb = (model.get("mamba_state_mb_per_seq") or 0) / 1e3
    per_seq_gb = (isl + osl) * kvb / 1e9 + state_gb
    max_b_mem = max(int(budget_gb / per_seq_gb), 0)
    bw = gpu["bw_gbps"] * gpus * BW_EFF
    # per sequence per step: KV for the average decode context (isl + osl/2) + SSM state read + written
    seq_read = (isl + osl / 2) * kvb / 1e9 + 2 * state_gb

    tflops = gpu[PRECISIONS[precision][1]] * gpus

    def step_s(b):
        mem_s = (weights_read_gb(model, precision, b) + b * seq_read) / bw
        compute_s = 2 * model["params_active_b"] * 1e9 * b / (tflops * 1e12 * MFU_DECODE)
        return max(mem_s, compute_s) + STEP_OVERHEAD_S

    floor_ms = step_s(1) * 1e3
    best = 0
    for b in range(1, max_b_mem + 1):
        if step_s(b) * 1e3 <= itl_slo_ms:
            best = b
        else:
            break
    ttft_ms = 2 * model["params_active_b"] * 1e9 * isl / (tflops * 1e12 * MFU) * 1e3
    res.update(fits=True, decode_floor_ms=round(floor_ms, 1), max_batch_mem=max_b_mem, ttft_est_ms=round(ttft_ms))
    if best == 0:
        return dict(res, max_batch_slo=0, tok_s=0.0, note=f"physics veto: ITL floor {floor_ms:.0f} ms > SLO {itl_slo_ms} ms")
    tok_s = best / step_s(best)
    note = "" if ttft_ms <= ttft_slo_ms else f"TTFT {ttft_ms:.0f} ms > SLO at zero queueing"
    if best == max_b_mem:
        note = (note + "; " if note else "") + "memory-bound batch"
    return dict(res, max_batch_slo=best, itl_ms=round(step_s(best) * 1e3, 1), tok_s=round(tok_s), note=note)


def rank(model_name, isl, osl, itl_slo_ms, ttft_slo_ms, precisions, clouds=None, use_spot=False, nvaie=0.0, models=None):
    gpus_db, insts, models = load("gpus.yaml"), load("instances.yaml"), (models or load("models.yaml"))
    model = models[model_name]
    if model.get("nim_precisions"):          # only precisions that exist as serving profiles for this model
        precisions = [p for p in precisions if p in model["nim_precisions"]] or list(model["nim_precisions"])
    rows = []
    for inst in insts:
        if clouds and inst["cloud"] not in clouds:
            continue
        gpu = gpus_db[inst["gpu"]]
        price = inst["spot"] if use_spot and inst.get("spot") else inst["on_demand"]
        price += nvaie * inst["gpus"]
        best = None
        for p in precisions:
            r = evaluate(model, gpu, p, isl, osl, itl_slo_ms, ttft_slo_ms, inst["gpus"])
            if best is None or (r.get("tok_s") and not best.get("tok_s")):
                best = r
            elif r.get("tok_s") and best.get("tok_s"):
                # more decode throughput wins; within 5% the faster prefill (TTFT) wins (NVFP4 vs W4A16 on Blackwell)
                if r["tok_s"] > best["tok_s"] * (1 + TIE) or (r["tok_s"] >= best["tok_s"] * (1 - TIE) and r["ttft_est_ms"] < best["ttft_est_ms"]):
                    best = r
        usd = price / (best["tok_s"] * 3600) * 1e6 if best.get("tok_s") else math.inf
        rows.append({**inst, **best, "usd_per_hour": round(price, 3), "usd_per_1m_tok": round(usd, 3) if usd != math.inf else None,
                     "verified": verified_note(model, inst["gpu"])})
    rows.sort(key=lambda r: (r["usd_per_1m_tok"] is None, r["usd_per_1m_tok"] or 0))
    return rows


def main():
    signal.signal(signal.SIGPIPE, signal.SIG_DFL)   # allow `| head` without a traceback
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="llama-3.1-8b")
    ap.add_argument("--models-file", help="extra models YAML (e.g. addon-nemotron35/models.yaml)")
    ap.add_argument("--isl", type=int, default=600)
    ap.add_argument("--osl", type=int, default=200)
    ap.add_argument("--itl-slo-ms", type=float, default=40)
    ap.add_argument("--ttft-slo-ms", type=float, default=1000)
    ap.add_argument("--precisions", default="bf16,fp8,nvfp4")
    ap.add_argument("--clouds", default="")
    ap.add_argument("--spot", action="store_true")
    ap.add_argument("--nvaie", type=float, default=0.0, help="NVIDIA AI Enterprise $/GPU-hour to add (marketplace list: 1.0)")
    a = ap.parse_args()
    models = load("models.yaml")
    if a.models_file:
        models.update(yaml.safe_load(Path(a.models_file).read_text()))
    rows = rank(a.model, a.isl, a.osl, a.itl_slo_ms, a.ttft_slo_ms, a.precisions.split(","),
                [c for c in a.clouds.split(",") if c], a.spot, a.nvaie, models)
    print(f"{a.model}  ISL={a.isl} OSL={a.osl}  ITL SLO={a.itl_slo_ms:g} ms  TTFT SLO={a.ttft_slo_ms:g} ms  "
          f"{'spot' if a.spot else 'on-demand'}" + (f" + NVAIE ${a.nvaie}/GPU-h" if a.nvaie else ""))
    print(f"{'cloud':<6}{'instance':<39}{'gpu':<17}{'prec':<6}{'$/h':>7}{'floor ms':>9}{'batch':>6}{'tok/s':>7}{'$/1M tok':>9}  note")
    for r in rows:
        usd = f"{r['usd_per_1m_tok']:.3f}" if r.get("usd_per_1m_tok") is not None else "-"
        note = (r.get("note", "") + " " + r["verified"]).strip()
        print(f"{r['cloud']:<6}{r['instance']:<39}{r['gpu']:<17}{r['precision']:<6}{r['usd_per_hour']:>7.3f}"
              f"{r.get('decode_floor_ms', '-')!s:>9}{r.get('max_batch_slo', '-')!s:>6}{r.get('tok_s', '-')!s:>7}{usd:>9}  {note}")


if __name__ == "__main__":
    main()
