#!/usr/bin/env python3
"""Is decode at the memory roofline? Compare a measured inter-token latency with the bytes-per-step floor (🔁 Lab 01).

Usage:
  python3 profiling/roofline_check.py --gpu H100-SXM --batch 1 --ctx 2600 --measured-itl-ms 6.1
  python3 profiling/roofline_check.py --gpu H100-SXM --aiperf results/aiperf/c16        # ITL p50, ISL, concurrency
  python3 profiling/roofline_check.py --gpu L40S --precision fp8 --kv fp8 --batch 32 --ctx 3000   # floor only
  python3 profiling/roofline_check.py --gpu H100-SXM --sweep 1,4,16,32,64,128,256      # floor vs batch size
  python3 profiling/roofline_check.py results/bench/vllm-base.json    # per level, ITL estimated from the agent bench
Model: --config archmath/configs/llama-3.1-8b.json (the default; any dense Llama-style config.json works; MoE, MLA
and hybrid models belong in archmath/, M2).
Why: one decode step reads every weight once plus every sequence's KV cache. For Llama-3.1-8B BF16 on an H100 at
batch 1 that is ~15.4 GB, a ~4.6 ms floor. Measure 12 ms and you are paying for overhead, not for bandwidth.
GPU numbers = archmath/archmath.py GPUS (dense tensor TFLOPS, peak HBM bandwidth; same as Lab 04).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO / "archmath" / "configs" / "llama-3.1-8b.json"
GPUS = {
    "H100-SXM": {"mem_gb": 80, "bw_tbs": 3.35, "bf16_tflops": 989, "fp8_tflops": 1979, "fp4_tflops": None},
    "H100-NVL": {"mem_gb": 94, "bw_tbs": 3.9, "bf16_tflops": 835, "fp8_tflops": 1671, "fp4_tflops": None},
    "H100-PCIe": {"mem_gb": 80, "bw_tbs": 2.0, "bf16_tflops": 756, "fp8_tflops": 1513, "fp4_tflops": None},
    "H200-SXM": {"mem_gb": 141, "bw_tbs": 4.8, "bf16_tflops": 989, "fp8_tflops": 1979, "fp4_tflops": None},
    "A100-80GB-SXM": {"mem_gb": 80, "bw_tbs": 2.039, "bf16_tflops": 312, "fp8_tflops": None, "fp4_tflops": None},
    "L40S": {"mem_gb": 48, "bw_tbs": 0.864, "bf16_tflops": 362, "fp8_tflops": 733, "fp4_tflops": None},
    "B200-in-GB200": {"mem_gb": 186, "bw_tbs": 8.0, "bf16_tflops": 2500, "fp8_tflops": 5000, "fp4_tflops": 10000},
}
PRECISION_BYTES = {"bf16": 2.0, "fp8": 1.0, "nvfp4": 0.5625}     # NVFP4: 4-bit values + an FP8 scale per 16
LLAMA_31_8B = {"hidden_size": 4096, "num_hidden_layers": 32, "num_attention_heads": 32, "num_key_value_heads": 8,
               "head_dim": 128, "intermediate_size": 14336, "vocab_size": 128256, "tie_word_embeddings": False,
               "hidden_act": "silu", "_name": "llama-3.1-8b (built in)"}


def load_config(path: str | None) -> dict:
    p = Path(path) if path else DEFAULT_CONFIG
    if p.exists():
        cfg = json.loads(p.read_text())
        cfg.setdefault("_name", p.stem)
        return cfg
    if path:
        raise FileNotFoundError(path)
    return dict(LLAMA_31_8B)


def dense_model(cfg: dict) -> dict:
    """Parameter and KV-cache sizes of a dense decoder (Llama, Mistral, Qwen2 style: GQA + gated MLP)."""
    t = cfg.get("text_config") or cfg
    if any(k in t for k in ("num_local_experts", "n_routed_experts", "num_experts", "kv_lora_rank")):
        raise ValueError("MoE / MLA config: use python3 archmath/archmath.py (M2) for its roofline")
    h, L = int(t["hidden_size"]), int(t["num_hidden_layers"])
    nh = int(t["num_attention_heads"])
    nkv = int(t.get("num_key_value_heads") or nh)
    hd = int(t.get("head_dim") or h // nh)
    inter, vocab = int(t["intermediate_size"]), int(t["vocab_size"])
    gated = str(t.get("hidden_act", "silu")).lower() in ("silu", "swish", "swiglu", "gelu_pytorch_tanh")
    attn = h * nh * hd + 2 * h * nkv * hd + nh * hd * h
    mlp = (3 if gated else 2) * h * inter
    block_linear = L * (attn + mlp)
    norms = L * 2 * h + h
    embed = vocab * h
    lm_head = 0 if t.get("tie_word_embeddings") else vocab * h
    return {"name": cfg.get("_name", "model"), "layers": L, "hidden": h, "heads": nh, "kv_heads": nkv, "head_dim": hd,
            "block_linear_params": block_linear, "norm_params": norms, "embed_params": embed,
            "lm_head_params": lm_head, "total_params": block_linear + norms + embed + lm_head,
            "kv_bytes_per_token_bf16": 2 * L * nkv * hd * 2, "vocab": vocab}


def step_floor(m: dict, gpu: str, precision: str = "bf16", kv: str = "bf16", batch: int = 1, ctx: int = 2600,
               gpus: int = 1) -> dict:
    """Bytes and FLOPs of ONE decode step for `batch` sequences that each attend over `ctx` tokens.
    Block linears are stored at `precision`; the lm_head and norms stay BF16 (what FP8/NVFP4 checkpoints do); the
    input-embedding table is gathered (batch rows), not streamed. --gpus divides both evenly (ideal TP)."""
    g = GPUS[gpu]
    head = m["lm_head_params"] or m["embed_params"]                  # tied embeddings: the lm_head reads the table
    w = m["block_linear_params"] * PRECISION_BYTES[precision] + (m["norm_params"] + head) * 2.0 \
        + batch * m["hidden"] * 2.0
    kv_tok = m["kv_bytes_per_token_bf16"] * (0.5 if kv == "fp8" else 1.0)
    kv_read = batch * ctx * kv_tok
    flops = batch * (2.0 * (m["block_linear_params"] + head) + 4.0 * m["layers"] * m["heads"] * m["head_dim"] * ctx)
    peak = g["bf16_tflops"]
    if precision == "fp8" and g["fp8_tflops"]:
        peak = g["fp8_tflops"]
    if precision == "nvfp4":
        peak = g["fp4_tflops"] or g["fp8_tflops"] or g["bf16_tflops"]
    total = (w + kv_read) / gpus
    t_mem = total / (g["bw_tbs"] * 1e12)
    t_cmp = flops / gpus / (peak * 1e12)
    step = max(t_mem, t_cmp)
    return {"model": m["name"], "gpu": gpu, "gpus": gpus, "precision": precision, "kv": kv, "batch": batch, "ctx": ctx,
            "weights_gb": w / 1e9 / gpus, "kv_gb": kv_read / 1e9 / gpus, "bytes_gb": total / 1e9,
            "gflops": flops / 1e9 / gpus, "t_mem_ms": t_mem * 1e3, "t_compute_ms": t_cmp * 1e3,
            "bound": "memory" if t_mem >= t_cmp else "compute", "floor_ms": step * 1e3,
            "tok_s_per_seq": 1.0 / step, "tok_s_total": batch / step,
            "kv_free_gb": g["mem_gb"] * gpus * 0.9 / gpus - w / 1e9 / gpus,
            "fits": kv_read / gpus <= (g["mem_gb"] * 0.9 * 1e9 - w / gpus),
            "max_batch_kv": int(max(0.0, g["mem_gb"] * 0.9 * 1e9 - w / gpus) / (ctx * kv_tok / gpus)) if ctx else 0}


def compare(floor: dict, measured_itl_ms: float, gpu: str) -> dict:
    """Achieved bandwidth = the floor's bytes moved in the measured step time."""
    bw = floor["bytes_gb"] * 1e9 / (measured_itl_ms / 1e3)
    frac = bw / (GPUS[gpu]["bw_tbs"] * 1e12)
    return {"measured_itl_ms": measured_itl_ms, "achieved_tbs": bw / 1e12, "bw_fraction": frac,
            "floor_over_measured": floor["floor_ms"] / measured_itl_ms, "verdict": verdict(frac, floor["bound"])}


def verdict(frac: float, bound: str = "memory") -> str:
    if bound == "compute":
        return ("compute-bound at this batch: the lever is FLOPs - FP8 tensor cores, or a smaller batch per GPU if "
                "latency matters more than throughput")
    if frac > 1.05:
        return ("faster than the floor: several tokens per weight read (speculative decoding) or fewer bytes than "
                "assumed (check --precision/--kv and the real batch) - verify the acceptance rate")
    if frac >= 0.8:
        return ("at the roofline: only fewer bytes help - FP8 weights (~half), FP8 KV cache, a smaller model, or "
                "speculative decoding (more tokens per weight read)")
    if frac >= 0.5:
        return ("typical for a loaded server: scheduling, sampling and small kernels between steps - confirm CUDA "
                "graphs are on and that prefill chunks are not inflating ITL (profiling/analyze_trace.py)")
    return ("overhead-bound: the GPU is waiting - eager mode, CPU scheduling gaps, prefill interference or a shared "
            "GPU; profile it (profiling/torch_profile.sh) before buying more GPUs")


# ----------------------------------------------------------------------------------------------- AIPerf input
def find_export(path: str) -> Path:
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


def _metric(d: dict, key: str, stat: str):
    v = d.get(key)
    if isinstance(v, dict):
        x = v.get(stat, v.get("avg"))
        return float(x) if isinstance(x, (int, float)) else None
    return float(v) if isinstance(v, (int, float)) else None


def _concurrency(cfg) -> int | None:
    """Best effort: the profiling phase's concurrency from AIPerf's input_config (schema varies by version)."""
    found = []

    def walk(x, profiling=False):
        if isinstance(x, dict):
            prof = profiling or str(x.get("name", x.get("kind", ""))).lower() == "profiling"
            c = x.get("concurrency")
            if isinstance(c, int) and not isinstance(c, bool) and c > 0:
                found.append((prof, c))
            for v in x.values():
                walk(v, prof)
        elif isinstance(x, list):
            for v in x:
                walk(v, profiling)
    walk(cfg)
    if not found:
        return None
    prof = [c for p, c in found if p]
    return prof[0] if prof else found[0][1]


def read_aiperf(path: str, stat: str = "p50") -> dict:
    f = find_export(path)
    d = json.loads(f.read_text())
    itl = _metric(d, "inter_token_latency", stat)
    unit = (d.get("inter_token_latency") or {}).get("unit", "ms") if isinstance(d.get("inter_token_latency"), dict) \
        else "ms"
    if itl is not None and str(unit).lower() in ("ns", "nanoseconds"):
        itl /= 1e6
    elif itl is not None and str(unit).lower() in ("s", "sec", "seconds"):
        itl *= 1e3
    return {"file": str(f), "itl_ms": itl, "isl": _metric(d, "input_sequence_length", "avg"),
            "osl": _metric(d, "output_sequence_length", "avg"), "concurrency": _concurrency(d.get("input_config"))}


# ---------------------------------------------------------------------------------------- agent bench input
def guess_gpu(name: str) -> str | None:
    n = (name or "").upper()
    for key, gpu in (("GB200", "B200-in-GB200"), ("B200", "B200-in-GB200"), ("H100 NVL", "H100-NVL"),
                     ("H100-NVL", "H100-NVL"), ("H100 PCIE", "H100-PCIe"),
                     ("H100-PCIE", "H100-PCIe"), ("H100", "H100-SXM"), ("H200", "H200-SXM"), ("A100", "A100-80GB-SXM"),
                     ("L40S", "L40S")):
        if key in n:
            return gpu
    return None


def estimate_from_tasks(tasks: list, concurrency: int) -> dict | None:
    """Rough decode numbers from TaskResults (agent_bench): ITL ~ (LLM time - calls x TTFT) / (output - calls) tokens;
    sequences decoding at once ~ concurrency x LLM share of task time; context ~ prompt per call + output/2."""
    itl, share, ctx = [], [], []
    for t in tasks:
        calls, out = t.get("llm_calls") or 0, t.get("completion_tokens") or 0
        llm, ttft, lat = t.get("llm_time_s"), t.get("first_ttft_s"), t.get("latency_s")
        if not calls or out <= calls or llm is None or ttft is None or not lat:
            continue
        decode = llm - calls * ttft
        if decode <= 0:
            continue
        itl.append(decode / (out - calls) * 1e3)
        share.append(min(1.0, llm / lat))
        ctx.append((t.get("prompt_tokens") or 0) / calls + out / calls / 2)
    if not itl:
        return None
    itl.sort()
    return {"itl_ms": itl[len(itl) // 2], "batch": max(1, round(concurrency * sum(share) / len(share))),
            "ctx": int(sum(ctx) / len(ctx)), "tasks": len(itl)}


def from_bench(path: str, m: dict, gpu: str | None, precision: str | None, kv: str | None, gpus: int) -> dict:
    d = json.loads(Path(path).read_text())
    variant = str(d.get("variant") or "")
    gpu = gpu or guess_gpu(str(d.get("gpu") or "")) or "H100-SXM"
    precision = precision or ("fp8" if variant == "fp8" else "bf16")
    kv = kv or ("fp8" if variant == "fp8kv" else "bf16")
    # TP splits every layer's bytes across GPUs; PP runs the stages one after the other for each token, so a
    # pipeline's per-sequence decode floor is the single-GPU one (PP buys capacity, not latency).
    gpus = (d.get("gpus") or gpus) if variant == "tp2" else 1 if variant == "pp2" else gpus
    rows = []
    for lv in d.get("levels", []):
        c = lv["concurrency"]
        est = estimate_from_tasks([t for t in d.get("tasks", []) if t.get("concurrency") == c], c)
        if not est:
            continue
        floor = step_floor(m, gpu, precision, kv, est["batch"], est["ctx"], gpus)
        rows.append({"concurrency": c, **est, "floor_ms": floor["floor_ms"],
                     **compare(floor, est["itl_ms"], gpu)})
    return {"label": d.get("label"), "engine": d.get("engine"), "variant": variant, "gpu": gpu, "gpus": gpus,
            "precision": precision, "kv": kv, "levels": rows}


def render_bench(r: dict) -> str:
    out = [f"{r['label']} ({r['engine']}, {r['variant']}) on {r['gpus']}x {r['gpu']}, weights {r['precision']}, "
           f"KV {r['kv']} - ITL ESTIMATED from task timings (AIPerf ITL is the precise number)",
           f"{'conc':>5} {'~batch':>7} {'~ctx':>6} {'~ITL ms':>8} {'floor ms':>9} {'% peak BW':>10}  read it"]
    for x in r["levels"]:
        out.append(f"{x['concurrency']:>5} {x['batch']:>7} {x['ctx']:>6,} {x['itl_ms']:>8.2f} {x['floor_ms']:>9.2f} "
                   f"{x['bw_fraction'] * 100:>9.0f}%  {x['verdict'].split(':')[0]}")
    if not r["levels"]:
        out.append("  no usable tasks (need llm_time_s, first_ttft_s, completion_tokens per task)")
    return "\n".join(out)


# ------------------------------------------------------------------------------------------------------- main
def render(floor: dict, cmp: dict | None, m: dict) -> str:
    out = [f"{floor['model']} on {floor['gpus']}x {floor['gpu']} | weights {floor['precision']}, KV {floor['kv']} | "
           f"batch {floor['batch']} x {floor['ctx']:,} tokens of context",
           f"  params {m['total_params'] / 1e9:.2f} B | KV cache {m['kv_bytes_per_token_bf16'] / 1024:.0f} KiB/token "
           f"(BF16)",
           f"  bytes per decode step: weights {floor['weights_gb']:.2f} GB + KV {floor['kv_gb']:.2f} GB = "
           f"{floor['bytes_gb']:.2f} GB",
           f"  floor: memory {floor['t_mem_ms']:.2f} ms vs compute {floor['t_compute_ms']:.2f} ms -> "
           f"{floor['floor_ms']:.2f} ms/step ({floor['bound']}-bound), <= {floor['tok_s_per_seq']:,.0f} tok/s per "
           f"sequence, <= {floor['tok_s_total']:,.0f} tok/s total",
           f"  KV room after weights (90% of HBM): {floor['kv_free_gb']:.1f} GB -> at most {floor['max_batch_kv']:,} "
           f"sequences of {floor['ctx']:,} tokens" + ("" if floor["fits"] else "  ** this batch does NOT fit **")]
    if cmp:
        out.append(f"  measured ITL {cmp['measured_itl_ms']:.2f} ms -> {cmp['achieved_tbs']:.2f} TB/s achieved = "
                   f"{cmp['bw_fraction'] * 100:.0f}% of {GPUS[floor['gpu']]['bw_tbs']} TB/s peak "
                   f"(floor/measured {cmp['floor_over_measured'] * 100:.0f}%)")
        out.append(f"  read it: {cmp['verdict']}")
    out.append("  caveat: ITL under load includes prefill chunks of new requests; for this check use a decode-heavy "
               "window or low concurrency, and set --batch to vllm:num_requests_running.")
    return "\n".join(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bench", nargs="?", help="an agent_bench file (results/bench/<label>.json): check every level")
    ap.add_argument("--config", help=f"HF config.json (default {DEFAULT_CONFIG.relative_to(REPO)})")
    ap.add_argument("--gpu", choices=sorted(GPUS), help="default H100-SXM (bench mode: guessed from the file)")
    ap.add_argument("--gpus", type=int, default=1, help="tensor-parallel degree (ideal split, no all-reduce cost)")
    ap.add_argument("--precision", choices=sorted(PRECISION_BYTES), help="weights (default bf16; bench: variant)")
    ap.add_argument("--kv", choices=["bf16", "fp8"], help="KV cache dtype (default bf16; bench: variant)")
    ap.add_argument("--batch", type=int, help="sequences decoding together (default: AIPerf concurrency, else 1)")
    ap.add_argument("--ctx", type=int, help="mean context per sequence during decode (default: ISL + OSL/2, "
                                            "else 2600 = the agent's ~2K-token system prompt + history)")
    ap.add_argument("--measured-itl-ms", type=float, help="measured inter-token latency in ms")
    ap.add_argument("--aiperf", help="AIPerf artifact dir or profile_export_aiperf.json (ITL p50, ISL, concurrency)")
    ap.add_argument("--stat", default="p50", help="which ITL statistic to read from AIPerf (p50, avg, p90 ...)")
    ap.add_argument("--sweep", help="comma-separated batch sizes: print the floor table instead")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)

    m = dense_model(load_config(a.config))
    if a.bench:
        r = from_bench(a.bench, m, a.gpu, a.precision, a.kv, a.gpus)
        print(json.dumps(r, indent=2) if a.json else render_bench(r))
        return 0
    a.gpu, a.precision, a.kv = a.gpu or "H100-SXM", a.precision or "bf16", a.kv or "bf16"
    if a.sweep:
        rows = [step_floor(m, a.gpu, a.precision, a.kv, int(b), a.ctx or 2600, a.gpus) for b in a.sweep.split(",")]
        if a.json:
            print(json.dumps(rows, indent=2))
            return 0
        print(f"{m['name']} on {a.gpus}x {a.gpu}, weights {a.precision}, KV {a.kv}, ctx {a.ctx or 2600:,}")
        print(f"{'batch':>6} {'GB/step':>8} {'floor ms':>9} {'bound':>8} {'tok/s/seq':>10} {'tok/s total':>12}  KV fits?")
        for r in rows:
            print(f"{r['batch']:>6} {r['bytes_gb']:>8.2f} {r['floor_ms']:>9.2f} {r['bound']:>8} "
                  f"{r['tok_s_per_seq']:>10,.0f} {r['tok_s_total']:>12,.0f}  "
                  f"{'yes' if r['fits'] else 'NO - KV > free HBM'}")
        print(f"KV room after weights (90% of HBM): {rows[0]['kv_free_gb']:.1f} GB per GPU -> at most "
              f"{rows[0]['max_batch_kv']:,} sequences of {a.ctx or 2600:,} tokens (FP8 KV: --kv fp8 doubles it)")
        print("Prefix caching stores the shared ~2K-token system prompt ONCE: capacity rises well above this, but "
              "bytes per step do not (every sequence still reads its prefix).")
        return 0

    measured, batch, ctx, source = a.measured_itl_ms, a.batch, a.ctx, None
    if a.aiperf:
        ai = read_aiperf(a.aiperf, a.stat)
        source = ai["file"]
        measured = measured or ai["itl_ms"]
        batch = batch or ai["concurrency"]
        if ctx is None and ai["isl"]:
            ctx = int(ai["isl"] + (ai["osl"] or 0) / 2)
    batch, ctx = batch or 1, ctx or 2600
    floor = step_floor(m, a.gpu, a.precision, a.kv, batch, ctx, a.gpus)
    cmp = compare(floor, measured, a.gpu) if measured else None
    if a.json:
        print(json.dumps({"source": source, "floor": floor, "measured": cmp}, indent=2))
    else:
        if source:
            print(f"AIPerf: {source}")
        print(render(floor, cmp, m))
        if not math.isfinite(floor["floor_ms"]):
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
