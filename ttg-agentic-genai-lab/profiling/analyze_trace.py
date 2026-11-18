#!/usr/bin/env python3
"""Turn a GPU profile into the four numbers an SA talks about: GPU busy %, where the busy time goes, how the idle
time is spread, and the top kernels.

Usage:
  python3 profiling/analyze_trace.py results/profiles/torch/                 # newest vLLM GPU-worker trace in a dir
  python3 profiling/analyze_trace.py <file>.pt.trace.json.gz --top 20 --gaps 10 --json results/profiles/summary.json
  python3 profiling/analyze_trace.py --nsys-csv results/profiles/nsys/kern_sum.csv  # nsys stats cuda_gpu_kern_sum
Input 1: a PyTorch-profiler Chrome trace (.json or .json.gz, what vLLM's --profiler-config writes; the AsyncLLM
  CPU-only trace in the same directory is skipped). Input 2: the CSV of `nsys stats --report cuda_gpu_kern_sum
  --format csv` (kernel totals only, so no idle-gap analysis).
Categories come from kernel names (GEMM, attention, communication, norm/act/elementwise, quantize, sampling,
memcpy/memset, other). It is a heuristic: read the top-20 table before you quote a category share.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

GPU_CATS = {"kernel", "gpu_memcpy", "gpu_memset"}
CATEGORY_ORDER = ["GEMM", "attention", "communication", "norm/act/elementwise", "quantize", "sampling",
                  "memcpy/memset", "other"]
# Checked in this order: the first matching rule wins. Flash-attention kernels contain "cutlass", so attention comes
# before GEMM; NCCL kernels contain "reduce", so communication comes first; FP8 GEMMs contain "fp8"/"e4m3", so GEMM
# comes before quantize (which is left with the activation-quantization kernels).
RULES = [
    ("memcpy/memset", re.compile(r"^memcpy|^memset|memcpy[a-z]*to[a-z]|\bmemset\b")),
    ("communication", re.compile(r"nccl|all_?reduce|all_?gather|reduce_?scatter|alltoall|all_to_all|custom_?ar|"
                                 r"cross_device_reduce|one_?shot|two_?shot|multimem|\bsendrecv\b|mscclpp")),
    ("attention", re.compile(r"flash|fmha|attn|attention|paged|flashinfer|xqa|\bmla\b|_mla_|decode_attention|"
                             r"reshape_and_cache|concat_and_cache|rotary|rope|merge_attn|softmax_lse")),
    ("sampling", re.compile(r"sampl|topk|top_k|top_p|topp|argmax|argsort|radixsort|\bsort|gumbel|multinomial|"
                            r"exponential_|logit|penalt|softmaxforward|softmax_warp|log_softmax")),
    ("GEMM", re.compile(r"gemm|cutlass|cublas|xmma|nvjet|wgmma|matmul|splitk|\bmm\b|scaled_mm|triton_tem|gemv|"
                        r"ampere_|hopper_|bmm")),
    ("quantize", re.compile(r"quant|fp8|e4m3|nvfp4|fp4")),
    ("norm/act/elementwise", re.compile(r"norm|silu|gelu|act_and_mul|swiglu|elementwise|vectorized|unrolled|"
                                        r"triton_|reduce_kernel|copy|cat_|index|gather|scatter|fill|add|mul|"
                                        r"embedding|where|cast|arange|clamp")),
]


def classify(name: str, cat: str = "kernel") -> str:
    c = (cat or "").lower()
    if c in ("gpu_memcpy", "gpu_memset"):
        return "memcpy/memset"
    n = (name or "").lower()
    for label, rx in RULES:
        if rx.search(n):
            return label
    return "other"


# ------------------------------------------------------------------------------------------------- loading traces
def find_trace(path: Path) -> Path:
    """A file is used as is; in a directory the newest GPU-worker trace wins (the async_llm trace is CPU only)."""
    if path.is_file():
        return path
    cands = [p for p in path.rglob("*") if p.is_file() and re.search(r"\.json(\.gz)?$", p.name)
             and "async_llm" not in p.name and "capture_traces" not in p.parts]
    traced = [p for p in cands if ".pt.trace." in p.name] or cands
    if not traced:
        raise FileNotFoundError(f"no *.pt.trace.json(.gz) under {path}")
    return max(traced, key=lambda p: (p.stat().st_mtime, p.name))


def load_trace(path: Path) -> dict:
    raw = path.read_bytes()
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    data = json.loads(raw.decode("utf-8"))
    if isinstance(data, list):                      # bare event array is valid Chrome-trace JSON too
        data = {"traceEvents": data}
    return data


def gpu_events(trace: dict) -> list:
    out = []
    for e in trace.get("traceEvents", []):
        if e.get("ph") != "X" or str(e.get("cat", "")).lower() not in GPU_CATS:
            continue
        try:
            ts, dur = float(e["ts"]), float(e.get("dur", 0.0))
        except (KeyError, TypeError, ValueError):
            continue
        args = e.get("args") or {}
        out.append({"name": str(e.get("name", "")), "cat": str(e.get("cat", "")).lower(), "ts": ts, "dur": dur,
                    "stream": args.get("stream", e.get("tid")), "device": args.get("device", e.get("pid"))})
    out.sort(key=lambda x: x["ts"])
    return out


def runtime_calls(trace: dict) -> dict:
    """Launch API counts: cudaGraphLaunch means CUDA graphs are replaying; thousands of cudaLaunchKernel per
    decode step means eager mode (CPU launch overhead)."""
    counts: dict = defaultdict(int)
    for e in trace.get("traceEvents", []):
        if e.get("ph") == "X" and str(e.get("cat", "")).lower() in ("cuda_runtime", "cuda_driver"):
            n = str(e.get("name", ""))
            if "Graph" in n and "Launch" in n:
                counts["graph_launches"] += 1
            elif "LaunchKernel" in n:
                counts["kernel_launches"] += 1
            elif "Synchronize" in n:
                counts["syncs"] += 1
    return dict(counts)


# --------------------------------------------------------------------------------------------------- analysis
def busy_intervals(events: list) -> list:
    """Union of [ts, ts+dur) over all GPU events (all streams): the time at least one kernel or copy ran."""
    merged: list = []
    for e in events:
        s, t = e["ts"], e["ts"] + e["dur"]
        if merged and s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], t)
        else:
            merged.append([s, t])
    return merged


GAP_BUCKETS = [("<10us", 0, 10), ("10-100us", 10, 100), ("100us-1ms", 100, 1000), (">1ms", 1000, float("inf"))]


def summarize(events: list, top: int = 20, n_gaps: int = 10) -> dict:
    if not events:
        raise ValueError("no GPU events (kernel / gpu_memcpy / gpu_memset) in this trace - was CUDA activity "
                         "recorded? (vLLM: profiler 'torch' records CPU+CUDA on NVIDIA GPUs by default)")
    merged = busy_intervals(events)
    t0, t1 = merged[0][0], max(m[1] for m in merged)
    window = t1 - t0
    busy = sum(b - a for a, b in merged)
    gaps = [(merged[i][1], merged[i + 1][0] - merged[i][1]) for i in range(len(merged) - 1)]
    gaps = [(s, d) for s, d in gaps if d > 0]
    buckets = []
    for label, lo, hi in GAP_BUCKETS:
        sel = [d for _, d in gaps if lo <= d < hi]
        buckets.append({"bucket": label, "count": len(sel), "total_ms": sum(sel) / 1e3})
    largest = sorted(gaps, key=lambda g: -g[1])[:n_gaps]

    kern_total = sum(e["dur"] for e in events) or 1.0      # summed over streams (can exceed busy with overlap)
    by_cat: dict = defaultdict(float)
    per_kernel: dict = defaultdict(lambda: [0, 0.0, ""])
    for e in events:
        c = classify(e["name"], e["cat"])
        by_cat[c] += e["dur"]
        k = per_kernel[e["name"]]
        k[0] += 1
        k[1] += e["dur"]
        k[2] = c
    cats = [{"category": c, "total_ms": by_cat[c] / 1e3, "pct": 100.0 * by_cat[c] / kern_total}
            for c in CATEGORY_ORDER if by_cat.get(c)]
    kernels = sorted(({"name": n, "count": v[0], "total_ms": v[1] / 1e3, "avg_us": v[1] / v[0],
                       "pct": 100.0 * v[1] / kern_total, "category": v[2]} for n, v in per_kernel.items()),
                     key=lambda r: -r["total_ms"])
    return {"window_ms": window / 1e3, "busy_ms": busy / 1e3, "busy_pct": 100.0 * busy / window if window else 100.0,
            "idle_ms": (window - busy) / 1e3, "n_events": len(events), "n_gaps": len(gaps),
            "streams": len({e["stream"] for e in events}), "gap_buckets": buckets,
            "largest_gaps": [{"at_ms": (s - t0) / 1e3, "gap_us": d} for s, d in largest],
            "categories": cats, "top_kernels": kernels[:top], "n_kernels": len(kernels)}


def diagnose(s: dict, calls: dict | None = None) -> list:
    """Plain-language reading of the numbers (what Jay would say next)."""
    pct = {c["category"]: c["pct"] for c in s["categories"]}
    out = []
    if s.get("busy_pct") is not None and s["busy_pct"] < 70:
        out.append(f"GPU busy only {s['busy_pct']:.0f}% of the window: the GPU waits on the CPU (scheduler, "
                   "Python, kernel launches) or on too little work - raise concurrency, keep CUDA graphs on.")
    if calls:
        if calls.get("graph_launches", 0) == 0 and calls.get("kernel_launches", 0) > 0:
            out.append(f"No cudaGraphLaunch and {calls['kernel_launches']:,} kernel launches: eager mode - every "
                       "decode step pays CPU launch latency for every kernel (compare the 'eager' variant).")
        elif calls.get("graph_launches", 0) > 0:
            out.append(f"CUDA graphs are replaying ({calls['graph_launches']:,} cudaGraphLaunch calls).")
    small = next((b for b in s.get("gap_buckets", []) if b["bucket"] == "<10us"), None)
    if small and s.get("n_gaps") and small["count"] > 0.8 * s["n_gaps"] and s.get("busy_pct", 100) >= 70:
        out.append("Most idle gaps are under 10 us: back-to-back kernels - the remaining lever is fewer bytes, not "
                   "less overhead.")
    if pct.get("GEMM", 0) > 50:
        out.append(f"GEMM is {pct['GEMM']:.0f}% of GPU time: decode streams every weight once per step - FP8 weights "
                   "halve those bytes (profiling/roofline_check.py, 🔁 Lab 01).")
    if pct.get("attention", 0) > 30:
        out.append(f"Attention + KV cache is {pct['attention']:.0f}%: long contexts - prefix caching, FP8 KV cache "
                   "and shorter prompts all cut it.")
    if pct.get("communication", 0) > 10:
        out.append(f"Communication is {pct['communication']:.0f}%: tensor-parallel all-reduce on the critical path - "
                   "NVLink vs PCIe matters here (M7).")
    if pct.get("sampling", 0) > 5:
        out.append(f"Sampling is {pct['sampling']:.0f}%: 128K-token vocabulary softmax/top-k every step - check "
                   "sampling params and batch size.")
    if pct.get("memcpy/memset", 0) > 5:
        out.append(f"Copies are {pct['memcpy/memset']:.0f}%: host-device traffic - look for CPU offload, pageable "
                   "memory or per-step tensor copies.")
    if pct.get("quantize", 0) > 3:
        out.append(f"Quantize kernels are {pct['quantize']:.0f}%: the activation-quantization cost of FP8 - still "
                   "worth it if the GEMM time dropped more.")
    return out


# ------------------------------------------------------------------------------------------------------- nsys
def _num(s: str) -> float:
    try:
        return float(str(s).replace(",", "").strip() or 0)
    except ValueError:
        return 0.0


def parse_nsys_kern_sum(text: str) -> list:
    """Rows of `nsys stats --report cuda_gpu_kern_sum --format csv`; preamble lines before the header are skipped."""
    lines = text.splitlines()
    start = next((i for i, ln in enumerate(lines) if "Total Time" in ln and "Name" in ln), None)
    if start is None:
        raise ValueError("no cuda_gpu_kern_sum header (Time (%), Total Time (ns), ..., Name) found")
    rows = []
    for r in csv.DictReader(io.StringIO("\n".join(lines[start:]))):
        name = r.get("Name") or ""
        total = next((_num(v) for k, v in r.items() if k and k.startswith("Total Time")), 0.0)
        inst = next((_num(v) for k, v in r.items() if k and k.startswith("Instances")), 0.0)
        if name:
            rows.append({"name": name, "total_ns": total, "instances": int(inst)})
    return rows


def summarize_nsys(rows: list, top: int = 20) -> dict:
    total = sum(r["total_ns"] for r in rows) or 1.0
    by_cat: dict = defaultdict(float)
    for r in rows:
        r["category"] = classify(r["name"])
        by_cat[r["category"]] += r["total_ns"]
    cats = [{"category": c, "total_ms": by_cat[c] / 1e6, "pct": 100.0 * by_cat[c] / total}
            for c in CATEGORY_ORDER if by_cat.get(c)]
    kernels = sorted(({"name": r["name"], "count": r["instances"], "total_ms": r["total_ns"] / 1e6,
                       "avg_us": r["total_ns"] / max(r["instances"], 1) / 1e3, "pct": 100.0 * r["total_ns"] / total,
                       "category": r["category"]} for r in rows), key=lambda r: -r["total_ms"])
    return {"kernel_time_ms": total / 1e6, "categories": cats, "top_kernels": kernels[:top], "n_kernels": len(rows),
            "busy_pct": None}


# ------------------------------------------------------------------------------------------------------ output
def _short(name: str, n: int = 88) -> str:
    return name if len(name) <= n else name[: n - 3] + "..."


def _dur(us: float) -> str:
    return f"{us / 1e3:,.2f} ms" if us >= 1000 else f"{us:,.1f} us"


def render(s: dict, source: str, calls: dict | None = None) -> str:
    out = [f"Trace: {source}"]
    if s.get("window_ms") is not None:
        out.append(f"Window {s['window_ms']:,.1f} ms | GPU busy {s['busy_pct']:.1f}% | idle {s['idle_ms']:,.1f} ms in "
                   f"{s['n_gaps']:,} gaps | {s['n_events']:,} GPU events on {s['streams']} stream(s)")
    else:
        out.append(f"Kernel time {s['kernel_time_ms']:,.1f} ms over {s['n_kernels']:,} kernel names (nsys summary: "
                   "no timeline, so no idle-gap analysis)")
    if calls:
        out.append(f"Launch APIs: cudaGraphLaunch x {calls.get('graph_launches', 0):,} | kernel launches x "
                   f"{calls.get('kernel_launches', 0):,} | synchronizes x {calls.get('syncs', 0):,}")
    out.append("")
    out.append("Where the GPU time goes (share of summed kernel time):")
    for c in s["categories"]:
        out.append(f"  {c['category']:<22} {c['pct']:5.1f}%  {c['total_ms']:10,.2f} ms")
    if s.get("gap_buckets"):
        out.append("")
        out.append("Idle gaps between GPU work: " + " | ".join(
            f"{b['bucket']}: {b['count']:,} ({b['total_ms']:,.2f} ms)" for b in s["gap_buckets"]))
        if s["largest_gaps"]:
            out.append("  largest: " + ", ".join(f"{_dur(g['gap_us'])} at +{g['at_ms']:,.2f} ms"
                                                 for g in s["largest_gaps"][:5]))
    out.append("")
    out.append(f"Top {len(s['top_kernels'])} kernels:")
    out.append(f"  {'%':>5}  {'total ms':>10}  {'count':>7}  {'avg us':>9}  {'category':<20}  name")
    for k in s["top_kernels"]:
        out.append(f"  {k['pct']:5.1f}  {k['total_ms']:10,.2f}  {k['count']:7,}  {k['avg_us']:9,.1f}  "
                   f"{k['category']:<20}  {_short(k['name'])}")
    notes = diagnose(s, calls)
    if notes:
        out.append("")
        out.append("Read it:")
        out.extend(f"  - {n}" for n in notes)
    return "\n".join(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("trace", nargs="?", help="trace file or directory (torch profiler output)")
    ap.add_argument("--nsys-csv", help="CSV from: nsys stats --report cuda_gpu_kern_sum --format csv <rep>")
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--gaps", type=int, default=10, help="largest idle gaps to keep")
    ap.add_argument("--json", help="also write the summary as JSON here")
    a = ap.parse_args(argv)
    if a.nsys_csv:
        s = summarize_nsys(parse_nsys_kern_sum(Path(a.nsys_csv).read_text(errors="replace")), a.top)
        source, calls = a.nsys_csv, None
    elif a.trace:
        path = find_trace(Path(a.trace))
        trace = load_trace(path)
        s = summarize(gpu_events(trace), a.top, a.gaps)
        source, calls = str(path), runtime_calls(trace)
        s["launch_calls"] = calls
    else:
        ap.error("give a trace file/directory or --nsys-csv")
    print(render(s, source, calls))
    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json).write_text(json.dumps({"source": source, **s, "notes": diagnose(s, calls)}, indent=2))
        print(f"\nwrote {a.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
