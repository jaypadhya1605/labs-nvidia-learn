#!/usr/bin/env python3
"""Cold start, itemized: how long an engine takes from container start to "ready", and where that time goes.

Usage:
  engines/logs.sh --save results/logs/vllm-base.log          # docker logs WITH timestamps (-t)
  python3 profiling/coldstart.py results/logs/*.log          # one block per log
  python3 profiling/coldstart.py results/logs/vllm-base.log --engine vllm --json
Parses (verified against vLLM 0.31 and SGLang 0.5.22 log lines):
  vLLM    "Time spent downloading weights for <model>: N seconds", "Loading weights took N seconds",
          "Model loading took X GiB memory and N seconds", "Available KV cache memory: X GiB",
          "GPU KV cache size: N tokens, Maximum concurrency for M tokens per request: Kx",
          "Graph capturing finished in N secs, took X GiB",
          "init engine (profile, create kv cache, warmup model) took N s (compilation: C s)", "Application startup complete"
  SGLang  "Load weight end. elapsed=N s, ... mem usage=X GB.", "KV Cache is allocated. dtype: D, #tokens: N, ...",
          "Capture <name> CUDA graph end. elapsed=N s, mem usage=X GB", "The server is fired up and ready to roll!"
  trtllm-serve / NIM / hf: wall time from docker timestamps to "Application startup complete" (uvicorn).
Why: on a storm day the autoscaler adds GPUs; a replica that needs 4 minutes to start serves nobody for 4 minutes.
Cold start = download (empty cache) + weight load (storage bandwidth) + compile + CUDA graph capture + server up.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

NUM = r"([0-9][0-9,]*\.?[0-9]*)"
PATTERNS = {
    "download_s": re.compile(r"Time spent downloading weights for \S+: " + NUM + r" seconds"),
    "load_weights_s": re.compile(r"Loading weights took " + NUM + r" seconds"),
    "model_load": re.compile(r"Model loading took " + NUM + r" GiB memory and " + NUM + r" seconds"),
    "kv_mem_gib": re.compile(r"Available KV cache memory: " + NUM + r" GiB"),
    "kv_tokens": re.compile(r"KV cache size: " + NUM + r" tokens, Maximum concurrency for " + NUM +
                            r" tokens per request: " + NUM + r"x"),
    "graph": re.compile(r"Graph capturing finished in " + NUM + r" secs?, took " + NUM + r" GiB"),
    "init": re.compile(r"init engine \(profile, create kv cache, warmup model\) took " + NUM +
                       r" s(?: \(compilation: " + NUM + r" s)?"),
    "sg_load": re.compile(r"Load weight end\. elapsed=" + NUM + r" s,.*?mem usage=" + NUM + r" GB"),
    "sg_kv": re.compile(r"is allocated\. dtype: (\S+), #tokens: " + NUM +
                        r", (?:KV size: " + NUM + r" GB|K size: " + NUM + r" GB, V size: " + NUM + r" GB)"),
    "sg_graph": re.compile(r"Capture .*?CUDA graph end\. elapsed=" + NUM + r" s, mem usage=" + NUM + r" GB"),
}
READY = {
    "vllm": re.compile(r"Application startup complete"),
    "sglang": re.compile(r"The server is fired up and ready to roll"),
    "trtllm": re.compile(r"Application startup complete|Uvicorn running on"),
    "nim": re.compile(r"Application startup complete|Uvicorn running on"),
    "hf": re.compile(r"Application startup complete|Uvicorn running on"),
}
TS_DOCKER = re.compile(r"^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})(?:\.(\d+))?Z\s")      # docker logs -t
TS_SGLANG = re.compile(r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")                      # [2026-10-17 22:30:01 TP0]
TS_VLLM = re.compile(r"\b(?:INFO|WARNING|DEBUG|ERROR)\s+(\d{2}-\d{2} \d{2}:\d{2}:\d{2})")  # INFO 10-17 22:30:01


def _f(s: str | None) -> float | None:
    return float(s.replace(",", "")) if s else None


def line_ts(line: str, year: int = 2026) -> datetime | None:
    m = TS_DOCKER.match(line)
    if m:
        frac = (m.group(2) or "0")[:6].ljust(6, "0")
        return datetime.strptime(m.group(1) + "." + frac, "%Y-%m-%dT%H:%M:%S.%f")
    m = TS_SGLANG.search(line)
    if m:
        return datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")
    m = TS_VLLM.search(line)
    if m:
        return datetime.strptime(f"{year}-{m.group(1)}", "%Y-%m-%d %H:%M:%S")
    return None


def detect_engine(text: str) -> str:
    t = text
    if "Model loading took" in t or "vllm" in t.lower() and "init engine" in t:
        return "vllm"
    if "Load weight end" in t or "sglang" in t.lower():
        return "sglang"
    if "hf-naive" in t or "with transformers - no batching" in t:
        return "hf"
    if "tensorrt" in t.lower() or "trtllm" in t.lower():
        return "trtllm"
    if "nim" in t.lower():
        return "nim"
    return "vllm" if "vllm" in t.lower() else "unknown"


def parse_log(text: str, engine: str | None = None) -> dict:
    eng = engine or detect_engine(text)
    r: dict = {"engine": eng}
    first = ready = None
    graph_s = graph_gib = 0.0
    n_graph = 0
    for line in text.splitlines():
        ts = line_ts(line)
        if ts and first is None:
            first = ts
        if ready is None and READY.get(eng, READY["vllm"]).search(line):
            ready = ts
            r["ready_seen"] = True
        m = PATTERNS["download_s"].search(line)
        if m:
            r["download_s"] = r.get("download_s", 0.0) + _f(m.group(1))
        m = PATTERNS["load_weights_s"].search(line)
        if m:
            r["load_weights_s"] = _f(m.group(1))
        m = PATTERNS["model_load"].search(line)
        if m:
            r["weights_gib"], r["model_load_s"] = _f(m.group(1)), _f(m.group(2))
        m = PATTERNS["kv_mem_gib"].search(line)
        if m:
            r["kv_mem_gib"] = _f(m.group(1))
        m = PATTERNS["kv_tokens"].search(line)
        if m:
            r["kv_tokens"], r["max_model_len"], r["max_concurrency"] = int(_f(m.group(1))), int(_f(m.group(2))), \
                _f(m.group(3))
        m = PATTERNS["graph"].search(line)
        if m:
            graph_s += _f(m.group(1))
            graph_gib += _f(m.group(2))
            n_graph += 1
        m = PATTERNS["init"].search(line)
        if m:
            r["engine_init_s"] = _f(m.group(1))
            if m.group(2):
                r["compile_s"] = _f(m.group(2))
        m = PATTERNS["sg_load"].search(line)
        if m:
            r["load_weights_s"], r["weights_gib"] = _f(m.group(1)), _f(m.group(2))
        m = PATTERNS["sg_kv"].search(line)
        if m:
            r["kv_dtype"], r["kv_tokens"] = m.group(1), int(_f(m.group(2)))
            r["kv_mem_gib"] = _f(m.group(3)) if m.group(3) else (_f(m.group(4)) or 0) + (_f(m.group(5)) or 0)
        m = PATTERNS["sg_graph"].search(line)
        if m:
            graph_s += _f(m.group(1))
            graph_gib += _f(m.group(2))
            n_graph += 1
    if n_graph:
        r["graph_capture_s"], r["graph_gib"] = graph_s, graph_gib
    if first and ready:
        r["cold_start_s"] = (ready - first).total_seconds()
    load_s = r.get("load_weights_s") or r.get("model_load_s")
    if load_s and r.get("weights_gib"):
        r["load_gib_s"] = r["weights_gib"] / load_s if load_s > 0 else None
    return r


def suggestions(r: dict) -> list:
    out = []
    if r.get("download_s"):
        out.append(f"{r['download_s']:.0f} s downloading weights: pre-stage them (a warm HF cache volume, the NIM cache, "
                   "or a PVC / node-local cache in Kubernetes - M9). A cold download is a scale-out tax.")
    gbs = r.get("load_gib_s")
    if gbs is not None and gbs < 1.5:
        out.append(f"Weights load at {gbs:.2f} GiB/s: storage-bound. Copy weights to local NVMe, or stream them: vLLM "
                   "--load-format runai_streamer --model-loader-extra-config '{\"concurrency\":16}' (vllm[runai]).")
    if r.get("compile_s") and r["compile_s"] > 20:
        out.append(f"torch.compile took {r['compile_s']:.0f} s: keep ~/.cache/vllm on a volume (launch.sh mounts "
                   "VLLM_CACHE_HOST) and the next start of the same config reuses it.")
    if r.get("graph_capture_s") and r["graph_capture_s"] > 15:
        out.append(f"CUDA graph capture took {r['graph_capture_s']:.0f} s: fewer capture sizes start faster (vLLM "
                   "--compilation-config '{\"max_cudagraph_capture_size\": 64}') but small batches lose speed.")
    if r.get("cold_start_s") is None:
        out.append("No wall time: save the log with docker timestamps (engines/logs.sh --save uses docker logs -t), "
                   "and check the engine reached its ready line.")
    return out


def render(path: str, r: dict) -> str:
    def row(label, value, note=""):
        return f"  {label:<40} {value:>14}  {note}".rstrip()

    def s(x):
        return "-" if x is None else f"{x:,.1f} s"

    out = [f"{path}  ({r['engine']})", row("cold start: first log line -> ready", s(r.get("cold_start_s")))]
    if r.get("download_s") is not None:
        out.append(row("weights download", s(r.get("download_s")), "only on an empty cache"))
    load_s = r.get("load_weights_s") or r.get("model_load_s")
    note = ""
    if r.get("weights_gib"):
        note = f"{r['weights_gib']:.2f} GiB" + (f" -> {r['load_gib_s']:.2f} GiB/s" if r.get("load_gib_s") else "")
    if load_s is not None:
        out.append(row("weights load", s(load_s), note))
    if r.get("engine_init_s") is not None:
        out.append(row("engine init (profile, KV cache, warm-up)", s(r["engine_init_s"]),
                       f"torch.compile {r['compile_s']:.1f} s" if r.get("compile_s") else ""))
    if r.get("graph_capture_s") is not None:
        out.append(row("CUDA graph capture", s(r["graph_capture_s"]), f"{r.get('graph_gib', 0):.2f} GiB"))
    if r.get("kv_tokens"):
        kv = f"{r['kv_tokens']:,} tokens"
        extra = f"{r['kv_mem_gib']:.1f} GiB" if r.get("kv_mem_gib") else ""
        if r.get("max_concurrency"):
            extra += f", {r['max_concurrency']:.1f}x requests of {r['max_model_len']:,} tokens"
        out.append(row("KV cache", kv, extra))
    tips = suggestions(r)
    if tips:
        out.append("  next:")
        out.extend(f"    - {t}" for t in tips)
    return "\n".join(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("logs", nargs="+")
    ap.add_argument("--engine", choices=sorted(READY), help="override auto-detection")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    results = []
    for p in a.logs:
        r = parse_log(Path(p).read_text(errors="replace"), a.engine)
        r["log"] = p
        results.append(r)
    if a.json:
        print(json.dumps(results, indent=2))
    else:
        print("\n\n".join(render(r["log"], r) for r in results))
    return 0


if __name__ == "__main__":
    sys.exit(main())
