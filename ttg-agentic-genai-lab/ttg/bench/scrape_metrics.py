"""Read an engine's own metrics: prefix-cache hit rate, queue depth, token counters.

  python -m ttg.bench.scrape_metrics --url http://localhost:8000/metrics          # vLLM / SGLang / mock
  python -m ttg.bench.scrape_metrics --url http://localhost:8000/v1/metrics       # NIM
  python -m ttg.bench.scrape_metrics --url http://localhost:8000/metrics --raw    # every cache / queue line
Names differ per engine and version, so this matches by substring and prints what it found:
  vLLM 0.31  vllm:prefix_cache_hits_total / vllm:prefix_cache_queries_total (tokens), vllm:num_requests_waiting
  SGLang     sglang:cache_hit_rate (gauge, needs --enable-metrics), sglang:num_queue_reqs
  trtllm-serve  JSON iteration stats (needs enable_iter_perf_stats: true) - KV block reuse counters
"""
from __future__ import annotations

import argparse
import json
import re
import sys

import httpx

LINE = re.compile(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{[^}]*\})?\s+([-+0-9.eEinfNa]+)")


def fetch(url: str, timeout: float = 5.0) -> str:
    r = httpx.get(url, timeout=timeout)
    r.raise_for_status()
    return r.text


def parse_prometheus(text: str) -> dict:
    """Sum every series per metric name (labels dropped)."""
    out: dict = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        m = LINE.match(line.strip())
        if m:
            try:
                out[m.group(1)] = out.get(m.group(1), 0.0) + float(m.group(3))
            except ValueError:
                pass
    return out


def _find(parsed: dict, *needles) -> float | None:
    for name, val in parsed.items():
        if all(n in name for n in needles):
            return val
    return None


def _kv_records(obj, out):
    """Every dict that carries TensorRT-LLM KV-cache counters (reusedBlocks / missedBlocks), in document order."""
    if isinstance(obj, dict):
        if "reusedBlocks" in obj or "missedBlocks" in obj:
            out.append(obj)
        for v in obj.values():
            _kv_records(v, out)
    elif isinstance(obj, list):
        for x in obj:
            _kv_records(x, out)
    return out


def _trtllm(data) -> dict:
    """trtllm-serve /metrics returns the iteration stats queued since the last poll. reusedBlocks and missedBlocks
    are LIFETIME counters of the KV cache manager, so the newest record is the current total (summing records
    would double count). Two snapshots then give a window ratio exactly like vLLM's counters."""
    records = data if isinstance(data, list) else [data]
    records = sorted((r for r in records if isinstance(r, dict)), key=lambda r: r.get("iter", 0))
    kv = _kv_records(records[-1], []) if records else []
    if not kv:
        return {"engine": "trtllm", "hits": None, "queries": None, "hit_ratio": None, "waiting": None,
                "note": "no iteration stats since the last poll (idle engine, or enable_iter_perf_stats is off)"}
    k = kv[-1]
    reused, missed = float(k.get("reusedBlocks") or 0), float(k.get("missedBlocks") or 0)
    ratio = k.get("cacheHitRate")
    if ratio is None and reused + missed:
        ratio = reused / (reused + missed)
    return {"engine": "trtllm", "hits": reused, "queries": reused + missed, "hit_ratio": ratio, "waiting": None,
            "iter": records[-1].get("iter")}


def cache_stats(text: str) -> dict:
    """Returns {"engine", "hits", "queries", "hit_ratio", "waiting"} with None where the engine does not say."""
    s = text.lstrip()
    if s.startswith("{") or s.startswith("["):                          # trtllm-serve iteration stats (JSON)
        return _trtllm(json.loads(s))
    p = parse_prometheus(text)
    hits, queries = _find(p, "prefix_cache_hits"), _find(p, "prefix_cache_queries")
    if hits is not None and queries:
        return {"engine": "vllm", "hits": hits, "queries": queries, "hit_ratio": hits / queries,
                "waiting": _find(p, "num_requests_waiting")}
    rate = _find(p, "cache_hit_rate")
    if rate is not None:
        return {"engine": "sglang", "hits": None, "queries": None, "hit_ratio": rate,
                "waiting": _find(p, "num_queue_reqs")}
    return {"engine": "unknown", "hits": None, "queries": None, "hit_ratio": None,
            "waiting": _find(p, "waiting") or _find(p, "queue")}


def delta(before: dict, after: dict) -> dict:
    """Hit ratio over a window from two cumulative snapshots (vLLM and TRT-LLM counters). Gauges (SGLang), or a
    missing 'before': the 'after' value (for TRT-LLM that is the lifetime ratio - relaunch for a clean window)."""
    if before.get("hits") is not None and after.get("hits") is not None and after.get("queries") \
            and after["queries"] >= (before.get("queries") or 0):
        dq = after["queries"] - before["queries"]
        dh = after["hits"] - before["hits"]
        return {"engine": after["engine"], "hits": dh, "queries": dq, "hit_ratio": (dh / dq) if dq else None}
    return {"engine": after.get("engine"), "hit_ratio": after.get("hit_ratio")}


def snapshot(url: str | None) -> dict | None:
    if not url:
        return None
    try:
        return cache_stats(fetch(url))
    except Exception as e:                       # metrics are optional; never fail a benchmark on them
        return {"engine": "unreachable", "error": str(e)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://localhost:8000/metrics")
    ap.add_argument("--raw", action="store_true", help="print every cache/queue/token line")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    text = fetch(a.url)
    stats = cache_stats(text)
    if a.raw:
        for line in text.splitlines():
            if re.search(r"cache|queue|waiting|running|tokens", line, re.I) and not line.startswith("#"):
                print(line)
    if a.json:
        print(json.dumps(stats, indent=1, default=str))
    else:
        hr = stats.get("hit_ratio")
        print(f"engine={stats['engine']} prefix-cache hit ratio={'n/a' if hr is None else f'{hr:.1%}'} "
              f"waiting={stats.get('waiting')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
