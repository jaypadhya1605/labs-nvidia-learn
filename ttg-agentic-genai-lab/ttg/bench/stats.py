"""Small, dependency-free statistics shared by the bench, the ledger and the gates."""
from __future__ import annotations

import math
from collections import Counter


def pct(values, q: float):
    """Percentile with linear interpolation (q in 0..1). None for an empty list."""
    xs = sorted(v for v in values if v is not None)
    if not xs:
        return None
    if len(xs) == 1:
        return xs[0]
    pos = q * (len(xs) - 1)
    lo, hi = math.floor(pos), math.ceil(pos)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def mean(values):
    xs = [v for v in values if v is not None]
    return sum(xs) / len(xs) if xs else None


def usd_per_1k(tasks_per_s, gpu_hourly_usd, gpus) -> float | None:
    if not tasks_per_s or not gpu_hourly_usd:
        return None
    return gpu_hourly_usd * gpus / (tasks_per_s * 3600.0) * 1000.0


def summarize_level(results: list, concurrency: int, wall_s: float, gpu_hourly_usd: float = 0.0, gpus: int = 1,
                    engine_cache: dict | None = None) -> dict:
    n = len(results)
    tps = n / wall_s if wall_s > 0 else 0.0
    # Only tasks whose engine REPORTED cached tokens count: "not reported" (None) is unknown, not a 0% hit rate.
    reported = [r for r in results if r.get("cached_tokens") is not None and r.get("prompt_tokens")]
    prompt = sum(r["prompt_tokens"] for r in reported)
    cached = sum(r["cached_tokens"] for r in reported)
    viol = Counter(v.split(":")[0] for r in results for v in r["violations"])
    out = {
        "concurrency": concurrency, "n": n, "wall_s": round(wall_s, 3), "tasks_per_s": round(tps, 4),
        "success_rate": round(sum(r["success"] for r in results) / n, 4) if n else 0.0,
        "p50_latency_s": _r(pct([r["latency_s"] for r in results], 0.50)),
        "p95_latency_s": _r(pct([r["latency_s"] for r in results], 0.95)),
        "p50_first_ttft_s": _r(pct([r["first_ttft_s"] for r in results], 0.50)),
        "p95_first_ttft_s": _r(pct([r["first_ttft_s"] for r in results], 0.95)),
        "mean_llm_calls": _r(mean([r["llm_calls"] for r in results])),
        "mean_prompt_tokens": _r(mean([r["prompt_tokens"] for r in results])),
        "mean_cached_tokens": _r(mean([r["cached_tokens"] for r in results])),
        "mean_completion_tokens": _r(mean([r["completion_tokens"] for r in results])),
        "cache_hit_ratio": round(cached / prompt, 4) if prompt else None,
        "retries_per_task": _r(mean([r["retries"] for r in results])),
        "errors": sum(1 for r in results if r.get("error")),
        "usd_per_1k_tasks": _r(usd_per_1k(tps, gpu_hourly_usd, gpus)),
        "tasks_per_gpu_hour": round(tps * 3600 / max(gpus, 1), 1),
        "top_violations": dict(viol.most_common(5)),
    }
    if engine_cache:
        out["engine_prefix_hit_ratio"] = engine_cache.get("hit_ratio")
    return out


def _r(x, nd=4):
    return None if x is None else round(x, nd)
