"""Plover Assist API - the thin product layer in front of NVIDIA NIM.

  POST /v1/prior-auth/summarize  domain endpoint: variant routing (stable/canary LoRA), prompt by style mode,
                                 schema + faithfulness validation, Prometheus metrics
  POST /v1/chat/completions      OpenAI-compatible passthrough (streaming supported) - for AIPerf and the workshop
  GET  /v1/models, /healthz, /readyz, /metrics
Configuration is environment-only so the same image runs unchanged on AKS, GKE, EKS, OKE and RKE2.
"""
import json
import os
import time
import uuid

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, Counter, Histogram, generate_latest
from pydantic import BaseModel

from . import prompts, routing, schema

LLM_URL = os.environ.get("LLM_URL", "http://plover-llm:8000").rstrip("/")
BASE_MODEL = os.environ.get("BASE_MODEL", "meta/llama-3.1-8b-instruct")
STABLE_MODEL = os.environ.get("STABLE_MODEL", "") or BASE_MODEL
CANARY_MODEL = os.environ.get("CANARY_MODEL", "")
CANARY_WEIGHT = int(os.environ.get("CANARY_WEIGHT", "0") or 0)
TIMEOUT = float(os.environ.get("REQUEST_TIMEOUT_S", "120"))
# Model-specific request knobs, e.g. {"chat_template_kwargs": {"enable_thinking": false}} for Nemotron 3.5 (add-on)
LLM_EXTRA_BODY = json.loads(os.environ.get("LLM_EXTRA_BODY", "") or "{}")

REGISTRY = CollectorRegistry()   # module-level registry: one per process, re-created on reload (tests)
REQS = Counter("plover_api_requests_total", "Requests", ["route", "variant", "outcome"], registry=REGISTRY)
LAT = Histogram("plover_api_latency_seconds", "End-to-end latency", ["route"],
                buckets=(0.25, 0.5, 1, 2, 4, 8, 16, 32, 64), registry=REGISTRY)
VALID = Counter("plover_api_schema_valid_total", "Summaries by schema validity", ["variant", "valid"], registry=REGISTRY)
PTOK = Counter("plover_api_prompt_tokens_total", "Prompt tokens sent to the LLM", ["variant"], registry=REGISTRY)

app = FastAPI(title="Plover Assist API", version="1.0.0")


def http_client() -> httpx.AsyncClient:
    """Indirection so tests can inject a mock transport."""
    return httpx.AsyncClient(timeout=TIMEOUT)


def mode_for(model: str) -> str:
    return "long-prompt" if model == BASE_MODEL else "adapter"


class SummarizeRequest(BaseModel):
    case_text: str
    request_id: str | None = None
    max_tokens: int = 400


@app.post("/v1/prior-auth/summarize")
async def summarize(req: SummarizeRequest):
    t0 = time.perf_counter()
    rid = req.request_id or str(uuid.uuid4())
    variant = routing.choose_variant(rid, CANARY_WEIGHT if CANARY_MODEL else 0)
    model = CANARY_MODEL if variant == "canary" else STABLE_MODEL
    body = {"model": model, "messages": prompts.build_messages(req.case_text, mode_for(model)),
            "temperature": 0.0, "max_tokens": req.max_tokens, **LLM_EXTRA_BODY}
    try:
        async with http_client() as client:
            r = await client.post(f"{LLM_URL}/v1/chat/completions", json=body)
        r.raise_for_status()
        out = r.json()
    except httpx.HTTPError as exc:
        REQS.labels("summarize", variant, "upstream_error").inc()
        return JSONResponse({"error": f"LLM call failed: {exc}", "variant": variant}, status_code=502)
    text = out["choices"][0]["message"].get("content") or ""
    obj = schema.extract_json(text)
    errors = schema.validate(obj)
    hallucinated = schema.unfaithful_codes(obj, req.case_text)
    valid = not errors and not hallucinated
    usage = out.get("usage") or {}
    PTOK.labels(variant).inc(usage.get("prompt_tokens", 0) or 0)
    VALID.labels(variant, str(valid).lower()).inc()
    REQS.labels("summarize", variant, "ok" if valid else "invalid").inc()
    LAT.labels("summarize").observe(time.perf_counter() - t0)
    return {"request_id": rid, "variant": variant, "model": model, "schema_valid": valid,
            "errors": errors + [f"code not in request: {c}" for c in hallucinated],
            "summary": obj, "usage": usage}


@app.post("/v1/chat/completions")
async def chat(request: Request):
    body = await request.json()
    body.setdefault("model", BASE_MODEL)
    t0 = time.perf_counter()
    if body.get("stream"):
        async def relay():
            async with http_client() as client:
                async with client.stream("POST", f"{LLM_URL}/v1/chat/completions", json=body) as r:
                    async for chunk in r.aiter_raw():
                        yield chunk
            REQS.labels("chat", "passthrough", "ok").inc()
            LAT.labels("chat").observe(time.perf_counter() - t0)
        return StreamingResponse(relay(), media_type="text/event-stream")
    async with http_client() as client:
        r = await client.post(f"{LLM_URL}/v1/chat/completions", json=body)
    REQS.labels("chat", "passthrough", "ok" if r.status_code < 400 else "error").inc()
    LAT.labels("chat").observe(time.perf_counter() - t0)
    return JSONResponse(r.json(), status_code=r.status_code)


@app.get("/v1/models")
async def models():
    async with http_client() as client:
        r = await client.get(f"{LLM_URL}/v1/models")
    return JSONResponse(r.json(), status_code=r.status_code)


@app.get("/healthz")
async def healthz():
    return {"ok": True, "stable": STABLE_MODEL, "canary": CANARY_MODEL or None, "canary_weight": CANARY_WEIGHT}


@app.get("/readyz")
async def readyz():
    try:
        async with http_client() as client:
            r = await client.get(f"{LLM_URL}/v1/health/ready")
        ok = r.status_code == 200
    except httpx.HTTPError:
        ok = False
    return JSONResponse({"llm_ready": ok}, status_code=200 if ok else 503)


@app.get("/metrics")
async def metrics():
    return PlainTextResponse(generate_latest(REGISTRY), media_type=CONTENT_TYPE_LATEST)
