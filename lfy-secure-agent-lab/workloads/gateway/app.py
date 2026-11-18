#!/usr/bin/env python3
"""Lanternfly guarded gateway: OpenAI-compatible /v1/chat/completions where EVERY agent LLM call runs through NeMo
Guardrails (jailbreak detection + content safety on input, content safety + PHI masking on output).

Why not `nemoguardrails server`? We want OpenAI-format responses (so AIPerf, garak, and NeMo Agent Toolkit call it
unchanged), per-rail Prometheus metrics, and an explicit FAIL_MODE switch (closed = refuse, open = bypass) to drill on.
"""
import json
import logging
import os
import time
import uuid

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from nemoguardrails import LLMRails, RailsConfig
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

CONFIG_DIR = os.environ.get("GUARDRAILS_CONFIG", "/work/config")
FAIL_MODE = os.environ.get("FAIL_MODE", "closed").lower()
MAIN_LLM_URL = os.environ.get("MAIN_LLM_URL", "http://triage-llm.lfy-agent.svc:8000/v1/chat/completions")
MODEL_NAME = os.environ.get("SERVED_MODEL_NAME", "meta/llama-3.1-8b-instruct")
REFUSAL = "BLOCKED_BY_POLICY: this request or response violated Lanternfly safety policy."

REQUESTS = Counter("guardrail_requests_total", "Guarded requests by outcome", ["outcome"])
STOPS = Counter("guardrail_rail_stops_total", "Rails that stopped a request", ["rail"])
ACTIVATIONS = Counter("guardrail_rail_activations_total", "Rails executed", ["rail"])
ERRORS = Counter("guardrail_errors_total", "Errors in rails or the main LLM", ["fail_mode"])
LATENCY = Histogram("guardrail_latency_seconds", "End-to-end guarded latency", buckets=(0.1, 0.25, 0.5, 1, 2, 4, 8, 16, 32, 64))
RAIL_LATENCY = Histogram("guardrail_rail_latency_seconds", "Latency per rail", ["rail"],
                         buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 4, 8))

logging.basicConfig(level=logging.INFO)
app = FastAPI(title="Lanternfly guarded gateway")
rails = LLMRails(RailsConfig.from_path(CONFIG_DIR))


def completion(content, guard):
    return {"id": f"chatcmpl-{uuid.uuid4().hex[:24]}", "object": "chat.completion", "created": int(time.time()), "model": MODEL_NAME,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}, "guardrails": guard}


async def direct_llm(messages, params):
    async with httpx.AsyncClient(timeout=120) as client:
        r = await client.post(MAIN_LLM_URL, json={"model": MODEL_NAME, "messages": messages, **params})
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]


@app.post("/v1/chat/completions")
async def chat(request: Request):
    body = await request.json()
    messages = [{"role": m["role"], "content": m.get("content") or ""} for m in body.get("messages", [])
                if m.get("role") in ("system", "user", "assistant")]
    params = {k: body[k] for k in ("temperature", "top_p", "max_tokens") if k in body}
    if "max_completion_tokens" in body and "max_tokens" not in params:
        params["max_tokens"] = body["max_completion_tokens"]
    guard = {"blocked": False, "stopped_by": None, "rails": [], "fail_mode": FAIL_MODE}
    t0 = time.perf_counter()
    try:
        res = await rails.generate_async(messages=messages, options={"log": {"activated_rails": True}, "llm_params": params})
        resp = res.response
        content = resp[-1]["content"] if isinstance(resp, list) and resp else (resp if isinstance(resp, str) else str(resp))
        for rail in (res.log.activated_rails if res.log else []):
            ACTIVATIONS.labels(rail.name).inc()
            guard["rails"].append(rail.name)
            if getattr(rail, "duration", None) is not None:
                RAIL_LATENCY.labels(rail.name).observe(rail.duration)
            if getattr(rail, "stop", False):
                guard["blocked"], guard["stopped_by"] = True, rail.name
                STOPS.labels(rail.name).inc()
        if guard["blocked"]:
            content = REFUSAL
        REQUESTS.labels("blocked" if guard["blocked"] else "allowed").inc()
    except Exception as exc:  # safety model down, timeout, parse error ...
        ERRORS.labels(FAIL_MODE).inc()
        logging.exception("guardrails error")
        if FAIL_MODE == "open":  # DANGEROUS by design (drill 6): availability over safety, no checks at all
            guard.update({"bypassed": True, "error": str(exc)[:200]})
            content = await direct_llm(messages, params)
            REQUESTS.labels("bypassed").inc()
        else:
            guard.update({"blocked": True, "stopped_by": "error", "error": str(exc)[:200]})
            content = REFUSAL
            REQUESTS.labels("error").inc()
    LATENCY.observe(time.perf_counter() - t0)
    out = completion(content, guard)
    if body.get("stream"):
        async def sse():  # honest streaming: the answer only exists after the output rails ran, so TTFT = full latency
            chunk = {"id": out["id"], "object": "chat.completion.chunk", "created": out["created"], "model": MODEL_NAME,
                     "choices": [{"index": 0, "delta": {"role": "assistant", "content": content}, "finish_reason": "stop"}]}
            yield f"data: {json.dumps(chunk)}\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(sse(), media_type="text/event-stream")
    return JSONResponse(out)


@app.get("/v1/models")
async def models():
    return {"object": "list", "data": [{"id": MODEL_NAME, "object": "model", "owned_by": "lanternfly-guarded"}]}


@app.get("/healthz")
async def healthz():
    return {"ok": True, "fail_mode": FAIL_MODE}


@app.get("/metrics")
async def metrics():
    return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)
