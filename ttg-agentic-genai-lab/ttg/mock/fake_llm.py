"""Mock engine: an OpenAI-compatible server that behaves like a small, well-trained agent model on a GPU.

  uvicorn ttg.mock.fake_llm:app --port 8000                      # then TTG_BASE_URL=http://localhost:8000/v1
  MOCK_PREFIX_CACHE=0 uvicorn ttg.mock.fake_llm:app --port 8000  # like vllm --no-enable-prefix-caching
  MOCK_FAULTS=invalid_json_once,wrong_amount ...                 # drills and negative tests
  TTG_MODEL=ttg-llm+invalid_json_once+slow                       # same faults per client, via the model name

What it emulates (all tunable with MOCK_* env vars, all approximate - it is a teaching aid, not a GPU model):
- the right next action for Tern Care, computed from the prompt by the same policy oracle the scorer uses;
- prefix caching on 16-token blocks: only uncached prompt tokens pay prefill time, and usage reports
  prompt_tokens_details.cached_tokens like vLLM with --enable-prompt-tokens-details;
- TTFT = base + prefill time; streaming at an inter-token latency that grows with the running batch; a fixed
  number of batch slots (requests beyond it queue, which is where TTFT explodes at high concurrency);
- verbose thoughts when the system prompt asks for step-by-step reasoning (drill 10: tokens per task explode);
- vLLM-style Prometheus counters at /metrics (vllm:prefix_cache_hits_total, vllm:prefix_cache_queries_total ...).
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import time
import uuid
from collections import OrderedDict

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse

from ttg import world
from ttg.agent.protocol import ProtocolError, format_action, format_final, parse_step

BLOCK_TOKENS = 16
CHARS_PER_TOKEN = 4
MODELS = ["ttg-llm", "ttg-short", "meta-llama/Llama-3.1-8B-Instruct", "meta/llama-3.1-8b-instruct", "mock-wrong"]


def _f(name, default):
    return float(os.environ.get(name, default))


class Engine:
    def __init__(self):
        self.reset()

    def reset(self):
        self.cache: OrderedDict = OrderedDict()
        self.capacity = int(_f("MOCK_CACHE_BLOCKS", 200_000))
        self.slots = asyncio.Semaphore(int(_f("MOCK_MAX_BATCH", 48)))
        self.running = 0
        self.waiting = 0
        self.c = {"prefix_cache_queries": 0, "prefix_cache_hits": 0, "prompt_tokens": 0, "generation_tokens": 0,
                  "requests": 0}

    def prefix_lookup(self, text: str) -> tuple:
        """Returns (prompt_tokens, cached_tokens) and inserts the prompt's blocks (vLLM-style hash chain)."""
        tokens = max(1, math.ceil(len(text) / CHARS_PER_TOKEN))
        step = BLOCK_TOKENS * CHARS_PER_TOKEN
        h, hit_blocks, still_hitting = "", 0, os.environ.get("MOCK_PREFIX_CACHE", "1") == "1"
        for i in range(0, len(text) - step + 1, step):
            h = hashlib.sha1((h + text[i:i + step]).encode()).hexdigest()
            if still_hitting and h in self.cache:
                hit_blocks += 1
                self.cache.move_to_end(h)
            else:
                still_hitting = False
                self.cache[h] = True
                if len(self.cache) > self.capacity:
                    self.cache.popitem(last=False)
        cached = min(tokens, hit_blocks * BLOCK_TOKENS) if os.environ.get("MOCK_PREFIX_CACHE", "1") == "1" else 0
        self.c["prefix_cache_queries"] += tokens
        self.c["prefix_cache_hits"] += cached
        self.c["prompt_tokens"] += tokens
        return tokens, cached


ENGINE = Engine()
app = FastAPI(title="ttg-mock-engine")

THOUGHTS = {"flight_status": "Check the flight first.", "loyalty_profile": "Check the loyalty tier.",
            "policy_search": "Confirm the policy for this case.", "rebooking_options": "Cancelled: find seats.",
            "book_flight": "Book the earliest option with enough seats.",
            "hotel_availability": "Overnight: check hotel rooms at the origin.",
            "book_hotel": "Book the cheapest hotel with enough rooms.", "read_receipt": "Read the uploaded receipt.",
            "issue_voucher": "Issue the voucher for the whole party."}
VERBOSE = ("Let me think step by step. First I restate the Tern Care Commitments: controllable causes are crew and "
           "mechanical, rebooking goes to the earliest option with enough seats, meals depend on long disruption, "
           "hotels depend on overnight stays, credits depend on cause and status, receipts are capped per traveler, "
           "and vouchers above three hundred dollars need approval. Next I list every fact I know from the "
           "observations so far, check each rule against those facts, consider the alternatives, and only then "
           "choose the single next action that moves the case forward without breaking any rule. ") * 2


def _render(messages: list) -> str:
    parts = []
    for m in messages:
        c = m.get("content")
        if isinstance(c, list):
            c = " ".join(p.get("text", "") for p in c if isinstance(p, dict) and p.get("type") == "text")
        parts.append(f"<|{m.get('role')}|>\n{c or ''}")
    return "\n".join(parts) + "\n<|assistant|>\n"


def _has_image(messages: list) -> bool:
    return any(isinstance(m.get("content"), list) and any(isinstance(p, dict) and p.get("type") == "image_url"
                                                         for p in m["content"]) for m in messages)


def agent_reply(messages: list, model: str, faults: set) -> str:
    """The oracle's next step for this conversation (plus any injected fault)."""
    first_user = next((m.get("content") for m in messages if m.get("role") == "user"), "") or ""
    if not isinstance(first_user, str) or not first_user.startswith("Disrupted traveler: "):
        return ""
    public = json.loads(first_user[len("Disrupted traveler: "):])
    assistant = [m.get("content") or "" for m in messages if m.get("role") == "assistant"]
    done = 0
    for text in assistant:
        try:
            if parse_step(text).kind == "action":
                done += 1
        except ProtocolError:
            pass
    system = next((m.get("content") for m in messages if m.get("role") == "system"), "") or ""
    verbose = "verbose" in faults or "think step by step in great detail" in system
    if "invalid_json_once" in faults and not assistant:
        return "Sure! Let me check the flight status for you first."
    plan = world.expected_plan(public)
    if "no_final" in faults and done >= len(plan) - 1:
        return format_action("policy_search", {"query": "anything else"}, "Double-check the policy.")
    step = plan[min(done, len(plan) - 1)]
    if "final" in step:
        final = dict(step["final"])
        if "wrong_amount" in faults or "wrong" in model:
            final["vouchers"] = [{**v, "amount_usd": v["amount_usd"] * 2} for v in final["vouchers"]]
        return format_final(final, (VERBOSE if verbose else "") + "All actions done; write the guest message.")
    args = dict(step["args"])
    if step["action"] == "issue_voucher" and ("wrong_amount" in faults or "wrong" in model):
        args["amount_usd"] = args["amount_usd"] * 2                         # over-compensation: a policy violation
    thought = (VERBOSE if verbose else "") + THOUGHTS.get(step["action"], "Next step.")
    return format_action(step["action"], args, thought)


def filler(n_tokens: int) -> str:
    words = "storm rebook hotel voucher policy flight guest crew seats overnight".split()
    return " ".join(words[i % len(words)] for i in range(max(1, n_tokens)))


@app.get("/v1/models")
async def models():
    return {"object": "list", "data": [{"id": m, "object": "model", "owned_by": "ttg-mock"} for m in MODELS]}


@app.get("/health")
@app.get("/v1/health/ready")
@app.get("/v1/health/live")
async def health():
    return {"status": "ok"}


@app.post("/reset")
async def reset():
    ENGINE.reset()
    return {"status": "reset"}


@app.get("/metrics")
@app.get("/v1/metrics")
async def metrics():
    c, label = ENGINE.c, 'model_name="ttg-llm"'
    lines = []
    for name, kind, val in [("prefix_cache_queries_total", "counter", c["prefix_cache_queries"]),
                            ("prefix_cache_hits_total", "counter", c["prefix_cache_hits"]),
                            ("prompt_tokens_total", "counter", c["prompt_tokens"]),
                            ("generation_tokens_total", "counter", c["generation_tokens"]),
                            ("request_success_total", "counter", c["requests"]),
                            ("num_requests_running", "gauge", ENGINE.running),
                            ("num_requests_waiting", "gauge", ENGINE.waiting)]:
        lines += [f"# TYPE vllm:{name} {kind}", f"vllm:{name}{{{label}}} {val}"]
    return PlainTextResponse("\n".join(lines) + "\n")


@app.post("/v1/chat/completions")
async def chat(request: Request):
    body = await request.json()
    model = body.get("model", "ttg-llm")
    messages = body.get("messages") or []
    faults = {f.strip() for f in (os.environ.get("MOCK_FAULTS", "") + "," +
                                  request.headers.get("x-mock-faults", "")).split(",") if f.strip()}
    faults |= {f for f in model.split("+")[1:] if f}            # e.g. TTG_MODEL=ttg-llm+invalid_json_once
    speed = _f("MOCK_SPEED", 1.0) * (5.0 if "slow" in faults else 1.0)
    max_tokens = int(body.get("max_tokens") or body.get("max_completion_tokens") or 256)
    if _has_image(messages):
        text = json.dumps({"merchant": "unknown", "date": world.OPS_DATE, "category": "meal", "total_usd": 0.0,
                           "items": [], "note": "the mock engine cannot see images; use a real VLM for M8"})
    else:
        text = agent_reply(messages, model, faults)
        if not text:                                   # AIPerf / smoke tests: synthetic text of the requested size
            want = max_tokens if body.get("ignore_eos") or body.get("min_tokens") else min(max_tokens, 48)
            text = filler(want)
    out_tokens = max(1, math.ceil(len(text) / CHARS_PER_TOKEN))
    if out_tokens > max_tokens:                         # the engine would stop at max_tokens: truncated JSON
        text, out_tokens = text[: max_tokens * CHARS_PER_TOKEN], max_tokens
        finish = "length"
    else:
        finish = "stop"
    prompt_text = _render(messages)
    ENGINE.waiting += 1
    await ENGINE.slots.acquire()
    ENGINE.waiting -= 1
    ENGINE.running += 1
    prompt_tokens, cached = ENGINE.prefix_lookup(prompt_text)
    ttft = (_f("MOCK_TTFT_BASE_S", 0.02) + (prompt_tokens - cached) / 1000 * _f("MOCK_PREFILL_S_PER_1K", 0.035)) * speed
    itl = _f("MOCK_ITL_S", 0.004) * (1 + _f("MOCK_ITL_GROWTH", 0.03) * ENGINE.running) * speed
    usage = {"prompt_tokens": prompt_tokens, "completion_tokens": out_tokens, "total_tokens": prompt_tokens + out_tokens,
             "prompt_tokens_details": {"cached_tokens": cached}}
    rid, created = f"chatcmpl-{uuid.uuid4().hex[:12]}", int(time.time())

    def release():
        ENGINE.running -= 1
        ENGINE.slots.release()
        ENGINE.c["generation_tokens"] += out_tokens
        ENGINE.c["requests"] += 1

    if not body.get("stream"):
        try:
            await asyncio.sleep(ttft + itl * out_tokens)
        finally:
            release()
        return JSONResponse({"id": rid, "object": "chat.completion", "created": created, "model": model,
                             "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                                          "finish_reason": finish}], "usage": usage})
    include_usage = bool((body.get("stream_options") or {}).get("include_usage"))

    async def gen():
        try:
            def chunk(delta, fin=None):
                return "data: " + json.dumps({"id": rid, "object": "chat.completion.chunk", "created": created,
                                              "model": model, "choices": [{"index": 0, "delta": delta,
                                                                           "finish_reason": fin}]}) + "\n\n"
            await asyncio.sleep(ttft)
            yield chunk({"role": "assistant", "content": ""})
            step = 4 * CHARS_PER_TOKEN                     # ~4 tokens per streamed chunk
            for i in range(0, len(text), step):
                yield chunk({"content": text[i:i + step]})
                await asyncio.sleep(itl * 4)
            yield chunk({}, finish)
            if include_usage:
                yield "data: " + json.dumps({"id": rid, "object": "chat.completion.chunk", "created": created,
                                             "model": model, "choices": [], "usage": usage}) + "\n\n"
            yield "data: [DONE]\n\n"
        finally:
            release()

    return StreamingResponse(gen(), media_type="text/event-stream")
