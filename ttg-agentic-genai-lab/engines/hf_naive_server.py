#!/usr/bin/env python3
"""The "before" baseline: a naive OpenAI-compatible server around Hugging Face transformers `model.generate`.

This is what many customers run first - and what the accelerated stacks are measured against in the M10 ledger:
  - ONE request at a time (an asyncio.Lock): no continuous batching, so 16 concurrent agents queue behind each other
  - a fresh KV cache per request (no PagedAttention, no prefix caching): the ~2K-token system prompt is recomputed
    on every one of the ~7 LLM calls of every task
  - no CUDA graphs, no fused kernels beyond what PyTorch picks, no FP8, no speculative decoding

Usage (engines/launch.sh hf base runs it inside the vLLM image, which already has torch + transformers + fastapi):
  python3 engines/hf_naive_server.py --model meta-llama/Llama-3.1-8B-Instruct --served-model-name ttg-llm --port 8000
Endpoints: GET /health, GET /v1/models, POST /v1/chat/completions (stream or not; stream_options.include_usage),
POST /reset_prefix_cache (a no-op, so the bench's --reset-url works on every engine).
response_format / tools are accepted and ignored (the Lab 05 agent speaks plain JSON anyway).
torch and transformers are imported lazily, so `--help` and the unit tests run without them.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import time
import uuid
from typing import Iterator, Protocol

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse


class Backend(Protocol):
    """What the server needs from a model. HFBackend is the real one; tests inject a fake."""

    last_completion_tokens: int

    def prompt_tokens(self, messages: list) -> int: ...

    def stream(self, messages: list, max_tokens: int, temperature: float) -> Iterator[str]: ...


class HFBackend:
    """transformers generate() with TextIteratorStreamer - deliberately nothing clever."""

    def __init__(self, model_id: str, dtype: str = "bfloat16", device: str = "cuda"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._torch = torch
        self.device = device
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, dtype=getattr(torch, dtype)).to(device).eval()
        self.last_completion_tokens = 0

    def _encode(self, messages):
        return self.tok.apply_chat_template(messages, add_generation_prompt=True, return_dict=True,
                                            return_tensors="pt").to(self.device)

    def prompt_tokens(self, messages):
        return int(self._encode(messages)["input_ids"].shape[-1])

    def stream(self, messages, max_tokens, temperature):
        import threading

        from transformers import TextIteratorStreamer

        inputs = self._encode(messages)
        n_prompt = int(inputs["input_ids"].shape[-1])
        # timeout: a stuck generate() raises queue.Empty here instead of holding the request lock forever
        streamer = TextIteratorStreamer(self.tok, skip_prompt=True, skip_special_tokens=True, timeout=600)
        kwargs = dict(**inputs, max_new_tokens=int(max_tokens), streamer=streamer,
                      do_sample=temperature > 0, pad_token_id=self.tok.eos_token_id)
        if temperature > 0:
            kwargs["temperature"] = float(temperature)
        result = {}

        def run():
            try:
                with self._torch.inference_mode():
                    out = self.model.generate(**kwargs)
                result["n"] = int(out.shape[-1]) - n_prompt
            except Exception as e:                  # OOM, bad input: unblock the reader, then report the error
                result["error"] = e
                streamer.end()

        t = threading.Thread(target=run, daemon=True)
        t.start()
        for piece in streamer:
            if piece:
                yield piece
        t.join()
        if "error" in result:
            raise RuntimeError(f"generate() failed: {result['error']}")
        self.last_completion_tokens = result.get("n", 0)


app = FastAPI(title="hf-naive-baseline")
app.state.backend = None
app.state.served = "ttg-llm"
app.state.lock = None
app.state.requests = 0


def _lock() -> asyncio.Lock:
    if app.state.lock is None:
        app.state.lock = asyncio.Lock()
    return app.state.lock


def _usage(prompt: int, completion: int) -> dict:
    return {"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": prompt + completion,
            "prompt_tokens_details": {"cached_tokens": 0}}       # nothing is ever cached here - that is the point


def _chunk(cid: str, model: str, delta: dict, finish=None) -> str:
    return "data: " + json.dumps({"id": cid, "object": "chat.completion.chunk", "created": int(time.time()),
                                  "model": model, "choices": [{"index": 0, "delta": delta,
                                                               "finish_reason": finish}]}) + "\n\n"


@app.get("/health")
async def health():
    if app.state.backend is None:
        return JSONResponse({"status": "loading"}, status_code=503)
    return {"status": "ok"}


@app.get("/v1/models")
async def models():
    return {"object": "list", "data": [{"id": app.state.served, "object": "model", "owned_by": "hf-naive"}]}


@app.post("/reset_prefix_cache")
async def reset_prefix_cache():
    """Same path as vLLM's dev endpoint so one bench command (--reset-url) works everywhere. Nothing to reset here."""
    return {"status": "ok", "cached_blocks": 0}


@app.post("/v1/chat/completions")
async def chat(request: Request):
    backend: Backend | None = app.state.backend
    if backend is None:
        return JSONResponse({"error": {"message": "model is still loading"}}, status_code=503)
    body = await request.json()
    messages = body.get("messages") or []
    max_tokens = int(body.get("max_tokens") or body.get("max_completion_tokens") or 256)
    temperature = float(body.get("temperature") or 0.0)
    model = body.get("model") or app.state.served
    cid = "chatcmpl-" + uuid.uuid4().hex[:24]
    include_usage = bool((body.get("stream_options") or {}).get("include_usage"))
    app.state.requests += 1

    if not body.get("stream"):
        async with _lock():
            n_prompt = await asyncio.to_thread(backend.prompt_tokens, messages)
            text = "".join(await asyncio.to_thread(lambda: list(backend.stream(messages, max_tokens, temperature))))
            n_out = backend.last_completion_tokens
        finish = "length" if n_out >= max_tokens else "stop"
        return {"id": cid, "object": "chat.completion", "created": int(time.time()), "model": model,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": finish}],
                "usage": _usage(n_prompt, n_out)}

    async def events():
        async with _lock():                        # the whole stream holds the lock: requests run one at a time
            n_prompt = await asyncio.to_thread(backend.prompt_tokens, messages)
            yield _chunk(cid, model, {"role": "assistant", "content": ""})
            it = backend.stream(messages, max_tokens, temperature)
            done = object()
            while True:
                piece = await asyncio.to_thread(next, it, done)
                if piece is done:
                    break
                yield _chunk(cid, model, {"content": piece})
            n_out = backend.last_completion_tokens
        yield _chunk(cid, model, {}, finish="length" if n_out >= max_tokens else "stop")
        if include_usage:
            yield "data: " + json.dumps({"id": cid, "object": "chat.completion.chunk", "created": int(time.time()),
                                         "model": model, "choices": [], "usage": _usage(n_prompt, n_out)}) + "\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(events(), media_type="text/event-stream")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="Hugging Face id or local path")
    ap.add_argument("--served-model-name", default="ttg-llm")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args(argv)
    import uvicorn

    app.state.served = a.served_model_name
    print(f"Loading {a.model} ({a.dtype}) with transformers - no batching, no KV paging, no prefix cache ...")
    app.state.backend = HFBackend(a.model, a.dtype, a.device)
    uvicorn.run(app, host=a.host, port=a.port, log_level="info")


if __name__ == "__main__":
    main()
