"""OpenAI-compatible chat client that measures what an SA needs: TTFT, end-to-end latency, prompt / completion /
cached tokens per call. Works against vLLM, SGLang, trtllm-serve, NIM, build.nvidia.com and the lab's mock engine."""
from __future__ import annotations

import asyncio
import time
from dataclasses import asdict, dataclass

import openai
from openai import AsyncOpenAI

from ttg.agent.config import Settings
from ttg.agent.protocol import response_format


@dataclass
class LLMResult:
    text: str
    ttft_s: float
    latency_s: float
    prompt_tokens: int | None
    completion_tokens: int | None
    cached_tokens: int | None
    finish_reason: str | None

    def as_dict(self) -> dict:
        d = asdict(self)
        d.pop("text")
        return d


class LLMClient:
    def __init__(self, settings: Settings, http_client=None):
        self.s = settings
        self.client = AsyncOpenAI(base_url=settings.base_url, api_key=settings.api_key, timeout=settings.timeout_s,
                                  max_retries=1, http_client=http_client)
        self._usage_in_stream = True      # some servers reject stream_options; we fall back once and remember

    async def complete(self, messages: list) -> LLMResult:
        """One streamed chat completion. TTG_TIMEOUT_S bounds the WHOLE call (the HTTP client's timeout is per read)."""
        return await asyncio.wait_for(self._complete(messages), timeout=self.s.timeout_s)

    async def _complete(self, messages: list) -> LLMResult:
        kwargs = {"model": self.s.model, "messages": messages, "max_tokens": self.s.max_tokens,
                  "temperature": self.s.temperature, "stream": True}
        if self._usage_in_stream:
            kwargs["stream_options"] = {"include_usage": True}
        rf = response_format(self.s.structured)
        if rf:
            kwargs["response_format"] = rf
        t0 = time.perf_counter()
        try:
            stream = await self.client.chat.completions.create(**kwargs)
        except openai.BadRequestError as e:
            if self._usage_in_stream and "stream_options" in str(e):
                self._usage_in_stream = False
                return await self._complete(messages)
            raise
        ttft, parts, usage, finish = None, [], None, None
        async for chunk in stream:
            if getattr(chunk, "usage", None):
                usage = chunk.usage
            for choice in chunk.choices or []:
                delta = choice.delta
                if delta is not None and delta.content:
                    if ttft is None:
                        ttft = time.perf_counter() - t0
                    parts.append(delta.content)
                if choice.finish_reason:
                    finish = choice.finish_reason
        latency = time.perf_counter() - t0
        cached = None
        if usage is not None:
            details = getattr(usage, "prompt_tokens_details", None)
            cached = getattr(details, "cached_tokens", None) if details is not None else None
        return LLMResult(text="".join(parts), ttft_s=ttft if ttft is not None else latency, latency_s=latency,
                         prompt_tokens=getattr(usage, "prompt_tokens", None),
                         completion_tokens=getattr(usage, "completion_tokens", None),
                         cached_tokens=cached, finish_reason=finish)

    async def aclose(self) -> None:
        await self.client.close()
