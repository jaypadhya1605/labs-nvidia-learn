# Feature request (PRD) - Per-request prefix-cache accounting, tied to the agent session

**For:** NVIDIA NIM for LLMs and NVIDIA Dynamo product teams; upstream issues for vLLM, SGLang and TensorRT-LLM.
**From:** NVIDIA solutions architect, Tern Travel Group PoC (fictional customer). **Priority:** high for agent
workloads. **Customer timing:** before peak storm season (December).

## Problem
Agent traffic lives on prefix reuse. Tern's agent makes ~6.5 LLM calls per task, and an estimated ~96% of its prompt
tokens repeat from the previous call. When reuse breaks, cost and TTFT jump, but nothing tells the operator which
session, which step, or which prompt change broke it. Engines report cache hits as global counters, and per-response
cached tokens only when a flag is set. Neither says where a prompt stopped matching the cache.

## Impact (measured in the PoC)
- Drill 1 (one timestamp at the top of the system prompt): cached prompt tokens ~95% -> 0%; on the mock engine,
  tasks per second at concurrency 8 fell from 3.82 to 3.09. On the H100: $[measured] -> $[measured] per 1,000
  disruptions, p95 TTFT per call [measured] -> [measured].
- The prompt still produced correct answers, so the eval gate passed. Only the perf gate and the bill caught it.
- Time to root cause in the drill, without the per-request data: [measured] minutes. The engine side gave no pointer.

## Current state (to verify against the current docs before filing)
| Engine | What exists today, as we understand it | Gap |
|---|---|---|
| vLLM | Global Prometheus counters (prefix-cache queries and hits); `usage.prompt_tokens_details.cached_tokens` when started with `--enable-prompt-tokens-details` | Off by default; no session or step attribution; no divergence point |
| SGLang | RadixAttention hit statistics in server logs and metrics; cached tokens per request in response metadata | To verify: field names, OpenAI-compatible `usage` support |
| TensorRT-LLM (trtllm-serve) | KV-cache block reuse statistics in iteration stats | To verify: per-request reporting through the OpenAI API |
| NIM for LLMs | Inherits the backend engine; metrics at `/v1/metrics` | To verify: which fields pass through per backend |
| Dynamo | KV-aware routing computes prefix overlap per request | To verify: whether the overlap is exposed per request |

## Proposal
1. **Always return cached tokens** in the OpenAI-compatible field `usage.prompt_tokens_details.cached_tokens` when
   prefix caching is on (opt-out flag instead of opt-in).
2. **Accept a session id** (an `x-session-id` header or request `metadata`) and propagate it to OpenTelemetry spans
   and as exemplars on the prefix-cache metrics, so a dashboard can jump from "hit ratio dropped" to the sessions that
   dropped it.
3. **Debug mode: report the divergence point.** For a request in a known session, return the number of prompt tokens
   that matched the longest cached prefix (`prefix_match_tokens`). A timestamp on line 1 shows up as a match length near
   0 on every call, which points straight at the changed byte.
4. **Per-adapter hit ratio** for multi-LoRA serving, so a new adapter that changes the prompt shows up on its own line.

## API sketch
```json
// request
POST /v1/chat/completions
x-session-id: task-S0016-7f3a
{"model": "ttg-llm", "messages": [...], "max_tokens": 256}

// response (debug mode)
"usage": {"prompt_tokens": 2831, "completion_tokens": 41,
          "prompt_tokens_details": {"cached_tokens": 2768},
          "prefix_match_tokens": 2768}
```

## Acceptance criteria
1. Cached tokens are returned on 100% of responses when prefix caching is on, with no flag.
2. The session id appears on the engine's spans and on the cache-metric exemplars; it is opaque (no prompt text).
3. With drill 1 injected, `prefix_match_tokens` is under 20 on every call; healthy, it is at least the shared
   system prompt (~1,840 tokens) from the second call of a task on.
4. Throughput overhead under 1% at concurrency 64 (`python -m ttg.bench.agent_bench`, `vllm-base` vs the feature on).
5. Works with LoRA adapters, FP8 weights and an FP8 KV cache.

## Alternatives considered
| Alternative | Why it is not enough |
|---|---|
| Hash prompts client-side and guess hits | Misses engine-side eviction and block boundaries; every client reimplements it |
| Log full requests and diff them | Prompt text in logs is a privacy problem (guest data) and expensive at storm-day volume |
| Agent-side tracing only | Shows that tokens were sent, not whether the engine reused them |

## Workaround until then
The perf gate on $ per 1,000 tasks (ADR-09), cached tokens per call recorded by the agent's LLM client
(`ttg/agent/llm.py`), and the review rule from ADR-02: nothing dynamic above the shared prefix.
