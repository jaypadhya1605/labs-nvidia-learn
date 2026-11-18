# Storm day: what accelerated inference does to the cost of an AI agent

*Draft. Every `[measured]` comes from `results/ledger/` with its GPU, engine versions and date. Tern Travel Group is a
fictional company and every traveler in this post is synthetic.*

At 6 a.m. a winter storm closes three hubs. By noon, 40,000 travelers have a cancelled or badly delayed flight, and
every one of them wants the same three answers: when do I leave, where do I sleep, and what am I owed. At 11 minutes
per conversation that is about 7,330 agent-hours of work, or roughly 1,220 people on the phones at once for six hours.
No contact center has that. So guests wait three hours, and the agents who do pick up apply a 12-page compensation
policy from memory, under pressure. Some guests get too little. Many get too much.

I built a small AI agent for that day, then measured what it costs to run on GPUs. The interesting part was not the
agent. It was how much the serving stack changes the bill.

## The agent
The Disruption Concierge handles one traveler at a time. It checks the flight and the loyalty tier, looks up the
policy, rebooks a cancelled flight, books a hotel when one is owed, issues meal, credit and receipt vouchers, and
writes a short message. Any voucher above $300 waits for a human. It runs as a LangGraph state machine. The tools sit
in two MCP servers: one for flights and hotels, one for loyalty, policy and vouchers. The model is Llama 3.1 8B
Instruct, served behind an OpenAI-compatible API.

Every answer is checked. A policy oracle knows exactly which bookings and voucher amounts each of 200 synthetic
travelers is owed, so "success" means the agent did precisely the right things, not that its message sounded right.

## The surprise: agents read far more than they write
A simple case (a short weather delay) takes 4 model calls. A cancelled flight for a family of four, with a hotel and a
taxi receipt, takes 12. On average the agent makes 6.5 calls per traveler. Each call resends the whole conversation:
the instructions, the tool catalog, the policy digest, and every step so far.

That adds up to about 15,100 prompt tokens per traveler and only about 240 output tokens. And about 96% of those
prompt tokens are the same on every call: the ~1,840-token system prompt and the steps the model has already seen.
(These are estimates at four characters per token; the measured numbers are in the table below.)

So agent traffic is not chat traffic. It is long, repetitive prompts with short answers, many times per task. That
shape decides which optimizations matter.

## What I measured
One H100 GPU, four engines (vLLM, SGLang, TensorRT-LLM and NVIDIA NIM), the same model and byte-identical prompts.
The agent speaks one JSON action format on every engine, so the only thing that changes between runs is the engine.
Each configuration ran the 200 golden travelers at rising concurrency, and I kept the highest load whose p95 time per
traveler stayed under 12 seconds with at least 95% of travelers handled exactly by the policy. Then one unit: GPU
dollars per 1,000 handled disruptions.

| Step | Tasks per GPU-hour | $ per 1,000 disruptions |
|---|---|---|
| A naive server, one request at a time | [measured] | [measured] |
| + continuous batching (vLLM, prefix caching off) | [measured] | [measured] |
| + prefix caching | [measured] | [measured] |
| + FP8 weights | [measured] | [measured] |
| + n-gram speculative decoding | [measured] | [measured] |
| + a LoRA that bakes the policy in (short prompt) | [measured] | [measured] |

Best engine at the operating point: [measured]. Storm-day capacity, with one spare GPU per region: [measured] GPUs in
the US and [measured] in the EU, for about $[measured] of GPU time for the whole storm.

## Three lessons

**1. One timestamp can erase your prefix cache.** Prefix caching (vLLM's automatic prefix caching, SGLang's
RadixAttention, TensorRT-LLM's KV block reuse) computes a shared prompt once and reuses it. It only works up to the
first token that differs. I added one line, `Current time: ...`, to the top of the system prompt, the kind of change
that passes code review because it "helps the model know the date". Answers stayed correct. The cache hit ratio
dropped from about 95% to 0%, and on the mock engine throughput at 8 concurrent travelers fell from 3.82 to 3.09 tasks
per second. On the GPU it cost [measured] per 1,000 disruptions, because every call re-read 1,840 tokens it had read a
moment before. The rule I now apply everywhere: nothing dynamic above the shared prefix.

**2. Speculative decoding is a light-load tool.** Tool calls copy flight numbers, ids and amounts straight from the
context, so cheap n-gram drafts are often right. At low concurrency that cut latency by [measured]. At storm-day load
the GPU is busy anyway, and checking drafts that turn out wrong burns compute other travelers needed: [measured]. The
honest answer to "should we turn it on?" is "at which load?"

**3. Thinking out loud is expensive.** Asking the agent to "think step by step in great detail" did not make it more
correct. On the mock engine, with the usual 256-token limit, the JSON got cut off and every task failed. With the limit
raised to 1,024, output tokens per traveler went up about 8 times, and p95 latency blew through the 12-second target.
On the GPU: [measured]. Output tokens are the slow ones: the model makes them one at a time. One sentence of reasoning
per step was enough here.

## What this means if you are building agents
- **Size from traces, not from chat.** The prompt grows with every step. Measure your longest call before you set a
  context limit, and before you pick a GPU.
- **Gate on the task, not the tokens.** A quantized model, a new engine or a LoRA adapter ships only if task success
  on the golden set holds. Tokens per second never told me whether a voucher amount was right.
- **Price the decision.** "$ per 1,000 disruptions at a 12-second p95" is a number a CFO can compare with overtime and
  over-compensation. Tokens per second is not.

## Try it
The repo runs on a laptop with a mock engine, so you can see the shape of every result without a GPU, and on one
rented H100 for the real numbers: [repo link]. The workshop notebook takes 90 minutes, and it starts by breaking the
prefix cache on purpose.

*This is a personal project. Tern Travel Group, its policy and its travelers are fictional, and all data is synthetic.
Results: [measured] on [GPU], [engine versions], [date].*
