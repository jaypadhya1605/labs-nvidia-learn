# M13 - Break-fix drills (agent traffic)

Same rules as Labs 02-04: one person injects, the other diagnoses **from symptoms only**, fixes it, then says the
one-liner to the customer from memory. Solo: inject, wait a day, then diagnose. Drills 1, 2, 8, 8b and 10 run on a
laptop with the mock engine; the others need the M4 GPU VM (drill 5 needs 2 GPUs).

```bash
drills/inject.sh list                       # the drills and what each needs
eval "$(drills/inject.sh 1)"                # inject drill 1 into THIS shell (stdout holds only export/unset lines)
eval "$(drills/inject.sh 1 revert)"         # undo it; then `. env/lab.env` restores your own defaults
python drills/diff_runs.py results/bench/mock-base.json results/bench/mock-drill1.json   # what changed, what moved
```
`drills/inject.sh` only prints. What to run next (restart the mock, relaunch the engine, start a container) goes to
stderr. Numbers marked **mock** were measured on the mock engine: they show the shape of the failure, never a GPU result.

| # | Needs | What the diagnoser sees | Say it (after the fix) |
|---|---|---|---|
| 1 | laptop | Same success, higher cost per task, higher TTFT | "Your prompt cache is only as good as the first byte that changes." |
| 2 | laptop (GPU variant) | Retries per task climb; success holds (laptop) or collapses (GPU) | "A retry that succeeds still costs a full LLM call." |
| 3 | GPU | About a third of the tasks fail with HTTP 400, always the long ones | "Agents grow their prompt every step; we size the context from your traces." |
| 4 | GPU | A change that helped at concurrency 1 hurts at 64 | "Speculation is a light-load tool; at storm-day load we measure it." |
| 5 | 2 GPUs | Tensor parallel 2 is no faster than one GPU | "Tensor parallelism is only as fast as the link between the GPUs." |
| 6 | GPU | `read_receipt` p95 goes from under a second to seconds; the agent slows down too | "An image is billed in tokens; we cap the resolution and measure accuracy there." |
| 7 | GPU | The engine refuses to start | "We give inference whole GPUs or MIG slices, so capacity is a number we can plan." |
| 8 | laptop | The guest message promises a rebooking that never happened | "A timeout doesn't mean it didn't happen, and a message isn't proof that it did." |
| 8b | laptop | The agent hangs ~30 s at startup, then dies | "Every dependency gets a connect timeout and a call timeout." |
| 9 | GPU | The LoRA adapter will not load | "An adapter is a contract with the server; we check it in CI." |
| 10 | laptop | A "smarter" prompt: success drops to 0%, or cost goes up 6x | "Every output token is paid for in latency." |

## How to diagnose (the same commands every time)
```bash
python -m ttg.agent.run --scenario S0016                                 # 11 tools: every step, every observation
python -m ttg.agent.run --n 20 --concurrency 4 --json > results/run.json # per task: error, violations, tool_time_s
python drills/diff_runs.py <good.json> <bad.json>                        # settings that differ + metrics that moved
python -m ttg.bench.scrape_metrics --url http://localhost:8000/metrics   # prefix cache, queue, tokens (engine side)
DRY_RUN=1 engines/launch.sh vllm base                                    # GPU: the exact engine command, EXTRA_ARGS too
docker logs ttg-engine 2>&1 | tail -50; nvidia-smi; nvidia-smi topo -m   # GPU: what the engine and the node say
```

---

## Drill 1 - A harmless prompt change, a bigger bill [laptop; the GPU shows the real cost]
**Inject:** `eval "$(drills/inject.sh 1)"`, then rerun your bench as `mock-drill1` (or `vllm-drill1`).

**Symptom.** Success stays at 100%. Cost per task goes up, and on a GPU so does TTFT on every call. **Mock:** cached
prompt tokens drop from ~94-96% to 0%; tasks per second at concurrency 8 drop from 3.82 to 3.09. On a GPU every call
re-prefills the ~1,840-token system prompt plus the history, so the effect is larger.

**Diagnose.** `cache_hit_ratio` in the two bench files, or prefix-cache hits vs queries in `scrape_metrics`.
`diff_runs.py` shows the setting that changed (`bust_cache: False -> True`); in real life it is a prompt commit.

**Root cause.** `TTG_BUST_CACHE=1` puts a fresh timestamp at the top of the system prompt on every LLM call. Prefix
caching (vLLM automatic prefix caching, SGLang RadixAttention, TensorRT-LLM KV block reuse) reuses KV blocks only up to
the first token that differs. The first token now differs on every call.

**Fix.** Revert. Anything per-request (time, request id, guest data) goes after the shared prefix, never above it
(ADR-02).

**Prevention.** The perf gate fails on $ per 1,000 tasks (ADR-09). A unit test builds the system prompt twice and
compares the bytes. Review rule: nothing dynamic above the history.

**Say it:** "Your prompt cache is only as good as the first byte that changes. We keep the shared prefix byte-identical
and put anything personal after it."

## Drill 2 - Retries out of nowhere [laptop; GPU variant]
**Inject:** `eval "$(drills/inject.sh 2)"`. Laptop: restart the mock in the same shell. GPU: relaunch vLLM.

**Symptom.** **Mock**, first 20 golden tasks at concurrency 4: success stays 20/20, but retries per task go from 0 to
1.0, LLM calls per task from 6.6 to 7.6, prompt tokens per task from 15,548 to 18,032 (+16%) and p50 task latency from
1.42 to 1.60 s. The retry hides the problem; the bill shows it. GPU variant: the replies are not JSON at all, tasks end
with `error: protocol: ...` after `TTG_MAX_RETRIES=2` retries, and success collapses.

**Diagnose.** `retries_per_task` in the bench file. `python -m ttg.agent.run --scenario S0005` shows the raw first
reply. GPU: `DRY_RUN=1 engines/launch.sh vllm base` shows `--chat-template /drills/...`; compare it with the
`chat_template` in the model's `tokenizer_config.json`.

**Root cause.** Laptop: the engine returns broken JSON on the first call of each task (`MOCK_FAULTS=invalid_json_once`).
GPU: a chat template with no role headers and no generation prompt
([drills/files/broken-chat-template.jinja](files/broken-chat-template.jinja)). The model sees one block of text and
continues it, instead of answering as the assistant.

**Fix.** Drop the custom template: the model's own template is the default. Then turn on constrained decoding
(`TTG_STRUCTURED=json_schema`) so the engine can only emit valid action JSON. It makes invalid JSON impossible; it does
not fix a wrong template. The eval gate still has to pass.

**Prevention.** Eval gate (success >= 95%), plus an alert when `retries_per_task` goes above 0.05. Send one known
request after every engine launch.

**Say it:** "A retry that succeeds still costs a full LLM call. We track retries per task, because they are the first
sign that the model and the serving stack disagree about the format."

## Drill 3 - Only the long tasks fail [GPU]
**Inject:** `eval "$(drills/inject.sh 3)"` (a context limit copied from a chat demo), relaunch, bench as `vllm-drill3`.

**Symptom.** About a third of the tasks fail (estimate from the oracle trajectories at 4 characters per token; your
tokenizer decides), always the long ones: cancelled, hotel, receipt, 7 to 11 tool calls. The error is an HTTP 400 that
mentions the maximum context length (the wording varies by version). They fail late: a task that dies on its 9th call
has already paid for 8.

**Diagnose.** Group `tasks[].error` by `tool_calls` in `results/bench/vllm-drill3.json`. The engine log shows the
context length it started with. Per-call prompts in this agent (estimate): the longest call of a task is ~2,700 tokens
at the median and ~3,300 at most, and each request also reserves `TTG_MAX_TOKENS` (256).

**Root cause.** `--max-model-len 3072`. The prompt grows with every step (append-only history), and prompt plus
`max_tokens` must fit in the context.

**Fix.** Size the context from traces: the longest call (~3,300 tokens) + 256 + margin, so at least 4,096; 8,192 is
comfortable on an 80 GB GPU. Or compact the history: the `policy_search` observation (~295 tokens) is the biggest one.
The opposite mistake, 131,072, costs nothing per request in vLLM's paged KV cache, but it can stop the engine from
starting on a small GPU or a MIG slice: the KV cache must hold one full-length sequence (16 GiB for Llama 3.1 8B BF16).

**Prevention.** Record the longest call per task in the bench. Fail CI when it plus `max_tokens` exceeds 80% of the
deployed context length (Helm `maxModelLen`).

**Say it:** "Agents grow their prompt with every step. We size the context window from your traces, not from a chat demo."

## Drill 4 - Faster alone, slower in a crowd [GPU]
**Inject:** nothing to break. This drill is a measurement: `vllm base` and `vllm spec`, same levels (1, 16, 64).

**Symptom (expected; measure yours).** At concurrency 1, `spec` lowers task latency: tool calls copy flight numbers,
ids and amounts from the context, so n-gram drafts are often accepted. At 64, `spec` delivers fewer tasks per second
than `base` and costs more per 1,000 tasks.

**Diagnose.** The two ledger rows side by side. Acceptance from the engine:
`python -m ttg.bench.scrape_metrics --raw | grep -i spec` (vLLM exports accepted and draft token counters; names vary
by version).

**Root cause.** At low batch, decode is bandwidth-bound: checking 4 draft tokens rides on the same weight read. But
speculation multiplies the tokens per step by k+1: at batch 64 with 4 drafts the target verifies 320 tokens per step,
past the H100's BF16 ridge (~295 FLOP/byte = 989 TFLOPS / 3.35 TB/s). The step turns compute-bound, and every
rejected draft burns FLOPs that other requests needed.

**Fix.** Decide per operating point (ADR-08): on only where the ledger says it wins. Or fewer draft tokens (2 instead
of 4).

**Prevention.** The perf gate runs at the operating-point concurrency, not at 1.

**Say it:** "Speculative decoding is a latency tool for light load. At storm-day load we measure it, and we turn it off
if it costs throughput."

## Drill 5 - Two GPUs, no faster [2 GPUs]
**Inject:** `eval "$(drills/inject.sh 5)"`, relaunch `vllm tp2`, bench with `--gpus 2`. The story: someone fixed a
hang on another VM with `NCCL_P2P_DISABLE=1` and `--disable-custom-all-reduce`, and the workaround stayed in the
launch config.

**Symptom.** Tensor parallel 2 is no faster per token than one GPU, sometimes slower. Tasks per GPU-hour roughly halve:
you pay for two GPUs.

**Diagnose.** `docker logs ttg-engine 2>&1 | grep "NCCL INFO" | grep -E "via|NVLS" | head`: `via SHM` means the
all-reduce goes through host memory; a healthy NVLink node shows `via P2P/...` or NVLS. `nvidia-smi topo -m` (NV# =
NVLink; PIX/PXB/PHB/SYS = PCIe paths). Bus bandwidth with and without P2P: `parallel/nccl_tests.sh 2` vs
`parallel/nccl_tests.sh 2 --p2p-off` (M7).

**Root cause.** Decode with TP runs 2 all-reduces per layer: 64 per token for 32 layers. The messages are small, so
latency per all-reduce dominates. With P2P off they go through host memory. vLLM's own custom all-reduce (which uses
P2P directly, not NCCL) was off too; with it on, `NCCL_P2P_DISABLE` alone mostly hits large messages.

**Fix.** Remove both workarounds. If the hardware really has no P2P (some PCIe VMs), run 2 data-parallel replicas
(`engines/launch.sh vllm dp2`) instead of TP.

**Prevention.** A node acceptance test (nccl-tests + topology) before any benchmark. Save `nvidia-smi topo -m` with
every result. No `NCCL_*` override in a launch config without a ticket.

**Say it:** "Tensor parallelism is only as fast as the link between the GPUs. We test the link before we benchmark the
model."

## Drill 6 - Receipts got slow [GPU]
**Inject:** `eval "$(drills/inject.sh 6)"`, then `python drills/fixtures.py receipts-photo` (the M8 receipts as
12 MP phone photos). Start the agent's LLM and the `ttg-vlm` container as in
[docs/06-multimodal.md](../docs/06-multimodal.md) section 7, but **without** `--mm-processor-kwargs`. Then
`python -m vlm.agent_receipts --concurrency 8`. The story: a mobile-app release stopped resizing photos, and the VLM
was redeployed without its size cap.

**Symptom.** `read_receipt` p95 jumps from under a second to seconds. With both engines on one GPU, the agent's own
TTFT rises too: the VLM's prefill competes for the same SMs. Under load a receipt read can hit `TTG_TOOL_TIMEOUT_S`
(10 s), and the reimbursement goes missing.

**Diagnose.** `usage.prompt_tokens` of one VLM request, or the prompt-token counters of the `ttg-vlm` engine
(`python -m ttg.bench.scrape_metrics --url http://localhost:8001/metrics`) before and after one receipt.
`tool_time_s` per task in the results. The token math: `vlm/vlm_tokens.py`; TTFT against resolution:
`vlm/vlm_sweep.py` (M8).

**Root cause.** Qwen2.5-VL turns every 28 x 28-pixel patch into one token. A 12 MP photo (3024 x 4032) is
108 x 144 = 15,552 tokens; the same receipt at 896 x 896 is 1,024. The checkpoint's default `max_pixels` (12,845,056)
lets a 12 MP photo through untouched, so nothing capped it once the app stopped resizing.

**Fix.** Cap it at the server again: `--mm-processor-kwargs '{"max_pixels": 1003520}'` (at most 1,280 tokens; a
12 MP photo becomes ~1,230), and resize on upload in the app (a 1,344-pixel long side is 1,728 tokens, docs/06). Then
rerun the M8 accuracy check: too small misreads small print.

**Prevention.** A size cap at the API edge and on the server, not just in the app. Prompt tokens per image on the VLM
dashboard. `vlm/vlm_sweep.py` in the perf gate.

**Say it:** "An image is billed in tokens. We cap the resolution at what the model needs to read the receipt, and we
measure accuracy at that cap."

## Drill 7 - The engine will not start [GPU]
**Inject:** `eval "$(drills/inject.sh 7)"`; it prints a `docker run` that holds 20 GB on GPU 0. Run it, then relaunch.

**Symptom.** `engines/launch.sh` waits, then the container exits. The log says the free memory on the device is less
than the desired GPU memory utilization (0.9). Older vLLM versions started with a tiny KV cache instead: fewer agents
at once, and queueing.

**Diagnose.** `nvidia-smi` (the Processes table), `nvidia-smi --query-compute-apps=pid,process_name,used_memory
--format=csv`, `docker ps`, `docker logs ttg-engine | tail -30`.

**Root cause.** `--gpu-memory-utilization 0.90` asks for 90% of the GPU (~71 GB of 80 GB). Another process holds 20 GB,
so only ~59 GB is free.

**Fix.** Stop the other process (`docker rm -f ttg-gpu-hog`). If the GPU must be shared, lower `GPU_MEM_UTIL` on purpose
(0.70 fits next to 20 GB) and accept a smaller KV cache: fewer agents at once. Better: MIG for hard isolation
(🔁 Lab 02).

**Prevention.** One engine per GPU. In Kubernetes, GPU requests hand out whole GPUs. A preflight that checks free
memory before launch.

**Say it:** "On a shared GPU the engine sizes its KV cache from what is left. We give inference whole GPUs or MIG
slices, so capacity is a number we can plan."

## Drill 8 - The agent says it rebooked; the system of record says no [laptop]
**Inject:** `eval "$(drills/inject.sh 8)"`, then `python -m ttg.agent.run --n 20 --concurrency 4`.

**Symptom.** **Mock**, first 20 golden tasks: success drops from 20/20 to 5/20. The 5 that pass owe nothing: lookups only,
no booking and no voucher. The rest show `missing_rebooking`, `missing_voucher:credit`,
`missing_voucher:reimbursement`, `missing_voucher:meal`. Every observation reads
`{"error":"tool <name> timed out after 0.001s"}`, yet the guest message still promises the new flight and the vouchers.

**Diagnose.** `python -m ttg.agent.run --scenario S0016` prints each observation. `tool_time_s` is near zero. The
violations in the bench file point at write actions.

**Root cause.** `TTG_TOOL_TIMEOUT_S=0.001` is shorter than any tool call. The model (here the mock; a real model may do
better or worse) treats an error observation as done and writes the message from its plan.

**Fix.** A realistic timeout (the default is 10 s; size it from the tools' p99). An explicit rule in the agent: an error
observation is not success; retry once with the same idempotency key, or hand off to a human. Build the guest message
from tool results, not from the plan. A timed-out call may still have happened on the server, so retrying a booking
without an idempotency key can book twice (`duplicate_booking`).

**Prevention.** The eval gate scores actions against the system of record (`ttg.world.score`), not the message. Alert on
the tool error rate.

**Say it:** "A timeout doesn't mean it didn't happen, and an agent's message isn't proof that it did. We score actions
against the system of record."

## Drill 8b - The agent API hangs at startup [laptop]
**Inject:** `eval "$(drills/inject.sh 8b)"`, start only the guest MCP server, then `time python -m ttg.agent.run --n 1`.

**Symptom.** Measured on a Linux workstation: the agent hangs ~30 s, then dies with `httpx.ConnectTimeout` inside an
`ExceptionGroup`. No task runs. The 120 s tool timeout never comes into play. (If your network answers "unreachable"
at once, you get a fast `ConnectError`: same lesson.)

**Diagnose.** The traceback ends in an HTTP connect. `curl -m 5 "$TTG_MCP_OPS_URL"`. Check which URLs the agent uses
(`settings` in the bench file, or `env | grep TTG_MCP`). In Kubernetes: `kubectl get endpoints` and the NetworkPolicy.

**Root cause.** The MCP client connects when the agent starts, to list the tools. A per-call timeout does not cover
connection setup, and `TTG_MCP_OPS_URL` points at an address that never answers.

**Fix.** The right URL. Fail fast: a connect timeout of a few seconds, and a readiness probe that checks both MCP
servers (`scripts/wait_for.py` does this in CI), so Kubernetes never routes guests to an agent without tools.

**Prevention.** Readiness probes on the agent API; NetworkPolicy tests in the M9 pipeline.

**Say it:** "Every dependency gets two timeouts, connect and call, and the agent isn't ready until its tools are."

## Drill 9 - The adapter will not load [GPU]
**Inject:** `python drills/fixtures.py lora-rank --rank 32`, `eval "$(drills/inject.sh 9)"`, relaunch `vllm lora`.

**Symptom.** The engine fails to start, or starts without the adapter. The log says the LoRA rank (32) is greater than
`max_lora_rank` (16). Requests for `ttg-short` fail with "model not found".

**Diagnose.** `docker logs ttg-engine 2>&1 | grep -i lora`. `grep '"r"' "$ADAPTER_DIR/ttg-short/adapter_config.json"`.
`DRY_RUN=1 engines/launch.sh vllm lora` shows `--max-lora-rank 16`; the Helm chart has `lora.maxRank`.

**Root cause.** The server reserves LoRA buffers for a maximum rank, and an adapter above it cannot load. Here only the
config says 32 (the fixture fakes it); in real life someone retrained with r=32.

**Fix.** Here: revert (point `ADAPTER_DIR` back at the real adapter). In real life: retrain at r <= 16, or raise the
limit on purpose (`EXTRA_ARGS='--max-lora-rank 32'` and Helm `lora.maxRank: 32`), which reserves more GPU memory per
adapter slot. Never raise the limit for this fixture: its scaling (alpha / r) would be wrong.

**Prevention.** A CI check that the adapter's r <= `lora.maxRank` before deploy. The LoRA release gate (🔁 Lab 04 M8)
compares quality too.

**Say it:** "An adapter is a contract with the server: rank, target modules, base model. We check the contract in CI,
before the adapter meets production."

## Drill 10 - A smarter prompt, a broken agent [laptop]
**Inject:** `eval "$(drills/inject.sh 10)"` ("think step by step in great detail"), bench as `mock-drill10`; then
`export TTG_MAX_TOKENS=1024` and bench again as `mock-drill10b`.

**Symptom.** **Mock:** with the default `TTG_MAX_TOKENS=256` the JSON gets cut off: protocol errors, 3 retries, 0%
success. With 1,024: success comes back, but output tokens per task go from 269 to 2,185 (8x), p95 task latency from
2.9 s to 18 s (it misses the 12 s SLO) and $ per 1,000 tasks goes up ~6x.

**Diagnose.** `finish_reason: length` on the cut-off calls, `mean_completion_tokens` and `retries_per_task` in the bench
file, and `diff_runs.py` (`thoughts: brief -> verbose`).

**Root cause.** Output tokens are the slow tokens: decode makes them one at a time. Verbose reasoning multiplies them,
and at a fixed `max_tokens` it truncates the JSON.

**Fix.** Brief thoughts (one sentence, the default prompt) and `max_tokens` 256. If the task really needs reasoning, use
a reasoning model with its own budget, and measure success and $ per 1,000 tasks together.

**Prevention.** The perf gate on p95 and $ per 1,000 tasks; the eval gate catches truncation.

**Say it:** "Every output token is paid for in latency. We let the agent think in one sentence and spend the GPU on
actions."

---

**Checkpoint:** you diagnosed at least six drills from symptoms only, each in under 10 minutes, and for each one you can
name the metric that showed it first.

**Say it:** "We broke it on purpose before your storm day did: ten failure modes, each with the metric that catches it
and the gate that stops it reaching production."
