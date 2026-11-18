# M13 - Interview drill: NVIDIA Senior Solutions Architect, Gen AI (JR2019035)

Answer each question out loud in under 90 seconds, then check the artifact named under it. Lab numbers below are
estimates or placeholders (`__`). Before any interview, replace them with YOUR measured numbers from `results/`.

## A. The two first-hand questions (verbatim from earlier candidates)

### "Describe the structure of the NVL72 as best as you can."
"GB200 NVL72 is one liquid-cooled rack that behaves like one very large GPU. The building block is the GB200
superchip: one Grace CPU and two Blackwell GPUs, joined by NVLink-C2C. The rack holds 36 of them, so 36 Grace CPUs
and 72 Blackwell GPUs, packed as 18 compute trays of two superchips each. Nine NVLink switch trays connect all 72 GPUs
into a single NVLink domain: fifth-generation NVLink at 1.8 TB/s per GPU and about 130 TB/s across the rack. Memory is
13.4 TB of HBM3E at 576 TB/s, plus 17 TB of LPDDR5X on the Grace side.

Why it matters: a DGX H100 has an NVLink domain of 8 GPUs at 900 GB/s each. NVL72 makes it 72 GPUs at twice the
per-GPU bandwidth. Tensor and expert parallelism for a large MoE model stay on NVLink instead of crossing the network,
and prefill and decode can be split across GPUs inside one rack (Dynamo). Racks scale out over InfiniBand or
Spectrum-X Ethernet."
- Proof: [docs/04-gpu-cluster-architecture.md](04-gpu-cluster-architecture.md), M7.
- Sources: [NVIDIA GB200 NVL72](https://www.nvidia.com/en-us/data-center/gb200-nvl72/) (36 CPUs, 72 GPUs, 1.8 TB/s,
  130 TB/s, 13.4 TB HBM3E at 576 TB/s, 17 TB LPDDR5X, liquid-cooled). The 18 compute trays and 9 switch trays are the
  widely published tray layout: confirm against NVIDIA's current material before you quote it.

### "What was your main goal when optimizing an ML model?"
"My main goal is the customer's metric at their SLO: cost per decision at the latency they promised, not a benchmark
number. The first job is making the measurement honest. I partnered with an internal engineering team on a LoRA
fine-tune of a 4B Gemma model on NVIDIA A100s, and the fine-tuned model looked worse than base. It turned out the base
had been scored through vLLM and the fine-tuned model through a custom inference path: two serving stacks, two sets
of decoding defaults. Once both ran through the same harness, engine and decoding settings, the comparison meant
something.

In my Lab 05 I did the same thing at scale. Four engines get byte-identical prompts through one JSON protocol, and the
ledger reports tasks per GPU-hour and dollars per 1,000 disruptions at a 12-second p95. Then I optimize in order of
leverage: prefix caching first, because ~96% of the agent's prompt tokens repeat (the shared system prompt alone
is about three quarters of them); then FP8; then speculative decoding for the JSON tool steps; then baking the
policy into a LoRA. The result was __ tasks per GPU-hour, from __ at baseline."
- Proof: M10 ledger, [docs/09-results-and-exec-readout.md](09-results-and-exec-readout.md).

## B. Questions by JD line
Each answer is 3-5 sentences. The line under it names the module or file that proves it.

### Model architectures and their computational complexity
**1. Walk me through the cost of a dense transformer, then GQA, MLA, MoE and a hybrid Mamba model.**
Prefill costs about 2 x parameters FLOPs per token and is compute-bound. Decode reads every weight per token and is
bandwidth-bound: Llama 3.1 8B in BF16 streams ~15 GB of weights per step (the 1 GB embedding table is gathered, not
streamed), a ~4.6 ms floor on an H100 SXM. GQA shrinks the KV cache:
8 KV heads instead of 32 gives 128 KiB per token instead of 512 KiB. MLA (DeepSeek-V3) caches a compressed latent,
about 69 KiB per token for a 671B model. MoE decouples memory (total parameters) from compute (active parameters):
DeepSeek-V3 is 671B total and ~37B active. Hybrid Mamba models keep a fixed-size state for most layers, so only a few
attention layers grow KV with context.
- Proof: M2, `archmath/archmath.py`, [docs/03-model-complexity.md](03-model-complexity.md).

**2. Why is an MoE model fast at batch 1 but not proportionally at batch 64?**
At batch 1 a token reads only its top-k experts, so bytes per step track active parameters. As the batch grows,
tokens route to different experts. With uniform routing the share of experts touched per layer is 1 - (1 - k/E)^B.
For 8 of 128 experts that is ~6% at batch 1 and ~87% at batch 32. Decode bytes per step then approach the full model,
while FLOPs per token stay at the active size. That is why large MoE serving needs expert parallelism and large NVLink
domains.
- Proof: M2 optional MoE run (🔁 Lab 04's Nemotron 3.5 Lightning NIM).

**3. What changes at 128K context?**
KV memory grows linearly: Llama 3.1 8B needs 16 GiB of KV for one 131,072-token sequence. Prefill attention grows
quadratically and dominates the MLP at long context. Every decode step also reads the whole KV, so bandwidth per step
grows with context. The levers are chunked prefill, FP8 KV cache, KV offload, context parallelism (ring attention) and
hybrid SSM models.
- Proof: M2 table, drill 3.

### Deploying and optimizing LLM inference (vLLM, TensorRT-LLM, SGLang)
**4. vLLM, SGLang or TensorRT-LLM: how do you choose?**
vLLM has the broadest model support and day-0 coverage, with PagedAttention and automatic prefix caching. SGLang's
RadixAttention prefix tree and fast structured output suit agents with shared prefixes. TensorRT-LLM has NVIDIA's most
optimized kernels (FP8, NVFP4), in-flight batching and KV block reuse, at the price of more configuration. NIM
packages an optimized engine per GPU as a supported microservice. I choose by measuring the customer's traffic on
identical prompts, not someone else's benchmark.
- Proof: M4, `engines/launch.sh`, `results/bench/*-base.json`.

**5. Why does prefix caching matter so much for agents?**
An agent resends its whole context on every step. Tern's agent makes ~6.5 calls per task with a ~1,840-token shared
system prompt, so an estimated ~96% of prompt tokens repeat. vLLM's automatic prefix caching, SGLang's RadixAttention
and TensorRT-LLM's KV block reuse compute those tokens once. The way to lose it is anything dynamic at the top of the
prompt, like a timestamp. I watch `vllm:prefix_cache_hits` against `vllm:prefix_cache_queries`, or `cached_tokens` per
response.
- Proof: M4-M5, `python -m ttg.bench.scrape_metrics`, drill 1.

**6. When does speculative decoding help, and when does it hurt?**
A cheap drafter proposes k tokens and the target model verifies them in one pass. Decode is bandwidth-bound, so the
verification rides on the same weight read and accepted tokens are almost free. It helps at low concurrency with high
acceptance. N-gram prompt lookup suits tool calls that copy ids and amounts from the context. It hurts at high
concurrency: k drafts multiply the tokens per step by k+1, so at batch 64 with k=4 the verify pass handles 320 tokens,
past the H100's BF16 ridge (~295 FLOP/byte). The step turns compute-bound, and rejected drafts waste FLOPs.
- Proof: M5 (`engines/launch.sh vllm spec`), drill 4, ADR-08.

**7. FP8, NVFP4, INT4 AWQ: what are the trade-offs?**
FP8 halves weight bytes versus BF16 on Hopper, Ada and Blackwell tensor cores and is usually close to lossless.
NVFP4 (Blackwell) is 4-bit with small blocks that each carry an FP8 scale. It cuts bytes about 4x and needs
calibration with ModelOpt. INT4 AWQ is weight-only: great for memory-bound decode, but dequantization slows
compute-bound prefill. An FP8 KV cache doubles KV capacity. I gate every option on the task metric (the golden set),
not on perplexity.
- Proof: M5 (`vllm fp8`, `trtllm fp8`, `fp8kv`), ADR-07.

**8. A customer's p95 TTFT doubled after a release. What are your first three checks?**
First, did the prompt change? Compare cached tokens and the prefix-cache hit ratio before and after. Second, did load
or input length change? Look at prompt tokens per call and `num_requests_waiting`, because queueing shows up as TTFT.
Third, did the engine configuration change: chunked prefill, eager mode, precision, max-model-len or GPU memory
utilization? Then profile.
- Proof: drills 1, 3 and 7; M5.

### Training and fine-tuning (PyTorch, Hugging Face)
**9. Memory math: full fine-tune vs LoRA for Llama 3.1 8B.**
Full fine-tuning with mixed-precision Adam needs about 16 bytes per parameter: 8B x 16 = ~128 GB before activations,
so several GPUs with FSDP. LoRA freezes the 16 GB BF16 base and trains low-rank adapters. Rank 16 on all seven linear
projections is ~42M parameters (~0.5%), whose optimizer state is under 1 GB. That fits one 80 GB GPU. LoRA changes
behavior and format well; it adds little new knowledge.
- Proof: M6 (`finetune/`), [docs/05-parallelism-math.md](05-parallelism-math.md).

**10. How did you build fine-tuning data for the agent?**
The policy oracle (`ttg.world.expected_plan`) gives exact trajectories for free. Each step becomes a prompt-completion
example (conversation so far, then the next JSON action), so TRL trains on the completion only. The prompt is the
short variant, 560 characters instead of 7,360, so the policy moves into the weights. A different seed is held out,
and an optional teacher on build.nvidia.com is kept only where it matches the oracle.
- Proof: M6, ADR-05.

**11. DDP vs FSDP: what changes in memory and communication?**
DDP keeps a full copy of the model, gradients and optimizer on every GPU, and all-reduces gradients each step:
2(N-1)/N x gradient bytes per GPU. FSDP (ZeRO-3) shards parameters, gradients and optimizer states, so memory per GPU
falls to about 16 x parameters / N plus activations. The price is all-gathers in forward and backward plus a
reduce-scatter, about 1.5x DDP's traffic. For an 8B LoRA, DDP is enough; full fine-tuning needs FSDP2.
- Proof: M7 training scaling run.

### GPU cluster architecture and parallelism
**12. Explain TP, PP, DP, EP and CP with the math.**
DP replicates the model and all-reduces gradients: 2(N-1)/N x gradient bytes per GPU per step. TP splits each layer's
matrices and needs 2 all-reduces per layer in forward (and 2 in backward) of batch x sequence x hidden x bytes. PP splits
layers into stages, sends activations point to point, and wastes a bubble of (P-1)/(m+P-1) for m micro-batches. EP
places experts on different GPUs with two all-to-alls per MoE layer. CP splits the sequence and passes KV blocks
around a ring. Rule of thumb: TP inside an NVLink domain, PP across nodes, DP on top, EP for MoE, CP for very long context.
- Proof: M7, `parallel/parallelism_calc.py`, [docs/05-parallelism-math.md](05-parallelism-math.md).

**13. Estimate the pretraining time for an 8B model on 1T tokens on 64 H100s.**
Training FLOPs are about 6 x parameters x tokens: 6 x 8e9 x 1e12 = 4.8e22. 64 GPUs x 989 TFLOPS (BF16 dense) x 40%
MFU is about 2.5e16 FLOP/s. That gives ~1.9e6 seconds, about 22 days, or ~18 days at 50% MFU. Then add checkpointing,
evaluation and failure recovery.
- Proof: M7 calculator.

**14. Why does tensor parallelism over PCIe hurt decode?**
Each decode step runs 2 all-reduces per layer: 64 per token for a 32-layer model. The messages are small (batch x
hidden x 2 bytes), so link latency dominates, not bandwidth. NVLink keeps each one short; PCIe, and especially PCIe
without peer-to-peer, adds milliseconds per token. Across PCIe, prefer data-parallel replicas.
- Proof: M7 TP sweep, drill 5 (`NCCL_P2P_DISABLE=1`).

**15. What does NCCL bus bandwidth mean, and how do you test a node?**
`all_reduce_perf` reports algorithm bandwidth (bytes / time) and bus bandwidth, which scales it by 2(n-1)/n for
all-reduce so results compare across GPU counts. I run nccl-tests from 8 bytes to 4 GB inside the NGC PyTorch
container, check `nvidia-smi topo -m`, and read `NCCL_DEBUG=INFO` for the transport (P2P, NVLS, SHM, NET). Then I
compare large-message bus bandwidth with the interconnect spec and small-message latency with the TP pattern.
- Proof: M7, `parallel/nccl_tests.sh`, `parallel/compare_nccl.py`.

### DevOps: Docker, Kubernetes, GitLab, Linux
**16. How would you deploy this agent on Kubernetes?**
One Helm chart: the engine as a Deployment with `Recreate`, a GPU limit, `/dev/shm`, a startupProbe for the model
download and a PVC model cache; the agent API and both MCP servers as their own Deployments. Secrets come from a vault
through External Secrets (🔁 Lab 04). NetworkPolicy lets only the agent reach the engine and the MCP servers. The API
scales on CPU, the engine on queue depth, and GitLab deploys with `helm upgrade --install`.
- Proof: M9, `deploy/helm/ttg`, [docs/02-architecture-and-adrs.md](02-architecture-and-adrs.md).

**17. What do your GitLab gates check?**
Lint and unit tests first. The eval gate starts the mock engine and MCP servers, runs 40 agent tasks and fails under
95% success. The perf gate runs on a GPU runner and fails if p95 task latency or $ per 1,000 tasks regresses more than
10% against the recorded baseline. Then the image builds, and production deploy is manual. Baselines are re-recorded
deliberately, with a reason, never automatically.
- Proof: M9, `.gitlab-ci.yml`, `scripts/gate.py`, ADR-09.

### Agentic AI frameworks, tools and protocols (LangChain, LangGraph, MCP)
**18. LangChain, LangGraph, MCP: what does each do in your design?**
MCP is the contract between the agent and its tools: each tool lives in a server with a typed schema, usable from any
framework. LangChain is the integration layer: `langchain-mcp-adapters` loads the MCP tools. LangGraph is the
orchestration: an explicit state machine with nodes, edges, retries and a step budget. NIM or another engine serves the
model behind an OpenAI-compatible API, and NVIDIA's NeMo Agent Toolkit can profile and evaluate such workflows.
- Proof: M3, `ttg/agent/graph.py`, ADR-03 and ADR-04.

**19. Native tool calling or a JSON protocol?**
For one engine and one model, native tool calling is the right default. For a bake-off across vLLM, SGLang,
TensorRT-LLM and NIM, the parsers differ (`llama3_json`, `llama3`, `--tool_parser`). So the agent speaks a JSON action
protocol, validated against the MCP schemas: identical prompts, comparable outputs, and retries I can count.
Constrained decoding (`TTG_STRUCTURED=json_schema`) removes invalid JSON where the engine supports it.
- Proof: ADR-01, drill 2.

**20. How do you stop an agent from giving away money or following injected instructions?**
The policy is the oracle: every prompt, model, LoRA and engine is scored on 200 golden scenarios for exact bookings
and voucher amounts, so over-compensation is a violation. Hard limits live in tools, not prompts: `issue_voucher`
returns `PENDING_APPROVAL` above $300, and the final answer must flag it. Tool output is untrusted data (🔁 Lab 03),
so we plant injections and measure. MCP servers expose only the actions needed, behind NetworkPolicy, and every call is
traced.
- Proof: M3, `ttg.world.score`, ADR-11.

### Multimodal LLMs and VLMs
**21. How do images change the cost of a request?**
An image becomes tokens. Qwen2.5-VL turns each 28 x 28-pixel patch into one token: an 896 x 896 receipt is 1,024
tokens, a 12 MP phone photo (3024 x 4032) 15,552, so ~15x the prefill and KV. Cross-attention VLMs such as Llama 3.2 Vision keep
image features out of the decoder sequence (up to 4 tiles of 1,601 tokens, attended through cross-attention layers).
The policy: cap `max_pixels`, crop to the region you need, and measure TTFT against resolution.
- Proof: M8, `vlm/vlm_tokens.py`, `vlm/vlm_sweep.py`, [docs/06-multimodal.md](06-multimodal.md), drill 6.

### Profiling latency, throughput, memory and I/O
**22. The agent is slow. Where do you start?**
Split task latency into LLM time and tool time (`llm_time_s`, `tool_time_s`), then calls per task. For the LLM part,
separate TTFT from decode and check cached tokens. High TTFT points to prefill, cache misses or queueing; high
inter-token latency points to batch size, precision, TP over PCIe or speculation at high load. Slow tools point to the
MCP backends and their timeouts. Then profile the engine (vLLM's torch profiler via `/start_profile`, or Nsight Systems)
to see GEMM, attention and all-reduce time against CPU scheduling gaps.
- Proof: M5, `profiling/`.

**23. How do you profile memory and I/O, not just latency?**
Memory is weights plus KV plus activations: the engine log shows the KV capacity at startup, and KV usage and
preemptions show it under load. DCGM's DRAM-active metric shows how close decode runs to the bandwidth roof (🔁 Lab 01
roofline). For I/O, cold start is image pull plus weight load plus warm-up. I measure weight-load GB/s from network
storage, local NVMe and a pre-filled cache. For training, data-loader throughput must beat the GPU step time.
- Proof: M5, `profiling/`, drill 7.

### Workshops, collateral, design wins and product feedback
**24. How do you run a hands-on workshop for a mixed audience?**
Start from their use case, not the product list, and show one picture of the stack. Run three exercises on laptops
against the mock engine, so nobody waits for a GPU, and ask people to predict each result before they run it. Show the
GPU part live, with a recording as backup. They leave with the repo and the notebook.
- Proof: M12, [workshop/agenda.md](../workshop/agenda.md).

**25. How do you turn a PoC into a design win?**
Agree on success criteria up front, each measured by a named tool, in a signed PoC charter. Run on the customer's
policy and traffic shape (synthetic data), and report in their unit: dollars per 1,000 disruptions. Keep a technical
win plan (champion, veto holders, competition) and a mutual action plan with dates. End with a readout that makes a
recommendation and names the decision you need.
- Proof: M11, [sales/poc-charter.md](../sales/poc-charter.md), [sales/technical-win-plan.md](../sales/technical-win-plan.md).

**26. Tell me about a feature request you drafted.**
Engines report prefix-cache hits as global counters, or per-response `cached_tokens` only when enabled. So you cannot
tie a cache miss to the agent session or prompt change that caused it. I wrote a PRD: the problem; the impact,
quantified with the cache-busting drill (from $__ to $__ per 1,000 disruptions); a repro; a proposed API (session id on
spans and metric exemplars, cached tokens always returned, a prefix-divergence offset in debug mode); acceptance
criteria; and the alternatives.
- Proof: M11, [sales/feature-request.md](../sales/feature-request.md).

**27. A new model launches today. What do you do before the customer call?**
Size it from its config.json in 30 minutes: parameters, active parameters, KV per token, GPUs needed. Run it on the
same golden set and gates as the current model. Then brief the customer only on what we measured. 🔁 Lab 04's
Nemotron 3.5 add-on is the template.
- Proof: M2 (`archmath/`), M9 gates.

## C. Hospitality and travel beyond IROPS
Have one sentence ready for each. Name the NVIDIA piece, then the business number it moves.
| Use case | NVIDIA stack | What to say |
|---|---|---|
| Revenue management and demand forecasting | RAPIDS (cuDF, cuML), XGBoost on GPU | "Re-forecast every night's price for every room and seat in minutes, not overnight." |
| Contact-center voice AI | Speech AI (ASR, TTS) + an LLM on NIM under a latency budget | 🔁 your real work: a voice AI system architected for 500K+ calls a month for a Fortune 500 health plan |
| Multilingual guest messaging | Nemotron on NIM + NeMo Guardrails | "One agent, every guest language, the same policy." |
| Digital humans at front desks and kiosks | NVIDIA ACE | "A concierge that speaks, at 3 a.m., in nine countries." |
| Travel-document and receipt intelligence | VLMs (M8) + NeMo Retriever extraction | "Read the receipt, not the guest's description of it." |
| Personalized offers | Merlin recommenders | "Ancillaries that fit this trip, not last year's." |
| Crew, fleet and rail scheduling | cuOpt | "Re-plan crews after a storm in minutes." |
| Loyalty fraud and abuse | RAPIDS, graph analytics | "Find points rings before the statement goes out." |

## D. A note on the posting
The posting names the Hospitality and Travel team ("hotel and transportation chains"), and one sentence also says
"Auto customers". That is probably template text. Do not point it out in the application. If the team also covers
automotive or transportation, your FCA (Stellantis) plant-floor ML work and your work with a Fortune 50 automaker at
Microsoft are directly relevant: say so plainly.

## E. Numbers to have ready (replace with yours)
| Number | Lab estimate or placeholder | Where it comes from |
|---|---|---|
| LLM calls per task | ~6.5 (4-12) | oracle trajectories, `ttg.world` |
| Prompt tokens per task | ~15,100 (p95 ~25,000) | oracle trajectories, characters / 4 |
| Ideal prefix-cache hit ratio | ~96% | same |
| Measured cache hit ratio, best engine | __ | `results/bench/*.json` |
| Tasks per GPU-hour, best configuration | __ | ledger |
| $ per 1,000 disruptions at the 12 s SLO | __ | ledger |
| Storm-day GPUs (US + EU, N+1) | __ + __ | [docs/09](09-results-and-exec-readout.md) section 5 |
| Llama 3.1 8B decode floor, H100 SXM, batch 1 | ~4.6 ms per token (BF16), ~2.5 ms (FP8) | ~15 GB streamed (16.1 GB minus the gathered embedding) / 3.35 TB/s; `profiling/roofline_check.py` |

**Checkpoint:** you can answer every question in section B in under 90 seconds without notes, and each answer names
the artifact that proves it.

**Say it:** "I don't quote a benchmark. I measure your traffic on your policy, show you the cost per decision at your
latency target, and tell you which optimization bought what."
