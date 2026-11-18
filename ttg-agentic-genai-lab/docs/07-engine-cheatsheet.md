# 07 - Engine cheat sheet: vLLM vs SGLang vs TensorRT-LLM vs NIM vs Dynamo (M4, M5)

**The answer first:** for an agent, the engine feature that matters most is how well it reuses the shared prompt
prefix; after that, how it batches many short decode steps. All four engines here do continuous batching, paged KV
memory, CUDA graphs and chunked prefill. They differ in how they manage and share the KV cache, which quantized
checkpoints they load, and how much of the stack NVIDIA supports for you. Flags below were checked on 2026-10-17
against vLLM 0.31.0, SGLang 0.5.22, the trtllm-serve 1.4.0rc0 docs and AIPerf 0.13.0 (see the contract in the README).

## What each one is
| | vLLM | SGLang | TensorRT-LLM (`trtllm-serve`) | NIM for LLMs | Dynamo |
|---|---|---|---|---|---|
| What it is | Open-source serving engine | Open-source serving engine + frontend language | NVIDIA's LLM inference library; PyTorch backend + OpenAI server | NVIDIA's packaged, supported microservice | NVIDIA's distributed serving framework |
| KV memory | PagedAttention: fixed-size blocks, block table per request | RadixAttention: a radix tree of token sequences, LRU eviction | Paged KV cache with block reuse | Whatever the selected profile's engine uses | Orchestrates engines; KV-aware routing, KV offload to CPU/SSD |
| Prefix caching | Automatic prefix caching, hash per block (`--enable-prefix-caching`) | On by default (`--disable-radix-cache` turns it off) | `kv_cache_config.enable_block_reuse` | `NIM_ENABLE_KV_CACHE_REUSE` (verify for your tag) | Routes requests to the worker that already holds their prefix |
| You choose it when | Broadest model coverage, fastest day-0 support, big community | Heavy prefix sharing, agents, structured output, fast scheduler | Peak tokens per GPU on NVIDIA hardware, FP8/NVFP4 ModelOpt paths | Enterprise support, security patches, one container per model, per-GPU tuned profiles | Multi-node, disaggregated prefill/decode, many replicas |

🔁 Lab 04 M6 ran NIM vs trtllm-serve vs Triton on one prompt shape. 🔁 Lab 02 used Dynamo for disaggregated serving.

## The flags this lab uses (engines/launch.sh)
| Capability | vLLM 0.31 (`vllm serve MODEL`) | SGLang 0.5.22 (`python -m sglang.launch_server`) | trtllm-serve (`trtllm-serve serve MODEL`) |
|---|---|---|---|
| Name the model | `--served-model-name ttg-llm` | `--served-model-name ttg-llm` | `--served_model_name ttg-llm` |
| Context window | `--max-model-len 16384` | `--context-length 16384` | `--max_seq_len 16384` |
| GPU memory for KV | `--gpu-memory-utilization 0.90` | `--mem-fraction-static 0.85` | YAML `kv_cache_config.free_gpu_memory_fraction: 0.85` |
| Prefix caching off | `--no-enable-prefix-caching` | `--disable-radix-cache` | YAML `enable_block_reuse: false` |
| Cached tokens in usage | `--enable-prompt-tokens-details --enable-force-include-usage` | `--enable-cache-report` | iteration stats on `/metrics` |
| FP8 weights | ModelOpt FP8 checkpoint, or `--quantization fp8` (online) | checkpoint, or `--quantization fp8` | ModelOpt FP8 checkpoint only |
| FP8 KV cache | `--kv-cache-dtype fp8` | `--kv-cache-dtype fp8_e4m3` | `kv_cache_config: {dtype: fp8}` in the options YAML |
| No CUDA graphs | `--enforce-eager` | `--disable-cuda-graph` | YAML `cuda_graph_config: null` (verify) |
| Speculative decoding | `--speculative-config '{"method":"ngram",...}'` (also `eagle3`, `draft_model`, `mtp`, `suffix`) | `--speculative-algorithm NGRAM\|EAGLE\|EAGLE3` | YAML `speculative_config` (NGram, Eagle3; verify keys) |
| Structured output | xgrammar via `response_format` (`--structured-outputs-config` to tune) | `--grammar-backend` (xgrammar) | `guided_decoding_backend` |
| LoRA | `--enable-lora --lora-modules ttg-short=/adapters/ttg-short --max-lora-rank 16`; request model `ttg-short` | `--enable-lora --lora-paths ttg-short=/adapters/ttg-short`; request model `ttg-llm:ttg-short` | LLM API `lora_config` (stretch) |
| Tensor / pipeline / data parallel | `--tensor-parallel-size`, `--pipeline-parallel-size`, `--data-parallel-size`, `--enable-expert-parallel` | `--tp-size`, `--pp-size`, `--dp-size`, `--ep-size` | `--tp_size`, `--pp_size`, `--ep_size`, `--cp_size` |
| Native tool calling | `--enable-auto-tool-choice --tool-call-parser llama3_json` | `--tool-call-parser llama3` | `--tool_parser` |

Speculative decoding for agents: n-gram (prompt lookup) drafts tokens by copying from the prompt, and tool-call
JSON copies flight numbers and ids from the observations - cheap wins. vLLM's `suffix` method (needs the
`arctic-inference` package) also learns from earlier responses, which suits repetitive agent loops - a stretch.

Why this lab's agent does NOT use native tool calling: parsers differ per engine and per model, so the bake-off
would compare parsers, not engines. The agent speaks one JSON object per turn and validates it itself
(`ttg/agent/protocol.py`). Native tool calling is an M4 stretch.

## Metrics to scrape (`python -m ttg.bench.scrape_metrics --url ...`)
| Question | vLLM `/metrics` | SGLang `/metrics` | NIM `/v1/metrics` | trtllm-serve `/metrics` |
|---|---|---|---|---|
| Prefix-cache hit rate | `vllm:prefix_cache_hits_total / vllm:prefix_cache_queries_total` (+ `vllm:prompt_tokens_cached`) | `sglang:cache_hit_rate` | engine-dependent names (vLLM-style for vLLM profiles) | JSON iteration stats (`enable_iter_perf_stats: true`) |
| Queue depth (autoscale on this) | `vllm:num_requests_waiting` | `sglang:num_queue_reqs` | same as the profile's engine | iteration stats |
| KV memory pressure | `vllm:kv_cache_usage_perc`, `vllm:num_preemptions` | `sglang:token_usage` | | |
| Latency histograms | `vllm:time_to_first_token_seconds`, `vllm:inter_token_latency_seconds`, `vllm:request_queue_time_seconds` | `sglang:time_to_first_token_seconds`, `sglang:inter_token_latency_seconds`, `sglang:queue_time_seconds` | | |
| Speculative acceptance | `vllm:spec_decode_num_accepted_tokens / vllm:spec_decode_num_draft_tokens` | `sglang:spec_accept_rate`, `sglang:spec_accept_length` | | |

Counters appear with a `_total` suffix in the Prometheus text format. Snapshot before and after a run and subtract.

## Quantization paths in one breath
- **FP8 (Hopper, Ada, Blackwell):** halves weight bytes, so the decode floor roughly halves (🔁 Lab 01 roofline). Online
  FP8 (`--quantization fp8`) is convenient; a ModelOpt checkpoint (`nvidia/Llama-3.1-8B-Instruct-FP8`) is calibrated
  and loads the same on vLLM, SGLang and TensorRT-LLM - use it for a fair bake-off.
- **NVFP4 (Blackwell only):** 4-bit weights with FP8 scales per 16 values; needs an NVFP4 checkpoint (🔁 Lab 04 GKE RTX
  PRO 6000).
- **FP8 KV cache:** halves KV bytes per token, so twice as many concurrent agent conversations fit.
- **Always re-run the quality gate** (the golden-set success rate) after quantizing: an agent that issues the wrong
  voucher amount is not faster, it is wrong faster.

## What to recommend - the Say it lines
- **Agent with a long shared system prompt, many short steps:** "Prefix caching is your biggest lever. SGLang's radix
  cache and vLLM's automatic prefix caching both compute your 2,000-token instructions once; keep that prefix
  byte-identical and you pay for it once per GPU, not once per call."
- **Peak throughput per GPU on NVIDIA hardware, stable model:** "TensorRT-LLM with an FP8 ModelOpt checkpoint usually
  gives the most tokens per GPU-hour; we measure it on your traffic, not on a benchmark."
- **Enterprise, regulated, many teams:** "NIM gives you the same engines as supported, security-patched containers with
  profiles tuned per GPU - you buy back the engineering time, and it runs in your own cloud tenant."
- **Many replicas, multi-node, long prompts:** "Dynamo adds KV-aware routing and disaggregated prefill and decode on
  top of these engines, so a cache hit on one GPU is not a cache miss on its neighbor."
- **Day-0 model:** "vLLM and SGLang usually run a new open model the day it ships; NIM and TensorRT-LLM follow with
  tuned profiles - start on one, move when the tuned profile lands."

## Checkpoint
1. Why does an agent benefit more from prefix caching than a single-turn chatbot? (Hint: 7 calls per task share the
   same 2K-token prefix, and each call extends the previous one.)
2. Which three flags make vLLM report `cached_tokens` per request, and why does the bench want them?
3. Why does the lab compare FP8 with the same ModelOpt checkpoint on every engine instead of online FP8?
4. Where does the speculative-decoding acceptance rate come from on vLLM and on SGLang?
5. A customer insists on native tool calling. What changes in the bake-off, and what must you hold constant?
