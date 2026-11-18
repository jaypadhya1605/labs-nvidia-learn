# 08 - Profiling playbook: latency, throughput, memory, I/O (M5, 3 h, 1x H100)

**Goal:** find the bottleneck of the Disruption Concierge on one GPU and fix it with numbers, not opinions.
Work top-down: the task first, then the GPU, then the kernels, then the bytes. Change one thing per run.

## The method - five questions, in this order
| # | Question | Tool | The number that answers it |
|---|---|---|---|
| 1 | Where does task time go? | `python -m ttg.bench.agent_bench` | LLM time vs tool time per task, LLM calls, tokens, cache-hit ratio |
| 2 | Is the GPU busy, and on what? | `profiling/dcgm_watch.sh` | SM active, tensor-pipe active, DRAM active (not `nvidia-smi` "util") |
| 3 | What runs on the GPU? | `profiling/torch_profile.sh`, `profiling/nsys_capture.sh` | GPU busy %, share per kernel category, idle gaps, top-20 kernels |
| 4 | Is decode at the roofline? | `profiling/roofline_check.py` | achieved TB/s vs peak (🔁 Lab 01) |
| 5 | How fast can a new replica serve? | `profiling/coldstart.py` | download, weight-load GB/s, compile, CUDA graph capture |

An agent task is 4-12 dependent LLM calls on a ~2K-token byte-identical prefix. So the first suspects are always
prefill of that prefix (is the cache hitting?), then decode bytes per step, then CPU time between steps.

## Commands
```bash
# 1. Task level (the line engines/launch.sh prints, with --metrics-url and --reset-url filled in)
python -m ttg.bench.agent_bench --label vllm-base --engine vllm --levels 1,4,16,32 --n 48 \
  --gpu-name H100-80GB --gpus 1 --gpu-hourly 2.49 \
  --metrics-url http://localhost:8000/metrics --reset-url http://localhost:8000/reset_prefix_cache

# 2. GPU counters next to a bench run (DCGM if installed, nvidia-smi dmon always)
profiling/dcgm_watch.sh 180 results/profiles/gpu-vllm-base &
python -m ttg.bench.agent_bench --label vllm-base-c16 --levels 16 --n 64 --out results/profiles/gpu-vllm-base/bench.json

# 3. Kernel level: PyTorch profiler (60 engine steps of real agent traffic), then nsys for the timeline
profiling/torch_profile.sh base          # -> results/profiles/torch/base/summary.json
profiling/torch_profile.sh eager         # same traffic, CUDA graphs off
profiling/nsys_capture.sh base           # -> results/profiles/nsys/base/base.nsys-rep + kern_sum.csv
python3 profiling/analyze_trace.py results/profiles/torch/base/

# 4. Roofline: floor vs measured ITL (from AIPerf on the agent's real prompts, 🔁 Labs 02-04)
python3 profiling/roofline_check.py --gpu H100-SXM --aiperf results/aiperf/c16
python3 profiling/roofline_check.py --gpu H100-SXM --sweep 1,4,16,32,64,128

# 5. Cold start: save logs WITH docker timestamps, start the same variant twice, compare
engines/logs.sh --save results/logs/vllm-base-1.log
python3 profiling/coldstart.py results/logs/vllm-base-*.log
```

## What the floor says before you measure (Llama-3.1-8B, H100 SXM, 2,600 tokens of context)
`roofline_check.py --sweep` prints these from the model config - arithmetic, not measurements:

| Batch | Bytes per decode step | Floor per step | Ceiling, tokens/s total |
|---|---|---|---|
| 1 | 15.4 GB (weights 15.0 + KV 0.34) | 4.6 ms | ~218 |
| 16 | 20.5 GB | 6.1 ms | ~2,600 |
| 64 | 36.8 GB | 11.0 ms | ~5,800 |
| 128 | 58.6 GB | 17.5 ms | ~7,300 |

Read it: decode is memory-bound at every batch size here. Weights dominate at small batch, KV reads at large batch.
FP8 weights cut the batch-1 floor to ~2.5 ms. Measured ITL always sits above the floor; the gap is what this
module explains (CPU time between steps, prefill chunks mixed into decode, sampling, small kernels).

## Experiments - one change per run, same traffic, same order
| # | Experiment | Run | Look at | What you should see (direction) |
|---|---|---|---|---|
| E1 | Prefix caching off | `engines/launch.sh vllm nocache` vs `base` | cache-hit ratio, TTFT, p95 task latency | hit ratio ~0, every call re-prefills ~2K tokens, TTFT and p95 rise |
| E2 | CUDA graphs off | `torch_profile.sh eager` vs `base` | GPU busy %, launch APIs, ITL | thousands of `cudaLaunchKernel`, more small gaps, higher ITL at low batch |
| E3 | FP8 weights | `engines/launch.sh vllm fp8` | ITL, roofline fraction, success rate | lower ITL at small batch (half the weight bytes); success rate must not drop |
| E4 | FP8 KV cache | `fp8kv` | `GPU KV cache size` in the log, p95 at c=32 | about 2x KV tokens; helps when the queue, not the step, is the problem |
| E5 | N-gram speculative decoding | `spec` | `vllm:spec_decode_num_accepted_tokens / ..._draft_tokens`, ITL | tool-call JSON copies ids from the prompt, so drafts get accepted; gain shrinks at high batch |
| E6 | Concurrency sweep | `--levels 1,4,16,32,64` | tasks/s, p95, `vllm:num_requests_waiting` | the knee where p95 crosses the 12 s SLO = your per-GPU capacity |
| E7 | Verbose thoughts | `TTG_THOUGHTS=verbose TTG_MAX_TOKENS=1024` | completion tokens per task, p95 | on the mock: ~8x output tokens, ~4x p95; with 256 max tokens the JSON truncates (0% success) |
| E8 | Cache-busting prompt | `TTG_BUST_CACHE=1` | cache-hit ratio | a timestamp at the top of the system prompt kills every prefix hit |
| E9 | Prompt baking (M6) | `engines/launch.sh vllm lora`, `TTG_PROMPT_VARIANT=short` | prompt tokens per call, KV bytes per step | ~1.7K fewer tokens per call: fewer KV bytes read every decode step, cheaper cold-cache prefill |
| E10 | Cold start, twice | `coldstart.py` on both logs | download s, load GiB/s, compile s, graph capture s | second start: no download, torch.compile cache reused (VLLM_CACHE_HOST) |

Record each run's bench file, profile summary and log. The M10 ledger turns them into the before/after story.

## Reading the profile (analyze_trace.py output)
- **GPU busy % below ~70%:** the GPU waits on the CPU - scheduler, Python, kernel launches - or on too little work.
  Check that CUDA graphs are replaying (`cudaGraphLaunch` count) and raise concurrency.
- **GEMM above ~50%:** weight streaming dominates decode. FP8 weights halve those bytes.
- **Attention + KV above ~30%:** long contexts. Prefix caching, FP8 KV and shorter prompts (M6) all cut it.
- **Communication above ~10% (tp2):** the all-reduce is on the critical path - NVLink vs PCIe (M7).
- **Many gaps under 10 us, busy above 90%:** back-to-back kernels. Only fewer bytes help now (roofline).
- The categories come from kernel names. Read the top-20 table before you quote a share.
- vLLM also writes its own CUDA-time table next to the trace (`torch_profiler_dump_cuda_time_total`, on by default).

## Memory and I/O checks
- KV capacity: `GPU KV cache size: N tokens, Maximum concurrency for 16,384 tokens per request: Kx` (vLLM log),
  `KV Cache is allocated ... #tokens: N` (SGLang). `roofline_check.py` shows when a batch no longer fits.
- Prefix caching stores the shared system prompt once, so capacity rises - but every sequence still reads it.
- Weight load: `coldstart.py` reports GiB/s. Below ~1.5 GiB/s the disk is the bottleneck. Try local NVMe or
  `--load-format runai_streamer --model-loader-extra-config '{"concurrency":16}'` (needs `vllm[runai]`).
- PCIe: `dcgm_watch.sh` fields 1009/1010. High RX during decode means something copies host to device per step.

## Checkpoint
1. nvidia-smi says 100% while tensor-pipe active is 15%. Is the GPU saturated? What does DRAM active tell you?
2. Why does the eager variant hurt ITL more at batch 1 than at batch 32?
3. The roofline check says 92% of peak bandwidth at batch 4. Name two changes that still make decode faster.
4. Prefix caching is on, yet the hit ratio is 0.3. Name two causes (hint: E8, and cache size under load).
5. A new replica takes 4 minutes to serve. Which three numbers from `coldstart.py` do you fix first, and how?

## Say it
- "Your GPU looks 100% busy, but tensor cores are idle most of the time - decode is waiting on memory, not math."
- "We profiled sixty engine steps of your real agent traffic. Most GPU time is weight streaming, so FP8 is the
  first lever - and we ship it only if the golden-set success rate holds."
- "Prefix caching makes your 2,000-token policy prompt nearly free after the first call - one timestamp in that
  prompt would undo it."
- "Before we add GPUs for the storm, we make each new replica start faster: warm weights on local NVMe and a
  reused compile cache, measured line by line from the start-up log."
