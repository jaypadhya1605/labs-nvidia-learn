# engines/ - one model, five serving stacks, one fair bake-off (M4, M5, M7, M8)

`engines/launch.sh <engine> <variant>` starts exactly one engine as container `ttg-engine` on port 8000. Every engine
serves the same checkpoint under the same name (`ttg-llm`), so the agent, its byte-identical prompts and the bench
never change between runs - only the engine does. 🔁 Lab 04 M6 compared NIM vs trtllm-serve vs Triton on one
prior-auth prompt; this time the traffic is a multi-step agent, where the shared prefix dominates.

| Engine | Image (env/pins.env) | Health | Metrics | Variants |
|---|---|---|---|---|
| vllm | `vllm/vllm-openai:v0.31.0` | `/health` | `/metrics` (`vllm:prefix_cache_hits`, `vllm:num_requests_waiting`) | all |
| sglang | `lmsysorg/sglang` (pinned tag) | `/health` | `/metrics` with `--enable-metrics` (`sglang:cache_hit_rate`) | all but vlm |
| trtllm | `nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc29` | `/health` | `/metrics` JSON (`enable_iter_perf_stats`) | base, nocache, fp8, fp8kv, eager, tp2, pp2 |
| nim | `nvcr.io/nim/meta/llama-3.1-8b-instruct:2.0.13` | `/v1/health/ready` | `/v1/metrics` | base, nocache, tp2, fp8 (needs `NIM_MODEL_PROFILE`) |
| hf | naive transformers server in the vLLM image | `/health` | none | base |

Unsupported combinations fail fast with the reason (for example `trtllm spec` points you at the stretch YAML).
`DRY_RUN=1` prints the exact docker command; `engines/preflight.sh <engine>` greps every flag it uses in the pinned
image's `--help` before you rent a GPU.

## The bake-off procedure (M4) - same GPU, same model, same prompts, same order
1. **Preflight on the laptop:** `engines/preflight.sh vllm` (and sglang, trtllm). Fix any `MISSING` before renting.
2. **One GPU VM, one engine at a time:** `engines/launch.sh <engine> base`. `launch.sh` removes the previous container,
   waits for health, writes `results/engine-current.json`, and prints the `export TTG_BASE_URL=... TTG_MODEL=...`
   line plus the exact bench command for that engine (metrics and reset URLs filled in).
   The port is published on `127.0.0.1` only (`BIND`): from the laptop use `ssh -L 8000:localhost:8000 <vm>`.
3. **Warm up, then measure:** the first requests compile kernels and capture CUDA graphs. The bench's `--warmup N`
   runs N tasks first and discards them.
4. **Measure the agent, not just the engine** (the line `launch.sh` printed, for example):
   `python -m ttg.bench.agent_bench --label vllm-base --engine vllm --levels 1,4,16,32 --n 48 --gpu-name H100-80GB --gpus 1 --gpu-hourly 2.49 --metrics-url http://localhost:8000/metrics --reset-url http://localhost:8000/reset_prefix_cache`
   -> `results/bench/vllm-base.json`: task p50/p95 latency, LLM calls, tokens, success rate, cache-hit ratio,
   $ per 1,000 tasks. `--reset-url` empties the prefix cache before each concurrency level, so an earlier level
   cannot inflate the hit rate of the next one. Reset endpoints: vLLM `POST /reset_prefix_cache` (launch.sh sets
   `VLLM_SERVER_DEV_MODE=1` - lab only, it also exposes `/sleep` and `/collective_rpc`), SGLang `/flush_cache`,
   the mock `POST /reset`, hf a no-op. trtllm-serve and NIM have none: relaunch for a cold cache.
5. **Measure the engine with AIPerf on the agent's real prompts** (🔁 Labs 02-04):
   `python -m ttg.bench.export_aiperf --n 64 --out results/aiperf` then replay `sessions.jsonl` with
   `--custom-dataset-type multi_turn --system-prompt-file results/aiperf/system_prompt.txt` (commands in the README, M4).
6. **Record the cache:** `python -m ttg.bench.scrape_metrics --url http://localhost:8000/metrics` before and after.
7. **Save the logs** for the cold-start numbers: `engines/logs.sh --save results/logs/<label>.log`, then
   `python3 profiling/coldstart.py results/logs/*.log`.
8. **Stop:** `engines/stop.sh`. Next engine. Keep the order the same on every GPU you test.
9. **Ledger (M10):** `python -m ttg.bench.ledger results/bench/*.json --order hf-base,vllm-nocache,vllm-base,... --out results/ledger`

What to record per run: engine + image tag, variant, GPU + driver (`nvidia-smi`), model + precision, the bench file,
the AIPerf export, the metrics snapshot. The M10 ledger reads the bench files; everything else is evidence.
`hf base` serves one request at a time, so its bench line uses `--levels 1,4 --n 8` (bigger runs only time out).

## Variants at a glance
| Variant | vLLM | SGLang | TensorRT-LLM | What it teaches |
|---|---|---|---|---|
| base | `--enable-prefix-caching` | RadixAttention (default) | `kv_cache_config.enable_block_reuse: true` | the agent prefix is computed once |
| nocache | `--no-enable-prefix-caching` | `--disable-radix-cache` | `enable_block_reuse: false` | what prefix caching is worth for agents |
| fp8 | FP8 checkpoint (or `FP8_MODE=online` -> `--quantization fp8`) | same | ModelOpt FP8 checkpoint | half the weight bytes per decode step |
| fp8kv | `--kv-cache-dtype fp8` | `--kv-cache-dtype fp8_e4m3` | `kv_cache_config.dtype: fp8` (`trtllm/fp8kv.yaml`) | twice the KV capacity |
| spec | `--speculative-config` n-gram | `--speculative-algorithm NGRAM` | stretch | tool-call JSON copies ids from the prompt |
| eager | `--enforce-eager` | `--disable-cuda-graph` | `cuda_graph_config: null` (verify) | CPU launch overhead in decode |
| lora | `--enable-lora --lora-modules ttg-short=...` | `--lora-paths ttg-short=...` (model `ttg-llm:ttg-short`) | - | M6 prompt-baked adapter |
| tp2 / pp2 / dp2 | `--tensor/pipeline/data-parallel-size N` | `--tp-size/--pp-size/--dp-size N` | `--tp_size/--pp_size N` | M7 parallelism |
| vlm | Qwen2.5-VL-7B, `--limit-mm-per-prompt` | - | - | M8 image tokens |

`hf base` is the "before" picture for the ledger: one request at a time, no paged KV, no prefix cache.
