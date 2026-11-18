# Perf-gate baselines

The GitLab `perf-gate` job runs on the GPU runner, manually or on a schedule. It benchmarks the agent on one engine and
compares the result with a **committed baseline** in this folder. The rule is the same as 🔁 Lab 04's perf gate:
baselines move on purpose, never automatically.

| File | Engine / variant | GPU (`nvidia-smi` name) | Levels |
|---|---|---|---|
| `vllm-base-h100.json` | vLLM `base` (BF16, prefix caching on) | NVIDIA H100 80GB HBM3 | 1, 4, 16 |

The folder starts empty. The first `perf-gate` run exits with code 2 and prints the command that records a baseline.

## What the gate checks
`scripts/gate.py perf` compares every concurrency level that appears in BOTH files:
- **success_rate >= 0.95.** A faster engine that breaks the Tern Care policy fails, whatever its latency.
- **p95 task latency** may regress by at most **10%**.
- **$ per 1,000 disruptions** may regress by at most **10%**. This check is skipped when either file has
  `gpu_hourly_usd` 0.
- **Same GPU.** If the candidate ran on a different `gpu` than the baseline, that is an input error (exit 2), not a
  regression. Record one baseline per GPU.

## Record or re-record a baseline
The safest source is the perf-gate job itself, because it uses the same runner, flags and GPU name as every later
comparison:
1. Run `perf-gate` (CI/CD > Pipelines > play). Download its artifact `results/bench/vllm-base.json`.
2. Turn it into a baseline. Give a reason, and note that the per-task rows are dropped so the diff stays reviewable:
   ```bash
   python scripts/gate.py baseline results/bench/vllm-base.json --out bench/baselines/vllm-base-h100.json \
     --reason "first baseline: vLLM v0.31.0, H100 SXM, agent prompt v1"
   ```
3. Commit it in its **own merge request**. The reviewer reads the `baseline.reason` field and the numbers.

You can also record by hand on the GPU VM, with exactly the flags the job uses:
```bash
engines/launch.sh vllm base
python -m ttg.bench.agent_bench --label vllm-base --engine vllm --levels 1,4,16 --n 48 \
  --gpu-name "$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)" --gpu-hourly "$GPU_HOURLY_USD" --gpus 1 \
  --metrics-url http://localhost:8000/metrics --reset-url http://localhost:8000/reset_prefix_cache
```

**Re-record when** you changed something on purpose and the new numbers are the new normal: an engine upgrade, a new
GPU, a new prompt version. **Never re-record** to turn a red gate green. Find the regression first (M5 profiling).
