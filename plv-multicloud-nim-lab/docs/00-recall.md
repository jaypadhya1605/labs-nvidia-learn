# Labs 02 + 03 recall - spaced repetition for Lab 04

Before each Lab 04 module, answer its recall questions out loud **without notes**, then check. Lab 04 deliberately
re-runs Lab 02/03 concepts on new clouds (🔁 marks them everywhere) - the second and third reps are what make them stick.

## Where Labs 02 and 03 show up again
| Concept (where you learned it) | Lab 04 module | What is different this time |
|---|---|---|
| Decode = bytes / bandwidth; prefill = FLOPs (L02 M3) | M2, M3, add-on A1 | selector across 4 clouds + roofline + hybrid Mamba-MoE state |
| CUDA bandwidth probe (L02 M3) | M3 | one Job for every GPU (`-arch=native`) |
| GPU Operator, operator-managed driver on AKS (L02 M2) | M3-M4 | + GKE driver installer, EKS/OKE AMIs, host driver in the DC |
| NIMCache filter, NIMService, profiles, groupID 0 (L02 M4) | M4, M9 | same chart on 5 platforms; plain Deployment in the air gap |
| MIG geometries + noisy neighbor (L02 M8) | M8 | train on one 3g.40gb slice while serving on the other |
| Prometheus rules, Grafana, DCGM energy (L02 M6) | M4-M5 | same rules (`plv:`), per-cluster, live $/1M tokens panel |
| AIPerf sweeps, goodput, $/1k (L02 M7, L03 M9) | M5, M6, A4 | per cloud; measured vs predicted |
| HPA on queue depth, cold start timeline (L02 M9) | M7 | scale-from-zero on each cloud's autoscaler |
| Triton + TensorRT engine (L02 M5) | M6 | engines are per GPU: the portability drill |
| Perf gate (L02) / safety gate (L03) | M8 | LoRA release gate - same shape, model artifact |
| Prompt-injection eval (L03 M10) | M8 agentic suite, A3 | model-level, ranks models on day 0 |
| Network policy as the last control (L03) | M4, M12 drill 9 | per-namespace grants in Git |
| Break-fix drills (L02/L03) | M12 | multi-cloud + GitOps self-heal |
| 5-part exec readout (L02/L03) | M13 | audience: ISV VP Eng + CFO + CSP partners |

## Recall questions
1. (M2) Llama 3.1 8B BF16 decode floor at batch 1 on an A100 80GB PCIe, and on an L4?
2. (M2) KV bytes per token for Llama 3.1 8B BF16 - and what is it for Nemotron 3.5 Lightning?
3. (M3) Why does the AKS GPU pool use `gpu_driver = "None"`, and what must the GPU Operator then own?
4. (M3) What does the bandwidth probe's read-only kernel model, and why use ~35% of free memory for buffers?
5. (M4) What does a NIMCache filter (engine, precision, TP, LoRA) buy you?
6. (M4) Why does a NIM need `groupID: 0`?
7. (M4) What is a NIM profile, and why is it per GPU AND per GPU count?
8. (M5) What is AIPerf goodput, and how do you pick the operating point of a sweep?
9. (M5) How do you turn a sweep into $ per 1,000 summaries?
10. (M5) Why `max by (UUID)` before summing DCGM power under MIG?
11. (M6) Why can't you copy a TensorRT plan from an A100 to an RTX PRO 6000?
12. (M7) What signal does the HPA scale NIM on, and why not GPU utilization?
13. (M8) What did MIG isolate in the noisy-neighbor test that time-slicing does not?
14. (M8) What does a release gate compare, and when do you re-baseline?
15. (M8) Why is tool output "untrusted data" (Lab 03), and how do you measure it?
16. (M12) Why do resets in this lab mostly mean "re-enable auto-sync"?
17. (M13) The five parts of the one-page readout?

## Answers
1. 16.06 GB / 1,935 GB/s ≈ 8.3 ms (~120 tok/s per sequence); L4: 16.06 / 300 ≈ 53 ms - vetoed for a 40 ms ITL SLO unless FP8.
2. 2 (K,V) x 32 layers x 8 KV heads x 128 x 2 B = 128 KiB. Lightning: 6 attention layers x 2 KV heads -> 6 KiB, plus ~49 MB fixed SSM state per sequence.
3. The operator owns driver + toolkit (operator-managed mode, runtimeClass `nvidia`). Never mix with AKS-managed drivers.
4. Decode streams weights/KV and writes almost nothing; buffers far larger than L2 measure HBM/GDDR, not cache.
5. Only the profile you serve is downloaded: smaller cache, faster and deterministic start.
6. NIM images update CA certificates at start and need group-0 file permissions; otherwise CrashLoopBackOff.
7. A pre-built engine configuration (engine, precision, TP/PP, LoRA). Kernels and memory plans depend on the GPU and on how many GPUs share the model.
8. Requests/s that met every SLO you passed. Operating point = highest throughput level whose p95 still meets the SLO.
9. $/hour / (SLO-compliant summaries/s x 3,600) x 1,000 (+ NVAIE $/GPU-hour if licensed hourly).
10. Device-level fields repeat on every MIG instance series; summing them overstates power.
11. TensorRT plans are compiled for a GPU architecture (and TensorRT version). Rebuild per GPU or build with a hardware-compatibility level (ampere+) and accept some speed cost.
12. Queue depth (`vllm:num_requests_waiting`): it measures unmet demand; GPU utilization is ~100% busy long before latency breaks.
13. SMs, L2 and memory bandwidth in hardware; time-slicing only interleaves contexts on shared hardware.
14. Candidate vs baseline on the metrics that matter (latency, throughput / quality, injection / schema, tokens) with tolerances; re-baseline deliberately, never automatically.
15. Tool output can contain attacker-written text that the model may obey; plant injections in tool results and measure how often the final action follows them.
16. Drills hot-patch live objects after pausing Argo CD; Git still holds the truth, so re-enabling self-heal repairs the cluster.
17. Situation, what we tested, results table, recommendation, risks + next steps.
