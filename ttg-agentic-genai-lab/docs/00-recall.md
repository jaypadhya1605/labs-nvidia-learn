# Labs 01-04 recall - spaced repetition for Lab 05

Before each Lab 05 module, answer its recall questions out loud **without notes**, then check the answers at the
bottom. Lab 05 re-runs earlier concepts on a new kind of traffic (an agent) on purpose. Every 🔁 in this lab marks
one of these concepts coming back for another rep.

## Where Labs 01-04 (and the Oct 17 mock round) show up again
| Concept (where you learned it) | Lab 05 module | What is different this time |
|---|---|---|
| Decode = bytes / bandwidth; prefill = FLOPs (Lab 01 vLLM sweeps on an L40S, Lab 02 M3) | M2, M5 | per architecture: dense, GQA, MLA, MoE, hybrid Mamba, VLM |
| KV-cache bytes per token, GQA (Lab 02, mock round Q1) | M2, M5, M7 | MLA and hybrid state; KV capacity caps how many agents run at once |
| Roofline + CUDA bandwidth probe (Lab 01, Lab 04 M3) | M5 | the profiler shows where the time actually goes, not just the floor |
| AIPerf sweeps, goodput, operating point (Lab 02 M7, Lab 04 M5) | M4 | multi-turn agent replay with one shared system prompt |
| NIM profiles; NIM vs trtllm-serve (Lab 04 M6) | M4 | four engines, one prompt protocol, task-level metrics |
| LoRA adapter + release gate (Lab 04 M8) | M6, M9 | "prompt baking" with TRL + PEFT; the gate now runs in GitLab |
| MIG partitions, Dynamo disaggregated serving (Lab 02) | M7, drill 7 | parallelism math; why NVL72 makes one big NVLink domain |
| Guardrails, prompt injection, tool output is untrusted (Lab 03) | M3, M13 | MCP tool results are data, never instructions; money needs approval |
| Multi-cloud NIM, GitOps, External Secrets (Lab 04 M3-M4) | M9 | same AKS/GKE clusters, a new chart, GitLab CI instead of GitHub Actions |
| Perf gate (Lab 02) and LoRA release gate (Lab 04) | M9 | eval gate (task success) + perf gate (p95, $ per 1,000 tasks) |
| Cold-start timeline (Lab 02 M9, Lab 04 M7) | M5 | I/O: weight load from network vs local NVMe vs a pre-filled cache |
| Prior-auth: evidence in, policy-grounded decision out (Labs 02-04) | M3 | Tern Care Commitments: flight + tier + policy in, rebooking + vouchers out |
| PHI sovereignty (Labs 02-04) | M1, M9 | GDPR: EU guests stay in the EU region; PCI: payment tokens only |
| $ per 1,000 summaries (Labs 02-04) | M10 | $ per 1,000 disruptions and tasks per GPU-hour |
| Break-fix drills + 5-part readout (Labs 02-04) | M13, [docs/09](09-results-and-exec-readout.md) | agent drills: cache busting, verbose thoughts, context too small |
| Mock round (Oct 17): 70B sizing, NIM pod debugging, NIM YAML, TTFT/ITL client, benchmark knee | M2, M4, M7, M9 | the same questions, asked about agent traffic |

## Recall questions
1. (M2) What is the batch-1 decode floor for Llama 3.1 8B BF16 on an A100 80GB PCIe? On an H100 SXM?
2. (M2) KV bytes per token for Llama 3.1 8B BF16? For Llama 3.1 70B?
3. (M2) What is decode's arithmetic intensity at batch 1, and what is the H100's BF16 ridge point?
4. (M5) Prefill or decode: which one is compute-bound, and what is its FLOP count per token?
5. (M7) Llama 3.1 70B FP8 on 2x H100 80GB with tensor parallelism 2: how many 8K-token sequences fit? On 1x H100?
6. (M7) Best-case decode speed for that 2x H100 FP8 setup, per stream?
7. (M4) What is AIPerf goodput, and how do you pick the operating point of a sweep?
8. (M4) In the mock round's benchmark, throughput went flat, ITL stayed flat and TTFT exploded. What was happening?
9. (M4) What is a NIM profile, and why is it per GPU AND per GPU count?
10. (M4) Why did the mock round's Python client start the TTFT timer only after the semaphore?
11. (M6) What does a release gate compare, and when do you re-baseline?
12. (M7) What does MIG isolate that time-slicing does not?
13. (M7) How many GPUs share one NVLink domain in a DGX H100, and at what bandwidth per GPU?
14. (M5) Why can't you copy a TensorRT plan from an A100 to another GPU?
15. (M9) Why does a one-GPU engine Deployment use `strategy: Recreate`?
16. (M9) Why does a NIM pod need a startupProbe, not just a livenessProbe?
17. (M9) Which signal should autoscale an LLM engine, and why not GPU utilization?
18. (M3) Why is tool output "untrusted data", and how did Lab 03 measure it?
19. (M10) Turn a benchmark into $ per 1,000 tasks. What is the formula?
20. (M13) Name the five parts of the one-page executive readout.

## Answers
1. 16.06 GB / 1,935 GB/s ≈ 8.3 ms per token (~120 tokens/s). H100 SXM: 16.06 GB / 3,350 GB/s ≈ 4.8 ms (~208 tokens/s).
2. 2 (K and V) x 32 layers x 8 KV heads x 128 x 2 bytes = 128 KiB. 70B: 2 x 80 x 8 x 128 x 2 bytes = 320 KiB.
3. About 1 FLOP per byte (2 FLOPs per 2-byte weight). H100 SXM BF16: 989 TFLOPS / 3.35 TB/s ≈ 295 FLOPs per byte.
   That gap is why batching is the lever.
4. Prefill is compute-bound: about 2 x active parameters FLOPs per token, plus attention. Decode is bandwidth-bound.
5. About 30 sequences (about 60 with an FP8 KV cache). One H100 FP8 fits only 2: weights take 70.6 GB of 80 GB.
6. Each GPU reads its 35.3 GB half of the weights per step at 3.35 TB/s: about 10.5 ms per token, ~95 tokens/s per stream.
7. Goodput = requests per second that met every SLO you passed. Operating point = the highest load level whose p95
   still meets the SLO. Never sell the plateau.
8. Queueing. The replica was saturated (prefill compute or KV capacity). Each token was not slower; requests waited.
   Check KV-cache usage and waiting requests to tell compute from memory.
9. A pre-built engine configuration: engine, precision, TP/PP, LoRA. Kernels and memory plans depend on the GPU and on
   how many GPUs share the model.
10. Otherwise TTFT includes the client's own queueing, and you blame the server for your semaphore.
11. Candidate vs baseline on the metrics that matter (quality, latency, tokens) with tolerances. Re-baseline
    deliberately, never automatically.
12. SMs, L2 cache and memory bandwidth, in hardware. Time-slicing only interleaves contexts on shared hardware.
13. Eight H100s through NVSwitch, NVLink 4 at 900 GB/s bidirectional per GPU.
14. TensorRT plans are compiled for one GPU architecture and TensorRT version. Rebuild per GPU, or build with a
    hardware-compatibility level and accept some speed cost.
15. A rolling update would schedule a second pod that waits forever for a second GPU.
16. The first start downloads the model. Without a startupProbe, the livenessProbe kills the pod mid-download.
17. Queue depth (for example `vllm:num_requests_waiting`). It measures unmet demand. GPU utilization reads ~100% long
    before latency breaks.
18. Tool output can carry attacker-written text that the model may obey. Lab 03 planted injections in tool results and
    measured how often the final action followed them.
19. $/hour / (SLO-compliant tasks per second x 3,600) x 1,000. Use the operating point, not the peak.
20. Situation, what we tested, results table, recommendation, risks and next steps.

**Checkpoint:** you answered at least 16 of the 20 without notes. Re-read the lab behind every miss before its Lab 05
module.

**Say it:** "Agent traffic is the same physics I measured in Labs 01-04 - bytes per token, FLOPs per prompt, queueing at
the knee - with one new twist: every step resends the conversation."
