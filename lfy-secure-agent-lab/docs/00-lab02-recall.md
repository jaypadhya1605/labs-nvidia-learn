# Lab 02 recall - spaced repetition for Lab 03

Before each Lab 03 module, answer its recall question out loud **without notes**, then check. If you miss one, re-read that
Lab 02 module tonight. Repetition is the design: Lab 03 reuses Lab 02's platform so every concept gets a second rep.

## Where Lab 02 shows up again
| Lab 02 concept | Lab 03 module | What's different this time |
|---|---|---|
| Bicep landing zone, Spot GPU pool, `--gpu-driver none` | M1 | + ACR, + Cilium network policy |
| GPU Operator, MIG all-balanced, DCGM | M2 | identical scripts |
| Decode is bandwidth-bound; prefill is compute-bound | M3, M9 | now explains the *safety tax* |
| NIMCache -> NIMService, PB6 groupID 0, profiles | M4 | four models, two of them safety models |
| Prediction C: 8B BF16 on a 2g slice leaves ~2 GiB KV | M4, M9 | the content-safety model lives exactly there |
| Prediction B: prefix caching | M4, M9 | agent loops resend the same system prompt every step |
| Namespaces + ResourceQuota on MIG resources | M4, M13 | tenants are now "agent" and "safety" |
| Prometheus, recording rules, Grafana, Azure remote write | M6, M9 | + guardrail and agent metrics |
| AIPerf sweeps, goodput, energy + $ per 1,000 | M9 | direct vs. guarded = the price of safety |
| Noisy-neighbor isolation | M9 | red-team flood on the safety slice vs. the agent slice |
| CI perf gate | M8 | becomes a *safety* release gate |
| Break-fix drills | M13 | drills 1-4 are Lab 02's |
| Executive readout, "5 sentences" | M14 | new audience: an ISV's CTO and CISO |

## Recall questions (answers below)
1. (M1) Why does the GPU pool use `--gpu-driver none`, and what must the GPU Operator then enable?
2. (M2) What does the `nvidia.com/mig.config` label do, and how do you know reconfiguration finished?
3. (M3) Why does a 3g.40gb slice get about half the full GPU's memory bandwidth but only three-sevenths of its compute?
4. (M3) Llama-3.1-8B BF16 decode floor at batch 1 on a full A100?
5. (M4) What does a NIMCache filter (engine, precision, TP, LoRA) buy you?
6. (M4) Why does a PB6 NIM need `groupID: 0`?
7. (M4) KV-cache bytes per token for Llama-3.1-8B BF16?
8. (M4) Why can't 8B BF16 serve 32 chats on an A100 2g.20gb slice?
9. (M6) Why `max by (UUID)` before summing DCGM power under MIG?
10. (M6) Why does the DCGM ServiceMonitor set `honorLabels`?
11. (M9) What is AIPerf "goodput"?
12. (M9) How do you turn a sweep into $ per 1,000 decisions?
13. (M9) What did MIG isolate in the noisy-neighbor test that time-slicing did not?
14. (M8) What does a perf gate compare, and when do you re-baseline?
15. (M14) What are the five parts of the one-page readout?

## Answers
1. The operator owns driver + toolkit (operator-managed mode): `driver.enabled`, `toolkit.enabled`, runtimeClass `nvidia`. Never mix with AKS-managed drivers.
2. It selects a named geometry from `<node>-mig-config`; MIG Manager applies it. Done when `nvidia.com/mig.config.state=success`.
3. Bandwidth follows memory slices (4 of 8); compute follows compute slices (3 of 7).
4. 16.06 GB / 1,935 GB/s = ~8.3 ms per token (~120 tok/s per sequence).
5. Downloads only the profile you will serve: smaller caches, faster, deterministic startup.
6. PB6 images run `update-ca-certificates` at start and need group 0 permissions, otherwise CrashLoopBackOff.
7. 2 (K,V) x 32 layers x 8 KV heads x 128 dims x 2 bytes = 128 KiB.
8. ~15 GiB of weights on a ~19.6 GiB slice leaves ~2 GiB for KV: ~16k tokens, ~15-20 chats at 800 tokens each.
9. Device-level fields repeat on every MIG instance series; summing them overstates power ~4x.
10. To keep the workload's `namespace`/`pod` labels instead of `exported_*` labels.
11. Requests per second that met every SLO you passed (`--goodput time_to_first_token:... inter_token_latency:...`).
12. VM $/h x slice share / (SLO-compliant req/s x 3,600) x 1,000.
13. SMs, L2, and memory bandwidth in hardware; time-slicing only interleaves contexts on shared hardware.
14. Candidate vs. baseline TTFT p95, ITL p95, throughput, with tolerances. Re-baseline deliberately (new GPU, accepted new version), never automatically.
15. Situation, what we tested, results table, recommendation, risks + next steps.
