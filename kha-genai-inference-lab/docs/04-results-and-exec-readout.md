# Results log and executive readout

Record numbers as you go. Every cell should trace to a file under `results/`.

## M2 - GPU layout
| Mode | nvidia.com resources advertised | GPU product | Driver | Notes |
|---|---|---|---|---|
| MIG all-balanced | | | | |

## M3 - Bandwidth probe (CUDA C++)
| Slice | SMs | Read GB/s | Copy GB/s | % of peak | Decode floor ms/token |
|---|---|---|---|---|---|
| 3g | | | | | |
| 2g | | | | | |
| 1g | | | | | |
| full (M8-D) | | | | | |

## M4 - NIM tenants
| Tenant | Profile selected | Cache download (min) | Pod start -> Ready (min) | max-model-len | KV cache size in logs |
|---|---|---|---|---|---|
| priorauth-llm | | | | | |
| shared-llm | | | | | |
| policy-embed | | | | | |

## M5 - Triage: PyTorch vs TensorRT on a 1g slice
| Engine | Batch 1 p50 ms | Batch 8 p50 ms | Throughput (seq/s) | Val accuracy / urgent F1 |
|---|---|---|---|---|
| PyTorch eager FP32 | | | | |
| PyTorch eager FP16 | | | | |
| TensorRT FP16 on Triton | | | | |

## M7 - Operating points
| Tenant | Concurrency at SLO | req/s | TTFT p95 | ITL p95 | Tokens/J | kWh / 1k decisions | $ / 1k (slice, Spot) | $ / 1k (slice, PAYG) |
|---|---|---|---|---|---|---|---|---|
| Prior-auth (caching on) | | | | | | | | |
| Prior-auth (caching off) | | | | | | | | |
| Member chat | | | | | | | | |
| Claims batch (throughput) | | | | | | | | |

## M8 - Isolation (victim = prior-auth at c=8, aggressor = batch flood at c=64)
| Mode | TTFT p95 alone | TTFT p95 flooded | change | ITL p95 alone | ITL p95 flooded | change |
|---|---|---|---|---|---|---|
| MIG | | | | | | |
| Time-slicing | | | | | | |
| MPS | | | | | | |
| Full GPU, exclusive (reference) | | n/a | | | n/a | |

KAI reclaim: time from `um-interactive` creation to Running: ____ s; batch pod evicted: ____

## M9 - Autoscaling and cold start
| Step | Seconds from scale trigger |
|---|---|
| HPA desired replicas 2 | |
| Cluster autoscaler scale-up event | |
| Node Ready | |
| GPU Operator validated + MIG configured | |
| NIM image pulled | |
| Model loaded, pod Ready | |

## M10 - Dynamo on prefix-heavy traffic (48 policies, Zipf 1.1, c=16)
| Variant | TTFT p50 | TTFT p95 | ITL p95 | Output tok/s | Prefix-cache hit ratio | Tokens/J |
|---|---|---|---|---|---|---|
| vLLM, round-robin | | | | | | |
| vLLM, KV router | | | | | | |
| vLLM, disaggregated 1P+1D | | | | | | |
| TensorRT-LLM, round-robin | | | | | | |

## M11 - Drills
| # | Time to diagnose | What gave it away | Note for the runbook |
|---|---|---|---|

---

# Executive readout (one page) - template

**To:** CIO, CFO, VP Utilization Management, Sustainability, CISO  **From:** <you>  **Date:** <date>

**Situation.** Three GenAI workloads ran on four dedicated GPUs with no telemetry; a time-slicing incident degraded
member chat for 90 minutes during open enrollment.

**What we tested.** One GPU partitioned with MIG serving all three workloads plus two supporting models, on AKS with
NVIDIA NIM; isolation under a batch flood; latency-throughput curves; autoscaling; KV-aware routing with Dynamo.

**Results.**
| Question | Answer |
|---|---|
| Does batch still hurt prior-auth? | TTFT p95 changed __% under flood with MIG vs. __% with time-slicing |
| Cost per 1,000 prior-auth decisions | $__ (Spot) / $__ (PAYG) at the SLO operating point |
| Energy per 1,000 decisions | __ kWh (method: DCGM energy counter, whole-GPU attribution) |
| GPUs needed for today's load | __ (from 4) |
| Open-enrollment readiness | scale-out in __ min; scheduled pre-scaling recommended |

**Recommendation.** (1) ... (2) ... (3) ...
**Risks.** Spot eviction for SLO tenants; member-chat concurrency ceiling on A100 2g; cold-start minutes.
**Next 30/60/90 days.** Pilot with UM nurses; FP8 on H100 NVL evaluation; production hardening (private cluster).
