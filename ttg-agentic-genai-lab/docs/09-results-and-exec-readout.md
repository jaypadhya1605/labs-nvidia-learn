# M10 - Results and the executive readout

**The ledger turns every benchmark file into one number the CFO understands: GPU dollars per 1,000 disruptions at
the 12-second SLO.** Fill every `[fill from results/ledger]` with YOUR measurement and keep the source file next to it.
Numbers from the mock engine never go into this document.

## 1. Produce the results
Launch one engine configuration at a time (`engines/launch.sh <engine> <variant>`), run the same golden set at the
same load levels, then stop it (`engines/stop.sh`). Labels follow `<engine>-<variant>`.

```bash
. env/pins.env && . env/lab.env                     # GPU_HOURLY_USD and the TTG_* defaults
COMMON="--n 96 --levels 1,4,16,32 --gpu-hourly ${GPU_HOURLY_USD} --gpus 1"   # GPUs per replica: 2 for tp2/pp2
python -m ttg.bench.agent_bench --label hf-base      --engine hf     --n 24 --levels 1,2 --gpu-hourly ${GPU_HOURLY_USD} --gpus 1
python -m ttg.bench.agent_bench --label vllm-nocache --engine vllm   ${COMMON}
python -m ttg.bench.agent_bench --label vllm-base    --engine vllm   ${COMMON}
python -m ttg.bench.agent_bench --label sglang-base  --engine sglang ${COMMON}
python -m ttg.bench.agent_bench --label trtllm-base  --engine trtllm ${COMMON}
python -m ttg.bench.agent_bench --label nim-base     --engine nim    ${COMMON}
python -m ttg.bench.agent_bench --label vllm-fp8     --engine vllm   ${COMMON}
python -m ttg.bench.agent_bench --label vllm-spec    --engine vllm   ${COMMON}
TTG_PROMPT_VARIANT=short TTG_MODEL=ttg-short \
python -m ttg.bench.agent_bench --label vllm-lora    --engine vllm   ${COMMON}
python -m ttg.bench.ledger results/bench/*.json --slo-p95-s 12 \
  --order hf-base,vllm-nocache,vllm-base,vllm-fp8,vllm-spec,vllm-lora --out results/ledger
```
The ledger writes `results/ledger/ledger.md` (the table below), `ledger.csv`, `ledger.json` and
`results/ledger/waterfall.png` (the chart for the readout and the blog; it needs at least two priced runs).

**How it picks the operating point:** for each file, the load level with the most tasks per second whose p95 task
latency is within the SLO (`--slo-p95-s 12`) and whose success rate is at least `--min-success` (0.95). Its tasks per
second become tasks per GPU-hour and $ per 1,000 disruptions: `$/hour x GPUs / (tasks per second x 3,600) x 1,000`
(🔁 Labs 02-04). A row marked "NO" met the SLO at no measured level: add lower levels or fix the configuration.

## 2. Results table
| Configuration | Operating concurrency | Tasks/s | Tasks per GPU-hour | p95 task latency | Success | Cache hit ratio | Prompt / output tokens per task | $ per 1,000 disruptions | Source |
|---|---|---|---|---|---|---|---|---|---|
| hf-base (naive server, batch 1) | [fill from results/ledger] | | | | | | | | `results/bench/hf-base.json` |
| vllm-nocache | [fill from results/ledger] | | | | | | | | `results/bench/vllm-nocache.json` |
| vllm-base | [fill from results/ledger] | | | | | | | | `results/bench/vllm-base.json` |
| sglang-base | [fill from results/ledger] | | | | | | | | `results/bench/sglang-base.json` |
| trtllm-base | [fill from results/ledger] | | | | | | | | `results/bench/trtllm-base.json` |
| nim-base | [fill from results/ledger] | | | | | | | | `results/bench/nim-base.json` |
| vllm-fp8 | [fill from results/ledger] | | | | | | | | `results/bench/vllm-fp8.json` |
| vllm-spec | [fill from results/ledger] | | | | | | | | `results/bench/vllm-spec.json` |
| vllm-lora (short prompt) | [fill from results/ledger] | | | | | | | | `results/bench/vllm-lora.json` |

Expectations to check your numbers against (estimates, not results): ~6.5 LLM calls, ~15,100 prompt tokens and ~240
output tokens per task; an ideal prefix cache reuses ~96% of prompt tokens. A cache hit ratio far below 85% on a
caching engine means something dynamic sits in the prefix (drill 1).

## 3. The acceleration waterfall
Read `results/ledger/waterfall.png` left to right, in `--order`. The first and last bars are GPU $ per 1,000
disruptions; each bar between is the change from the step before (green = cheaper, red = more expensive):

| Step | What changed | $ per 1,000 disruptions | Tasks per GPU-hour | Change |
|---|---|---|---|---|
| Baseline | naive Transformers server, one request at a time | [fill from results/ledger] | | - |
| + continuous batching, paged KV cache | vLLM with prefix caching off | [fill from results/ledger] | | [fill] |
| + prefix caching | the shared ~1,840-token system prompt computed once | [fill from results/ledger] | | [fill] |
| + FP8 weights | half the weight bytes per decode step | [fill from results/ledger] | | [fill] |
| + n-gram speculation | only if it wins at the operating point (ADR-08) | [fill from results/ledger] | | [fill] |
| + prompt baking (LoRA, short prompt) | fewer prompt tokens per call | [fill from results/ledger] | | [fill] |

## 4. PoC criteria
| # | Criterion | Target | Result | Pass? | Evidence |
|---|---|---|---|---|---|
| 1 | Policy-compliant task success, golden set | >= 95% | [fill from results/ledger] | | `results/bench/<best>.json` |
| 2 | p95 task latency at the storm-day operating point | <= 12 s | [fill from results/ledger] | | ledger |
| 3 | GPU cost per 1,000 disruptions | <= $2.00 | [fill from results/ledger] | | ledger |
| 4 | Storm-day capacity plan with N+1 | a GPU count | [fill: section 5] | | this document |
| 5 | Same prompts on vLLM, SGLang, TensorRT-LLM, NIM | 4 engines | [fill] | | `results/bench/*-base.json` |
| 6 | Vouchers above $300 sent for approval | 0 misses | [fill: `approval_flag_wrong` count] | | bench `tasks[].violations` |
| 7 | GitLab eval + perf gates green | both | [fill] | | pipeline link |

## 5. Storm-day capacity plan
- Peak load: 40,000 disruptions in 6 hours = 1.85 tasks/s on average; plan for a 2x peak hour = **3.7 tasks/s**.
- Per GPU at the SLO: **[fill from results/ledger] tasks/s** (best configuration that passes criteria 1-3). The
  ledger's "Storm-day GPUs" column is the one-region answer (replicas for 3.7 tasks/s, plus one spare).
- US region GPUs = ceil(3.7 x 0.78 / per-GPU tasks/s) + 1. EU region GPUs = ceil(3.7 x 0.22 / per-GPU tasks/s) + 1.
  The golden set has 45 EU residents out of 200 (22%); use Tern's real share.
- Storm-day GPU cost = GPUs x 6 hours x $/GPU-hour = **$[fill]**, against ~7,330 agent-hours of handle time.

## 6. The one-page readout (5 parts - 🔁 Labs 02-04)
Audience: Rosa Delgado (VP Guest Operations), Ken Ito (CIO), Grace Owusu (CFO). One page, six result rows at most.

1. **Situation.** Storm days strand ~40,000 travelers in 6 hours; at 11 minutes each that is ~7,330 agent-hours and a
   3-hour queue. Over-compensation is common because the policy is applied by hand.
2. **What we tested.** A Disruption Concierge agent (LangGraph + MCP) on Llama 3.1 8B, served by vLLM, SGLang,
   TensorRT-LLM and NIM on [fill: GPU]; 200 synthetic golden scenarios scored against Tern Care Commitments v3;
   load levels up to [fill] concurrent tasks; FP8, speculation and a LoRA that bakes in the policy.
3. **Results.** [fill from results/ledger: success, p95 latency, tasks per GPU-hour, $ per 1,000 disruptions for the
   recommended configuration, plus the baseline row for contrast].
4. **Recommendation.** [fill: e.g. "Production on <engine> with <precision> on <GPU>, prefix caching on, speculation
   <on/off>; N US GPUs + M EU GPUs on storm days; LoRA after one more policy cycle."]
5. **Risks and next steps.** Policy changes need a retrain or prompt update behind the gate; GPU reservations before
   winter; human approval staffing for vouchers above $300; GDPR review of the EU deployment; owners and dates in
   [sales/mutual-action-plan.md](../sales/mutual-action-plan.md).

## Five sentences to say (fill in)
"On your 200-scenario golden set the agent applied Tern Care exactly in __% of cases, and every voucher above $300
went to a human. At your storm-day load it held a 12-second p95 at __ tasks per second per GPU. That is $__ of GPU
per 1,000 disruptions, versus __ agent-hours today. The biggest single win was __, which took us from __ to __ tasks
per GPU-hour. We recommend __ GPUs in the US and __ in the EU for storm days, and the decision we need from you is __."

**Checkpoint:** every number in the readout has a source file in `results/`, and you can say why the operating point is
not the throughput peak.

**Say it:** "Here is what one disruption costs you on each engine, measured on your policy at your latency target;
the waterfall shows which optimization bought what."
