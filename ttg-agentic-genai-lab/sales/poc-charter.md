# PoC charter - Tern Disruption Concierge on NVIDIA accelerated inference (4 weeks)

**Customer:** Tern Travel Group (fictional). **Dates:** 2026-10-26 to 2026-11-24 (readout); decision by 2026-12-01.
**Data:** synthetic only: 200 golden scenarios from `ttg.world` (seed 7), scored against Tern Care Commitments v3
([data/policies/tern-care-v3.md](../data/policies/tern-care-v3.md)). No real guest data enters the PoC.

## Business goal
Clear a storm-day backlog of ~40,000 disrupted travelers in 6 hours, with the policy applied exactly, at a GPU cost per
disruption the CFO can compare with overtime and over-compensation.

## Scope
**In:** the Disruption Concierge agent (rebooking, hotels, meal, credit and receipt vouchers, guest message, human
approval above $300); four inference engines on one GPU type; FP8, speculative decoding and a LoRA adapter; receipt
reading with a VLM; GitLab eval and perf gates; a US and an EU deployment of the same Helm chart.
**Out:** production integration with Tern's live reservation system, voice channels, bag tracing, real guest data.

## Success criteria (each measured with a named tool)
Criteria 1-4 decide the PoC. Criteria 5-10 are the evidence the platform and security teams asked for.

| # | Criterion | Target | Measured by |
|---|---|---|---|
| 1 | Policy-compliant task success on the golden set | >= 95% | `python -m ttg.bench.agent_bench` (`success_rate`, scored by `ttg.world.score`) |
| 2 | p95 end-to-end task latency at the storm-day operating point | <= 12 s | `agent_bench` load levels + `python -m ttg.bench.ledger --slo-p95-s 12` |
| 3 | GPU cost per 1,000 disruptions at that operating point | <= $2.00 | ledger `usd_per_1k_tasks`, at Tern's own GPU price |
| 4 | Storm-day capacity plan | GPUs per region, N+1 | ledger storm-day columns + [docs/09](../docs/09-results-and-exec-readout.md) section 5 |
| 5 | Same prompts and protocol on vLLM, SGLang, TensorRT-LLM and NIM | 4 engines measured | `engines/launch.sh` + `agent_bench`; token-level replay with `python -m ttg.bench.export_aiperf` + AIPerf |
| 6 | Prefix-cache hit ratio at the operating point | >= 85% | `cache_hit_ratio` in the bench file + `python -m ttg.bench.scrape_metrics` |
| 7 | Vouchers above $300 sent for human approval | 0 misses | `approval_flag_wrong` violations in the bench file = 0 |
| 8 | EU residents served only by the EU deployment; no card numbers in prompts or traces | 45 of 45 EU golden travelers; 0 card numbers | routing test on the EU golden travelers; trace review by the CISO's team |
| 9 | GitLab eval and perf gates green on the release branch | both green | `.gitlab-ci.yml` jobs `eval-gate` and `perf-gate` (`scripts/gate.py`) |
| 10 | Tern's engineers can run and repair it | workshop delivered; 10 drills run by Tern on-call | [workshop/](../workshop/README.md), [drills/](../drills/README.md) |

## Environment
- One GPU VM in Tern's cloud tenant (1x H100 80GB) for M4-M8; one 8x H100 node for 2 hours for the parallelism tests.
- Tern's Kubernetes cluster (US region) plus a namespace in the EU region for the residency test.
- Engine and model versions pinned in `env/pins.env`; every result file records its GPU, engine and settings.

## Roles
| Who | Role in the PoC |
|---|---|
| Rosa Delgado, VP Guest Operations | Sponsor; accepts criteria 1-4 |
| Amina Yusuf, Head of Contact Center | Champion; reviews guest messages and the approval flow |
| Ken Ito, CIO | Environment, GPUs, Kubernetes, GitLab |
| Lars Berg, CISO and DPO | Criterion 8; security review |
| Grace Owusu, CFO | Criterion 3 baseline: today's storm-day cost |
| Loyalty team | Signs off the policy oracle on the golden set |
| NVIDIA solutions architect | Runs the bake-off, the gates, the workshop and the readout |
| NVIDIA account manager | Commercial path: NVIDIA AI Enterprise, GPU capacity with the cloud partner |

## Timeline
Week 1: environment, golden set signed off, baseline. Week 2: engine bake-off. Week 3: optimization, LoRA, VLM,
gates, EU deployment. Week 4: workshop, drills, results readout. Owners and dates: [mutual-action-plan.md](mutual-action-plan.md).

## Exit
- **Go:** criteria 1-4 pass and 7-8 show zero misses: production pilot on one hub before peak storm season.
- **Fix and retest (one week):** a criterion misses by less than 10%.
- **Stop:** criterion 1 below 90% after prompt and LoRA work, or criterion 8 fails the CISO's review.

## Sign-off
| Name | Role | Date |
|---|---|---|
| Rosa Delgado | Sponsor | |
| Ken Ito | CIO | |
| Lars Berg | CISO and DPO | |
| NVIDIA solutions architect | | |
