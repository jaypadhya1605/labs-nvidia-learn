# PoC evaluation report - Tern Disruption Concierge (template)

Fill every `[fill]` from a file in `results/` and keep the file path next to the number. Mock-engine numbers never
appear in this report. Tern Travel Group is fictional; all data is synthetic.

**Prepared by:** [fill: your name], NVIDIA solutions architect. **Date:** [fill]. **Readout:** 2026-11-24.

## 1. Summary (three sentences)
[fill: what we tested, the headline result against criteria 1-3, and the recommendation.]

## 2. Results against the charter
| # | Criterion | Target | Result | Pass? | Evidence |
|---|---|---|---|---|---|
| 1 | Policy-compliant task success, golden set | >= 95% | [fill] | | `results/bench/<best>.json` |
| 2 | p95 task latency at the operating point | <= 12 s | [fill] | | `results/ledger/ledger.md` |
| 3 | GPU $ per 1,000 disruptions | <= $2.00 | [fill] | | `results/ledger/ledger.csv` |
| 4 | Storm-day capacity plan, N+1 | GPUs per region | [fill: US + EU] | | [docs/09](../docs/09-results-and-exec-readout.md) section 5 |
| 5 | Four engines, same prompts | 4 measured | [fill] | | `results/bench/*-base.json` |
| 6 | Prefix-cache hit ratio at the operating point | >= 85% | [fill] | | bench `cache_hit_ratio`, `scrape_metrics` |
| 7 | Vouchers above $300 sent for approval | 0 misses | [fill] | | `approval_flag_wrong` count |
| 8 | EU residency; no card numbers in traces | 45/45; 0 | [fill] | | routing test, CISO trace review |
| 9 | GitLab eval and perf gates | both green | [fill] | | pipeline link |
| 10 | Workshop and drills | delivered; 10 run | [fill] | | survey results, drill log |

## 3. What we tested
- **Workload:** 200 synthetic golden travelers, Tern Care Commitments v3, 6.5 LLM calls and 3-11 tool calls per task.
- **Hardware:** [fill: GPU model, count, cloud, region, $ per GPU-hour used for pricing].
- **Software:** [fill from `env/pins.env`: engine versions, NIM version, model ids, LoRA adapter id].
- **Load:** concurrency levels [fill]; operating point chosen as in [docs/09](../docs/09-results-and-exec-readout.md).

## 4. Results
Paste `results/ledger/ledger.md` here, then the waterfall (`results/ledger/waterfall.png`).

| Configuration | Tasks per GPU-hour | p95 task latency | Success | $ per 1,000 disruptions |
|---|---|---|---|---|
| [fill: baseline] | | | | |
| [fill: recommended] | | | | |

## 5. Issues found and how we resolved them
| Issue | How we found it | Resolution | Drill |
|---|---|---|---|
| [fill: e.g. cache hit ratio dropped after a prompt change] | [fill: perf gate] | [fill] | [drills/README.md](../drills/README.md) drill 1 |
| [fill] | | | |

## 6. Recommendation
[fill: engine, precision, speculation on/off, LoRA or full prompt, GPUs per region for storm days, and what happens
first: a production pilot on one hub in shadow mode, humans approving every action for the first storm.]

## 7. Risks and next steps
| Risk | Mitigation | Owner | Date |
|---|---|---|---|
| Policy change between now and the first storm | change behind the eval gate; retrain the LoRA or update the prompt | Loyalty team + platform | [fill] |
| GPU capacity in December | reservation placed by 2026-12-04 | CIO + cloud partner | 2026-12-04 |
| Approval queue overload on storm days | staffing plan for vouchers above $300 | Head of Contact Center | [fill] |
| GDPR review of the EU deployment | DPO review with the trace sample | CISO and DPO | [fill] |

## Appendix - reproduce every number
```bash
. env/pins.env && . env/lab.env
engines/launch.sh <engine> <variant>
python -m ttg.bench.agent_bench --label <engine>-<variant> --engine <engine> --levels 1,4,16,32 --n 96 \
  --gpu-hourly "${GPU_HOURLY_USD}" --gpus 1
python -m ttg.bench.ledger results/bench/*.json --slo-p95-s 12 --out results/ledger
```
Result files: [fill: list]. Pipeline: [fill: link]. Versions: `env/pins.env` at commit [fill].
