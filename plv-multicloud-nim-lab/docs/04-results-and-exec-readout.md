# Results and executive readout (M13)

Fill every `__` with YOUR measurement and keep the source file next to it. Estimates from the selector are allowed
only in the "predicted" column.

## Results table
| Criterion (PoC charter) | AKS (A100 80GB, Spot) | GKE (RTX PRO 6000, Spot) | DC (RKE2, air gap) | Source |
|---|---|---|---|---|
| IaC + GitOps from zero | __ min | __ min | __ min (manual helm) | your notes / Argo CD |
| Schema-valid (base / adapter) | __% / __% | __% / __% | __% | `mlops/eval_gate.py` |
| Hallucinated codes | __ | __ | __ | eval gate |
| Operating point (conc, tok/s, ITL p95) | __ | __ | __ | `bench/compare_clouds.py` |
| Predicted tok/s (selector) and gap | __ (__%) | __ (__%) | __ | same |
| $/1M output tokens (GPU only / + NVAIE) | __ / __ | __ / __ | __ (amortized) | `bench/cost_model.py` |
| $/1,000 summaries at 50% duty | __ | __ | __ | same |
| TTFT p95 streaming | __ ms | __ ms | __ ms | AIPerf |
| Cold start (node + image + model) | __ s | __ s | __ s | `bench/coldstart_timeline.py` |
| Adapter: prompt-token ratio, canary -> promote time | __, __ min | __, __ min | n/a | M8 |
| Agentic success / injection success | __% / __% | __% / __% | - | `tests/agentic` |
| Drills passed | __/10 | | | `drills/` |

## One-page readout (5 parts - 🔁 Labs 02/03)
1. **Situation** - one product, three customer environments, cost and residency blockers.
2. **What we tested** - same chart on AKS + GKE + air-gapped DC; quality gate; benchmarks; LoRA release; managed vs self-hosted.
3. **Results** - the table above, 6 rows max, each with a source.
4. **Recommendation** - e.g. "Production on AKS with H100 NVL FP8 (verified SKU) for the payer; GKE G4 NVFP4 for the
   health system; the RKE2 pattern for St. Brigid. LoRA style adapter as the default path. NVAIE via each marketplace
   for the first year, annual subscription once utilization is stable above __%."
5. **Risks and next steps** - GPU reservations, verified-SKU migration, Nemotron 3.5 evaluation (add-on), owners and dates.

## Five sentences to say (fill in)
"Plover now runs the same NIM-based product on Azure, Google Cloud and in a disconnected hospital data center from one
chart. At your target load we measured __ summaries per second within the 8-second SLO on AKS and __ on GKE, which
is $__ and $__ per thousand summaries including NVIDIA AI Enterprise. The style adapter cut prompt tokens by __x with
the same accuracy, and we shipped and rolled it back with Git commits in under __ minutes. The air-gapped install made
zero outbound connections and kept no NGC key in any pod. We recommend __, and the decision we need from you is __."
