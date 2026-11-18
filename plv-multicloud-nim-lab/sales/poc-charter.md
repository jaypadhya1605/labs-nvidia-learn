# PoC charter - Plover Assist on NVIDIA NIM, multi-cloud (4 weeks)

**Sponsor:** Plover VP Engineering · **Technical owner:** NVIDIA SA (AI Cloud Services) · **Partners:** Azure and Google
partner SAs · **Data:** synthetic only.

## Business goal
Unblock three customer types with one product: a payer on Azure, a health system on Google Cloud, a hospital that
requires an air-gapped on-prem install - at a lower and predictable cost per summary.

## Success criteria (all measured with the lab's tools)
| # | Criterion | Target | Evidence |
|---|---|---|---|
| 1 | Same chart + GitOps on AKS and GKE, IaC from zero | < 60 min per cluster | `infra/tf.sh`, Argo CD screenshots |
| 2 | Summary quality | schema-valid >= 98% (base), >= 97% (adapter), 0 hallucinated codes on 120 held-out cases | `mlops/eval_gate.py` |
| 3 | Latency at target load | p95 summary <= 8 s, streaming TTFT p95 <= 1 s at 20 concurrent reviewers | AIPerf, `bench/run_sweep.sh` |
| 4 | Cost | $/1,000 summaries <= the agreed target at 50% duty cycle, per cloud | `bench/compare_clouds.py`, `bench/cost_model.py` |
| 5 | LoRA as a release | >= 3x fewer prompt tokens; canary and rollback by Git commit < 10 min | `mlops/promote.sh`, Grafana |
| 6 | Air-gapped install | zero egress, image list = contract, no NGC key in pods | `datacenter/verify-airgap.sh` |
| 7 | Operability | dashboards, alerts, 10 drills passed, weekly sync held 4x | `drills/`, `fleet/` |

## Out of scope
Production PHI, HA across regions, EKS/OKE (stretch only), model training beyond the style adapter.

## Timeline
W1 infra + GitOps + baseline (M0-M4) · W2 benchmarks, engines, scale (M5-M7) · W3 LoRA MLOps + data center (M8-M9) ·
W4 managed comparison, readout, go/no-go (M10-M13).

## Exit
Go/no-go with VP Eng + CFO delegate: production architecture, NVIDIA AI Enterprise path (marketplace hourly vs
annual subscription), GPU reservation plan per cloud, owners.
