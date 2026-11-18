# Customer brief - Plover Health AI (fictional ISV)

**Company:** Plover Health AI, ~180 people, Series C. **Product:** Plover Assist - summarizes prior-authorization
requests into a strict JSON contract that nurse reviewers act on in 90 seconds. **Today:** a hosted closed LLM API +
a ~700-token style guide on every call.

## Why they called NVIDIA (and their CSPs)
| Customer (fictional) | Environment | Blocker today |
|---|---|---|
| Cardinal Mutual Health Plan (payer) | Azure, East US | PHI must stay in the payer's Azure tenant; cost per summary at 2M/month |
| Lakeshore Health System (provider) | Google Cloud | standardizing on GKE; wants the model in its VPC |
| St. Brigid Medical Center (hospital) | on-prem, **disconnected** | no Internet from the clinical network; change-controlled images |
| Prospects | AWS, Oracle Cloud | "can you run where we run?" |

## Requirements
- **One product, every platform:** the same artifact on AKS, GKE (EKS/OKE next) and an air-gapped data center.
- **Quality:** schema-valid JSON, no invented CPT/ICD-10 codes, conservative recommendations.
- **Latency:** p95 summary <= 8 s; streaming TTFT <= 1 s for the interactive reviewer UI.
- **Cost:** predictable $/1,000 summaries per cloud; licensing that fits cloud commits.
- **Change safety:** new styles (LoRA) and models ship as gated, reversible releases.
- **Next:** an agent that checks coverage criteria and clinical notes (tool calls) - safely.

## What NVIDIA brings (the PoC)
NIM (optimized, supported inference per GPU) + NIM Operator + GPU Operator, deployed by GitOps from one Helm chart;
measured price-performance per cloud; LoRA MLOps with a release gate; an air-gapped reference install; a weekly
operating cadence; and day-0 readiness for new NVIDIA models (Nemotron 3.5).

## The JD behind this lab (NVIDIA Senior SA, AI Cloud Services)
You are the main technical contact between NVIDIA, Plover, and the CSP partner SAs: you build the PoC, deploy NVIDIA AI
software at scale on cloud GPUs, support the AE and DevRel in closing it, run the weekly technical meetings, and deliver
the workshop. The README's coverage map ties each JD line to a module.
