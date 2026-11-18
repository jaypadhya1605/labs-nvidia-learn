# Customer brief - Kestrel Health Alliance (fictional)

> Training scenario. Kestrel Health Alliance is fictional and every byte of data in this lab is synthetic.

## Who they are
Kestrel Health Alliance (KHA) is an integrated payer-provider in the Great Lakes region: a health plan with about
1.8 million members and a network of 11 hospitals. KHA runs on Azure. Its AI platform team supports three business
units that each shipped a generative-AI workload in the past year - on separate, hand-built GPU VMs.

## The three workloads
| Workload | Owner | Traffic shape | What "good" means (SLO) |
|---|---|---|---|
| **Prior-Auth Clinical Summarizer** | Utilization Management (UM), ~140 nurses | Long prompts: clinical notes + the relevant medical-policy excerpt (~6,000 tokens in, ~350 out). Many requests reuse the same policy text. | TTFT p95 <= 1.5 s, inter-token latency (ITL) p95 <= 60 ms, at 12 concurrent reviewers |
| **Member Services Assistant** | Member Experience (call center + chat) | Short turns (~600 in, ~200 out); 5x spikes during open enrollment | TTFT p95 <= 0.8 s, ITL p95 <= 50 ms, at 32 concurrent chats |
| **Claims Note Coder** (batch) | Payment Integrity | ~50,000 notes/night (~1,200 in, ~150 out) | Done by 6 a.m.; must never degrade the two interactive workloads |

Supporting models: a policy-retrieval **embedding** model that feeds the prior-auth prompt, and a small **urgency
triage** classifier that flags requests likely to need expedited review.

## Current state
- Four A100 VMs, one per team plus an idle standby. Nobody can say how busy they are.
- No GPU telemetry, no per-tenant cost, no SLO dashboard, no energy numbers.
- Hand-built model-server containers; each team pinned different versions; nobody owns upgrades.
- **The incident:** during open enrollment the claims batch overran into business hours on a GPU it shared with
  Member Services through time-slicing. Chat latency tripled for 90 minutes and the call center escalated to the CIO.

## Stakeholders
| Stakeholder | What they said | What this lab delivers |
|---|---|---|
| CIO | "One platform, not four science projects. Scale up for open enrollment, down at night." | Multi-tenant AKS + NVIDIA stack, autoscaling, runbooks |
| CFO | "What does one prior-auth decision cost us to run?" | $ per 1,000 decisions, self-hosted vs. API break-even |
| VP Utilization Management | "My nurses can't wait. And batch must never slow us down again." | SLO dashboard + isolation experiment (MIG vs. time-slicing vs. MPS) |
| Sustainability Office | "We report energy per AI decision in the ESG report." | kWh per 1,000 decisions from DCGM energy counters, with the method |
| CISO | "PHI stays in our tenant. No third-party hosted inference." | Self-hosted NIM inside KHA's subscription; synthetic data only in the lab |
| Platform Engineering lead | "Give my team something they can operate at 2 a.m." | Break-fix drills, observability, pinned versions, CI performance gate |

## Requirements
**Functional:** OpenAI-compatible endpoint per workload; retrieval embeddings; triage classifier; tenant isolation;
chargeback per business unit.
**Non-functional:** the SLOs above; PHI never leaves the tenant; reproducible infrastructure as code; upgrade and
rollback path; Spot-tolerant non-production.

## Engagement exit criteria (= lab exit criteria)
1. Consolidate three workloads + two supporting models onto one GPU with hardware isolation, and **prove** it:
   prior-auth TTFT p95 changes < 10% while the batch floods its neighbor.
2. A latency-throughput curve per workload and a chosen SLO-compliant operating point.
3. $ per 1,000 prior-auth decisions and kWh per 1,000 decisions, with a reproducible method.
4. Autoscaling on queue depth plus a measured cold-start timeline.
5. KV-aware routing quantified on policy-heavy traffic; an honest verdict on disaggregated serving for this hardware.
6. A runbook with rehearsed break-fix drills.
7. A one-page executive readout with a recommendation.

## Discovery questions the SA asks (and why)
1. At what p95 latency does a nurse switch tabs? *(turns UX into an SLO)*
2. How much of each prompt is shared across requests - policy text, system prompts? *(prefix caching, KV routing)*
3. What is the peak-to-average ratio, and is the peak predictable? *(autoscaling vs. scheduled capacity)*
4. Who owns the GPU budget, and how will you charge back? *(namespaces, quotas, queues)*
5. Where is the PHI boundary? *(self-hosted NIM vs. hosted API)*
6. What quality bar was measured for the current model? *(headroom for FP8 or a smaller model)*
7. What happens today when a GPU node dies at 2 a.m.? *(operability, runbooks)*
8. Do your target VM families have InfiniBand/RDMA? *(whether disaggregated serving is even on the table)*

## Out of scope
Fine-tuning the LLMs, clinical validation of outputs, production network hardening (private cluster, egress
firewall, private endpoints), and any real PHI.
