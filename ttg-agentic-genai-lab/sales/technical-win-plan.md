# Technical win plan - Tern Travel Group (fictional)

**Opportunity:** the Disruption Concierge in production on NVIDIA accelerated inference before peak storm season,
then the rest of guest operations (multilingual messaging, voice, revenue management). **Decision:** 2026-12-01.

## Account map
| Person | Role in the decision | What they need | Status |
|---|---|---|---|
| Rosa Delgado, VP Guest Operations | Sponsor, signs the business case | backlog cleared on storm days, guest satisfaction | supportive; wants the storm-day plan |
| Amina Yusuf, Head of Contact Center | Champion | agents trust it, clean approval queue, escalation | champion; co-presents the readout |
| Ken Ito, CIO | Platform owner, technical decision | runs on their cloud and Kubernetes, GitLab, on-call | neutral; worried about operating GPUs |
| Lars Berg, CISO and DPO | Veto | GDPR residency, PCI scope, audit trail | cautious; review starts 2026-11-04 |
| Grace Owusu, CFO | Approver | $ per 1,000 disruptions vs overtime and over-compensation | wants a number, not a range |
| Loyalty team | Policy owner | policy applied exactly; no over-compensation | signs off the golden set |
| Cloud partner account team | GPU capacity, marketplace purchase | committed spend, region capacity | aligned; reservation by 2026-12-04 |

## Why NVIDIA wins (proof, not adjectives)
| Claim | Proof in the PoC |
|---|---|
| The lowest cost per handled disruption at Tern's latency target | ledger: $[measured] per 1,000 at a 12 s p95 (criterion 3) |
| Exact policy, every time | [measured]% on 200 golden scenarios; 0 approval misses (criteria 1, 7) |
| Choice without lock-in | the same agent on vLLM, SGLang, TensorRT-LLM and NIM (criterion 5); MCP tools work with any framework |
| Runs in Tern's tenant, EU data in the EU | one Helm chart, two regions (criterion 8) |
| Supported in production | NIM with NVIDIA AI Enterprise support; pinned, tested versions |
| Tern can operate it | GitLab gates, 10 drills, a workshop for their engineers (criteria 9, 10) |

## Competition and how to respond
| Alternative | Their pitch | Our response |
|---|---|---|
| Hosted closed-model APIs | "No GPUs to run, best model quality" | Guest data stays in Tern's tenant; storm-day volume makes per-token pricing the biggest variable cost; we show measured $ per disruption, they show a price list |
| DIY open-source engines on a generic GPU cloud | "Same engines, cheaper GPUs" | Same open engines, plus NIM's optimized profiles, support and a tested upgrade path; the ledger compares them on the same traffic, so the choice is Tern's |
| Seasonal temps or an outsourced contact center | "People, not software" | ~7,330 agent-hours in 6 hours is ~1,220 people at once; the agent handles the policy work and leaves the hard cases to Tern's best agents |

## Risks
| Risk | Signal | Mitigation | Owner |
|---|---|---|---|
| The CISO's review runs past the decision date | review not started by 2026-11-04 | send the data-flow diagram and trace sample in week 1 | NVIDIA SA |
| GPUs unavailable in the EU region in December | no reservation by 2026-12-04 | cloud partner escalation; fall back to the next EU region | NVIDIA account manager |
| A policy change mid-PoC | Loyalty team announces v4 | rerun the golden set behind the eval gate; retrain the LoRA | Loyalty team + SA |
| The CFO compares GPU cost with the wrong baseline | "GPUs are expensive" | put overtime, temps and over-compensation per storm next to $ per 1,000 disruptions | Grace Owusu + SA |
| The 8B misses 95% | criterion 1 at 90-95% | prompt and LoRA iteration; the 70B as a measured fallback (ADR-06) | SA |

## Feedback to the product teams
The prefix-cache accounting request ([feature-request.md](feature-request.md)), filed with the measured impact.

## Next 3 actions
1. Send the security pack (data flow, residency, PCI scope, trace sample) to Lars Berg by 2026-10-30.
2. Get Grace Owusu's storm-day cost baseline (overtime, temps, vouchers) by 2026-11-06.
3. Book the readout with Rosa, Ken and Grace for 2026-11-24, with Amina presenting the guest messages.
