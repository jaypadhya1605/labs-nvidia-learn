# Technical win plan - Plover Health AI

## Account map
| Person (fictional) | Role in the decision | What they need to see | Owner of the relationship |
|---|---|---|---|
| Priya Raman, VP Engineering | Sponsor / economic influence | cost curve, customer unblock, risk | NVIDIA AE + SA |
| Dana Ortiz, Platform lead | Technical champion | same chart everywhere, GitOps, drills | SA |
| Marcus Lee, ML lead | Evaluator | eval gate, LoRA pipeline, model choice (Nemotron) | SA + DevRel |
| Security / compliance | Veto | air gap, secrets in vaults, no PHI in hosted APIs | SA + CSP partner |
| CFO office | Approver | $/1k summaries, licensing channel, commit drawdown | AE + CSP partner |
| Azure / Google partner SAs | Co-sell | GPU capacity, marketplace private offers | SA |

## Why NVIDIA wins (proof, not adjectives)
1. One NIM + one chart across AKS, GKE and an air-gapped DC (M4, M9).
2. Measured price-performance per cloud from their traffic (M5), engine choice explained (M6).
3. LoRA-as-a-release with a gate (M8) - their style, their IP, cheaper prompts.
4. Day-0 model readiness (Nemotron 3.5 add-on) - they adopt new models with the same gates.

## Competition and how to respond
- **Hosted closed API:** fine for prototypes; PHI residency, cost at volume and on-prem customers decide it (M10 scorecard).
- **DIY vLLM:** NIM = supported, per-GPU optimized profiles, security patching; same OpenAI API (M6 shows the engine underneath).
- **Another GPU cloud:** the chart runs there too - the value is the software and the operating model, not the VM.

## Risks
| Risk | Mitigation | Owner |
|---|---|---|
| GPU quota (A100 / G4) late | request in week 0; Spot for PoC; selector alternatives | SA + CSP |
| A100 PCIe not on NIM's verified list | PoC accepted risk; production on verified SKU (H100 NVL / RTX PRO 6000) | SA |
| Spot eviction during demos | pause discipline, recorded demo fallback | SA |
| Team new to Kubernetes GPU ops | workshop + drills + weekly sync | SA |
| Budget timing | marketplace hourly NVAIE first, annual later | AE |

## Next 3 actions
1. Send the PoC charter for signature (sponsor) - this week.
2. Quota tickets with both partner SAs - this week.
3. Book the 2-hour workshop for week 2.
