# Discovery questions (M1) - Tern Travel Group (fictional)

Ask the business questions first, then the technical ones. Write the answer and the number next to each question: the
PoC charter is built from them. People are fictional; their roles are the ones you will meet on a real account.

## Sponsor - Rosa Delgado, VP Guest Operations
| # | Question | Why we ask |
|---|---|---|
| 1 | How many travelers were disrupted on your worst storm day last winter, and over how many hours? | Sizes the peak (here 40,000 in 6 h) |
| 2 | What happens to a guest who waits three hours? Rebooks elsewhere, calls again, complains? | The cost of the queue, in her words |
| 3 | What share of storm-day contacts are rebookings, hotels, vouchers, or something else ("where is my bag")? | Scopes what the agent must do on day one |
| 4 | What would make you call this a success after the first storm? | The criterion she will judge us on |
| 5 | Who else must agree before this goes live? | The decision map |

## Champion - Amina Yusuf, Head of Contact Center
| # | Question | Why we ask |
|---|---|---|
| 6 | What is the average handle time per disruption, and how much of it is policy lookup? | 11 minutes today; the agent's baseline |
| 7 | Who approves large vouchers today, how fast, and what is the limit? | The $300 approval rule and the queue behind it |
| 8 | When should the agent hand a guest to a person? | Escalation path; a criterion for trust |
| 9 | What would your agents need to see to trust it? | Audit trail and explanations in the UI |

## Platform owner - Ken Ito, CIO
| # | Question | Why we ask |
|---|---|---|
| 10 | Where does it have to run: which cloud, which regions, Kubernetes already? | Deployment target (🔁 Lab 04 AKS/GKE) |
| 11 | Can you get GPUs in both regions before December, and on what terms (reserved, on-demand)? | Capacity is a procurement lead time |
| 12 | What is your CI/CD, and who is on call for it? | GitLab gates (M9); ownership after the PoC |
| 13 | Which systems of record must the agent call, and do they have APIs with test environments? | MCP server scope (ADR-03) |
| 14 | What is your observability stack? | Prometheus/Grafana and traces (ADR-04) |

## Veto holder - Lars Berg, CISO and DPO
| # | Question | Why we ask |
|---|---|---|
| 15 | Which guests are EU residents, and where must their data be processed and stored? | GDPR: the EU deployment (ADR-11) |
| 16 | Is any hosted model API allowed to see guest data? | Rules out hosted endpoints for production data |
| 17 | What must the audit trail contain, and for how long? | `TTG_TRACE_DIR` retention, every tool call traced |
| 18 | How do refunds work today, and where do card numbers live? | PCI scope: payment tokens only, never in the LLM |
| 19 | What is your review process for a new AI system, and how long does it take? | Goes on the mutual action plan |

## Approver - Grace Owusu, CFO
| # | Question | Why we ask |
|---|---|---|
| 20 | What does a storm day cost you today: overtime, temporary staff, vouchers? | The baseline the $ per 1,000 disruptions competes with |
| 21 | How much over-compensation do you estimate per storm? | Policy exactness is money |
| 22 | When is the budget decided, and is this capex or opex? | Timing of the decision |

## Data and policy owners - Ops Control (Tern Air) and the Loyalty team
| # | Question | Why we ask |
|---|---|---|
| 23 | How fresh is the flight status feed, and how fast is the rebooking inventory API under load? | Tool latency budget inside 12 s |
| 24 | How often does the compensation policy change, and who signs off a change? | Retraining or prompt updates behind the gate |
| 25 | Which exceptions do your best agents make that the policy doesn't say? | What the agent must escalate, not invent |

## Technical deep dive - ML and platform engineers
| # | Question | Why we ask |
|---|---|---|
| 26 | Which latency do guests tolerate in the app before they call instead? | The 12-second p95 SLO |
| 27 | Do you have conversation logs we can turn into synthetic scenarios (no real guest data)? | The golden set's realism |
| 28 | Which models have you tried, and how did you evaluate them? | Avoid repeating their experiments; reuse their metrics |
| 29 | Which languages must the guest message support? | Model choice, guardrails, evaluation per locale |
| 30 | Who will own the agent's prompts, adapters and gates after the PoC? | Enablement plan (workshop, drills) |

**Checkpoint:** every success criterion in the PoC charter traces back to an answer here.
