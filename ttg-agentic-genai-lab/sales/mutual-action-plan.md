# Mutual action plan (example dates for a PoC starting 2026-10-26)

Shared with Tern. Every row has one owner on each side, and the plan ends in production before peak storm season, not
at the readout. People are fictional.

| Date | Milestone | Tern owner | NVIDIA owner | Done when |
|---|---|---|---|---|
| Mon 2026-10-26 | Kickoff; discovery with Guest Ops, Contact Center, Loyalty | Rosa Delgado | SA | [discovery-questions.md](discovery-questions.md) answered |
| Fri 2026-10-30 | PoC charter signed | Rosa Delgado, Ken Ito, Lars Berg | SA | [poc-charter.md](poc-charter.md) signed |
| Fri 2026-10-30 | Security pack sent: data flow, residency, PCI scope, trace sample | Lars Berg | SA | review scheduled |
| Mon 2026-11-02 | GPU VM and Kubernetes namespaces ready (US + EU) | Ken Ito | SA + cloud partner | `engines/preflight.sh` passes; Helm dry run on both clusters |
| Wed 2026-11-04 | Security and privacy review starts | Lars Berg | SA | first review meeting held |
| Fri 2026-11-06 | Golden set and policy oracle signed off; storm-day cost baseline shared | Loyalty team, Grace Owusu | SA | 200 scenarios accepted; baseline in the business case |
| Tue 2026-11-10 | Engine bake-off: vLLM, SGLang, TensorRT-LLM, NIM | Ken Ito (observer) | SA | `results/bench/*-base.json` for 4 engines |
| Fri 2026-11-13 | Optimization, LoRA, VLM receipts; GitLab gates green | Platform team | SA | ledger + green pipeline |
| Tue 2026-11-17 | Workshop for Tern engineers (90 minutes) | Amina Yusuf (invites), platform team | SA | survey results |
| Thu 2026-11-19 | EU deployment test: residency evidence; drills run by Tern on-call | Ken Ito, Lars Berg | SA | criterion 8 evidence; drill log |
| Tue 2026-11-24 | Results readout (one page, 5 parts) | Rosa Delgado, Ken Ito, Grace Owusu | SA + account manager | [poc-evaluation-report.md](poc-evaluation-report.md) delivered |
| Tue 2026-12-01 | Decision: production pilot go / no-go | Rosa Delgado, Grace Owusu | account manager | signed decision |
| Fri 2026-12-04 | GPU reservation for storm season; NVIDIA AI Enterprise via the cloud marketplace | Ken Ito, Grace Owusu | account manager + cloud partner | reservation confirmed in both regions |
| Mon 2026-12-14 | Production pilot live on one hub, shadow mode (humans approve every action) | Amina Yusuf | SA | first storm-day drill passed |

US Thanksgiving (Thu 2026-11-26) is a no-meeting week for the US team: the readout is the Tuesday before.

**Checkpoint:** every PoC criterion has a date on this plan, and the last row is a production step, not a meeting.

**Say it:** "Here is the plan that gets you live before the first big storm; the dates on your side are the ones only
your team can move."
