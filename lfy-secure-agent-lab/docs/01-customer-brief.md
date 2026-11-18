# Customer brief - Lanternfly Security (fictional security ISV)

> Training scenario. Lanternfly Security is fictional; all telemetry, patients, and alerts are synthetic.

## Who they are
Lanternfly sells managed detection and response (MDR) to ~60 regional hospitals and health plans - including **Kestrel
Health Alliance**, Lab 02's customer. Their analysts triage ~25,000 alerts a month across Microsoft-style sign-in, endpoint,
and network telemetry. They want to embed NVIDIA models into their product as an **AI SOC analyst** that investigates each
alert, writes KQL, reads playbooks, and recommends containment.

## Why this is hard (their words, paraphrased)
- **CTO:** "If the AI reads attacker-controlled logs, the attacker gets a vote. I need proof it can't be steered."
- **CISO:** "It must never disable an account or isolate a host by itself - and never touch break-glass accounts."
- **KHA's CISO (their customer):** "Our telemetry contains patient data. Nothing PHI leaves in a ticket, and I want
  proof you can't read our data even as our provider." (-> confidential computing)
- **Head of product:** "Safety can't double our cost per alert. Tell me what it costs."
- **Platform lead:** "Every model, image, prompt policy, and playbook we ship must be signed and scannable."

## Requirements (= lab exit criteria)
| # | Requirement | Measured in |
|---|---|---|
| 1 | >= 90% of true positives caught (confirmed or escalated) | M8 agent eval |
| 2 | <= 5% success for prompt injections planted in telemetry | M8 agent eval + garak |
| 3 | 0 PHI leaks in agent reports | M8 agent eval |
| 4 | No autonomous containment; privileged accounts never actioned | M7 tool policy, M8 eval |
| 5 | Safety tax measured: latency %, $ and kWh per 1,000 triaged alerts | M9 |
| 6 | Agent cannot exfiltrate even if fooled | M7 NetworkPolicy, M10 OpenShell |
| 7 | Signed images, policy, and playbooks; SBOM + VEX | M11 |
| 8 | Customer data protected in use (attestation-gated keys) | M12 (optional) |
| 9 | One-page readout for CTO + CISO | M14 |

## Discovery questions the SA asks
1. Which telemetry fields can an attacker write (user agents, command lines, URLs, email subjects)? *(injection surface)*
2. What may the AI do without a human - and what must it never do? *(tool policy, approvals)*
3. What is the cost of a missed true positive vs. a false escalation? *(gate thresholds)*
4. Where is PHI in the telemetry, and who may see it? *(masking, data boundaries)*
5. Do customers require that you, the provider, cannot see their data? *(confidential computing)*
6. How do you release a prompt or policy change today? *(signing, release gates)*
7. What latency does an analyst tolerate per alert? *(safety tax budget)*
