# M13 - Break-fix drills

Same rules as Lab 02: inject, diagnose from symptoms only, fix, then say the one-liner from memory.
Drills 1-4 are Lab 02 drills on purpose - if they take you longer than 3 minutes now, re-read Lab 02's drill table.

```bash
drills/inject.sh 5          # break
drills/inject.sh reset 5    # repair
```

| # | Drill | Symptom | Root cause | Fix | Say it |
|---|---|---|---|---|---|
| 1 | 🔁 Whole-GPU request on a MIG node | Pod `Pending`, `Insufficient nvidia.com/gpu` | Mixed strategy advertises slices only | Request the slice resource | "With MIG you schedule slices, not GPUs." |
| 2 | 🔁 PB6 NIM with wrong groupID | triage-llm `CrashLoopBackOff` | PB6 images need `groupID: 0` | Restore groupID | "PB6 NIMs need group 0." |
| 3 | 🔁 Missing GPU toleration | `untolerated taint {nvidia.com/gpu}` | GPU pool is tainted | Add tolerations | "GPU pods must opt in to GPU nodes." |
| 4 | 🔁 Tenant over quota | `exceeded quota: gpu-quota` | lfy-safety owns one 2g slice | Free the slice or raise quota deliberately | "Quotas make MIG slices enforceable budgets." |
| 5 | Safety model unreachable | Every agent step returns `BLOCKED_BY_POLICY`; `guardrail_errors_total` climbs; alert `LFYGuardrailFailClosed` | Wrong safety NIM URL in the policy | Restore the config | "Fail-closed turns a safety outage into an availability incident - which is the right trade for a SOC." |
| 6 | Fail-open misconfiguration | Unsafe prompts answered; responses carry `"bypassed": true` | `FAIL_MODE=open` plus a broken safety model | `FAIL_MODE=closed`, fix the model | "Fail-open silently removes every control; it must be a conscious, alerted decision." |
| 7 | Tampered policy (after M11) | Gateway stuck in `Init:Error`; `verify-policy` log shows verification failed | Someone removed the jailbreak rail without re-signing | Restore, or change it properly and re-sign | "A safety policy is a release artifact: signed, reviewed, gated." |
| 8 | Poisoned playbook | After M11: agent `Init:Error`. Before M11: defended agent cites a "VPN range" for a Tor IP | Unsigned community content entered the retrieval corpus | Restore the signed bundle | "RAG content is supply chain too - sign the corpus, not just the model." |
| 9 | Content-safety false positive | Benign malware-analysis text blocked as S17 | A general-purpose taxonomy meets security language | Tune the category wording, re-sign, re-run the red-team gate | "Safety models need domain tuning; measure false blocks like you measure misses." |
| 10 | Exfiltration attempt from the agent pod | `urlopen` times out | NetworkPolicy allows only DNS, gateway, KQL, local NIMs | (none - that's the control working) | "Network policy is the control that still works when the model is fooled." |
