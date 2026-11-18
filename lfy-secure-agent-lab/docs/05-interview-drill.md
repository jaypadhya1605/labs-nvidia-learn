# Interview drill - say it without notes

## Five sentences per theme (fill the blanks with YOUR numbers)
**Threat model.** "An agent that reads telemetry gives the attacker a voice, because logs contain attacker-written
fields. I planted injections in user agents, command lines, and URLs, and measured how often they changed the verdict
or triggered an action. Undefended, __% succeeded. With layered defenses, __%. No single layer was enough; the tool
policy and network policy stopped what the classifiers missed."

**Guardrails.** "Every LLM call passes a gateway running NeMo Guardrails: jailbreak detection and content safety on the
way in, content safety and PHI masking on the way out. It fails closed and pages when a safety model is down. Rails are
config, so the config is signed and verified before the pod starts. The cost was +__% latency per step, mostly
content-safety prefill on a 2g slice. I'd move safety checks to tool outputs and final reports to cut that in half."

**Red teaming.** "I use garak for model-level probes and a scenario harness for agent-level attacks, because agents fail
in workflows, not single prompts. Both feed a release gate in CI, like a performance gate. A prompt or policy change that
raises attack success or PHI leaks doesn't ship."

**Runtime security.** "The model proposes and code decides: read-only KQL, validated inputs, propose-only containment,
protected accounts. The agent pod has no internet route, and agent-written code runs only in an OpenShell sandbox with
kernel-enforced file and network policy. Those controls hold even when the model is fooled."

**Confidential AI.** "For customers who require that we can't see their data, the model key is released by Key Vault only
after Azure Attestation verifies an SEV-SNP confidential VM, and the H100 runs in CC mode with its own attestation. The
operator never holds the plaintext key."

## Questions
1. **Why isn't a jailbreak classifier enough?** Indirect injections look like data; classifiers miss paraphrases and
   flag benign security text. Defense in depth plus least-privilege tools.
2. **Fail-open or fail-closed?** Closed for a SOC: availability incident beats a silent control loss. Alert either way.
3. **How do you measure an agent's safety?** Scenario-based evals with ground truth: attack success, detection rate,
   task accuracy, leaks, latency - per change, gated in CI.
4. **What does the safety tax cost and how do you cut it?** Extra prefill-bound model calls per step; parallel rails,
   fewer checkpoints, smaller or faster safety models, slice placement.
5. **SBOM vs. VEX?** SBOM lists components; VEX states whether a known CVE actually affects the product.
6. **What does model signing prove?** Integrity and origin of the artifact bytes; not that the model is safe.
7. **Explain remote attestation in two sentences.** Hardware signs a measurement of what's running; a verifier checks it
   against expected values and only then releases secrets.
8. **MCP risks?** Tool poisoning, over-broad tools, injected tool outputs, confused-deputy credentials - enforce policy in
   the tool server, least privilege, and treat outputs as untrusted.
9. **Where do NVIDIA products fit?** NIM for serving, NeMo Guardrails + Nemotron safety models for rails, garak / NeMo
   Auditor for red teaming, NeMo Agent Toolkit for agent workflows, OpenShell for sandboxing, H100/Blackwell CC for
   confidential AI.
10. **Lab 02 callback:** why is the safety model's slice the bottleneck? Prefill is compute-bound and the 2g slice has
    2/7 of the SMs; KV room is tiny after 8B BF16 weights (prediction C).
