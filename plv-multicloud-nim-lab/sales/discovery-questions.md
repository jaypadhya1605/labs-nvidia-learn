# Discovery questions (M1) - Plover Health AI and its CSP partners

Ask in this order; stop when you can fill the PoC charter. Everything here is fictional.

## Business (VP Eng, CFO delegate)
1. What does one prior-auth summary cost you today (API spend / summaries), and what must it cost next year?
2. Which customers are blocked today, and why (data residency, PHI in a third-party API, on-prem requirement)?
3. What happens if the summarizer is down for an hour - who notices first?
4. Who signs the production purchase, and through which channel (Azure/Google marketplace commit, enterprise agreement)?

## Workload (ML lead, product)
5. Volume: summaries per day, peak per minute, growth; interactive (streaming) vs batch share.
6. Shape: input/output tokens per request (we measured ~1,400 in / ~300 out with the style guide).
7. Quality bar: what is a wrong summary - schema break, missing evidence, invented code? Who reviews?
8. Model constraints: license, Llama vs Nemotron preference, LoRA ownership, how often the style changes.
9. What is next - agents (tool calls into policy/coverage systems), document images (faxes)?

## Platform (platform lead, security)
10. Clouds and regions per customer; existing Kubernetes (AKS/GKE/EKS/OKE/on-prem RKE2/OpenShift)?
11. GPU quota and reservations today; Spot tolerance for non-production?
12. GitOps/CI stack; who may run kubectl in production?
13. Secrets: Key Vault / Secret Manager / Secrets Manager / OCI Vault; workload identity in use?
14. Air-gapped sites: registry (Harbor/Artifactory), change windows, who carries images across the gap?
15. Observability: Prometheus/Grafana or Azure Monitor/Cloud Monitoring; on-call ownership?

## CSP partner SAs (Azure, Google)
16. Marketplace path for Plover (private offers) and for NVIDIA AI Enterprise (counts toward commit?).
17. GPU capacity outlook in the customer regions; reservation options; Spot/preemptible policy.
18. Managed options they will propose (Foundry managed compute, Vertex/Model Garden) - when are they the right answer?
