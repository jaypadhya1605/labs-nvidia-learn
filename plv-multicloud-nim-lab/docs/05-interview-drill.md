# Interview drill - NVIDIA Senior Solutions Architect, AI Cloud Services

Answer out loud in under 90 seconds, then check the artifact. Fill the blanks with YOUR measured numbers.

## Five sentences per theme
**Multi-cloud deployment.** "I deployed one NIM-based product to AKS and GKE from the same Helm chart through Argo CD;
the per-cloud diff is a 20-line values file - storage class, tolerations, precision. Terraform builds both clusters;
GPU plumbing differs (operator-managed driver on AKS, Google's installer on GKE), the product does not. Secrets come
from Key Vault and Secret Manager through External Secrets with workload identity, so nothing sensitive is in Git.
From zero to a passing helm test took __ minutes per cloud."

**Choosing GPUs.** "I start with physics: decode reads every weight once per token, so an 8B BF16 model can't beat ~53 ms
per token on an L4 - that's a no for a 40 ms SLO before anyone buys anything. The selector ranks every instance by $/1M
tokens at the SLO, including the roofline at large batch and the NVAIE license. Then AIPerf replaces the estimates: on
AKS we measured __ tok/s vs __ predicted. And I check the NIM verified list per model - the A100 PCIe isn't on it for
Llama 3.1 8B, so production goes to a verified SKU."

**Engines.** "NIM picks an optimized profile per GPU - BF16 on Ampere, FP8 on Hopper/Ada, NVFP4 on Blackwell. I compared
it with trtllm-serve on the same GPU and preset: __. For non-LLM models I build TensorRT engines with Triton, and
I learned the hard way that engines are per GPU architecture - the portability drill shows the error and the two fixes."

**MLOps.** "Plover's style guide costs ~700 prompt tokens per call. I distilled it into a LoRA adapter: teacher outputs
filtered for schema and faithfulness, PEFT training tracked in MLflow, an eval gate, ORAS push to GHCR by digest, and a
Git commit that loads the adapter and routes 10% canary traffic by model name. Prompt tokens dropped __x; rollback is a
one-line commit."

**Data center.** "For the hospital I built an air-gapped RKE2 cluster: host driver, GPU Operator against RKE2's
containerd, a registry mirror so manifests keep their image names, and the model pre-cached with download-to-cache. The
image list is computed from the rendered charts - including the ClusterPolicy components people forget. No NGC key in
any pod, zero egress, verified."

**Customer cadence.** "Every week I send a sync pack: fleet status across clouds, releases from Git, upstream changes
from our pins, open issues, and the decisions we need. Today it included Nemotron 3.5: I sized it from its config in 30
minutes, ran it on AKS with the same chart, and put it through the same gates before saying anything to the customer."

## Likely questions
1. How do you decide between a managed endpoint and self-hosted NIM? (docs/06)
2. Walk me through what happens when a Spot GPU node is reclaimed. (M7, drill 5)
3. A customer says "our TTFT doubled after an upgrade" - first three commands? (drills README)
4. How does NVIDIA AI Enterprise licensing work on AWS/Azure/GCP marketplaces? ($1/GPU-hour + instance; or $4,500/GPU/year)
5. Why GitOps and not CI-driven deploys for GPU workloads?
6. How would you run the same thing on OKE or EKS? (stretch Terraform, docs/07)
7. Explain MIG vs time-slicing to a platform engineer. (🔁 Lab 02)
8. What is a NIM profile and how do you pick one in an air gap?
9. How do you measure prompt-injection resistance for an agent? (🔁 Lab 03, tests/agentic)
10. A new model launches today - what do you do before the customer call? (add-on)
