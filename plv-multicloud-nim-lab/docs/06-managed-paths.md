# Managed and marketplace paths (M10) - checked 2026-10-15

The JD: *work with cloud service providers to build and demonstrate solutions*. Customers will ask "why not the CSP's
managed option?" - answer with this matrix and a scorecard from `managed/compare.py`, not with opinions.
Verify before quoting: these change monthly. Hosted endpoints = **synthetic data only** unless a BAA covers PHI.

## What exists, per cloud
| Cloud | Managed NVIDIA path | Billing | Status / notes |
|---|---|---|---|
| Azure | **NIM on Foundry (classic) managed compute** - subscribe once per hub to the "NVIDIA NIM SaaS offer" (Marketplace) | Azure ML VM/hour **+ NVIDIA per-GPU-hour surcharge**, per minute; 90-day trial | [Learn](https://learn.microsoft.com/azure/foundry-classic/how-to/deploy-models-managed-pay-go) |
| Azure | **Foundry (new) managed compute**, runtime "NVIDIA NIM" (TensorRT-LLM backend, NIM API) | hourly per accelerator (A100/H100/MI300X) | **Preview, no SLA** - [Learn](https://learn.microsoft.com/azure/foundry/concepts/managed-compute-overview) |
| Azure | **Pay-per-token Nemotron via Fireworks on Foundry**: `FW-Nemotron-Lightning-3.5-30B-A3B`, `FW-Nemotron-3-Ultra-NVFP4` (GA) | per token | not NIM - [Learn](https://learn.microsoft.com/azure/foundry/how-to/fireworks/fireworks-model-catalog) |
| Azure | AKS + NIM | self-managed Helm (this lab) | [NVIDIA](https://docs.nvidia.com/nim/large-language-models/latest/deployment/csp-deployment/azure.html) |
| Google | **NIM on GKE** Marketplace app (pick region, GPU, model; ~15-20 min) | pricing/NVAIE terms not stated in docs - ask | [NVIDIA](https://docs.nvidia.com/nim/cloud/gke/latest/overview.html) |
| Google | **NIM as a Vertex AI custom container** (`gcloud ai models upload` -> endpoint) | consumption; can scale to zero | [NVIDIA](https://docs.nvidia.com/nim/large-language-models/latest/deployment/csp-deployment/google-cloud.html) |
| Google | **Nemotron in Model Garden** (Nemotron 3 Nano one-click into a private VPC; Nemotron 3 Super listed); Vertex is now "Gemini Enterprise Agent Platform" | - | [Google](https://cloud.google.com/blog/products/compute/google-cloud-ai-infrastructure-at-nvidia-gtc-2026) |
| AWS | **NVAIE AMI on Marketplace** (NIM, NeMo, Riva) | **$1/GPU-hour + EC2** | [AWS](https://aws.amazon.com/marketplace/pp/prodview-ozgjkov6vq3l6) |
| AWS | **Bedrock serverless**: `nvidia.nemotron-super-3-120b`, `nvidia.nemotron-nano-3-30b` (InvokeModel/Converse/OpenAI-compatible) | per token | [model cards](https://docs.aws.amazon.com/bedrock/latest/userguide/model-cards.html) |
| AWS | **Bedrock Marketplace / SageMaker JumpStart NIMs** (instance + listing price); Lightning on JumpStart since 2026-08-17 | instance + listing | [AWS](https://aws.amazon.com/blogs/machine-learning/nvidia-nemotron-3-5-lightning-now-available-in-amazon-sagemaker-jumpstart/) |
| Oracle | **NVAIE built into the OCI Console** ("Enable NVIDIA AI Enterprise" on bare-metal GPU shapes) | per GPU-hour, Universal Credits | [Docs](https://docs.oracle.com/en-us/iaas/Content/Compute/Tasks/nvidia-ai-enterprise.htm) |
| Oracle | OCI Data Science model deployment (NIM copied to OCIR); OCI Generative AI dedicated clusters | - | [Oracle](https://blogs.oracle.com/ai-and-datascience/oci-data-science-now-offers-nvidia-nemotron-models) |
| NVIDIA | **build.nvidia.com** hosted NIM endpoints (`https://integrate.api.nvidia.com/v1`) | free trial | **prototyping/testing only** (API Trial ToS); production needs NVAIE - [FAQ](https://docs.api.nvidia.com/nim/docs/product) |
| NVIDIA | **DGX Cloud Lepton** - GPU marketplace across partner clouds (dev pods, batch, NIM endpoints) | not published | status unclear (docs: early access) - [Docs](https://docs.nvidia.com/dgx-cloud/lepton/get-started/workspace) |

**NVIDIA AI Enterprise list price** (updated 2026-09-02): $4,500/GPU/year (1 yr), $13,500 (3 yr), $18,000 (5 yr),
$22,500 perpetual; **$1/GPU-hour + instance on AWS/Azure/GCP/OCI marketplaces** -
[licensing guide](https://docs.nvidia.com/ai-enterprise/planning-resource/licensing-guide/latest/pricing.html).
Break-even: $4,500 / 8,760 h ≈ $0.51/h -> the annual subscription wins above ~51% utilization of a GPU.

## Decision matrix (Plover)
| Question | Managed (Foundry / Vertex / Bedrock / SageMaker) | Self-hosted NIM (AKS/GKE/DC) |
|---|---|---|
| PHI in the customer's tenant / VPC? | possible (managed compute in their subscription) - check the BAA and region | yes, by construction |
| Disconnected hospital? | no | yes (M9) |
| Same artifact on every cloud? | no - three APIs, three catalogs | yes - one chart |
| LoRA style adapter? | varies by service | yes (NIM LoRA profiles) |
| Ops burden | low | medium (GitOps, drills, weekly sync) |
| Cost at Plover's volume | per-token or per-instance + surcharge | $/GPU-hour + NVAIE; cheapest at high utilization |
| Speed to first demo | fastest | ~1 day with this repo |

**Plover's answer:** prototype and benchmark against managed endpoints (same suites), ship self-hosted NIM where PHI,
on-prem or LoRA ownership matter - and let each CSP's marketplace carry NVAIE so it draws down the customer's commit.

## Hands-on
```bash
cp managed/endpoints.example.yaml managed/endpoints.yaml      # enable what you can reach; export the *_API_KEY vars
python3 managed/compare.py --endpoints managed/endpoints.yaml --self-hosted-summaries-per-hour <M5 number> --out results/managed.md
```
