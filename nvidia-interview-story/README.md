# NVIDIA interview story

**Live page:** https://jaypadhya1605.github.io/labs-nvidia-learn/

One healthcare research agent, re-engineered gate by gate from a hosted model call to NVIDIA
inference on Azure Kubernetes Service. I'm an Azure Solutions Engineer working with healthcare
payers and providers. This folder is the interview-ready view of that work: what was built,
what was measured, what broke, and what comes next.

> Personal learning project. Built on public documents and synthetic data. Customer names
> are fictional. Not a customer deployment and not an official Microsoft or NVIDIA resource.

## The problem

Northstar Provider Operations (fictional) must find what changed between the 2025 and 2026
editions of a health plan's public Marketplace provider manual: 343 pages in total. Answers
must be under 50 words, cite an edition and page, pass code checks that the cited page
contains the fact, and be approved by a person before any procedure changes.

Five golden checks act as traps: a real routing change, a new outside vendor, a new restriction
layered on an older rule, an ID that did not change, and a 2025 addendum rule that looks new in
2026. Expected answers are written independently and never shown to the model.

```text
Ask -> Azure AI Search (page-level, both editions, dated addenda) -> Nemotron (structured JSON)
    -> citation validation in Python -> human approval
```

## Six gates

| Gate | Focus | Status | Evidence / next step |
| --- | --- | --- | --- |
| G1 | Cited research agent: Python, Azure AI Search, hosted Nemotron, OpenTelemetry | **Passed** | 5/5 live golden checks, 18 unit tests, traces in Foundry |
| G2 | Quality and baseline: AI-Q blueprint, MCP, NeMo Guardrails, AIPerf, MLflow | **In progress** | AI-Q 2.2.1 running on AKS with Entra-only auth via a token-proxy sidecar (7 tests); NVIDIA embeddings (2048-d) with the key in Key Vault; pharmacy report prototype. Next: MCP, Guardrails, streaming TTFT/ITL, 20 reviewed cases |
| G3 | NVIDIA inference on AKS: NIM, NIM Operator, GPU scheduling, workload identity | **Infra verified** | NIM serving Nemotron 3.5 Lightning on an A100 80 GB Spot node with observed GPU activity. Next: NIMCache/NIMService, Helm, golden checks through the GPU path |
| G4 | GPU systems: CUDA kernel, Nsight, DCGM, trtllm-serve, Triton + TensorRT-LLM, vLLM vs NIM | Planned | Controlled comparison on one checkpoint with AIPerf and a quality gate |
| G5 | Security and recovery: hardened guardrails, 401/403 and isolation tests, failure drills | Planned | |
| G6 | Release: Terraform, Helm, GitHub Actions, Argo CD, MLflow gates, timed rollback | Planned | |

## What one A100 showed

Smoke-test observations from an AKS Spot node (A100 80 GB PCIe, driver 580, CUDA 13.0), not a benchmark:

- **The NIM ran on vLLM.** The selected profile was `vllm-int4-tp1-pp1` (vLLM 0.32.0), not TensorRT-LLM.
  Comparing that NIM with standalone vLLM would mostly measure packaging.
- **Ampere has no native FP8/FP4.** Low-precision weights ran weight-only through Marlin kernels.
- **75 GB of VRAM in use** was mostly KV cache that vLLM pre-allocates, not model size.
- **99% utilization at 246 W.** nvidia-smi utilization is time-based; DCGM SM activity and Nsight show real saturation.
- **0.67–0.82 s for 3 concurrent non-streaming requests.** This is not TTFT, which needs a streaming path and AIPerf.
- **About 3 minutes to ready,** plus a 10.8 GB image pull and a ~21 GB model cache. A NIMCache on a PVC fixes restarts.

## Lessons

1. **Agent fan-out is a capacity problem.** One AI-Q deep report made 49 model calls plus 20
   throttled retries. Hosted TPM limits count `max_tokens`. Fix: a dedicated deployment and a
   per-report call budget.
2. **A blueprint isn't proof of correctness.** The AI-Q report missed the pharmacy routing change
   that the keyword baseline caught. Independent golden answers now gate every variant.
3. **Fit the blueprint to the identity model.** AI-Q's provider takes only a static key, so a
   loopback sidecar swaps in refreshed Entra tokens from workload identity.
4. **Reasoning tokens are real tokens.** Hidden reasoning consumed the whole 8,192-token budget.
   Thinking is off for strict-JSON tasks; truncated answers fail.
5. **Ask which engine is under the package.** Engine comparisons hold weights, tokenizer, GPU and
   prompts constant, use ABBA ordering and at least three repetitions, and run quality first.
6. **Quota isn't capacity.** Check physical allocation at run time and scale GPUs to zero after use.

## Contents

| Path | What it is |
| --- | --- |
| [site/index.html](site/index.html) | The GitHub Pages interview page |
| [site/img/cards/](site/img/cards/) | Field notes: lab on a page, Azure → NVIDIA map, inference stack, A100 evidence, performance cheat sheet |
| [site/img/diagrams/](site/img/diagrams/) | Architecture diagrams for each gate. G5/G6 and the reference architecture are target designs |
| [cards-src/](cards-src/) | HTML sources for the field notes; `python cards-src/render.py` re-renders them with headless Edge |

Other lab guides in this repo: [kha-genai-inference-lab](../kha-genai-inference-lab/),
[lfy-secure-agent-lab](../lfy-secure-agent-lab/), [plv-multicloud-nim-lab](../plv-multicloud-nim-lab/),
[ttg-agentic-genai-lab](../ttg-agentic-genai-lab/).
