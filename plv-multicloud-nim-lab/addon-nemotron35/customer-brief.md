# Nemotron 3.5 - what it means for Plover (customer brief, day 0)

**Bottom line:** keep Llama 3.1 8B + the LoRA adapter for summaries; start a 2-week evaluation of **Nemotron 3.5
Lightning** for the agent workflow and of **Super VL** for scanned/faxed clinical documents. Decide with Plover's
own gate (quality, injection resistance, $/1k), not with launch benchmarks.

## What launched (facts)
- **Super VL 120B-A12B** (2026-10-15): open weights, multimodal (text, image, document, video), hybrid Mamba-2 /
  attention MoE (512 experts, 22 active), 1M-token config window, NVFP4 checkpoint for Blackwell. NIM profiles:
  NVFP4 TP1/TP2, BF16 TP2/4/8 (LoRA).
- **Bolt 74B-A9B 24GB** (2026-10-15): pruned Super with 2-bit experts for workstations/PCs; Bolt E4B (edge) on 2026-10-23.
- **Lightning 30B-A3B** (since 2026-08-11): 256K native context, NIM with BF16/W4A16/NVFP4 profiles **with LoRA**.

## Why it matters for you (our measurements and physics, not marketing)
| | Llama 3.1 8B (today) | Lightning 30B-A3B | Super VL 120B-A12B |
|---|---|---|---|
| KV cache per token | 128 KiB | 6 KiB (+49 MB fixed state/seq) | 4 KiB FP8 (+170 MB state/seq) |
| Memory per sequence at 128K context | ~17 GB | ~0.85 GB | ~0.7 GB |
| Fits on | 1x A100 / L40S / RTX PRO 6000 | 1x A100 80GB (W4A16 recommended) | 1x RTX PRO 6000 96GB (NVFP4) |
| Selector estimate, 3K-in/400-out agent step, A100 Spot | 2,168 tok/s @30 ms ITL | ~6,900 tok/s (W4A16) | n/a (Blackwell NVFP4: ~480 tok/s @50 ms) |
| LoRA in NIM | yes (today's adapter) | yes (retrain the adapter) | BF16 profiles only (TP2+) |

Estimates are first-order (physics model, `selector/`); the evaluation replaces them with AIPerf measurements.

## Options
1. **Do nothing now** - Llama 8B + adapter meets the PoC SLOs; revisit at the next model review.
2. **Lightning for the agent (recommended to evaluate)** - long tool transcripts are where 6 KiB/token beats 128 KiB/token;
   retrain the style adapter on Lightning only if the agentic and summary gates pass.
3. **Super VL for document intake** - faxed/scanned records are a real prior-auth bottleneck; needs Blackwell
   (GKE G4 or Azure NCv6) and a security review of image handling.

## Unknowns we will close in the evaluation
Verified-GPU status for the new NIMs on A100 PCIe / RTX PRO 6000; the license text for production use; prompt-injection
resistance with thinking on vs off; cost at your real concurrency.

## 60-second version for the call
"Yesterday's Nemotron 3.5 release matters to you in one place first: the agent workflow. Lightning keeps about 20 times
less memory per token of context than Llama 8B, so long tool transcripts get much cheaper, and it already ships as a NIM
with LoRA support. We'll run it through the same quality and injection gates as your current model on your AKS
cluster - two weeks, synthetic data - and bring you a measured cost per thousand summaries before anything changes in
production."
