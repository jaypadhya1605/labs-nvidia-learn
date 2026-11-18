# A0 - Launch-day claims check: Nemotron 3.5 (2026-10-15)

Rule (the SA's credibility rule): **every number you repeat to a customer is either verified by you, or labeled
"NVIDIA-reported".** Config files and checkpoint sizes are evidence you can read in minutes; benchmarks are not.

| # | Claim (launch coverage) | Evidence you can check today | Status | How this lab checks it |
|---|---|---|---|---|
| 1 | Super VL is a 120B hybrid MoE with ~12B active | `config.json`: 88 layers = 40 Mamba-2 + 40 latent-MoE + 8 attention; 512 routed experts, 22 per token; NeMo AutoModel page: 121B (124B with MTP head) | **Verified (architecture)** | `selector/model_sizer.py` on `tests/fixtures/nemotron-3.5-super-vl-120b-a12b.config.json` |
| 2 | 1M-token context (Super VL) | `max_position_embeddings: 1048576` | **Verified (config)**; what a NIM profile *serves* is a separate check | `list-model-profiles` + `--max-model-len` |
| 3 | Lightning has 1M context | Config + NIM docs: native **262,144**; longer needs `VLLM_ALLOW_LONG_MAX_MODEL_LEN` and quality validation | **Partially true** - say "256K native" | config fixture |
| 4 | NVFP4 checkpoint for Blackwell, trained with quantization-aware distillation | `quantization_config`: experts NVFP4 (group 16), Mamba/attention FP8, **KV cache FP8**, embeddings/latent/router kept high precision; 75.01 GB on disk | **Verified (format)**; QAD accuracy = NVIDIA-reported | `addon-nemotron35/quant_report.py` |
| 5 | Bolt 74B-A9B "fits in 24 GB" via 2-bit | Experts **IQ2_XXS (2.0625 bits, GGML packing)**, Mamba projections W4A16 NVFP4, attention Q8_0; checkpoint **24.08 GB**; Puzzle-pruned (240-496 experts/layer, top-9) | **True for a 24 GiB workstation card at low concurrency; false for cloud "24 GB" GPUs** (L4/A10/A10G leave no room for state) | selector: `weights do not fit` on L4/A10; fits L40S |
| 6 | Bolt runs in llama.cpp, Ollama, LM Studio, vLLM | GGML IQ2 packing in the config | **Plausible**; not a NIM profile today | - |
| 7 | "Nearly 5x faster than other models in its class on a single GPU" | none public beyond NVIDIA's charts | **NVIDIA-reported** | A4: measure YOUR traffic with AIPerf |
| 8 | Dynamo + KV-aware routing: 8x users, -85% TTFT on 16 B200 | NVIDIA-reported; needs B200s | **NVIDIA-reported** | 🔁 Lab 02 ran the same mechanism (Dynamo KV routing) at small scale |
| 9 | Context reuse preserves attention caches AND Mamba states | consistent with the architecture (fixed SSM state per sequence) | **Plausible, architecture-consistent** | - |
| 10 | Available on NIM | NIM 2.0.13 support matrix lists `nemotron-3.5-super-vl-120b-a12b` (NVFP4 TP1/TP2; BF16 TP2/4/8 with LoRA) and `nemotron-3.5-lightning` (BF16/W4A16/NVFP4, with and without LoRA); Lightning container `nemotron-3.5-lightning-30b-a3b` documented with tag `2.0.9-variant` | **Verified (docs)**; verified-GPU lists for these models: check before quoting | `scripts/check-images.sh`, NGC catalog |
| 11 | Day-0 on clouds | Lightning (released **2026-08-11**): OpenRouter, Microsoft Foundry via Fireworks (GA), SageMaker JumpStart (2026-08-17), OCI Enterprise AI. Super VL: OpenRouter, Hugging Face (card points to a NIM). Bolt: Hugging Face only | **Mixed** - see docs/06-managed-paths.md | `managed/compare.py` |

Sources: SiliconANGLE launch article (2026-10-15); NVIDIA NeMo AutoModel model page for Super VL; NIM LLM 2.0.13 support
matrix and the Lightning get-started guide (NIM 2.0.10 docs); Hugging Face model repos (config.json, file sizes);
OpenRouter model pages; Microsoft Learn (Foundry Fireworks catalog); AWS ML blog (JumpStart); Oracle AI blog (Aug 2026).
