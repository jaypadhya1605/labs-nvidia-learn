# 03 - Model architectures and their computational complexity (M2)

**Goal:** size any model from its `config.json` in 30 minutes: parameters, active parameters, bytes per decode step,
KV cache per token, FLOPs. Then say how many GPUs it needs and what bounds it: memory bandwidth or compute.
🔁 Lab 01 roofline, Lab 02 KV math, Lab 04 model sizer (Nemotron 3.5 Lightning), mock round Q1.

**Time:** 90 min on a laptop. Optional: +30 min on 1x H100 to measure MoE decode against the prediction (section 8).
**JD line:** "Model architectures and their computational complexity."
**Tool:** `archmath/archmath.py` - pure Python, no GPU, no torch. Every number below is its real output.

## 0. Predict first (5 min, no tools)
Write your answers down. Section 2 checks them.
1. KV bytes per token for Llama 3.1 8B in BF16? For DeepSeek-V3 (671B)?
2. Qwen3-30B-A3B has 3.3B active parameters. Is it faster than dense Llama 3.1 8B at batch 1? At batch 32?
3. At 128K context, what share of a Llama 3.1 8B decode step is attention?

## 1. Six formulas size any decoder
| Quantity | Formula | Llama 3.1 8B |
|---|---|---|
| Parameters | embeddings + L x (attention + MLP or experts) + lm_head | 8.03B |
| Weight bytes | parameters x bytes: BF16 2, FP8 1, NVFP4 0.5625 (4 bits + one FP8 scale per 16 values) | 16.1 GB BF16 |
| KV per token (GQA) | 2 (K and V) x layers x KV heads x head_dim x bytes | 2 x 32 x 8 x 128 x 2 B = 128 KiB |
| Decode FLOPs per token | 2 x active matmul parameters + attention over the context | 19.3 GFLOP at 8K |
| Prefill FLOPs | 2 x parameters x tokens + causal attention (~ tokens squared) | 44.2 TFLOP for 3,000 tokens |
| Step time (roofline) | max(bytes read / bandwidth, FLOPs / peak) | 4.64 ms on H100 SXM at 4K context |

The rule from Lab 01: decode reads every weight once per step, for the whole batch. At batch 1 that is about 1 FLOP
per byte. The H100 SXM ridge point is 989 TFLOPS / 3.35 TB/s = 295 FLOP per byte. So decode is bandwidth-bound until
the batch is in the hundreds. Prefill is compute-bound after a few hundred tokens.

Two details the naive formula misses (both matter on a whiteboard):
- The input-embedding table is a lookup, not a matmul. Decode reads b rows of it, not 1.05 GB. So Llama 3.1 8B
  streams 15.0 GB per step, not 16.1 GB.
- Every sequence's KV cache is read every step. At batch 32 and 4K context that is 17.2 GB: more than the weights.

## 2. Run the calculator (10 min)
```bash
python3 -m archmath.archmath                     # every bundled config (archmath/configs/*.json)
python3 -m archmath.archmath --notes             # the assumptions behind every number
python3 -m archmath.archmath --config Qwen/Qwen3-30B-A3B-Instruct-2507   # any HF repo id or local config.json
```
Real output (2026-10-17):

| model | architecture | total B | active B | weights GB bf16 / fp8 / nvfp4 | KV KiB/token (bf16) | fixed state MB/seq | decode GFLOP/token @8K | max context |
|---|---|---|---|---|---|---|---|---|
| deepseek-v3 | MLA, MoE 8/256 + shared | 671.0 (+11.6 MTP) | 37.6 | 1,342.1 / 672.9 / 380.1 | 68.6 | - | 212.4 | 163,840 |
| llama-3.1-70b | dense, GQA 64:8 | 70.6 | 70.6 | 141.1 / 72.7 / 42.7 | 320.0 | - | 160.5 | 131,072 |
| llama-3.1-8b | dense, GQA 32:8 | 8.0 | 8.0 | 16.1 / 9.1 / 6.0 | 128.0 | - | 19.3 | 131,072 |
| llama-3.2-11b-vision | dense, GQA 32:8, cross-attention VLM (8 xattn layers) | 10.6 | 10.6 | 21.2 / 12.5 / 8.7 | 128.0 | - | 22.8 | - |
| mixtral-8x7b | GQA 32:8, MoE 2/8 | 46.7 | 12.9 | 93.4 / 47.0 / 26.6 | 128.0 | - | 29.8 | 32,768 |
| nemotron-3.5-lightning-30b-a3b | hybrid: 23 Mamba-2 + 6 attention, GQA 32:2, MoE 6/128 + shared | 31.6 (+1.3 MTP) | 3.6 | 63.2 / 32.3 / 18.8 | 6.0 | 49.1 | 7.3 | 262,144 |
| qwen2.5-vl-7b | dense, GQA 28:4, decoder-only VLM | 8.3 | 8.3 | 16.6 / 10.1 / 7.2 | 56.0 | - | 17.4 | 128,000 |
| qwen3-30b-a3b | GQA 32:4, MoE 8/128 | 30.5 | 3.4 | 61.1 / 31.2 / 18.1 | 96.0 | - | 12.5 | 40,960 |

Sanity checks against the model cards: DeepSeek-V3 671B total / 37B active; Mixtral 46.7B / 12.9B; Qwen3-30B-A3B
30.5B / 3.3B; Qwen2.5-VL-7B 8.29B; Lightning 32.9B with the MTP head (🔁 Lab 04's sizer: same 6 KiB and 49.1 MB).
Two small gaps you can explain: Qwen3's 3.4B counts both the embedding and the output head as "active" (the card's
3.3B leaves the lookup out), and DeepSeek's checkpoint is 685B = 671B + 14B of MTP module, because the MTP layer
stores its own copy of the embedding and output head (~1.9B) on top of the 11.6B the tool counts.
FP8 and NVFP4 columns keep embeddings, lm_head, norms and routers in BF16, as quantized checkpoints usually do.

**Checkpoint:** your prediction 1 should read 128 KiB (Llama 3.1 8B) and 68.6 KiB (DeepSeek-V3).

## 3. The architectures, one at a time (30 min)

### Dense with GQA: Llama 3.1 8B and 70B
Grouped-query attention shares each K/V head across several query heads: 32 query heads, 8 KV heads in the 8B.
KV per token drops 4x versus full multi-head attention (128 KiB instead of 512 KiB). FLOPs do not change.
The 70B has 80 layers and still 8 KV heads: 320 KiB per token. 🔁 Lab 02 KV math.

### MLA: DeepSeek-V3
Multi-head latent attention caches one compressed latent per layer instead of per-head K and V:
(512 latent + 64 RoPE key) x 61 layers x 2 bytes = 68.6 KiB per token for a 671B model. That is less than the
dense 8B's 128 KiB. The price is FLOPs: decode attention runs in the "absorbed" form (FlashMLA), so attention is
66% of a DeepSeek-V3 decode step at 8K context (section 5), against 22% for Llama 3.1 8B.

### MoE: Mixtral 8x7B, Qwen3-30B-A3B, DeepSeek-V3
A router sends each token to k of E experts. Memory follows TOTAL parameters; FLOPs per token follow ACTIVE ones.
Bytes per decode step follow neither: they follow the experts the whole batch touches. With uniform routing a layer
touches U(b) = E x (1 - (1 - k/E)^b) experts at batch b.
```bash
python3 -m archmath.archmath --coverage --batch 1,8,32,64,256
```
| model | batch | experts touched per layer | routed weights read | weights GB/step (bf16) | share of all weights |
|---|---|---|---|---|---|
| qwen3-30b-a3b | 1 | 8.0 of 128 | 6% | 6.1 | 10% |
| qwen3-30b-a3b | 8 | 51.6 of 128 | 40% | 25.8 | 43% |
| qwen3-30b-a3b | 32 | 111.8 of 128 | 87% | 53.1 | 88% |
| qwen3-30b-a3b | 64 | 125.9 of 128 | 98% | 59.5 | 98% |
| deepseek-v3 | 1 | 8.0 of 256 | 3% | 73.3 | 5% |
| deepseek-v3 | 32 | 163.3 of 256 | 64% | 866.7 | 65% |
| deepseek-v3 | 64 | 222.4 of 256 | 87% | 1,168.8 | 87% |
| mixtral-8x7b | 8 | 7.2 of 8 | 90% | 84.1 | 90% |
| nemotron-3.5-lightning-30b-a3b | 32 | 100.5 of 128 | 78% | 49.8 | 80% |

(Rows trimmed; the command prints every model at every batch.) Real routers are skewed toward hot experts, so
measured coverage is lower. The trend is the point: by batch 32 an "A3B" model reads most of its 30B weights every
step. That is why large MoE serving spreads experts over many GPUs (expert parallelism, [docs/05](05-parallelism-math.md))
and why NVL72's 72-GPU NVLink domain matters ([docs/04](04-gpu-cluster-architecture.md)).

### Hybrid Mamba-2 + MoE: Nemotron 3.5 Lightning 🔁 Lab 04
23 Mamba-2 layers keep a fixed-size state per sequence; only 6 attention layers (2 KV heads) grow a KV cache.
KV is 6 KiB per token, plus 49.1 MB of fixed state per sequence. At 4K context the state is bigger than the KV
(24 MiB). At 128K the KV is 768 MiB, where Llama 3.1 8B needs 16 GiB. The state is read AND written every step,
so at batch 32 it is 3.1 GB of traffic per step, more than the KV.

### VLMs: decoder-only vs cross-attention (details in M8, [docs/06](06-multimodal.md))
Qwen2.5-VL puts image tokens into the text sequence: a 12 MP photo is 15,552 tokens at 56 KiB each.
Llama 3.2 Vision keeps them out: one `<|image|>` token in the text, up to 4 x 1,601 vision tokens read through 8
cross-attention layers. Same question, two very different KV and prefill bills.

## 4. The decode roofline per architecture (15 min)
```bash
python3 -m archmath.archmath --roofline --gpu H100-SXM --batch 1,32 --ctx 4096 \
  --model llama-3.1-8b,qwen3-30b-a3b,nemotron-3.5-lightning-30b-a3b,mixtral-8x7b,deepseek-v3
```
Decode roofline on 1x H100-SXM, BF16 weights, BF16 KV, context 4,096 tokens (ridge point 295 FLOP/byte):

| model | batch | weights GB/step | KV+state GB/step | FLOP/byte | t_mem ms | t_compute ms | bound | tok/s per seq | tok/s total | fits |
|---|---|---|---|---|---|---|---|---|---|---|
| llama-3.1-8b | 1 | 15.01 | 0.54 | 1 | 4.64 | 0.02 | memory | 215 | 215 | yes |
| llama-3.1-8b | 32 | 15.01 | 17.18 | 17 | 9.61 | 0.56 | memory | 104 | 3,330 | yes |
| qwen3-30b-a3b | 1 | 6.08 | 0.40 | 1 | 1.94 | 0.01 | memory | 516 | 516 | yes |
| qwen3-30b-a3b | 32 | 53.09 | 12.88 | 5 | 19.69 | 0.30 | memory | 51 | 1,625 | yes |
| nemotron-3.5-lightning-30b-a3b | 1 | 6.46 | 0.12 | 1 | 1.96 | 0.01 | memory | 509 | 509 | yes |
| nemotron-3.5-lightning-30b-a3b | 32 | 49.81 | 3.95 | 4 | 16.05 | 0.22 | memory | 62 | 1,994 | yes |
| mixtral-8x7b | 1 | 25.50 | 0.54 | 1 | 7.77 | 0.03 | memory | 129 | 129 | NO |
| mixtral-8x7b | 32 | 93.13 | 17.18 | 8 | 32.93 | 0.89 | memory | 30 | 972 | NO |
| deepseek-v3 | 1 | 73.25 | 0.29 | 2 | 21.95 | 0.14 | memory | 46 | 46 | NO |
| deepseek-v3 | 32 | 866.70 | 9.21 | 5 | 261.47 | 4.62 | memory | 4 | 122 | NO |

How to read it:
- Every row is memory-bound. FLOP/byte stays far below 295, even at batch 32.
- Prediction 2: at batch 1 the A3B MoE is 2.4x faster than dense 8B (516 vs 215 tokens/s). At batch 32 it is 2x
  SLOWER per token (19.7 ms vs 9.6 ms), because it reads 53 GB of experts per step against 15 GB.
- The hybrid wins at batch 32 among the 30B MoEs: small KV, and its state traffic grows slower than KV would.
- "fits NO" means the BF16 weights alone exceed 90% of 80 GB: Mixtral needs 2 GPUs or FP8. DeepSeek-V3 does not fit
  one 8x H100 node even in FP8 (673 GB of weights against 640 GB of HBM, before any KV): plan an 8x H200 node
  (1,128 GB), an 8x B200 node, or two H100 nodes.
- Naive check (🔁 Lab 01): 16.06 GB / 3.35 TB/s = 4.8 ms for the 8B. The tool says 4.64 ms: 1.05 GB less
  (embedding gather) and 0.54 GB more (KV at 4K). Real kernels reach ~80-90% of bandwidth: expect ~5.5 ms measured.

## 5. Context length changes the bill (10 min)
```bash
python3 -m archmath.archmath --attention-share --model llama-3.1-8b,llama-3.1-70b,qwen3-30b-a3b,deepseek-v3,nemotron-3.5-lightning-30b-a3b
```
| model | 8K prefill / decode | 32K prefill / decode | 128K prefill / decode |
|---|---|---|---|
| llama-3.1-8b | 13% / 22% | 38% / 53% | 71% / 82% |
| llama-3.1-70b | 7% / 13% | 24% / 38% | 56% / 71% |
| qwen3-30b-a3b | 37% / 51% | 70% / 81% | 90% / 94% |
| deepseek-v3 | 22% / 66% | 53% / 88% | 82% / 97% |
| nemotron-3.5-lightning-30b-a3b | 6% / 11% | 22% / 33% | 53% / 66% |

Prediction 3: 82% of a Llama 3.1 8B decode step is attention at 128K. One 131,072-token sequence holds 16 GiB of KV.
Small-active MoEs hit the wall first (Qwen3: 51% at 8K), because their MLP share is tiny.
The hybrid has the lowest share: only 6 of its layers attend. Levers at long context: chunked prefill, FP8 KV cache,
KV offload, context parallelism ([docs/05](05-parallelism-math.md)) and hybrid SSM models.

## 6. Prefill and the agent's TTFT (10 min)
Each Tern agent step sends ~2,000-3,500 tokens (a ~1,840-token shared system prompt plus the conversation so far).
```bash
python3 -m archmath.archmath --prefill 3000 --gpu H100-SXM \
  --model llama-3.1-8b,llama-3.1-70b,qwen3-30b-a3b,nemotron-3.5-lightning-30b-a3b,qwen2.5-vl-7b
```
| model | TFLOP | ms @100% | ms @50% | KV written MB (bf16) |
|---|---|---|---|---|
| llama-3.1-8b | 44.2 | 44.7 | 89.5 | 393.2 |
| llama-3.1-70b | 422.5 | 427.2 | 854.4 | 983.0 |
| qwen3-30b-a3b | 19.9 | 20.1 | 40.3 | 294.9 |
| nemotron-3.5-lightning-30b-a3b | 17.9 | 18.1 | 36.2 | 18.4 |
| qwen2.5-vl-7b | 41.0 | 41.4 | 82.8 | 172.0 |

An agent task makes ~6.5 LLM calls, so uncached prefill alone costs ~0.6 s per task on the 8B at 50% MFU.
With prefix caching only the new ~150-300 tokens per step are computed: about 10x less. That is the M4-M5 story,
and why a timestamp at the top of the system prompt (drill 1) is so expensive.

## 7. Mock round Q1, one more rep (5 min)
"Llama 3.1 70B in FP8 on 2x H100 with tensor parallelism 2: best-case decode speed per stream?"
```bash
python3 -m archmath.archmath --roofline --model llama-3.1-70b --precision fp8 --gpus 2 --batch 1,32 --ctx 8192
```
| model | batch | weights GB/step | KV+state GB/step | FLOP/byte | t_mem ms | t_compute ms | bound | tok/s per seq | tok/s total | fits |
|---|---|---|---|---|---|---|---|---|---|---|
| llama-3.1-70b | 1 | 35.28 | 1.34 | 2 | 10.93 | 0.04 | memory | 91 | 91 | yes |
| llama-3.1-70b | 32 | 35.28 | 42.95 | 33 | 23.35 | 1.30 | memory | 43 | 1,370 | yes |

The mock-round answer (35.3 GB / 3.35 TB/s = 10.5 ms, ~95 tokens/s) was right on weights. A complete answer adds
the KV read at 8K (+0.4 ms) and the 160 tensor-parallel all-reduces per token (+1.6 ms, [docs/05](05-parallelism-math.md)
E4): ~12.5 ms, ~80 tokens/s per stream.

## 8. Optional on 1x H100 (30 min): measure MoE vs dense decode
Prediction from section 4: dense 8B goes from 4.6 ms to 9.6 ms per token between batch 1 and 32. Qwen3-30B-A3B goes
from 1.9 ms to 19.7 ms. Measure inter-token latency (ITL) at the same shape (4,000 tokens in, 256 out):
```bash
MODEL=Qwen/Qwen3-30B-A3B-Instruct-2507 engines/launch.sh vllm base     # 61 GB BF16: fits one 80 GB H100
for c in 1 8 32; do
  aiperf profile --model ttg-llm --url http://localhost:8000 --endpoint-type chat --streaming \
    --tokenizer Qwen/Qwen3-30B-A3B-Instruct-2507 --concurrency "$c" --request-count $((c * 8)) \
    --warmup-request-count "$c" --isl 4000 --isl-stddev 0 --osl 256 --osl-stddev 0 \
    --extra-inputs ignore_eos:true --artifact-dir "results/m2/qwen3-moe-c$c" --ui-type simple
done
engines/stop.sh
engines/launch.sh vllm base                                           # dense Llama 3.1 8B: repeat the loop
```
Use a different `--isl` from the agent's system prompt so prefix caching cannot hide anything. Check the engine log
before you trust the batch-32 row: vLLM prints the maximum concurrency its KV cache holds at this length. Qwen3-30B-A3B
in BF16 (61 GB) at the default `GPU_MEM_UTIL=0.90` leaves ~13 GB for KV, and 32 x 4,256 tokens needs ~13.1 GB, so
"batch 32" can quietly run as ~31 plus a queue. If the log says less than 32, use `--isl 3500` or `GPU_MEM_UTIL=0.93`.
🔁 Lab 04: the same loop against the Nemotron 3.5 Lightning NIM (`PIN_NIM_LIGHTNING_IMAGE:PIN_NIM_LIGHTNING_TAG`)
adds the hybrid row.

| model | batch | predicted ms/token | measured ITL p50 ms | measured / predicted |
|---|---|---|---|---|
| llama-3.1-8b | 1 | 4.64 | __ | __ |
| llama-3.1-8b | 32 | 9.61 | __ | __ |
| qwen3-30b-a3b | 1 | 1.94 | __ | __ |
| qwen3-30b-a3b | 32 | 19.69 | __ | __ |

Expect measured / predicted around 1.1-1.4 for dense (bandwidth efficiency, kernel launches) and below the MoE
prediction at batch 32 if routing is skewed. Write your ratio next to every future estimate.

## Checkpoint
- [ ] KV per token from memory: 128 KiB (8B), 320 KiB (70B), 68.6 KiB (DeepSeek-V3 MLA), 6 KiB + 49.1 MB (Lightning).
- [ ] You can explain why the A3B MoE beats dense 8B at batch 1 and loses at batch 32 (expert coverage).
- [ ] You can name the attention share of an 8B decode step at 8K and 128K (22%, 82%).
- [ ] You can size a new model from its config.json with `--config <repo id>` in under 30 minutes.
- [ ] (GPU) Your measured / predicted ratios are in the section 8 table.

## Say it
"Before we pick GPUs, I size the model from its config: total and active parameters, bytes per decode step, KV per
token at your context, and where batch 32 lands on the roofline. A '3B-active' MoE is cheap at batch 1 and reads
most of its 30 billion weights at batch 32, so it needs a different GPU plan than a dense model."

## Whiteboard drills (60 seconds each)
1. **"How much KV cache does Llama 3.1 70B need for 32 users at 8K?"** 2 x 80 x 8 x 128 x 2 B = 320 KiB per token.
   x 8,192 x 32 = 80 GiB, as much as one H100. FP8 KV halves it.
2. **"Why is decode memory-bound?"** Each step reads every weight for ~1 FLOP per byte at batch 1. The H100's ridge
   is 295 FLOP/byte. Batching raises intensity until KV reads take over.
3. **"MLA vs GQA?"** MLA caches a 576-dim latent per layer: 68.6 KiB per token for 671B parameters. It shrinks KV
   ~5x versus the 70B's GQA, but absorbed decode attention costs more FLOPs.
4. **"A customer wants 128K context on an 8B model. What changes?"** 16 GiB of KV per sequence, attention 82% of
   decode FLOPs, quadratic prefill. Use chunked prefill, FP8 KV, prefix caching, maybe context parallelism.
5. **"Mamba vs attention for serving?"** A fixed state instead of a growing KV: memory flat in context, cheap long
   prompts. Hybrids keep a few attention layers for recall, so they still grow some KV.

## Sources
- Configs (verified 2026-10-17, see each file's `_source`): [meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)
  (values via the ungated mirror unsloth/Llama-3.1-8B-Instruct), [Llama-3.1-70B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-70B-Instruct),
  [deepseek-ai/DeepSeek-V3](https://huggingface.co/deepseek-ai/DeepSeek-V3), [mistralai/Mixtral-8x7B-v0.1](https://huggingface.co/mistralai/Mixtral-8x7B-v0.1),
  [Qwen/Qwen3-30B-A3B](https://huggingface.co/Qwen/Qwen3-30B-A3B), [Qwen/Qwen3-30B-A3B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507)
  (30.5B total, 3.3B active, 48 layers, 32 Q / 4 KV heads, 128 experts, 8 active),
  [Qwen/Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct),
  [meta-llama/Llama-3.2-11B-Vision-Instruct](https://huggingface.co/meta-llama/Llama-3.2-11B-Vision-Instruct),
  nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16 (🔁 Lab 04 fixture).
- [DeepSeek-V3 technical report](https://arxiv.org/abs/2412.19437) (MLA, MoE 256 routed + 1 shared, 671B / 37B).
- [Mixtral of Experts](https://arxiv.org/abs/2401.04088) (46.7B total, 12.9B active).
- [GQA paper](https://arxiv.org/abs/2305.13245); [Mamba-2 / SSD](https://arxiv.org/abs/2405.21060).
- [NVIDIA H100](https://www.nvidia.com/en-us/data-center/h100/) (H100 SXM: 3.35 TB/s, 989 TFLOPS dense BF16 = half the
  sparse 1,979 figure); [GB200 NVL72](https://www.nvidia.com/en-us/data-center/gb200-nvl72/) for the B200 row.
- [Qwen/Qwen3-30B-A3B-Instruct-2507 vLLM usage](https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507) (`vllm serve`).
