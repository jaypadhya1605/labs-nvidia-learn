# Sizing worksheet - predict first, then measure

Write your predictions BEFORE M3/M7 and compare. The gap between prediction and measurement is what you talk about
in interviews. Numbers below are first-principles estimates, not benchmarks.

## 1. Model facts (Llama-3.1-8B-Instruct)
| Item | Value | How |
|---|---|---|
| Parameters | 8.03 B | model card |
| Weights, BF16 | 16.06 GB (14.96 GiB) | 8.03e9 x 2 bytes |
| Weights, FP8 | ~8.5-9 GB | ~1 byte/param; embeddings and LM head often stay BF16 |
| KV cache per token, BF16 | 128 KiB | 2 (K,V) x 32 layers x 8 KV heads x 128 head dim x 2 bytes |
| KV per prior-auth request | ~0.8 GiB | (6,000 + 350) tokens x 128 KiB |
| KV per member-chat request | ~0.1 GiB | (600 + 200) tokens x 128 KiB |

## 2. Hardware facts
| | A100 80GB PCIe (NC A100 v4) | H100 NVL 94GB (NC H100 v5) |
|---|---|---|
| Peak HBM bandwidth | 1,935 GB/s | 3,900 GB/s |
| Dense tensor TFLOPS | 312 (BF16) | ~835 (BF16), ~1,670 (FP8) |
| Ridge point (FLOP/byte) | ~161 (BF16) | ~214 (BF16), ~428 (FP8) |
| MIG profiles used | 3g.40gb, 2g.20gb, 1g.10gb | 3g.47gb, 2g.24gb, 1g.12gb |
| FP8 | no | yes |

MIG splits memory into 8 slices and compute into 7. A slice's bandwidth scales with its **memory** slices:
3g = 4/8, 2g = 2/8, 1g = 1/8 of peak. Its compute scales with its **compute** slices: 3g = 3/7, 2g = 2/7, 1g = 1/7.

## 3. Decode is memory-bandwidth-bound
Each generated token streams all weights (plus the KV cache) from HBM, but does only ~2 FLOPs per weight per
sequence. Arithmetic intensity is about the batch size, far below the ridge point, so:
**time per token >= bytes read / bandwidth.** Batch-1 floor from weights alone:

| Slice | A100 BF16 floor | H100 NVL FP8 floor |
|---|---|---|
| Full GPU | 16.06 GB / 1,935 GB/s = **8.3 ms** (~120 tok/s) | ~9 GB / 3,900 GB/s = **2.3 ms** |
| 3g | / 968 GB/s = **16.6 ms** (~60 tok/s) | / 1,950 GB/s = **4.6 ms** |
| 2g | / 484 GB/s = **33 ms** (~30 tok/s) | / 975 GB/s = **9.2 ms** |

Real ITL is higher (achieved bandwidth ~80-90% of peak, KV reads, scheduling overhead, prefill chunks mixed in).
**Prediction A (A100):** member-chat ITL p95 on the 2g slice will sit near or above the 50 ms SLO beyond ~8-12
concurrent chats. M3 gives you measured bandwidth; M7 gives you measured ITL.

## 4. Prefill is compute-bound
Prefill FLOPs ~= 2 x params x prompt tokens + attention (~2 x n^2 x d_model x layers for causal attention).
For a 6,000-token prior-auth prompt: 2 x 8.03e9 x 6,000 + 2 x 6,000^2 x 4,096 x 32 ~= **~106 TFLOP**.

| Slice | Effective compute (50% MFU) | Predicted TTFT, no prefix reuse |
|---|---|---|
| A100 full | ~156 TFLOPS | ~0.7 s |
| A100 3g (3/7 of SMs) | ~67 TFLOPS | **~1.6 s** (SLO is 1.5 s) |
| H100 NVL 3g, FP8 | ~358 TFLOPS | ~0.3 s |

**Prediction B (A100):** prior-auth TTFT p95 misses the 1.5 s SLO on a 3g slice unless the shared policy prefix
(~half of each prompt) is served from the prefix cache. Prefix caching is a requirement here, not a tuning knob -
test it in M7 with `bench/run_sweep.sh prefix-priorauth` with caching on and off.

## 5. KV-cache capacity -> max concurrency
Usable KV ~= slice memory x gpu-memory-utilization - weights - activations/CUDA graphs (~1.2-1.8 GiB).

| Tenant on slice | KV budget | Tokens | Concurrent requests | Target |
|---|---|---|---|---|
| Prior-auth, A100 3g.40gb (util 0.90) | 39.4 x 0.90 - 15.0 - 1.5 ~= **19 GiB** | ~155k | ~24 | 12 - OK |
| Member chat, A100 2g.20gb (util 0.95, small batch buffers) | 19.6 x 0.95 - 15.0 - 1.4 ~= **2 GiB** | ~16k | ~15-20 | **32 - MISS** |
| Prior-auth, H100 3g.47gb (FP8 weights) | ~44 x 0.90 - 8.4 - 1.5 ~= **30 GiB** | ~245k | ~38 | 12 - OK |
| Member chat, H100 2g.24gb (FP8 weights) | ~22 x 0.90 - 8.4 - 1.5 ~= **10 GiB** | ~80k | ~100 | 32 - OK |

**Prediction C (A100):** an 8B BF16 model on a 20 GB slice cannot hold 32 concurrent chats. Options to put in the
readout: H100 NVL with FP8; a different geometry (e.g. 4g+3g, giving up the 1g slices); a smaller model for member
chat; or giving member chat the bigger slice during open-enrollment hours.

## 6. Energy and cost method (what the scripts compute)
- **Energy:** increase of `DCGM_FI_DEV_TOTAL_ENERGY_CONSUMPTION` (mJ) over the AIPerf profiling window, `max by (UUID)`
  per physical GPU (MIG repeats device-level fields on every instance series - summing overstates energy).
- **Attribution:** benchmarks run one tenant at a time and the whole GPU's energy is attributed to the run. Idle
  slices still draw power, so this is the honest marginal cost. For shared operation, split by slice share or by
  `DCGM_FI_PROF_GR_ENGINE_ACTIVE` per instance - and say which method you used.
- **tokens/J** = output tokens / joules.  **kWh per 1,000 decisions** = J / 3.6e6 / requests x 1,000.
- **$ per 1,000 decisions** = VM $/h x slice share / (SLO-compliant requests/s x 3,600) x 1,000.
  Example arithmetic only: 0.8 req/s on a 3g A100 Spot slice -> 0.6788 x 3/7 / 2,880 x 1,000 ~= $0.10.
  Same on PAYG ($3.673/h) ~= $0.55. Your measured req/s replaces 0.8.

## 7. Your predictions (fill in before measuring)
| Quantity | Prediction | Measured (M3/M7) | Why the gap? |
|---|---|---|---|
| Read bandwidth, full / 3g / 2g / 1g (GB/s) | | | |
| Prior-auth TTFT p95 at c=12, caching on / off | | | |
| Member ITL p95 at c=16 / c=32 | | | |
| Max member concurrency within SLO | | | |
| Tokens/J at the operating point | | | |
