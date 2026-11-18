# Instance-selection worksheet (M2) - physics first, then prices, then measurements

## 1. The model (🔁 Lab 02)
- Weights fit? `params x bytes(precision) + extra + 2 GB overhead <= GPU memory x 0.9`.
- Decode step = **bytes moved / bandwidth** (weights + per-sequence KV + 2 x SSM state for hybrids) - but never
  faster than **2 x active params x batch / FLOPs** (the roofline: large batches turn decode compute-bound).
- Largest batch that meets the ITL SLO and fits memory -> tokens/s -> **$/1M tokens = $/h / (tok/s x 3,600) x 1e6**.
- TTFT ≈ 2 x active params x prompt tokens / (tensor FLOPs x 0.45).

## 2. Run it
```bash
python3 selector/instance_selector.py --spot                      # Llama 3.1 8B, 600-in/200-out, ITL 40 ms
python3 selector/instance_selector.py --spot --nvaie 1.0          # + NVIDIA AI Enterprise via marketplace
python3 selector/instance_selector.py --spot --isl 3000 --osl 400 --itl-slo-ms 30   # agent-step shape
python3 selector/fetch_prices.py                                  # refresh prices from public feeds (Azure/AWS/OCI)
```

## 3. What it says today (Spot, 2026-10-15 prices; estimates, not measurements)
| Instance | GPU | Precision | Decode floor | Batch @40 ms | tok/s | $/1M tok | + NVAIE $1/GPU-h |
|---|---|---|---|---|---|---|---|
| Azure NC40ads H100 v5 | H100 NVL | FP8 | 4.6 ms | 711 | 27,902 | $0.013 | $0.023 |
| Azure NC24ads A100 v4 (lab AKS) | A100 80GB PCIe | BF16 | 12.4 ms | 369 (compute roof) | 9,228 | $0.020 | $0.051 |
| Azure NC144lds RTX PRO 6000 v6 | RTX PRO 6000 | NVFP4 | 5.6 ms | 479 | 11,995 | $0.024 | $0.047 |
| GCP g4-standard-48 (lab GKE) | RTX PRO 6000 | NVFP4 | 5.6 ms | 479 | 11,995 | $0.038 | - |
| AWS g6e.xlarge (on-demand) | L40S | FP8 | 13.8 ms | 198 | 4,962 | $0.104 | - |
| GCP g2-standard-8 | L4 | FP8 | 35.8 ms | 11 | 277 | $0.512 | - |

Read it like an SA:
- **The L4 is vetoed in BF16** (53 ms floor > 40 ms SLO) and barely rescued by FP8 - and it is not on the NIM verified list.
- **On Spot, the license can cost more than the GPU**: +$1/GPU-hour turns $0.020 into $0.051 on the A100. For always-on
  capacity compare the annual subscription ($4,500/GPU/year ≈ $0.51/h at 100% use).
- **Verified is per model**: the A100 PCIe runs the generic profile fine (Labs 02/03) but is not on Llama 3.1 8B's
  verified list in NIM 2.0.13 - say so in the readout and plan production on a verified SKU.
- **Prices move**: `fetch_prices.py` shows what changed; Google has no keyless price feed (update by hand).

## 4. Calibrate with measurements
1. M3 bandwidth probe -> replace `BW_EFF = 0.80` with your measured read GB/s / datasheet GB/s.
2. M5 sweeps -> `bench/compare_clouds.py` prints measured vs predicted; a gap > 30% is a finding (wrong profile, CPU
   bottleneck in the API, a noisy node, or the selector's assumptions - write down which).
