# 05 - Parallelism math, then measure it on 8x H100 (M7)

**Goal:** on a whiteboard, compute memory per GPU, bytes on the wire and time for data (DP, ZeRO, FSDP), tensor (TP),
pipeline (PP), expert (EP) and context (CP) parallelism. Then measure NCCL, TP vs PP vs DP serving, and DDP vs FSDP2
training on one 8x H100 node, and write measured / predicted next to every number.
🔁 mock round Q1, Lab 02 (Dynamo, MIG), M2 `archmath`, M6 LoRA job, [docs/04](04-gpu-cluster-architecture.md) ladder.

**Time:** 1 h paper on a laptop + 2 h on one 8x H100 SXM node (~$32-49 per hour on demand, checked 2026-10-10;
check your provider).
**JD line:** "GPU cluster architecture and parallelism" and "training and fine-tuning".
**Tools:** `parallel/parallelism_calc.py` (math), `nccl_tests.sh` + `compare_nccl.py` (fabric), `sweep_parallel.sh`
(serving), `train_scaling.sh` + `scaling_report.py` (training). Every table in sections 4-5 is real calculator output.

## 0. Predict first (5 min, no tools)
1. Full fine-tune of Llama 3.1 8B with Adam: how many GB of model state? Does plain data parallelism fit on 80 GB?
2. Llama 3.1 70B FP8 decode: TP 8 uses 4x the GPUs of TP 2. How much faster is one stream?
3. A healthy 8x H100 SXM node: all-reduce bus bandwidth as a % of 450 GB/s?

## 1. Memory per GPU
**Model states: 16 bytes per trained parameter** (mixed precision Adam): BF16 weights 2 + BF16 gradients 2 + FP32
master weights 4 + Adam m 4 + Adam v 4. Llama 3.1 8B: 8.03B x 16 = 128 GB before one activation.

| Data-parallel flavor (N GPUs) | Weights | Gradients | Optimizer (12 B) | Bytes per parameter per GPU |
|---|---|---|---|---|
| Plain DP / DDP | full | full | full | 16 |
| ZeRO-1 | full | full | / N | 4 + 12/N |
| ZeRO-2 | full | / N | / N | 2 + 14/N |
| ZeRO-3 = FSDP full shard | / N | / N | / N | 16/N |

TP and PP divide all of it by t x p. MoE experts are divided by the expert-parallel size instead (EP reuses DP ranks).
LoRA: the frozen base costs 2 bytes per parameter (sharded only by FSDP); only adapters pay 16 bytes.

**Activations** (Korthikanti et al. 2022, BF16, per transformer layer, s = sequence, b = micro-batch, h = hidden,
a = heads):
- no parallelism: s x b x h x (34 + 5as/h) bytes. The 5as/h term is the attention score matrix.
- FlashAttention or selective recompute: the 5as/h term disappears.
- TP with sequence parallelism: everything / t. TP without SP: s x b x h x (10 + 24/t + 5as/(h t)).
- Full recompute: keep only the layer input, 2 x s x b x h per layer, and recompute the rest in backward.
- Pipeline 1F1B: the first stage holds min(p, m) micro-batches of its L/p layers.

## 2. Communication per step (each GPU, ring algorithms, message M)
| Parallelism | Pattern | Bytes per GPU per step |
|---|---|---|
| DP (DDP, ZeRO-1/2) | all-reduce of gradients, once per step, overlaps backward | 2(n-1)/n x 2 bytes x params |
| ZeRO-3 / FSDP | all-gather weights (forward), all-gather (backward), reduce-scatter gradients, per layer | 1.5x DDP |
| TP (Megatron) | 2 all-reduces per layer forward + 2 backward, of s x b x h x 2 bytes | 4 x L x 2(t-1)/t x M per micro-batch |
| TP at decode | 2 all-reduces per layer per token of batch x h x 2 bytes: tiny, so latency-bound | alpha x 2L per token |
| PP | send/receive activations between stages; bubble = (p-1) / (m + p-1) | s x b x h x 2 bytes per micro-batch |
| EP (MoE) | all-to-all dispatch + combine per MoE layer | tokens x k x h x bytes x (ep-1)/ep x 2 |
| CP (ring attention) | pass K,V chunks around the ring, c-1 times per layer | (s/c) x b x 2 x kv_heads x d x 2 bytes x (c-1) |

**Time on the wire (alpha-beta model):** t = messages x alpha + bytes / (link rate x efficiency).
Big messages pay bandwidth; small messages pay alpha (~10-25 us). Section 6 measures both on your node.

## 3. Time: 6 x N x T
Training FLOPs = 6 x parameters x tokens (2 forward + 4 backward); LoRA is ~4NT (no weight gradients for the
frozen base). Divide by GPUs x peak x MFU. Example: Llama-class 8B on 1T tokens, 64 H100 (989 TFLOPS BF16 dense),
40% MFU: 4.8e22 / (64 x 989e12 x 0.4) = 1.9e6 s = 22 days. Then add checkpoints, evals and failures.

## 4. Five worked examples (real calculator output)
```bash
python3 -m parallel.parallelism_calc --examples      # E1-E5 below
python3 -m parallel.parallelism_calc --links         # link rates and alpha it assumes
```

### E1 - the M6 LoRA job: Llama 3.1 8B, rank 16 on q,k,v,o,gate,up,down, 1x H100, seq 2048 x 4
Adapter parameters: 41,943,040 (0.52% of the model). 1.5M training tokens per epoch at 35% MFU.

| plan | GPUs | states GB/GPU | activations GB/GPU | total GB/GPU | fits | compute ms/step | DP comm ms | TP comm ms | PP comm ms | PP bubble | all comm ms | epoch time |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| LoRA r=16, no recompute | 1 | 16.7 | 36.5 | 53.2 | yes | 760 | 0 | 0 | 0.0 | 0% | 0 | 2.3 min |
| LoRA r=16, full recompute | 1 | 16.7 | 3.3 | 20.0 | yes | 760 | 0 | 0 | 0.0 | 0% | 0 | 2.3 min |

LoRA turns a 128 GB problem into 17 GB of states. Activations are now the biggest item: gradient checkpointing
cuts them 11x. It costs one extra forward pass, which the table does not charge: +2NT on LoRA's 4NT is ~50% more
compute (the familiar ~33% is for full training's 6NT).

### E2 - full fine-tune: Llama 3.1 8B on one 8x H100 node (NVLink 4), seq 4096 x 2 per GPU, full recompute
| plan | GPUs | states GB/GPU | activations GB/GPU | total GB/GPU | fits | compute ms/step | DP comm ms | TP comm ms | PP comm ms | PP bubble | all comm ms |
|---|---|---|---|---|---|---|---|---|---|---|---|
| DP (no sharding) | 8 | 128.5 | 3.3 | 131.8 | NO | 998 | 78 | 0 | 0.0 | 0% | 78 |
| ZeRO-1 | 8 | 44.2 | 3.3 | 47.5 | yes | 998 | 78 | 0 | 0.0 | 0% | 78 |
| ZeRO-2 | 8 | 30.1 | 3.3 | 33.4 | yes | 998 | 78 | 0 | 0.0 | 0% | 78 |
| ZeRO-3 / FSDP | 8 | 16.1 | 3.3 | 19.3 | yes | 998 | 118 | 0 | 0.0 | 0% | 118 |
| ZeRO-3 on PCIe Gen4 (L40S-style node) | 8 | 16.1 | 3.3 | 19.3 | yes | 998 | 1,649 | 0 | 0.0 | 0% | 1,649 |

Prediction 1: 128 GB of states, so plain DP does not fit. ZeRO-1 already fits. FSDP moves 1.5x the bytes of DDP:
fine on NVLink (118 ms against ~1 s of compute, and it overlaps). The PCIe row keeps the H100's compute to isolate
the link: 1.6 s of communication per step is more than the whole compute. A real L40S also computes ~2.7x slower
(~2.7 s per step), so there FSDP's traffic is ~60% of the step - prefer DDP-style plans (or LoRA) on PCIe nodes.

### E3 - pretraining: Llama 3.1 70B on 2,048 H100 (TP 8 inside a node, PP 4 x DP 64 over InfiniBand NDR), 15T tokens, 40% MFU
| plan | GPUs | states GB/GPU | activations GB/GPU | total GB/GPU | fits | compute ms/step | DP comm ms | TP comm ms | PP comm ms | PP bubble | all comm ms | days | GPU-hours |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| TP8 x PP4 x DP64, ZeRO-1, m=32 | 2048 | 9.2 | 22.8 | 32.0 | yes | 8,766 | 217 | 1,696 | 216.3 | 9% | 2,129 | 99 | 4.88 M |

Meta reports 7.0M H100-hours for Llama 3.1 70B. The 6NT estimate at 40% MFU gives 4.88M: the gap is lower real MFU,
restarts, evaluations and long-context stages. TP communication takes ~8x the DP time: that is why TP stays
inside NVLink.

### E4 - inference: Llama 3.1 70B FP8 decode at 8K context, TP 2 / 4 / 8 (NVLink 4 vs PCIe Gen4)
| link | TP | batch | weight+KV read ms | all-reduce ms | step ms | tok/s per seq | tok/s per GPU |
|---|---|---|---|---|---|---|---|
| nvlink4 | 2 | 1 | 10.93 | 1.61 | 12.54 | 80 | 40 |
| nvlink4 | 2 | 32 | 23.35 | 1.83 | 25.18 | 40 | 635 |
| nvlink4 | 4 | 1 | 5.47 | 1.61 | 7.08 | 141 | 35 |
| nvlink4 | 4 | 32 | 11.68 | 1.95 | 13.63 | 73 | 587 |
| nvlink4 | 8 | 1 | 2.73 | 1.61 | 4.35 | 230 | 29 |
| nvlink4 | 8 | 32 | 5.84 | 2.01 | 7.85 | 127 | 510 |
| pcie4 | 2 | 1 | 10.93 | 3.30 | 14.23 | 70 | 35 |
| pcie4 | 2 | 32 | 23.35 | 6.48 | 29.83 | 34 | 536 |
| pcie4 | 4 | 1 | 5.47 | 3.35 | 8.82 | 113 | 28 |
| pcie4 | 4 | 32 | 11.68 | 8.12 | 19.79 | 51 | 404 |
| pcie4 | 8 | 1 | 2.73 | 3.38 | 6.11 | 164 | 20 |
| pcie4 | 8 | 32 | 5.84 | 8.93 | 14.77 | 68 | 271 |

Prediction 2: TP 8 is 2.9x faster than TP 2 per stream, not 4x. 80 layers x 2 all-reduces = 160 tiny messages per
token, ~1.6 ms of pure latency that does not shrink with TP. Tokens per GPU fall as TP grows: buy TP for latency,
replicas for throughput. Over PCIe the all-reduce costs more than the weight read at TP 8.

### E5 - MoE decode: DeepSeek-V3 all-to-all per step, 128 tokens per GPU, FP8 dispatch + BF16 combine, 58 MoE layers
| placement | GB sent per GPU per step | all-to-all ms per step |
|---|---|---|
| EP 8 inside one 8x H200 node (NVLink 4; 671 GB FP8 does not fit 8x H100) | 1.12 | 4.3 |
| EP 64 across 8 nodes, IB NDR | 1.26 | 34.3 |
| EP 72 inside one GB200 NVL72 | 1.26 | 2.9 |

Production MoE serving overlaps the all-to-all with compute (two micro-batches, DeepEP-style kernels). The point
stands: across InfiniBand the all-to-all costs ~10x what it costs inside one NVLink domain. This is the NVL72 pitch
in one number ([docs/04](04-gpu-cluster-architecture.md) section 4).

### More combinations (your own plans)
```bash
python3 -m parallel.parallelism_calc --train --model qwen3-30b-a3b --dp 8 --ep 8 --cp 2 --zero 1 --seq 4096
```
| plan | GPUs | states GB/GPU | activations GB/GPU | total GB/GPU | fits | compute ms/step | DP comm ms | TP comm ms | PP comm ms | PP bubble | all comm ms |
|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen3-30b-a3b TP1 PP1 DP8 EP8 CP2 ZeRO-1 | 16 | 43.6 | 6.8 | 50.4 | yes | 104 | 36 | 0 | 0.0 | 0% | 71 |

EP all-to-all 11.27 GB/step (33.2 ms); CP ring 0.40 GB/step (1.1 ms). The CP ranks hold the same weights, so they join
the gradient all-reduce and the ZeRO shards: the sync group is DP x CP = 16. Without EP the same plan needs 145.0 GB of
states per GPU. A small-active MoE trains on few FLOPs, so the all-to-all becomes a third of the step.

## 5. Mock round Q1, completed
"Llama 3.1 70B FP8 on 2x H100 with TP 2: best-case decode speed per stream?" Three layers of the answer:
1. Weights: 35.3 GB per GPU / 3.35 TB/s = 10.5 ms (the mock-round answer).
2. Plus the KV read at 8K: 10.93 ms ([docs/03](03-model-complexity.md) section 7).
3. Plus 160 all-reduces at ~10 us: 12.54 ms, ~80 tokens/s per stream (E4).
Then measure: real kernels run at ~80-90% of bandwidth, so expect ~14 ms. Section 6 tells you your alpha.

## 6. On the 8x H100 node (2 h)
Setup (10 min): `scripts/gpu-vm-bootstrap.sh --pull vllm,pytorch` (Docker, NVIDIA toolkit, logins, pinned images).

### Step 1 - measure the fabric (20 min)
```bash
parallel/nccl_tests.sh 8                     # topology + all_reduce, all_gather, reduce_scatter, alltoall, 8 B-4 GB
python3 -m parallel.compare_nccl results/nccl/<run>/all_reduce_perf.log --link nvlink4
```
What the report looks like (run on `parallel/samples/all_reduce_perf_format_example.txt`, a hand-made FORMAT
EXAMPLE shaped like an 8x H100 node - not a measurement):

| size | time us | algbw GB/s | busbw GB/s | % of link |
|---|---|---|---|---|
| 8 B | 18.0 | 0.00 | 0.00 | 0% |
| 1 MB | 23.1 | 45.40 | 79.45 | 18% |
| 16 MB | 99.6 | 168.52 | 294.91 | 66% |
| 256 MB | 1,322.9 | 202.92 | 355.10 | 79% |
| 4 GB | 20,896.3 | 205.54 | 359.69 | 80% |

- Peak bus bandwidth 359.7 GB/s at 4 GB = 80% of the nvlink4 rate -> use EFFICIENCY = 0.80. (That is the ring shape;
  with NVLink SHARP on, busbw can read above 400 GB/s - compare `--nvls-off` before you set EFFICIENCY.)
- Bandwidth regime starts around 32 MB (90% of peak); below that you pay latency, not bytes.
- Smallest message took 18.0 us -> use alpha ~ 18 us for decode-time TP all-reduces.

Put YOUR efficiency and alpha into `LINKS` / `EFFICIENCY` in `parallel/parallelism_calc.py` and rerun E4.

### Step 2 - two fabric drills (15 min)
```bash
parallel/nccl_tests.sh 8 --nvls-off          # NVLink SHARP off: how much did in-switch reduction buy?
parallel/nccl_tests.sh 8 --p2p-off           # drill 5: no GPU peer-to-peer, traffic through host memory
python3 -m parallel.compare_nccl results/nccl/*/all_reduce_perf.log --link nvlink4    # side by side
```
Read the `NCCL INFO` lines the script prints: the transport changes from P2P/NVLS to SHM. Customer symptom of the
P2P case: "TP 8 is slower than TP 2" on a VM that hides NVLink or blocks peer access.

### Step 3 - TP vs PP vs DP for serving the agent (60 min)
```bash
export HF_TOKEN=...                  # Llama license accepted (or put it in env/lab.env)
parallel/sweep_parallel.sh 70b       # nvidia/Llama-3.1-70B-Instruct-FP8: TP 2, TP 4, TP 8, PP 2 x TP 4
parallel/sweep_parallel.sh 8b        # Llama 3.1 8B: TP 1, then TP 2, PP 2, DP 2 on 2 GPUs
python3 -m ttg.bench.ledger results/bench/vllm-70b-*.json results/bench/vllm-8b-*.json --slo-p95-s 12
```
Each configuration runs the real agent (`ttg.bench.agent_bench`, levels 1, 8, 32) and AIPerf (3,000 tokens in, 100
out, concurrency 32). `parallel/sweep_parallel.sh --dry-run` prints every command first. Predictions at 4K context:
```bash
python3 -m parallel.parallelism_calc --infer --model llama-3.1-70b --precision fp8 --tp 2,4,8 --batch 1,32 --ctx 4096
python3 -m parallel.parallelism_calc --infer --model llama-3.1-8b --tp 1,2 --batch 1,16,32 --ctx 4096
```
| config | GPUs | predicted ITL ms at c32 | measured ITL p50 ms (AIPerf) | agent p95 task s at c32 | $ per 1,000 tasks |
|---|---|---|---|---|---|
| 70B FP8 TP 2 | 2 | 18.8 | __ | __ | __ |
| 70B FP8 TP 4 | 4 | 10.4 | __ | __ | __ |
| 70B FP8 TP 8 | 8 | 6.2 | __ | __ | __ |
| 70B FP8 PP 2 x TP 4 | 8 | ~10.4 (two stages in series) | __ | __ | __ |
| 8B TP 1 | 1 | 9.6 | __ | __ | __ |
| 8B TP 2 | 2 | 5.5 | __ | __ | __ |
| 8B PP 2 | 2 | ~9.6 (no latency gain) | __ | __ | __ |
| 8B DP 2 (two replicas, 16 each) | 2 | 7.0 | __ | __ | __ |

What to look for:
- TP lowers inter-token latency; its gain flattens as all-reduce latency takes a bigger share (E4).
- PP does not lower per-token latency: stages run in series. It buys capacity across nodes, and throughput only when
  several micro-batches are in flight.
- DP doubles KV capacity and scales throughput linearly with no communication. On NVLink the calculator still
  predicts TP 2 above DP 2 at 32 concurrent sequences (5,828 vs 4,542 tokens/s), because decode is bandwidth-bound and
  TP splits the weight read. Your measurement decides; past ~64 concurrent agents, replicas usually win.
- For the agent, judge on p95 task latency at the SLO and $ per 1,000 tasks, not on tokens/s.

### Step 4 - DDP vs FSDP2 for the M6 LoRA job (25 min)
```bash
TRAIN_ARGS="<your M6 train_lora.py flags, with its max-steps flag set to ~60>" parallel/train_scaling.sh 1 2 4 8
python3 -m parallel.scaling_report results/scaling/          # the newest run under results/scaling/
```
The script runs the same job four ways: 1 GPU (`python3 finetune/train_lora.py`), DDP
(`accelerate launch --multi_gpu --num_processes N finetune/train_lora.py`) and FSDP2
(`accelerate launch --config_file finetune/accelerate_fsdp2.yaml --num_processes N finetune/train_lora.py`). The
per-GPU batch stays fixed (weak scaling). The plain M6 FSDP2 command is
`accelerate launch --config_file finetune/accelerate_fsdp2.yaml finetune/train_lora.py ...` (GPU count from the YAML);
the script only adds `--num_processes N` to vary N. Predictions for 8 GPUs:
```bash
python3 -m parallel.parallelism_calc --train --model llama-3.1-8b --lora-rank 16 --seq 2048 --mbs 4 --dp 8 --zero 0 --mfu 0.35
python3 -m parallel.parallelism_calc --train --model llama-3.1-8b --lora-rank 16 --seq 2048 --mbs 4 --dp 8 --zero 3 --mfu 0.35
```
| plan | GPUs | states GB/GPU | activations GB/GPU | total GB/GPU | fits | compute ms/step | DP comm ms | TP comm ms | PP comm ms | PP bubble | all comm ms |
|---|---|---|---|---|---|---|---|---|---|---|---|
| llama-3.1-8b TP1 PP1 DP8 EP1 CP1 ZeRO-0 LoRA r=16 | 8 | 16.7 | 36.5 | 53.2 | yes | 760 | 0 | 0 | 0.0 | 0% | 0 |
| llama-3.1-8b TP1 PP1 DP8 EP1 CP1 ZeRO-3 LoRA r=16 | 8 | 2.1 | 36.5 | 38.6 | yes | 760 | 80 | 0 | 0.0 | 0% | 80 |

DDP all-reduces only the 84 MB of LoRA gradients: expect ~95%+ scaling efficiency. FSDP2 shards the frozen 16 GB base
(16.7 -> 2.1 GB per GPU) but gathers it every forward and backward: ~80 ms of traffic per step, mostly hidden by
prefetch. For an 8B LoRA, DDP is enough. FSDP2 earns its keep when the frozen base does not fit (70B on 80 GB GPUs).

| run | samples/s | speedup | scaling efficiency | peak GiB per GPU |
|---|---|---|---|---|
| single-1gpu | __ | 1.00x | 100% | __ |
| ddp-8gpu | __ | __ | __ | __ |
| fsdp2-8gpu | __ | __ | __ | __ |

### Optional - the PCIe contrast on 4x L40S (30 min, a few dollars)
```bash
parallel/nccl_tests.sh 4 && python3 -m parallel.compare_nccl results/nccl/<run>/all_reduce_perf.log --link pcie4
python3 -m parallel.parallelism_calc --infer --model llama-3.1-8b --tp 1,2 --batch 1,32 --ctx 4096 --link pcie4 --gpu L40S
```
The calculator predicts 8B TP 2 at 10.3 ms per token on L40S (1.3 ms of it all-reduce) against 18.0 ms on one GPU.
Measure, then plug your alpha back in.

## Checkpoint
- [ ] From memory: 16 bytes per parameter; ZeRO-1/2/3 bytes per parameter; 2(n-1)/n; (p-1)/(m+p-1); 6NT.
- [ ] You can explain why TP 8 gives 2.9x over TP 2 for 70B decode, not 4x (E4).
- [ ] Your node's measured EFFICIENCY and alpha are in `parallelism_calc.py`, and E4 is rerun with them.
- [ ] The Step 3 and Step 4 tables are filled with measured numbers and the ledger has a row per configuration.
- [ ] You can say when to choose TP, PP, DP, EP and CP in one sentence each (section 8 of docs/04).

## Say it
"Tensor parallelism buys latency and costs an all-reduce on every layer of every token, so it lives inside NVLink.
Replicas buy throughput for free. On your 8-GPU node I would serve the 70B as two TP-4 replicas unless the latency SLO
needs TP 8 - and here is the measured table that says which."

## Whiteboard drills (60 seconds each)
1. **"Full fine-tune of a 70B on 8x H100: does it fit?"** 70.6B x 16 B = 1.13 TB of states; FSDP over 8 gives
   141 GB per GPU. No: shard over 16+ GPUs (two nodes), or use LoRA/QLoRA on one node.
2. **"Your FSDP job scales badly from 8 to 16 GPUs."** First the fabric: with 8x 400 Gb/s NICs and GPUDirect RDMA,
   NCCL spreads each collective over all 8 NICs and large all-gathers still reach ~360 GB/s of bus bandwidth across
   two nodes. One NIC in use, `NET/Socket` instead of `NET/IB`, or no GDRDMA in the `NCCL_DEBUG=INFO` log is 8-50x
   slower (fix `NCCL_IB_HCA`). Then latency: every per-layer collective now has twice the ring steps and network
   hops. Use hybrid sharding (shard inside the node, replicate across), bigger buckets and micro-batches.
3. **"PP 4 with 4 micro-batches: how much idle time?"** (4-1)/(4+4-1) = 43%. With 32 micro-batches: 9%.
4. **"How much KV does TP 8 save per GPU?"** KV heads split across GPUs: 8 KV heads / TP 8 = 1 head per GPU, so 1/8
   of the KV per GPU. Beyond TP = KV heads, KV gets replicated.
5. **"Why EP for MoE, not TP?"** Experts are independent matrices: EP places whole experts on GPUs and moves tokens
   (all-to-all) instead of splitting every matrix and all-reducing every layer.

## Sources
- [Megatron-LM: Shoeybi et al. 2019](https://arxiv.org/abs/1909.08053) (tensor parallelism, 2 all-reduces per layer
  forward); [Narayanan et al. 2021](https://arxiv.org/abs/2104.04473) (PTD-P, 1F1B, bubble (p-1)/m).
- [Korthikanti et al. 2022, Reducing Activation Recomputation](https://arxiv.org/abs/2205.05198) (34 + 5as/h, SP,
  selective recompute).
- [ZeRO: Rajbhandari et al. 2019](https://arxiv.org/abs/1910.02054) (16 bytes per parameter, stages 1-3);
  [PyTorch FSDP paper](https://arxiv.org/abs/2304.11277).
- [Scaling laws, Kaplan et al. 2020](https://arxiv.org/abs/2001.08361) (C = 6NT).
- [Llama 3.1 model card](https://huggingface.co/meta-llama/Llama-3.1-70B-Instruct) (7.0M H100-80GB GPU-hours for
  70B; 15T+ tokens).
- [nccl-tests PERFORMANCE.md](https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md) (algbw, busbw =
  algbw x 2(n-1)/n for all-reduce); [DeepEP](https://github.com/deepseek-ai/DeepEP) (overlapped expert all-to-all).
- [nvidia/Llama-3.1-70B-Instruct-FP8](https://huggingface.co/nvidia/Llama-3.1-70B-Instruct-FP8) (ModelOpt FP8, served
  with `vllm serve` directly); vLLM 0.31 parallel flags per the lab contract (README).
- Assumptions, not measurements: alpha 10-30 us per message and 80% NCCL efficiency in `LINKS`. Replace them with
  Step 1's numbers.
