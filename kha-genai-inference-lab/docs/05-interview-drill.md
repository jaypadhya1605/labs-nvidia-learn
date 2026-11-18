# Interview drill - say it without notes

For each module, practice the five sentences out loud, then answer the questions. Use YOUR measured numbers.

## Five sentences per module
**M0 framing.** "The customer had four GPUs, no telemetry, and a time-slicing incident. I turned three business asks
into SLOs: TTFT and inter-token latency per tenant. I sized from first principles before touching hardware: decode is
bandwidth-bound, prefill is compute-bound, and KV cache sets concurrency. That predicted two problems: prior-auth TTFT
on an A100 3g slice needs prefix caching, and member chat can't reach 32 concurrent on a 2g slice. Every later
measurement tests those predictions."

**M2 MIG.** "I let the GPU Operator own the full stack on pools created with gpu-driver none. MIG Manager applies the
all-balanced geometry from a node label, and the device plugin's mixed strategy advertises each slice as its own
resource. DCGM exporter gives power, energy, and per-instance activity. Repartitioning is a drain-relabel-wait
operation. I wrapped that in one script so it is repeatable at 2 a.m."

**M3 physics.** "I wrote a CUDA kernel that streams memory to measure achievable bandwidth per slice. The 3g slice got
about half of the full GPU's bandwidth because it owns half the memory slices, even though it has only three-sevenths
of the compute. Dividing weight bytes by that bandwidth gives the per-token decode floor. My measured ITL landed
within __% of the floor at low batch. That's how I explain latency to a customer without hand-waving."

**M7 performance and power.** "I swept concurrency with AIPerf and chose the highest load that still met both SLOs.
At that point prior-auth ran __ requests per second at __ tokens per joule. That is $__ per 1,000 decisions on Spot
and __ kWh per 1,000 decisions. Prefix caching cut TTFT p95 from __ to __ seconds because half of every prompt is
shared policy text."

**M8 isolation.** "Under a batch flood at 64 concurrent, prior-auth TTFT moved __% with MIG and __% with
time-slicing. MIG isolates SMs, L2, and memory bandwidth in hardware; time-slicing only interleaves contexts. MPS
sits in between: memory limits and compute shares, but one fault domain. I recommend MIG for SLO tenants and
time-slicing only for dev or bursty non-SLO work. KAI queues add guaranteed quotas with borrowing when teams share
GPU pools."

**M10 Dynamo.** "On traffic where 48 policies are reused with a Zipf distribution, KV-aware routing raised prefix-cache
hits from __% to __% and cut TTFT p95 by __%. Disaggregated prefill and decode was slower on NC-series because KV
transfer had no RDMA fabric - I'd revisit it on InfiniBand hardware with longer prompts. TensorRT-LLM gave __% more
throughput than vLLM at equal KV-cache size. Equal KV size matters because the two engines define memory fractions
differently."

## Questions you should be able to answer
1. **Why is LLM decode memory-bandwidth-bound?** Each token reads every weight once but does ~2 FLOPs per weight per
   sequence, so arithmetic intensity ~= batch size - far below the ridge point (~161 FLOP/byte on A100). Batching
   amortizes the weight reads; that is why continuous batching raises throughput almost for free until KV-cache or
   compute limits.
2. **MIG vs. time-slicing vs. MPS?** MIG: hardware partitions (SMs, L2, memory, bandwidth), fault isolation, fixed
   geometry, A100/H100/Blackwell only. Time-slicing: oversubscription by context switching, no memory isolation.
   MPS: concurrent kernels with memory limits and compute share, shared fault domain.
3. **How do you size a deployment?** Weights + KV per token x context x concurrency must fit; bandwidth sets the ITL
   floor; compute sets TTFT; then measure a latency-throughput curve and pick the SLO-compliant knee.
4. **"GPU utilization is 100% but throughput is low" - what do you check?** GPU_UTIL only means a kernel was
   running. Look at SM active, tensor pipe active, DRAM active; batch size and KV-cache usage (preemption); prefix
   hits; tokenization or CPU bottlenecks; prefill/decode interference.
5. **When is disaggregated serving worth it?** Long prompts plus tight ITL SLOs, where prefill bursts hurt decode -
   and only with a fast KV-transfer fabric (NVLink, InfiniBand/RDMA). Otherwise transfer cost eats the gain.
6. **What does KV-aware routing do?** Sends each request to the worker that already caches the longest prefix of
   its prompt, balancing against load, so shared prefixes are computed once instead of on every worker.
7. **NIM vs. vLLM vs. TensorRT-LLM?** NIM LLM 2.x packages vLLM with NVIDIA support, security scanning, profiles, and
   an operator lifecycle. TensorRT-LLM is the peak-performance engine, used through Dynamo or TRT-LLM-based NIMs.
   Choose on support model, performance evidence, and operability.
8. **How do you autoscale LLM inference?** On queue depth or KV-cache pressure, not GPU utilization; pre-scale for
   predictable peaks because node + image + model cold start takes minutes; keep scale-down slow.
9. **How do you measure energy per request honestly?** DCGM energy counter over the benchmark window, per physical
   GPU (max by UUID under MIG), attribute idle slices explicitly, report the method alongside the number.
10. **FP8 vs. INT4 vs. BF16?** FP8 needs Hopper or newer; roughly halves weight bytes (faster decode, more KV room)
    with small quality loss that must be measured on the customer's task. INT4 weight-only helps bandwidth further
    but costs more quality and can slow prefill.
11. **Why run the GPU Operator on AKS at all?** Runtime MIG reconfiguration, DCGM profiling metrics, sharing modes,
    and one driver lifecycle across clouds. AKS-managed drivers are simpler; never mix the two on a pool.
12. **What did you NOT cover?** Large-scale distributed training. Say so, then describe how you would learn it
    (NeMo, Megatron-style tensor/pipeline parallelism, NCCL on InfiniBand).
