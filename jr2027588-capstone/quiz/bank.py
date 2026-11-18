"""Question bank for the JR2027588 capstone.

Every question has: id, phase (P0-P13), level (basic | intermediate | advanced), kind (mcq | num | open),
prompt, answer, why (the explanation), source (the lab file that teaches it).

- mcq:  options = list of strings, answer = index of the correct option (0-based). Auto-graded.
- num:  answer = number, tol = allowed relative error (0.10 = +-10%), unit = string. Auto-graded.
- open: answer = model answer; keys = the points a good answer must hit. You grade yourself (y / p / n).

Numbers come from the labs (Labs 02-05). Where a number is a datasheet value, the source says so.
"""

L4 = "plv-multicloud-nim-lab"   # Lab 04 - the centerpiece, built for JR2027588
L5 = "ttg-agentic-genai-lab"    # Lab 05
L2 = "kha-genai-inference-lab"  # Lab 02


def mcq(id, phase, level, prompt, options, answer, why, source):
    return dict(id=id, phase=phase, level=level, kind="mcq", prompt=prompt, options=options,
                answer=answer, why=why, source=source)


def num(id, phase, level, prompt, answer, unit, tol, why, source):
    return dict(id=id, phase=phase, level=level, kind="num", prompt=prompt, answer=answer, unit=unit,
                tol=tol, why=why, source=source)


def opn(id, phase, level, prompt, answer, keys, source):
    return dict(id=id, phase=phase, level=level, kind="open", prompt=prompt, answer=answer, keys=keys,
                why=answer, source=source)


QUESTIONS = [
    # ------------------------------------------------------------------ P0 setup
    mcq("P0-B1", "P0", "basic",
        "In WSL2, where should the labs live for speed and correct line endings?",
        ["/mnt/c/Users/<you>/nvidia-labs", "~/nvidia-labs on the Linux filesystem", "A OneDrive folder",
         "Anywhere; WSL makes no difference"], 1,
        "/mnt/c is the Windows disk through a translation layer: slow, and Windows editors can add CRLF line "
        "endings that break bash scripts. Copy the labs into your Linux home.",
        "nvidia-labs/START-HERE.md"),
    mcq("P0-B2", "P0", "basic",
        "Which data may you send to build.nvidia.com hosted endpoints?",
        ["Real PHI if the customer signs a BAA", "De-identified claims", "Synthetic data only",
         "Anything under 1 MB"], 2,
        "Hosted endpoints sit outside the customer's tenant and run under the API Trial terms: prototyping with "
        "synthetic data only (for example Synthea FHIR bundles).",
        f"{L4}/docs/06-managed-paths.md"),
    mcq("P0-I1", "P0", "intermediate",
        "Your laptop is Windows on ARM64 with no NVIDIA GPU. Where do the GPU steps of this lab run?",
        ["In WSL with CUDA on the ARM CPU", "On cloud GPU nodes (AKS/GKE) or a rented GPU VM, driven from WSL",
         "In Docker Desktop with GPU passthrough", "They cannot run at all"], 1,
        "The laptop only drives: Terraform, kubectl, helm and Python tools run in WSL; CUDA kernels, NIM and "
        "benchmarks run on cloud GPUs.",
        "jr2027588-capstone/WINDOWS-SETUP.md"),
    opn("P0-A1", "P0", "advanced",
        "The Plover API image must be linux/amd64 but your laptop is ARM64. Give two ways to build and push it.",
        "1) Let GitHub Actions build it: push a v1.0.0 tag and release.yml builds on an amd64 runner. "
        "2) Build locally with docker buildx --platform linux/amd64 after registering QEMU emulation "
        "(docker run --privileged --rm tonistiigi/binfmt --install amd64); slower but works.",
        ["CI on an amd64 runner (release.yml tag)", "buildx cross-build with QEMU/binfmt",
         "the cluster nodes are amd64, so an arm64 image would fail with exec format error"],
        f"{L4}/README.md (M4) + .github/workflows/release.yml"),

    # ------------------------------------------------------------------ P1 physics
    mcq("P1-B1", "P1", "basic",
        "LLM decode (generating one token at a time) is limited mainly by:",
        ["GPU compute (FLOPs)", "GPU memory bandwidth", "PCIe bandwidth", "CPU speed"], 1,
        "Each new token re-reads every weight (plus the KV cache) but does only ~2 FLOPs per weight per sequence. "
        "Arithmetic intensity ~= batch size, far below the GPU's ridge point.",
        f"{L5}/docs/03-model-complexity.md"),
    mcq("P1-B2", "P1", "basic",
        "Prefill (reading the prompt) is limited mainly by ____ and sets ____.",
        ["bandwidth; inter-token latency", "compute; time to first token (TTFT)", "network; throughput",
         "KV cache; context length"], 1,
        "Prefill processes all prompt tokens in parallel: ~2 x parameters FLOPs per token, compute-bound. "
        "It determines TTFT.",
        f"{L5}/docs/03-model-complexity.md"),
    num("P1-B3", "P1", "basic",
        "Llama 3.1 8B in BF16: about how many GB of weights? (8.03B parameters x 2 bytes)",
        16.06, "GB", 0.05,
        "8.03e9 x 2 bytes = 16.06 GB. FP8 halves it (~8 GB); NVFP4 roughly quarters it.",
        f"{L4}/docs/00-recall.md"),
    num("P1-I1", "P1", "intermediate",
        "Batch-1 decode floor for Llama 3.1 8B BF16 (16.06 GB) on an A100 80GB PCIe (1,935 GB/s), in ms per token?",
        8.3, "ms", 0.06,
        "16.06 GB / 1,935 GB/s = 8.3 ms per token (~120 tok/s per stream). The lab's selector adds a measured "
        "bandwidth-efficiency factor, so it prints a higher floor.",
        f"{L4}/docs/00-recall.md"),
    num("P1-I2", "P1", "intermediate",
        "KV-cache bytes per token for Llama 3.1 8B BF16, in KiB? (32 layers, 8 KV heads, head dim 128)",
        128, "KiB", 0.02,
        "2 (K and V) x 32 layers x 8 KV heads x 128 dims x 2 bytes = 131,072 B = 128 KiB per token. "
        "An 8K-token conversation holds ~1 GiB.",
        f"{L4}/docs/00-recall.md"),
    mcq("P1-I3", "P1", "intermediate",
        "Why is the L4 vetoed for Llama 3.1 8B BF16 at a 40 ms inter-token SLO?",
        ["It has too little memory for the weights", "Its 300 GB/s bandwidth gives a ~53 ms decode floor",
         "It does not support BF16", "NIM does not run on Ada GPUs"], 1,
        "16.06 GB / 300 GB/s ~= 53 ms per token before any queueing: physics says no before anyone buys it. "
        "FP8 barely rescues it, and it is not on NIM's verified list for that model.",
        f"{L4}/docs/03-instance-selection.md"),
    num("P1-A1", "P1", "advanced",
        "H100 SXM BF16 ridge point: 989 TFLOPS / 3.35 TB/s = about how many FLOPs per byte?",
        295, "FLOP/byte", 0.05,
        "989e12 / 3.35e12 ~= 295. Decode at batch 1 runs at ~1 FLOP/byte, so batching (continuous batching) is "
        "the lever: throughput rises almost free until KV capacity or the compute roof stops it.",
        f"{L5}/docs/00-recall.md"),
    num("P1-A2", "P1", "advanced",
        "Llama 3.1 70B FP8 on 2x H100 80GB (TP 2): about how many 8K-token sequences fit in KV cache (BF16 KV, "
        "320 KiB/token)?",
        30, "sequences", 0.2,
        "Weights ~70.6 GB split across 160 GB leaves ~80 GB minus overhead for KV. 8,192 x 320 KiB = 2.5 GiB per "
        "sequence -> ~30 sequences (~60 with an FP8 KV cache). One H100 fits only ~2.",
        f"{L5}/docs/00-recall.md"),
    opn("P1-A3", "P1", "advanced",
        "A customer asks: 'Which GPU should we run Llama 3.1 8B on, at 600 input / 200 output tokens and a 40 ms "
        "ITL SLO?' Walk through your method in 60 seconds.",
        "Start with physics: weight bytes / bandwidth gives the decode floor per GPU and precision, which vetoes "
        "GPUs that can't meet 40 ms (L4 in BF16). Then the batch that still meets the SLO (bandwidth until the "
        "compute roof, as the A100 at batch 369), tokens/s and $/1M tokens per instance including the NVAIE "
        "license. Check the NIM verified-GPU list for the model. Then replace estimates with AIPerf measurements "
        "on the shortlist and pick the cheapest SLO-compliant operating point.",
        ["decode floor = bytes / bandwidth", "veto by SLO", "max batch at SLO / compute roof",
         "$/1M tokens incl. NVAIE", "NIM verified list", "measure with AIPerf"],
        f"{L4}/docs/03-instance-selection.md"),

    # ------------------------------------------------------------------ P2 discovery
    mcq("P2-B1", "P2", "basic",
        "A good PoC success criterion is:",
        ["'The model feels fast'", "'p95 ITL <= 40 ms at the operating point, measured by AIPerf'",
         "'Customer is happy'", "'Use the latest GPU'"], 1,
        "Every criterion names a metric, a target and the tool that measures it.",
        f"{L4}/sales/poc-charter.md"),
    mcq("P2-B2", "P2", "basic",
        "In discovery, which question matters most before sizing GPUs?",
        ["Which logo do they prefer", "Traffic shape: input/output tokens, peak concurrency, latency SLO",
         "How many engineers they have", "Which IDE they use"], 1,
        "Sizing depends on prompt and output lengths, concurrency and the SLO. Everything else comes later.",
        f"{L4}/sales/discovery-questions.md"),
    opn("P2-I1", "P2", "intermediate",
        "Name four things a PoC charter must contain.",
        "Scope and use case; success criteria with targets and the tool that measures each; timeline and "
        "owners on both sides; environment and data rules (synthetic data, where it runs); exit decision "
        "(what happens if it passes).",
        ["measurable criteria + tool", "timeline + owners", "data/environment rules", "exit decision"],
        f"{L4}/sales/poc-charter.md"),
    opn("P2-A1", "P2", "advanced",
        "A health plan wants a prior-auth summarizer but PHI can't leave their Azure tenant. How do you shape "
        "the PoC?",
        "Prototype on hosted endpoints with synthetic data only; run the real PoC as self-hosted NIM inside "
        "their tenant (AKS, private network, Key Vault via External Secrets), so PHI never leaves. Success "
        "criteria: schema-valid rate, faithfulness, p95 latency and $ per 1,000 summaries at the SLO. Let "
        "the Azure Marketplace carry NVIDIA AI Enterprise so it draws down their commit.",
        ["synthetic data on hosted endpoints", "self-hosted NIM in their tenant", "measurable criteria",
         "marketplace / commit drawdown"],
        f"{L4}/docs/01-customer-brief.md"),

    # ------------------------------------------------------------------ P3 platform
    mcq("P3-B1", "P3", "basic",
        "Why does every GPU node pool in the lab carry the taint nvidia.com/gpu?",
        ["To make GPUs faster", "So only pods that tolerate it (GPU workloads) land on expensive GPU nodes",
         "Kubernetes requires it", "To enable MIG"], 1,
        "Taints keep ordinary pods off GPU nodes. GPU pods must opt in with a toleration (drill 1).",
        f"{L4}/drills/README.md"),
    mcq("P3-B2", "P3", "basic",
        "What does the NVIDIA GPU Operator install and manage?",
        ["Only the CUDA toolkit", "Driver, container toolkit, device plugin, DCGM exporter, MIG manager",
         "The NIM containers", "Terraform state"], 1,
        "It owns the whole GPU software stack on the node so it is the same on every cloud.",
        f"{L2}/docs/05-interview-drill.md"),
    mcq("P3-I1", "P3", "intermediate",
        "Why does the AKS GPU pool use gpu_driver = \"None\"?",
        ["AKS can't install drivers", "So the GPU Operator owns the driver; never mix AKS-managed and "
         "operator-managed drivers on one pool", "To save cost", "Because A100 needs no driver"], 1,
        "Operator-managed mode gives runtime MIG reconfiguration, DCGM profiling and one driver lifecycle "
        "across clouds.",
        f"{L4}/docs/07-csp-cheatsheet.md"),
    mcq("P3-I2", "P3", "intermediate",
        "On GKE, who installs the NVIDIA driver in this lab?",
        ["The GPU Operator", "Google's driver installer DaemonSet (pool set to INSTALLATION_DISABLED)",
         "The NIM Operator", "Argo CD"], 1,
        "GPU plumbing differs per cloud; the product (chart) doesn't. That's the point of the cheat sheet.",
        f"{L4}/docs/07-csp-cheatsheet.md"),
    mcq("P3-A1", "P3", "advanced",
        "The CUDA bandwidth probe uses buffers of ~35% of free memory. Why so large?",
        ["To fill the KV cache", "Buffers far larger than L2 measure HBM/GDDR bandwidth, not cache",
         "To trigger MIG", "To test PCIe"], 1,
        "A read-only streaming kernel models decode (read weights, write almost nothing). Small buffers would "
        "fit in L2 and overstate bandwidth. Expect ~80-90% of the datasheet number.",
        f"{L4}/docs/00-recall.md"),
    opn("P3-A2", "P3", "advanced",
        "Explain pull-based GitOps with Argo CD app-of-apps and the sync waves this lab uses.",
        "A root Application per cluster points at gitops/; it creates child Applications. The cluster pulls "
        "from Git (no CI credentials into the cluster). Sync waves order them: driver -> operators -> "
        "secrets (External Secrets) -> observability -> product. Self-heal reverts manual drift, so Git is "
        "the control plane and kubectl is read-only in production.",
        ["root app creates child apps", "pull, not push", "waves: driver, operators, secrets, observability, "
         "product", "self-heal: Git is the source of truth"],
        f"{L4}/README.md (M3)"),

    # ------------------------------------------------------------------ P4 NIM
    mcq("P4-B1", "P4", "basic",
        "What is a NIM?",
        ["A GPU model", "A packaged, supported inference microservice: optimized engine + OpenAI-compatible API",
         "A Kubernetes scheduler", "A training framework"], 1,
        "NIM = NVIDIA Inference Microservice. Same engines (TensorRT-LLM / vLLM class), tuned profiles per GPU, "
        "security-patched, part of NVIDIA AI Enterprise.",
        f"{L5}/docs/07-engine-cheatsheet.md"),
    mcq("P4-B2", "P4", "basic",
        "Which two custom resources does the NIM Operator use?",
        ["Deployment and Service", "NIMCache and NIMService", "ClusterPolicy and NodePool",
         "Application and AppProject"], 1,
        "NIMCache downloads model profiles once to storage; NIMService runs the microservice from that cache.",
        f"{L4}/README.md (M4)"),
    mcq("P4-I1", "P4", "intermediate",
        "What is a NIM profile, and why is it per GPU AND per GPU count?",
        ["A user account", "A pre-built engine configuration (engine, precision, TP/PP, LoRA); kernels and "
         "memory plans depend on the GPU and how many GPUs share the model", "A billing tier",
         "A Helm values file"], 1,
        "Always pin the profile that list-model-profiles shows on THAT GPU (drill 2).",
        f"{L4}/docs/00-recall.md"),
    mcq("P4-I2", "P4", "intermediate",
        "A NIM pod is in CrashLoopBackOff right after start; the log mentions certificates and permissions. "
        "The likely missing setting is:",
        ["groupID: 0", "a GPU toleration", "a larger PVC", "an HPA"], 0,
        "NIM images update CA certificates at start and need group-0 file permissions.",
        f"{L4}/docs/00-recall.md"),
    mcq("P4-A1", "P4", "advanced",
        "Why does a NIM pod need a startupProbe, not just a livenessProbe?",
        ["Startup probes are faster", "First start downloads the model; without a startupProbe the "
         "livenessProbe kills the pod mid-download", "Liveness probes don't work on GPUs",
         "NIM has no health endpoint"], 1,
        "Model download and engine load take minutes. Also: a one-GPU engine Deployment uses strategy Recreate "
        "so a rolling update doesn't wait forever for a second GPU.",
        f"{L5}/docs/00-recall.md"),
    opn("P4-A2", "P4", "advanced",
        "What exactly differs between the AKS and GKE deployments of the same product?",
        "Only clusters/<cloud>/values.yaml (~20 lines): storage class and NIM cache (Azure Files RWX NIMCache "
        "vs emptyDir), tolerations, precision/profile per GPU (BF16 on A100, NVFP4 on RTX PRO 6000), and the "
        "secret store (Key Vault vs Secret Manager via External Secrets with workload identity). Same chart, "
        "same GitOps.",
        ["one chart, per-cloud values file", "storage/cache", "profile/precision per GPU", "secret store"],
        f"{L4}/docs/05-interview-drill.md"),

    # ------------------------------------------------------------------ P5 benchmarking
    mcq("P5-B1", "P5", "basic",
        "TTFT and ITL stand for:",
        ["Total time / idle time", "Time to first token / inter-token latency",
         "Tokens to finish / input token length", "Throughput / latency"], 1,
        "TTFT is what the user waits before text appears (prefill + queueing); ITL is the gap between tokens "
        "(decode).",
        f"{L5}/docs/10-interview-drill.md"),
    mcq("P5-B2", "P5", "basic",
        "What is goodput?",
        ["Peak tokens per second", "Requests per second that met every SLO", "GPU utilization",
         "Tokens per joule"], 1,
        "Customers pay for goodput, not the plateau of a throughput curve.",
        f"{L4}/docs/00-recall.md"),
    num("P5-I1", "P5", "intermediate",
        "A node costs $1.20/hour and serves 2.0 SLO-compliant summaries per second at the operating point. "
        "Cost per 1,000 summaries in dollars?",
        0.1667, "$", 0.05,
        "$/hour / (summaries per second x 3,600) x 1,000 = 1.20 / 7,200 x 1,000 = $0.167. Add NVAIE $/GPU-hour "
        "if licensed hourly.",
        f"{L4}/docs/00-recall.md"),
    mcq("P5-I2", "P5", "intermediate",
        "How do you pick the operating point from a concurrency sweep?",
        ["The highest throughput level", "The highest load level whose p95 still meets the SLO",
         "Concurrency 1", "Where GPU utilization hits 100%"], 1,
        "Never sell the plateau: past the knee, requests just queue.",
        f"{L5}/docs/00-recall.md"),
    mcq("P5-A1", "P5", "advanced",
        "In a sweep, throughput goes flat, ITL stays flat, and TTFT explodes. What is happening?",
        ["The GPU is overheating", "Queueing: the replica is saturated (prefill compute or KV capacity), so "
         "requests wait", "The network is down", "Decode got slower"], 1,
        "Each token isn't slower; requests wait. Check KV-cache usage and waiting requests to tell compute "
        "from memory.",
        f"{L5}/docs/00-recall.md"),
    mcq("P5-A2", "P5", "advanced",
        "Under MIG, why do you use max by (UUID) before summing DCGM power?",
        ["It's faster", "Device-level fields repeat on every MIG instance series; summing them overstates power",
         "UUIDs are random", "DCGM requires it"], 1,
        "Report energy per physical GPU over the benchmark window, and say how you measured it.",
        f"{L4}/docs/00-recall.md"),
    opn("P5-A3", "P5", "advanced",
        "Your measured tokens/s on AKS is 30% below the selector's prediction. Give three likely reasons and "
        "how you'd check each.",
        "1) Real bandwidth efficiency below the assumed BW_EFF: run the CUDA probe and put the measured ratio in. "
        "2) Batch never reached the predicted size: KV capacity or max-num-seqs limits; check engine logs and "
        "KV usage. 3) Overheads not in the model: scheduler/CPU, tokenization, small-prompt prefill "
        "interference; profile and check GPU busy %. Also check that AIPerf ran close to the cluster (no WAN "
        "round trips) and the prompt shape matches.",
        ["measured bandwidth vs assumed", "batch/KV limits", "CPU/scheduler/prefill overhead",
         "benchmark setup (client location, prompt shape)"],
        f"{L4}/README.md (M5)"),

    # ------------------------------------------------------------------ P6 engines + precision
    mcq("P6-B1", "P6", "basic",
        "Which statement about TensorRT-LLM is right?",
        ["It's a GPU", "NVIDIA's LLM inference library: optimized kernels, in-flight batching, paged KV cache, "
         "FP8/NVFP4", "A Kubernetes operator", "A model hub"], 1,
        "Choose it for peak tokens per GPU on NVIDIA hardware with a stable model.",
        f"{L5}/docs/07-engine-cheatsheet.md"),
    mcq("P6-B2", "P6", "basic",
        "FP8 weights mainly speed up decode because they:",
        ["Use fewer GPUs", "Halve the bytes read per token, so the bandwidth floor roughly halves",
         "Skip attention", "Shorten the prompt"], 1,
        "Decode is bandwidth-bound: fewer bytes per weight = faster tokens and more room for KV cache. "
        "Always re-run the quality gate after quantizing.",
        f"{L5}/docs/07-engine-cheatsheet.md"),
    mcq("P6-I1", "P6", "intermediate",
        "Which precision is native to Blackwell only?",
        ["BF16", "FP8", "NVFP4", "INT8"], 2,
        "NVFP4: 4-bit weights with FP8 scales per 16 values; needs an NVFP4 checkpoint. FP8 runs on Hopper, "
        "Ada and Blackwell.",
        f"{L5}/docs/07-engine-cheatsheet.md"),
    mcq("P6-I2", "P6", "intermediate",
        "Why can't you copy a TensorRT plan from an A100 to an RTX PRO 6000?",
        ["File size limits", "Plans are compiled for one GPU architecture and TensorRT version",
         "License restrictions", "Different CUDA drivers can't read files"], 1,
        "Rebuild per GPU, or build with a hardware-compatibility level (ampere+) and accept some speed cost. "
        "ONNX is portable; engines are not (drill 10).",
        f"{L4}/drills/README.md"),
    mcq("P6-A1", "P6", "advanced",
        "When is disaggregated prefill/decode (Dynamo) worth it?",
        ["Always", "Long prompts + tight ITL SLOs where prefill bursts hurt decode, AND a fast KV-transfer "
         "fabric (NVLink, InfiniBand/RDMA)", "Only on a single GPU", "Only for training"], 1,
        "Without a fast fabric, KV transfer eats the gain (Lab 02 measured it slower on NC-series without RDMA).",
        f"{L2}/docs/05-interview-drill.md"),
    mcq("P6-A2", "P6", "advanced",
        "What does KV-aware routing do?",
        ["Routes by GPU temperature", "Sends each request to the worker already caching the longest prefix "
         "of its prompt, balanced against load", "Round-robin across pods", "Moves KV cache to disk"], 1,
        "Shared prefixes (system prompts, policies) are computed once instead of on every replica.",
        f"{L2}/docs/05-interview-drill.md"),
    opn("P6-A3", "P6", "advanced",
        "A customer asks: 'NIM, vLLM or TensorRT-LLM - which should we use?' Give your 60-second answer.",
        "It depends on support model, performance evidence and operability. vLLM/SGLang: broadest model "
        "coverage and day-0 support. TensorRT-LLM: peak tokens per GPU on NVIDIA hardware with calibrated "
        "FP8/NVFP4 checkpoints. NIM: the same engine families packaged with tuned per-GPU profiles, security "
        "patches and enterprise support, running in your own tenant. Dynamo on top for multi-replica, "
        "disaggregated serving. I'd measure your traffic on two of them, check API feature parity "
        "(conformance tests), and decide on cost per request at your SLO.",
        ["vLLM/SGLang day-0 coverage", "TRT-LLM peak perf", "NIM = packaged + supported + per-GPU profiles",
         "Dynamo for scale-out", "measure on their traffic + conformance"],
        f"{L5}/docs/07-engine-cheatsheet.md"),

    # ------------------------------------------------------------------ P7 scale
    mcq("P7-B1", "P7", "basic",
        "Which signal should autoscale an LLM engine?",
        ["GPU utilization", "Queue depth (e.g. vllm:num_requests_waiting) or KV-cache pressure", "CPU usage",
         "Pod count"], 1,
        "Queue depth measures unmet demand. GPU utilization reads ~100% long before latency breaks.",
        f"{L4}/docs/00-recall.md"),
    mcq("P7-B2", "P7", "basic",
        "MIG (Multi-Instance GPU) partitions a GPU into up to how many isolated instances?",
        ["2", "4", "7", "16"], 2,
        "Up to 7 on A100/H100/Blackwell, each with its own SMs, L2 slice and memory bandwidth.",
        f"{L2}/docs/05-interview-drill.md"),
    mcq("P7-I1", "P7", "intermediate",
        "What does MIG isolate that time-slicing does not?",
        ["Nothing", "SMs, L2 cache and memory bandwidth, in hardware", "Only memory capacity", "CPU cores"], 1,
        "Time-slicing only interleaves contexts on shared hardware. MPS sits in between (memory limits, "
        "compute share, one fault domain).",
        f"{L4}/docs/00-recall.md"),
    opn("P7-I2", "P7", "intermediate",
        "Break a GPU cold start into its parts and give one fix for each.",
        "1) Node provisioning (Spot capacity, quota): keep one warm replica or reserve capacity. 2) Image pull "
        "(NIM images are large): image streaming (GKE) or pre-pulled images. 3) Model load: a pre-filled cache "
        "(NIMCache on RWX storage) instead of re-downloading to emptyDir. Then pre-scale for known peaks and "
        "scale down slowly.",
        ["node provisioning", "image pull", "model download/load", "warm replica / pre-scale"],
        f"{L4}/README.md (M7)"),
    num("P7-A1", "P7", "advanced",
        "Pipeline parallelism with P = 4 stages and m = 12 micro-batches: what fraction of time is the bubble? "
        "(formula (P-1)/(m+P-1))",
        0.2, "fraction", 0.05,
        "(4-1)/(12+4-1) = 3/15 = 0.20. More micro-batches shrink the bubble.",
        f"{L5}/docs/05-parallelism-math.md"),
    mcq("P7-A2", "P7", "advanced",
        "Where should tensor parallelism live, and why?",
        ["Across nodes over Ethernet", "Inside the NVLink domain: it runs 2 all-reduces per layer per step, "
         "small latency-bound messages", "On PCIe only", "On CPU"], 1,
        "TP inside NVLink, PP across nodes, DP on top, EP for MoE, CP for very long context. Over PCIe, prefer "
        "data-parallel replicas.",
        f"{L5}/docs/10-interview-drill.md"),
    opn("P7-A3", "P7", "advanced",
        "Describe the GB200 NVL72 in 60 seconds.",
        "One liquid-cooled rack that behaves like one big GPU. Building block: GB200 superchip = 1 Grace CPU + "
        "2 Blackwell GPUs over NVLink-C2C. 36 superchips = 36 Grace CPUs + 72 GPUs, all in one NVLink domain "
        "(NVLink 5, 1.8 TB/s per GPU, ~130 TB/s across the rack). A DGX H100's NVLink domain is 8 GPUs at "
        "900 GB/s each. So TP and expert parallelism for big MoE models stay on NVLink, and Dynamo can split "
        "prefill/decode inside one rack; racks scale out over InfiniBand or Spectrum-X.",
        ["36 Grace + 72 Blackwell", "one NVLink domain", "1.8 TB/s per GPU, ~130 TB/s", "vs DGX H100 8 GPUs",
         "why: TP/EP on NVLink, scale out over IB/Spectrum-X"],
        f"{L5}/docs/04-gpu-cluster-architecture.md"),
    num("P7-A4", "P7", "advanced",
        "InfiniBand NDR is 400 Gb/s per NIC. How many GB/s is that?",
        50, "GB/s", 0.02,
        "400 / 8 = 50 GB/s per NIC per direction, vs NVLink 4 at 450 GB/s per direction: leaving the NVLink "
        "domain costs ~9x.",
        f"{L5}/docs/04-gpu-cluster-architecture.md"),

    # ------------------------------------------------------------------ P8 MLOps / LoRA
    mcq("P8-B1", "P8", "basic",
        "What is a LoRA adapter?",
        ["A new GPU", "A small set of low-rank weights trained on top of a frozen base model",
         "A Kubernetes plugin", "A quantization format"], 1,
        "Small, cheap to train and swap; NIM can serve several adapters on one base model.",
        f"{L4}/README.md (M8)"),
    mcq("P8-I1", "P8", "intermediate",
        "What does the release gate compare, and when do you re-baseline?",
        ["Nothing; it always passes", "Candidate vs baseline on the metrics that matter, with tolerances; "
         "re-baseline deliberately, never automatically", "Only file size", "Only training loss"], 1,
        "Quality (schema, faithfulness), latency, tokens. The gate decides, not the demo.",
        f"{L4}/docs/00-recall.md"),
    opn("P8-A1", "P8", "advanced",
        "Describe the LoRA release path in this lab end to end.",
        "Distil a dataset from today's product (teacher outputs filtered for schema and faithfulness); train with "
        "PEFT on a MIG slice while the other slice serves, tracked in MLflow; eval gate vs baseline; push the "
        "adapter as an OCI artifact (ORAS) pinned by digest; a Git commit loads it and routes 10% canary by "
        "model name; promote or roll back with one commit.",
        ["distilled dataset", "PEFT on a MIG slice + MLflow", "eval gate", "ORAS artifact by digest",
         "canary + rollback via Git"],
        f"{L4}/README.md (M8)"),
    opn("P8-A2", "P8", "advanced",
        "Tell the Gemma LoRA evaluation story and the lesson.",
        "A fine-tuned Gemma 4B looked worse than base. The base was scored through vLLM and the fine-tune through "
        "a custom inference path: two serving stacks, two decoding defaults. I re-ran both through the same "
        "engine, chat template and sampling settings. Lesson: lock the serving stack before comparing models; "
        "otherwise you measure the paths, not the fine-tune.",
        ["base via vLLM vs tuned via custom path", "same engine/template/sampling", "lesson: measure models, "
         "not paths"],
        f"{L5}/docs/10-interview-drill.md"),

    # ------------------------------------------------------------------ P9 air gap
    mcq("P9-B1", "P9", "basic",
        "In an air-gapped data center, how does the NIM get its model?",
        ["Downloads from NGC at start", "Pre-cached (download-to-cache on the connected side, then moved in)",
         "From Hugging Face", "It can't run air-gapped"], 1,
        "No NGC key in any pod, zero egress; images come from a local registry mirror.",
        f"{L4}/README.md (M9)"),
    mcq("P9-I1", "P9", "intermediate",
        "After a tag bump, pods show ErrImagePull ... i/o timeout in the air gap. Root cause?",
        ["GPU failure", "The new tag isn't in the mirror; the image list wasn't re-staged",
         "Wrong toleration", "Expired license"], 1,
        "In a disconnected DC the image list IS the change request (drill 8).",
        f"{L4}/drills/README.md"),
    opn("P9-A1", "P9", "advanced",
        "What evidence pack does a hospital security team need to sign off an air-gapped deployment?",
        "Proof of zero egress (network policy/NSG deny-all, verification run), images only from the internal "
        "mirror (computed image list from rendered charts, including ClusterPolicy components), no NGC key or "
        "secrets in pods, model pre-cached with checksums, helm test passing inside the gap.",
        ["zero egress proof", "mirrored images only (computed list)", "no keys in pods", "pre-cached model",
         "helm test inside the gap"],
        f"{L4}/README.md (M9)"),

    # ------------------------------------------------------------------ P10 managed vs self-hosted
    mcq("P10-B1", "P10", "basic",
        "NVIDIA AI Enterprise on AWS/Azure/GCP/OCI marketplaces is priced at:",
        ["Free", "$1 per GPU-hour plus the instance", "$10 per GPU-hour", "Per token only"], 1,
        "Or $4,500/GPU/year list (1-year). Marketplace purchases can draw down the customer's cloud commit.",
        f"{L4}/docs/06-managed-paths.md"),
    num("P10-I1", "P10", "intermediate",
        "NVAIE annual list $4,500/GPU/year vs $1/GPU-hour: above roughly what utilization (%) does the annual "
        "subscription win?",
        51, "%", 0.06,
        "$4,500 / 8,760 h ~= $0.51/h, so the annual wins above ~51% utilization of a GPU.",
        f"{L4}/docs/06-managed-paths.md"),
    opn("P10-I2", "P10", "intermediate",
        "When do you recommend a managed endpoint vs self-hosted NIM?",
        "Prototype on managed/hosted endpoints (fast, per-token). Ship self-hosted NIM where PHI or data "
        "sovereignty, on-prem/air gap, LoRA adapter ownership, or high steady utilization (cheaper per token) "
        "matter. Let each CSP marketplace carry NVAIE so it counts against the cloud commit.",
        ["prototype managed", "self-host for PHI/sovereignty/on-prem/adapters/utilization", "marketplace commit"],
        f"{L4}/docs/06-managed-paths.md"),
    mcq("P10-A1", "P10", "advanced",
        "On Spot A100 at $0.679/h, adding NVAIE at $1/GPU-hour roughly does what to $/1M tokens?",
        ["Nothing", "More than doubles it: the license can cost more than the Spot GPU", "Halves it",
         "Adds 1%"], 1,
        "The selector shows $0.020 -> $0.051 per 1M tokens. For always-on capacity, compare the annual "
        "subscription.",
        f"{L4}/docs/03-instance-selection.md"),

    # ------------------------------------------------------------------ P11 operate + troubleshoot
    mcq("P11-B1", "P11", "basic",
        "A NIM pod is Pending with 'untolerated taint {nvidia.com/gpu}'. Fix?",
        ["Bigger node", "Add the GPU toleration (in Git, via the chart)", "Restart the cluster",
         "Delete the taint"], 1,
        "GPU pods must opt in to GPU nodes - on every cloud (drill 1).",
        f"{L4}/drills/README.md"),
    mcq("P11-B2", "P11", "basic",
        "You scale the NIM to 0 with kubectl; minutes later it's back. Why?",
        ["A bug", "Argo CD self-heal reverts drift from Git", "The HPA", "Kubernetes ignores scale 0"], 1,
        "Git is the control plane; change Git, not the cluster (drill 6).",
        f"{L4}/drills/README.md"),
    mcq("P11-I1", "P11", "intermediate",
        "Second NIM replica Pending: 'Insufficient nvidia.com/gpu', autoscaler can't add nodes. Root cause?",
        ["Bad image", "Replicas exceed GPU capacity: pool max nodes or regional quota", "Wrong profile",
         "NetworkPolicy"], 1,
        "Autoscaling stops at your quota - request quota before the launch (drill 7).",
        f"{L4}/drills/README.md"),
    opn("P11-I2", "P11", "intermediate",
        "Name the six commands you run first on any cloud when the product breaks.",
        "kubectl get nimservice,nimcache,externalsecret,pods; describe pod (tail); logs of the LLM pod; "
        "get events -A sorted by time; get applications in argocd; fleet/fleet_report.py for the fleet view.",
        ["get nimservice/nimcache/externalsecret/pods", "describe pod", "logs", "events", "argocd applications",
         "fleet report"],
        f"{L4}/drills/README.md"),
    opn("P11-A1", "P11", "advanced",
        "'GPU utilization is 100% but throughput is low.' What do you check?",
        "GPU_UTIL only means a kernel was running. Check DCGM SM active, tensor-pipe active and DRAM active; "
        "batch size and KV-cache usage (preemption); prefix-cache hits; tokenization or CPU/scheduler "
        "bottlenecks; prefill/decode interference. Then profile (Nsight Systems / torch profiler) for idle gaps.",
        ["GPU_UTIL is misleading", "SM/tensor/DRAM active", "batch + KV usage/preemption", "CPU/scheduler",
         "profile"],
        f"{L2}/docs/05-interview-drill.md"),
    opn("P11-A2", "P11", "advanced",
        "A customer says 'our TTFT doubled after an upgrade.' First three moves?",
        "1) Diff what changed: pins/images/profile in Git and the NIM profile actually selected (logs, "
        "list-model-profiles). 2) Re-run the same AIPerf sweep and compare against the perf-gate baseline: is "
        "it queueing (TTFT up, ITL flat) or slower prefill? 3) Check prefix-cache hit ratio, KV capacity and "
        "max batch/precision settings in the new version; roll back via Git if the SLO is breached.",
        ["diff pins/profile in Git", "re-run the baseline sweep", "queueing vs prefill", "cache/KV settings",
         "rollback via Git"],
        f"{L4}/docs/05-interview-drill.md"),

    # ------------------------------------------------------------------ P12 teach + win
    mcq("P12-B1", "P12", "basic",
        "The five parts of the one-page executive readout are:",
        ["Intro, body, end, Q&A, thanks", "Situation, what we tested, results table, recommendation, risks + "
         "next steps", "Architecture, code, tests, CI, deploy", "Price, discount, terms, legal, signature"], 1,
        "Same shape in every lab; fill it with your measured numbers.",
        f"{L4}/docs/00-recall.md"),
    opn("P12-I1", "P12", "intermediate",
        "What is a technical win plan, and how does it differ from a mutual action plan?",
        "Technical win plan: the SA's plan to prove the solution (criteria, risks, evidence, stakeholders on the "
        "technical side). Mutual action plan: dated steps both sides commit to through to the decision "
        "(PoC, security review, procurement), owned jointly with sales.",
        ["technical win = proof + criteria + evidence", "MAP = dated joint steps to a decision"],
        f"{L4}/sales/technical-win-plan.md"),
    opn("P12-A1", "P12", "advanced",
        "Give the 5-minute exec readout opening for Plover, with placeholders for your numbers.",
        "'You asked whether one product can run on Azure, Google Cloud and a disconnected hospital at your "
        "latency target. We tested it for four weeks with seven criteria. It runs on all three from one chart; "
        "at a 40 ms ITL target the cost is $__ per 1,000 summaries on Azure and $__ on Google Cloud. We "
        "recommend __ for production, with these two risks and next steps.'",
        ["the business question", "what was tested", "the number per cloud at the SLO", "recommendation",
         "risks/next steps"],
        f"{L4}/docs/04-results-and-exec-readout.md"),

    # ------------------------------------------------------------------ P13 interview mix
    opn("P13-I1", "P13", "intermediate",
        "'Why NVIDIA, and why this role?' (60 seconds)",
        "Every customer AI conversation ends as a GPU conversation: capacity, latency, cost per token. NVIDIA owns "
        "the stack under all of it. This role sits where I already work, between the cloud provider and the "
        "customer; I know how a hyperscaler sizes, sells and deploys GPU workloads, and I've built the NVIDIA "
        "side hands-on: NIM on AKS and GKE, AIPerf price-performance, engine bake-offs.",
        ["GPU is the bottleneck of every AI deal", "CSP + customer middle", "hands-on NVIDIA proof"],
        "jr2027588-capstone/README.md (P13)"),
    num("P13-A1", "P13", "advanced",
        "Pretraining FLOPs for an 8B model on 1T tokens (6 x N x D), in units of 1e22?",
        4.8, "x1e22 FLOPs", 0.03,
        "6 x 8e9 x 1e12 = 4.8e22. On 64 H100s at 989 TFLOPS x 40% MFU (~2.5e16 FLOP/s) that's ~22 days.",
        f"{L5}/docs/10-interview-drill.md"),
    opn("P13-A2", "P13", "advanced",
        "Design: a CSP customer wants Llama 3.1 70B for 2,000 concurrent chat users at p95 TTFT < 1 s and "
        "ITL < 50 ms. Outline your answer.",
        "Clarify traffic (input/output lengths, peak vs average). Size: FP8 70B ~70 GB weights; KV 320 KiB/token "
        "(160 with FP8 KV); per-replica capacity from KV and the ITL floor (TP2 on H100 ~10.5 ms/token). "
        "Replicas = peak concurrency / sequences per replica at the SLO, + N+1. Serving: NIM (TRT-LLM profile) "
        "on GPU node pools, TP inside NVLink, Dynamo KV-aware routing for shared prefixes, autoscale on queue "
        "depth with warm capacity. Measure with AIPerf, pick the operating point, price $/1M tokens incl. NVAIE, "
        "and set quota before launch.",
        ["clarify traffic", "weights + KV math", "replicas at SLO + N+1", "TP in NVLink / NIM / Dynamo",
         "autoscale on queue depth", "measure + price"],
        "jr2027588-capstone/README.md (P13)"),
]

PHASES = {
    "P0": "Setup on Windows + WSL2",
    "P1": "GPU + LLM inference physics",
    "P2": "Customer discovery + PoC charter",
    "P3": "Multi-cloud GPU Kubernetes platform (IaC + GitOps)",
    "P4": "Ship NIM on two clouds",
    "P5": "Benchmark + price-performance",
    "P6": "Engines + precision (TensorRT-LLM, Triton, NIM, Dynamo)",
    "P7": "Scale: autoscaling, cold start, MIG, parallelism, NVL72",
    "P8": "MLOps: LoRA as a release",
    "P9": "Air-gapped data center",
    "P10": "Managed vs self-hosted + CSP marketplaces",
    "P11": "Operate + troubleshoot",
    "P12": "Teach + win: workshop, readout, win plan",
    "P13": "Interview loop",
}
