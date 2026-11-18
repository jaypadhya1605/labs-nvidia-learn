# Lab 05 - Agentic AI, quantified: a travel-disruption agent on the NVIDIA GenAI stack

**Tern Travel Group** (fictional: 140 hotels, a 60-aircraft regional airline, a rail partner) loses its contact center
on every winter-storm day: ~40,000 disrupted travelers inside 6 hours, 11 minutes of agent time each, 3-hour queues.
In this lab you are the NVIDIA **Senior Solutions Architect, Gen AI** on the Hospitality and Travel team who helps
Tern build a **Disruption Concierge** agent - and then proves, in dollars per disruption and GPUs per storm, what
accelerated computing is worth to it. The agent checks the flight, rebooks, books a hotel when owed, issues
vouchers under a written policy, reads receipt photos with a VLM and drafts the guest message, with a human approval
gate for large vouchers.

Built for **JR2019035** (US-CA-Remote; applications open until at least **2026-10-20**). Same style as Labs 02-04 -
and on purpose it **repeats them**: every 🔁 marks a concept from an earlier lab (or the Oct 17 mock technical round)
getting its next rep ([docs/00-recall.md](docs/00-recall.md)). Synthetic data only; every company and traveler is fictional.

## What you will build
- **The agent:** a LangGraph state machine (think -> act), 9 tools behind two **MCP** servers (stdio on the laptop,
  streamable HTTP in Kubernetes), an engine-agnostic JSON action protocol, byte-identical (cache-stable) prompts and an
  approval gate for any voucher over $300.
- **A world you can score:** 200 golden storm-day travelers, a written policy
  ([Tern Care Commitments v3](data/policies/tern-care-v3.md)), an oracle and a scorer that names every violation
  (`wrong_amount:credit:600.0!=450.0`, `missing_hotel`, ...). "Is the agent good enough" becomes a number.
- **A mock engine** (OpenAI-compatible; emulates prefix caching, TTFT, ITL and faults), so the whole pipeline runs on a
  laptop and in CI before you rent a GPU.
- **An engine bake-off for agent traffic:** vLLM, SGLang, TensorRT-LLM, NIM and a naive Transformers baseline behind one
  launcher, one model name and one set of prompts; a task-level benchmark plus an AIPerf replay of the real sessions.
- **Profiling and optimization:** PyTorch profiler, Nsight Systems, DCGM, a trace analyzer, a roofline check and a cold-start
  parser; one change per experiment (prefix caching, CUDA graphs, FP8, FP8 KV cache, n-gram speculation, structured output).
- **LoRA "prompt baking":** SFT data from the oracle, TRL + PEFT on one GPU, DDP vs FSDP2 on eight, a tool-call eval and
  the adapter served next to the base model.
- **Architecture and cluster math:** a model-complexity calculator (dense, GQA, MLA, MoE, hybrid Mamba, VLMs), a
  parallelism calculator (DP/ZeRO/FSDP, TP, PP, EP, CP), nccl-tests against theory, TP/PP/DP sweeps on 8x H100.
- **A multimodal step:** synthetic receipt images, exact image-token math for two VLM families, extraction accuracy and
  TTFT vs resolution.
- **Ship it:** Dockerfile, a Helm chart (engine = vLLM | SGLang | TRT-LLM | NIM | mock), GitLab CI with an eval gate and a
  perf gate, a kind dry run, then 🔁 Lab 04's AKS or GKE cluster.
- **The acceleration ledger:** the operating point of every configuration (p95 <= 12 s and >= 95% policy-compliant), $ per
  1,000 disruptions, storm-day GPUs with N+1, and a waterfall chart.
- **The customer side:** discovery, PoC charter, POC evaluation report, a feature request, technical win plan and mutual
  action plan; a 90-minute workshop with a notebook; a blog draft; 10 break-fix drills; an interview drill.

## JD coverage map
| JD line (JR2019035) | Module | Proof you can show |
|---|---|---|
| Hands-on technical mentorship on the NVIDIA GenAI stack | M3-M6, M12 | the agent repo; workshop + notebook that runs on any laptop |
| Guide customers to develop and deploy Agentic AI workflows | M3, M9 | LangGraph + MCP agent; Helm chart; CI gates |
| ...while quantifying the benefits of accelerated computing | M4, M5, M10 | [ledger](docs/09-results-and-exec-readout.md): tasks/GPU-hour, p95, $ per 1k disruptions, storm-day GPUs |
| Demos and POCs | M1, M3, M11 | [PoC charter](sales/poc-charter.md) with a tool per criterion; [POC evaluation](sales/poc-evaluation-report.md) |
| Draft requirements for missing features | M11 | [feature request](sales/feature-request.md) with a repro and acceptance criteria |
| Educate on new technologies; training and workshops | M12 | [workshop/](workshop/): agenda, slides outline, facilitator guide, exercises, notebook |
| Collateral (notebook/blog) for industry use cases | M12 | [blog draft](collateral/blog.md) filled from the ledger |
| Partner with engineering, product and sales on design wins | M11 | [technical win plan](sales/technical-win-plan.md), [mutual action plan](sales/mutual-action-plan.md) |
| Product feedback through POC evaluations | M11 | POC evaluation report + feature request |
| Latest model architectures and their computational complexity | M2 | `archmath`: params, active params, KV bytes, FLOPs, roofline, MoE coverage ([docs/03](docs/03-model-complexity.md)) |
| Deploying and optimizing LLM inference with vLLM, TRT-LLM, SGLang | M4, M5 | one launcher, 11 variants, task-level + AIPerf results ([docs/07](docs/07-engine-cheatsheet.md)) |
| Training and fine-tuning with PyTorch / HF Transformers | M6 | TRL SFTTrainer + PEFT LoRA; tool-call eval |
| GPU cluster architecture; parallel processing for training and inference | M7 | nccl-tests vs theory; TP/PP/DP sweeps; DDP vs FSDP2 ([docs/04](docs/04-gpu-cluster-architecture.md)) |
| Docker, Kubernetes, GitLab, Linux CLI, Shell | M9, everywhere | Dockerfile, Helm, `.gitlab-ci.yml`, bash 3.2-safe scripts |
| Communication; leading workshops | M12, M13 | exec readout, workshop rehearsal, interview drill |
| *Stand out:* LangChain, LangGraph, MCP | M3 | StateGraph agent, two FastMCP servers, `langchain-mcp-adapters` |
| *Stand out:* multimodal LLMs and VLMs | M8 | Qwen2.5-VL on vLLM + hosted Llama 3.2 Vision; token math ([docs/06](docs/06-multimodal.md)) |
| *Stand out:* LLMs at scale on AWS, Azure or GCP | M9 | the same chart on 🔁 Lab 04's AKS or GKE (values per cloud) |
| *Stand out:* profiling latency, throughput, memory and I/O | M5 | profiles, DCGM counters, roofline check, cold start ([docs/08](docs/08-profiling-playbook.md)) |
| *Stand out:* mathematical understanding of parallelization | M7 | `parallelism_calc` with every formula ([docs/05](docs/05-parallelism-math.md)) |

## Time and cost
| Path | Modules | Hands-on time | GPU cost (on-demand) |
|---|---|---|---|
| **Fast path** (interview-ready) | M0, M2, M3, M4 (vLLM vs SGLang only), M7 on paper, M13 interview drill | ~9 h | ~$7-15 |
| **Full path** | M0-M13 | ~27 h | ~$90-155 |
| Cheaper M7 | a 4x H100 node: the 4-GPU subset (TP 1/2/4, nccl-tests on 4, DDP and FSDP2 on 4); 8-way rows stay on paper | ~same | saves ~$30-50 |

What drives the cost (on-demand list prices checked 2026-10-10 by IntuitionLabs; prices move, check yours):
- **1x H100 80GB** for M4-M6 and M8, ~12 h: **$1.99-4.29 per hour** (RunPod Community PCIe to Lambda SXM). The
  cheap end is often **H100 PCIe**: 2.0 TB/s instead of SXM's 3.35, so decode runs ~1.7x slower. Predict with
  `--gpu H100-PCIe` and put the real GPU in every bench label and `--gpu-name`.
- **8x H100 SXM node** for M7, ~2 h: **$31.92-49.24 per hour** (Lambda at $3.99 per GPU-hour; CoreWeave per node).
  The hyperscalers list more per GPU-hour on demand (AWS p5 $6.88, Google A3 $11.06, Azure ND H100 v5 $12.29).
- **Hosted endpoints** (build.nvidia.com) for M3 and M8: free developer credits; synthetic prompts only.
- **M9 cloud step:** 🔁 Lab 04's AKS (~$1.20 per hour) or GKE (~$1.95 per hour) for ~2 h, or kind on the laptop for $0.

Pause discipline (🔁 Lab 04): stop the GPU VM between sessions - its disk still bills, the GPU does not. An L40S 48GB
works for M4-M6 with smaller numbers; write the GPU name into every bench label.

## Repository map
```
ttg/            core package: world.py (scenarios, policy oracle, scorer) · mcp/ (ttg-ops, ttg-guest servers) ·
                agent/ (LangGraph graph, JSON protocol, prompts, LLM client, FastAPI) · mock/ (mock engine) ·
                bench/ (agent_bench, stats, scrape_metrics, export_aiperf, ledger)
data/policies/  Tern Care Commitments v3 - the policy the agent must follow
archmath/       M2: model-complexity calculator + real config.json files (Llama, Qwen3 MoE, DeepSeek-V3, Nemotron 3.5)
engines/        M4: launch.sh <engine> <variant> for vllm | sglang | trtllm | nim | hf, flag preflight, naive HF server
profiling/      M5: torch profiler, Nsight Systems, DCGM, trace analyzer, roofline check, cold start
finetune/       M6: oracle SFT data, TRL + PEFT LoRA, accelerate DDP / FSDP2 configs, tool-call eval, LoRA serving
parallel/       M7: parallelism calculator, nccl-tests, TP / PP / DP sweeps, DDP vs FSDP2 scaling
vlm/            M8: receipt images, image-token math, extraction scorer, TTFT vs resolution sweep
deploy/         M9: Dockerfile, Helm chart (deploy/helm/ttg), values for kind / AKS / GKE
scripts/        preflight, GPU VM bootstrap, image check, manifest validation, CI gate, load helpers
env/            pins.env (ONE source of truth for versions) + lab.env.example
bench/baselines perf-gate baselines (re-recorded on purpose, never by accident)
docs/           00 recall · 01 customer · 02 architecture + ADRs · 03 model complexity · 04 GPU cluster architecture ·
                05 parallelism math · 06 multimodal · 07 engine cheat sheet · 08 profiling playbook ·
                09 results + exec readout · 10 interview drill
sales/ workshop/ collateral/ drills/      M11-M13
tests/          offline suite: mock engine, no GPU, no torch
.gitlab-ci.yml  lint -> test -> eval-gate -> perf-gate (GPU runner, manual) -> build -> deploy   (.github/ mirror)
```

## Prerequisites
- A personal laptop (macOS bash 3.2 is fine) with Python 3.11+, git and `pip install -r requirements-dev.txt`.
  Docker, helm, kubectl and kind for M9 (`scripts/preflight.sh` tells you what is missing).
- Keys, all personal: **NVIDIA API key** (build.nvidia.com, hosted endpoints), **NGC API key** (NIM and TensorRT-LLM
  images), **Hugging Face token** with the Llama 3.1 license accepted (🔁 Lab 04). Qwen2.5-VL is Apache-2.0, no gate.
- A GPU cloud account that rents 1x H100 80GB by the hour (Brev, Lambda, RunPod or any cloud) and an 8x H100 SXM node
  for 2 hours. Ubuntu 22.04 or 24.04 with the NVIDIA driver; `scripts/gpu-vm-bootstrap.sh` does the rest.
- Optional: 🔁 Lab 04's AKS or GKE cluster for M9's cloud step, and a GitLab project (free tier) for the pipeline.

---

## M0 - Setup + recall (45 min, laptop) 🔁 Labs 01-04
```bash
unzip ttg-agentic-genai-lab.zip && cd ttg-agentic-genai-lab   # then push to your own GitHub/GitLab repo
python3 -m venv .venv && . .venv/bin/activate && pip install -r requirements-dev.txt
cp env/lab.env.example env/lab.env && $EDITOR env/lab.env     # keys + GPU_HOURLY_USD (never committed: .gitignore)
. env/lab.env && . env/pins.env
scripts/preflight.sh                                          # python, packages, keys present, optional tools
python3 -m pytest -q tests                                    # the whole offline suite (~890 tests), ~2 min
python -m ttg.world --n 200 --seed 7 --out data/scenarios.jsonl --stats
```
Then close the repo and answer the 20 questions in [docs/00-recall.md](docs/00-recall.md) before you read the answers.
**Checkpoint:** tests green; the stats line shows the golden mix: 106 cancelled / 94 delayed, 100 controllable,
35 overnight, 23 hotel bookings, 66 receipts, 26 approvals, 3-11 tool calls per traveler.

## M1 - Discovery + PoC charter (45 min)
Read [docs/01-customer-brief.md](docs/01-customer-brief.md). Role-play [sales/discovery-questions.md](sales/discovery-questions.md)
(a friend plays Tern's VP Guest Operations), then fill [sales/poc-charter.md](sales/poc-charter.md). Every success
criterion names the tool in this repo that will measure it:

| Criterion | Target | Measured by |
|---|---|---|
| Policy-compliant task success | >= 95% of the 200 golden travelers | `ttg.world.score` inside `agent_bench` |
| p95 end-to-end task latency at the storm operating point | <= 12 s | `agent_bench` levels -> `ledger` |
| GPU cost per 1,000 disruptions | <= $2.00 | `ledger` ($ per 1k tasks) |
| Storm-day capacity: 40,000 travelers in 6 h, peak x2, N+1 | a GPU count | `ledger --storm-tasks 40000 --storm-hours 6` |
| EU guest data stays in the EU; card numbers never reach the LLM | by design | [ADRs](docs/02-architecture-and-adrs.md), NetworkPolicy, payment tokens only |

**Say it:** "We'll measure your agent the way you pay for it - per handled disruption, at storm-day load, with a
pass or fail on your own policy - not as tokens per second."

## M2 - Model architectures + computational complexity (90 min; +30 min GPU optional) 🔁 Lab 01 roofline, Lab 02 KV math, Lab 04 sizer
```bash
python3 -m archmath.archmath                                    # every bundled architecture in one table
python3 -m archmath.archmath --roofline --gpu H100-SXM --batch 1,8,32 --ctx 4096
python3 -m archmath.archmath --roofline --model llama-3.1-70b --precision fp8 --gpus 2 --batch 1   # 🔁 mock round Q1
python3 -m archmath.archmath --coverage --batch 1,8,32,64,256   # MoE: unique experts touched per decode step
python3 -m archmath.archmath --attention-share                  # the quadratic part at 8K / 32K / 128K
python3 -m archmath.archmath --prefill 3000 --gpu H100-SXM      # TTFT floor for a 3,000-token agent prompt
python3 -m archmath.archmath --notes                            # every assumption behind the numbers
```
Work through [docs/03-model-complexity.md](docs/03-model-complexity.md). **Checkpoint questions:** Why is Llama 3.1 8B's
KV cache 128 KiB per token while DeepSeek-V3 (61 layers, MLA) stores less per token than that? Why does a "3B active"
MoE stream far more than 3B parameters' worth of bytes per decode step at batch 32? Which part of a long agent prompt
grows quadratically, and at what context does it start to matter? The optional GPU experiment (Qwen3-30B-A3B MoE vs
dense Llama 3.1 8B, ITL at batch 1 vs 32; 🔁 Lab 04's Nemotron 3.5 Lightning NIM adds the hybrid row) is in the same doc.
**Say it:** "Active parameters set the FLOPs, but the bytes you stream per step set the latency - and for an MoE at
batch 32 that is most of the experts, not 3 billion parameters."

## M3 - The agentic workflow: LangGraph + MCP (2 h, laptop + hosted)
Start on the mock engine, then point the same code at a hosted model. Nothing in the agent changes - only `TTG_*`.
```bash
uvicorn ttg.mock.fake_llm:app --port 8000 --log-level warning &         # OpenAI-compatible mock engine
export TTG_BASE_URL=http://localhost:8000/v1 TTG_MODEL=ttg-llm TTG_TRACE_DIR=results/traces   # one JSON trace per task
python -m ttg.agent.run --scenario S0003                                # thought -> action -> observation, step by step
python -m ttg.agent.run --n 20 --concurrency 4                          # "20/20 policy-compliant"
python -m ttg.mcp.ops_server --transport streamable-http --port 8101 &   # the same tools as network services
python -m ttg.mcp.guest_server --transport streamable-http --port 8102 &
TTG_MCP_MODE=http python -m ttg.agent.run --n 8                         # what Kubernetes runs (M9)
```
Read [ttg/agent/graph.py](ttg/agent/graph.py) (the state machine) and [ttg/agent/protocol.py](ttg/agent/protocol.py)
(one JSON action per step, with fix-it feedback on bad JSON). [docs/02](docs/02-architecture-and-adrs.md) says why the
lab uses a JSON action protocol instead of each engine's native tool calling: it keeps the bake-off fair.
Now a real model - **synthetic travelers only** on hosted endpoints:
```bash
export TTG_BASE_URL=https://integrate.api.nvidia.com/v1 TTG_API_KEY=$NVIDIA_API_KEY TTG_MODEL=meta/llama-3.1-8b-instruct
python -m ttg.agent.run --n 20 --concurrency 2
python -m ttg.agent.run --n 20 --concurrency 2 --json > results/hosted-8b.json
python3 -c "import json,collections; r=json.load(open('results/hosted-8b.json')); print(collections.Counter(v.split(':')[0] for x in r for v in x['violations']))"
TTG_MODEL=meta/llama-3.3-70b-instruct python -m ttg.agent.run --n 20 --concurrency 2    # a bigger model, same policy
```
**Checkpoint:** the mock scores 20/20; you can name the hosted 8B's top two violation types and show one failing trace
(`results/traces/`). Write down the 8B vs 70B success rates - M6 tries to close that gap with a LoRA.
**Say it:** "The agent is a state machine you can test: every step is one JSON action, every tool is an MCP server, and
every run is scored against your policy - so 'is it good enough' is a number, not a demo."

## M4 - Engine bake-off for agent traffic (3 h, 1x H100) 🔁 Lab 04 bake-off, Labs 01-04 AIPerf
On the GPU VM (run the bench there too: from the laptop, every LLM call would add a WAN round trip - 6 calls x 60 ms
is ~0.4 s per task):
```bash
git clone <your repo> ttg && cd ttg            # or scp the zip
cp env/lab.env.example env/lab.env && $EDITOR env/lab.env    # HF_TOKEN, NGC_API_KEY, GPU_HOURLY_USD
scripts/gpu-vm-bootstrap.sh --pull all         # driver check, Docker + Container Toolkit, Python 3.12 venv, logins, images
. .venv/bin/activate && . env/lab.env && . env/pins.env
for e in vllm sglang trtllm nim; do engines/preflight.sh $e; done    # every flag the launcher uses exists in that image
```
Then one engine at a time, always in the same order. `launch.sh` waits for health and prints the exact bench line
for that engine (metrics and cache-reset URLs filled in) - copy it, change only the label:
```bash
engines/launch.sh hf base         # the "before": transformers generate(), one request at a time
python -m ttg.bench.agent_bench --label hf-base --engine hf --levels 1,4 --n 8 \
    --gpu-name H100-80GB --gpus 1 --gpu-hourly $GPU_HOURLY_USD
engines/launch.sh vllm nocache    # prefix caching OFF
engines/launch.sh vllm base       # prefix caching ON, CUDA graphs ON
python -m ttg.bench.agent_bench --label vllm-base --engine vllm --levels 1,4,16,32,64 --n 48 \
    --gpu-name H100-80GB --gpus 1 --gpu-hourly $GPU_HOURLY_USD \
    --metrics-url http://localhost:8000/metrics --reset-url http://localhost:8000/reset_prefix_cache
engines/launch.sh sglang nocache && engines/launch.sh sglang base     # RadixAttention off / on
engines/launch.sh trtllm base && engines/launch.sh trtllm fp8         # KV block reuse; ModelOpt FP8 checkpoint
engines/launch.sh nim base                                            # 🔁 Lab 04's NIM, same model
python -m ttg.bench.scrape_metrics --url http://localhost:8000/metrics   # prefix-cache hit ratio, queue depth
engines/logs.sh --save results/logs/vllm-base.log && engines/stop.sh     # keep the logs for M5's cold start
```
The engine view, with AIPerf replaying the agent's real sessions (multi-turn, shared system prompt):
```bash
python -m ttg.bench.export_aiperf --n 64 --out results/aiperf
pip install "aiperf==${PIN_AIPERF}"
aiperf profile --model ttg-llm --url http://localhost:8000 --endpoint-type chat --streaming \
    --tokenizer meta-llama/Llama-3.1-8B-Instruct --system-prompt-file results/aiperf/system_prompt.txt \
    --input-file results/aiperf/sessions.jsonl --custom-dataset-type multi_turn \
    --extra-inputs max_tokens:96 --concurrency 16 --benchmark-duration 120 --artifact-dir results/aiperf/vllm-base-c16
```
Stretch: the industry agentic replay, `aiperf profile ... --scenario inferencex-agentx-mvp` (15 minutes or more per run).
[engines/README.md](engines/README.md) has the full procedure and what to record per run;
[docs/07](docs/07-engine-cheatsheet.md) compares the engines.
**Checkpoint:** one table of tasks per second at p95 <= 12 s for hf, vLLM, SGLang, TRT-LLM and NIM; prefix-cache hit
ratio near 90% with caching on and 0% with it off; one sentence on why `hf-base` misses the SLO at concurrency 4.
**Say it:** "For agents, the shared prefix is most of every prompt: an engine that computes it once wins more than any
kernel tweak - and we measured it per task, not per token."

## M5 - Profile and optimize: latency, throughput, memory, I/O (3 h, 1x H100)
The method in [docs/08-profiling-playbook.md](docs/08-profiling-playbook.md): measure, name the bottleneck class
(compute, memory bandwidth, CPU/scheduler, communication, I/O), change ONE thing, measure again.
```bash
profiling/torch_profile.sh                        # vLLM + --profiler-config, agent load, /start_profile ... /stop_profile
python3 profiling/analyze_trace.py results/profiles/torch/          # GPU busy %, time by kernel class, idle gaps, top 20
profiling/dcgm_watch.sh                           # SM active, tensor active, DRAM active, PCIe bytes -> CSV
profiling/nsys_capture.sh                         # Nsight Systems on the same window (cudaProfilerApi capture range)
python3 profiling/roofline_check.py results/bench/vllm-base.json    # measured ITL vs the bytes/bandwidth floor 🔁 Lab 01
python3 profiling/coldstart.py results/logs/*.log                   # weight-load GB/s: local NVMe vs network disk
```
Then the experiments - one launcher variant or one env var each, one bench label each:

| Experiment | Change | Expect |
|---|---|---|
| CUDA graphs off | `engines/launch.sh vllm eager` | ITL up at low concurrency: CPU launch overhead is back |
| FP8 weights | `engines/launch.sh vllm fp8` | half the weight bytes per decode step: lower ITL, more room for KV |
| FP8 KV cache | `engines/launch.sh vllm fp8kv` | twice the KV capacity: more sessions before preemption |
| n-gram speculation | `engines/launch.sh vllm spec` (and `sglang spec`) | faster tool-call JSON at low concurrency; the gain shrinks under load |
| Structured output | `TTG_STRUCTURED=json_schema` | malformed JSON becomes impossible (a wrong argument can still retry) |
| Cache-busting prompt (drill 1) | `TTG_BUST_CACHE=1` | cached tokens drop to 0; TTFT and $ per task rise |
| Verbose reasoning (drill 10) | `TTG_THOUGHTS=verbose TTG_MAX_TOKENS=1024` | ~8x output tokens per task; p95 blows through the SLO |

**Checkpoint:** a profile summary (GPU busy %, top kernel classes) and one sentence per experiment: what moved, and why.
**Say it:** "We profile before we tune: the trace tells us whether we're bound by bandwidth, compute, the CPU or I/O,
and every change gets its own row in the ledger."

## M6 - Fine-tune with PyTorch + Hugging Face: LoRA "prompt baking" (2.5 h, 1x H100) 🔁 Lab 04 LoRA
The full system prompt is ~2,000 tokens and every LLM step re-reads it. Train the policy into an adapter, then serve it
with the 560-character short prompt:
```bash
python3 finetune/make_sft_data.py                     # oracle SFT data: 1,000 travelers (seed 11) -> 6,170 steps; 200 held out
python3 finetune/train_lora.py --dry-run              # the plan without torch: steps, 41.9M trainable params (0.52%), memory
# Training needs torch + the Hugging Face stack: run it inside the NGC PyTorch container on the GPU VM
docker run --rm -it --gpus all --ipc=host -v "$PWD":/lab -w /lab -e HF_TOKEN \
    -v ~/.cache/huggingface:/root/.cache/huggingface "nvcr.io/nvidia/pytorch:${PIN_NGC_RELEASE}-py3" bash
. env/pins.env && pip install "trl==${PIN_TRL}" "peft==${PIN_PEFT}" "transformers==${PIN_TRANSFORMERS}" \
    "accelerate==${PIN_ACCELERATE}" "datasets==${PIN_DATASETS}"      # inside the container
python3 finetune/train_lora.py                        # TRL SFTTrainer + PEFT LoRA r=16, BF16 -> adapters/ttg-short
exit                                                  # back on the VM host
engines/launch.sh vllm lora                           # ONE engine serves the base model (ttg-llm) and the adapter (ttg-short)
python3 finetune/eval_toolcalls.py --model ttg-llm --system-variant full      # before
python3 finetune/eval_toolcalls.py --model ttg-llm --system-variant short     # control: short prompt, no training
python3 finetune/eval_toolcalls.py --model ttg-short --system-variant short   # after
TTG_PROMPT_VARIANT=short TTG_MODEL=ttg-short python -m ttg.bench.agent_bench --label vllm-lora --engine vllm \
    --variant lora --levels 1,4,16,32,64 --n 48 --gpu-name H100-80GB --gpus 1 --gpu-hourly $GPU_HOURLY_USD \
    --metrics-url http://localhost:8000/metrics --reset-url http://localhost:8000/reset_prefix_cache
```
All three evals go through the **same** engine, chat template and sampling settings. If the base model is scored
through one inference path and the adapter through another, you are measuring the paths, not the fine-tune. The
control row is why the adapter exists: the short prompt alone should fail. [finetune/README.md](finetune/README.md)
has the details.
**Checkpoint:** the adapter's action and argument exact-match on the held-out travelers is at least the base model's with
the full prompt; prompt tokens per LLM call drop by ~1,800; `vllm-lora` gets its own ledger row.
**Say it:** "We baked a 2,000-token policy prompt into a rank-16 adapter: fewer prefill tokens on every step, the same
policy compliance - and the eval decides, not a demo."

## M7 - GPU cluster architecture + parallelism math (1 h on paper + 2 h on 8x H100) 🔁 mock round Q1, Lab 02 Dynamo
On paper first - [docs/04-gpu-cluster-architecture.md](docs/04-gpu-cluster-architecture.md) (DGX H100, NVSwitch,
rail-optimized InfiniBand, GB200 NVL72) and [docs/05-parallelism-math.md](docs/05-parallelism-math.md):
```bash
python3 -m parallel.parallelism_calc --links          # the link bandwidth and latency table everything uses
python3 -m parallel.parallelism_calc --examples       # five worked examples (training and inference)
python3 -m parallel.parallelism_calc --train --model llama-3.1-8b --dp 8 --zero 3 --seq 4096 --mbs 2
python3 -m parallel.parallelism_calc --infer --model llama-3.1-70b --precision fp8 --tp 1,2,4,8 --batch 1,32 --link nvlink4
```
Then on the 8x H100 SXM node (bootstrap it like M4):
```bash
nvidia-smi topo -m && nvidia-smi nvlink -s            # NV18 between every GPU pair = NVLink 4 through NVSwitch
parallel/nccl_tests.sh 8                              # all_reduce, all_gather, reduce_scatter, alltoall: 8 B .. 4 GB
parallel/nccl_tests.sh 8 --nvls-off                   # NVLink SHARP off: no in-switch reduction
python3 -m parallel.compare_nccl results/nccl/ --link nvlink4    # measured bus bandwidth vs the calculator's link rate
parallel/sweep_parallel.sh                            # vLLM: 70B FP8 at TP 2/4/8 and PP 2 x TP 4; 8B at TP 1/2, PP 2, DP 2
# inside the NGC PyTorch container from M6 (torch + TRL + accelerate); each run saves to results/scaling/, not adapters/
TRAIN_ARGS="--max-steps 60" parallel/train_scaling.sh  # the M6 LoRA job: 1 GPU vs DDP x8 vs FSDP2 x8
python3 -m parallel.scaling_report results/scaling/   # samples/s, speedup, scaling efficiency, peak memory per GPU
```
On a 4x node instead: `parallel/nccl_tests.sh 4`, `parallel/sweep_parallel.sh 8b` and `parallel/train_scaling.sh 1 2 4`
(and check `nvidia-smi topo -m`: a 4-GPU HGX board links the GPUs directly, without NVSwitch, so there is no NVLS).
**Checkpoint:** measured all-reduce bus bandwidth as a fraction of NVLink 4's ~450 GB/s per direction; for the 8B on
2 GPUs, which of TP 2, PP 2 and DP 2 wins at concurrency 1 and at 32 (and why); DDP vs FSDP2 samples per second per GPU
and peak memory. Practice out loud: "Describe the structure of the NVL72 as best as you can." (model answer in docs/04).
**Say it:** "Tensor parallelism lives inside the NVLink domain and data parallelism crosses the network - the math
tells us where every byte goes before we rent the cluster."

## M8 - Multimodal: reading receipt photos with a VLM (2 h, 1x H100 + hosted)
```bash
python3 -m vlm.make_receipts                          # 66 synthetic receipts (seed 7) + labels.jsonl -> data/receipts/
python3 -m vlm.make_receipts --noise 0.6 --out data/receipts/noisy   # skew, blur, sensor noise: a phone photo, not a scan
python3 -m vlm.vlm_tokens --table                     # Qwen2.5-VL: 1 token per 28x28 px; Llama 3.2 Vision: 1,601 per tile
engines/launch.sh vllm vlm                            # Qwen2.5-VL-7B-Instruct on vLLM, served as ttg-llm
python3 -m vlm.extract_receipt --dir data/receipts --base-url http://localhost:8000/v1 --model ttg-llm \
    --label qwen25vl-clean --concurrency 8            # field accuracy vs the ground truth
python3 -m vlm.extract_receipt --dir data/receipts/noisy --base-url http://localhost:8000/v1 --model ttg-llm \
    --label qwen25vl-noisy --concurrency 8
python3 -m vlm.vlm_sweep --base-url http://localhost:8000/v1 --model ttg-llm --label qwen25vl   # tokens + TTFT vs resolution
python3 -m vlm.extract_receipt --dir data/receipts --n 20 --concurrency 2 --base-url https://integrate.api.nvidia.com/v1 \
    --model meta/llama-3.2-11b-vision-instruct --api-key-env NVIDIA_API_KEY --label hosted-llama32v   # synthetic images only
```
Then put the VLM inside the agent: with `TTG_VLM_BASE_URL` set, the MCP guest server answers `read_receipt` by reading
the image. The text model and the VLM share the H100 as two vLLM instances (45% and 40% of memory); the exact commands
are in [docs/06](docs/06-multimodal.md) section 7. The money question is one command:
```bash
export TTG_VLM_BASE_URL=http://localhost:8001/v1 TTG_VLM_MODEL=ttg-vlm
python3 -m vlm.agent_receipts --concurrency 8 --label qwen25vl     # the 35 receipt tasks: oracle pass, then VLM pass
```
**Checkpoint:** field and reimbursement accuracy for clean vs noisy images and self-hosted vs hosted; measured image tokens
equal the prediction; the resize you would put in production; how many of the 35 tasks the VLM changes, and the
violations that did it (`wrong_amount:reimbursement`).
**Say it:** "An image is prefill: one 12-megapixel phone photo is 15,552 tokens on Qwen2.5-VL, several times the whole
agent prompt - so we set a resolution policy before we set a GPU budget."

## M9 - Ship it: Docker, Helm, GitLab CI gates, Kubernetes (2 h) 🔁 Lab 04 chart, clusters and perf gate
[deploy/README.md](deploy/README.md) is the walkthrough. On the laptop, the whole chart runs on kind with the mock engine:
```bash
scripts/validate-manifests.sh                         # helm template every values file | kubeconform -strict
docker build -f deploy/Dockerfile -t ttg-agent:dev .  # one image: agent API, MCP servers or mock, chosen by command
kind create cluster --name ttg --image "$PIN_KIND_NODE_IMAGE" && kind load docker-image ttg-agent:dev --name ttg
helm upgrade --install ttg deploy/helm/ttg -n ttg --create-namespace -f deploy/values/kind.yaml --wait
helm test ttg -n ttg                                  # one golden disruption through the API
kubectl -n ttg port-forward svc/ttg-agent 8080:8080 &
python -m ttg.bench.agent_bench --label kind-storm --api http://localhost:8080 --levels 8,32 --n 64
```
GitLab (the JD names it): push the repo and watch `.gitlab-ci.yml` run **lint -> test -> eval-gate -> perf-gate ->
build -> deploy**. The eval gate runs 40 golden travelers on the mock and fails below 95% policy compliance; the perf
gate is manual on a GPU runner and fails on a 10% p95 or cost regression against
[bench/baselines/](bench/baselines/README.md). Then the cloud: the same chart on 🔁 Lab 04's cluster with
`-f deploy/values/aks.yaml` (or `gke.yaml`) - the engine is a values change, not a new deployment.
**Checkpoint:** `helm test` passes on kind; set `TTG_MODEL=mock-wrong` in the eval-gate job and watch the pipeline stop.
**Say it:** "Every merge runs your golden travelers through the agent: if policy compliance drops below 95% or p95
regresses by 10%, the pipeline stops - the gate decides, not the demo."

## M10 - Quantify: the acceleration ledger (1 h)
```bash
python -m ttg.bench.ledger results/bench/*.json --slo-p95-s 12 --min-success 0.95 \
    --order hf-base,vllm-nocache,vllm-base,sglang-base,trtllm-base,trtllm-fp8,nim-base,vllm-fp8,vllm-spec,vllm-lora \
    --storm-tasks 40000 --storm-hours 6 --peak-factor 2 --out results/ledger
```
`results/ledger/ledger.md` lists the operating point per configuration; `waterfall.png` shows $ per 1,000 disruptions
step by step from the naive baseline. Fill [docs/09-results-and-exec-readout.md](docs/09-results-and-exec-readout.md)
with YOUR numbers and record a 5-minute readout for Tern's VP Guest Operations, CIO and CFO.
**Checkpoint:** you can say, from your own runs, the x capacity vs `hf-base`, the $ per 1,000 disruptions vs the $2.00
target, and the storm-day GPU count with one spare replica.
**Say it:** "On storm day you need this many GPUs instead of that many: that is the acceleration, priced per disruption."

## M11 - Design win: POC evaluation, feature request, win plan (1 h)
Fill [sales/poc-evaluation-report.md](sales/poc-evaluation-report.md) from the ledger: what passed, what did not, what
it would take. Finish [sales/feature-request.md](sales/feature-request.md) (per-session prefix-cache accounting across
engines, with the lab's repro) - that is the "draft requirements for missing features" line of the JD, done the way
product teams can use it. Update [sales/technical-win-plan.md](sales/technical-win-plan.md) and
[sales/mutual-action-plan.md](sales/mutual-action-plan.md).
**Say it:** "Here is what we proved, what we could not, and the one product request that closes the gap - with your
numbers attached."

## M12 - Teach it: workshop, notebook, blog (2 h)
```bash
python3 workshop/build_notebook.py                    # builds workshop/ttg-workshop.ipynb (runs on the mock)
jupyter-execute --timeout=600 workshop/ttg-workshop.ipynb   # nbclient's runner: proves every cell runs on a laptop
```
Rehearse [workshop/agenda.md](workshop/agenda.md) (90 minutes, hands-on) with the
[slides outline](workshop/slides-outline.md), [facilitator guide](workshop/facilitator-guide.md) and
[exercises](workshop/exercises.md). Then finish the [blog draft](collateral/blog.md) with numbers from your ledger;
[collateral/README.md](collateral/README.md) covers review and publishing.
**Say it:** "Everyone leaves with an agent running on their laptop and a number they measured themselves."

## M13 - Break-fix drills + interview drill (2 h) 🔁 Labs 02-04
One person injects, the other diagnoses from symptoms only, fixes it and says the one-liner:
```bash
drills/inject.sh list                                                 # 1-10 + 8b, and which ones need a GPU
python -m ttg.bench.agent_bench --label drill-base --levels 8 --n 24  # a healthy run to compare against
eval "$(drills/inject.sh 1)"                                          # inject drill 1 into THIS shell (export/unset only)
python -m ttg.bench.agent_bench --label drill-1 --levels 8 --n 24     # now diagnose from the symptoms
python3 drills/diff_runs.py results/bench/drill-base.json results/bench/drill-1.json   # settings + metrics that moved
eval "$(drills/inject.sh 1 revert)" && . env/lab.env                  # undo, restore your defaults
```
Eleven drills, from a cache-busting timestamp to NCCL P2P off: [drills/README.md](drills/README.md).
Then [docs/10-interview-drill.md](docs/10-interview-drill.md): ~25 questions across every JD line, two first-hand
questions with model answers, and a hospitality and travel section (use cases beyond disruptions).

---

## Teardown
```bash
engines/stop.sh                                     # on each GPU VM, then terminate the VM in the provider console
kind delete cluster --name ttg
helm uninstall ttg -n ttg                           # on the Lab 04 cluster, then pause or delete it (🔁 Lab 04 tf.sh)
```

## What was validated before you got this (sandbox, no GPU)
The full offline test suite (888 tests): the world and oracle (200/200 golden travelers policy-compliant), the JSON protocol, both MCP
servers over stdio and streamable HTTP, the agent end to end on the mock (success, invalid-JSON repair, step guard,
engine-down handling, both drills), the bench, ledger, metrics parsing, AIPerf export and the agent API; `archmath`
against known values (Llama 3.1 8B KV = 128 KiB/token, DeepSeek-V3 ~37B active); the parallelism formulas; VLM token
math and receipt rendering; every `engines/launch.sh` variant in dry-run mode with flags checked against the vLLM 0.31.0,
SGLang 0.5.22 and AIPerf 0.13.0 wheels and the TensorRT-LLM docs; the trace analyzer; SFT data and the training dry run;
`helm template` + `kubeconform -strict` for every values file and engine kind; pins; the CI config; every docs link.
On the mock (emulated timing, laptop CPU - the shape, not GPU numbers): prefix caching moved 32 concurrent tasks from
7.9 to 9.7 tasks per second at a 94% hit ratio; the cache-bust drill took the hit ratio to 0%; verbose reasoning
multiplied output tokens per task by ~8.
Three independent reviews then checked the core code, the numbers and hardware facts in the docs, and the GPU scripts
against the real vLLM, SGLang, TensorRT-LLM, TRL, PEFT and accelerate sources; every confirmed finding is fixed and the
core fixes carry regression tests (`tests/test_core_regressions.py`). The agent image was built and all four roles ran
as uid 10001 on a read-only root filesystem.
**Not run here:** anything that needs a GPU, your keys, gated weights or a cloud account - those are the modules.

## Known caveats (verify on the day)
- `engines/preflight.sh <engine>` before every rental: image tags and flags move faster than this README.
- TensorRT-LLM speculative-decoding YAML keys differ by release, so `trtllm spec` is a stretch, not a launcher variant.
- NIM's served-model-name variable and FP8 profile IDs differ by NIM version: check `list-model-profiles` (🔁 Lab 04).
- vLLM's cache-reset endpoint needs `VLLM_SERVER_DEV_MODE=1` (the launcher sets it, lab only). trtllm-serve and NIM have
  no reset endpoint: relaunch for a cold cache.
- Hosted endpoints may reject `response_format`; the agent then reports `llm:` errors, so run `TTG_STRUCTURED` on your own engine.
- Prices and GPU availability change weekly - re-check before you quote a $ per 1,000 disruptions to anyone.
