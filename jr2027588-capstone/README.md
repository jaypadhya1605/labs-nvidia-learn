# JR2027588 Capstone: everything the Senior Solutions Architect, AI Cloud Services role needs, in one lab

One end-to-end path through Labs 02-05, cut down to **only what JR2027588 asks for**. Each phase has the same five
parts:

1. **Why**: the job-description line it proves.
2. **Learn**: three rungs, Basic, then Intermediate, then Advanced. Each rung points to the exact doc to read.
3. **Do**: exact commands, copied from the lab READMEs that were tested in the sandbox.
4. **Checkpoint**: what "done" looks like.
5. **Knowledge session**: `python quiz/quiz.py --phase Pn`, with basic, intermediate and advanced questions
   graded on the spot. Misses come back the next day in `--review`.

The code stays where it was built and tested. This capstone is the route through it, with nothing extra. You work
from `~/nvidia-labs/` and each phase says which folder to `cd` into.

| Folder | Role in this capstone |
|---|---|
| `plv-multicloud-nim-lab/` (Lab 04) | **The spine.** It was built for JR2027588: NIM on AKS and GKE, GitOps, AIPerf price-performance, engines, LoRA MLOps, air gap, sales and workshop material |
| `ttg-agentic-genai-lab/` (Lab 05) | Physics and GPU-systems depth only: `archmath` (model complexity, roofline), the parallelism calculator, GPU cluster architecture (NVLink, NVL72), engine cheat sheet, profiling playbook |
| `kha-genai-inference-lab/` (Lab 02) | Concepts only: MIG vs time-slicing, Dynamo KV-aware routing and disaggregation (read the M10 results and YAMLs) |
| `lfy-secure-agent-lab/` (Lab 03) | Not used. Security is a different req (JR2027074) |

## JD coverage: every line, the phase that proves it

| JR2027588 says | Phase | What you can show |
|---|---|---|
| Work with cloud service providers to develop and demonstrate solutions on NVIDIA hardware and software | P3, P4, P10 | Same chart on AKS + GKE; CSP cheat sheet; managed-vs-self-hosted scorecard |
| Build and deploy solutions at scale using NVIDIA AI software on cloud GPU platforms | P3, P4, P7 | GitOps fleet, HPA on queue depth, cold-start table per cloud |
| Custom proofs of concept for customers' business needs | P2, P5, P12 | PoC charter with a tool per criterion; results table; exec readout |
| Partner with sales and developer relations to identify and close opportunities | P2, P12 | Discovery, technical win plan, mutual action plan |
| Regular technical customer meetings: project updates, features, new technology, debugging | P11 | Weekly sync pack, what's-new, 10 break-fix drills |
| Presentations and workshops on NVIDIA products | P12 | Workshop agenda, slide outline, executed notebook, facilitator guide |
| *Required:* deploying AI/ML in AWS, GCP, Azure or OCI | P3, P10 | Terraform for AKS + GKE (EKS/OKE stretch) |
| *Required:* DevOps/MLOps: containers, Kubernetes, data center deployments | P3, P8, P9 | Argo CD, MLflow + release gate, air-gapped RKE2 |
| *Stand out:* NVIDIA GPUs and SDKs (CUDA, Triton, TensorRT-LLM) | P1, P3, P6 | CUDA bandwidth probe; Triton/TensorRT portability; trtllm-serve vs NIM |
| *Stand out:* full SDLC: system design, architecture, testing | all | ADRs, ~70 tests, CI workflows, perf gate |
| *Stand out:* Python + system-level GPU experience | P1, P5, P7, P11 | archmath, selector, parallelism math, profiling |
| *Stand out:* professional cloud SA certification | P10 | The CSP cheat sheet doubles as AZ-305 practice |

## Deliberately left out (and why)
- **LangGraph, MCP and agents** (Lab 05 M3): not in this JD. One line is enough in an interview: "agents are the same
  inference physics with a long shared prefix."
- **VLM receipts** (Lab 05 M8) and **Lab 05's agent-traffic engine bake-off** (M4): Lab 04 M6 covers engines for this
  role.
- **Guardrails and red-teaming** (Lab 03): a different req.
- **Lab 02's landing zone**: it repeats what Lab 04 builds. You keep Lab 02's concepts (MIG, Dynamo) through reading
  and quizzes.

## Paths, time and cost

| Path | Phases | Hands-on time | Cloud cost |
|---|---|---|---|
| **Fast path** (interview-ready) | P0-P7, P10-P13. P7's GPU part is optional | ~30 h | ~$40-50 (Lab 04 fast path on Spot, paused between sessions) |
| **Full path** | P0-P13 | ~45 h | ~$110-150 |

Running costs from Lab 04 (2026-10-15 list prices; check yours): **AKS ~$1.20/h**, **GKE ~$1.95/h**. **Pause every
night**: `infra/tf.sh aks pause`, `infra/tf.sh gke pause`.

**Rules:**
- Personal cloud accounts, personal keys and synthetic data only.
- Never use a company subscription or work login.
- Keys live in each lab's `env/lab.env`, never in `.txt` files.

## Daily rhythm (the knowledge sessions)
- **Start of every session, 10 min:** `python quiz/quiz.py --review`. Spaced repetition: a miss returns the next day,
  and a hit returns in 3, then 7, then 14 days.
- **End of every phase:** `python quiz/quiz.py --phase Pn`. Pass at 80% or more, then move on. Below 80%, redo that
  phase's **Learn** rungs and retry.
- **Mock days:** `python quiz/quiz.py --mock 12`, then `interview` with the coach.
- `python quiz/quiz.py` alone shows status: every phase, your score and what is due.

The quiz runs in Windows PowerShell (`py quiz\quiz.py`) or in WSL (`python3 quiz/quiz.py`). Both use the same
progress file if you run them from the same folder.

## 12-day plan (Day 1 = Wed Nov 18, 2026)
Weekdays ~3 h, weekends ~8 h. **(core)** = the fast path. If you fall behind, drop non-core work first, never the
mocks.

| Day | Date | Work |
|---|---|---|
| 1 | Wed Nov 18 | (core) P0 setup + request GPU quotas today (Azure A100 Spot, GCP G4 Spot). P1 Learn: Basic and Intermediate |
| 2 | Thu Nov 19 | (core) P1 Do + Advanced + quiz. P2 + quiz |
| 3 | Fri Nov 20 | (core) P3: both clusters, GitOps, CUDA probe. If one quota is pending, start with the other cloud |
| 4 | Sat Nov 21 | (core) P4 + P5, quizzes. **Mock 1** (`--mock 12` + coach `interview`) |
| 5 | Sun Nov 22 | (core) P6 (NIM precisions, trtllm-serve, Triton portability). P7 on paper + HPA/cold start. Pause clusters |
| 6 | Mon Nov 23 | (core) P10 + P11 drills 1-7 on the clusters, quizzes |
| 7 | Tue Nov 24 | (core) P12 workshop + recorded 5-minute readout |
| 8 | Wed Nov 25 | P8 part 1 (MIG + dataset) **or** P7 paper deep-dive. **Mock 2**: system design (P13-A2) |
| 9 | Thu Nov 26 | Thanksgiving: `--review` only (15 min) |
| 10 | Fri Nov 27 | P8 part 2 **or** P9 air gap, or re-run your two weakest phases (`--stats`) |
| 11 | Sat Nov 28 | (core) **Mock 3**: full loop. Push the repo. **Tear down every cluster** |
| 12 | Sun Nov 29 | Cheat sheet, ten 1-minute answers, questions for interviewers. No new labs |

If the technical round is booked sooner:
- **5-day track:** P0-P5 on the cloud, P6-P7 on paper, P13 mocks.
- **2-day track:** no cloud. P1, P2, P6, P7 and P10 on paper, every quiz, two mocks.

---

## P0 - Setup on Windows + WSL2 (60-90 min, laptop, $0)
**Why:** a working toolchain; nothing else runs without it.

**Learn**
- *Basic:* labs run in WSL2 Ubuntu, from `~/nvidia-labs` on the Linux filesystem. Your laptop is ARM64 with no
  NVIDIA GPU, so every GPU step runs on cloud nodes that the laptop drives. Read [WINDOWS-SETUP.md](WINDOWS-SETUP.md).
- *Intermediate:* keys and pins. `env/lab.env` holds secrets and is git-ignored. `env/pins.env` is the single source
  of truth for versions, and CI fails if anything drifts from it.
- *Advanced:* the Plover API image must be `linux/amd64`. On an ARM laptop, build it in CI by pushing a `v1.0.0` tag
  (`release.yml`) or cross-build with QEMU (WINDOWS-SETUP.md section 5).

**Do** (Ubuntu terminal). Virtual environments live in `~/.venvs`, outside the repos, so they never get committed:
```bash
cd ~/nvidia-labs/plv-multicloud-nim-lab
python3 -m venv ~/.venvs/plv && . ~/.venvs/plv/bin/activate && pip install -r requirements-dev.txt
cp env/lab.env.example env/lab.env && $EDITOR env/lab.env              # subscriptions, project, keys, GITHUB_OWNER
source env/lab.env
scripts/preflight.sh                                                   # tools, logins, GPU quota, NGC images, HF access
python3 -m pytest -q tests                                             # ~70 tests, no cloud needed
```
Argo CD pulls from **your** GitHub repo, so publish the lab first:
- Make it a **public** repo. It holds synthetic data only, and `.gitignore` already keeps `env/lab.env` out.
- A public repo lets Argo CD pull without credentials.
```bash
git init -b main && git add . && git commit -m "Lab 04 baseline"
gh repo create plv-multicloud-nim-lab --public --source . --push       # must match GIT_REPO_URL in env/lab.env
gitops/set-repo-url.sh && git add gitops && git commit -m "point GitOps at my fork" && git push
scripts/check-images.sh                                                # every pinned nvcr.io image pullable with YOUR key
```
**Checkpoint:** preflight passes; tests green; the repo is on GitHub; `check-images.sh` shows every image pullable.
**Knowledge session:** `python3 ~/nvidia-labs/jr2027588-capstone/quiz/quiz.py --phase P0`

---

## P1 - GPU + LLM inference physics (3-4 h, laptop, $0)
**Why:** "system-level experience with GPU-based systems", plus every sizing question a customer asks. This is the
single most important phase.

**Learn**
- *Basic:* prefill is compute-bound and sets TTFT. Decode is memory-bandwidth-bound and sets ITL. Weights = parameters
  x bytes (Llama 3.1 8B BF16 = 16.06 GB). Read `ttg-agentic-genai-lab/docs/03-model-complexity.md`, sections 1-3.
- *Intermediate:* decode floor = bytes / bandwidth (8.3 ms on an A100 PCIe; 53 ms on an L4, so the L4 is vetoed). KV
  bytes per token = 2 x layers x KV heads x head dim x bytes (128 KiB for the 8B, 320 KiB for the 70B). Read
  `plv-multicloud-nim-lab/docs/03-instance-selection.md`.
- *Advanced:* roofline and ridge point (H100: 989 / 3.35 ≈ 295 FLOP/byte); why batching is nearly free until the
  compute roof (A100 batch 369); MoE "active" parameters vs bytes streamed; 70B FP8 on 2x H100 (~30 sequences at 8K,
  ~10.5 ms/token). Read the rest of `docs/03-model-complexity.md` and `ttg-agentic-genai-lab/docs/00-recall.md`
  answers 1-6.

**Do:**
```bash
cd ~/nvidia-labs/ttg-agentic-genai-lab
python3 -m venv ~/.venvs/ttg && . ~/.venvs/ttg/bin/activate && pip install -r requirements-dev.txt
python3 -m archmath.archmath                                    # every bundled architecture in one table
python3 -m archmath.archmath --roofline --gpu H100-SXM --batch 1,8,32 --ctx 4096
python3 -m archmath.archmath --roofline --model llama-3.1-70b --precision fp8 --gpus 2 --batch 1
python3 -m archmath.archmath --coverage --batch 1,8,32,64,256   # MoE: unique experts touched per decode step
python3 -m archmath.archmath --prefill 3000 --gpu H100-SXM      # TTFT floor for a 3,000-token prompt
python3 -m archmath.archmath --notes                            # every assumption behind the numbers
deactivate && cd ~/nvidia-labs/plv-multicloud-nim-lab && . ~/.venvs/plv/bin/activate
python3 selector/instance_selector.py --spot                          # Llama 3.1 8B, 600 in / 200 out, ITL 40 ms
python3 selector/instance_selector.py --spot --nvaie 1.0              # + NVIDIA AI Enterprise via marketplace
python3 selector/instance_selector.py --spot --isl 3000 --osl 400 --itl-slo-ms 30
python3 selector/fetch_prices.py                                      # what moved since 2026-10-15
```
**Checkpoint:** without notes, you can answer the four checkpoint questions in Lab 04 M2:
- Why is the L4 vetoed in BF16?
- Why does the A100's batch stop at 369?
- Why can the NVAIE license cost more than a Spot GPU?
- Why is the A100 PCIe not NIM-verified for Llama 3.1 8B?

**Knowledge session:** `--phase P1` (the biggest bank, with three advanced numeric questions).
**Say it:** "Active parameters set the FLOPs, but the bytes you stream per token set the latency, so I size from
bandwidth first and measure second."

---

## P2 - Customer discovery + PoC charter (45-60 min, $0)
**Why:** "custom proofs of concept" and "partner with sales to identify and close opportunities."

**Learn**
- *Basic:* discovery is about traffic shape (input/output tokens, peak concurrency) and the SLO before any hardware
  talk. Read `plv-multicloud-nim-lab/docs/01-customer-brief.md`.
- *Intermediate:* a charter lists criteria, targets, the tool that measures each, owners, dates, data rules and the
  exit decision. Read `sales/poc-charter.md`.
- *Advanced:* regulated data. Prototype on hosted endpoints with synthetic data; the real PoC is self-hosted NIM in
  the customer's tenant; NVAIE goes through the marketplace so it draws down their commit. Read `docs/06-managed-paths.md`.

**Do:** role-play `sales/discovery-questions.md` (the coach can play the VP Eng), then fill `sales/poc-charter.md`.
Every criterion must name the tool in the repo that measures it.
**Checkpoint:** seven criteria, each with a target and a tool.
**Knowledge session:** `--phase P2`
**Say it:** "We'll prove one product on Azure, Google Cloud and a disconnected hospital in four weeks, with seven
measurable criteria, and you'll get a measured cost per thousand summaries per cloud, not a quote."

---

## P3 - Multi-cloud GPU Kubernetes platform: IaC + GitOps (2-3 h, cloud)
**Why:** "deploying AI/ML in AWS, GCP, Azure or OCI" and "containers, Kubernetes, data center deployments."

**Learn**
- *Basic:* GPU node pools are tainted, Spot, and scale 0..N. The GPU Operator owns the driver, container toolkit,
  device plugin, DCGM exporter and MIG manager.
- *Intermediate:* GPU plumbing differs per cloud. On AKS, `gpu_driver = "None"` lets the operator own the driver; GKE
  uses Google's installer. Secrets come through External Secrets with workload identity. Read
  `docs/07-csp-cheatsheet.md`, the whole table plus "Gotchas we hit".
- *Advanced:* pull-based GitOps, app-of-apps, sync waves (driver, then operators, secrets, observability, product)
  and self-heal. The CUDA probe measures real bandwidth (~80-90% of datasheet); you feed that ratio back into the
  selector's `BW_EFF`.

**Do** (Lab 04 M3, verbatim):
```bash
cd ~/nvidia-labs/plv-multicloud-nim-lab && . ~/.venvs/plv/bin/activate && source env/lab.env
infra/tf.sh aks up && infra/tf.sh gke up            # clusters, GPU pools (tainted, Spot, scale 0..N), vaults, registries
scripts/put-secrets.sh aks && scripts/put-secrets.sh gke                 # NGC key + HF token into Key Vault / Secret Manager
scripts/render-cluster-config.sh aks && scripts/render-cluster-config.sh gke   # ClusterSecretStore + values from TF outputs
git add clusters/*/generated && git commit -m "render cluster config" && git push
platform/install-argocd.sh aks && platform/install-argocd.sh gke     # Argo CD + root app: the cluster pulls the rest
kubectl --context plv-aks -n argocd get applications -w               # waves: driver -> operators -> ESO -> observability
```
CUDA bandwidth probe, one manifest for both GPUs:
```bash
for c in aks gke; do
  kubectl --context plv-$c create ns plv-bench --dry-run=client -o yaml | kubectl --context plv-$c apply -f -
  kubectl --context plv-$c -n plv-bench create configmap bw-probe --from-file=bench/physics/bandwidth_probe.cu
  scripts/render.sh bench/physics/bandwidth-job.yaml | kubectl --context plv-$c apply -f -
done
kubectl --context plv-aks -n plv-bench logs -f job/bw-probe ; kubectl --context plv-gke -n plv-bench logs -f job/bw-probe
```
**Checkpoint:** all Argo CD apps Synced/Healthy on both clusters. Measured bandwidth ratio written down (A100 PCIe
1,935 GB/s and RTX PRO 6000 1,597 GB/s on the datasheets).
**Knowledge session:** `--phase P3`. **Pause tonight:** `infra/tf.sh aks pause && infra/tf.sh gke pause`

---

## P4 - Ship NIM on two clouds with one chart (90 min, cloud)
**Why:** "build and deploy solutions at scale using NVIDIA's AI software on cloud-based GPU platforms."

**Learn**
- *Basic:* a NIM is a packaged inference microservice with an OpenAI-compatible API. The NIM Operator runs it through
  NIMCache (download once) and NIMService (serve).
- *Intermediate:* profiles. A profile is engine + precision + TP + LoRA, built for one GPU type and GPU count, so pin
  what `list-model-profiles` shows. Also `groupID: 0` (CA cert updates) and why a startupProbe matters.
- *Advanced:* the porting cost is one values file per cloud. Conformance tests prove the API contract (guided JSON,
  usage in streams, clean 4xx). Read `docs/05-interview-drill.md`, the "Multi-cloud deployment" paragraph.

**Do** (Lab 04 M4). The ARM laptop builds the amd64 image through CI; see WINDOWS-SETUP.md section 5:
```bash
git tag v1.0.0 && git push origin v1.0.0                   # release.yml builds ghcr.io/<you>/plover-api:1.0.0 on amd64
# then on GitHub: Packages -> plover-api -> Package settings -> Change visibility -> Public (the clusters pull it anonymously)
kubectl --context plv-aks -n plv-serving get nimcache,nimservice,pods -w        # AKS: NIMCache downloads once (RWX)
kubectl --context plv-aks -n plv-serving exec deploy/plover-llm -- list-model-profiles 2>/dev/null \
  || kubectl --context plv-aks -n plv-serving exec $(kubectl --context plv-aks -n plv-serving get pod -l app.kubernetes.io/name=plover-llm -o name | head -1) -- list-model-profiles
$EDITOR clusters/aks/values.yaml clusters/gke/values.yaml   # llm.profile = the compatible LoRA profile on THAT GPU
git commit -am "pin NIM profiles per GPU" && git push
kubectl --context plv-aks -n plv-serving port-forward svc/plover-llm 8000 &
kubectl --context plv-aks -n plv-serving port-forward svc/plover-api 8080 &
python3 -m pytest -q tests/conformance --endpoint http://localhost:8000          # the OpenAI contract, live
curl -s localhost:8080/v1/prior-auth/summarize -H 'content-type: application/json' \
  -d "{\"case_text\": $(python3 -c 'import sys,json; sys.path.insert(0,"app"); from plover_api import synth; print(json.dumps(synth.make_case(1)["text"]))')}" | jq .
git diff --no-index clusters/aks/values.yaml clusters/gke/values.yaml            # THE porting cost, on one screen
```
Repeat the profile, port-forward and conformance steps with `--context plv-gke`.
**Checkpoint:** both clusters return `"schema_valid": true`; Grafana shows the Plover dashboard.
**Knowledge session:** `--phase P4`
**Say it:** "Same chart, same GitOps, one 20-line values file per cloud. The GPU plumbing differs; the product
doesn't."

---

## P5 - Benchmark + price-performance per cloud (90 min, cloud)
**Why:** PoCs are won on measured numbers: "build PoCs that address critical business needs."

**Learn**
- *Basic:* TTFT, ITL, throughput, goodput (requests per second that met every SLO).
- *Intermediate:* the operating point is the highest load whose p95 still meets the SLO. Cost per 1,000 =
  $/hour / (SLO-compliant requests per second x 3,600) x 1,000, plus NVAIE if licensed hourly.
- *Advanced:* reading the knee. Flat throughput, flat ITL and exploding TTFT mean queueing. Energy uses
  `max by (UUID)` under MIG. Measured vs predicted gaps: bandwidth efficiency, batch limits, CPU overhead.

**Do** (Lab 04 M5):
```bash
kubectl --context plv-aks -n monitoring port-forward svc/kps-prometheus 9090:9090 &      # energy + $ columns
kubectl --context plv-gke -n monitoring port-forward svc/kps-prometheus 9091:9090 &
PROM_URL=http://localhost:9090 bench/run_sweep.sh aks synth-600-200 "1 8 32 64 128 256"
PROM_URL=http://localhost:9091 bench/run_sweep.sh gke synth-600-200 "1 8 32 64 128 256"
bench/run_sweep.sh aks llm-long "1 4 16 32"                    # the real product shape (style guide prompts)
python3 bench/compare_clouds.py results/*-synth-600-200-* --itl-slo-ms 40 --precision bf16
python3 bench/cost_model.py --node-hourly 0.679 --nvaie-hourly 1.0 --req-per-s <from llm-long> --isl 1400 --osl 300 \
    --duty-cycle 0.5 --api-in-per-1m <your API price> --api-out-per-1m <...>
```
Arm the perf gate: `bench/baselines/README.md`.
**Checkpoint:** a table of measured tok/s, ITL p95 and $/1M tokens per cloud next to the selector's prediction, plus
one sentence on the biggest gap.
**Knowledge session:** `--phase P5`. Log your numbers in `PROGRESS.md`; the interview answers use them.
**Say it:** "I don't quote a benchmark. I measure your traffic and show cost per summary at your latency target."

---

## P6 - Engines + precision: TensorRT-LLM, Triton, NIM, Dynamo (2-3 h, cloud + reading)
**Why:** stand-out line: "hands-on experience with NVIDIA GPUs and SDKs (CUDA, Triton, TensorRT-LLM)."

**Learn**
- *Basic:* what vLLM, SGLang, TensorRT-LLM, NIM and Dynamo each are. Read `ttg-agentic-genai-lab/docs/07-engine-cheatsheet.md`,
  "What each one is".
- *Intermediate:* precision. FP8 (Hopper, Ada, Blackwell) halves weight bytes; NVFP4 is Blackwell only; an FP8 KV
  cache doubles KV capacity; re-run the quality gate after any change. TensorRT engines are per GPU, while ONNX is
  portable. Read the same doc's "Quantization paths".
- *Advanced:* Dynamo. KV-aware routing helps when shared prefixes don't fit each worker's cache. Disaggregated
  prefill/decode pays off only with a fast KV-transfer fabric (Lab 02 measured it slower without RDMA). Read
  `kha-genai-inference-lab/README.md` M10, the YAMLs in `kha-genai-inference-lab/dynamo/`, and
  `kha-genai-inference-lab/docs/05-interview-drill.md` Q5-Q7.

**Do** (Lab 04 M6):
1. NIM precisions on Blackwell (GKE): commit `llm.precision`/`llm.profile` for FP8, then NVFP4. Sweep each with
   `bench/run_sweep.sh gke synth-600-200 "1 8 32 64 128 256"`.
2. trtllm-serve vs NIM on the same GPU:
   ```bash
   engines/bakeoff.sh gke pause && engines/bakeoff.sh gke trtllm            # NVFP4 checkpoint on Blackwell
   PLV_URL=http://trtllm-serve.plv-serving.svc:8000 PLV_MODEL=nvidia/Llama-3.1-8B-Instruct-NVFP4 \
     bench/run_sweep.sh gke synth-600-200 "1 8 32 64 128 256"
   kubectl --context plv-gke -n plv-serving port-forward svc/trtllm-serve 8002:8000 &
   python3 -m pytest -q tests/conformance --endpoint http://localhost:8002 --model nvidia/Llama-3.1-8B-Instruct-NVFP4
   engines/bakeoff.sh gke trtllm-down
   ```
3. Triton + TensorRT, built where it is served: `ENGINE_MODE=build engines/bakeoff.sh gke triton`, then
   `engines/bakeoff.sh gke triton-perf`. Then drill 10 (`drills/README.md`, "Drill 10"): the A100 plan on the RTX PRO
   6000.
4. `engines/bakeoff.sh gke resume` and `engines/bakeoff.sh aks resume`.

**Checkpoint:** FP8 vs NVFP4 vs trtllm-serve numbers in one table; the drill 10 error message understood; the
conformance difference between NIM and raw trtllm-serve written down.
**Knowledge session:** `--phase P6`
**Say it:** "NIM is TensorRT-LLM- and vLLM-class engines chosen per GPU and kept current by NVIDIA; trtllm-serve is
the same engine family without the packaging. ONNX is portable, TensorRT engines are per GPU."

---

## P7 - Scale: autoscaling, cold start, MIG, parallelism, NVL72 (cloud 75 min + paper 2 h)
**Why:** "at scale" in the JD, plus the system-level GPU questions every SA loop asks.

**Learn**
- *Basic:* autoscale on queue depth, not GPU utilization. MIG makes up to 7 hardware partitions; time-slicing only
  interleaves.
- *Intermediate:* cold start = node + image pull + model load, so keep a warm replica, use a pre-filled cache and
  pre-scale. Read `kha-genai-inference-lab/docs/05-interview-drill.md` Q2 and Q8.
- *Advanced:* the bandwidth ladder (HBM 3,350, NVLink 4 450 per direction, IB NDR 50 GB/s). TP inside NVLink, PP
  across nodes (bubble (P-1)/(m+P-1)), DP on top, EP for MoE. NVL72 in 60 seconds. Read
  `ttg-agentic-genai-lab/docs/04-gpu-cluster-architecture.md` and `docs/05-parallelism-math.md`.

**Do, cloud** (Lab 04 M7; needs 48 vCPU of A100 quota, otherwise skip to paper):
```bash
cd ~/nvidia-labs/plv-multicloud-nim-lab
AZ_GPU_MAX_NODES=2 infra/tf.sh aks up                  # needs 48 vCPU A100 quota
yq -i '.llm.autoscaling.enabled = true | .llm.autoscaling.maxReplicas = 2' clusters/aks/values.yaml
git commit -am "AKS: HPA on queue depth (max 2)" && git push
bench/run_sweep.sh aks synth-600-200 "256 384"          # sustained queue -> HPA -> pending pod -> autoscaler adds a node
kubectl --context plv-aks -n plv-serving get hpa -w
python3 bench/coldstart_timeline.py --context plv-aks -n plv-serving --newest -l app.kubernetes.io/name=plover-llm
python3 bench/coldstart_timeline.py --context plv-gke -n plv-serving --newest -l app.kubernetes.io/name=plover-llm
```
Revert the HPA commit afterwards.

**Do, paper** (laptop, $0):
```bash
cd ~/nvidia-labs/ttg-agentic-genai-lab && . ~/.venvs/ttg/bin/activate
python3 -m parallel.parallelism_calc --links          # the link bandwidth and latency table
python3 -m parallel.parallelism_calc --examples       # five worked examples (training and inference)
python3 -m parallel.parallelism_calc --infer --model llama-3.1-70b --precision fp8 --tp 1,2,4,8 --batch 1,32 --link nvlink4
python3 -m parallel.parallelism_calc --train --model llama-3.1-8b --dp 8 --zero 3 --seq 4096 --mbs 2
```
**Checkpoint:** a cold-start table per cloud with a recommendation; the NVL72 answer said out loud in under 60 s
(model answer: `ttg-agentic-genai-lab/docs/10-interview-drill.md`, section A).
**Knowledge session:** `--phase P7`
**Say it:** "Tensor parallelism lives inside the NVLink domain and data parallelism crosses the network. The math
tells us where every byte goes before we rent the cluster."

---

## P8 - MLOps: LoRA as a product release (2.5 h, cloud; full path)
**Why:** "DevOps/MLOps" required; "full SDLC: testing" stand-out.

**Learn**
- *Basic:* LoRA = small low-rank weights on a frozen base; NIM serves several adapters on one model.
- *Intermediate:* release gate = candidate vs baseline with tolerances; re-baseline deliberately, never
  automatically.
- *Advanced:* the full release path: distilled dataset, PEFT on a MIG slice, MLflow, gate, ORAS artifact by digest,
  canary by Git commit, one-line rollback. Plus your Gemma story: measure models, not serving paths.

**Do:** Lab 04 README **M8**, run exactly as written there (MIG `all-3g.40gb`, dataset job, noisy neighbor, train,
fetch/push adapter, canary, eval gate, promote/rollback). Finish with `mlops/mig.sh aks all-disabled`.
**Checkpoint:** an adapter promoted or rolled back by the gate; prompt tokens per summary before and after.
**Knowledge session:** `--phase P8`
**Say it:** "The adapter is an OCI artifact pinned by digest, the release is a Git commit, and the gate decides.
Rollback is one line."

---

## P9 - Air-gapped data center (2.5 h, cloud VM; full path)
**Why:** "data center deployments" is required. Health systems ask for it.

**Learn**
- *Basic:* no egress means a local registry mirror, a pre-cached model and no NGC key in any pod.
- *Intermediate:* the image list is computed from rendered charts, including ClusterPolicy components; a tag bump is a
  change request (drill 8).
- *Advanced:* the evidence pack a hospital security team signs.

**Do:** Lab 04 README **M9**, exactly as written (`infra/tf.sh aks pause && infra/tf.sh dc up` ...
`datacenter/verify-airgap.sh`). Finish with `infra/tf.sh dc down` and `infra/tf.sh aks resume`.
**Checkpoint:** `helm test` passes inside the air gap; the evidence pack is complete.
**Knowledge session:** `--phase P9`

---

## P10 - Managed vs self-hosted + CSP marketplaces (60 min, $0-low)
**Why:** working with CSPs means knowing what each one sells for NVIDIA AI, and how it's bought.

**Learn**
- *Basic:* NVAIE is $1/GPU-hour on the AWS/Azure/GCP/OCI marketplaces, or $4,500/GPU/year list.
- *Intermediate:* break-even is ~51% utilization ($4,500 / 8,760 h). On Spot, the license can cost more than the GPU.
  Marketplace purchases draw down the cloud commit.
- *Advanced:* the recommendation: prototype managed, ship self-hosted where PHI, on-prem, adapter ownership or steady
  utilization matter. Each CSP's path is in `docs/06-managed-paths.md` and `docs/07-csp-cheatsheet.md`.

**Do** (Lab 04 M10):
```bash
cd ~/nvidia-labs/plv-multicloud-nim-lab && . ~/.venvs/plv/bin/activate
cp managed/endpoints.example.yaml managed/endpoints.yaml && $EDITOR managed/endpoints.yaml    # synthetic data only
python3 managed/compare.py --endpoints managed/endpoints.yaml --self-hosted-summaries-per-hour <from P5> --out results/managed.md
```
**Checkpoint:** `results/managed.md` scorecard with your P5 number.
**Knowledge session:** `--phase P10`
**Say it:** "Prototype on managed endpoints, ship self-hosted NIM where PHI, on-prem or adapter ownership matter, and
let each marketplace carry NVIDIA AI Enterprise so it draws down your cloud commit."

---

## P11 - Operate + troubleshoot with the customer (2 h, cloud)
**Why:** "regular technical customer meetings ... debugging sessions."

**Learn**
- *Basic:* the six diagnosis commands (`drills/README.md`, "How to diagnose"). Taints, Pending pods, CrashLoopBackOff.
- *Intermediate:* GitOps self-heal (change Git, not the cluster), quota-bound autoscaling, the weekly sync pack
  (`docs/08-customer-cadence.md`).
- *Advanced:* "GPU util 100% but throughput low" and "TTFT doubled after an upgrade". Profiling method: measure, name
  the bottleneck class, change one thing. Read `ttg-agentic-genai-lab/docs/08-profiling-playbook.md`.

**Do** (Lab 04 M11 + M12):
```bash
cd ~/nvidia-labs/plv-multicloud-nim-lab && . ~/.venvs/plv/bin/activate
python3 fleet/fleet_report.py --out results/fleet.md
python3 fleet/whats_new.py --json results/whats-new.json
python3 fleet/weekly_sync.py --fleet results/fleet.json --whats-new results/whats-new.json --out results/weekly-sync.md
drills/inject.sh aks 2          # break: the coach injects, you diagnose from symptoms only
drills/inject.sh aks reset 2    # repair
```
Run drills 1-7 (fast path) or 1-10 (full path), using the table in `drills/README.md`.
**Checkpoint:** each drill diagnosed in under 5 minutes with its one-liner said out loud.
**Knowledge session:** `--phase P11`

---

## P12 - Teach + win: workshop, readout, technical win plan (2 h, $0)
**Why:** "presentations and workshops on NVIDIA products" and "partner with sales to close."

**Learn**
- *Basic:* the five-part readout: situation, what we tested, results table, recommendation, risks and next steps.
- *Intermediate:* technical win plan vs mutual action plan.
- *Advanced:* a 5-minute readout to a VP Eng, CFO and CSP partners, with your measured numbers.

**Do:**
- Rehearse `workshop/agenda.md` with `workshop/plover-workshop.ipynb` (runs on the mock),
  `workshop/slides-outline.md` and `workshop/facilitator-guide.md`.
- Fill `docs/04-results-and-exec-readout.md` with your numbers and record a 5-minute readout.
- Update `sales/technical-win-plan.md` and `sales/mutual-action-plan.md`.

**Checkpoint:** a recorded readout under 5 minutes, every number from your own runs.
**Knowledge session:** `--phase P12`

---

## P13 - Interview loop (mock days)
- `python quiz/quiz.py --mock 12`: weighted toward advanced questions in phases you've started.
- Coach: `interview` (one question at a time, scored 1-5, top 3 fixes).
- Drill docs, in this order:
  1. `plv-multicloud-nim-lab/docs/05-interview-drill.md`: six themes and ten likely questions. This one is the
     closest to this role.
  2. `kha-genai-inference-lab/docs/05-interview-drill.md`: twelve questions.
  3. `ttg-agentic-genai-lab/docs/10-interview-drill.md`: only section A (NVL72; "main goal when optimizing"), the
     GPU cluster and parallelism questions, and the profiling questions.
- Your stories: Molina voice AI (scale, latency, capacity planning) and the Gemma evaluation confound (rigor).
- **Knowledge session:** `--phase P13` (system design and "why NVIDIA").

## Teardown (Day 11, and any time you stop for more than a day)
```bash
cd ~/nvidia-labs/plv-multicloud-nim-lab
infra/tf.sh dc down; infra/tf.sh gke down; infra/tf.sh aks down
```

## What's in this folder
```
README.md                this lab
WINDOWS-SETUP.md         WSL2 on Windows ARM64, tools, the amd64 image question
PROGRESS.md              your tracker: phases, numbers, quiz scores, cloud meter
quiz/quiz.py             knowledge sessions (stdlib Python; Windows or WSL)
quiz/bank.py             74 questions: basic / intermediate / advanced per phase, each with its source file
QUIZ-BANK.md             printable bank with answers (read after a session, not before)
CLAUDE.md                makes Claude Code act as the coach
coach/jr2027588-coach.agent.md   the coach; copy it to ~/nvidia-labs/.github/agents/ for Copilot Chat
```
