# Lab 04 - One NIM product on every cloud (and an air-gapped data center)

**Plover Health AI** (fictional ISV) sells a prior-authorization summarizer to a payer on Azure, a health system on
Google Cloud, and a hospital that is disconnected from the Internet. In this lab you are the NVIDIA **Senior Solutions
Architect, AI Cloud Services** who makes that one product run on NVIDIA NIM everywhere - picked by physics, deployed by
GitOps, measured per cloud, improved with LoRA, operated with a weekly cadence - and who briefs the customer on a model
released today (**Nemotron 3.5 add-on**).

Built for **JR2027588** (Santa Clara / remote; applications open until at least **2026-10-18**). Same style as Labs 02
and 03 - and on purpose it **repeats them**: every 🔁 marks a Lab 02/03 concept getting its next rep
([docs/00-recall.md](docs/00-recall.md)).

## What you will build
- Terraform for **AKS (A100 80GB)** and **GKE (RTX PRO 6000 Blackwell)** - required - plus EKS, OKE and a data-center VM.
- **Pull-based GitOps** (Argo CD app-of-apps per cluster): GPU Operator, NIM Operator, External Secrets (cloud vaults),
  kube-prometheus-stack, the Plover chart. The only per-cloud difference is `clusters/<cloud>/values.yaml`.
- **Plover API** (FastAPI): schema + faithfulness validation, sticky canary routing to **LoRA adapters**, metrics.
- An **instance selector** (roofline physics + real prices for 4 clouds) and a **model sizer** that reads `config.json`.
- **AIPerf** price-performance per cloud, measured vs predicted; **NIM vs trtllm-serve vs Triton/TensorRT** bake-off.
- **LoRA MLOps**: distillation dataset, PEFT training on a MIG slice, MLflow, release gate, ORAS artifact, canary by commit.
- **Air-gapped RKE2**: registry mirror, image list computed from charts, model pre-cached, zero egress, no NGC key.
- **Managed vs self-hosted** scorecard (Foundry, Vertex/GKE Marketplace, Bedrock/SageMaker, OCI, build.nvidia.com).
- **Customer cadence**: fleet report, what's-new checker, weekly sync pack; 10 break-fix drills; a 2-hour workshop
  (executed notebook); PoC charter, technical win plan, mutual action plan; interview drill.

## JD coverage map
| JD line (JR2027588) | Module | Proof you can show |
|---|---|---|
| Work with CSPs to build and demonstrate solutions using NVIDIA technology | M3-M4, M10, [docs/07](docs/07-csp-cheatsheet.md) | same chart on AKS + GKE; CSP cheat sheet; managed-vs-self-hosted scorecard |
| Deploy NVIDIA AI software at scale on cloud GPU platforms | M3-M4, M7, M9 | GitOps fleet; HPA on queue depth; cold-start timelines per cloud; air gap |
| Build PoCs that address customers' critical business needs | M1, M5, M8, M13 | [PoC charter](sales/poc-charter.md); results table; exec readout |
| Support sales and developer relations in identifying and closing opportunities | M1, M13, [sales/](sales/) | discovery, technical win plan, mutual action plan |
| Regular technical meetings: updates, new features, troubleshooting | M11, M12, add-on | weekly sync pack; what's-new; drills; Nemotron 3.5 brief |
| Presentations and workshops on NVIDIA products | M13, [workshop/](workshop/) | agenda, slide outline, executed notebook, facilitator guide |
| Deploying AI/ML on AWS, GCP, Azure or OCI | M3 | Terraform for all four (AKS + GKE applied; EKS + OKE stretch) |
| DevOps/MLOps: containers, Kubernetes, data center deployments | M3, M8, M9 | Argo CD, MLflow + gate, RKE2 air gap |
| *Preferred:* CUDA, Triton, TensorRT-LLM | M3, M6 | CUDA bandwidth probe; Triton/TensorRT engine portability; trtllm-serve |
| *Preferred:* SDLC, system design and testing | [docs/02](docs/02-architecture-and-adrs.md), tests, CI | 10 ADRs; ~70 tests; CI workflows |
| *Preferred:* Python + GPU systems; cloud certifications | everywhere; docs/07 | the lab doubles as AZ-305 / GCP PCA practice |

## Time and cost
| Path | Modules | Hands-on time | Cloud cost (Spot, with pause discipline) |
|---|---|---|---|
| **Fast path** (interview-ready) | M0-M5, M11, M12 (drills 1-7), M13 | ~12 h | ~$40-50 |
| **Full path** | M0-M13 + add-on | ~32 h | ~$110-150 |
| Stretch | EKS and/or OKE in M3-M5 | +4 h each | +$10-20 each |

Running costs per hour (2026-10-15 list prices): **AKS ~$1.20/h** (2x D4ds_v5 $0.452 + A100 Spot $0.679 + ACR/Files/LB),
**GKE ~$1.95/h** (2x e2-standard-4 ~$0.27 + G4 Spot $1.629 + disks; one zonal cluster's fee is covered by GKE's free
tier), **DC VM ~$0.80/h** (A100 Spot + 512 GB disk). Pause everything between sessions:
`infra/tf.sh aks pause`, `infra/tf.sh gke pause`, `infra/tf.sh dc pause` - and `down` when finished.

## Repository map
```
infra/        Terraform: azure (AKS), gcp (GKE), aws (EKS), oci (OKE), dc-sim (VM)  +  tf.sh (up/down/pause/airgap)
platform/     GPU Operator values per cloud, kps, prometheus-adapter, ESO, Argo CD, templates -> clusters/<c>/generated
gitops/       Argo CD app-of-apps per cluster (sync waves: driver -> operators -> secrets -> observability -> product)
charts/plover Helm chart: NIMService/NIMCache or air-gap Deployment, LoRA init containers, API, ExternalSecrets, toolbox
clusters/     values per cluster - THE porting cost (aks, gke, eks, oke, rke2)
app/          Plover API (FastAPI) + synthetic case generator + output schema
selector/     instance_selector.py, model_sizer.py, fetch_prices.py, gpus/instances/models.yaml
bench/        🔁 AIPerf sweeps per cluster, energy + $, compare_clouds.py, cost model, perf gate, CUDA probe
engines/      M6: trtllm-serve, Triton + TensorRT triage engine (portability), bakeoff.sh (GitOps pause)
mlops/        M8: dataset, LoRA training, eval gate, ORAS push, promote (Git), MIG, k8s Jobs + MLflow
datacenter/   M9: bringup, registry mirror, image list, prefetch, air-gapped deploy, verification
managed/      M10: endpoints + scorecard (managed vs self-hosted)
fleet/        M11: fleet report, what's new, weekly sync, action log
drills/       M12: 10 break-fix drills
workshop/     M13: agenda, slides outline, facilitator guide, notebook (executed in CI on the mock)
sales/        discovery, PoC charter, technical win plan, mutual action plan
addon-nemotron35/  Nemotron 3.5 day-0 kit: claims check, sizing, deploy, eval, brief, workshop segment
tests/        unit, chart policy, pins, conformance + agentic suites (mock by default), fixtures (real configs)
docs/         00 recall · 01 customer · 02 architecture/ADRs · 03 instance selection · 04 results/readout ·
              05 interview drill · 06 managed paths · 07 CSP cheat sheet · 08 customer cadence
```

## Prerequisites
- Azure subscription with **A100 Spot quota** (NCADSA100v4 family, 24 vCPU; 48 for M7 scale-out) and a Google Cloud
  project with **G4 / RTX PRO 6000 Spot quota** in your zone. Optional: AWS (G and VT vCPUs), OCI (A10).
- NGC API key (NIM images), Hugging Face token with the **Llama 3.1 license accepted**, a GitHub account (fork + GHCR).
- Laptop (macOS bash 3.2 is fine): `az`, `gcloud` + `gke-gcloud-auth-plugin`, `kubectl`, `helm`, `terraform`, `jq`,
  `yq` (mikefarah v4), `envsubst` (`brew install gettext`), `oras`, `git`, Python 3.11+ with `pip install -r requirements-dev.txt`.

---

## M0 - Setup (60 min)
```bash
gh repo fork <this repo> --clone && cd plv-multicloud-nim-lab        # or create your own repo from the zip
cp env/lab.env.example env/lab.env && $EDITOR env/lab.env              # subscriptions, project, keys, GITHUB_OWNER
source env/lab.env
gitops/set-repo-url.sh && git add gitops && git commit -m "point GitOps at my fork" && git push
scripts/preflight.sh                                                   # tools, logins, GPU quota, NGC images, HF access
python3 -m pytest -q tests                                             # everything that needs no cloud: ~70 tests
```
**Checkpoint:** preflight passes; `scripts/check-images.sh` shows every pinned `nvcr.io` image as pullable with YOUR key.
Pins live in `env/pins.env` (one source of truth; CI fails if GitOps or the chart drift from it).

## M1 - Discovery and the PoC charter (45 min)
Read [docs/01-customer-brief.md](docs/01-customer-brief.md). Role-play [sales/discovery-questions.md](sales/discovery-questions.md)
(ask a friend to be the VP Eng), then fill [sales/poc-charter.md](sales/poc-charter.md) - every success criterion must
name the tool in this repo that will measure it.
**Say it:** "We'll prove one product on Azure, Google Cloud and a disconnected hospital in four weeks, with seven
measurable criteria - and you'll get a cost per thousand summaries per cloud, measured, not quoted."

## M2 - Instance selection on paper (45 min) 🔁 Lab 02 physics
```bash
python3 selector/instance_selector.py --spot                          # Llama 3.1 8B, 600 in / 200 out, ITL 40 ms
python3 selector/instance_selector.py --spot --nvaie 1.0              # + NVIDIA AI Enterprise via marketplace
python3 selector/instance_selector.py --spot --isl 3000 --osl 400 --itl-slo-ms 30
python3 selector/fetch_prices.py                                      # what moved since 2026-10-15
```
Work through [docs/03-instance-selection.md](docs/03-instance-selection.md). **Checkpoint questions:** Why is the L4
vetoed in BF16? Why does the A100's batch stop at 369 (hint: the compute roof)? Why can the NVAIE license cost more than a
Spot GPU? Why is the lab's A100 PCIe *not* NIM-verified for Llama 3.1 8B while A100 SXM4 is?

## M3 - Infrastructure as code + GitOps platform on two clouds (2 h)
```bash
infra/tf.sh aks up && infra/tf.sh gke up            # clusters, GPU pools (tainted, Spot, scale 0..N), vaults, registries
scripts/put-secrets.sh aks && scripts/put-secrets.sh gke                 # NGC key + HF token into Key Vault / Secret Manager
scripts/render-cluster-config.sh aks && scripts/render-cluster-config.sh gke   # ClusterSecretStore + values from TF outputs
git add clusters/*/generated && git commit -m "render cluster config" && git push
platform/install-argocd.sh aks && platform/install-argocd.sh gke     # Argo CD + root app: the cluster pulls the rest
kubectl --context plv-aks -n argocd get applications -w               # waves: driver -> operators -> ESO -> observability
```
Then 🔁 the CUDA bandwidth probe - ONE manifest for both GPUs (`-arch=native`):
```bash
for c in aks gke; do
  kubectl --context plv-$c create ns plv-bench --dry-run=client -o yaml | kubectl --context plv-$c apply -f -
  kubectl --context plv-$c -n plv-bench create configmap bw-probe --from-file=bench/physics/bandwidth_probe.cu
  scripts/render.sh bench/physics/bandwidth-job.yaml | kubectl --context plv-$c apply -f -
done
kubectl --context plv-aks -n plv-bench logs -f job/bw-probe ; kubectl --context plv-gke -n plv-bench logs -f job/bw-probe
```
Expect read bandwidth at ~80-90% of datasheet (A100 PCIe 1,935 GB/s; RTX PRO 6000 1,597 GB/s); put YOUR ratio into the
selector's `BW_EFF`. Compare the two clusters' GPU plumbing with [docs/07](docs/07-csp-cheatsheet.md): operator-owned
driver on AKS vs Google's installer on GKE. **Stretch:** `infra/tf.sh eks up` / `infra/tf.sh oke up`, same steps.

## M4 - Ship the product with the same chart (90 min) 🔁 Lab 02 NIM
Build and publish the API image once (public GHCR package - synthetic data, no secrets):
```bash
docker buildx build --platform linux/amd64 -f app/Dockerfile -t ghcr.io/${GITHUB_OWNER_LC}/plover-api:1.0.0 --push .   # or tag v1.0.0 -> release.yml
```
Argo CD already syncs `plover` (wave 0). Watch NIM come up, then pin the profile you actually run:
```bash
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
**Checkpoint:** both clusters return `"schema_valid": true`; Grafana (kps) shows the Plover dashboard. If tool calls come
back as text later (M8 agentic suite), add the tool-call parser flags via `llm.extraArgs` - a commit, not a hotfix.

## M5 - Price-performance per cloud (90 min) 🔁 Lab 02/03 AIPerf
```bash
kubectl --context plv-aks -n monitoring port-forward svc/kps-prometheus 9090:9090 &      # energy + $ columns
kubectl --context plv-gke -n monitoring port-forward svc/kps-prometheus 9091:9090 &
PROM_URL=http://localhost:9090 bench/run_sweep.sh aks synth-600-200 "1 8 32 64 128 256"
PROM_URL=http://localhost:9091 bench/run_sweep.sh gke synth-600-200 "1 8 32 64 128 256"
bench/run_sweep.sh aks llm-long "1 4 16 32"                    # the real product shape (style guide prompts)
python3 bench/compare_clouds.py results/*-synth-600-200-* --itl-slo-ms 40 --precision bf16   # both clusters serve BF16 until M6
python3 bench/cost_model.py --node-hourly 0.679 --nvaie-hourly 1.0 --req-per-s <from llm-long> --isl 1400 --osl 300 \
    --duty-cycle 0.5 --api-in-per-1m <your API price> --api-out-per-1m <...>
```
Arm the perf gate ([bench/baselines/README.md](bench/baselines/README.md)). **Checkpoint:** a table with measured
tok/s, ITL p95 and $/1M tokens per cloud next to the selector's prediction - and one sentence explaining the biggest gap.

## M6 - What engine is underneath? (2 h) 🔁 Lab 02 Triton/TensorRT
1. **NIM precisions on Blackwell (GKE):** commit `llm.precision`/`llm.profile` for FP8, then NVFP4; sweep each with
   `bench/run_sweep.sh gke synth-600-200 ...`. Same GPU, same chart - tokens/s and TTFT move with bytes per weight.
2. **trtllm-serve vs NIM** on the same GPU, same preset (GitOps paused so Argo CD does not restore the NIM mid-test):
   ```bash
   engines/bakeoff.sh gke pause && engines/bakeoff.sh gke trtllm            # NVFP4 checkpoint on Blackwell
   PLV_URL=http://trtllm-serve.plv-serving.svc:8000 PLV_MODEL=nvidia/Llama-3.1-8B-Instruct-NVFP4 \
     bench/run_sweep.sh gke synth-600-200 "1 8 32 64 128 256"
   kubectl --context plv-gke -n plv-serving port-forward svc/trtllm-serve 8002:8000 &
   python3 -m pytest -q tests/conformance --endpoint http://localhost:8002 --model nvidia/Llama-3.1-8B-Instruct-NVFP4
   engines/bakeoff.sh gke trtllm-down
   ```
   The conformance run is the point: which API promises (guided JSON, usage in streams, clean 4xx) does the raw engine
   keep that Plover relies on? Feature parity is part of "which engine", not just tokens/s.
3. **Triton + TensorRT, built where it is served:** `ENGINE_MODE=build engines/bakeoff.sh gke triton`, then
   `engines/bakeoff.sh gke triton-perf`; repeat on AKS. Then **drill 10** ([drills/README.md](drills/README.md)): copy
   the A100 plan to the RTX PRO 6000 and read the error; rebuild with `ENGINE_MODE=build-compat` (ampere+) and price it.
4. `engines/bakeoff.sh gke resume` and `engines/bakeoff.sh aks resume` - Argo CD restores the product from Git.

**Say it:** "NIM is TensorRT-LLM- and vLLM-class engines chosen per GPU and kept current by NVIDIA; trtllm-serve is
the same engine family without the packaging. ONNX is portable, TensorRT engines are per GPU."
(Triton's TensorRT-LLM backend image is not published for 26.09 at the time of writing - check NGC before planning on it.)

## M7 - Scale out, scale to zero, cold start (75 min) 🔁 Lab 02 HPA
```bash
AZ_GPU_MAX_NODES=2 infra/tf.sh aks up                  # needs 48 vCPU A100 quota
yq -i '.llm.autoscaling.enabled = true | .llm.autoscaling.maxReplicas = 2' clusters/aks/values.yaml
git commit -am "AKS: HPA on queue depth (max 2)" && git push
bench/run_sweep.sh aks synth-600-200 "256 384"          # sustained queue -> HPA -> pending pod -> autoscaler adds a node
kubectl --context plv-aks -n plv-serving get hpa -w
python3 bench/coldstart_timeline.py --context plv-aks -n plv-serving --newest -l app.kubernetes.io/name=plover-llm
python3 bench/coldstart_timeline.py --context plv-gke -n plv-serving --newest -l app.kubernetes.io/name=plover-llm
```
Compare: node provisioning (Spot), image pull (GKE image streaming), model load (AKS NIMCache RWX vs GKE emptyDir
re-download). **Checkpoint:** a cold-start table per cloud and a recommendation (keep 1 warm replica? Filestore? reservations?).
Revert the HPA commit afterwards (or keep `maxReplicas: 2` only where quota allows).

## M8 - LoRA as a product release (2.5 h) 🔁 Lab 02 MIG + Lab 03 gate
```bash
mlops/mig.sh aks all-3g.40gb                           # 2 x 3g.40gb: one serves, one trains
yq -i '.llm.gpuResource = "nvidia.com/mig-3g.40gb"' clusters/aks/values.yaml && git commit -am "AKS: NIM on a MIG slice" && git push
scripts/render.sh mlops/k8s/mlflow.yaml | kubectl --context plv-aks apply -f -
kubectl --context plv-aks apply -f mlops/k8s/hf-token-externalsecret.yaml
kubectl --context plv-aks -n plv-mlops create configmap plover-mlops-code --from-file=mlops/ \
  --from-file=app/plover_api/ --from-file=tests/mock_openai.py
kubectl --context plv-aks apply -f mlops/k8s/dataset-job.yaml        # teacher = today's product (long prompt)
kubectl --context plv-aks -n plv-mlops wait --for=condition=complete job/plover-dataset --timeout=60m
bench/noisy_neighbor.sh aks                            # experiment FIRST: a throwaway training run next to serving
ADAPTER_NAME=plover-style-v1 scripts/render.sh mlops/k8s/train-lora-job.yaml | kubectl --context plv-aks apply -f -
kubectl --context plv-aks -n plv-mlops wait --for=condition=complete job/plover-train --timeout=4h
mlops/fetch_adapter.sh aks plover-style-v1 && mlops/push_adapter.sh results/adapters/plover-style-v1 style-v1
mlops/promote.sh aks canary plover-style-v1 ghcr.io/${GITHUB_OWNER_LC}/plover-adapters@sha256:<digest> 10 && git push
kubectl --context plv-aks -n plv-serving rollout status deploy/plover-llm --timeout=30m   # NIM restarts with the adapter
kubectl --context plv-aks -n plv-serving port-forward svc/plover-llm 8000 &   # fresh port-forward: the pod was replaced
kubectl --context plv-aks -n plv-mlops port-forward svc/mlflow 5000 &
pip install "mlflow==${PIN_MLFLOW}" && export MLFLOW_TRACKING_URI=http://localhost:5000
python3 mlops/eval_gate.py --endpoint http://localhost:8000 --candidate plover-style-v1 \
  --mlflow-run-id "$(cat results/adapters/plover-style-v1/mlflow_run_id)"
mlops/promote.sh aks promote plover-style-v1 && git push          # or: mlops/promote.sh aks rollback
python3 -m pytest -q tests/agentic --endpoint http://localhost:8000 --scorecard results/agentic-llama8b.json
```
Watch Grafana: schema-valid ratio by variant and **prompt tokens per summary** (the LoRA payoff). Finish with
`mlops/mig.sh aks all-disabled` and revert `gpuResource`. **Say it:** "The adapter is an OCI artifact pinned by digest,
the release is a Git commit, the gate decides - rollback is one line."

## M9 - The hospital data center, air-gapped (2.5 h)
```bash
infra/tf.sh aks pause && infra/tf.sh dc up              # free A100 quota, then the DC-sim VM (RKE2 host)
rsync -az --exclude .git --exclude results ./ ${DC_USER}@${DC_IP}:plv/      # incl. env/lab.env (NGC key used in staging only)
ssh ${DC_USER}@${DC_IP} 'cd plv && sudo -E bash datacenter/bringup.sh'        # driver (reboot once), RKE2, helm
ssh ${DC_USER}@${DC_IP} 'cd plv && sudo -E bash datacenter/stage-images.sh && sudo -E bash datacenter/prefetch-model.sh'
ssh ${DC_USER}@${DC_IP} 'shred -u plv/env/lab.env'      # the key leaves the "data center" with the staging step
infra/tf.sh dc airgap                                   # NSG denies all Internet egress from now on
datacenter/kubeconfig.sh                                # then keep: ssh -N -L 16443:127.0.0.1:6443 ${DC_USER}@${DC_IP}
datacenter/deploy-airgapped.sh && datacenter/verify-airgap.sh
drills/inject.sh rke2 8 && drills/inject.sh rke2 reset 8
```
**Checkpoint:** helm test passes inside the air gap; the evidence pack (no egress, mirrored images only, no key in pods)
is what the hospital's security team signs. `infra/tf.sh dc down` and `infra/tf.sh aks resume` when done.

## M10 - Managed vs self-hosted (60 min)
Read [docs/06-managed-paths.md](docs/06-managed-paths.md) (what each CSP offers for NVIDIA AI today, with sources), then:
```bash
cp managed/endpoints.example.yaml managed/endpoints.yaml && $EDITOR managed/endpoints.yaml    # synthetic data only
python3 managed/compare.py --endpoints managed/endpoints.yaml --self-hosted-summaries-per-hour <M5> --out results/managed.md
```
**Say it:** "Prototype on managed endpoints, ship self-hosted NIM where PHI, on-prem or adapter ownership matter - and
let each marketplace carry NVIDIA AI Enterprise so it draws down your cloud commit."

## M11 - Operate the fleet with the customer (45 min)
```bash
python3 fleet/fleet_report.py --out results/fleet.md
python3 fleet/whats_new.py --json results/whats-new.json
python3 fleet/weekly_sync.py --fleet results/fleet.json --whats-new results/whats-new.json --out results/weekly-sync.md
```
Practice the upgrade flow in [docs/08](docs/08-customer-cadence.md): bump one pin in a branch, watch CI enforce it
(`tests/chart/test_pins.py`), perf-gate one cluster, merge, let Argo CD roll it out.

## M12 - Break-fix drills (90 min) 🔁 Labs 02/03
Pair up; `drills/inject.sh <cluster> <n>`, diagnose from symptoms, fix, say the one-liner. Table: [drills/README.md](drills/README.md).

## M13 - Workshop, readout, technical win (2 h)
- Rehearse [workshop/agenda.md](workshop/agenda.md) with the notebook (`workshop/plover-workshop.ipynb`, runs on the mock
  or `PLV_ENDPOINT`), [slides outline](workshop/slides-outline.md) and [facilitator guide](workshop/facilitator-guide.md).
- Fill [docs/04-results-and-exec-readout.md](docs/04-results-and-exec-readout.md) with YOUR numbers; record a 5-minute readout.
- Update [sales/technical-win-plan.md](sales/technical-win-plan.md) and the [mutual action plan](sales/mutual-action-plan.md).
- Interview drill: [docs/05-interview-drill.md](docs/05-interview-drill.md).

## Add-on - Nemotron 3.5 day-0 launch kit (4 h)
[addon-nemotron35/README.md](addon-nemotron35/README.md): claims check against the real `config.json` and quantization
configs, sizing hybrid Mamba-MoE models (SSM state is read and written every step), Lightning on AKS with the same chart,
the Plover gates (conformance, quality, agentic with thinking on/off, injection), AIPerf vs OpenRouter pricing, and a
customer brief + 10-minute workshop segment.

---

## Teardown
```bash
infra/tf.sh dc down; infra/tf.sh gke down; infra/tf.sh aks down     # (eks/oke if used)
```
GHCR packages and the GitHub fork stay (free). Delete cloud secrets if you created them outside Terraform.

## What was validated before you got this (sandbox, no cloud)
`terraform validate` for all five stacks; `helm lint` + `helm template` for every cluster and the add-on overlays;
kubeconform against the CRD schemas (NIMService/NIMCache, ExternalSecret/ClusterSecretStore, Argo CD Application,
ServiceMonitor/PrometheusRule); shellcheck on every script; the full pytest suite (unit, chart policy, pins, conformance,
agentic, MLOps, fleet, data center) against the mock, plus the same suites over real HTTP (mock NIM server and the
Plover API passthrough); drills, bake-off, MIG and add-on scripts dry-run against stubbed kubectl/helm; the workshop
notebook executed end-to-end; `fleet/whats_new.py` and `selector/fetch_prices.py` against live public sources; model
sizing against the real Nemotron 3.5 configs and checkpoint sizes from Hugging Face.
**Not run here:** anything that needs your cloud accounts, GPUs or NGC key - those are the modules.

## Known caveats (verify on the day)
- NIM LoRA profile IDs differ by GPU and version - always pin what `list-model-profiles` shows (M4).
- The A100 80GB PCIe is not on NIM 2.0.13's verified list for Llama 3.1 8B (fine for the PoC; production on a verified SKU).
- GKE G4 and Azure NCv6 capacity is regional; check quota first. NCv6 is SR-IOV vGPU (no power telemetry).
- Triton's TensorRT-LLM backend image is not available for 26.09 at the time of writing.
- EKS/OKE stacks are validated but not exercised end-to-end; OKE GPU notes say "verify".
- Nemotron 3.5 Super VL NIM tag and its verified GPUs: the chart refuses `VERIFY` placeholders until you check NGC.
