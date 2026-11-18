# M9 Ship it: Docker, Helm, GitLab CI gates, Kubernetes (2 h)

You ship the Disruption Concierge the way a customer would run it in production: one container image, one Helm chart,
and a GitLab pipeline in which a release has to pass an **eval gate** (is it still policy-compliant?) and then a
**perf gate** (did p95 or $ per 1,000 disruptions get worse?). You rehearse it on a laptop for $0 with kind and the CPU
mock engine. Then you install the same chart on 🔁 Lab 04's AKS or GKE GPU cluster and run the storm-day load test.

```
              GitLab CI: lint -> test -> eval-gate (mock, every push) -> perf-gate (GPU runner) -> build -> deploy (manual)
                                                                                                        |
  namespace ttg                                                                                         v helm upgrade
  +-----------------------------------------------------------------------------------------------------------------+
  |  ttg-agent (Deployment + HPA on CPU)  --OpenAI API-->  ttg-engine (vllm | sglang | trtllm | nim | mock)           |
  |   FastAPI + LangGraph, /v1/disruptions     :8000         Recreate, GPU limit, /dev/shm, startupProbe, model cache |
  |        |  MCP streamable HTTP                            HPA or KEDA on queue depth (vllm:num_requests_waiting)   |
  |        +--> ttg-mcp-ops   :8101  (flights, rebooking, hotels)   no egress at all                                   |
  |        +--> ttg-mcp-guest :8102  (loyalty, policy, vouchers, receipts -> VLM in M8)                                |
  |  NetworkPolicy: agent egress = engine + 2 MCP servers + DNS. Nothing else, so a fooled model cannot exfiltrate.  |
  |  ServiceMonitor: engine /metrics + agent /metrics (ttg_task_latency_seconds, ttg_task_success_total)            |
  +-----------------------------------------------------------------------------------------------------------------+
```

| File | What it is |
|---|---|
| `deploy/Dockerfile` | ONE CPU image for four roles: agent API, MCP ops, MCP guest, mock engine. Runs as uid 10001 and works with a read-only root filesystem. The synthetic receipts (M8) are baked in. |
| `deploy/requirements-runtime.txt` | Only what `ttg/` imports, pinned to `env/pins.env` |
| `deploy/helm/ttg/` | The chart. `values.schema.json` rejects `latest` tags, secrets in `extraEnv`, and unknown variants |
| `deploy/values/kind.yaml` | Laptop: mock engine, no keys, no GPU |
| `deploy/values/aks.yaml` | 🔁 Lab 04 AKS: A100 80GB Spot, Premium SSD model cache, External Secrets from Key Vault |
| `deploy/values/gke.yaml` | 🔁 Lab 04 GKE: RTX PRO 6000 Blackwell (G4) Spot, local-SSD emptyDir cache (ADR-05) |
| `deploy/values/vlm-addon.yaml` | M8 in the cluster: a second release, engine only, that serves the receipt VLM |
| `.gitlab-ci.yml` | Primary CI (JR2019035 names GitLab). `.github/workflows/ci.yml` mirrors it |
| `scripts/gate.py` | The eval and perf gates, plus `baseline` (see `bench/baselines/README.md`) |
| `scripts/validate-manifests.sh` | Renders every values file and every engine kind x variant, then runs `kubeconform -strict` |

## 1. Build the image (10 min)
```bash
scripts/preflight.sh                                   # tools for M9: docker, kind, kubectl, helm, kubeconform
docker build -f deploy/Dockerfile -t ttg-agent:dev .
docker run --rm --user 10001 --read-only ttg-agent:dev python -c "import ttg.agent.api, ttg.mcp.ops_server; print('ok')"
```
There is one image and the chart picks the role through the command: `uvicorn ttg.agent.api:app`,
`python -m ttg.mcp.{ops,guest}_server --transport streamable-http`, or `uvicorn ttg.mock.fake_llm:app`. The GPU work
lives in the engine images (vLLM, SGLang, TensorRT-LLM, NIM), so this image has no torch and no CUDA.

**Checkpoint:** the image starts as a non-root user with a read-only root filesystem.

## 2. Laptop rehearsal on kind ($0, 20 min)
```bash
. env/pins.env
kind create cluster --name ttg --image "$PIN_KIND_NODE_IMAGE"         # kind >= $PIN_KIND
kind load docker-image ttg-agent:dev --name ttg
helm upgrade --install ttg deploy/helm/ttg -n ttg --create-namespace -f deploy/values/kind.yaml --wait
helm test ttg -n ttg                                                  # posts S0001 through the agent, expects success
kubectl -n ttg port-forward svc/ttg-agent 8080:8080 &
curl -s localhost:8080/v1/disruptions -H 'content-type: application/json' -d '{"scenario_id": "S0003"}' | python3 -m json.tool
curl -s localhost:8080/metrics | grep ttg_task
```
**Gotcha worth a story: `421 Invalid Host header`.** FastMCP turns on DNS-rebinding protection by default and only
accepts `localhost` as the Host header. On the laptop and in CI the agent connects to `localhost:8101`, so all is fine.
In the cluster it connects to `ttg-mcp-ops:8101` and every tool call fails. The chart therefore passes
`TTG_MCP_ALLOWED_HOSTS` with the Service names, and `ttg/mcp/_common.py` turns it into FastMCP
`TransportSecuritySettings`, so the protection stays on. `tests/test_deploy_mcp_host.py` replays the exact Host header
the agent sends.

**Checkpoint:** `helm test` reports `Succeeded` and the curl call returns `"success": true` with several `llm_calls`.

## 3. NetworkPolicy drill (🔁 Lab 03, 10 min)
kind's default CNI (kindnet, through kube-network-policies) enforces NetworkPolicy, so this drill runs on the laptop.
```bash
# the agent reaches the engine ...
kubectl -n ttg exec deploy/ttg-agent -- python -c "import httpx; print(httpx.get('http://ttg-engine:8000/v1/models').status_code)"
# ... but not the internet (expect ConnectTimeout / ConnectError)
kubectl -n ttg exec deploy/ttg-agent -- python -c "import httpx; httpx.get('https://example.com', timeout=5)"
# a pod that is not the agent cannot call the MCP tools (expect a timeout)
kubectl -n ttg run probe --rm -i --restart=Never --image=ttg-agent:dev --image-pull-policy=Never -- \
  python -c "import httpx; httpx.post('http://ttg-mcp-ops:8101/mcp', json={}, timeout=5)"
```
**Checkpoint:** you can explain each of the four policies: engine ingress, the two MCP servers (ingress from the agent
only, `egress: []`), and the agent's egress allow-list. You can also say why port 443 shows up in the agent's egress only
on the hosted-endpoint path (`engine.enabled=false`).

## 4. The same chart on a GPU cluster (🔁 Lab 04 AKS or GKE, 30 min + model download)
You need these from Lab 04: the cluster (GPU pool labelled `plv.io/pool=gpu`), the GPU Operator, External Secrets with
the `plv-cloud-secrets` ClusterSecretStore (keys `plv-hf-token`, `plv-ngc-api-key`), and kube-prometheus-stack in
`monitoring`.
```bash
scripts/check-images.sh                              # every pinned engine tag exists BEFORE you pay for a GPU
kubectl create namespace ttg
kubectl -n ttg create secret docker-registry gitlab-registry --docker-server=registry.gitlab.com \
  --docker-username="$DEPLOY_TOKEN_USER" --docker-password="$DEPLOY_TOKEN"      # GitLab deploy token: read_registry
helm upgrade --install ttg deploy/helm/ttg -n ttg -f deploy/values/aks.yaml \
  --set image.repository="$TTG_IMAGE" --set image.tag="$(git rev-parse --short HEAD)" --wait --timeout 45m
kubectl -n ttg logs -f deploy/ttg-engine             # first start downloads ~16 GB; startupProbe allows 30 min
helm test ttg -n ttg
```
The model cache is a PVC on AKS and a local-SSD emptyDir on GKE (Lab 04 ADR-05). The PVC is annotated
`helm.sh/resource-policy: keep`, so `helm uninstall` does not throw away 16 GB of weights.

**Switch engines with flags.** The variant names are the same as in `engines/launch.sh`, so the M4 numbers carry over
(`tests/test_deploy_chart.py` checks that the flag sets match):
```bash
helm upgrade ttg deploy/helm/ttg -n ttg --reuse-values --set engine.kind=sglang --set engine.variant=base
helm upgrade ttg deploy/helm/ttg -n ttg --reuse-values --set engine.kind=nim                     # NGC key + pull secret
helm upgrade ttg deploy/helm/ttg -n ttg --reuse-values --set engine.kind=vllm --set engine.variant=fp8kv
```
Combinations that would crash on the GPU fail at `helm template` time instead: `trtllm` + `spec`, `nim` + `fp8`, `lora`
without a PVC, and TensorRT-LLM with online FP8. The engine uses the `Recreate` strategy because one GPU cannot hold two
pods during a rolling update. A swap therefore takes the engine down for one restart. When the switch must be
zero-downtime, install a second engine release and move `agent.baseUrl`.

**M6 adapter (variant `lora`).** Copy `adapters/ttg-short` into a PVC, then switch to it:
```bash
kubectl -n ttg apply -f - <<'EOF'
apiVersion: v1
kind: PersistentVolumeClaim
metadata: {name: ttg-adapters}
spec: {accessModes: [ReadWriteOnce], resources: {requests: {storage: 1Gi}}}
---
apiVersion: v1
kind: Pod
metadata: {name: adapter-loader}
spec:
  containers: [{name: c, image: busybox:1.37, command: [sleep, "3600"], volumeMounts: [{name: a, mountPath: /adapters}]}]
  volumes: [{name: a, persistentVolumeClaim: {claimName: ttg-adapters}}]
EOF
kubectl -n ttg wait --for=condition=Ready pod/adapter-loader
kubectl -n ttg cp adapters/ttg-short adapter-loader:/adapters/ttg-short && kubectl -n ttg delete pod adapter-loader
helm upgrade ttg deploy/helm/ttg -n ttg --reuse-values --set engine.variant=lora --set engine.lora.pvcName=ttg-adapters
```
The agent switches by itself to the adapter's model name (`ttg-short` on vLLM, `ttg-llm:ttg-short` on SGLang) and to
the short prompt.

**M8 receipts through a VLM in the cluster.** Use `deploy/values/vlm-addon.yaml`; the commands are in its header. The
receipts are synthetic and rendered into the image at build time.

**Checkpoint:** `helm test` passes on the GPU cluster, and `kubectl -n ttg exec deploy/ttg-agent -- python -c
"import httpx; print(httpx.get('http://ttg-engine:8000/metrics').text[:400])"` shows prefix-cache counters.

## 5. Storm day: load test + autoscaling (30 min)
The customer's peak is about 3.7 tasks/s with an SLO of p95 <= 12 s ([docs/01](../docs/01-customer-brief.md)).
Little's law gives concurrency = throughput x latency. At about 8 s per task that is 3.7 x 8 ≈ 30 tasks in flight, so
test 16, 32 and 64.
```bash
# engine replicas on queue depth: Lab 04's prometheus-adapter already exposes vllm_num_requests_waiting
helm upgrade ttg deploy/helm/ttg -n ttg --reuse-values --set engine.autoscaling.enabled=true \
  --set engine.autoscaling.maxReplicas=2
kubectl -n ttg port-forward svc/ttg-agent 8080:8080 &
python -m ttg.bench.agent_bench --label storm-aks --engine vllm --api http://localhost:8080 --levels 16,32,64 --n 128
python scripts/gate.py eval results/bench/storm-aks.json --min-success 0.95      # still compliant under load?
python scripts/load_api.py --url http://localhost:8080 --concurrency 64 --n 256  # raw HTTP view: status codes
kubectl -n ttg get hpa -w
```
SGLang exposes its queue under another name. Add this rule to Lab 04's prometheus-adapter values, or use
`engine.autoscaling.mode=keda`, which queries Prometheus directly:
```yaml
- seriesQuery: 'sglang:num_queue_reqs{namespace!="",pod!=""}'
  resources: {overrides: {namespace: {resource: namespace}, pod: {resource: pod}}}
  name: {as: "sglang_num_queue_reqs"}
  metricsQuery: 'sum(<<.Series>>{<<.LabelMatchers>>}) by (<<.GroupBy>>)'
```
Expect the agent pods (CPU HPA) to scale in seconds. A new engine replica needs a free GPU and minutes of start-up, and
on Spot that GPU may not exist. Hence the 10-minute scale-down window. Before a forecast storm, raise
`engine.autoscaling.minReplicas`; do not wait for the queue. Grafana:
`histogram_quantile(0.95, sum(rate(ttg_task_latency_seconds_bucket[5m])) by (le))` next to `vllm:num_requests_waiting`.

**Checkpoint:** success >= 0.95 at c=64, and the p95 at c=32 goes into [docs/09](../docs/09-results-and-exec-readout.md)
next to the GPU count.

## 6. GitLab CI/CD (20 min)
| Stage | Job | Runs | Fails when |
|---|---|---|---|
| lint | shellcheck, ruff (syntax and undefined names), helm-lint | every push | a script or the chart is broken |
| test | pytest (JUnit), manifests (`kubeconform -strict`) | every push | any unit, chart, pin or CI test fails |
| eval-gate | mock engine + both MCP servers over HTTP, 40 golden-set tasks | every push | task success < 0.95 |
| perf-gate | `engines/launch.sh` on the GPU runner, bench at c = 1, 4, 16 | schedule, or manual on main | p95 or $/1k tasks > baseline + 10%, success < 0.95 |
| build | kaniko -> `$CI_REGISTRY_IMAGE/ttg-agent:<sha>` | main, tags | - |
| deploy | `helm upgrade --install` + `helm test`: staging (AKS) -> production (GKE) | manual | the helm test fails |

Setup, once:
- **GitLab agent for Kubernetes** in each cluster. Then set `KUBE_CONTEXT=<group>/<project>:<agent>`, and grant
  `ci_access` to this project in `.gitlab/agents/<agent>/config.yaml`.
- **GPU runner:** create a project runner with the tag `gpu` (Settings > CI/CD > Runners). On the GPU VM run
  `gitlab-runner register --url https://gitlab.com --token glrt-... --executor shell`, then
  `scripts/gpu-vm-bootstrap.sh`. The script installs uv system-wide, which the job uses to build its Python 3.12 venv,
  and adds `gitlab-runner` to the docker group, which `engines/launch.sh` needs. CI/CD variables: `GPU_HOURLY_USD`,
  `HF_TOKEN`, `NGC_API_KEY` (masked).
- **Baseline:** record the first one deliberately (`bench/baselines/README.md`). Until then the perf gate exits 2 and
  prints the command that records it.
- **Schedule:** run the perf gate nightly. An engine image bump goes through an MR, and the perf gate has to pass on it.

**Checkpoint:** you can say why the eval gate comes BEFORE the perf gate. An engine that is faster but wrong must never
reach the latency comparison.

## 7. When it breaks
| Symptom | Cause | Fix |
|---|---|---|
| Engine pod `Pending` | no GPU node, or a missing toleration | `kubectl describe pod`; check `engine.nodeSelector` and `tolerations`, and that the GPU Operator is ready |
| Engine restarts while downloading weights | startupProbe ran out | raise `engine.startupFailureThreshold` (x 10 s) |
| Agent: every tool call fails, MCP logs `421` | Host header not allowed | `TTG_MCP_ALLOWED_HOSTS` (section 2) |
| Agent `404 model not found` | `TTG_MODEL` is not a served name | leave `agent.model` empty: the chart derives it (LoRA names differ per engine) |
| NIM `401` on pull or download | NGC key or `ngc-secret` | `kubectl -n ttg get externalsecret`; Lab 04 Key Vault / Secret Manager entries |
| Engine HPA shows `<unknown>` | no prometheus-adapter rule | `kubectl get --raw /apis/custom.metrics.k8s.io/v1beta1 \| jq .` |
| `Bus error` / NCCL shm errors with tp2 | `/dev/shm` too small | `engine.shmSize` |

## 8. Clean up
```bash
helm uninstall ttg -n ttg && kubectl -n ttg delete pvc ttg-engine-cache   # the PVC is kept on purpose: delete it to stop paying
kind delete cluster --name ttg
```
Scale the GPU pool to 0 with Lab 04's Terraform. An idle GPU node left on overnight still bills about 12 GPU-hours.

**Say it:** "One image, one chart, two gates. The chart refuses configurations that would crash on a GPU, the
NetworkPolicy makes tools the only way out, and GitLab will not promote a build that is slower or less
policy-compliant than the committed baseline. On storm day the agent scales on CPU in seconds, and the GPUs scale on
queue depth, with warm capacity planned ahead because a GPU takes minutes to arrive."
