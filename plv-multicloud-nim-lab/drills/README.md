# M12 - Break-fix drills (multi-cloud)

Same rules as Labs 02 and 03: someone injects, you diagnose **from symptoms only**, fix, then say the one-liner to
the customer from memory. Drill 1 is a Lab 02 drill on purpose - if it takes you more than 3 minutes, re-read Lab 02.

```bash
drills/inject.sh aks 2          # break (pair up: one person injects, the other diagnoses)
drills/inject.sh aks reset 2    # repair (most resets = re-enable Argo CD auto-sync and let Git heal)
```
Drills that hot-patch live objects first pause auto-sync on BOTH `plv-root` and `plover` - with app-of-apps, the root
would otherwise self-heal the child's sync policy back on and quietly undo your break (try it once to see).

| # | Drill | Symptom | Root cause | Fix | Say it |
|---|---|---|---|---|---|
| 1 | 🔁 Missing GPU toleration | Pod `Pending`: `untolerated taint {nvidia.com/gpu}` | Every cloud's GPU pool is tainted (Terraform) | Add the toleration (the chart does) | "GPU pods must opt in to GPU nodes - on every cloud." |
| 2 | Incompatible NIM profile | NIM `CrashLoopBackOff`; log says the profile is not compatible / needs 2 GPUs | Someone pinned a TP2 profile on a 1-GPU node | `list-model-profiles` on THAT GPU; pin a compatible profile in Git | "NIM profiles are per GPU and per GPU count - we pin the one we measured." |
| 3 | Secret renamed in the cloud vault | `ExternalSecret` `SecretSyncedError`; new NIM pod `CreateContainerConfigError: secret "ngc-api-secret" not found` | Remote key renamed; synced copy deleted | Fix the remote key in Git (or the vault); force-sync | "Secrets live in your vault, not in Git - a rename there is a production change." |
| 4 | Canary routed to an adapter NIM does not serve | ~50% of summarize calls return 502; `plover_api_requests_total{variant="canary",outcome="upstream_error"}` climbs | `CANARY_MODEL` set by hand, adapter not in `llm.lora.adapters` | Roll back (weight 0) or load the adapter via Git | "A canary is only as safe as its release path - adapters and routing ship in one commit." |
| 5 | No headroom for a lost GPU node | NIM `Pending`: `node(s) were unschedulable`; autoscaler: max node group size reached | GPU node cordoned (maintenance / Spot reclaim) and `max_count=1` | Uncordon or raise max nodes + quota | "N+1 GPU capacity is a cost decision - make it explicitly, per environment." |
| 6 | Manual change vs GitOps | You scale the NIM to 0; minutes later it is back | Argo CD `selfHeal` reverts drift from Git | Change Git, not the cluster | "Git is the control plane; kubectl is read-only in production." |
| 7 | Scale-out blocked | 2nd NIM replica `Pending`; `Insufficient nvidia.com/gpu`; autoscaler cannot add nodes | Replicas > GPU capacity (pool max, regional quota) | Quota + `gpu_max_nodes` in Terraform, then replicas/HPA in Git | "Autoscaling stops at your quota - we request quota before the launch, not during it." |
| 8 | Air gap: image not in the mirror (rke2) | `ErrImagePull ... i/o timeout` for `llama-3.1-8b-instruct:2.0.14` | Tag bumped without re-running the mirror | Re-stage (`datacenter/stage-images.sh` from a connected side) or roll back | "In a disconnected DC the image list IS the change request." |
| 9 | NetworkPolicy blocks MLOps (aks) | Dataset/eval jobs time out calling the NIM; probe `curl` times out | Ingress rule for `plv-mlops` removed | Restore `networkPolicy.allowNamespaces` in Git | "Default-deny is working; access is granted per namespace, in Git." |
| 10 | TensorRT plan from another GPU | Triton log: engine incompatible with this device / compute capability; model `UNAVAILABLE` | A100-built plan copied to the RTX PRO 6000 | Build per GPU (`ENGINE_MODE=build`) or `build-compat` (ampere+) and measure the cost | "ONNX is portable; TensorRT engines are built for the GPU that runs them." |

## Drill 10 (manual, M6 portability)

```bash
engines/bakeoff.sh aks pause && ENGINE_MODE=build engines/bakeoff.sh aks triton        # A100 (sm_80) plan
POD=$(kubectl --context plv-aks -n plv-serving get pod -l app=triage-triton -o name | head -1)
kubectl --context plv-aks -n plv-serving cp ${POD#pod/}:/work/triage/model_repository/triage_trt/1/model.plan results/a100.plan -c triton
engines/bakeoff.sh gke pause && ENGINE_MODE=import engines/bakeoff.sh gke triton         # RTX PRO 6000 (sm_120) waits for a plan
GPOD=$(kubectl --context plv-gke -n plv-serving get pod -l app=triage-triton -o name | head -1)
kubectl --context plv-gke -n plv-serving cp results/a100.plan ${GPOD#pod/}:/work/triage/import/model.plan -c build-engine
kubectl --context plv-gke -n plv-serving logs ${GPOD#pod/} -c triton | grep -i -E "engine|compat|error"
# then rebuild with ENGINE_MODE=build-compat on AKS, copy again, and compare trtexec/perf_analyzer numbers
engines/bakeoff.sh aks resume && engines/bakeoff.sh gke resume
```

## How to diagnose (the same 6 commands on every cloud)

```bash
kubectl --context <ctx> -n plv-serving get nimservice,nimcache,externalsecret,pods
kubectl --context <ctx> -n plv-serving describe pod -l app.kubernetes.io/name=plover-llm | tail -30
kubectl --context <ctx> -n plv-serving logs -l app.kubernetes.io/name=plover-llm --tail=80
kubectl --context <ctx> get events -A --sort-by=.lastTimestamp | tail -30
kubectl --context <ctx> -n argocd get applications
python3 fleet/fleet_report.py     # the fleet-wide view you show the customer
```
