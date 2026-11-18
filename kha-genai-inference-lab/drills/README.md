# M11 - Break-fix drills (2 a.m. runbook practice)

Rules: inject, then diagnose from **symptoms only** (events, describe, logs, metrics) before reading the root cause.
Time yourself. Write the one-line "Say it" answer from memory afterwards. Reset each drill before the next.

```bash
drills/inject.sh 1          # break
drills/inject.sh reset 1    # repair
```

First-look commands for every drill:
```bash
kubectl get pods -A -o wide | grep -v -E 'Running|Completed'
kubectl describe pod <pod> -n <ns> | sed -n '/Events/,$p'
kubectl get node -l kha.io/pool=mig -o json | jq '.items[0].metadata.labels | with_entries(select(.key|test("nvidia.com/(mig|device-plugin|gpu.product)")))'
platform/set-gpu-mode.sh status
kubectl logs -n gpu-operator -l app=nvidia-mig-manager -c nvidia-mig-manager --tail=40
```

| # | Drill | Symptom you should find | Root cause | Fix | Say it |
|---|---|---|---|---|---|
| 1 | Whole-GPU request on a MIG node | Pod `Pending`, `Insufficient nvidia.com/gpu` | Mixed strategy advertises only `nvidia.com/mig-*` resources; a MIG-enabled GPU has no `nvidia.com/gpu` left | Request the slice resource (e.g. `nvidia.com/mig-1g.10gb`) or change the node's mode | "With MIG mixed strategy you schedule slices, not GPUs - the resource name is the contract." |
| 2 | Invalid MIG profile label | `nvidia.com/mig.config.state=failed`; MIG Manager log says the config is not found | Label names a config that is not in `<node>-mig-config` | Use a config from the ConfigMap (`kubectl get cm -n gpu-operator`), e.g. `all-balanced` | "MIG Manager is declarative: the node label selects a named geometry; invalid names fail closed." |
| 3 | Reconfigure a busy GPU | State stuck/failed; log mentions GPU clients or processes | MIG geometry cannot change while workloads hold the GPU | Drain GPU pods first (`platform/set-gpu-mode.sh` does cordon + evict + relabel + wait) | "Repartitioning is a maintenance operation: drain, relabel, wait for success, uncordon." |
| 4 | PB6 NIM with wrong groupID | `CrashLoopBackOff`; log shows `update-ca-certificates` permission denied | NIM LLM 2.x PB6 images need `groupID: 0` | Restore `groupID: 0` | "PB6 NIMs need group 0 at startup - it's in the release notes and it bites everyone once." |
| 5 | Missing GPU toleration | `Pending`, `untolerated taint {nvidia.com/gpu: present}` | GPU pools are tainted so only GPU workloads land there | Add the `nvidia.com/gpu` (and Spot) tolerations | "Taints keep CPU pods off $4/hour nodes; GPU pods must opt in with tolerations." |
| 6 | Bad NGC credentials | `ImagePullBackOff`, `401 Unauthorized` from nvcr.io | Pull secret has a wrong key (username must be literally `$oauthtoken`) | Recreate `ngc-secret` from Key Vault (`workloads/create-secrets.sh`) | "NGC pulls use `$oauthtoken` plus the API key; I keep the key in Key Vault and project it." |
| 7 | Context too long for the slice | shared-llm restarts; vLLM log: KV cache cannot hold `max_model_len` | 8B weights leave ~1-2 GiB for KV on a 20 GB slice; 131k tokens needs ~16 GiB | Restore `--max-model-len` to what the slice can hold (see sizing worksheet) | "Max context is a memory decision: KV bytes per token x max length must fit after weights." |
| 8 | HPA can't read its metric (after M9) | `kubectl get hpa` shows `<unknown>`; `FailedGetPodsMetric` | HPA asks for a metric prometheus-adapter does not expose | Use `vllm_num_requests_waiting` (or add an adapter rule) | "If the HPA shows unknown, check the custom metrics API before blaming the app." |
| 9 | Energy double counting (analysis) | `sum(POWER_USAGE)` is ~4x the real draw in MIG mode | Device-level DCGM fields repeat on every MIG instance series | `max by (UUID)` before summing; use the energy counter for reports | "Per-instance series repeat device power; aggregate by physical GPU UUID or you'll overstate energy." |
| 10 | Tenant over quota | `kubectl apply` fails: `exceeded quota: gpu-quota` | Namespace ResourceQuota caps `requests.nvidia.com/mig-3g.40gb` | Raise the quota deliberately or free the slice | "Quotas on extended resources turn MIG slices into chargeable, enforceable tenant budgets." |

Stretch drills (no script): evict the Spot GPU node (`az vmss simulate-eviction` on the node pool's VMSS) and time
recovery; delete the NIMCache PVC and watch a NIMService fail to start; stop DCGM exporter and see which alerts fire.
