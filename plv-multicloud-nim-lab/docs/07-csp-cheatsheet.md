# CSP cheat sheet - what is actually different per platform

| | AKS (Azure) | GKE (Google) | EKS (AWS, stretch) | OKE (Oracle, stretch) | RKE2 (data center) |
|---|---|---|---|---|---|
| Lab GPU | NC24ads A100 v4 (A100 80GB PCIe) | g4-standard-48 (RTX PRO 6000) | g6e.xlarge (L40S) | VM.GPU.A10.1 (A10) | Spot A100 VM |
| Driver | GPU Operator (`gpu_driver = "None"` on the pool) | Google driver installer DaemonSet (pool: `INSTALLATION_DISABLED`) | AL2023 NVIDIA AMI | OKE GPU image | host LTSB driver (R580) |
| Toolkit | GPU Operator | GPU Operator (`installDir /home/kubernetes/bin/nvidia`, `RUNTIME_CONFIG_SOURCE=file`) | AMI | image | GPU Operator (`CONTAINERD_SOCKET` = RKE2's) |
| Device plugin | GPU Operator (MIG mixed) | GPU Operator (label `gke-no-default-nvidia-gpu-device-plugin=true`) | GPU Operator | OKE add-on (operator's disabled) | GPU Operator |
| GPU taint | pool taint `nvidia.com/gpu` (+ spot taint) | pool taint | node group taint | none in API -> `infra/tf.sh` taints | none (single node) |
| Precision for Llama 8B | BF16 (Ampere) | NVFP4 (Blackwell) | FP8 (Ada) | BF16 (Ampere, not verified) | BF16 |
| NIM cache | NIMCache on Azure Files Premium RWX | emptyDir (Filestore minimum too costly for a PoC) | emptyDir | emptyDir | hostPath, pre-filled |
| Secrets -> ESO auth | Key Vault, workload identity (UAMI + federated credential) | Secret Manager, WIF principal (no GSA) | Secrets Manager, EKS Pod Identity | OCI Vault, workload principal (enhanced cluster) | Kubernetes secrets only (offline) |
| NetworkPolicy | Cilium (enforced) | Dataplane V2 (enforced) | off until VPC CNI network policy (or a policy engine) is enabled | flannel: not enforced | Canal (enforced) |
| Default StorageClass | managed-csi | standard-rwo | check: may be none on new clusters | oci-bv | none (local-path optional) |
| Image registry | ACR (AcrPull for kubelet) | Artifact Registry | ECR | OCIR | local mirror via registries.yaml |
| Pause to save money | `infra/tf.sh aks pause` (stop cluster) | GPU pool to 0 | node group desired 0 | pool size 0 | `az vm deallocate` |
| Marketplace NVAIE | Azure Marketplace / Foundry surcharge | GCP Marketplace (private offers count to commit) | AWS Marketplace AMI $1/GPU-h | built into OCI Console | annual subscription |

**Gotchas we hit:** azurerm 5.x requires `node_provisioning_profile` on AKS; GKE >= 1.33 needs the toolkit to read
containerd config from file; GKE needs a ResourceQuota for GPU Operator pods in system priority classes; OKE has no
node-pool taint field; EKS AL2023 NVIDIA AMI already has the toolkit (never install a second one); Azure NCv6
RTX PRO 6000 is SR-IOV vGPU (GRID driver, no power telemetry).
