#!/usr/bin/env bash
# Terraform wrapper for every cloud in Lab 04: apply/destroy, write env/outputs-<c>.env, create a named kube context.
# Usage: infra/tf.sh <aks|gke|eks|oke|dc> <up|down|output|kubeconfig|pause|resume|airgap|reconnect>
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck source=/dev/null
. env/lab.env
# shellcheck source=scripts/lib.sh
. scripts/lib.sh
C="${1:?usage: infra/tf.sh <aks|gke|eks|oke|dc> <action>}"
A="${2:?action: up|down|output|kubeconfig|pause|resume|airgap|reconnect}"

ssh_pub() { for f in "${HOME}/.ssh/id_ed25519.pub" "${HOME}/.ssh/id_rsa.pub"; do [ -f "$f" ] && { cat "$f"; return; }; done; die "no SSH public key (ssh-keygen -t ed25519)"; }

case "$C" in
  aks) D=infra/terraform/azure
       set -- -var "subscription_id=${AZ_SUBSCRIPTION_ID:?}" -var "location=${AZ_LOCATION}" -var "resource_group=${AZ_RG}" \
              -var "gpu_vm_size=${AZ_GPU_SKU}" -var "gpu_priority=${AZ_GPU_PRIORITY}" -var "gpu_max_nodes=${AZ_GPU_MAX_NODES}" ;;
  gke) D=infra/terraform/gcp
       set -- -var "project=${GCP_PROJECT:?}" -var "region=${GCP_REGION}" -var "zone=${GCP_ZONE}" \
              -var "gpu_profile=${GKE_GPU_PROFILE}" -var "gpu_spot=${GKE_GPU_SPOT}" -var "gpu_max_nodes=${GKE_GPU_MAX_NODES:-1}" ;;
  eks) D=infra/terraform/aws
       set -- -var "region=${AWS_REGION}" -var "gpu_instance_type=${EKS_GPU_INSTANCE}" ;;
  oke) D=infra/terraform/oci
       set -- -var "region=${OCI_REGION}" -var "tenancy_ocid=${OCI_TENANCY_OCID:?}" -var "compartment_ocid=${OCI_COMPARTMENT_OCID:?}" -var "gpu_shape=${OKE_GPU_SHAPE}" ;;
  dc)  D=infra/terraform/dc-sim
       AIR=false; if [ "$A" = "airgap" ]; then AIR=true; fi
       set -- -var "subscription_id=${AZ_SUBSCRIPTION_ID:?}" -var "location=${AZ_LOCATION}" -var "vm_size=${DC_VM_SKU}" -var "priority=${DC_VM_PRIORITY}" \
              -var "admin_ssh_public_key=$(ssh_pub)" -var "my_ip_cidr=$(curl -s https://api.ipify.org)/32" -var "airgapped=${AIR}" ;;
  *) die "unknown cloud $C" ;;
esac

out() { terraform -chdir="$D" output -raw "$1"; }

write_outputs() {
  f="env/outputs-${C}.env"
  case "$C" in
    aks) { echo "export AZ_AKS_NAME=\"$(out aks_name)\""; echo "export AZ_KV_NAME=\"$(out kv_name)\""; echo "export AZ_KV_URI=\"$(out kv_uri)\"";
           echo "export AZ_ESO_CLIENT_ID=\"$(out eso_client_id)\""; echo "export AZ_TENANT_ID=\"$(out tenant_id)\"";
           echo "export AZ_ACR_NAME=\"$(out acr_name)\""; echo "export AZ_ACR_LOGIN_SERVER=\"$(out acr_login_server)\""; } > "$f" ;;
    gke) { echo "export GKE_CLUSTER=\"$(out cluster_name)\""; echo "export GCP_AR_REPO=\"$(out ar_repo)\""; } > "$f" ;;
    eks) { echo "export EKS_CLUSTER=\"$(out cluster_name)\""; echo "export AWS_ECR_REPO=\"$(out ecr_repo)\""; } > "$f" ;;
    oke) { echo "export OKE_CLUSTER_ID=\"$(out cluster_id)\""; echo "export OCI_VAULT_OCID=\"$(out vault_ocid)\"";
           echo "export OCI_SECRET_OCID_PLV_NGC_API_KEY=\"$(out secret_ocid_ngc)\""; echo "export OCI_SECRET_OCID_PLV_HF_TOKEN=\"$(out secret_ocid_hf)\""; } > "$f" ;;
    dc)  { echo "export DC_IP=\"$(out public_ip)\""; } > "$f" ;;
  esac
  echo "wrote $f"
}

kubeconfig() {
  # shellcheck source=/dev/null
  . "env/outputs-${C}.env"
  case "$C" in
    aks) az aks get-credentials -g "${AZ_RG}" -n "${AZ_AKS_NAME}" --context "${KCTX_AKS}" --overwrite-existing ;;
    gke) gcloud container clusters get-credentials "${GKE_CLUSTER}" --zone "${GCP_ZONE}" --project "${GCP_PROJECT}"
         kubectl config delete-context "${KCTX_GKE}" >/dev/null 2>&1 || true
         kubectl config rename-context "gke_${GCP_PROJECT}_${GCP_ZONE}_${GKE_CLUSTER}" "${KCTX_GKE}" ;;
    eks) aws eks update-kubeconfig --name "${EKS_CLUSTER}" --region "${AWS_REGION}" --alias "${KCTX_EKS}" ;;
    oke) tmp="$(mktemp)"
         oci ce cluster create-kubeconfig --cluster-id "${OKE_CLUSTER_ID}" --file "$tmp" --region "${OCI_REGION}" --token-version 2.0.0 --kube-endpoint PUBLIC_ENDPOINT
         KUBECONFIG="$tmp" kubectl config rename-context "$(KUBECONFIG="$tmp" kubectl config current-context)" "${KCTX_OKE}"
         KUBECONFIG="${HOME}/.kube/config:$tmp" kubectl config view --flatten > "${tmp}.merged" && mv "${tmp}.merged" "${HOME}/.kube/config" && rm -f "$tmp"
         # OKE node pools have no taint field: taint GPU nodes so only GPU workloads land there (same contract as other clouds)
         kubectl --context "${KCTX_OKE}" taint nodes -l plv.io/pool=gpu nvidia.com/gpu=present:NoSchedule --overwrite ;;
    dc)  echo "RKE2 kubeconfig: run datacenter/bringup.sh on the VM, then datacenter/kubeconfig.sh here (SSH tunnel)." ;;
  esac
}

case "$A" in
  up|airgap|reconnect)
    terraform -chdir="$D" init -input=false
    terraform -chdir="$D" apply "$@"
    write_outputs
    [ "$A" = "up" ] && kubeconfig
    [ "$A" = "airgap" ] && echo "Outbound Internet is now DENIED for the DC VM (NSG rule deny-internet-egress)."
    true ;;
  down)   terraform -chdir="$D" destroy "$@" && rm -f "env/outputs-${C}.env" ;;
  output) terraform -chdir="$D" output ;;
  kubeconfig) kubeconfig ;;
  pause|resume)
    # shellcheck source=/dev/null
    . "env/outputs-${C}.env"
    n=0; [ "$A" = "resume" ] && n=1
    case "$C" in
      aks) if [ "$A" = "pause" ]; then az aks stop -g "${AZ_RG}" -n "${AZ_AKS_NAME}"; else az aks start -g "${AZ_RG}" -n "${AZ_AKS_NAME}"; kubeconfig; fi ;;
      gke) G="--cluster ${GKE_CLUSTER} --zone ${GCP_ZONE} --project ${GCP_PROJECT}"
           # the pool autoscales 0..N: a Pending NIM pod would bring a GPU node right back, so pause = autoscaler OFF + 0 nodes
           # shellcheck disable=SC2086
           if [ "$A" = "pause" ]; then gcloud container node-pools update gpu $G --no-enable-autoscaling --quiet
           else gcloud container node-pools update gpu $G --enable-autoscaling --min-nodes 0 --max-nodes "${GKE_GPU_MAX_NODES:-1}" --quiet; fi
           gcloud container clusters resize "${GKE_CLUSTER}" --node-pool gpu --num-nodes "$n" --zone "${GCP_ZONE}" --project "${GCP_PROJECT}" --quiet
           echo "GKE has no cluster stop: the GPU pool is at ${n}; the 2 system nodes keep billing (~\$0.27/h). Destroy when done." ;;
      eks) aws eks update-nodegroup-config --cluster-name "${EKS_CLUSTER}" --nodegroup-name gpu --scaling-config "minSize=0,maxSize=2,desiredSize=${n}" --region "${AWS_REGION}" >/dev/null
           echo "EKS GPU node group desired=${n}; the control plane (\$0.10/h) and system nodes keep billing." ;;
      oke) pool="$(oci ce node-pool list --compartment-id "${OCI_COMPARTMENT_OCID}" --cluster-id "${OKE_CLUSTER_ID}" --name gpu --query 'data[0].id' --raw-output)"
           oci ce node-pool update --node-pool-id "$pool" --size "$n" --force >/dev/null; echo "OKE GPU pool size=${n}" ;;
      dc)  if [ "$A" = "pause" ]; then az vm deallocate -g rg-plv-dc -n plv-dc-node; else az vm start -g rg-plv-dc -n plv-dc-node; fi ;;
    esac ;;
  *) die "unknown action $A" ;;
esac
