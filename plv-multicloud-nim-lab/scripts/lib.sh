# shellcheck shell=bash
# Shared helpers (bash 3.2 compatible). Source after env/lab.env.
die() { echo "ERROR: $*" >&2; exit 1; }

# ctx <cluster>  -> kubectl context name for aks|gke|eks|oke|rke2
ctx() {
  case "$1" in
    aks) echo "${KCTX_AKS}" ;; gke) echo "${KCTX_GKE}" ;; eks) echo "${KCTX_EKS}" ;;
    oke) echo "${KCTX_OKE}" ;; rke2) echo "${KCTX_RKE2}" ;;
    *) die "unknown cluster '$1' (aks|gke|eks|oke|rke2)" ;;
  esac
}

# k <cluster> <kubectl args...>  -> kubectl pinned to that cluster's context
k() { _c="$1"; shift; kubectl --context "$(ctx "${_c}")" "$@"; }

# h <cluster> <helm args...>
h() { _c="$1"; shift; helm --kube-context "$(ctx "${_c}")" "$@"; }

valid_cluster() { case "$1" in aks|gke|eks|oke|rke2) return 0 ;; *) return 1 ;; esac; }
