#!/usr/bin/env bash
# Laptop preflight for Lab 05 (macOS bash 3.2 or Linux): Python + packages, optional tools per module, keys.
# FAIL = M0 cannot run; warn = a later module needs it. Nothing here needs a GPU.
# Usage: scripts/preflight.sh
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck source=/dev/null
. env/pins.env
if [ -f env/lab.env ]; then
  # shellcheck source=/dev/null
  . env/lab.env
fi
FAILS=0
ok()   { printf '  [ok]   %s\n' "$*"; }
warn() { printf '  [warn] %s\n' "$*"; }
bad()  { printf '  [FAIL] %s\n' "$*"; FAILS=$((FAILS + 1)); }

echo "Python"
if command -v python3 >/dev/null 2>&1 && python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)'; then
  ok "python3 $(python3 -c 'import platform; print(platform.python_version())')"
else
  bad "python3 >= 3.11 required (brew install python@3.12 / apt install python3.12)"
fi
for mod in yaml httpx fastapi uvicorn openai langgraph langchain_mcp_adapters mcp pytest; do
  if python3 -c "import ${mod}" >/dev/null 2>&1; then ok "import ${mod}"; else bad "import ${mod} - run: pip install -r requirements-dev.txt"; fi
done
if python3 -c "import ttg" >/dev/null 2>&1; then ok "lab package ttg importable"; else warn "ttg not importable: run from the repo root"; fi

echo "Tools (by module)"
check_tool() {   # check_tool <binary> <why>
  if command -v "$1" >/dev/null 2>&1; then ok "$1 ($2)"; else warn "$1 missing ($2)"; fi
}
check_tool docker "M9 image build; GPU VM engines"
check_tool kind "M9 laptop cluster - kind ${PIN_KIND}+, kind create cluster --image ${PIN_KIND_NODE_IMAGE}"
check_tool kubectl "M9"
check_tool helm "M9 chart - pinned ${PIN_HELM}"
check_tool kubeconform "M9 scripts/validate-manifests.sh - pinned ${PIN_KUBECONFORM}"
check_tool shellcheck "CI lint"
check_tool jq "handy for JSON results"
if command -v helm >/dev/null 2>&1; then
  hv="$(helm version --short 2>/dev/null | cut -d+ -f1)"
  case "${hv}" in v3.*) ok "helm ${hv} is Helm 3" ;; *) warn "helm ${hv}: the lab pins Helm 3 (${PIN_HELM})" ;; esac
fi

echo "Keys (env/lab.env)"
if [ ! -f env/lab.env ]; then warn "env/lab.env missing: cp env/lab.env.example env/lab.env"; fi
if [ -n "${NVIDIA_API_KEY:-}" ]; then ok "NVIDIA_API_KEY (M3/M8 hosted endpoints - synthetic data only)"; else warn "NVIDIA_API_KEY empty (M3 hosted run, M8 hosted VLM)"; fi
if [ -n "${HF_TOKEN:-}" ]; then ok "HF_TOKEN (gated Llama 3.1 on the GPU VM)"; else warn "HF_TOKEN empty (M4-M7 on the GPU VM)"; fi
if [ -n "${NGC_API_KEY:-}" ]; then ok "NGC_API_KEY (NIM + nvcr.io)"; else warn "NGC_API_KEY empty (NIM in M4, scripts/check-images.sh)"; fi
case "${GPU_HOURLY_USD:-}" in
  ''|0|0.0) warn "GPU_HOURLY_USD not set: \$ per 1,000 disruptions will be blank" ;;
  *) ok "GPU_HOURLY_USD=${GPU_HOURLY_USD} for ${GPU_NAME:-your GPU} (check your provider's price before each session)" ;;
esac

echo "Pins"
ok "vLLM ${PIN_VLLM_TAG} · SGLang ${PIN_SGLANG_TAG} · TensorRT-LLM ${PIN_TRTLLM_TAG} · NIM ${PIN_NIM_LLM_TAG} · AIPerf ${PIN_AIPERF}"

if [ "${FAILS}" -gt 0 ]; then echo "${FAILS} blocking problem(s)"; exit 1; fi
echo "Preflight passed. Next: python3 -m pytest -q"
