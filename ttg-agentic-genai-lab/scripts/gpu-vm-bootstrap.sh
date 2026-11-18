#!/usr/bin/env bash
# Bootstrap a rented Ubuntu 22.04/24.04 GPU VM (Brev, Lambda, RunPod or any cloud) for M4-M8: NVIDIA driver check,
# Docker + NVIDIA Container Toolkit, Python 3.12 venv (uv), nvcr.io + Hugging Face logins, pinned engine images, GPU
# topology. It also prepares the GitLab perf-gate runner (shell executor) when a gitlab-runner user exists.
# Idempotent: safe to re-run after a reboot or on a new VM.
# Usage (on the VM, from the repo root):
#   cp env/lab.env.example env/lab.env && $EDITOR env/lab.env      # HF_TOKEN, NGC_API_KEY, GPU_HOURLY_USD
#   scripts/gpu-vm-bootstrap.sh                    # pulls vllm + sglang (M4 fast path)
#   scripts/gpu-vm-bootstrap.sh --pull all         # + trtllm, nim, NGC PyTorch (nsys, nccl-tests, training)
#   scripts/gpu-vm-bootstrap.sh --no-pull          # tools and logins only
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck source=/dev/null
. env/pins.env
if [ -f env/lab.env ]; then
  # shellcheck source=/dev/null
  . env/lab.env
fi

PULL="vllm,sglang"
while [ $# -gt 0 ]; do
  case "$1" in
    --pull) PULL="${2:?--pull needs a list: vllm,sglang,trtllm,nim,pytorch | all}"; shift 2 ;;
    --no-pull) PULL=""; shift ;;
    -h|--help) sed -n '2,11p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
if [ "${PULL}" = "all" ]; then PULL="vllm,sglang,trtllm,nim,pytorch"; fi

SUDO=""
if [ "$(id -u)" -ne 0 ]; then SUDO="sudo"; fi
say() { printf '\n== %s\n' "$*"; }
die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

say "1/7 NVIDIA driver"
command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi not found: pick a GPU image with the NVIDIA driver preinstalled (every listed provider offers one)"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader

say "2/7 Docker"
if ! command -v docker >/dev/null 2>&1; then
  curl -fsSL https://get.docker.com | ${SUDO} sh
  if [ -n "${SUDO}" ]; then ${SUDO} usermod -aG docker "$(id -un)"; echo "added $(id -un) to the docker group (new shells pick it up)"; fi
fi
DOCKER="docker"
if ! docker info >/dev/null 2>&1; then DOCKER="${SUDO} docker"; fi
${DOCKER} --version

say "3/7 NVIDIA Container Toolkit"
if ! ${DOCKER} info 2>/dev/null | grep -qi 'nvidia'; then
  if ! command -v nvidia-ctk >/dev/null 2>&1; then
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
      | ${SUDO} gpg --dearmor --yes -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
      | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
      | ${SUDO} tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null
    ${SUDO} apt-get update -qq
    ${SUDO} apt-get install -y -qq nvidia-container-toolkit
  fi
  ${SUDO} nvidia-ctk runtime configure --runtime=docker
  ${SUDO} systemctl restart docker
fi
${DOCKER} run --rm --gpus all ubuntu:24.04 nvidia-smi -L

say "4/7 Python 3.12 venv for the bench and the gates (uv ${PIN_UV}; Ubuntu 22.04 ships Python 3.10)"
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf "https://astral.sh/uv/${PIN_UV}/install.sh" | ${SUDO} env UV_INSTALL_DIR=/usr/local/bin UV_NO_MODIFY_PATH=1 sh
fi
uv --version
if [ ! -x .venv/bin/python ]; then uv venv -q --python 3.12 .venv; fi
uv pip install -q --python .venv -r requirements-dev.txt
echo "lab venv ready: . .venv/bin/activate"
if id gitlab-runner >/dev/null 2>&1; then
  ${SUDO} usermod -aG docker gitlab-runner      # the perf-gate job runs engines/launch.sh (docker) as this user
  echo "gitlab-runner added to the docker group: sudo gitlab-runner restart"
fi

say "5/7 Registry and Hugging Face logins"
if [ -n "${NGC_API_KEY:-}" ]; then
  printf '%s' "${NGC_API_KEY}" | ${DOCKER} login nvcr.io -u "\$oauthtoken" --password-stdin
else
  echo "NGC_API_KEY empty: nvcr.io pulls of NIM images will fail (TensorRT-LLM and NGC PyTorch are public)"
fi
if [ -n "${HF_TOKEN:-}" ]; then
  echo "HF_TOKEN set: engines/launch.sh passes it to the containers (gated Llama 3.1 needs the license accepted)"
else
  echo "HF_TOKEN empty: meta-llama/Llama-3.1-8B-Instruct is gated - set it in env/lab.env before M4"
fi

say "6/7 Pinned images (${PULL:-none})"
OLDIFS="${IFS}"; IFS=","
for what in ${PULL}; do
  IFS="${OLDIFS}"
  case "${what}" in
    vllm) ref="${PIN_VLLM_IMAGE}:${PIN_VLLM_TAG}" ;;
    sglang) ref="${PIN_SGLANG_IMAGE}:${PIN_SGLANG_TAG}" ;;
    trtllm) ref="${PIN_TRTLLM_IMAGE}:${PIN_TRTLLM_TAG}" ;;
    nim) ref="${PIN_NIM_LLM_IMAGE}:${PIN_NIM_LLM_TAG}" ;;
    pytorch) ref="${PIN_PYTORCH_IMAGE}:${PIN_NGC_RELEASE}-py3" ;;
    *) die "unknown image key: ${what} (vllm, sglang, trtllm, nim, pytorch)" ;;
  esac
  echo "pulling ${ref}"
  ${DOCKER} pull -q "${ref}"
  IFS=","
done
IFS="${OLDIFS}"

say "7/7 GPU topology (M7: NV# = NVLink hops, PIX/PHB/SYS = PCIe paths)"
nvidia-smi topo -m || true
mkdir -p results
nvidia-smi --query-gpu=index,name,memory.total,driver_version,pci.bus_id --format=csv > results/gpu-vm.csv
echo "GPU facts saved to results/gpu-vm.csv. Next: engines/launch.sh vllm base"
