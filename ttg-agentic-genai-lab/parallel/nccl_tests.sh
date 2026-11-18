#!/usr/bin/env bash
# M7 - measure the GPU fabric before you trust any parallelism plan.
# Builds NVIDIA/nccl-tests inside the NGC PyTorch container (it already ships CUDA + NCCL) and runs the four
# collectives that parallelism uses: all_reduce (DP, TP), all_gather + reduce_scatter (FSDP / ZeRO-3, sequence
# parallel) and alltoall (MoE expert parallelism), 8 B to 4 GB, on N GPUs of THIS node.
#
# Usage: parallel/nccl_tests.sh [ngpus] [--p2p-off] [--nvls-off] [--dry-run]
#   parallel/nccl_tests.sh 8               # 8x H100 SXM: NVLink 4 + NVSwitch (expect busbw ~80% of 450 GB/s)
#   parallel/nccl_tests.sh 8 --nvls-off    # NVLink SHARP off (NCCL_NVLS_ENABLE=0): in-switch reduction disabled
#   parallel/nccl_tests.sh 8 --p2p-off     # drill: NCCL_P2P_DISABLE=1 pushes traffic through host memory
#   parallel/nccl_tests.sh 4               # 4x L40S PCIe node: the contrast (no NVLink)
# Then:  python3 -m parallel.compare_nccl results/nccl/<run>/all_reduce_perf.log --link nvlink4
# Needs: Docker + NVIDIA Container Toolkit on the GPU VM, Internet access to github.com and nvcr.io.
set -euo pipefail
cd "$(dirname "$0")/.."
if [ -f env/pins.env ]; then
  # shellcheck source=/dev/null
  . env/pins.env
fi

NGPUS=8
DRY=0
SUFFIX=""
EXTRA_ENV=""
for arg in "$@"; do
  case "${arg}" in
    --p2p-off) SUFFIX="${SUFFIX}-p2poff"; EXTRA_ENV="${EXTRA_ENV} -e NCCL_P2P_DISABLE=1" ;;
    --nvls-off) SUFFIX="${SUFFIX}-nvlsoff"; EXTRA_ENV="${EXTRA_ENV} -e NCCL_NVLS_ENABLE=0" ;;
    --dry-run) DRY=1 ;;
    ''|*[!0-9]*) echo "unknown argument: ${arg}" >&2; exit 2 ;;
    *) NGPUS="${arg}" ;;
  esac
done

IMAGE="nvcr.io/nvidia/pytorch:${PIN_NGC_RELEASE:-26.09}-py3"
HOST="$(hostname -s 2>/dev/null || echo node)"
RUN="results/nccl/${HOST}-${NGPUS}gpu${SUFFIX}-$(date -u +%Y%m%dT%H%M%SZ)"
# NCCL_DEBUG=INFO prints the transport NCCL chose (NVLS / P2P / SHM / NET) into the log - read it.
# Single quotes on purpose: $(nproc), ${t} and ${NGPUS} expand INSIDE the container (NGPUS is passed with -e).
# shellcheck disable=SC2016
INNER='set -e
nvidia-smi topo -m > /out/topo.txt
nvidia-smi nvlink -s > /out/nvlink.txt 2>&1 || echo "no NVLink on this node" > /out/nvlink.txt
git clone --depth 1 https://github.com/NVIDIA/nccl-tests /tmp/nccl-tests > /out/build.log 2>&1
make -C /tmp/nccl-tests -j"$(nproc)" MPI=0 CUDA_HOME=/usr/local/cuda >> /out/build.log 2>&1
for t in all_reduce all_gather reduce_scatter alltoall; do
  /tmp/nccl-tests/build/${t}_perf -b 8 -e 4G -f 2 -g "${NGPUS}" 2> /out/${t}_perf.stderr | tee /out/${t}_perf.log
done'

# shellcheck disable=SC2086  # EXTRA_ENV is a list of -e flags on purpose
set -- docker run --rm --gpus all --ipc=host --network host --ulimit memlock=-1 --ulimit stack=67108864 \
  -e NGPUS="${NGPUS}" -e NCCL_DEBUG=INFO ${EXTRA_ENV} -v "${PWD}/${RUN}:/out" "${IMAGE}" bash -lc "${INNER}"

if [ "${DRY}" = "1" ]; then
  printf '%q ' "$@"
  echo
  exit 0
fi
mkdir -p "${RUN}"
echo "nccl-tests on ${NGPUS} GPUs (${IMAGE}) -> ${RUN}"
"$@"
grep -h "NCCL INFO" "${RUN}"/all_reduce_perf.stderr 2>/dev/null | grep -E "NVLS|P2P|SHM|NET/" | sort -u | head -5 || true
echo
echo "Compare with the link you think you have:"
echo "  python3 -m parallel.compare_nccl ${RUN}/all_reduce_perf.log --link nvlink4   # or pcie4 on an L40S node"
