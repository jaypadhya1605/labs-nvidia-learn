#!/usr/bin/env bash
# Build a TensorRT engine from the exported ONNX and lay out a Triton model repository.
# Runs in nvcr.io/nvidia/tensorrt:<NGC_RELEASE>-py3 on the SAME GPU type/slice class that will serve it:
# engines are specific to GPU architecture and TensorRT version.
set -euo pipefail
W="${WORK_DIR:-/models/triage}"
REPO="${W}/model_repository/triage_trt"
mkdir -p "${REPO}/1"
nvidia-smi -L || true
SHAPES="--minShapes=input_ids:1x16,attention_mask:1x16 --optShapes=input_ids:8x128,attention_mask:8x128 --maxShapes=input_ids:32x256,attention_mask:32x256"
# shellcheck disable=SC2086
if trtexec --onnx="${W}/onnx/model.onnx" --saveEngine="${REPO}/1/model.plan" ${SHAPES} --fp16 --skipInference > "${W}/trtexec_build.log" 2>&1; then
  echo "Built engine with FP16 kernels enabled"
else
  echo "FP16 build failed (tail of ${W}/trtexec_build.log below); retrying in FP32"
  tail -20 "${W}/trtexec_build.log"
  # shellcheck disable=SC2086
  trtexec --onnx="${W}/onnx/model.onnx" --saveEngine="${REPO}/1/model.plan" ${SHAPES} --skipInference > "${W}/trtexec_build_fp32.log" 2>&1
fi
cp /code/config.pbtxt "${REPO}/config.pbtxt"
trtexec --loadEngine="${REPO}/1/model.plan" --shapes=input_ids:8x128,attention_mask:8x128 \
  --iterations=200 --avgRuns=50 > "${W}/trtexec_bench.log" 2>&1 || true
echo "--- trtexec (batch 8, seq 128) ---"
grep -E "Throughput|Latency: min|GPU Compute Time: min" "${W}/trtexec_bench.log" | head -5 || true
ls -la "${REPO}" "${REPO}/1"
