#!/usr/bin/env bash
# Build a TensorRT engine from the portable ONNX and lay out a Triton model repository (🔁 Lab 02).
# Runs in nvcr.io/nvidia/tensorrt:<NGC_RELEASE>-py3 ON THE GPU THAT WILL SERVE IT.
#   ENGINE_MODE=build          per-architecture engine (fastest; only runs on this GPU architecture)
#   ENGINE_MODE=build-compat   --hardwareCompatibilityLevel=ampere+ (runs on Ampere and newer; some speed cost)
#   ENGINE_MODE=import         wait for a plan copied in from ANOTHER cloud (the portability drill)
set -euo pipefail
W="${WORK_DIR:-/work/triage}"
REPO="${W}/model_repository/triage_trt"
MODE="${ENGINE_MODE:-build}"
mkdir -p "${REPO}/1"
nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader || true
cp /code/config.pbtxt "${REPO}/config.pbtxt"
if [ "${MODE}" = "import" ]; then
  mkdir -p "${W}/import"
  echo "Waiting for a plan: kubectl cp <file> <ns>/<pod>:${W}/import/model.plan -c build-engine"
  until [ -f "${W}/import/model.plan" ]; do sleep 5; done
  sleep 3
  cp "${W}/import/model.plan" "${REPO}/1/model.plan"
  echo "Imported a plan built elsewhere - Triton will tell us if this GPU can run it."
  exit 0
fi
COMPAT=""
if [ "${MODE}" = "build-compat" ]; then COMPAT="--hardwareCompatibilityLevel=ampere+"; fi
SHAPES="--minShapes=input_ids:1x16,attention_mask:1x16 --optShapes=input_ids:8x128,attention_mask:8x128 --maxShapes=input_ids:32x256,attention_mask:32x256"
# shellcheck disable=SC2086
if trtexec --onnx="${W}/onnx/model.onnx" --saveEngine="${REPO}/1/model.plan" ${SHAPES} --fp16 ${COMPAT} --skipInference > "${W}/trtexec_build.log" 2>&1; then
  echo "Built engine (mode=${MODE}) with FP16 kernels enabled"
else
  echo "FP16 build failed (tail of ${W}/trtexec_build.log below); retrying in FP32"
  tail -20 "${W}/trtexec_build.log"
  # shellcheck disable=SC2086
  trtexec --onnx="${W}/onnx/model.onnx" --saveEngine="${REPO}/1/model.plan" ${SHAPES} ${COMPAT} --skipInference > "${W}/trtexec_build_fp32.log" 2>&1
fi
trtexec --loadEngine="${REPO}/1/model.plan" --shapes=input_ids:8x128,attention_mask:8x128 \
  --iterations=200 --avgRuns=50 > "${W}/trtexec_bench.log" 2>&1 || true
echo "--- trtexec (batch 8, seq 128, mode=${MODE}) ---"
grep -E "Throughput|Latency: min|GPU Compute Time: min" "${W}/trtexec_bench.log" | head -5 || true
ls -la "${REPO}/1"
