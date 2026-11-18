#!/usr/bin/env bash
# drills/inject.sh - M13 break-fix drills for the Tern Disruption Concierge (Lab 05).
# Usage: eval "$(drills/inject.sh <1-10|8b>)"           inject one drill into THIS shell
#        eval "$(drills/inject.sh <1-10|8b> revert)"    undo it (then `. env/lab.env` restores your own defaults)
#        drills/inject.sh list | help
# stdout carries only `export` and `unset` lines, so eval is safe. What to run next goes to stderr. Printing needs no
# GPU and changes nothing on its own. One person injects, the other diagnoses from symptoms only (drills/README.md).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
if [ -f "${ROOT}/env/pins.env" ]; then
  # shellcheck source=/dev/null
  . "${ROOT}/env/pins.env"
fi
PYTORCH_IMAGE="${PIN_PYTORCH_IMAGE:-nvcr.io/nvidia/pytorch}:${PIN_NGC_RELEASE:-26.09}-py3"

usage() {
  cat <<'EOF'
Usage: eval "$(drills/inject.sh <1-10|8b>)"          inject one drill into this shell
       eval "$(drills/inject.sh <1-10|8b> revert)"   undo it; `. env/lab.env` then restores your own defaults
       drills/inject.sh list                         the drills and what each one needs
       drills/inject.sh help                         this text
stdout holds only export/unset lines (safe to eval); what to run next goes to stderr.
Laptop drills use the mock engine (uvicorn ttg.mock.fake_llm:app --port 8000); [GPU] drills need the M4 GPU VM.
EOF
}

list() {
  cat <<'EOF'
 1  laptop   A harmless prompt change, a bigger bill
 2  laptop   Retries out of nowhere (GPU variant: an engine that will not speak JSON)
 3  GPU      Only the long tasks fail
 4  GPU      Faster alone, slower in a crowd
 5  2 GPUs   Two GPUs, no faster
 6  GPU      Receipts got slow
 7  GPU      The engine will not start
 8  laptop   The agent says it rebooked; the system of record says no
 8b laptop   The agent API hangs at startup
 9  GPU      The adapter will not load
10  laptop   A smarter prompt, a broken agent
EOF
}

note() { printf '%s\n' "$*" >&2; }
q() { printf "'%s'" "$(printf '%s' "$1" | sed "s/'/'\\\\''/g")"; }
ex() { printf 'export %s=%s\n' "$1" "$(q "$2")"; }
un() { printf 'unset %s\n' "$*"; }
relaunch() { note "Then: engines/stop.sh && engines/launch.sh $1"; }

inject() {
  ex TTG_DRILL "$1"
  case "$1" in
    1) ex TTG_BUST_CACHE 1
       note "Rerun the bench you already have a baseline for, under a new label, e.g.:"
       note "  python -m ttg.bench.agent_bench --label mock-drill1 --engine mock --levels 1,8 --n 24 \\"
       note "    --metrics-url http://localhost:8000/metrics --reset-url http://localhost:8000/reset"
       note "  python -m ttg.bench.ledger results/bench/mock-base.json results/bench/mock-drill1.json --out results/drills" ;;
    2) ex MOCK_FAULTS invalid_json_once
       ex EXTRA_DOCKER "-v ${ROOT}/drills/files/broken-chat-template.jinja:/drills/broken-chat-template.jinja:ro"
       ex EXTRA_ARGS "--chat-template /drills/broken-chat-template.jinja"
       note "Laptop: restart the mock in THIS shell (the mock reads MOCK_FAULTS): uvicorn ttg.mock.fake_llm:app --port 8000"
       note "GPU:    engines/stop.sh && engines/launch.sh vllm base"
       note "Both:   python -m ttg.agent.run --n 20 --concurrency 4" ;;
    3) ex MAX_MODEL_LEN 3072
       relaunch "vllm base     (MAX_MODEL_LEN also drives sglang --context-length and trtllm --max_seq_len)"
       note "Run:  python -m ttg.bench.agent_bench --label vllm-drill3 --engine vllm --levels 4 --n 48" ;;
    4) note "Nothing to export: this drill is a measurement. Same levels, two engine variants:"
       note "  engines/launch.sh vllm base && python -m ttg.bench.agent_bench --label vllm-base --engine vllm --levels 1,16,64 --n 128"
       note "  engines/stop.sh && engines/launch.sh vllm spec && python -m ttg.bench.agent_bench --label vllm-spec --engine vllm --levels 1,16,64 --n 128"
       note "  python -m ttg.bench.ledger results/bench/vllm-base.json results/bench/vllm-spec.json --order vllm-base,vllm-spec --out results/drills" ;;
    5) ex GPUS 0,1
       ex EXTRA_DOCKER "-e NCCL_P2P_DISABLE=1 -e NCCL_DEBUG=INFO"
       ex EXTRA_ARGS "--disable-custom-all-reduce"
       relaunch "vllm tp2"
       note "Run:  python -m ttg.bench.agent_bench --label vllm-drill5 --engine vllm --gpus 2 --levels 1,16 --n 48" ;;
    6) ex TTG_VLM_BASE_URL http://localhost:8001/v1
       ex TTG_VLM_MODEL ttg-vlm
       ex TTG_RECEIPT_DIR "${ROOT}/results/drills/receipts-12mp"
       note "Photos:  python -m vlm.make_receipts (if data/receipts is empty), then python drills/fixtures.py receipts-photo"
       note "Engines: the agent's LLM and the ttg-vlm container as in docs/06-multimodal.md section 7, but start ttg-vlm"
       note "         WITHOUT --mm-processor-kwargs (that is the fault: nothing caps the image size)."
       note "Run:  python -m vlm.agent_receipts --concurrency 8   (the 35 receipt tasks, VLM path)" ;;
    7) note "Start the other tenant (20 GB on GPU 0):"
       note "  docker run -d --rm --name ttg-gpu-hog --gpus device=0 ${PYTORCH_IMAGE} \\"
       note "    python -c \"import torch, time; x = torch.empty(20 * 1024**3, dtype=torch.uint8, device='cuda'); time.sleep(86400)\""
       relaunch "vllm base" ;;
    8) ex TTG_TOOL_TIMEOUT_S 0.001
       note "Run:  python -m ttg.agent.run --n 20 --concurrency 4" ;;
    8b) ex TTG_MCP_MODE http
        ex TTG_MCP_OPS_URL http://10.255.255.1:8101/mcp
        ex TTG_MCP_GUEST_URL http://localhost:8102/mcp
        ex TTG_TOOL_TIMEOUT_S 120
        note "Start the guest server only:  python -m ttg.mcp.guest_server --transport streamable-http --port 8102 &"
        note "Run (and time it):  time python -m ttg.agent.run --n 1" ;;
    9) ex ADAPTER_DIR "${ROOT}/results/drills/adapters-r32"
       note "Make the adapter:  python drills/fixtures.py lora-rank --rank 32   (copies adapters/ttg-short from M6)"
       relaunch "vllm lora" ;;
    10) ex TTG_THOUGHTS verbose
        note "Run:  python -m ttg.bench.agent_bench --label mock-drill10 --engine mock --levels 1,8 --n 24"
        note "Part 2, same shell:  export TTG_MAX_TOKENS=1024  and run it again as --label mock-drill10b" ;;
  esac
}

revert() {
  case "$1" in
    1) un TTG_DRILL TTG_BUST_CACHE ;;
    2) un TTG_DRILL MOCK_FAULTS EXTRA_DOCKER EXTRA_ARGS
       note "Restart the mock (laptop) or relaunch the engine (GPU) in this shell." ;;
    3) un TTG_DRILL MAX_MODEL_LEN; relaunch "vllm base" ;;
    4) un TTG_DRILL; relaunch "vllm base" ;;
    5) un TTG_DRILL GPUS EXTRA_DOCKER EXTRA_ARGS; relaunch "vllm base" ;;
    6) un TTG_DRILL TTG_VLM_BASE_URL TTG_VLM_MODEL TTG_RECEIPT_DIR
       note "Restart ttg-vlm with --mm-processor-kwargs '{\"max_pixels\": 1003520}' (docs/06 section 7)." ;;
    7) un TTG_DRILL; note "Then: docker rm -f ttg-gpu-hog"; relaunch "vllm base" ;;
    8) un TTG_DRILL TTG_TOOL_TIMEOUT_S ;;
    8b) un TTG_DRILL TTG_MCP_MODE TTG_MCP_OPS_URL TTG_MCP_GUEST_URL TTG_TOOL_TIMEOUT_S
        note "Stop the guest server you started on port 8102." ;;
    9) un TTG_DRILL ADAPTER_DIR; relaunch "vllm lora" ;;
    10) un TTG_DRILL TTG_THOUGHTS TTG_MAX_TOKENS ;;
  esac
}

DRILL="${1:-}"
MODE="${2:-inject}"
case "${DRILL}" in
  ""|-h|--help|help) usage; exit 0 ;;
  list) list; exit 0 ;;
  1|2|3|4|5|6|7|8|8b|9|10) ;;
  *) note "inject.sh: unknown drill '${DRILL}' (1-10 or 8b)"; usage >&2; exit 2 ;;
esac
case "${MODE}" in
  inject) inject "${DRILL}" ;;
  revert|reset) revert "${DRILL}" ;;
  *) note "inject.sh: unknown mode '${MODE}' (revert)"; exit 2 ;;
esac
case "${ROOT}" in *" "*) note "warning: the repo path contains a space; EXTRA_DOCKER/EXTRA_ARGS paths will split." ;; esac
if [ -t 1 ]; then note "(Nothing changed yet: run it as  eval \"\$(drills/inject.sh $*)\"  to apply.)"; fi
