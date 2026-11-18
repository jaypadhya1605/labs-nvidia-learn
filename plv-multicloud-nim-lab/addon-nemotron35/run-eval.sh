#!/usr/bin/env bash
# Add-on A3: one scorecard per model - conformance + agentic (thinking off AND on) + Plover summary quality/cost.
# Usage: addon-nemotron35/run-eval.sh <base-url> <model-id> <label> [usd_per_hour]
#   addon-nemotron35/run-eval.sh http://localhost:8010 nvidia/nemotron-3.5-lightning-30b-a3b lightning-aks 0.68
#   addon-nemotron35/run-eval.sh http://localhost:8000 meta/llama-3.1-8b-instruct llama8b-aks 0.68     # the baseline
set -euo pipefail
cd "$(dirname "$0")/.."
URL="${1:?base url}"; MODEL="${2:?model id}"; LABEL="${3:?label}"; HOURLY="${4:-0}"
OUT="results/addon/${LABEL}"; mkdir -p "${OUT}"
OFF='{"chat_template_kwargs": {"enable_thinking": false}}'; ON='{"chat_template_kwargs": {"enable_thinking": true}}'
case "${MODEL}" in meta/*) OFF='{}'; ON='{}' ;; esac      # Llama has no thinking switch
python3 -m pytest -q tests/conformance --endpoint "${URL}" --model "${MODEL}" --extra-body "${OFF}" | tail -1 | tee "${OUT}/conformance.txt" || true   # failures are findings, keep scoring
python3 tests/agentic/harness.py --endpoint "${URL}" --model "${MODEL}" --extra-body "${OFF}" --out "${OUT}/agentic-thinking-off.json"
python3 tests/agentic/harness.py --endpoint "${URL}" --model "${MODEL}" --extra-body "${ON}" --out "${OUT}/agentic-thinking-on.json" || true
python3 addon-nemotron35/scorecard.py run --endpoint "${URL}" --model "${MODEL}" --extra-body "${OFF}" --usd-per-hour "${HOURLY}" --out "${OUT}/plover.json"
echo "Compare: python3 addon-nemotron35/scorecard.py compare results/addon/*"
