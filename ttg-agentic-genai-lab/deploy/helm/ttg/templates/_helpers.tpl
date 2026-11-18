{{/* ---------- labels ---------- */}}
{{- define "ttg.labels" -}}
app.kubernetes.io/part-of: ttg
app.kubernetes.io/managed-by: {{ .Release.Service }}
helm.sh/chart: {{ .Chart.Name }}-{{ .Chart.Version }}
ttg.io/cloud: {{ .Values.cloud }}
{{- end }}

{{/* Pod labels: `app` is the selector every NetworkPolicy, Service and drill uses. */}}
{{- define "ttg.podLabels" -}}
app: {{ .name }}
app.kubernetes.io/name: {{ .name }}
app.kubernetes.io/component: {{ .component }}
app.kubernetes.io/part-of: ttg
{{- end }}

{{/* ---------- engine: support matrix (same variant names as engines/launch.sh) ---------- */}}
{{- define "ttg.supportedVariants" -}}
{{- $m := dict
  "mock" (list "base")
  "vllm" (list "base" "nocache" "fp8" "fp8kv" "spec" "eager" "lora" "tp2" "pp2" "dp2")
  "sglang" (list "base" "nocache" "fp8" "fp8kv" "spec" "eager" "lora" "tp2" "pp2" "dp2")
  "trtllm" (list "base" "nocache" "fp8" "fp8kv" "tp2" "pp2")
  "nim" (list "base" "nocache") -}}
{{- get $m .Values.engine.kind | join " " -}}
{{- end }}

{{- define "ttg.isGpuEngine" -}}
{{- if has .Values.engine.kind (list "vllm" "sglang" "trtllm" "nim") }}true{{ end -}}
{{- end }}

{{/* Fail early, at `helm template` time, instead of a CrashLoopBackOff on a rented GPU. */}}
{{- define "ttg.validate" -}}
{{- $e := .Values.engine -}}
{{- if $e.enabled }}
{{- $ok := splitList " " (include "ttg.supportedVariants" .) -}}
{{- if not (has $e.variant $ok) }}
{{- fail (printf "engine.variant=%s is not supported for engine.kind=%s (supported: %s). spec/eager/lora/dp2 are vLLM + SGLang experiments; NIM picks its engine from the profile." $e.variant $e.kind (join ", " $ok)) }}
{{- end }}
{{- if and (eq $e.variant "lora") (not $e.lora.pvcName) }}
{{- fail "engine.variant=lora needs engine.lora.pvcName: the PVC that holds the M6 adapter directory" }}
{{- end }}
{{- if and (eq $e.kind "nim") (eq .Values.secrets.mode "none") }}
{{- fail "engine.kind=nim needs secrets.mode existing|external (NGC_API_KEY + nvcr.io pull secret)" }}
{{- end }}
{{- if and (eq $e.kind "trtllm") (include "ttg.fp8Online" .) }}
{{- fail "engine.fp8Mode=online: TensorRT-LLM needs the pre-quantized ModelOpt checkpoint - use engine.fp8Mode=checkpoint (engine.fp8Model)" }}
{{- end }}
{{- if and $e.autoscaling.enabled (eq $e.kind "trtllm") }}
{{- fail "engine.autoscaling: trtllm-serve exposes JSON iteration stats, not a Prometheus queue gauge - autoscale vLLM, SGLang or NIM" }}
{{- end }}
{{- if and $e.autoscaling.enabled (eq $e.kind "mock") }}
{{- fail "engine.autoscaling is for GPU engines; the mock has no queue" }}
{{- end }}
{{- end }}
{{- if and (not $e.enabled) .Values.agent.enabled (not .Values.agent.baseUrl) }}
{{- fail "engine.enabled=false needs agent.baseUrl (e.g. https://integrate.api.nvidia.com/v1 - synthetic data only)" }}
{{- end }}
{{- end }}

{{- define "ttg.engineGpus" -}}
{{- if has .Values.engine.variant (list "tp2" "pp2" "dp2") }}2{{ else }}{{ .Values.engine.gpuCount }}{{ end -}}
{{- end }}

{{- define "ttg.engineImage" -}}
{{- if eq .Values.engine.kind "mock" -}}
{{ printf "%s:%s" .Values.image.repository .Values.image.tag }}
{{- else -}}
{{- $img := get .Values.engine.images .Values.engine.kind -}}
{{ printf "%s:%s" $img.repository (toString $img.tag) }}
{{- end -}}
{{- end }}

{{/* variant fp8 serves the ModelOpt FP8 checkpoint on every engine (same weights = fair bake-off), unless
     fp8Mode=online: vLLM / SGLang then quantize the BF16 weights at load. = engines/launch.sh FP8_MODE. */}}
{{- define "ttg.fp8Online" -}}
{{- if and (eq .Values.engine.variant "fp8") (eq .Values.engine.fp8Mode "online") }}true{{ end -}}
{{- end }}

{{- define "ttg.engineModel" -}}
{{- if and (eq .Values.engine.variant "fp8") (ne .Values.engine.fp8Mode "online") -}}
{{ .Values.engine.fp8Model }}
{{- else -}}
{{ .Values.engine.model }}
{{- end -}}
{{- end }}

{{- define "ttg.engineHealthPath" -}}
{{- if eq .Values.engine.kind "nim" }}/v1/health/ready{{ else if eq .Values.engine.kind "mock" }}/v1/models{{ else }}/health{{ end -}}
{{- end }}

{{- define "ttg.engineMetricsPath" -}}
{{- if eq .Values.engine.kind "nim" }}/v1/metrics{{ else }}/metrics{{ end -}}
{{- end }}

{{- define "ttg.queueMetric" -}}
{{- if .Values.engine.autoscaling.metric }}{{ .Values.engine.autoscaling.metric }}
{{- else if eq .Values.engine.kind "sglang" }}sglang_num_queue_reqs
{{- else }}vllm_num_requests_waiting{{ end -}}
{{- end }}

{{- define "ttg.queuePromQL" -}}
{{- if eq .Values.engine.kind "sglang" -}}
sum(sglang:num_queue_reqs{namespace="{{ .Release.Namespace }}"})
{{- else -}}
sum(vllm:num_requests_waiting{namespace="{{ .Release.Namespace }}"})
{{- end -}}
{{- end }}

{{/* ---------- engine command lines: flags verified against vLLM 0.31.0 / SGLang 0.5.22 / trtllm-serve 1.4 docs, and kept
     identical to engines/launch.sh (tests/test_deploy_chart.py compares the flag sets), so the M4 numbers carry to M9 ---------- */}}
{{- define "ttg.vllmArgs" -}}
{{- $e := .Values.engine -}}
{{- $a := list (include "ttg.engineModel" .) "--host" "0.0.0.0" "--port" "8000"
  "--served-model-name" $e.servedModelName
  "--max-model-len" (toString (int $e.maxModelLen))
  "--gpu-memory-utilization" (toString $e.gpuMemoryUtilization)
  "--enable-prompt-tokens-details" "--enable-force-include-usage" -}}
{{- if eq $e.variant "nocache" }}{{ $a = append $a "--no-enable-prefix-caching" }}{{ else }}{{ $a = append $a "--enable-prefix-caching" }}{{ end -}}
{{- if include "ttg.fp8Online" . }}{{ $a = concat $a (list "--quantization" "fp8") }}{{ end -}}
{{- if eq $e.variant "fp8kv" }}{{ $a = concat $a (list "--kv-cache-dtype" "fp8") }}{{ end -}}
{{- if eq $e.variant "spec" }}{{ $a = concat $a (list "--speculative-config" "{\"method\":\"ngram\",\"num_speculative_tokens\":4,\"prompt_lookup_max\":4,\"prompt_lookup_min\":2}") }}{{ end -}}
{{- if eq $e.variant "eager" }}{{ $a = append $a "--enforce-eager" }}{{ end -}}
{{- if eq $e.variant "lora" }}{{ $a = concat $a (list "--enable-lora" "--lora-modules" (printf "%s=%s" $e.lora.name $e.lora.path) "--max-lora-rank" (toString (int $e.lora.maxRank))) }}{{ end -}}
{{- if eq $e.variant "tp2" }}{{ $a = concat $a (list "--tensor-parallel-size" "2") }}{{ end -}}
{{- if eq $e.variant "pp2" }}{{ $a = concat $a (list "--pipeline-parallel-size" "2") }}{{ end -}}
{{- if eq $e.variant "dp2" }}{{ $a = concat $a (list "--data-parallel-size" "2") }}{{ end -}}
{{- toYaml (concat $a $e.extraArgs) -}}
{{- end }}

{{- define "ttg.sglangArgs" -}}
{{- $e := .Values.engine -}}
{{- $a := list "-m" "sglang.launch_server" "--model-path" (include "ttg.engineModel" .) "--host" "0.0.0.0" "--port" "8000"
  "--served-model-name" $e.servedModelName
  "--context-length" (toString (int $e.maxModelLen))
  "--mem-fraction-static" (toString $e.sglang.memFractionStatic)
  "--enable-metrics" "--enable-cache-report" -}}
{{- if eq $e.variant "nocache" }}{{ $a = append $a "--disable-radix-cache" }}{{ end -}}
{{- if include "ttg.fp8Online" . }}{{ $a = concat $a (list "--quantization" "fp8") }}{{ end -}}
{{- if eq $e.variant "fp8kv" }}{{ $a = concat $a (list "--kv-cache-dtype" "fp8_e4m3") }}{{ end -}}
{{- if eq $e.variant "spec" }}{{ $a = concat $a (list "--speculative-algorithm" "NGRAM" "--speculative-num-draft-tokens" "4") }}{{ end -}}
{{- if eq $e.variant "eager" }}{{ $a = append $a "--disable-cuda-graph" }}{{ end -}}
{{- if eq $e.variant "lora" }}{{ $a = concat $a (list "--enable-lora" "--lora-paths" (printf "%s=%s" $e.lora.name $e.lora.path) "--max-lora-rank" (toString (int $e.lora.maxRank))) }}{{ end -}}
{{- if eq $e.variant "tp2" }}{{ $a = concat $a (list "--tp-size" "2") }}{{ end -}}
{{- if eq $e.variant "pp2" }}{{ $a = concat $a (list "--pp-size" "2") }}{{ end -}}
{{- if eq $e.variant "dp2" }}{{ $a = concat $a (list "--dp-size" "2") }}{{ end -}}
{{- toYaml (concat $a $e.extraArgs) -}}
{{- end }}

{{/* TensorRT-LLM LLM API options (KvCacheConfig keys, docs 1.3.0rc27+): block reuse = prefix caching; dtype fp8 = FP8 KV
     cache (there is no --kv_cache_dtype CLI flag); iteration stats feed GET /metrics. Same keys as engines/trtllm/*.yaml. */}}
{{- define "ttg.trtllmOptions" -}}
{{- $e := .Values.engine -}}
kv_cache_config:
  enable_block_reuse: {{ ne $e.variant "nocache" }}
  free_gpu_memory_fraction: {{ $e.trtllm.kvCacheFreeFraction }}
{{- if eq $e.variant "fp8kv" }}
  dtype: fp8
{{- end }}
enable_iter_perf_stats: true
{{- end }}

{{/* trtllm-serve: newer images use `trtllm-serve serve MODEL`, older ones `trtllm-serve MODEL`; detect, don't guess. */}}
{{- define "ttg.trtllmScript" -}}
{{- $e := .Values.engine -}}
set -eu
SUB=""
if trtllm-serve --help 2>&1 | grep -qE '^[[:space:]]+serve([[:space:]]|$)'; then SUB="serve"; fi
NAME_ARGS=""
if trtllm-serve ${SUB} --help 2>&1 | grep -q -- '--served_model_name'; then NAME_ARGS="--served_model_name {{ $e.servedModelName }}"; fi
# shellcheck disable=SC2086
exec trtllm-serve ${SUB} {{ include "ttg.engineModel" . | quote }} --host 0.0.0.0 --port 8000 ${NAME_ARGS} \
  --max_seq_len {{ int $e.maxModelLen }} --max_batch_size {{ int $e.trtllm.maxBatchSize }} --max_num_tokens {{ int $e.trtllm.maxNumTokens }} \
  --extra_llm_api_options /config/extra-llm-api-options.yaml
{{- if eq $e.variant "tp2" }} --tp_size 2{{ end }}
{{- if eq $e.variant "pp2" }} --pp_size 2{{ end }}
{{- range $e.extraArgs }} {{ . | quote }}{{ end }}
{{- end }}

{{/* ---------- agent wiring ---------- */}}
{{- define "ttg.agentBaseUrl" -}}
{{- if .Values.agent.baseUrl }}{{ .Values.agent.baseUrl }}{{ else }}{{ printf "http://%s:8000/v1" .Values.engine.name }}{{ end -}}
{{- end }}

{{- define "ttg.agentModel" -}}
{{- $e := .Values.engine -}}
{{- if .Values.agent.model }}{{ .Values.agent.model }}
{{- else if and $e.enabled (eq $e.variant "lora") (eq $e.kind "sglang") }}{{ printf "%s:%s" $e.servedModelName $e.lora.name }}
{{- else if and $e.enabled (eq $e.variant "lora") }}{{ $e.lora.name }}
{{- else }}{{ $e.servedModelName }}{{ end -}}
{{- end }}

{{- define "ttg.agentPromptVariant" -}}
{{- if .Values.agent.promptVariant }}{{ .Values.agent.promptVariant }}
{{- else if and .Values.engine.enabled (eq .Values.engine.variant "lora") }}short
{{- else }}full{{ end -}}
{{- end }}

{{- define "ttg.securityContext" -}}
runAsNonRoot: true
allowPrivilegeEscalation: false
readOnlyRootFilesystem: true
capabilities:
  drop: ["ALL"]
{{- end }}
