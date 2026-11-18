{{- define "plover.labels" -}}
app.kubernetes.io/part-of: plover
app.kubernetes.io/managed-by: {{ .Release.Service }}
helm.sh/chart: {{ .Chart.Name }}-{{ .Chart.Version }}
plv.io/cloud: {{ .Values.cloud }}
{{- end }}

{{- define "plover.llmName" -}}{{ .Values.llm.name }}{{- end }}
{{- define "plover.apiName" -}}{{ .Values.api.name }}{{- end }}
{{- define "plover.cacheName" -}}{{ .Values.llm.name }}-cache{{- end }}

{{/* The model the API calls for "stable": an adapter name, or the base model. */}}
{{- define "plover.baseModel" -}}
{{- if .Values.llm.servedModelName }}{{ .Values.llm.servedModelName }}{{ else }}meta/llama-3.1-8b-instruct{{ end }}
{{- end }}

{{- define "plover.llmEnv" -}}
{{- if .Values.llm.servedModelName }}
- name: NIM_SERVED_MODEL_NAME
  value: {{ .Values.llm.servedModelName | quote }}
{{- end }}
{{- if and .Values.llm.profile (or (ne .Values.llm.cache.mode "nimcache") (eq .Values.llm.mode "deployment")) }}
- name: NIM_MODEL_PROFILE
  value: {{ .Values.llm.profile | quote }}
{{- end }}
{{- if and .Values.llm.lora.enabled .Values.llm.lora.adapters }}
- name: NIM_PEFT_SOURCE
  value: /scratch/loras
{{- if gt (int .Values.llm.lora.refreshSeconds) 0 }}
- name: NIM_PEFT_REFRESH_INTERVAL
  value: {{ .Values.llm.lora.refreshSeconds | quote }}
{{- end }}
{{- end }}
- name: NIM_PASSTHROUGH_ARGS
  value: {{ printf "--max-model-len %d %s" (int .Values.llm.maxModelLen) .Values.llm.extraArgs | trim | quote }}
{{- with .Values.llm.extraEnv }}
{{ toYaml . }}
{{- end }}
{{- end }}

{{/* Day-0 guard: values copied from a launch doc carry VERIFY placeholders until someone checks NGC. */}}
{{- define "plover.verifyGuard" -}}
{{- if contains "VERIFY" (printf "%s %s" .Values.llm.image.tag .Values.llm.profile) }}
{{- fail "llm.image.tag / llm.profile still say VERIFY: look them up (NGC catalog, list-model-profiles) before deploying" }}
{{- end }}
{{- end }}
