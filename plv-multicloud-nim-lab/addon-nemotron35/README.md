# Add-on: Nemotron 3.5 day-0 launch kit (≈4 hours)

A model launched today (2026-10-15: Super VL + Bolt; Lightning has been out since 2026-08-11). Your customer will
ask "should we switch?" on their next call. This module is the repeatable answer: **verify the claims, size it from
its config, run it on your clusters with the same chart, gate it like any release, and brief the customer.**

🔁 Lab 02 physics (bytes per token) · 🔁 Lab 03 prompt-injection eval · 🔁 M5 AIPerf · 🔁 M8 release gate · 🔁 M6 GitOps pause

| Step | Time | You produce |
|---|---|---|
| A0 Claims check | 30 min | `claims-check.md` filled with your own evidence |
| A1 Size from config | 30 min | model specs in `models.yaml`, selector table for 3 models x 4 clouds |
| A2 Deploy Lightning on AKS (Super VL on GKE: stretch) | 60 min | running NIM next to Plover, `list-model-profiles` output |
| A3 Evaluate | 45 min | scorecards: conformance, Plover quality, agentic (thinking off/on), injection |
| A4 Benchmark and price | 45 min | AIPerf sweep, measured vs predicted, $/1M tokens |
| A5 Brief + workshop | 30 min | `customer-brief.md`, 10-minute segment |

## A0 - Claims check
Read `claims-check.md`. Re-derive rows 1-5 yourself:
```bash
python3 selector/model_sizer.py --config nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16 --name super-vl --total-b 121 --active-b 12
python3 addon-nemotron35/quant_report.py nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4
python3 addon-nemotron35/quant_report.py nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB
```
**Checkpoint:** you can say, without notes, why "fits in 24 GB" does not mean "serves on an L4".

## A1 - Size it (🔁 Lab 02 physics, extended)
Hybrid Mamba-MoE changes the bytes per step: only 6-8 attention layers grow a KV cache, but every sequence carries a
fixed SSM state that is **read and written every decode step** (fp32 here: `mamba_ssm_cache_dtype`).
```bash
S="python3 selector/instance_selector.py --models-file addon-nemotron35/models.yaml --spot"
$S --model nemotron-3.5-lightning --isl 3000 --osl 400 --itl-slo-ms 30 --precisions bf16,w4a16,nvfp4
$S --model nemotron-3.5-super-vl  --isl 3000 --osl 400 --itl-slo-ms 50 --precisions bf16,nvfp4
$S --model nemotron-3.5-bolt-24gb --isl 2000 --osl 400 --itl-slo-ms 50 --precisions iq2
$S --model llama-3.1-8b           --isl 3000 --osl 400 --itl-slo-ms 30          # today's baseline, same shape
```
What you should see (estimates, 2026-10-15 prices): Lightning **W4A16 on the AKS A100 ≈ 6,900 tok/s** at a 30 ms ITL
vs ≈ 640 tok/s in BF16 (BF16 weights leave only ~4 GB for state) and ≈ 2,200 tok/s for Llama 8B; Super VL needs
Blackwell NVFP4 (75 GB on a 96 GB RTX PRO 6000, ≈ 480 tok/s at 50 ms); Bolt does not fit L4/A10 with any headroom.
**Checkpoint:** explain why per-sequence memory at 128K context is ~17 GB for Llama 8B but < 1 GB for both Nemotrons.

## A2 - Deploy next to Plover (same chart, separate release)
Make sure M8's MIG split is reverted (`mlops/mig.sh aks all-disabled`): Lightning BF16 needs the whole 80 GB.
```bash
source env/lab.env && scripts/check-images.sh "${PIN_NIM_LIGHTNING_IMAGE}:${PIN_NIM_LIGHTNING_TAG}"   # is the tag pullable with YOUR key?
addon-nemotron35/deploy.sh aks lightning up          # GitOps pause + NIM scale-down (M6), then helm install plv-addon
kubectl --context plv-aks -n plv-addon logs -f -l app.kubernetes.io/name=plover-llm     # ~63 GB BF16 download
addon-nemotron35/deploy.sh aks lightning profiles    # list-model-profiles on THIS GPU
```
Step 3: pin the profile NIM chose (`llm.profile` in `values/aks-lightning.yaml`). Step 4: switch `precision`/`profile`
to the W4A16 TP1 profile from the list, `helm upgrade` again, and compare KV-cache headroom in the NIM log.
Stretch: Super VL on GKE - `values/gke-super-vl.yaml` refuses to render until you replace `VERIFY-in-NGC` with the
real tag (the chart's day-0 guard). Driver ≥ 580 is required (NIM docs); the GKE installer and the AKS operator meet it.

## A3 - Evaluate with Plover's gates (🔁 Lab 03 + M8)
```bash
kubectl --context plv-aks -n plv-addon port-forward svc/plover-llm 8010:8000 &
addon-nemotron35/run-eval.sh http://localhost:8010 nvidia/nemotron-3.5-lightning-30b-a3b lightning-aks 0.68
addon-nemotron35/run-eval.sh http://localhost:8000 meta/llama-3.1-8b-instruct llama8b-aks 0.68   # baseline (after resume)
python3 addon-nemotron35/scorecard.py compare results/addon/*
```
Look for: `tool_call_in_content` (parser flags), injection success with thinking ON vs OFF, hallucinated codes.
**Checkpoint:** a one-line verdict per model: "passes / fails the Plover gate because ...".

## A4 - Benchmark and price (🔁 M5)
```bash
PLV_URL=http://plover-llm.plv-addon.svc:8000 PLV_LLM_NS=plv-addon PLV_MODEL=nvidia/nemotron-3.5-lightning-30b-a3b \
PLV_TOKENIZER=nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16 bench/run_sweep.sh aks synth-3000-400 "1 8 32 64 128 192"
python3 bench/compare_clouds.py results/aks-synth-3000-400-* --itl-slo-ms 30 --models-file addon-nemotron35/models.yaml
```
(The AIPerf toolbox stays in plv-serving; the overlay's `networkPolicy.allowNamespaces` lets it reach plv-addon.) Compare measured $/1M tokens with OpenRouter's Lightning price ($0.025 in / $0.12 out
per 1M, 2026-10-15) - self-hosting wins only above a utilization you can now compute (`bench/cost_model.py`).

## A5 - Brief the customer
Edit `customer-brief.md` with YOUR measured numbers (replace every estimate), rehearse `workshop-segment.md`, then
`addon-nemotron35/deploy.sh aks lightning down` (uninstalls and resumes GitOps - Argo CD restores Plover's NIM).
