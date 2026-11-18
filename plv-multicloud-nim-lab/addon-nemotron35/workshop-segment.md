# Workshop segment (10 min): "Day-0 with a new model - Nemotron 3.5"

**Goal:** show how an SA turns a launch into a decision in one afternoon. Slide 13 + one live terminal.

1. (2 min) **Read the config, not the press release.** Run `python3 selector/model_sizer.py --config
   tests/fixtures/nemotron-3.5-super-vl-120b-a12b.config.json --name super-vl --total-b 121 --active-b 12`.
   Point at: 8 attention layers out of 88 -> 8 KiB/token KV; 170 MB fixed SSM state per sequence.
2. (2 min) **"2-bit" decoded.** `python3 addon-nemotron35/quant_report.py nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB`
   -> IQ2_XXS experts, NVFP4/Q8_0 elsewhere, 24.08 GB on disk -> great on a 24 GiB workstation, not a cloud L4 server.
3. (3 min) **Physics.** `python3 selector/instance_selector.py --models-file addon-nemotron35/models.yaml --spot
   --model nemotron-3.5-lightning --isl 3000 --osl 400 --itl-slo-ms 30` - W4A16 on A100 vs BF16: same GPU, ~10x the
   SLO-compliant batch, because weights shrink 3.5x and the per-sequence state is small.
4. (2 min) **Gates, not vibes.** Show `results/addon/*` from `scorecard.py compare`: conformance, schema-valid,
   hallucinated codes, agentic success, injection success (thinking off/on).
5. (1 min) **Close:** "New model = same gates, same chart, same GitOps. That is why you can adopt it in two weeks."
