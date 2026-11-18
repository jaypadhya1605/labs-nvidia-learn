# finetune/ - M6: LoRA "prompt baking" with PyTorch + Hugging Face (2.5 h, 1x H100)

**Goal:** teach Tern Care Commitments to the weights, so the agent runs on the 560-char system prompt instead of
the 7,360-char one - with the same decisions. Then prove it with numbers: tool-call accuracy, tokens, latency.

Why it pays even with prefix caching: the cache saves the prefill of the long prompt (and stores it once), but
every decoding sequence still READS its whole context from the KV cache each step. At batch 32, ~2,300 tokens of
context per sequence cost a 7.4 ms floor per step on an H100; ~600 tokens cost 5.2 ms
(`profiling/roofline_check.py --batch 32 --ctx 2300` vs `--ctx 600`; arithmetic, not a measurement).
Cold caches (new replicas, evictions under load) and per-token pricing on hosted endpoints gain too.

## Steps
```bash
# 1. Data - laptop, seconds. Oracle targets (= the mock engine's answers), one example per agent step.
python3 finetune/make_sft_data.py            # train: seed 11, 1,000 tasks -> 6,170 steps; eval: seed 13, 200 -> 1,266
python3 finetune/make_sft_data.py --variant full --out data/sft-full   # same targets behind the full prompt

# 2. Plan - no GPU, no torch: steps, trainable params (41.9 M = 0.52% for r=16), memory per GPU
python3 finetune/train_lora.py --dry-run

# 3. Train - GPU VM, NGC PyTorch container (keep its torch; add the pinned Hugging Face stack)
docker run --rm -it --gpus all --ipc=host -v "$PWD":/lab -w /lab -e HF_TOKEN \
  -v ~/.cache/huggingface:/root/.cache/huggingface nvcr.io/nvidia/pytorch:${PIN_NGC_RELEASE}-py3 bash
pip install "trl==${PIN_TRL}" "peft==${PIN_PEFT}" "transformers==${PIN_TRANSFORMERS}" \
  "accelerate==${PIN_ACCELERATE}" "datasets==${PIN_DATASETS}"      # values from env/pins.env
python3 finetune/train_lora.py               # -> adapters/ttg-short + results/finetune/lora-r16-short-1gpu.json

# 4. Serve base + adapter in ONE engine (same GPU, same kernels), then evaluate three ways
finetune/serve_lora.sh vllm                  # or: finetune/serve_lora.sh sglang
python3 finetune/eval_toolcalls.py --model ttg-llm   --system-variant full    # before
python3 finetune/eval_toolcalls.py --model ttg-llm   --system-variant short   # control: short prompt, no training
python3 finetune/eval_toolcalls.py --model ttg-short --system-variant short   # after

# 5. Task level on the golden set (real tool calls, the number the customer buys)
TTG_PROMPT_VARIANT=short TTG_MODEL=ttg-short python -m ttg.bench.agent_bench --label vllm-lora --engine vllm \
  --levels 1,4,16,32 --n 48 --metrics-url http://localhost:8000/metrics --reset-url http://localhost:8000/reset_prefix_cache
```

| Row | Model | Prompt | Record |
|---|---|---|---|
| before | base | full (7,360 chars) | step exact %, final exact %, invalid JSON %, prompt tokens per call |
| control | base | short (560 chars) | the same - this is why the adapter exists |
| after | base + LoRA | short | the same; then golden-set success and p95 from the agent bench |

🔁 M7: the same script scales out. `accelerate launch --config_file finetune/accelerate_ddp.yaml ...` (DDP) and
`.../accelerate_fsdp2.yaml` (FSDP2) on 2+ GPUs: compare tokens/s per GPU and peak memory in results/finetune/.

## Pitfalls this code already handles (know them for the interview)
- TRL's `max_length` defaults to 1024 and truncates the END - the completion. `train_lora.py` uses 4096 / 8192 and
  tokenizes the longest examples first, failing before training if any would be cut.
- TRL loads a model given by name in float32 unless `model_init_kwargs={"dtype": "bfloat16"}` - 32 GB, not 16 GB.
- Prompt-completion data trains on the completion only (`completion_only_loss=True`): the model learns the steps,
  not to repeat observations.
- FSDP checkpoints are sharded; the adapter is saved with `FULL_STATE_DICT`, so vLLM and SGLang can load one file.
- Adapter rank must be <= the engine's `--max-lora-rank` (vLLM accepts 1, 8, 16, 32, 64, ...); `serve_lora.sh`
  raises it when needed.
- Eval prompts are teacher-forced (the oracle's history): a step-level policy score. The agent bench is the
  end-to-end score. Report both.

Optional: `--teacher` rejection-samples a hosted model on build.nvidia.com (synthetic data only) for more natural
thoughts - the student learns the teacher's wording behind the short prompt (context distillation).

## Checkpoint
1. Why does the control row (base model, short prompt) matter more than the before row?
2. Your adapter scores 99% step exact but the golden-set success rate drops. Name two likely causes.
3. What would break if the SFT targets used a different JSON layout than `ttg/agent/protocol.py`?
4. When is DDP faster than FSDP2 for this job, and when do you need FSDP2 anyway?

## Say it
"We baked your 2,000-token policy into a 42-million-parameter adapter. If it matches the base model on your
golden set, every call carries a 140-token system prompt instead of ~1,800 - and every decode step reads that much
less memory."
