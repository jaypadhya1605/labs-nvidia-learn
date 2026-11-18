#!/usr/bin/env python3
"""LoRA "prompt baking" with TRL + PEFT: Llama-3.1-8B learns Tern Care, so the agent can run on the 560-char prompt.

Usage:
  python3 finetune/train_lora.py --dry-run                         # resolved config + size math; no torch needed
  python3 finetune/train_lora.py                                   # 1 GPU (H100 80GB: about one epoch per session)
  torchrun --nproc_per_node 2 finetune/train_lora.py               # DDP on 2 GPUs
  accelerate launch --config_file finetune/accelerate_ddp.yaml finetune/train_lora.py
  accelerate launch --config_file finetune/accelerate_fsdp2.yaml finetune/train_lora.py      # FSDP2 (M7 contrast)
Defaults: LoRA r=16, alpha=32, dropout 0.05 on q,k,v,o,gate,up,down; BF16 weights; lr 2e-4 cosine, 3% warm-up;
1 epoch; per-device batch 4 x grad-accum 4; completion-only loss (prompt-completion data); max_length 4096 for the
short prompt, 8192 for the full one (TRL's default of 1024 would silently cut the completion off).
Reports: tokens/s (TRL's num_tokens counter / wall time), peak GPU memory, mean step time -> results/finetune/.
Saves the adapter (adapter_config.json + adapter_model.safetensors) to adapters/ttg-short, which is what
`engines/launch.sh vllm lora` (or sglang lora) serves. Data: python3 finetune/make_sft_data.py first.
Container: nvcr.io/nvidia/pytorch:<PIN_NGC_RELEASE>-py3 + pip install trl peft transformers accelerate datasets
at the env/pins.env versions.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_TARGETS = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
DEFAULT_MODEL = os.environ.get("PIN_AGENT_MODEL", "meta-llama/Llama-3.1-8B-Instruct")
MODEL_CONFIGS = {"llama-3.1-8b": REPO / "archmath" / "configs" / "llama-3.1-8b.json"}


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--data-dir", default=str(REPO / "data" / "sft"))
    ap.add_argument("--train", help="default: <data-dir>/train.jsonl")
    ap.add_argument("--eval", help="default: <data-dir>/eval.jsonl")
    ap.add_argument("--out", default=str(REPO / "adapters" / os.environ.get("PIN_LORA_NAME", "ttg-short")))
    ap.add_argument("--run-name", default="", help="results/finetune/<run-name>.json (default: derived)")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--max-steps", type=int, default=-1, help="> 0 overrides --epochs (quick runs)")
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--warmup", type=float, default=0.03, help="< 1 = fraction of steps, >= 1 = steps")
    ap.add_argument("--scheduler", default="cosine")
    ap.add_argument("--batch", type=int, default=4, help="per-device micro-batch")
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--max-length", type=int, default=0, help="0 = 4096 (short prompt data) or 8192 (full)")
    ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--targets", default=DEFAULT_TARGETS)
    ap.add_argument("--max-train-examples", type=int, default=0, help="0 = all")
    ap.add_argument("--max-eval-examples", type=int, default=256)
    ap.add_argument("--eval-steps", type=int, default=0, help="0 = evaluate once at the end")
    ap.add_argument("--packing", action="store_true", help="pack examples into max_length blocks (verify first)")
    ap.add_argument("--no-gradient-checkpointing", action="store_true")
    ap.add_argument("--logging-steps", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--report-to", default="none")
    ap.add_argument("--dry-run", action="store_true", help="print the plan and exit (no torch / trl import)")
    a = ap.parse_args(argv)
    a.train = a.train or str(Path(a.data_dir) / "train.jsonl")
    a.eval = a.eval or str(Path(a.data_dir) / "eval.jsonl")
    return a


# --------------------------------------------------------------------------------------------- pure size math
def count_lines(path: str) -> int:
    p = Path(path)
    if not p.exists():
        return 0
    with p.open("rb") as f:
        return sum(1 for _ in f)


def read_stats(data_dir: str) -> dict:
    p = Path(data_dir) / "stats.json"
    return json.loads(p.read_text()) if p.exists() else {}


def model_config(model: str) -> dict | None:
    key = "llama-3.1-8b" if "llama-3.1-8b" in model.lower() else None
    p = MODEL_CONFIGS.get(key) if key else None
    if p and p.exists():
        return json.loads(p.read_text())
    cfg = Path(model) / "config.json"
    return json.loads(cfg.read_text()) if cfg.exists() else None


def lora_shapes(cfg: dict) -> dict:
    """(in_features, out_features) of each LoRA target in one decoder layer of a Llama-style model."""
    h, nh = cfg["hidden_size"], cfg["num_attention_heads"]
    nkv = cfg.get("num_key_value_heads") or nh
    hd = cfg.get("head_dim") or h // nh
    inter = cfg["intermediate_size"]
    return {"q_proj": (h, nh * hd), "k_proj": (h, nkv * hd), "v_proj": (h, nkv * hd), "o_proj": (nh * hd, h),
            "gate_proj": (h, inter), "up_proj": (h, inter), "down_proj": (inter, h)}


def lora_trainable_params(cfg: dict, r: int, targets: list) -> int:
    """LoRA adds A (r x in) and B (out x r) per target matrix: r * (in + out) parameters each."""
    shapes = lora_shapes(cfg)
    unknown = [t for t in targets if t not in shapes]
    if unknown:
        raise ValueError(f"unknown LoRA targets {unknown}; known: {sorted(shapes)}")
    return cfg["num_hidden_layers"] * sum(r * (shapes[t][0] + shapes[t][1]) for t in targets)


def total_params(cfg: dict) -> int:
    shapes = lora_shapes(cfg)
    per_layer = sum(i * o for i, o in shapes.values()) + 2 * cfg["hidden_size"]
    emb = cfg["vocab_size"] * cfg["hidden_size"]
    return cfg["num_hidden_layers"] * per_layer + cfg["hidden_size"] + emb * (1 if cfg.get("tie_word_embeddings") else 2)


def resolve_max_length(a, stats: dict) -> int:
    if a.max_length:
        return a.max_length
    return 8192 if stats.get("variant") == "full" else 4096


def plan(a) -> dict:
    """Everything that can be decided without a GPU, so --dry-run (and CI) can check it."""
    stats = read_stats(a.data_dir)
    world = int(os.environ.get("WORLD_SIZE", "1"))
    fsdp = os.environ.get("ACCELERATE_USE_FSDP", "false").lower() in ("1", "true", "yes")
    n_train = count_lines(a.train)
    if a.max_train_examples:
        n_train = min(n_train, a.max_train_examples)
    n_eval = min(count_lines(a.eval), a.max_eval_examples) if a.max_eval_examples else count_lines(a.eval)
    eff = a.batch * a.grad_accum * world
    steps_epoch = math.ceil(n_train / eff) if n_train else 0
    steps = a.max_steps if a.max_steps > 0 else math.ceil(steps_epoch * a.epochs)
    targets = [t.strip() for t in a.targets.split(",") if t.strip()]
    max_len = resolve_max_length(a, stats)
    p = {"model": a.model, "train": a.train, "eval": a.eval, "out": a.out, "n_train": n_train, "n_eval": n_eval,
         "data_variant": stats.get("variant"), "world_size": world, "fsdp": fsdp,
         "effective_batch": eff, "steps_per_epoch": steps_epoch, "total_steps": steps,
         "lora": {"r": a.r, "alpha": a.alpha, "dropout": a.dropout, "targets": targets},
         "max_length": max_len, "packing": a.packing, "gradient_checkpointing": not a.no_gradient_checkpointing}
    cfg = model_config(a.model)
    if cfg:
        trainable, total = lora_trainable_params(cfg, a.r, targets), total_params(cfg)
        shard = world if fsdp else 1
        tr = stats.get("train", {})
        mean_tokens = (tr.get("approx_train_tokens", 0) / tr["examples"]) if tr.get("examples") else 0
        micro_tokens = a.batch * mean_tokens
        p["lora"].update({"trainable_params": trainable, "trainable_pct": round(100.0 * trainable / total, 3)})
        p["memory_gb_per_gpu"] = {
            "base_weights_bf16": round(total * 2 / 1e9 / shard, 2),
            "lora_weights_grads_adam_fp32": round(trainable * 16 / 1e9 / shard, 2),
            # rough: the loss upcasts logits to fp32 - vocab 128K makes this the largest activation
            "logits_fp32_rough": round(micro_tokens * cfg["vocab_size"] * 4 / 1e9, 2),
            "checkpointed_hidden_states_rough": round(micro_tokens * cfg["hidden_size"] * 2 *
                                                      cfg["num_hidden_layers"] / 1e9, 2)}
        p["tokens_per_epoch_approx"] = tr.get("approx_train_tokens")
        longest = tr.get("approx_max_example_tokens") or 0
        if longest and longest * 1.3 > max_len:
            p["warning"] = (f"longest example is ~{longest:,} tokens (chars/4; real tokenizers give up to ~1.3x): "
                            f"raise --max-length above {int(longest * 1.3):,}")
    return p


def sft_config_kwargs(a, p: dict) -> dict:
    """SFTConfig arguments (TRL 1.16 + transformers 5.20 names). Kept as a dict so the dry run prints it."""
    kw = {"output_dir": str(REPO / "results" / "finetune" / "checkpoints"), "num_train_epochs": a.epochs,
          "max_steps": a.max_steps, "per_device_train_batch_size": a.batch, "per_device_eval_batch_size": a.batch,
          "gradient_accumulation_steps": a.grad_accum, "learning_rate": a.lr, "lr_scheduler_type": a.scheduler,
          "warmup_steps": a.warmup, "logging_steps": a.logging_steps, "save_strategy": "no",
          "eval_strategy": "steps" if a.eval_steps else "no", "bf16": True,
          "gradient_checkpointing": not a.no_gradient_checkpointing, "max_length": p["max_length"],
          "packing": a.packing, "completion_only_loss": True, "report_to": a.report_to, "seed": a.seed,
          # TRL loads a model given by name in float32 unless told otherwise: 32 GB instead of 16 GB for 8B.
          "model_init_kwargs": {"dtype": "bfloat16"}}
    if a.eval_steps:
        kw["eval_steps"] = a.eval_steps
    return kw


def lora_config_kwargs(a) -> dict:
    return {"r": a.r, "lora_alpha": a.alpha, "lora_dropout": a.dropout, "bias": "none", "task_type": "CAUSAL_LM",
            "target_modules": [t.strip() for t in a.targets.split(",") if t.strip()]}


def run_name(a, p: dict) -> str:
    if a.run_name:
        return a.run_name
    w = p["world_size"]
    mode = f"fsdp2-{w}gpu" if p["fsdp"] else (f"ddp-{w}gpu" if w > 1 else "1gpu")
    return f"lora-r{a.r}-{p.get('data_variant') or 'data'}-{mode}"


# ------------------------------------------------------------------------------------------------ training
def check_lengths(tok, rows: list, max_len: int, k: int = 8) -> int:
    """Tokenize the k longest examples; fail before training if any would be truncated."""
    longest = sorted(rows, key=lambda r: -sum(len(m["content"]) for m in r["prompt"] + r["completion"]))[:k]
    worst = 0
    for r in longest:
        enc = tok.apply_chat_template(r["prompt"] + r["completion"], tokenize=True, return_dict=True)
        worst = max(worst, len(enc["input_ids"]))
    if worst > max_len:
        raise SystemExit(f"an example has {worst:,} tokens > max_length {max_len:,}: its completion would be cut off. "
                         f"Re-run with --max-length {int(worst * 1.1)}")
    return worst


def make_callback():
    from transformers import TrainerCallback

    class Throughput(TrainerCallback):
        """Wall-clock tokens/s from TRL's cumulative num_tokens metric, plus step times."""

        def __init__(self):
            self.t0, self.t_step, self.steps, self.tokens = None, None, [], 0

        def on_train_begin(self, args, state, control, **kw):
            self.t0 = time.perf_counter()

        def on_step_begin(self, args, state, control, **kw):
            self.t_step = time.perf_counter()

        def on_step_end(self, args, state, control, **kw):
            if self.t_step is not None:
                self.steps.append(time.perf_counter() - self.t_step)

        def on_log(self, args, state, control, logs=None, **kw):
            if logs and "num_tokens" in logs:
                self.tokens = float(logs["num_tokens"])

    return Throughput()


def train(a, p: dict) -> dict:
    import torch
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    tok = AutoTokenizer.from_pretrained(a.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    raw = [json.loads(line) for line in Path(a.train).read_text().splitlines() if line.strip()]
    worst = check_lengths(tok, raw, p["max_length"])

    ds_train = load_dataset("json", data_files=a.train, split="train").select_columns(["prompt", "completion"])
    if a.max_train_examples:
        ds_train = ds_train.shuffle(seed=a.seed).select(range(min(a.max_train_examples, len(ds_train))))
    ds_eval = None
    if Path(a.eval).exists() and a.max_eval_examples:
        ds_eval = load_dataset("json", data_files=a.eval, split="train").select_columns(["prompt", "completion"])
        ds_eval = ds_eval.shuffle(seed=a.seed).select(range(min(a.max_eval_examples, len(ds_eval))))

    cfg = SFTConfig(**sft_config_kwargs(a, p))
    trainer = SFTTrainer(model=a.model, args=cfg, train_dataset=ds_train, eval_dataset=ds_eval,
                         processing_class=tok, peft_config=LoraConfig(**lora_config_kwargs(a)))
    cb = make_callback()
    trainer.add_callback(cb)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    result = trainer.train()
    wall = time.perf_counter() - cb.t0
    metrics = {}
    if ds_eval is not None:
        metrics = trainer.evaluate()

    if trainer.is_fsdp_enabled:
        # The FSDP2 YAML checkpoints SHARDED_STATE_DICT (fast); the adapter we serve must be one full file.
        trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")
    trainer.save_model(a.out)

    steady = cb.steps[3:] or cb.steps
    report = {"run": run_name(a, p), "plan": p, "train_loss": result.training_loss, "eval": metrics,
              "wall_s": round(wall, 1), "steps": len(cb.steps), "mean_step_s": round(sum(steady) / max(len(steady), 1), 3),
              "tokens": cb.tokens, "tokens_per_s": round(cb.tokens / wall, 1) if wall else None,
              "tokens_per_s_per_gpu": round(cb.tokens / wall / p["world_size"], 1) if wall else None,
              "peak_mem_gib_rank0": round(torch.cuda.max_memory_allocated() / 2**30, 2) if torch.cuda.is_available()
              else None, "longest_example_tokens": worst, "adapter": a.out}
    if trainer.is_world_process_zero():
        out = REPO / "results" / "finetune" / f"{report['run']}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, default=str) + "\n")
        print(f"\n{report['run']}: {report['tokens_per_s']} tokens/s ({report['tokens_per_s_per_gpu']} per GPU), "
              f"peak {report['peak_mem_gib_rank0']} GiB on rank 0, {report['mean_step_s']} s/step, "
              f"train loss {report['train_loss']:.4f} -> adapter {a.out} | report {out}")
    return report


def main(argv=None) -> int:
    a = parse_args(argv)
    p = plan(a)
    if a.dry_run:
        print(json.dumps({"plan": p, "sft_config": sft_config_kwargs(a, p), "lora_config": lora_config_kwargs(a),
                          "run_name": run_name(a, p)}, indent=2))
        if not p["n_train"]:
            print(f"\nno training data at {a.train}: run python3 finetune/make_sft_data.py first", file=sys.stderr)
        return 0
    if not p["n_train"]:
        print(f"no training data at {a.train}: run python3 finetune/make_sft_data.py first", file=sys.stderr)
        return 2
    train(a, p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
