#!/usr/bin/env python3
"""M8 step 2 - LoRA fine-tune of Llama 3.1 8B Instruct on the distilled dataset, tracked in MLflow.

Runs in nvcr.io/nvidia/pytorch:<NGC_RELEASE>-py3 (mlops/k8s/train-lora-job.yaml) on a MIG 3g.40gb slice of the
AKS A100 - the other 3g.40gb slice keeps serving (🔁 Lab 02 MIG; bench/noisy_neighbor.sh measures the impact).
Output: <OUT_DIR>/<ADAPTER_NAME>/{adapter_config.json, adapter_model.safetensors} in Hugging Face PEFT format -
exactly what NIM loads from NIM_PEFT_SOURCE/<name>/ (the folder name becomes the model name).
Loss is computed on the assistant JSON only (prompt tokens masked): we teach the OUTPUT style, not the prompt.
"""
import json
import math
import os
import time
from pathlib import Path

BASE = os.environ.get("BASE_MODEL", "meta-llama/Llama-3.1-8B-Instruct")
DATA = Path(os.environ.get("DATA_DIR", "/data/mlops"))
OUT = Path(os.environ.get("OUT_DIR", "/data/adapters"))
NAME = os.environ.get("ADAPTER_NAME", "plover-style-v1")
CFG = {"rank": int(os.environ.get("LORA_RANK", "16")), "alpha": int(os.environ.get("LORA_ALPHA", "32")),
       "dropout": float(os.environ.get("LORA_DROPOUT", "0.05")), "lr": float(os.environ.get("LR", "2e-4")),
       "epochs": int(os.environ.get("EPOCHS", "2")), "batch": int(os.environ.get("BATCH", "4")),
       "grad_accum": int(os.environ.get("GRAD_ACCUM", "4")), "max_len": int(os.environ.get("MAX_LEN", "1024")),
       "targets": os.environ.get("LORA_TARGETS", "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj").split(",")}


def load(path):
    return [json.loads(line) for line in open(path) if line.strip()]


def encode(tok, row):
    """input_ids for the full chat; labels = -100 on everything except the assistant answer."""
    msgs = row["messages"]
    prompt_ids = tok.apply_chat_template(msgs[:-1], add_generation_prompt=True, tokenize=True)
    full_ids = tok.apply_chat_template(msgs, tokenize=True)
    if hasattr(prompt_ids, "input_ids"):          # newer tokenizers may return a BatchEncoding
        prompt_ids, full_ids = prompt_ids["input_ids"], full_ids["input_ids"]
    full_ids = full_ids[: CFG["max_len"]]
    labels = [-100] * min(len(prompt_ids), len(full_ids)) + full_ids[len(prompt_ids):]
    return {"input_ids": full_ids, "labels": labels[: len(full_ids)]}


def collate(batch, pad_id):
    import torch
    n = max(len(b["input_ids"]) for b in batch)
    ids = torch.full((len(batch), n), pad_id, dtype=torch.long)
    lab = torch.full((len(batch), n), -100, dtype=torch.long)
    att = torch.zeros((len(batch), n), dtype=torch.long)
    for i, b in enumerate(batch):
        ids[i, : len(b["input_ids"])] = torch.tensor(b["input_ids"])
        lab[i, : len(b["labels"])] = torch.tensor(b["labels"])
        att[i, : len(b["input_ids"])] = 1
    return {"input_ids": ids, "labels": lab, "attention_mask": att}


def eval_loss(model, loader, dev):
    import torch
    model.eval()
    tot, n = 0.0, 0
    with torch.no_grad():
        for b in loader:
            b = {k: v.to(dev) for k, v in b.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                tot += model(**b).loss.item()
            n += 1
    model.train()
    return tot / max(n, 1)


def main():
    import mlflow
    import torch
    from peft import LoraConfig, get_peft_model
    from torch.utils.data import DataLoader
    from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
    dev = "cuda"
    torch.manual_seed(7)
    tok = AutoTokenizer.from_pretrained(BASE)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    train = [encode(tok, r) for r in load(DATA / "train.jsonl")]
    val = [encode(tok, r) for r in load(DATA / "val.jsonl")]
    model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16).to(dev)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    lcfg = LoraConfig(r=CFG["rank"], lora_alpha=CFG["alpha"], lora_dropout=CFG["dropout"], target_modules=CFG["targets"],
                      bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(model, lcfg)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    tl = DataLoader(train, batch_size=CFG["batch"], shuffle=True, collate_fn=lambda b: collate(b, pad_id))
    vl = DataLoader(val, batch_size=CFG["batch"], shuffle=False, collate_fn=lambda b: collate(b, pad_id))
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=CFG["lr"], weight_decay=0.0)
    steps = math.ceil(len(tl) / CFG["grad_accum"]) * CFG["epochs"]
    sched = get_cosine_schedule_with_warmup(opt, max(1, int(0.05 * steps)), steps)
    mlflow.set_experiment(os.environ.get("MLFLOW_EXPERIMENT", "plover-style"))
    with mlflow.start_run(run_name=NAME) as run:
        mlflow.log_params({**{k: (",".join(v) if isinstance(v, list) else v) for k, v in CFG.items()},
                           "base_model": BASE, "adapter_name": NAME, "train_rows": len(train), "val_rows": len(val),
                           "trainable_params": trainable, "gpu": torch.cuda.get_device_name(0)})
        mlflow.log_metric("val_loss", eval_loss(model, vl, dev), step=0)
        step, t0 = 0, time.time()
        model.train()
        for epoch in range(CFG["epochs"]):
            for i, b in enumerate(tl):
                b = {k: v.to(dev) for k, v in b.items()}
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = model(**b).loss / CFG["grad_accum"]
                loss.backward()
                if (i + 1) % CFG["grad_accum"] == 0 or i + 1 == len(tl):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                    sched.step()
                    opt.zero_grad(set_to_none=True)
                    step += 1
                    mlflow.log_metric("train_loss", loss.item() * CFG["grad_accum"], step=step)
            v = eval_loss(model, vl, dev)
            mlflow.log_metric("val_loss", v, step=step)
            print(f"epoch {epoch + 1}/{CFG['epochs']} step {step}/{steps} val_loss={v:.4f} elapsed={time.time() - t0:.0f}s", flush=True)
        dest = OUT / NAME
        dest.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(dest)            # adapter_config.json + adapter_model.safetensors (PEFT format)
        mlflow.log_metric("train_minutes", (time.time() - t0) / 60)
        mlflow.log_artifacts(str(dest), artifact_path="adapter")
        (dest / "mlflow_run_id").write_text(run.info.run_id)
        print(f"saved {dest} (MLflow run {run.info.run_id}); files: {sorted(p.name for p in dest.iterdir())}")


if __name__ == "__main__":
    main()
