#!/usr/bin/env python3
"""Fine-tune DistilBERT for prior-auth urgency triage (PyTorch), export ONNX, record a PyTorch eager baseline.

Runs in nvcr.io/nvidia/pytorch:<NGC_RELEASE>-py3 on a 1g MIG slice (workloads/triage/triage-jobs.yaml).
Outputs under WORK_DIR: hf/ (checkpoint), onnx/model.onnx, metrics.json, pytorch_baseline.json, perf_data.json
"""
import copy
import json
import os
import random
import statistics
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

WORK = Path(os.environ.get("WORK_DIR", "/models/triage"))
BASE = os.environ.get("BASE_MODEL", "distilbert-base-uncased")
EPOCHS = int(os.environ.get("EPOCHS", "3"))
BATCH = int(os.environ.get("BATCH", "32"))
MAX_LEN = int(os.environ.get("MAX_LEN", "256"))
LR = float(os.environ.get("LR", "5e-5"))
LABELS = {"STANDARD": 0, "URGENT": 1}
SEED = 7


def load_rows():
    rows = []
    with open(WORK / "data" / "synthetic.jsonl") as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("label") in LABELS and len(r.get("text", "")) > 40:
                rows.append((r["text"], LABELS[r["label"]]))
    random.Random(SEED).shuffle(rows)
    return rows


def make_loader(rows, tok, batch_size, shuffle):
    def collate(batch):
        enc = tok([t for t, _ in batch], padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt")
        enc["labels"] = torch.tensor([y for _, y in batch])
        return enc
    return DataLoader(rows, batch_size=batch_size, shuffle=shuffle, collate_fn=collate, num_workers=0)


@torch.no_grad()
def evaluate(model, loader, dev):
    model.eval()
    tp = fp = fn = correct = n = 0
    for b in loader:
        b = {k: v.to(dev) for k, v in b.items()}
        pred = model(input_ids=b["input_ids"], attention_mask=b["attention_mask"]).logits.argmax(-1)
        y = b["labels"]
        correct += (pred == y).sum().item()
        n += y.numel()
        tp += ((pred == 1) & (y == 1)).sum().item()
        fp += ((pred == 1) & (y == 0)).sum().item()
        fn += ((pred == 0) & (y == 1)).sum().item()
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return {"n": n, "accuracy": round(correct / max(n, 1), 4), "urgent_precision": round(precision, 4),
            "urgent_recall": round(recall, 4), "urgent_f1": round(2 * precision * recall / max(precision + recall, 1e-9), 4)}


class Int32Wrapper(torch.nn.Module):
    """TensorRT/Triton-friendly signature: int32 inputs -> logits."""
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, input_ids, attention_mask):
        return self.m(input_ids=input_ids.long(), attention_mask=attention_mask.long()).logits


def export_onnx(model, out_path: Path):
    wrapper = Int32Wrapper(copy.deepcopy(model).float().cpu().eval()).eval()
    ids = torch.ones(2, 128, dtype=torch.int32)
    mask = torch.ones(2, 128, dtype=torch.int32)
    common = dict(input_names=["input_ids", "attention_mask"], output_names=["logits"],
                  dynamic_axes={"input_ids": {0: "batch", 1: "seq"}, "attention_mask": {0: "batch", 1: "seq"},
                                "logits": {0: "batch"}})
    try:
        torch.onnx.export(wrapper, (ids, mask), str(out_path), dynamo=False, opset_version=17, **common)
        print("ONNX export: TorchScript exporter, opset 17")
    except Exception as exc:  # noqa: BLE001 - newer PyTorch may drop the TorchScript exporter
        print(f"TorchScript exporter failed ({exc}); retrying with the dynamo exporter")
        torch.onnx.export(wrapper, (ids, mask), str(out_path), dynamo=True, **common)


@torch.no_grad()
def bench_eager(model, dev, dtype):
    m = copy.deepcopy(model).to(dev, dtype=dtype).eval()
    results = {}
    for bs in (1, 8, 32):
        ids = torch.randint(1000, 5000, (bs, 128), device=dev)
        mask = torch.ones_like(ids)
        for _ in range(10):
            m(input_ids=ids, attention_mask=mask)
        torch.cuda.synchronize()
        lat = []
        for _ in range(50):
            t = time.perf_counter()
            m(input_ids=ids, attention_mask=mask)
            torch.cuda.synchronize()
            lat.append((time.perf_counter() - t) * 1e3)
        lat.sort()
        results[f"bs{bs}_seq128"] = {"p50_ms": round(lat[len(lat) // 2], 3), "p95_ms": round(lat[int(len(lat) * 0.95) - 1], 3),
                                     "seq_per_s": round(bs * 1000 / statistics.mean(lat), 1)}
    return results


def main():
    torch.manual_seed(SEED)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    device_name = torch.cuda.get_device_name(0) if dev == "cuda" else "cpu"
    print(f"device: {device_name}  torch {torch.__version__}")
    rows = load_rows()
    split = int(len(rows) * 0.85)
    train, val = rows[:split], rows[split:]
    print(f"train={len(train)} val={len(val)} urgent_share={sum(y for _, y in rows) / max(len(rows), 1):.2f}")

    tok = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForSequenceClassification.from_pretrained(BASE, num_labels=2).to(dev)
    train_loader = make_loader(train, tok, BATCH, True)
    val_loader = make_loader(val, tok, 64, False)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    total_steps = EPOCHS * len(train_loader)
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * total_steps), total_steps)

    for epoch in range(EPOCHS):
        model.train()
        t0, running = time.time(), 0.0
        for b in train_loader:
            b = {k: v.to(dev) for k, v in b.items()}
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(dev == "cuda")):
                out = model(**b)
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            running += out.loss.item()
        print(f"epoch {epoch + 1}/{EPOCHS} loss={running / len(train_loader):.4f} "
              f"time={time.time() - t0:.1f}s val={evaluate(model, val_loader, dev)}")

    metrics = evaluate(model, val_loader, dev)
    metrics["device"] = device_name
    (WORK / "hf").mkdir(parents=True, exist_ok=True)
    model.save_pretrained(WORK / "hf")
    tok.save_pretrained(WORK / "hf")

    (WORK / "onnx").mkdir(parents=True, exist_ok=True)
    export_onnx(model, WORK / "onnx" / "model.onnx")
    import onnx  # noqa: PLC0415
    onnx.checker.check_model(str(WORK / "onnx" / "model.onnx"))
    print("ONNX model passes onnx.checker")

    if dev == "cuda":
        baseline = {"device": device_name, "fp32": bench_eager(model, dev, torch.float32),
                    "fp16": bench_eager(model, dev, torch.float16)}
        (WORK / "pytorch_baseline.json").write_text(json.dumps(baseline, indent=2))
        print("PyTorch eager baseline:", json.dumps(baseline))

    # Real tokenized samples for Triton perf_analyzer (shape excludes the batch dimension).
    enc = tok([t for t, _ in val[:64]], padding="max_length", truncation=True, max_length=128)
    perf = {"data": [{"input_ids": {"content": enc["input_ids"][i], "shape": [128]},
                      "attention_mask": {"content": enc["attention_mask"][i], "shape": [128]}}
                     for i in range(len(enc["input_ids"]))]}
    (WORK / "perf_data.json").write_text(json.dumps(perf))
    (WORK / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print("validation metrics:", metrics)


if __name__ == "__main__":
    main()
