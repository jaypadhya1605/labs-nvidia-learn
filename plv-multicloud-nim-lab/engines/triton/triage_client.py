#!/usr/bin/env python3
"""Classify one prior-auth note with the TensorRT triage model on Triton (run inside the bench toolbox).
Usage (bench toolbox, pip install tritonclient[http] numpy): python3 triage_client.py "free text of a request note"
"""
import os
import sys
import time

import numpy as np
import tritonclient.http as httpclient
from transformers import AutoTokenizer

URL = os.environ.get("TRITON_URL", "triage-triton.plv-serving.svc:8000")
TOKENIZER = os.environ.get("TRIAGE_TOKENIZER", "distilbert/distilbert-base-uncased")
text = " ".join(sys.argv[1:]) or (
    "68-year-old with new left-sided weakness since a fall two days ago; CT negative for bleed; "
    "MRI brain requested to rule out stroke before hospital discharge tomorrow."
)
tok = AutoTokenizer.from_pretrained(TOKENIZER)
enc = tok([text], padding="max_length", truncation=True, max_length=128, return_tensors="np")
ids = enc["input_ids"].astype(np.int32)
mask = enc["attention_mask"].astype(np.int32)
client = httpclient.InferenceServerClient(URL)
inputs = [httpclient.InferInput("input_ids", list(ids.shape), "INT32"),
          httpclient.InferInput("attention_mask", list(mask.shape), "INT32")]
inputs[0].set_data_from_numpy(ids)
inputs[1].set_data_from_numpy(mask)
t0 = time.perf_counter()
res = client.infer("triage_trt", inputs, outputs=[httpclient.InferRequestedOutput("logits")])
ms = (time.perf_counter() - t0) * 1e3
logits = res.as_numpy("logits")[0]
p = np.exp(logits - logits.max())
p /= p.sum()
verdict = "EXPEDITED REVIEW" if p[1] >= 0.5 else "normal queue"
print(f"P(urgent)={p[1]:.3f} -> {verdict}   ({ms:.1f} ms round trip)")
