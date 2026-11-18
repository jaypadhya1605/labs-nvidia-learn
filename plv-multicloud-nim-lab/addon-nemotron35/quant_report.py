#!/usr/bin/env python3
"""Add-on A1: what does "NVFP4" or "2-bit" ACTUALLY mean for this checkpoint? Read quantization_config, not the blog.
  python3 addon-nemotron35/quant_report.py nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB          # HF repo (config + sizes)
  python3 addon-nemotron35/quant_report.py path/to/config.json
Prints: quantization method/producer, KV-cache scheme, layers per module type per format, what stays high precision,
and (for HF repos) the real checkpoint size - the number that decides which GPU it fits.
"""
import collections
import json
import os
import sys
import urllib.request


def fetch(url):
    req = urllib.request.Request(url)
    if os.environ.get("HF_TOKEN"):
        req.add_header("Authorization", f"Bearer {os.environ['HF_TOKEN']}")
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)


def module_kind(name):
    if ".experts." in name:
        return "routed experts"
    if "shared" in name:
        return "shared experts"
    leaf = name.rsplit(".", 1)[-1]
    return {"in_proj": "mamba in_proj", "out_proj": "mamba out_proj", "q_proj": "attention", "k_proj": "attention",
            "v_proj": "attention", "o_proj": "attention", "lm_head": "lm_head"}.get(leaf, leaf)


def report(cfg, size_gb=None):
    q = cfg.get("quantization_config") or (cfg.get("llm_config") or {}).get("quantization_config")
    if not q:
        print("No quantization_config: a full-precision checkpoint (BF16/FP16).")
    else:
        print(f"method: {q.get('quant_method')}  algo: {q.get('quant_algo')}  producer: {q.get('producer')}")
        print(f"KV cache scheme: {q.get('kv_cache_scheme') or 'not quantized (BF16 KV)'}")
        table = collections.defaultdict(collections.Counter)
        for name, v in (q.get("quantized_layers") or {}).items():
            algo = v.get("quant_algo", "?") + (f" ({v['effective_bits']} bits)" if v.get("effective_bits") else "")
            table[module_kind(name)][algo] += 1
        print(f"\n{'module':<18}format (layers)")
        for kind, algos in sorted(table.items(), key=lambda kv: -sum(kv[1].values())):
            print(f"{kind:<18}" + ", ".join(f"{a} x{n}" for a, n in algos.most_common()))
        kept = collections.Counter(module_kind(n) for n in q.get("ignore") or [])
        if kept:
            print("\nkept in high precision: " + ", ".join(f"{k} x{n}" for k, n in kept.most_common(8)))
    if size_gb:
        print(f"\ncheckpoint weights on disk: {size_gb:.2f} GB  <- compare with GPU memory x 0.9 minus runtime overhead")


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else "nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB"
    if os.path.exists(src):
        report(json.load(open(src)))
        return
    cfg = fetch(f"https://huggingface.co/{src}/resolve/main/config.json")
    try:
        tree = fetch(f"https://huggingface.co/api/models/{src}/tree/main")
        size = sum(f.get("size") or 0 for f in tree if f["path"].endswith((".safetensors", ".gguf"))) / 1e9
    except Exception:  # noqa: BLE001
        size = None
    report(cfg, size)


if __name__ == "__main__":
    main()
