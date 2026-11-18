#!/usr/bin/env python3
"""Size ANY new model from its Hugging Face config.json in minutes - the day-0 skill (add-on: Nemotron 3.5).

Reads dense, MoE and hybrid Mamba-Transformer configs (Llama, Qwen, NemotronH incl. VLM wrappers and Puzzle
variants) and prints the selector's model spec:
  attention layers (only these grow a KV cache) -> KV bytes per token
  Mamba layers -> fixed SSM + conv state per sequence (dtype from mamba_ssm_cache_dtype; fp32 is common)
  experts -> routed experts and experts per token (per-layer average for Puzzle/NAS models)
Params total/active come from the model card or the checkpoint size - pass them with --total-b/--active-b.
  python3 selector/model_sizer.py --config tests/fixtures/nemotron-3.5-lightning-30b-a3b.config.json \
      --name nemotron-3.5-lightning --total-b 32.9 --active-b 3
  python3 selector/model_sizer.py --config nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16 ...   # HF repo id
"""
import argparse
import json
import os
import urllib.request

import yaml

TEXT_KEYS = ("text_config", "llm_config", "language_config", "language_model_config")
LAYER_KEYS = ("num_hidden_layers", "hybrid_override_pattern", "layers_block_type", "block_configs")
DTYPE_BYTES = {"float32": 4, "fp32": 4, "bfloat16": 2, "bf16": 2, "float16": 2, "fp16": 2, "auto": 2, None: 2}


def load_config(src):
    if os.path.exists(src):
        with open(src) as f:
            return json.load(f)
    url = f"https://huggingface.co/{src}/resolve/main/config.json"
    req = urllib.request.Request(url)
    if os.environ.get("HF_TOKEN"):
        req.add_header("Authorization", f"Bearer {os.environ['HF_TOKEN']}")
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def text_config(cfg):
    """VLM configs nest the language model; prefer the nested block when it describes layers."""
    for k in TEXT_KEYS:
        sub = cfg.get(k)
        if isinstance(sub, dict) and any(x in sub for x in LAYER_KEYS):
            return sub
    return cfg


def first(cfg, *keys, default=None):
    for k in keys:
        if cfg.get(k) is not None:
            return cfg[k]
    return default


def layer_counts(t):
    """(attention, mamba, total, how) from whichever layer description the config uses."""
    if isinstance(t.get("layers_block_type"), list):            # NemotronH (2026): ["mamba", "moe", "attention", ...]
        lbt = t["layers_block_type"]
        return lbt.count("attention"), lbt.count("mamba"), len(lbt), "layers_block_type"
    if isinstance(t.get("block_configs"), list):                 # Puzzle / NAS variants
        types = [b.get("block_type") for b in t["block_configs"]]
        return types.count("attention"), types.count("mamba"), len(types), "block_configs"
    if t.get("hybrid_override_pattern"):                         # NemotronH (2025): "M-M*-..." M=Mamba *=attention
        p = t["hybrid_override_pattern"]
        return p.count("*"), p.count("M"), len(p), "hybrid_override_pattern"
    if isinstance(t.get("layer_types"), list):                   # Qwen-style hybrids
        lt = [str(x) for x in t["layer_types"]]
        attn = sum(1 for x in lt if "attention" in x and "linear" not in x)
        return attn, len(lt) - attn, len(lt), "layer_types"
    n = first(t, "num_hidden_layers", "n_layer", "num_layers")
    return n, 0, n, "num_hidden_layers (dense)"


def experts(t):
    """Routed experts and experts per token; per-layer average when a NAS model varies them by layer."""
    if isinstance(t.get("block_configs"), list):
        moe = [b for b in t["block_configs"] if b.get("block_type") == "moe" and b.get("n_routed_experts")]
        if moe:
            e = sum(b["n_routed_experts"] for b in moe) / len(moe)
            k = sum(b.get("num_experts_per_tok", 0) for b in moe) / len(moe)
            return round(e, 1), round(k, 2), f"per-layer average over {len(moe)} MoE blocks"
    e = first(t, "n_routed_experts", "num_local_experts", "num_experts")
    k = first(t, "num_experts_per_tok", "moe_topk", "num_experts_per_token")
    return e, k, ""


def size(cfg):
    t = text_config(cfg)
    attn, mamba, total, how = layer_counts(t)
    heads = first(t, "num_attention_heads", "n_head")
    kv_heads = first(t, "num_key_value_heads", "num_kv_heads", default=heads)
    hidden = first(t, "hidden_size", "d_model")
    head_dim = first(t, "head_dim", "attention_head_dim", default=(hidden // heads if hidden and heads else None))
    notes = [f"{total} layers: {attn} attention + {mamba} Mamba (from {how})"]
    m_heads = first(t, "mamba_num_heads", "ssm_num_heads")
    m_head_dim = first(t, "mamba_head_dim", "ssm_head_dim")
    d_state = first(t, "ssm_state_size", "mamba_d_state", "state_size")
    state_mb = 0.0
    if mamba and m_heads and m_head_dim and d_state:
        ssm_dtype = t.get("mamba_ssm_cache_dtype")
        sb = DTYPE_BYTES.get(ssm_dtype, 2)
        d_inner = m_heads * m_head_dim
        conv = (d_inner + 2 * (t.get("n_groups") or 1) * d_state) * ((t.get("conv_kernel") or 4) - 1) * 2
        state_mb = mamba * (m_heads * m_head_dim * d_state * sb + conv) / 1e6
        notes.append(f"SSM state {ssm_dtype or 'bf16 (assumed)'}: {state_mb:.1f} MB per sequence, read+written every decode step")
    e, k, e_note = experts(t)
    if e_note:
        notes.append(f"experts: {e_note}")
    kv_per_tok = 2 * (attn or 0) * (kv_heads or 0) * (head_dim or 0) * 2
    ctx = first(t, "max_position_embeddings")
    if ctx:
        notes.append(f"max_position_embeddings={ctx:,} (the context the CONFIG supports - check what the NIM serves)")
    return {"kv_layers": attn, "kv_heads": kv_heads, "head_dim": head_dim, "mamba_layers": mamba,
            "mamba_state_mb_per_seq": round(state_mb, 1), "moe_experts": e, "moe_active_experts": k,
            "kv_bytes_per_token_bf16": kv_per_tok, "max_position_embeddings": ctx, "notes": notes}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="local config.json or Hugging Face repo id")
    ap.add_argument("--name", default="new-model")
    ap.add_argument("--total-b", type=float, required=True)
    ap.add_argument("--active-b", type=float, required=True)
    a = ap.parse_args()
    s = size(load_config(a.config))
    spec = {a.name: {"params_total_b": a.total_b, "params_active_b": a.active_b, "kv_layers": s["kv_layers"],
                     "kv_heads": s["kv_heads"], "head_dim": s["head_dim"], "kv_bytes": 2,
                     "mamba_state_mb_per_seq": s["mamba_state_mb_per_seq"], "moe_experts": s["moe_experts"],
                     "moe_active_experts": s["moe_active_experts"], "nim_verified_gpus": "unknown", "source": a.config}}
    print(f"# KV cache: {s['kv_bytes_per_token_bf16'] / 1024:.1f} KiB per token (bf16); Llama 3.1 8B is 128 KiB")
    for n in s["notes"]:
        print(f"# {n}")
    print(yaml.safe_dump(spec, sort_keys=False))


if __name__ == "__main__":
    main()
