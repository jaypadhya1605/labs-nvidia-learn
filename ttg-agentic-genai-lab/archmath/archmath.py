#!/usr/bin/env python3
"""M2 - what does a model ARCHITECTURE cost to serve? (🔁 Lab 01 roofline, Lab 02 KV math, Lab 04 model sizer)

Reads Hugging Face config.json files and computes, from first principles (no GPU, no torch):
  params      total and ACTIVE per token (an MoE routes each token to k of E experts)
  weights     GB at BF16 / FP8 / NVFP4 (capacity) and bytes STREAMED per decode step (what bandwidth pays for)
  KV cache    bytes per token: MHA/GQA, MLA (DeepSeek latent), hybrid Mamba (attention layers only + fixed SSM state)
  FLOPs       per decoded token = 2 x active matmul params + attention that grows with context
  roofline    decode step time on a GPU = max(bytes / bandwidth, FLOPs / peak) -> memory- or compute-bound
  attention   share of prefill and decode FLOPs that grows with context (8K / 32K / 128K)
  coverage    unique MoE experts touched per decode step at batch b (why MoE decode stops being "3B cheap")

Usage (from the repo root):
  python3 -m archmath.archmath                                    # every bundled config: summary table
  python3 -m archmath.archmath --roofline --gpu H100-SXM --batch 1,8,32 --ctx 4096
  python3 -m archmath.archmath --roofline --model llama-3.1-70b --precision fp8 --gpus 2 --batch 1   # 🔁 mock round Q1
  python3 -m archmath.archmath --coverage --batch 1,8,32,64,256  # MoE expert coverage vs batch
  python3 -m archmath.archmath --attention-share                  # quadratic share at 8K / 32K / 128K
  python3 -m archmath.archmath --prefill 3000 --gpu H100-SXM      # TTFT floor for a 3,000-token agent prompt
  python3 -m archmath.archmath --config my-config.json --json     # any local config.json (or an HF repo id)
  python3 -m archmath.archmath --notes                            # the assumptions behind every number
Every number is an estimate from the architecture. Measure on the GPU (M4/M5) and write your efficiency next to it.
"""
import argparse
import glob
import json
import os
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG_DIR = os.path.join(HERE, "configs")

# Dense (non-sparse) tensor-core TFLOPS and peak memory bandwidth. Same numbers as Lab 04 selector/gpus.yaml.
# B200 row: derived from NVIDIA's GB200 NVL72 page (2026-10-17): one GB200 superchip = 1 Grace + 2 Blackwell GPUs,
# 372 GB HBM3E, 16 TB/s, BF16 10 / FP8 20 / NVFP4 40 PFLOPS sparse ("dense is one-half sparse"). Per GPU = half a
# superchip. HGX B200 boards run at lower power - check that datasheet before quoting a B200 number.
GPUS = {
    "H100-SXM": {"mem_gb": 80, "bw_tbs": 3.35, "bf16_tflops": 989, "fp8_tflops": 1979, "fp4_tflops": None},
    "H100-NVL": {"mem_gb": 94, "bw_tbs": 3.9, "bf16_tflops": 835, "fp8_tflops": 1671, "fp4_tflops": None},
    "H100-PCIe": {"mem_gb": 80, "bw_tbs": 2.0, "bf16_tflops": 756, "fp8_tflops": 1513, "fp4_tflops": None},
    "H200-SXM": {"mem_gb": 141, "bw_tbs": 4.8, "bf16_tflops": 989, "fp8_tflops": 1979, "fp4_tflops": None},
    "A100-80GB-SXM": {"mem_gb": 80, "bw_tbs": 2.039, "bf16_tflops": 312, "fp8_tflops": None, "fp4_tflops": None},
    "L40S": {"mem_gb": 48, "bw_tbs": 0.864, "bf16_tflops": 362, "fp8_tflops": 733, "fp4_tflops": None},
    "B200-in-GB200": {"mem_gb": 186, "bw_tbs": 8.0, "bf16_tflops": 2500, "fp8_tflops": 5000, "fp4_tflops": 10000},
}

# Bytes per parameter. NVFP4 = 4-bit values + one FP8 (E4M3) scale per 16 values -> 0.5 + 1/16 = 0.5625.
PRECISION_BYTES = {"bf16": 2.0, "fp8": 1.0, "nvfp4": 0.5625}
DTYPE_BYTES = {"float32": 4, "fp32": 4, "bfloat16": 2, "bf16": 2, "float16": 2, "fp16": 2, None: 2}
TEXT_KEYS = ("text_config", "llm_config", "language_config", "language_model_config")
LAYER_KEYS = ("num_hidden_layers", "hybrid_override_pattern", "layers_block_type", "block_configs")

NOTES = [
    "Weights: block linears are quantized at the chosen precision; embeddings, lm_head, norms, routers and the vision "
    "tower stay BF16 (what FP8/NVFP4 checkpoints usually do - check the checkpoint's quantization_config).",
    "Decode bytes per step = every weight that is read (the input-embedding table is GATHERED, b rows, so it is not "
    "streamed) + the KV cache of every sequence in the batch (+ Mamba state read AND written for hybrids).",
    "MoE: active params = shared + top-k routed experts + attention + lm_head + embeddings (the convention model "
    "cards use). Expert coverage assumes uniform, independent routing: U(b) = E x (1 - (1 - k/E)^b). Real routers "
    "are skewed (hot experts), so measured coverage is lower.",
    "MLA (DeepSeek): the KV cache holds one compressed latent (kv_lora_rank) + one RoPE key (qk_rope_head_dim) per "
    "layer and token. Decode attention FLOPs use the absorbed form (FlashMLA): heads x (latent + rope + latent) x ctx.",
    "Hybrid Mamba (NemotronH): only attention layers grow a KV cache; each Mamba-2 layer keeps a fixed SSM + conv "
    "state per sequence (fp32 when mamba_ssm_cache_dtype says so). Same math as Lab 04 selector/model_sizer.py.",
    "Roofline: step time = max(bytes / bandwidth, FLOPs / dense peak). Real kernels reach ~80-90% of bandwidth and "
    "50-70% of peak; --gpus N divides bytes and FLOPs evenly (ideal tensor parallelism, no all-reduce - the cost of "
    "the all-reduce is parallel/parallelism_calc.py's job).",
    "Prefill: the lm_head runs for the last token only; causal attention costs heads x (d_qk + d_v) x L^2 per layer.",
]


# ----------------------------------------------------------------------------------------------- config loading
def load_config(src):
    """Bundled name (llama-3.1-8b), local path, or Hugging Face repo id (HF_TOKEN for gated repos)."""
    bundled = os.path.join(CONFIG_DIR, f"{src}.json")
    if os.path.exists(bundled):
        src = bundled
    if os.path.exists(src):
        with open(src) as f:
            cfg = json.load(f)
        cfg.setdefault("_name", os.path.splitext(os.path.basename(src))[0])
        return cfg
    url = f"https://huggingface.co/{src}/resolve/main/config.json"
    req = urllib.request.Request(url)
    if os.environ.get("HF_TOKEN"):
        req.add_header("Authorization", f"Bearer {os.environ['HF_TOKEN']}")
    with urllib.request.urlopen(req, timeout=60) as r:
        cfg = json.load(r)
    cfg.setdefault("_name", src.split("/")[-1].lower())
    cfg.setdefault("_source", src)
    return cfg


def bundled_names():
    return sorted(os.path.splitext(os.path.basename(p))[0] for p in glob.glob(os.path.join(CONFIG_DIR, "*.json")))


def text_config(cfg):
    """VLM configs nest the language model; prefer the nested block when it describes layers (🔁 Lab 04)."""
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


# ----------------------------------------------------------------------------------------------- architecture
def layer_plan(t):
    """One list of sublayers per layer: attn, xattn (cross-attention), mla, mlp, moe, mamba."""
    mla = t.get("kv_lora_rank") is not None
    att = "mla" if mla else "attn"
    if isinstance(t.get("layers_block_type"), list):                 # NemotronH: one block per layer
        kinds = {"mamba": ["mamba"], "moe": ["moe"], "attention": [att], "mlp": ["mlp"]}
        return [kinds[b] for b in t["layers_block_type"]]
    if t.get("hybrid_override_pattern"):                              # NemotronH 2025: M=Mamba *=attention -=MLP E=MoE
        kinds = {"M": ["mamba"], "*": [att], "-": ["mlp"], "E": ["moe"]}
        return [kinds[c] for c in t["hybrid_override_pattern"]]
    n = first(t, "num_hidden_layers", "n_layer", "num_layers")
    experts = first(t, "n_routed_experts", "num_local_experts", "num_experts", default=0) or 0
    cross = set(t.get("cross_attention_layers") or [])
    plan = []
    for i in range(n):
        a = "xattn" if i in cross else att
        moe = False
        if experts:
            if t.get("first_k_dense_replace") is not None:                # DeepSeek
                moe = i >= t["first_k_dense_replace"] and i % (t.get("moe_layer_freq") or 1) == 0
            elif t.get("decoder_sparse_step") is not None:                # Qwen MoE
                moe = i not in (t.get("mlp_only_layers") or []) and (i + 1) % t["decoder_sparse_step"] == 0
            else:                                                         # Mixtral: every layer
                moe = True
        plan.append([a, "moe" if moe else "mlp"])
    return plan


def mtp_plan(t):
    """Multi-token-prediction blocks (speculative heads) - counted separately, like model cards do."""
    if isinstance(t.get("mtp_layers_block_type"), list):
        kinds = {"attention": "attn", "moe": "moe", "mamba": "mamba", "mlp": "mlp"}
        return [[kinds[b] for b in t["mtp_layers_block_type"]]]
    n = t.get("num_nextn_predict_layers") or 0
    if n and t.get("kv_lora_rank") is not None:
        return [["mla", "moe"]] * n
    return []


def gated(t):
    act = str(first(t, "mlp_hidden_act", "hidden_act", default="silu")).lower()
    return act not in ("relu2", "gelu", "gelu_new", "relu")                  # relu2 (NemotronH) = up + down only


def dims(t):
    h = first(t, "hidden_size", "d_model")
    nh = first(t, "num_attention_heads", "n_head")
    kv = first(t, "num_key_value_heads", "num_kv_heads", default=nh)
    hd = first(t, "head_dim", default=h // nh)
    return h, nh, kv, hd


def mamba_dims(t, h):
    nh = first(t, "mamba_num_heads", "ssm_num_heads")
    hdim = first(t, "mamba_head_dim", "ssm_head_dim", default=64)
    d_inner = nh * hdim if nh else int((t.get("expand") or 2) * h)
    nh = nh or d_inner // hdim
    return {"d_inner": d_inner, "heads": nh, "head_dim": hdim, "groups": t.get("n_groups") or 1,
            "d_state": first(t, "ssm_state_size", "mamba_d_state", "state_size", default=128),
            "kernel": t.get("conv_kernel") or 4, "conv_bias": bool(t.get("use_conv_bias", True))}


def sublayer_params(kind, t, cfg):
    """(total, active) parameters of one sublayer."""
    h, nh, kv, hd = dims(t)
    if kind == "attn":
        p = h * nh * hd + 2 * h * kv * hd + nh * hd * h
        if cfg.get("_qkv_bias") or t.get("attention_bias"):
            p += nh * hd + 2 * kv * hd
        return p, p
    if kind == "xattn":                                       # Mllama cross-attention: K/V from projected vision states
        p = h * nh * hd + 2 * h * kv * hd + nh * hd * h + 2 * hd
        return p, p
    if kind == "mla":
        qd = t["qk_nope_head_dim"] + t["qk_rope_head_dim"]
        q = (h * t["q_lora_rank"] + t["q_lora_rank"] + t["q_lora_rank"] * nh * qd) if t.get("q_lora_rank") else h * nh * qd
        kv_a = h * (t["kv_lora_rank"] + t["qk_rope_head_dim"]) + t["kv_lora_rank"]
        kv_b = t["kv_lora_rank"] * nh * (t["qk_nope_head_dim"] + t["v_head_dim"])
        o = nh * t["v_head_dim"] * h
        p = q + kv_a + kv_b + o
        return p, p
    if kind == "mlp":
        inter = t["intermediate_size"]
        p = (3 if gated(t) else 2) * h * inter
        return p, p
    if kind == "moe":
        e = first(t, "n_routed_experts", "num_local_experts", "num_experts")
        k = first(t, "num_experts_per_tok", "moe_topk")
        inter = first(t, "moe_intermediate_size", "intermediate_size")
        one = (3 if gated(t) else 2) * h * inter
        n_shared = t.get("n_shared_experts") or 0
        shared_inter = first(t, "moe_shared_expert_intermediate_size", "shared_expert_intermediate_size", default=inter)
        shared = n_shared * (3 if gated(t) else 2) * h * shared_inter
        if not n_shared and t.get("shared_expert_intermediate_size"):    # Qwen2-MoE style single shared expert
            shared = (3 if gated(t) else 2) * h * t["shared_expert_intermediate_size"]
        router = h * e + (e if t.get("scoring_func") == "sigmoid" else 0)   # DeepSeek adds a correction bias
        return e * one + shared + router, k * one + shared + router
    if kind == "mamba":
        m = mamba_dims(t, h)
        conv_ch = m["d_inner"] + 2 * m["groups"] * m["d_state"]
        p = (h * (2 * m["d_inner"] + 2 * m["groups"] * m["d_state"] + m["heads"])
             + conv_ch * m["kernel"] + (conv_ch if m["conv_bias"] else 0)
             + 3 * m["heads"] + m["d_inner"] + m["d_inner"] * h)
        return p, p
    raise ValueError(f"unknown sublayer {kind}")


def vision_params(cfg):
    """Qwen2.5-VL style ViT + 2x2 merger (decoder-only VLMs). Cross-attention VLMs use _params_override."""
    v = cfg.get("vision_config") or {}
    if "depth" not in v:
        return 0
    d, inter, m = v["hidden_size"], v["intermediate_size"], v.get("spatial_merge_size", 2)
    block = (4 * d * d + 4 * d) + (3 * d * inter + 2 * inter + d) + 2 * d
    patch = v.get("in_chans", 3) * v.get("temporal_patch_size", 1) * v["patch_size"] ** 2 * d
    dm = d * m * m
    merger = d + (dm * dm + dm) + (dm * v["out_hidden_size"] + v["out_hidden_size"])
    return v["depth"] * block + patch + merger


def kv_bytes_per_token(t, plan, kv_bytes=2):
    _, _, kv, hd = dims(t)
    if t.get("kv_lora_rank") is not None:
        n = sum(1 for layer in plan if "mla" in layer)
        return n * (t["kv_lora_rank"] + t["qk_rope_head_dim"]) * kv_bytes
    n = sum(1 for layer in plan if "attn" in layer)
    return 2 * n * kv * hd * kv_bytes


def ssm_state_bytes(t, plan):
    """Fixed Mamba-2 state per sequence: SSM state (dtype from mamba_ssm_cache_dtype) + conv state (bf16)."""
    n = sum(1 for layer in plan if "mamba" in layer)
    if not n:
        return 0
    h = dims(t)[0]
    m = mamba_dims(t, h)
    sb = DTYPE_BYTES.get(t.get("mamba_ssm_cache_dtype"), 2)
    conv = (m["d_inner"] + 2 * m["groups"] * m["d_state"]) * (m["kernel"] - 1) * 2
    return n * (m["heads"] * m["head_dim"] * m["d_state"] * sb + conv)


def attn_flops_per_ctx_token(t, plan, mode):
    """FLOPs per (query token x context token), summed over attention layers. mode: decode | prefill."""
    _, nh, _, hd = dims(t)
    total = 0
    for layer in plan:
        if "attn" in layer:
            total += 2 * nh * (hd + hd)
        elif "mla" in layer:
            if mode == "decode":     # absorbed: scores over latent+rope, values from the latent
                total += 2 * nh * ((t["kv_lora_rank"] + t["qk_rope_head_dim"]) + t["kv_lora_rank"])
            else:                    # prefill materializes per-head K (nope+rope) and V
                total += 2 * nh * ((t["qk_nope_head_dim"] + t["qk_rope_head_dim"]) + t["v_head_dim"])
    return total


def analyze(cfg):
    """Everything archmath knows about one config, as a plain dict."""
    t = text_config(cfg)
    plan = layer_plan(t)
    h, nh, kv, hd = dims(t)
    vocab = first(t, "vocab_size", default=cfg.get("vocab_size"))
    tied = bool(first(t, "tie_word_embeddings", default=cfg.get("tie_word_embeddings", False)))
    counts = {"attn": 0, "mla": 0, "xattn": 0, "mlp": 0, "moe": 0, "mamba": 0}
    total = active = 0
    by_kind = {k: [0, 0] for k in counts}
    for layer in plan:
        for kind in layer:
            p, a = sublayer_params(kind, t, cfg)
            counts[kind] += 1
            by_kind[kind][0] += p
            by_kind[kind][1] += a
            total += p
            active += a
    norms = (2 * len(plan) + 1) * h
    embed = vocab * h
    head = 0 if tied else vocab * h
    vision = vision_params(cfg)
    total += norms + embed + head + vision
    active += norms + embed + head + vision
    mtp = 0
    for layer in mtp_plan(t):
        for kind in layer:
            mtp += sublayer_params(kind, t, cfg)[0]
        mtp += 2 * h * h + 3 * h                                           # eh_proj + norms
    override = cfg.get("_params_override") or {}
    if override.get("total_b"):        # e.g. Mllama: text side computed, vision tower = model-card total - text
        vision = max(0.0, override["total_b"] * 1e9 - total)
        total = active = override["total_b"] * 1e9
    experts = first(t, "n_routed_experts", "num_local_experts", "num_experts", default=0) or 0
    topk = first(t, "num_experts_per_tok", "moe_topk", default=0) or 0
    moe_layers = counts["moe"]
    routed_total = 0
    if moe_layers:
        one = (3 if gated(t) else 2) * h * first(t, "moe_intermediate_size", "intermediate_size")
        routed_total = moe_layers * experts * one
    quantizable = sum(v[0] for k, v in by_kind.items())                  # block linears (+ tiny router/bias terms)
    keep_bf16 = norms + embed + head + vision
    family = describe_family(cfg, t, counts, experts)
    kvb = kv_bytes_per_token(t, plan)
    window = t.get("sliding_window") if t.get("use_sliding_window") else None
    xattn_kv_per_vision_token = 2 * counts["xattn"] * kv * hd * 2
    return {
        "name": cfg.get("_name"), "source": cfg.get("_source", ""), "family": family,
        "layers": len(plan), "layer_counts": {k: v for k, v in counts.items() if v},
        "hidden": h, "heads": nh, "kv_heads": kv, "head_dim": hd, "vocab": vocab, "tied_embeddings": tied,
        "total_params": total, "active_params": active, "mtp_params": mtp,
        "checkpoint_params_b": cfg.get("_checkpoint_params_b"),
        "embed_params": embed, "lm_head_params": head, "vision_params": vision,
        "quantizable_params": quantizable, "bf16_params": keep_bf16,
        "experts": experts, "experts_per_token": topk, "moe_layers": moe_layers, "routed_expert_params": routed_total,
        "kv_bytes_per_token_bf16": kvb, "kv_window_tokens": window,
        "ssm_state_bytes_per_seq": ssm_state_bytes(t, plan),
        "xattn_kv_bytes_per_vision_token_bf16": xattn_kv_per_vision_token,
        "attn_flops_per_ctx_decode": attn_flops_per_ctx_token(t, plan, "decode"),
        "attn_flops_per_ctx_prefill": attn_flops_per_ctx_token(t, plan, "prefill"),
        "ssm_flops_per_token": 6 * mamba_dims(t, h)["d_inner"] * mamba_dims(t, h)["d_state"] * counts["mamba"]
        if counts["mamba"] else 0,
        "max_position_embeddings": first(t, "max_position_embeddings", default=cfg.get("max_position_embeddings")),
    }


def describe_family(cfg, t, counts, experts):
    parts = []
    if counts["mamba"]:
        parts.append(f"hybrid: {counts['mamba']} Mamba-2 + {counts['attn'] + counts['mla']} attention")
    if counts["mla"]:
        parts.append("MLA")
    elif counts["attn"]:
        _, nh, kv, _ = dims(t)
        parts.append("MHA" if kv == nh else ("MQA" if kv == 1 else f"GQA {nh}:{kv}"))
    if counts["xattn"]:
        parts.append(f"cross-attention VLM ({counts['xattn']} xattn layers)")
    if experts:
        k = first(t, "num_experts_per_tok", "moe_topk")
        parts.append(f"MoE {k}/{experts}" + (" + shared" if (t.get("n_shared_experts") or 0) else ""))
    if (cfg.get("vision_config") or {}).get("depth"):
        parts.append("decoder-only VLM")
    if not experts and not counts["mamba"]:
        parts.insert(0, "dense")
    return ", ".join(parts)


# ----------------------------------------------------------------------------------------------- derived costs
def weight_bytes(m, precision):
    """Capacity: every parameter that must sit in GPU memory."""
    return m["quantizable_params"] * PRECISION_BYTES[precision] + m["bf16_params"] * 2


def expert_coverage(experts, k, batch):
    """Expected unique experts touched by `batch` tokens with uniform routing: E x (1 - (1 - k/E)^b)."""
    if not experts or not k:
        return 0.0
    return experts * (1.0 - (1.0 - k / experts) ** batch)


def decode_weight_bytes(m, precision, batch):
    """Bytes of weights STREAMED per decode step (embedding table excluded; MoE = only experts touched)."""
    p = PRECISION_BYTES[precision]
    streamed_bf16 = m["bf16_params"] - m["embed_params"] - m["vision_params"]   # lm_head + norms (+ router)
    block = m["quantizable_params"]
    if m["experts"]:
        frac = expert_coverage(m["experts"], m["experts_per_token"], batch) / m["experts"]
        block = block - m["routed_expert_params"] + m["routed_expert_params"] * frac
    return block * p + streamed_bf16 * 2


def matmul_active(m):
    """Active parameters that are multiplied per token (input-embedding lookup is a gather, not a matmul)."""
    return m["active_params"] - m["embed_params"] - m["vision_params"]


def decode_flops_per_token(m, ctx):
    return 2 * matmul_active(m) + m["attn_flops_per_ctx_decode"] * ctx + m["ssm_flops_per_token"]


def prefill_flops(m, length):
    """FLOPs to prefill `length` tokens (lm_head for the last token only; causal attention ~ L^2)."""
    linear = 2 * (matmul_active(m) - m["lm_head_params"]) * length + 2 * m["lm_head_params"]
    attn = (m["attn_flops_per_ctx_prefill"] / 2) * length * length
    return linear + attn + m["ssm_flops_per_token"] * length


def attention_share(m, ctx, mode):
    if mode == "decode":
        a = m["attn_flops_per_ctx_decode"] * ctx
        return a / (a + 2 * matmul_active(m) + m["ssm_flops_per_token"])
    a = (m["attn_flops_per_ctx_prefill"] / 2) * ctx * ctx
    return a / prefill_flops(m, ctx)


def peak_tflops(gpu, precision):
    g = GPUS[gpu]
    if precision == "nvfp4":                       # no FP4 tensor cores before Blackwell: weight-only dequant
        return g["fp4_tflops"] or g["fp8_tflops"] or g["bf16_tflops"]
    if precision == "fp8":
        return g["fp8_tflops"] or g["bf16_tflops"]   # Ampere: no FP8 tensor cores -> BF16 math
    return g["bf16_tflops"]


def roofline(m, gpu, precision, batch, ctx, gpus=1, kv_precision="bf16"):
    kv_b = PRECISION_BYTES["fp8"] if kv_precision == "fp8" else 2
    kv = m["kv_bytes_per_token_bf16"] / 2 * kv_b
    if m["kv_window_tokens"]:
        ctx_kv = min(ctx, m["kv_window_tokens"])
    else:
        ctx_kv = ctx
    w = decode_weight_bytes(m, precision, batch)
    kv_read = batch * ctx_kv * kv
    state = batch * m["ssm_state_bytes_per_seq"] * 2             # read + write every step
    total_bytes = (w + kv_read + state) / gpus
    flops = batch * decode_flops_per_token(m, ctx) / gpus
    g = GPUS[gpu]
    t_mem = total_bytes / (g["bw_tbs"] * 1e12)
    t_cmp = flops / (peak_tflops(gpu, precision) * 1e12)
    step = max(t_mem, t_cmp)
    return {"model": m["name"], "gpu": gpu, "gpus": gpus, "precision": precision, "batch": batch, "ctx": ctx,
            "weight_gb_streamed": w / 1e9 / gpus, "kv_gb_read": (kv_read + state) / 1e9 / gpus,
            "bytes_gb": total_bytes / 1e9, "gflops": flops / 1e9, "intensity_flop_per_byte": flops / total_bytes,
            "ridge_flop_per_byte": peak_tflops(gpu, precision) * 1e12 / (g["bw_tbs"] * 1e12),
            "t_mem_ms": t_mem * 1e3, "t_compute_ms": t_cmp * 1e3, "bound": "memory" if t_mem >= t_cmp else "compute",
            "step_ms": step * 1e3, "tok_s_per_seq": 1 / step, "tok_s_total": batch / step,
            "fits": weight_bytes(m, precision) / gpus <= GPUS[gpu]["mem_gb"] * 1e9 * 0.9}


def prefill_time_ms(m, gpu, precision, length, gpus=1, mfu=1.0):
    return prefill_flops(m, length) / gpus / (peak_tflops(gpu, precision) * 1e12 * mfu) * 1e3


# ----------------------------------------------------------------------------------------------- tables
def fmt_b(x):
    return f"{x / 1e9:,.1f}"


def summary_rows(models, ctx):
    rows = []
    for m in models:
        rows.append({
            "model": m["name"], "family": m["family"],
            "total_b": m["total_params"] / 1e9, "active_b": m["active_params"] / 1e9,
            "mtp_b": m["mtp_params"] / 1e9,
            "weights_gb": {p: weight_bytes(m, p) / 1e9 for p in PRECISION_BYTES},
            "kv_kib_per_token": m["kv_bytes_per_token_bf16"] / 1024,
            "state_mb_per_seq": m["ssm_state_bytes_per_seq"] / 1e6,
            "decode_gflops_per_token": decode_flops_per_token(m, ctx) / 1e9,
            "max_context": m["max_position_embeddings"],
        })
    return rows


def md_table(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def print_summary(models, ctx):
    rows = []
    for r in summary_rows(models, ctx):
        w = r["weights_gb"]
        mtp = f" (+{r['mtp_b']:.1f} MTP)" if r["mtp_b"] else ""
        rows.append([r["model"], r["family"], f"{r['total_b']:,.1f}{mtp}", f"{r['active_b']:,.1f}",
                     f"{w['bf16']:,.1f} / {w['fp8']:,.1f} / {w['nvfp4']:,.1f}", f"{r['kv_kib_per_token']:,.1f}",
                     f"{r['state_mb_per_seq']:,.1f}" if r["state_mb_per_seq"] else "-",
                     f"{r['decode_gflops_per_token']:,.1f}", f"{r['max_context']:,}" if r["max_context"] else "-"])
    print(md_table(["model", "architecture", "total B", "active B", "weights GB bf16 / fp8 / nvfp4",
                    "KV KiB/token (bf16)", "fixed state MB/seq", f"decode GFLOP/token @{ctx // 1024}K",
                    "max context"], rows))


def print_roofline(models, gpu, precision, batches, ctx, gpus, kv_precision):
    rows = []
    for m in models:
        for b in batches:
            r = roofline(m, gpu, precision, b, ctx, gpus, kv_precision)
            rows.append([m["name"], b, f"{r['weight_gb_streamed']:.2f}", f"{r['kv_gb_read']:.2f}",
                         f"{r['intensity_flop_per_byte']:.0f}", f"{r['t_mem_ms']:.2f}", f"{r['t_compute_ms']:.2f}",
                         r["bound"], f"{r['tok_s_per_seq']:,.0f}", f"{r['tok_s_total']:,.0f}",
                         "yes" if r["fits"] else "NO"])
    ridge = peak_tflops(gpu, precision) * 1e12 / (GPUS[gpu]["bw_tbs"] * 1e12)
    print(f"Decode roofline on {gpus}x {gpu}, {precision} weights, {kv_precision} KV, context {ctx:,} tokens "
          f"(ridge point {ridge:.0f} FLOP/byte; per-GPU numbers, ideal TP split):")
    print(md_table(["model", "batch", "weights GB/step", "KV+state GB/step", "FLOP/byte", "t_mem ms", "t_compute ms",
                    "bound", "tok/s per seq", "tok/s total", "fits"], rows))


def print_coverage(models, batches):
    rows = []
    for m in models:
        if not m["experts"]:
            continue
        for b in batches:
            u = expert_coverage(m["experts"], m["experts_per_token"], b)
            dense_like = decode_weight_bytes(m, "bf16", b) / decode_weight_bytes(m, "bf16", 10 ** 6)
            rows.append([m["name"], b, f"{u:,.1f} of {m['experts']}", f"{u / m['experts']:.0%}",
                         f"{decode_weight_bytes(m, 'bf16', b) / 1e9:,.1f}", f"{dense_like:.0%}"])
    print("MoE expert coverage per decode step (uniform routing; real routers are skewed, so measured is lower):")
    print(md_table(["model", "batch", "experts touched per layer", "routed weights read", "weights GB/step (bf16)",
                    "share of all weights"], rows))


def print_attention_share(models, contexts):
    rows = []
    for m in models:
        cells = [m["name"]]
        for c in contexts:
            cells.append(f"{attention_share(m, c, 'prefill'):.0%} / {attention_share(m, c, 'decode'):.0%}")
        rows.append(cells)
    print("Share of FLOPs spent in attention that grows with context (prefill of L tokens / one decode step at L):")
    print(md_table(["model"] + [f"{c // 1024}K prefill / decode" for c in contexts], rows))


def print_prefill(models, gpu, precision, length, gpus, mfu):
    rows = []
    for m in models:
        rows.append([m["name"], f"{prefill_flops(m, length) / 1e12:,.1f}",
                     f"{prefill_time_ms(m, gpu, precision, length, gpus, 1.0):,.1f}",
                     f"{prefill_time_ms(m, gpu, precision, length, gpus, mfu):,.1f}",
                     f"{m['kv_bytes_per_token_bf16'] * length / 1e6:,.1f}"])
    print(f"Prefill of {length:,} tokens on {gpus}x {gpu} ({precision}); TTFT floor at 100% and {mfu:.0%} MFU:")
    print(md_table(["model", "TFLOP", "ms @100%", f"ms @{mfu:.0%}", "KV written MB (bf16)"], rows))


def parse_ints(s):
    return [int(float(x)) for x in str(s).split(",") if x.strip()]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="", help="comma list of bundled names (default: all): " + ", ".join(bundled_names()))
    ap.add_argument("--config", action="append", default=[], help="extra config.json path or HF repo id (repeatable)")
    ap.add_argument("--ctx", type=int, default=8192, help="context length for decode FLOPs/roofline (default 8192)")
    ap.add_argument("--gpu", default="H100-SXM", choices=sorted(GPUS))
    ap.add_argument("--gpus", type=int, default=1, help="ideal tensor-parallel split for the roofline (default 1)")
    ap.add_argument("--precision", default="bf16", choices=sorted(PRECISION_BYTES))
    ap.add_argument("--kv-precision", default="bf16", choices=["bf16", "fp8"])
    ap.add_argument("--batch", default="1,8,32", help="comma list of decode batch sizes")
    ap.add_argument("--roofline", action="store_true")
    ap.add_argument("--coverage", action="store_true", help="MoE expert coverage vs batch")
    ap.add_argument("--attention-share", action="store_true")
    ap.add_argument("--prefill", type=int, default=0, help="prefill length in tokens (TTFT floor table)")
    ap.add_argument("--mfu", type=float, default=0.5, help="model FLOPs utilization for the prefill estimate")
    ap.add_argument("--md", action="store_true", help="markdown tables (the default output)")
    ap.add_argument("--json", action="store_true", help="JSON instead of markdown")
    ap.add_argument("--notes", action="store_true", help="print the assumptions")
    a = ap.parse_args(argv)
    names = [n.strip() for n in a.model.split(",") if n.strip()] or bundled_names()
    unknown = [n for n in names if n not in bundled_names()]
    if unknown:
        ap.error(f"unknown model(s) {unknown}; bundled: {bundled_names()} (use --config for others)")
    cfgs = [load_config(n) for n in names] + [load_config(c) for c in a.config]
    models = [analyze(c) for c in cfgs]
    batches = parse_ints(a.batch)
    if a.json:
        out = {"assumptions": NOTES, "gpus": GPUS, "summary": summary_rows(models, a.ctx), "models": models}
        if a.roofline:
            out["roofline"] = [roofline(m, a.gpu, a.precision, b, a.ctx, a.gpus, a.kv_precision)
                               for m in models for b in batches]
        if a.coverage:
            out["coverage"] = [{"model": m["name"], "batch": b,
                                "experts_touched": expert_coverage(m["experts"], m["experts_per_token"], b)}
                               for m in models if m["experts"] for b in batches]
        if a.attention_share:
            out["attention_share"] = [{"model": m["name"], "ctx": c, "prefill": attention_share(m, c, "prefill"),
                                       "decode": attention_share(m, c, "decode")}
                                      for m in models for c in (8192, 32768, 131072)]
        if a.prefill:
            out["prefill"] = [{"model": m["name"], "tokens": a.prefill, "tflop": prefill_flops(m, a.prefill) / 1e12,
                               "ms_at_100pct": prefill_time_ms(m, a.gpu, a.precision, a.prefill, a.gpus, 1.0)}
                              for m in models]
        json.dump(out, sys.stdout, indent=2, default=str)
        print()
        return 0
    if a.notes:
        print("Assumptions:")
        for n in NOTES:
            print(f"- {n}")
        print()
    specific = a.roofline or a.coverage or a.attention_share or a.prefill
    if not specific:
        print_summary(models, a.ctx)
    if a.roofline:
        print_roofline(models, a.gpu, a.precision, batches, a.ctx, a.gpus, a.kv_precision)
        print()
    if a.coverage:
        print_coverage(models, batches)
        print()
    if a.attention_share:
        print_attention_share(models, (8192, 32768, 131072))
        print()
    if a.prefill:
        print_prefill(models, a.gpu, a.precision, a.prefill, a.gpus, a.mfu)
    return 0


if __name__ == "__main__":
    sys.exit(main())
