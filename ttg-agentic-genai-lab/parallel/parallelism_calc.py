#!/usr/bin/env python3
"""M7 - parallelism math you can do at a whiteboard (🔁 mock round Q1, Lab 02 Dynamo, M2 archmath).

Memory per GPU, bytes on the wire per step, and time - for data (DP / ZeRO / FSDP), tensor (TP), pipeline (PP),
expert (EP) and context (CP) parallelism, in training and in inference. Pure Python; model shapes come from
archmath/configs (any bundled name works).

Usage (from the repo root):
  python3 -m parallel.parallelism_calc --examples          # the five worked examples in docs/05 (markdown)
  python3 -m parallel.parallelism_calc --links             # the link bandwidth / latency table everything uses
  python3 -m parallel.parallelism_calc --train --model llama-3.1-8b --dp 8 --zero 3 --seq 4096 --mbs 2
  python3 -m parallel.parallelism_calc --train --model llama-3.1-8b --lora-rank 16 --seq 2048 --mbs 4 --tokens 1.5e6
  python3 -m parallel.parallelism_calc --train --model llama-3.1-70b --tp 8 --pp 4 --dp 64 --zero 1 \\
      --microbatches 32 --mbs 1 --seq 8192 --tokens 15e12 --mfu 0.4 --tp-link nvlink4 --dp-link ib-ndr
  python3 -m parallel.parallelism_calc --infer --model llama-3.1-70b --precision fp8 --tp 1,2,4,8 --batch 1,32 \\
      --link nvlink4                                        # TP decode: weight read vs all-reduce latency
Formulas and their sources are in docs/05-parallelism-math.md. Bandwidths are APPROXIMATE per-direction link rates;
replace EFFICIENCY and alpha with what nccl-tests measures on YOUR node (parallel/nccl_tests.sh + compare_nccl.py).
"""
import argparse
import json
import os
import sys

if __package__ in (None, ""):          # run as a file: put the repo root on sys.path
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from archmath import archmath as am  # noqa: E402

# Per GPU, per direction. alpha = per-message latency floor (order of magnitude; measure small all_reduce sizes).
LINKS = {
    "nvlink5": {"gbs": 900, "alpha_us": 10, "what": "NVLink 5 (Blackwell; GB200 NVL72 domain): 1.8 TB/s per GPU, both directions"},
    "nvlink4": {"gbs": 450, "alpha_us": 10, "what": "NVLink 4 + NVSwitch (H100 SXM, DGX/HGX H100): 900 GB/s per GPU, both directions"},
    "nvlink3": {"gbs": 300, "alpha_us": 10, "what": "NVLink 3 (A100 SXM): 600 GB/s per GPU, both directions"},
    "pcie5": {"gbs": 64, "alpha_us": 20, "what": "PCIe Gen5 x16: ~64 GB/s per direction"},
    "pcie4": {"gbs": 32, "alpha_us": 20, "what": "PCIe Gen4 x16 (L40S, A100 PCIe): ~32 GB/s per direction"},
    "ib-ndr": {"gbs": 50, "alpha_us": 25, "what": "InfiniBand NDR 400 Gb/s, one NIC per GPU (rail-optimized, ConnectX-7)"},
    "ib-xdr": {"gbs": 100, "alpha_us": 25, "what": "InfiniBand XDR 800 Gb/s, one NIC per GPU (Quantum-X800 class)"},
    "eth-400": {"gbs": 50, "alpha_us": 30, "what": "400 GbE RoCE, one NIC per GPU (Spectrum-X class)"},
}
EFFICIENCY = 0.8           # NCCL bus bandwidth on large messages is typically 70-90% of the link rate
GPU_PEAK_TFLOPS = {"H100-SXM": {"bf16": 989, "fp8": 1979}, "A100-80GB-SXM": {"bf16": 312, "fp8": 312},
                   "L40S": {"bf16": 362, "fp8": 733}, "B200-in-GB200": {"bf16": 2500, "fp8": 5000}}
GPU_MEM_GB = {"H100-SXM": 80, "A100-80GB-SXM": 80, "L40S": 48, "B200-in-GB200": 186}
USABLE = 0.92              # CUDA context, NCCL buffers, fragmentation: plan for ~92% of the card
BYTES = 2                  # BF16 activations / gradients on the wire


# ----------------------------------------------------------------------------------------------- collectives
def ring_allreduce_bytes(msg_bytes, n):
    """Bytes each GPU sends (and receives) in a ring all-reduce: reduce-scatter + all-gather = 2(n-1)/n x message."""
    return 0.0 if n <= 1 else 2.0 * (n - 1) / n * msg_bytes


def allgather_bytes(total_bytes, n):
    """All-gather (or reduce-scatter) of a tensor of total_bytes: each GPU moves (n-1)/n of it."""
    return 0.0 if n <= 1 else (n - 1) / n * total_bytes


def comm_time_s(nbytes, link, messages=1, efficiency=EFFICIENCY):
    """alpha-beta model: messages x latency + bytes / (link rate x efficiency)."""
    if nbytes <= 0 and messages <= 0:
        return 0.0
    lk = LINKS[link]
    return messages * lk["alpha_us"] * 1e-6 + nbytes / (lk["gbs"] * 1e9 * efficiency)


# ----------------------------------------------------------------------------------------------- memory
def model_state_bytes(params, dp=1, tp=1, pp=1, zero=0, lora_params=0.0, expert_params=0.0, ep=1):
    """Weights + gradients + Adam states per GPU, mixed precision (2 + 2 + 12 = 16 bytes per trained parameter).

    zero=0 plain DP · 1 shards optimizer states · 2 + gradients · 3 (= FSDP full shard) + weights.
    dp here is the GRADIENT-SYNC group: pass dp x cp when context parallelism is on (CP ranks hold the same weights
    and all-reduce their gradients, so Megatron's distributed optimizer shards over dp x cp).
    MoE: routed-expert weights are split over the expert-parallel group (ep) instead of TP, and ZeRO shards them
    over the expert-data-parallel group (dp / ep) - Megatron carves EP out of the DP x CP ranks.
    LoRA: the base model is frozen BF16 (2 bytes/param, sharded only by ZeRO-3/FSDP); the adapters carry the full
    16 bytes/param and are sharded by ZeRO like any trained parameter.
    """
    dense = (params - expert_params) / (tp * pp)
    experts = expert_params / (ep * pp)
    edp = max(1, dp // max(1, ep))

    def states(n, group):
        w, g, o = 2 * n, 2 * n, 12 * n
        if zero >= 1:
            o /= group
        if zero >= 2:
            g /= group
        if zero >= 3:
            w /= group
        return w + g + o

    if lora_params:
        frozen = 2 * dense / (dp if zero >= 3 else 1) + 2 * experts / (edp if zero >= 3 else 1)
        return frozen + states(lora_params / (tp * pp), dp)
    return states(dense, dp) + states(experts, edp)


def activation_bytes(seq, mbs, hidden, layers, heads, tp=1, pp=1, microbatches=1, sp=True, flash=True,
                     recompute="none"):
    """Activation memory per GPU, Korthikanti et al. 2022 (GPT-style block, BF16), approximate for SwiGLU models.

    per layer, no parallelism:          s*b*h*(34 + 5*a*s/h)    (5as/h = attention scores, softmax, dropout mask)
    TP without sequence parallel:       s*b*h*(10 + 24/t + 5*a*s/(h*t))
    TP with sequence parallel (SP):     s*b*h*(34 + 5*a*s/h) / t
    flash attention / selective recompute drop the 5as/h term; full recompute keeps 2*s*b*h (the layer input).
    Pipeline 1F1B: the first stage holds min(p, m) micro-batches of its layers/p layers in flight.
    """
    sbh = seq * mbs * hidden
    attn_term = 0.0 if (flash or recompute == "selective") else 5.0 * heads * seq / hidden
    if tp > 1 and not sp:
        per_layer = sbh * (10 + 24 / tp + attn_term / tp)
    else:
        per_layer = sbh * (34 + attn_term) / tp
    if recompute == "full":
        kept = 2 * sbh / (tp if sp else 1)
        stage_layers = layers / pp
        return kept * stage_layers * min(pp, microbatches) + per_layer      # + one layer rematerialized
    return per_layer * (layers / pp) * min(pp, microbatches)


def lora_params(m, rank, targets=("q", "k", "v", "o", "gate", "up", "down")):
    """Adapter parameters: r x (in + out) per targeted matrix per layer (dense-MLP layers; MoE experts skipped)."""
    h, nh, kv, hd = m["hidden"], m["heads"], m["kv_heads"], m["head_dim"]
    cfg = am.text_config(am.load_config(m["name"]))
    inter = cfg.get("intermediate_size") or 4 * h
    shapes = {"q": (h, nh * hd), "k": (h, kv * hd), "v": (h, kv * hd), "o": (nh * hd, h),
              "gate": (h, inter), "up": (h, inter), "down": (inter, h)}
    per_attn = sum(rank * sum(shapes[t]) for t in targets if t in ("q", "k", "v", "o"))
    per_mlp = sum(rank * sum(shapes[t]) for t in targets if t in ("gate", "up", "down"))
    attn_layers = m["layer_counts"].get("attn", 0)
    mlp_layers = m["layer_counts"].get("mlp", 0)
    return attn_layers * per_attn + mlp_layers * per_mlp


# ----------------------------------------------------------------------------------------------- training
def pipeline_bubble(pp, microbatches):
    """Idle fraction of a 1F1B / GPipe schedule: (p - 1) / (m + p - 1)."""
    return 0.0 if pp <= 1 else (pp - 1) / (microbatches + pp - 1)


def train_flops(active_params, tokens, lora=False):
    """6 x N x T for full training (2 forward + 4 backward); ~4 x N x T for LoRA (no weight gradients)."""
    return (4 if lora else 6) * active_params * tokens


def train_time_s(active_params, tokens, gpus, peak_tflops, mfu, lora=False):
    return train_flops(active_params, tokens, lora) / (gpus * peak_tflops * 1e12 * mfu)


def ep_alltoall_bytes(tokens, hidden, topk, ep, moe_layers, dispatch_bytes=BYTES, combine_bytes=BYTES, backward=True):
    """Per GPU per step: every token goes to k experts and comes back; (ep-1)/ep of that leaves the GPU."""
    if ep <= 1:
        return 0.0
    one_way = tokens * topk * hidden * (ep - 1) / ep
    per_layer = one_way * (dispatch_bytes + combine_bytes)
    return per_layer * moe_layers * (2 if backward else 1)


def cp_ring_bytes(seq, mbs, cp, kv_heads, head_dim, attn_layers, backward=True):
    """Ring attention: each GPU passes its K and V chunk (s/c tokens) around the ring c-1 times per layer."""
    if cp <= 1:
        return 0.0
    chunk = (seq / cp) * mbs * 2 * kv_heads * head_dim * BYTES
    return attn_layers * (cp - 1) * chunk * (2 if backward else 1)


def train_plan(m, *, dp=1, tp=1, pp=1, ep=1, cp=1, zero=0, seq=4096, mbs=1, microbatches=1, lora_rank=0,
               flash=True, recompute="none", sp=True, tokens=0.0, mfu=0.4, gpu="H100-SXM", precision="bf16",
               tp_link="nvlink4", dp_link="nvlink4", pp_link="nvlink4", ep_link="nvlink4", cp_link="nvlink4"):
    gpus = dp * tp * pp * cp                    # EP reuses data-parallel ranks (Megatron), so it adds no GPUs
    if ep > 1 and (dp * cp) % ep:
        raise ValueError(f"ep={ep} must divide dp x cp = {dp * cp} (expert parallelism reuses data-parallel ranks)")
    lp = lora_params(m, lora_rank) if lora_rank else 0
    expert_params = m["routed_expert_params"] if ep > 1 else 0.0
    sync = dp * cp                              # gradient all-reduce / ZeRO group: CP ranks share the weights too
    states = model_state_bytes(m["total_params"], dp=sync, tp=tp, pp=pp, zero=zero, lora_params=lp,
                               expert_params=expert_params, ep=ep)
    acts = activation_bytes(seq / cp, mbs, m["hidden"], m["layers"], m["heads"], tp=tp, pp=pp,
                            microbatches=microbatches, sp=sp, flash=flash, recompute=recompute)
    tokens_per_step = dp * mbs * seq * microbatches
    peak = GPU_PEAK_TFLOPS[gpu][precision]
    flops_step = train_flops(m["active_params"], tokens_per_step, lora=bool(lora_rank))
    t_compute = flops_step / (gpus * peak * 1e12 * mfu)
    # gradient sync over dp x cp ranks (LoRA: adapters only) per model-parallel shard; expert grads over
    # (dp x cp) / ep ranks
    edp = max(1, sync // max(1, ep))
    if lora_rank:
        grads = [(2 * lp / (tp * pp), sync)]
    else:
        grads = [(2 * (m["total_params"] - expert_params) / (tp * pp), sync), (2 * expert_params / (ep * pp), edp)]
    if zero >= 3:                                              # all-gather fwd + all-gather bwd + reduce-scatter
        dp_bytes = sum(3 * allgather_bytes(g, n) for g, n in grads)
        if lora_rank:                                          # FSDP + LoRA still gathers the FROZEN base, fwd + bwd
            dp_bytes += 2 * allgather_bytes(2 * (m["total_params"] - expert_params) / (tp * pp), sync)
            dp_bytes += 2 * allgather_bytes(2 * expert_params / (ep * pp), edp)
        dp_msgs = 3 * m["layers"] // pp
    else:
        dp_bytes = sum(ring_allreduce_bytes(g, n) for g, n in grads)
        dp_msgs = 1
    layers_stage = m["layers"] / pp
    tp_msg = (seq / cp) * mbs * m["hidden"] * BYTES
    tp_bytes = layers_stage * 4 * ring_allreduce_bytes(tp_msg, tp) * microbatches
    tp_msgs = int(layers_stage * 4 * microbatches) if tp > 1 else 0
    pp_bytes = 2 * (seq / cp) * mbs * m["hidden"] * BYTES * microbatches if pp > 1 else 0.0
    moe_layers = m["moe_layers"]
    ep_bytes = ep_alltoall_bytes((seq / cp) * mbs * microbatches, m["hidden"], m["experts_per_token"], ep,
                                 moe_layers / pp) if moe_layers else 0.0
    cp_bytes = cp_ring_bytes(seq, mbs, cp, m["kv_heads"], m["head_dim"],
                             (m["layer_counts"].get("attn", 0) + m["layer_counts"].get("mla", 0)) / pp) * microbatches
    comm = {
        "dp": {"bytes": dp_bytes, "s": comm_time_s(dp_bytes, dp_link, dp_msgs) if sync > 1 else 0.0, "link": dp_link},
        "tp": {"bytes": tp_bytes, "s": comm_time_s(tp_bytes, tp_link, tp_msgs) if tp > 1 else 0.0, "link": tp_link},
        "pp": {"bytes": pp_bytes, "s": comm_time_s(pp_bytes, pp_link, 2 * microbatches) if pp > 1 else 0.0,
               "link": pp_link},
        "ep": {"bytes": ep_bytes, "s": comm_time_s(ep_bytes, ep_link, 4 * moe_layers) if ep > 1 else 0.0,
               "link": ep_link},
        "cp": {"bytes": cp_bytes, "s": comm_time_s(cp_bytes, cp_link) if cp > 1 else 0.0, "link": cp_link},
    }
    bubble = pipeline_bubble(pp, microbatches)
    out = {
        "model": m["name"], "gpus": gpus, "dp": dp, "tp": tp, "pp": pp, "ep": ep, "cp": cp, "zero": zero,
        "lora_rank": lora_rank, "lora_params": lp, "seq": seq, "mbs": mbs, "microbatches": microbatches,
        "model_state_gb": states / 1e9, "activations_gb": acts / 1e9, "total_gb": (states + acts) / 1e9,
        "fits": (states + acts) <= GPU_MEM_GB[gpu] * 1e9 * USABLE, "gpu": gpu,
        "tokens_per_step": tokens_per_step, "compute_s_per_step": t_compute, "bubble": bubble,
        "comm": comm, "flops_per_step": flops_step,
    }
    if tokens:
        out["train_days"] = train_time_s(m["active_params"], tokens, gpus, peak, mfu, bool(lora_rank)) \
            / (1 - bubble) / 86400
        out["train_hours"] = out["train_days"] * 24
        out["gpu_hours"] = out["train_hours"] * gpus
    return out


# ----------------------------------------------------------------------------------------------- inference
def tp_decode(m, precision="bf16", tp=1, batch=1, ctx=8192, link="nvlink4", gpu="H100-SXM"):
    """Decode step with tensor parallelism: per-GPU roofline (archmath) + 2 all-reduces per layer of b*h*2 bytes.

    Small messages => the latency term alpha dominates: that is why TP decode scales sub-linearly and why TP
    across PCIe or across nodes hurts inter-token latency.
    """
    r = am.roofline(m, gpu, precision, batch, ctx, gpus=tp)
    per_layer_ar = 1 if m["layer_counts"].get("mamba") else 2      # NemotronH: one block (one all-reduce) per layer
    msgs = m["layers"] * per_layer_ar if tp > 1 else 0
    msg = batch * m["hidden"] * BYTES
    t_comm = comm_time_s(ring_allreduce_bytes(msg, tp) * msgs, link, msgs) if tp > 1 else 0.0
    step = r["step_ms"] / 1e3 + t_comm
    return {"model": m["name"], "precision": precision, "tp": tp, "batch": batch, "ctx": ctx, "link": link,
            "fits": r["fits"], "roofline_ms": r["step_ms"], "allreduce_ms": t_comm * 1e3, "step_ms": step * 1e3,
            "tok_s_per_seq": 1 / step, "tok_s_total": batch / step, "tok_s_per_gpu": batch / step / tp}


# ----------------------------------------------------------------------------------------------- output
def gb(x):
    return f"{x / 1e9:,.2f}"


def md(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def print_links():
    print(f"Link rates (approximate, per GPU per direction; NCCL efficiency assumed {EFFICIENCY:.0%}):")
    print(md(["link", "GB/s", "alpha (us)", "what"],
             [[k, v["gbs"], v["alpha_us"], v["what"]] for k, v in LINKS.items()]))


def train_row(label, p):
    c = p["comm"]
    comm_s = sum(v["s"] for v in c.values())
    return [label, p["gpus"], f"{p['model_state_gb']:,.1f}", f"{p['activations_gb']:,.1f}", f"{p['total_gb']:,.1f}",
            "yes" if p["fits"] else "NO",
            f"{p['compute_s_per_step'] * 1e3:,.0f}", f"{c['dp']['s'] * 1e3:,.0f}", f"{c['tp']['s'] * 1e3:,.0f}",
            f"{c['pp']['s'] * 1e3:,.1f}", f"{p['bubble']:.0%}", f"{comm_s * 1e3:,.0f}"]


TRAIN_HEADERS = ["plan", "GPUs", "states GB/GPU", "activations GB/GPU", "total GB/GPU", "fits", "compute ms/step",
                 "DP comm ms", "TP comm ms", "PP comm ms", "PP bubble", "all comm ms"]


def examples():
    m8 = am.analyze(am.load_config("llama-3.1-8b"))
    m70 = am.analyze(am.load_config("llama-3.1-70b"))
    ds = am.analyze(am.load_config("deepseek-v3"))
    out = []

    # E1 - the M6 job: LoRA on one H100
    rows = []
    for label, rc in (("LoRA r=16, no recompute", "none"), ("LoRA r=16, full recompute", "full")):
        p = train_plan(m8, lora_rank=16, seq=2048, mbs=4, recompute=rc, tokens=1.5e6, mfu=0.35)
        rows.append(train_row(label, p) + [f"{p['train_hours'] * 60:,.1f} min"])
    lp = lora_params(m8, 16)
    out.append("### E1 - M6 LoRA job: Llama 3.1 8B, rank 16 on q,k,v,o,gate,up,down, 1x H100, seq 2048 x 4\n")
    out.append(f"Adapter parameters: {lp:,} ({lp / m8['total_params']:.2%} of the model). "
               f"1.5M training tokens per epoch at 35% MFU.\n")
    out.append(md(TRAIN_HEADERS + ["epoch time"], rows))

    # E2 - full fine-tune of 8B on one 8x H100 node
    rows = []
    for label, zero in (("DP (no sharding)", 0), ("ZeRO-1", 1), ("ZeRO-2", 2), ("ZeRO-3 / FSDP", 3)):
        p = train_plan(m8, dp=8, zero=zero, seq=4096, mbs=2, recompute="full", mfu=0.4)
        rows.append(train_row(label, p))
    pcie = train_plan(m8, dp=8, zero=3, seq=4096, mbs=2, recompute="full", mfu=0.4, dp_link="pcie4")
    rows.append(train_row("ZeRO-3 on PCIe Gen4 (L40S-style node)", pcie))
    out.append("\n### E2 - full fine-tune: Llama 3.1 8B on one 8x H100 node (NVLink 4), seq 4096 x 2 per GPU, "
               "full recompute\n")
    out.append(md(TRAIN_HEADERS, rows))

    # E3 - 70B pretraining
    p = train_plan(m70, dp=64, tp=8, pp=4, zero=1, seq=8192, mbs=1, microbatches=32, recompute="selective",
                   tokens=15e12, mfu=0.4, tp_link="nvlink4", dp_link="ib-ndr", pp_link="ib-ndr")
    out.append("\n### E3 - pretraining: Llama 3.1 70B on 2,048 H100 (TP 8 inside a node, PP 4 x DP 64 over "
               "InfiniBand NDR), 15T tokens, 40% MFU\n")
    out.append(md(TRAIN_HEADERS + ["days", "GPU-hours"],
                  [train_row("TP8 x PP4 x DP64, ZeRO-1, m=32", p)
                   + [f"{p['train_days']:,.0f}", f"{p['gpu_hours'] / 1e6:,.2f} M"]]))

    # E4 - inference TP sweep (🔁 mock round Q1)
    rows = []
    for link in ("nvlink4", "pcie4"):
        for tp in (2, 4, 8):
            for b in (1, 32):
                r = tp_decode(m70, "fp8", tp, b, 8192, link)
                rows.append([link, tp, b, f"{r['roofline_ms']:.2f}", f"{r['allreduce_ms']:.2f}", f"{r['step_ms']:.2f}",
                             f"{r['tok_s_per_seq']:,.0f}", f"{r['tok_s_per_gpu']:,.0f}"])
    out.append("\n### E4 - inference: Llama 3.1 70B FP8 decode at 8K context, TP 2 / 4 / 8 (NVLink 4 vs PCIe Gen4)\n")
    out.append(md(["link", "TP", "batch", "weight+KV read ms", "all-reduce ms", "step ms", "tok/s per seq",
                   "tok/s per GPU"], rows))

    # E5 - MoE expert parallelism: where the all-to-all lives
    rows = []
    b_local = 128
    for label, ep, link in (("EP 8 inside one 8x H200 node (NVLink 4; 671 GB FP8 does not fit 8x H100)", 8, "nvlink4"),
                            ("EP 64 across 8 nodes, IB NDR", 64, "ib-ndr"),
                            ("EP 72 inside one GB200 NVL72", 72, "nvlink5")):
        nbytes = ep_alltoall_bytes(b_local, ds["hidden"], ds["experts_per_token"], ep, ds["moe_layers"],
                                   dispatch_bytes=1, combine_bytes=2, backward=False)
        t = comm_time_s(nbytes, link, 2 * ds["moe_layers"])
        rows.append([label, f"{nbytes / 1e9:,.2f}", f"{t * 1e3:,.1f}"])
    out.append(f"\n### E5 - MoE decode: DeepSeek-V3 all-to-all per step, {b_local} tokens per GPU, FP8 dispatch + "
               f"BF16 combine, {ds['moe_layers']} MoE layers\n")
    out.append(md(["placement", "GB sent per GPU per step", "all-to-all ms per step"], rows))
    out.append("\nProduction MoE serving overlaps the all-to-all with compute (two micro-batches, DeepEP-style "
               "kernels); the point stands: across InfiniBand the all-to-all costs ~10x what it costs inside one "
               "NVLink domain.")
    return "\n".join(out)


def parse_list(s, cast=int):
    return [cast(float(x)) for x in str(s).split(",") if x.strip()]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--examples", action="store_true")
    ap.add_argument("--links", action="store_true")
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--infer", action="store_true")
    ap.add_argument("--model", default="llama-3.1-8b", help="bundled archmath config name")
    ap.add_argument("--dp", type=int, default=1)
    ap.add_argument("--tp", default="1", help="--train: one value; --infer: comma list")
    ap.add_argument("--pp", type=int, default=1)
    ap.add_argument("--ep", type=int, default=1)
    ap.add_argument("--cp", type=int, default=1)
    ap.add_argument("--zero", type=int, default=0, choices=[0, 1, 2, 3])
    ap.add_argument("--seq", type=int, default=4096)
    ap.add_argument("--mbs", type=int, default=1, help="micro-batch size per GPU")
    ap.add_argument("--microbatches", type=int, default=1, help="micro-batches per step (gradient accumulation)")
    ap.add_argument("--lora-rank", type=int, default=0)
    ap.add_argument("--recompute", default="none", choices=["none", "selective", "full"])
    ap.add_argument("--no-flash", action="store_true")
    ap.add_argument("--no-sp", action="store_true", help="tensor parallel WITHOUT sequence parallelism")
    ap.add_argument("--tokens", type=float, default=0.0, help="training tokens (for days / GPU-hours)")
    ap.add_argument("--mfu", type=float, default=0.4)
    ap.add_argument("--gpu", default="H100-SXM", choices=sorted(GPU_PEAK_TFLOPS))
    ap.add_argument("--precision", default="bf16", choices=["bf16", "fp8", "nvfp4"])
    ap.add_argument("--batch", default="1,32", help="--infer: comma list of decode batch sizes")
    ap.add_argument("--ctx", type=int, default=8192)
    ap.add_argument("--link", default="nvlink4", choices=sorted(LINKS), help="--infer: TP link")
    for k in ("tp", "dp", "pp", "ep", "cp"):
        ap.add_argument(f"--{k}-link", default="nvlink4", choices=sorted(LINKS))
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    if not (a.examples or a.links or a.train or a.infer):
        a.examples = True
    if a.links:
        print_links()
        print()
    if a.examples:
        print(examples())
        print()
    m = am.analyze(am.load_config(a.model))
    if a.train:
        tp = parse_list(a.tp)[0]
        p = train_plan(m, dp=a.dp, tp=tp, pp=a.pp, ep=a.ep, cp=a.cp, zero=a.zero, seq=a.seq, mbs=a.mbs,
                       microbatches=a.microbatches, lora_rank=a.lora_rank, flash=not a.no_flash,
                       recompute=a.recompute, sp=not a.no_sp, tokens=a.tokens, mfu=a.mfu, gpu=a.gpu,
                       precision="fp8" if a.precision == "fp8" else "bf16", tp_link=a.tp_link,
                       dp_link=a.dp_link, pp_link=a.pp_link, ep_link=a.ep_link, cp_link=a.cp_link)
        if a.json:
            print(json.dumps(p, indent=2, default=float))
        else:
            label = f"{a.model} TP{tp} PP{a.pp} DP{a.dp} EP{a.ep} CP{a.cp} ZeRO-{a.zero}" + \
                (f" LoRA r={a.lora_rank}" if a.lora_rank else "")
            extra_h, extra = [], []
            if "train_days" in p:
                extra_h, extra = ["hours", "GPU-hours"], [f"{p['train_hours']:,.1f}", f"{p['gpu_hours']:,.0f}"]
            print(md(TRAIN_HEADERS + extra_h, [train_row(label, p) + extra]))
            if a.ep > 1 or a.cp > 1:
                print(f"\nEP all-to-all {gb(p['comm']['ep']['bytes'])} GB/step ({p['comm']['ep']['s'] * 1e3:,.1f} ms); "
                      f"CP ring {gb(p['comm']['cp']['bytes'])} GB/step ({p['comm']['cp']['s'] * 1e3:,.1f} ms)")
    if a.infer:
        rows, data = [], []
        for tp in parse_list(a.tp):
            for b in parse_list(a.batch):
                r = tp_decode(m, a.precision, tp, b, a.ctx, a.link, a.gpu if a.gpu in am.GPUS else "H100-SXM")
                data.append(r)
                rows.append([tp, b, "yes" if r["fits"] else "NO", f"{r['roofline_ms']:.2f}", f"{r['allreduce_ms']:.2f}",
                             f"{r['step_ms']:.2f}", f"{r['tok_s_per_seq']:,.0f}", f"{r['tok_s_per_gpu']:,.0f}"])
        if a.json:
            print(json.dumps(data, indent=2))
        else:
            print(f"TP decode for {a.model} ({a.precision}) at {a.ctx:,} context over {a.link} on {a.gpu}:")
            print(md(["TP", "batch", "fits", "weight+KV read ms", "all-reduce ms", "step ms", "tok/s per seq",
                      "tok/s per GPU"], rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
