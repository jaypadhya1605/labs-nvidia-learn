#!/usr/bin/env python3
"""M8 - how many tokens is an image? Exact for Qwen2.5-VL and Llama 3.2 Vision, then what those tokens cost.

Qwen2.5-VL (decoder-only VLM, self-hosted on vLLM): the vision encoder cuts the image into 14 x 14 pixel patches and
merges 2 x 2 patches into one token, so ONE TOKEN = 28 x 28 PIXELS after smart_resize (each side rounded to a multiple
of 28, total pixels kept between min_pixels and max_pixels). Image tokens sit in the text sequence: they fill the KV
cache of all 28 decoder layers and pay full prefill, exactly like text (🔁 Lab 02 KV math).
Llama 3.2 Vision (cross-attention VLM, hosted): the image is fitted onto a canvas of 560 x 560 tiles (up to 4). Each
tile is 40 x 40 patches + 1 class token = 1,601 vision tokens. The text model sees ONE <|image|> token and reads the
vision tokens through 8 cross-attention layers, so they never enter the 32 self-attention layers' KV cache.

Usage (from the repo root):
  python3 -m vlm.vlm_tokens --size 640x662 --size 3024x4032          # width x height, both models
  python3 -m vlm.vlm_tokens --size 3024x4032 --max-pixels 1003520    # vLLM --mm-processor-kwargs '{"max_pixels": ...}'
  python3 -m vlm.vlm_tokens --table                                  # the docs/06 table (receipt, phone photo, 4K)
  python3 -m vlm.vlm_tokens --table --json
Sources: Qwen2-VL image processor smart_resize (transformers), Qwen/Qwen2.5-VL-7B-Instruct preprocessor_config.json,
Mllama image processor get_optimal_tiled_canvas (transformers), meta-llama/Llama-3.2-11B-Vision-Instruct config.json.
"""
import argparse
import json
import math
import os
import sys

if __package__ in (None, ""):          # run as a file: put the repo root on sys.path
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from archmath import archmath as am  # noqa: E402

# Qwen/Qwen2.5-VL-7B-Instruct preprocessor_config.json (what vLLM uses unless --mm-processor-kwargs overrides it)
QWEN_PATCH, QWEN_MERGE = 14, 2
QWEN_FACTOR = QWEN_PATCH * QWEN_MERGE                  # 28 pixels per token side
QWEN_MIN_PIXELS = 3136                                  # 56 x 56
QWEN_MAX_PIXELS = 12845056                              # checkpoint default: 16,384 tokens x 784 pixels
QWEN_CLASS_MAX_PIXELS = 1003520                         # image-processor class default: 1,280 tokens
QWEN_WRAPPER_TOKENS = 2                                 # <|vision_start|> ... <|vision_end|> around the pad tokens
QWEN_VIT_WINDOW = 112 // QWEN_PATCH                     # window attention: 8 x 8 patches per window
QWEN_VIT_FULL_BLOCKS, QWEN_VIT_BLOCKS, QWEN_VIT_HIDDEN = 4, 32, 1280

# meta-llama/Llama-3.2-11B-Vision-Instruct vision_config
MLLAMA_TILE = 560
MLLAMA_MAX_TILES = 4
MLLAMA_TOKENS_PER_TILE = (560 // 14) ** 2 + 1           # 40 x 40 patches + 1 class token = 1,601

# A few image shapes the Tern agent meets (width x height)
EXAMPLES = [
    ("Tern receipt render (vlm/make_receipts.py, 4 items)", 640, 662),
    ("Same receipt resized to 448 px long side", 433, 448),
    ("Phone photo of a receipt, 12 MP", 3024, 4032),
    ("Same photo resized to 1344 px long side", 1008, 1344),
    ("Boarding-pass screenshot (phone, 1170 x 2532)", 1170, 2532),
    ("4K video frame (3840 x 2160)", 3840, 2160),
]


# ----------------------------------------------------------------------------------------------- Qwen2.5-VL
def qwen_smart_resize(height, width, factor=QWEN_FACTOR, min_pixels=QWEN_MIN_PIXELS, max_pixels=QWEN_MAX_PIXELS):
    """transformers Qwen2-VL smart_resize: both sides multiples of `factor`, pixels in [min, max], aspect kept."""
    if max(height, width) / min(height, width) > 200:
        raise ValueError(f"absolute aspect ratio must be smaller than 200, got {max(height, width) / min(height, width)}")
    h_bar = round(height / factor) * factor             # Python round(): half to even, same as the processor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def qwen_image_tokens(width, height, min_pixels=QWEN_MIN_PIXELS, max_pixels=QWEN_MAX_PIXELS):
    h, w = qwen_smart_resize(height, width, min_pixels=min_pixels, max_pixels=max_pixels)
    tokens = (h // QWEN_FACTOR) * (w // QWEN_FACTOR)
    return {"resized_w": w, "resized_h": h, "tokens": tokens, "prompt_tokens": tokens + QWEN_WRAPPER_TOKENS,
            "vit_patches": tokens * QWEN_MERGE * QWEN_MERGE}


def qwen_vit_flops(patches, vit_params):
    """Vision-encoder FLOPs (estimate): 2 x params x patches + attention (4 full-attention blocks, 28 windowed)."""
    linear = 2 * vit_params * patches
    full = QWEN_VIT_FULL_BLOCKS * 4 * patches * patches * QWEN_VIT_HIDDEN
    window = (QWEN_VIT_BLOCKS - QWEN_VIT_FULL_BLOCKS) * 4 * patches * min(patches, QWEN_VIT_WINDOW ** 2) * QWEN_VIT_HIDDEN
    return linear + full + window


# ----------------------------------------------------------------------------------------------- Llama 3.2 Vision
def mllama_aspect_ratios(max_tiles=MLLAMA_MAX_TILES):
    """(a, b) tile arrangements with a x b <= max_tiles - for 4: 1x1 1x2 1x3 1x4 2x1 2x2 3x1 4x1."""
    return [(a, b) for a in range(1, max_tiles + 1) for b in range(1, max_tiles + 1) if a * b <= max_tiles]


def mllama_canvas(width, height, tile=MLLAMA_TILE, max_tiles=MLLAMA_MAX_TILES):
    """transformers Mllama get_optimal_tiled_canvas -> (canvas_w, canvas_h).

    Scale = the limiting side's ratio for every candidate canvas. Prefer the SMALLEST upscale (scale >= 1); if the
    image is larger than every canvas, take the LARGEST downscale. Ties go to the canvas with the least area (padding).
    """
    canvases = [(a * tile, b * tile) for a, b in mllama_aspect_ratios(max_tiles)]     # (canvas_h, canvas_w)
    scales = [min(ch / height, cw / width) for ch, cw in canvases]
    up = [s for s in scales if s >= 1]
    chosen = min(up) if up else max(s for s in scales if s < 1)
    ties = [c for c, s in zip(canvases, scales) if s == chosen]
    ch, cw = min(ties, key=lambda c: c[0] * c[1])
    return cw, ch


def mllama_image_tokens(width, height, tile=MLLAMA_TILE, max_tiles=MLLAMA_MAX_TILES):
    cw, ch = mllama_canvas(width, height, tile, max_tiles)
    tiles = (cw // tile) * (ch // tile)
    return {"canvas_w": cw, "canvas_h": ch, "tiles": tiles, "vision_tokens": tiles * MLLAMA_TOKENS_PER_TILE,
            "text_tokens": 1}


# ----------------------------------------------------------------------------------------------- generic tiling
def tiled_tokens(width, height, tile=448, tokens_per_tile=256, max_tiles=12, thumbnail=True):
    """APPROXIMATE dynamic tiling (InternVL / NVLM-D / LLaVA-NeXT style): ceil-cover the image with tiles (capped),
    plus one thumbnail tile. Each model's processor differs in detail - use it for order of magnitude only."""
    cols, rows = math.ceil(width / tile), math.ceil(height / tile)
    while cols * rows > max_tiles:
        if cols >= rows:
            cols -= 1
        else:
            rows -= 1
    tiles = cols * rows + (1 if thumbnail and cols * rows > 1 else 0)
    return {"tiles": tiles, "tokens": tiles * tokens_per_tile}


# ----------------------------------------------------------------------------------------------- what it costs
def cost(width, height, max_pixels=QWEN_MAX_PIXELS, text_tokens=180, gpu="H100-SXM", mfu=0.5):
    """Tokens, KV bytes and a TTFT floor for one image + a short extraction prompt on each VLM."""
    qm = am.analyze(am.load_config("qwen2.5-vl-7b"))
    mm = am.analyze(am.load_config("llama-3.2-11b-vision"))
    q = qwen_image_tokens(width, height, max_pixels=max_pixels)
    seq = q["prompt_tokens"] + text_tokens
    vit = qwen_vit_flops(q["vit_patches"], qm["vision_params"])
    llm = am.prefill_flops(qm, seq)
    peak = am.peak_tflops(gpu, "bf16") * 1e12 * mfu
    ml = mllama_image_tokens(width, height)
    return {
        "width": width, "height": height, "max_pixels": max_pixels,
        "qwen": {**q, "kv_mib": seq * qm["kv_bytes_per_token_bf16"] / 2 ** 20,
                 "image_kv_mib": q["tokens"] * qm["kv_bytes_per_token_bf16"] / 2 ** 20,
                 "vit_tflop": vit / 1e12, "llm_prefill_tflop": llm / 1e12,
                 "ttft_floor_ms": (vit + llm) / peak * 1e3},
        "mllama": {**ml, "xattn_kv_mib": ml["vision_tokens"] * mm["xattn_kv_bytes_per_vision_token_bf16"] / 2 ** 20,
                   "text_kv_mib": (text_tokens + 1) * mm["kv_bytes_per_token_bf16"] / 2 ** 20},
        "assumptions": f"{text_tokens} text tokens around the image; {gpu} BF16 at {mfu:.0%} MFU; KV in BF16",
    }


def md(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    return "\n".join(out + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows])


def table(sizes, max_pixels=QWEN_MAX_PIXELS, text_tokens=180):
    q_rows, m_rows, data = [], [], []
    for label, w, h in sizes:
        c = cost(w, h, max_pixels, text_tokens)
        data.append({"label": label, **c})
        q, m = c["qwen"], c["mllama"]
        q_rows.append([label, f"{w} x {h}", f"{q['resized_w']} x {q['resized_h']}", f"{q['tokens']:,}",
                       f"{q['image_kv_mib']:,.1f}", f"{q['vit_tflop']:,.1f}", f"{q['llm_prefill_tflop']:,.1f}",
                       f"{q['ttft_floor_ms']:,.0f}"])
        m_rows.append([label, f"{m['canvas_w']} x {m['canvas_h']}", m["tiles"], f"{m['vision_tokens']:,}",
                       f"{m['xattn_kv_mib']:,.1f}", m["text_tokens"]])
    text = (f"Qwen2.5-VL-7B (max_pixels {max_pixels:,} = {max_pixels // 784:,} tokens max; {text_tokens} text tokens "
            f"of prompt; TTFT floor on H100 BF16 at 50% MFU):\n\n"
            + md(["image", "pixels (w x h)", "smart_resize", "image tokens", "image KV MiB", "ViT TFLOP",
                  "LLM prefill TFLOP", "TTFT floor ms"], q_rows)
            + "\n\nLlama 3.2 11B Vision (560 x 560 tiles, max 4, 1,601 vision tokens per tile, 8 cross-attention layers):\n\n"
            + md(["image", "canvas (w x h)", "tiles", "vision tokens", "cross-attn KV MiB", "tokens in the text sequence"],
                 m_rows))
    return text, data


def parse_size(s):
    w, h = s.lower().replace(" ", "").split("x")
    return int(w), int(h)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--size", action="append", default=[], help="WIDTHxHEIGHT in pixels (repeatable)")
    ap.add_argument("--table", action="store_true", help="the docs/06 example table")
    ap.add_argument("--max-pixels", type=int, default=QWEN_MAX_PIXELS,
                    help=f"Qwen max_pixels (checkpoint {QWEN_MAX_PIXELS:,}; class default {QWEN_CLASS_MAX_PIXELS:,})")
    ap.add_argument("--text-tokens", type=int, default=180, help="prompt text around the image")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    sizes = [(f"{w} x {h}", w, h) for w, h in map(parse_size, a.size)]
    if a.table or not sizes:
        sizes = EXAMPLES + sizes
    text, data = table(sizes, a.max_pixels, a.text_tokens)
    print(json.dumps(data, indent=2) if a.json else text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
