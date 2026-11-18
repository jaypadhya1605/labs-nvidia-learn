#!/usr/bin/env python3
"""M8 - image resolution vs tokens vs TTFT vs accuracy, measured against a live VLM (🔁 Lab 01 sweeps, M5 TTFT).

For each target size (long side in pixels) the sweep resizes ONE receipt photo, then:
  1. token check: prompt_tokens(with image) - prompt_tokens(same text, no image) = image tokens + 2 wrappers;
     compared with the exact prediction from vlm/vlm_tokens.py (Qwen2.5-VL smart_resize)
  2. TTFT: streamed requests, first content token; a random request id at the top of the system prompt defeats
     prefix caching, so every request pays the full vision encoder + prefill (--cache keeps it: the 🔁 M5 lesson)
  3. accuracy: does the extracted total still match the receipt at this resolution?

Usage (from the repo root, VLM running: engines/launch.sh vllm vlm):
  python3 -m vlm.vlm_sweep --dry-run                                  # predictions only, no server needed
  python3 -m vlm.vlm_sweep --base-url http://localhost:8000/v1 --model ttg-llm --label vllm-qwen25vl
  python3 -m vlm.vlm_sweep --sizes 448,896,1344,1792,2688 --repeats 8 --source R-S0003 --scale 3 --noise 0.3
  python3 -m vlm.vlm_sweep --cache --label vllm-qwen25vl-cache        # same image every time: cache hits
Output: results/vlm/sweep-<label>.csv and a markdown table.
"""
import argparse
import csv
import json
import os
import statistics
import sys
import time
import uuid

if __package__ in (None, ""):          # run as a file: put the repo root on sys.path
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vlm import extract_receipt as ex  # noqa: E402
from vlm import vlm_tokens as vt  # noqa: E402

FIELDS = ["long_side", "width", "height", "predicted_image_tokens", "measured_image_tokens", "prompt_tokens",
          "ttft_p50_ms", "ttft_p95_ms", "latency_p50_ms", "total_ok_rate", "predicted_ttft_floor_ms", "repeats"]


def source_image(ref, scale, noise, workdir):
    """Render one golden-set receipt big (a phone-camera crop), so every sweep size is a DOWNscale."""
    from ttg import world
    from vlm.make_receipts import label, render_receipt
    sc = next(s for s in world.make_scenarios(200, seed=7) if s.get("receipt") and s["receipt"]["receipt_ref"] == ref)
    img = render_receipt(sc["receipt"], airport=sc.get("origin", "DTW"), noise=noise, scale=scale)
    return img, label(sc)


def resized(img, long_side, workdir, ref):
    from PIL import Image
    w, h = img.size
    k = long_side / max(w, h)
    out = img.resize((max(1, round(w * k)), max(1, round(h * k))), Image.LANCZOS)
    path = os.path.join(workdir, f"{ref}-{long_side}.png")
    out.save(path)
    return path, out.size


def stream_once(client, url, body, headers, timeout_s):
    """POST a streaming chat completion. Returns (ttft_s, latency_s, text, usage)."""
    t0 = time.perf_counter()
    ttft, parts, usage = None, [], {}
    with client.stream("POST", url, json=body, headers=headers, timeout=timeout_s) as r:
        if r.status_code >= 400:
            raise RuntimeError(f"HTTP {r.status_code}: {r.read()[:300]!r}")
        for line in r.iter_lines():
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            chunk = json.loads(data)
            if chunk.get("usage"):
                usage = chunk["usage"]
            for ch in chunk.get("choices") or []:
                piece = (ch.get("delta") or {}).get("content") or ""
                if piece:
                    if ttft is None:
                        ttft = time.perf_counter() - t0
                    parts.append(piece)
    return ttft, time.perf_counter() - t0, "".join(parts), usage


def prompt_tokens(client, url, body, headers, timeout_s):
    b = dict(body, max_tokens=1, stream=False)
    b.pop("stream_options", None)
    r = client.post(url, json=b, headers=headers, timeout=timeout_s)
    if r.status_code >= 400:
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")
    return int(r.json()["usage"]["prompt_tokens"])


def predict(sizes, src_w, src_h, max_pixels):
    rows = []
    for s in sizes:
        k = s / max(src_w, src_h)
        w, h = max(1, round(src_w * k)), max(1, round(src_h * k))
        c = vt.cost(w, h, max_pixels)
        rows.append({"long_side": s, "width": w, "height": h, "predicted_image_tokens": c["qwen"]["tokens"],
                     "predicted_ttft_floor_ms": round(c["qwen"]["ttft_floor_ms"], 1),
                     "image_kv_mib": round(c["qwen"]["image_kv_mib"], 1)})
    return rows


def sweep(base_url, model, api_key, sizes, repeats, ref, scale, noise, cache, workdir, timeout_s, max_pixels,
          http_client=None):
    img, truth = source_image(ref, scale, noise, workdir)
    os.makedirs(workdir, exist_ok=True)
    client, owned = ex._client(http_client, timeout_s)
    url = ex.chat_url(base_url)
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    rows = []
    try:
        for p in predict(sizes, img.width, img.height, max_pixels):
            path, (w, h) = resized(img, p["long_side"], workdir, ref)
            with_img = ex.build_request(path, model, max_tokens=400, stream=True)
            no_img = ex.build_request(path, model, max_tokens=400, stream=True, image_mode="none")
            pt_img = prompt_tokens(client, url, with_img, headers, timeout_s)
            pt_txt = prompt_tokens(client, url, no_img, headers, timeout_s)
            ttfts, lats, oks = [], [], []
            for _ in range(repeats):
                body = with_img if cache else ex.build_request(path, model, max_tokens=400, stream=True,
                                                               nonce=uuid.uuid4().hex[:8])
                ttft, lat, text, _usage = stream_once(client, url, body, headers, timeout_s)
                ttfts.append((ttft or lat) * 1e3)
                lats.append(lat * 1e3)
                try:
                    fields = ex.normalize(ex.parse_json_reply(text))
                    oks.append(ex.score_one(fields, truth)["total_usd"])
                except ValueError:
                    oks.append(False)
            rows.append({**p, "width": w, "height": h, "measured_image_tokens": pt_img - pt_txt - vt.QWEN_WRAPPER_TOKENS,
                         "prompt_tokens": pt_img, "ttft_p50_ms": round(statistics.median(ttfts), 1),
                         "ttft_p95_ms": round(ex.pct(ttfts, 0.95), 1), "latency_p50_ms": round(statistics.median(lats), 1),
                         "total_ok_rate": sum(oks) / len(oks), "repeats": repeats})
    finally:
        if owned:
            client.close()
    return rows


def md(rows, measured=True):
    if measured:
        head = ["long side px", "sent (w x h)", "predicted tokens", "measured tokens", "TTFT p50 ms", "TTFT p95 ms",
                "floor ms", "total read right"]
        body = [[r["long_side"], f"{r['width']} x {r['height']}", f"{r['predicted_image_tokens']:,}",
                 f"{r['measured_image_tokens']:,}", f"{r['ttft_p50_ms']:,.0f}", f"{r['ttft_p95_ms']:,.0f}",
                 f"{r['predicted_ttft_floor_ms']:,.0f}", f"{r['total_ok_rate']:.0%}"] for r in rows]
    else:
        head = ["long side px", "sent (w x h)", "predicted image tokens", "image KV MiB", "TTFT floor ms (H100, 50% MFU)"]
        body = [[r["long_side"], f"{r['width']} x {r['height']}", f"{r['predicted_image_tokens']:,}",
                 f"{r['image_kv_mib']:,.1f}", f"{r['predicted_ttft_floor_ms']:,.0f}"] for r in rows]
    out = ["| " + " | ".join(head) + " |", "|" + "|".join("---" for _ in head) + "|"]
    return "\n".join(out + ["| " + " | ".join(str(c) for c in r) + " |" for r in body])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", default=os.environ.get("TTG_VLM_BASE_URL", "http://localhost:8000/v1"))
    ap.add_argument("--model", default=os.environ.get("TTG_VLM_MODEL", "ttg-llm"))
    ap.add_argument("--api-key-env", default="TTG_VLM_API_KEY", help="NAME of the env var holding the key")
    ap.add_argument("--api-key", default=None, help='the key itself, e.g. "$NVIDIA_API_KEY"')
    ap.add_argument("--sizes", default="448,896,1344,1792", help="long side in pixels, comma list")
    ap.add_argument("--repeats", type=int, default=8)
    ap.add_argument("--source", default="R-S0003", help="golden-set receipt to render")
    ap.add_argument("--scale", type=float, default=3.0, help="render scale of the source (3 = ~1,900 px wide)")
    ap.add_argument("--noise", type=float, default=0.3)
    ap.add_argument("--max-pixels", type=int, default=vt.QWEN_MAX_PIXELS, help="what the server uses (prediction only)")
    ap.add_argument("--cache", action="store_true", help="identical requests: let prefix caching hit")
    ap.add_argument("--timeout-s", type=float, default=120.0)
    ap.add_argument("--label", default="vlm")
    ap.add_argument("--dry-run", action="store_true", help="predictions only")
    a = ap.parse_args(argv)
    sizes = [int(x) for x in a.sizes.split(",") if x.strip()]
    workdir = os.path.join("results", "vlm", "sweep-images")
    if a.dry_run:
        img, _ = source_image(a.source, a.scale, a.noise, workdir)
        print(f"Source {a.source} rendered at scale {a.scale}: {img.width} x {img.height} px "
              f"(Qwen2.5-VL max_pixels {a.max_pixels:,})\n")
        print(md(predict(sizes, img.width, img.height, a.max_pixels), measured=False))
        return 0
    rows = sweep(a.base_url, a.model, a.api_key or os.environ.get(a.api_key_env, "not-used"), sizes, a.repeats, a.source, a.scale,
                 a.noise, a.cache, workdir, a.timeout_s, a.max_pixels)
    print(md(rows))
    out = os.path.join("results", "vlm", f"sweep-{a.label}.csv")
    with open(out, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        wr.writeheader()
        wr.writerows(rows)
    print(f"\n-> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
