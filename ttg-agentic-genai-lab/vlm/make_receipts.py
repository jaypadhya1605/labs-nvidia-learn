#!/usr/bin/env python3
"""M8 - render the golden set's receipts as images, so a VLM has something real to read (synthetic data only).

Every receipt comes from ttg.world (seed 7: 66 receipts - 45 meal, 21 taxi). The renderer draws what a thermal
printer prints: merchant, location, date, items, tax, TOTAL. No card digits are ever printed (PCI: card numbers never
reach the LLM). The ground truth goes to labels.jsonl next to the images, so vlm/extract_receipt.py can score a VLM.

Usage (from the repo root):
  python3 -m vlm.make_receipts                          # data/receipts/R-S0001.png ... + labels.jsonl (seed 7)
  python3 -m vlm.make_receipts --noise 0.6              # skew, blur, sensor noise: a phone photo, not a scan
  python3 -m vlm.make_receipts --ref R-S0003 --out /tmp/r --scale 2
  python3 -m vlm.make_receipts --n 1000 --seed 11 --out data/receipts-train
The MCP guest server reads data/receipts/<ref>.png when TTG_VLM_BASE_URL is set (ttg/mcp/guest_server.py).
Needs Pillow (deploy/requirements-runtime.txt).
"""
import argparse
import hashlib
import json
import os
import random
import sys
from datetime import date

if __package__ in (None, ""):          # run as a file: put the repo root on sys.path
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

FONT_FILES = ("DejaVuSansMono.ttf", "DejaVuSansMono-Bold.ttf")
WIDTH = 640                     # ~80 mm thermal paper at 203 dpi is 576 dots; a little margin around it
COLS = 45                       # characters per line when no font is measured (render_receipt measures it)
LOCATIONS = {"meal": "Terminal {gate} - Gate {num}", "taxi": "Trip to {airport} Airport"}
DATE_FORMATS = ("%Y-%m-%d", "%m/%d/%Y", "%d %b %Y")


def _rng(ref):
    return random.Random(int(hashlib.sha256(("render|" + ref).encode()).hexdigest()[:12], 16))


def _font(size, bold=False):
    from PIL import ImageFont
    try:
        return ImageFont.truetype(FONT_FILES[1 if bold else 0], size)
    except OSError:
        return ImageFont.load_default(size=size)          # Pillow >= 10.1 scales the built-in font


def receipt_lines(receipt, airport="DTW", cols=COLS):
    """The text of a receipt, line by line: [(text, style)] with style in title|center|body|total|small."""
    ref = receipt.get("receipt_ref") or receipt.get("ref") or "R-UNKNOWN"
    rng = _rng(ref)
    cat = receipt.get("category", "meal")
    d = date.fromisoformat(receipt["date"])
    when = f"{d.strftime(rng.choice(DATE_FORMATS))}  {rng.randint(5, 22):02d}:{rng.randint(0, 59):02d}"
    loc = LOCATIONS.get(cat, "{airport}").format(gate=rng.choice("ABC"), num=rng.randint(2, 40), airport=airport)
    order = f"Cab #{rng.randint(100, 999)}" if cat == "taxi" else f"Order #{rng.randint(1000, 9999)}"
    rule = "-" * cols

    def row(left, amount):
        right = f"{amount:,.2f}"
        return f"{left[:cols - len(right) - 1]:<{cols - len(right)}}{right}"

    lines = [(receipt["merchant"].upper(), "title"), (loc, "center"), (when, "center"),
             (f"{ref}  {order}", "center"), (rule, "body")]
    for it in receipt["items"]:
        lines.append((row(it["desc"], it["usd"]), "body"))
    lines.append((rule, "body"))
    subtotal = round(sum(it["usd"] for it in receipt["items"]), 2)
    if receipt.get("tax_usd"):
        lines += [(row("SUBTOTAL", subtotal), "body"), (row("TAX 6%", receipt["tax_usd"]), "body")]
    lines += [(row("TOTAL USD", receipt["total_usd"]), "total"), (rule, "body"),
              ("CARD PAYMENT - APPROVED", "center"), (f"AUTH {rng.randint(100000, 999999)}", "center"),
              ("Thank you for traveling with us!", "small")]
    return lines


def render_receipt(receipt, airport="DTW", noise=0.0, scale=1.0):
    """Pure function: receipt dict (ttg.world shape) -> PIL.Image (grayscale, like a thermal print). Deterministic."""
    from PIL import Image, ImageDraw, ImageFilter
    ref = receipt.get("receipt_ref") or receipt.get("ref") or "R-UNKNOWN"
    rng = _rng(ref + f"|noise{noise:.3f}")
    s = max(0.25, float(scale))
    fonts = {"title": _font(int(30 * s), True), "center": _font(int(20 * s)), "body": _font(int(26 * s * 0.8)),
             "total": _font(int(26 * s * 0.8), True), "small": _font(int(18 * s))}
    w = int(WIDTH * s)
    pad, gap = int(36 * s), int(10 * s)
    cols = int((w - 2 * pad) // fonts["body"].getlength("0"))
    lines = receipt_lines(receipt, airport, cols)
    heights = [fonts[st].getbbox("Ag")[3] + gap for _, st in lines]
    h = pad * 2 + sum(heights) + int(20 * s)
    img = Image.new("L", (w, h), 248)
    draw = ImageDraw.Draw(img)
    y = pad
    for (text, style), lh in zip(lines, heights):
        f = fonts[style]
        tw = draw.textlength(text, font=f)
        x = (w - tw) / 2 if style in ("title", "center", "small") else pad
        draw.text((x, y), text, fill=25, font=f)
        y += lh + (int(8 * s) if style == "title" else 0)
    if noise > 0:
        n = min(1.0, float(noise))
        sensor = Image.effect_noise((w, h), 40 + 60 * n)
        img = Image.blend(img, sensor, 0.10 * n)
        img = img.filter(ImageFilter.GaussianBlur(radius=0.9 * n))
        angle = rng.uniform(-3.0, 3.0) * n
        img = img.rotate(angle, resample=Image.BICUBIC, expand=True, fillcolor=60)
    return img


def label(scenario):
    """Ground truth for one receipt (what a perfect read returns) + what the policy then pays."""
    r = scenario["receipt"]
    return {"receipt_ref": r["receipt_ref"], "scenario_id": scenario["scenario_id"], "merchant": r["merchant"],
            "date": r["date"], "category": r["category"], "items": r["items"], "tax_usd": r["tax_usd"],
            "total_usd": r["total_usd"], "party_size": scenario["party_size"],
            "controllable": scenario["controllable"],
            "reimbursement_usd": round(min(r["total_usd"], 50.0 * scenario["party_size"]), 2)}


def generate(n=200, seed=7, out="data/receipts", noise=0.0, scale=1.0, refs=None):
    from ttg import world                                   # lazy: the renderer itself only needs Pillow
    scenarios = [s for s in world.make_scenarios(n, seed=seed) if s.get("receipt")]
    if refs:
        scenarios = [s for s in scenarios if s["receipt"]["receipt_ref"] in refs]
    os.makedirs(out, exist_ok=True)
    rows = []
    for sc in scenarios:
        img = render_receipt(sc["receipt"], airport=sc.get("origin", "DTW"), noise=noise, scale=scale)
        path = os.path.join(out, f"{sc['receipt']['receipt_ref']}.png")
        img.save(path, optimize=True)
        rows.append({**label(sc), "image": os.path.basename(path), "width": img.width, "height": img.height,
                     "bytes": os.path.getsize(path), "noise": noise})
    with open(os.path.join(out, "labels.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=200, help="scenarios to draw (seed 7, 200 = the golden set)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="data/receipts")
    ap.add_argument("--noise", type=float, default=0.0, help="0 = clean scan, 1 = skewed, blurred, noisy photo")
    ap.add_argument("--scale", type=float, default=1.0, help="2 = twice the pixels (a sharper photo)")
    ap.add_argument("--ref", action="append", default=[], help="only these receipt refs (repeatable)")
    a = ap.parse_args(argv)
    rows = generate(a.n, a.seed, a.out, a.noise, a.scale, set(a.ref) or None)
    if not rows:
        print("no receipts matched")
        return 1
    meal = sum(r["category"] == "meal" for r in rows)
    ws = sorted(r["width"] for r in rows)
    hs = sorted(r["height"] for r in rows)
    kb = sorted(r["bytes"] / 1024 for r in rows)
    print(f"{len(rows)} receipts ({meal} meal, {len(rows) - meal} taxi) -> {a.out}/ + labels.jsonl")
    print(f"size {ws[0]}-{ws[-1]} x {hs[0]}-{hs[-1]} px, {kb[0]:,.0f}-{kb[-1]:,.0f} KB PNG, noise {a.noise}")
    print(f"next: python3 -m vlm.vlm_tokens --size {ws[-1]}x{hs[-1]}   |   python3 -m vlm.extract_receipt --dir {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
