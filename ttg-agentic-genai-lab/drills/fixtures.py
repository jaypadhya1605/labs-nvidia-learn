#!/usr/bin/env python3
"""Fault fixtures for drills 6 and 9 (M13). Plain file operations: no GPU, no network, nothing outside results/.

  python drills/fixtures.py receipts-photo   # data/receipts/*.png -> results/drills/receipts-12mp/ at 3024x4032
  python drills/fixtures.py lora-rank --rank 32
                                          # adapters/ttg-short -> results/drills/adapters-r32/ttg-short with r=32

receipts-photo puts each receipt on a 3024x4032 canvas, a 12 MP phone photo: same text, 15,552 image tokens on
Qwen2.5-VL instead of ~1,000 (docs/06-multimodal.md).
lora-rank copies the M6 adapter and changes only "r" in adapter_config.json. The weights keep their real rank, so this
copy exists to trip the server's max-LoRA-rank check. Never serve it with a raised limit.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def receipts_photo(src: Path, dst: Path, size: tuple[int, int] = (3024, 4032)) -> list[Path]:
    from PIL import Image  # Pillow: installed with the M8 requirements

    files = sorted(src.glob("*.png"))
    if not files:
        raise SystemExit(f"no receipts in {src} - run: python -m vlm.make_receipts (M8)")
    dst.mkdir(parents=True, exist_ok=True)
    out = []
    for f in files:
        with Image.open(f) as im:
            im = im.convert("RGB")
            scale = min(size[0] * 0.9 / im.width, size[1] * 0.9 / im.height)
            im = im.resize((max(1, int(im.width * scale)), max(1, int(im.height * scale))), Image.LANCZOS)
            canvas = Image.new("RGB", size, (214, 210, 200))          # a table top behind the paper
            canvas.paste(im, ((size[0] - im.width) // 2, (size[1] - im.height) // 2))
            canvas.save(dst / f.name)
            out.append(dst / f.name)
    return out


def lora_rank(src: Path, dst: Path, rank: int) -> Path:
    cfg = src / "adapter_config.json"
    if not cfg.is_file():
        raise SystemExit(f"no adapter at {src} - train it first (M6), or pass --src")
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    data = json.loads((dst / "adapter_config.json").read_text(encoding="utf-8"))
    real = data.get("r")
    data["r"] = rank
    (dst / "adapter_config.json").write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    (dst / "DRILL-9-FAKE-RANK.txt").write_text(
        f"Drill 9 fixture: adapter_config.json says r={rank}; the weights are still r={real}. Do not serve it.\n",
        encoding="utf-8")
    return dst


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("receipts-photo", help="drill 6: receipts as full-resolution phone photos")
    r.add_argument("--src", default=str(REPO / "data" / "receipts"))
    r.add_argument("--dst", default=str(REPO / "results" / "drills" / "receipts-12mp"))
    r.add_argument("--size", default="3024x4032", help="WIDTHxHEIGHT (default: a 12 MP phone photo)")
    lr = sub.add_parser("lora-rank", help="drill 9: an adapter whose rank exceeds --max-lora-rank")
    lr.add_argument("--src", default=str(REPO / "adapters" / "ttg-short"))
    lr.add_argument("--dst", default=str(REPO / "results" / "drills" / "adapters-r32" / "ttg-short"))
    lr.add_argument("--rank", type=int, default=32)
    a = ap.parse_args(argv)
    if a.cmd == "receipts-photo":
        w, h = (int(x) for x in a.size.lower().split("x"))
        out = receipts_photo(Path(a.src), Path(a.dst), (w, h))
        print(f"{len(out)} receipts at {w}x{h} -> {a.dst}")
    else:
        print(f"adapter with r={a.rank} in its config -> {lora_rank(Path(a.src), Path(a.dst), a.rank)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
