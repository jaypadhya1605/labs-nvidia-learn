"""Render each cards-src/*.html card to ../site/img/cards/<name>.png with headless Edge, then trim the page.

Usage:  python render.py            (all cards)
        python render.py 01 03      (cards whose filename starts with these prefixes)
"""
import subprocess
import sys
from pathlib import Path

from PIL import Image

EDGE = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
SRC = Path(__file__).resolve().parent
OUT = SRC.parent / "site" / "img" / "cards"
WIDTH, MAX_H, SCALE = 1600, 4000, 2


def render(html: Path) -> Path:
    tmp = SRC / (html.stem + ".raw.png")
    subprocess.run(
        [EDGE, "--headless=new", "--disable-gpu", "--hide-scrollbars",
         f"--force-device-scale-factor={SCALE}", f"--window-size={WIDTH},{MAX_H}",
         f"--screenshot={tmp}", html.as_uri()],
        check=True, capture_output=True,
    )
    img = Image.open(tmp).convert("RGB")
    w, h = img.size
    px = img.load()
    # The body is magenta; the .page element is not. Find the last non-magenta row.
    bottom = h
    for y in range(h - 1, -1, -1):
        r, g, b = px[w // 2, y]
        if not (r > 240 and g < 20 and b > 240):
            bottom = y + 1
            break
    out = OUT / (html.stem + ".png")
    img.crop((0, 0, w, bottom)).save(out, optimize=True)
    tmp.unlink()
    return out


if __name__ == "__main__":
    prefixes = sys.argv[1:]
    for f in sorted(SRC.glob("*.html")):
        if not prefixes or any(f.name.startswith(p) for p in prefixes):
            print(render(f))
