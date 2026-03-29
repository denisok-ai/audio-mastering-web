#!/usr/bin/env python3
"""Rebuild mm-logo-lockup.png (+ icon-192/512, favicon) from mm-logo-source.png."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

try:
    from scipy import ndimage
except ImportError:
    ndimage = None

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "mm-logo-source.png"
LOCK = ROOT / "mm-logo-lockup.png"


def main() -> None:
    if not SRC.is_file():
        raise SystemExit(f"Missing {SRC} — drop the exported logo PNG there first.")
    im = Image.open(SRC).convert("RGB")
    a = np.array(im)
    h, w = a.shape[:2]
    r = a[:, :, 0].astype(np.float32)
    g = a[:, :, 1].astype(np.float32)
    b = a[:, :, 2].astype(np.float32)
    mx = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)
    chroma = mx - mn
    lum = (r + g + b) / 3.0
    bg = (lum > 198) & (chroma < 44)
    fg = ~bg
    if ndimage:
        fg = ndimage.binary_fill_holes(fg)
        k = np.ones((3, 3), dtype=bool)
        fg = ndimage.binary_opening(fg, structure=k, iterations=1)
        fg = ndimage.binary_closing(fg, structure=k, iterations=1)
    ys, xs = np.where(fg)
    pad = 12
    x0, x1 = max(0, xs.min() - pad), min(w, xs.max() + pad + 1)
    y0, y1 = max(0, ys.min() - pad), min(h, ys.max() + pad + 1)
    crop = a[y0:y1, x0:x1].copy()
    alpha = (fg[y0:y1, x0:x1].astype(np.uint8) * 255)
    rgba = np.dstack([crop, alpha])
    Image.fromarray(rgba, "RGBA").save(LOCK, optimize=True)
    print("Wrote", LOCK, Image.open(LOCK).size)

    lock = Image.open(LOCK).convert("RGBA")
    lw, lh = lock.size
    iw = int(lw * 0.34)
    icon = lock.crop((0, 0, iw, lh))
    side = max(iw, lh)
    sq = Image.new("RGBA", (side, side), (0, 0, 0, 0))
    ox, oy = (side - iw) // 2, (side - lh) // 2
    sq.paste(icon, (ox, oy), icon)
    for s in (192, 512):
        sq.resize((s, s), Image.Resampling.LANCZOS).save(ROOT / f"icon-{s}.png", optimize=True)
    sq.resize((32, 32), Image.Resampling.LANCZOS).save(ROOT.parent / "favicon.ico", format="ICO", sizes=[(32, 32)])
    print("Updated icon-192.png, icon-512.png, ../favicon.ico")


if __name__ == "__main__":
    main()
