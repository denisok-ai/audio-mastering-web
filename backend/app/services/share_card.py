# @file services/share_card.py
# @description PNG-карточка «после мастеринга» для шеринга.

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


def render_mastering_share_png(job: dict[str, Any]) -> Optional[bytes]:
    """1200×630 PNG с LUFS до/после и пресетом. Без Pillow — заглушка None."""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        logger.warning("Pillow not installed — share card disabled")
        return None

    w, h = 1200, 630
    img = Image.new("RGB", (w, h), (4, 4, 8))
    draw = ImageDraw.Draw(img)
    before = job.get("before_lufs")
    after = job.get("after_lufs")
    style = str(job.get("style") or "standard")

    try:
        font_lg = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 42)
        font_md = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 28)
        font_sm = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 22)
    except OSError:
        font_lg = font_md = font_sm = ImageFont.load_default()

    draw.text((48, 40), "Magic Master", fill=(255, 255, 255), font=font_lg)
    draw.text((48, 120), "Mastered track", fill=(160, 160, 190), font=font_md)

    y = 220
    draw.text((48, y), "Before", fill=(255, 120, 120), font=font_sm)
    btxt = f"{before:.1f} LUFS" if before is not None else "—"
    draw.text((48, y + 32), btxt, fill=(255, 200, 200), font=font_lg)

    draw.text((620, y), "After", fill=(120, 255, 160), font=font_sm)
    atxt = f"{after:.1f} LUFS" if after is not None else "—"
    draw.text((620, y + 32), atxt, fill=(200, 255, 220), font=font_lg)

    draw.text((48, 420), f"Preset: {style}", fill=(124, 58, 237), font=font_md)
    draw.text((48, 480), "magicmaster.pro", fill=(6, 182, 212), font=font_md)

    # Простая «волна» из оригинала (если есть)
    ob = job.get("original_bytes")
    on = job.get("original_filename") or "a.wav"
    if ob:
        try:
            from ..pipeline import load_audio_from_bytes

            audio, sr = load_audio_from_bytes(ob, on)
            if audio.ndim > 1:
                audio = np.mean(audio, axis=1)
            n = min(len(audio), int(sr * 12))
            chunk = max(1, n // 400)
            peaks = []
            for i in range(0, n, chunk):
                peaks.append(float(np.max(np.abs(audio[i : i + chunk]))))
            if peaks:
                mx = max(peaks) or 1.0
                peaks = [p / mx for p in peaks]
                x0, y0, x1, y1 = 48, 300, w - 48, 380
                pw = (x1 - x0) / len(peaks)
                cy = (y0 + y1) // 2
                for i, p in enumerate(peaks):
                    x = int(x0 + i * pw)
                    hh = int((y1 - y0) * 0.45 * p)
                    draw.line([(x, cy - hh), (x, cy + hh)], fill=(108, 75, 255), width=2)
        except Exception:  # noqa: BLE001
            pass

    import io

    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()
