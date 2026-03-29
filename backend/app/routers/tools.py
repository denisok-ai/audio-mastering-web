# @file routers/tools.py
# @description Публичные инструменты: бесплатный LUFS-анализ.

from __future__ import annotations

import logging
import time
from collections import defaultdict, deque
from typing import Optional

import numpy as np
from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from pydantic import BaseModel, Field
from scipy.signal import resample_poly

from ..config import settings
from ..helpers import allowed_file, check_audio_magic_bytes, get_client_ip
from ..pipeline import compute_lufs_timeline, load_audio_from_bytes, measure_lufs

logger = logging.getLogger(__name__)

router = APIRouter(tags=["tools"])

# IP -> deque of unix timestamps (последний час)
_lufs_hits: dict[str, deque[float]] = defaultdict(deque)
_LUFS_TOOL_MAX_MB = 50


def _check_lufs_rate_limit(ip: str) -> None:
    limit = int(getattr(settings, "lufs_tool_rate_per_hour", 20) or 20)
    now = time.time()
    window_start = now - 3600.0
    q = _lufs_hits[ip]
    while q and q[0] < window_start:
        q.popleft()
    if len(q) >= limit:
        raise HTTPException(
            429,
            f"Слишком много анализов с этого адреса. Лимит: {limit} в час.",
        )
    q.append(now)


def _true_peak_dbfs(audio: np.ndarray, sr: int) -> float:
    """Упрощённый True Peak: 4× oversampling, пик в dBFS."""
    if audio.size == 0:
        return -120.0
    x = np.asarray(audio, dtype=np.float64)
    if x.ndim == 1:
        ups = resample_poly(x, 4, 1)
    else:
        ups = np.column_stack([resample_poly(x[:, ch], 4, 1) for ch in range(x.shape[1])])
    peak = float(np.max(np.abs(ups)))
    return float(20 * np.log10(max(peak, 1e-12)))


def _loudness_range_lu(audio: np.ndarray, sr: int) -> float:
    """Грубая оценка LRA по блокам short-term LUFS."""
    timeline, _ = compute_lufs_timeline(audio, sr, block_sec=3.0, max_points=200)
    vals = [v for v in timeline if v is not None and v > -70]
    if len(vals) < 2:
        return 0.0
    arr = np.array(vals, dtype=np.float64)
    p10, p95 = np.percentile(arr, 10), np.percentile(arr, 95)
    return float(max(0.0, p95 - p10))


class LufsAnalyzeResponse(BaseModel):
    integrated_lufs: float
    true_peak_dbfs: float
    loudness_range_lu: float
    short_term_max_lufs: Optional[float] = None
    sample_rate: int
    channels: int
    duration_sec: float
    bit_depth: Optional[int] = None
    format: str
    spotify_ready: bool
    apple_ready: bool
    youtube_ready: bool
    podcast_ready: bool
    recommendations: list[str] = Field(default_factory=list)


@router.post("/api/tools/lufs-analyze", response_model=LufsAnalyzeResponse)
async def api_lufs_analyze(
    request: Request,
    file: UploadFile = File(...),
):
    """Бесплатный анализ громкости (LUFS, True Peak). Лимит запросов по IP за час."""
    ip = get_client_ip(request)
    _check_lufs_rate_limit(ip)

    fname = file.filename or "audio.wav"
    if not allowed_file(fname):
        raise HTTPException(400, "Формат: WAV, MP3 или FLAC.")
    data = await file.read()
    if len(data) > _LUFS_TOOL_MAX_MB * 1024 * 1024:
        raise HTTPException(400, f"Файл больше {_LUFS_TOOL_MAX_MB} МБ.")
    if not check_audio_magic_bytes(data, fname):
        raise HTTPException(400, "Содержимое не похоже на поддерживаемый аудиоформат.")

    try:
        audio, sr = load_audio_from_bytes(data, fname)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(400, f"Не удалось прочитать аудио: {e}") from e

    integrated = measure_lufs(audio, sr)
    if integrated is None or np.isnan(integrated):
        integrated = -70.0
    tp = _true_peak_dbfs(audio, sr)
    lra = _loudness_range_lu(audio, sr)
    timeline, _ = compute_lufs_timeline(audio, sr, block_sec=0.4, max_points=400)
    st_vals = [v for v in timeline if v is not None]
    st_max = float(max(st_vals)) if st_vals else None

    ch = 1 if audio.ndim == 1 else int(audio.shape[1])
    duration = float(len(audio) / sr)
    ext = fname.rsplit(".", 1)[-1].lower() if "." in fname else "wav"

    recs: list[str] = []
    if integrated < -16:
        recs.append(f"Трек относительно тихий ({integrated:.1f} LUFS). Для Spotify часто ориентируются на ~−14 LUFS.")
    elif integrated > -8:
        recs.append(f"Трек очень громкий ({integrated:.1f} LUFS). Платформы могут снизить громкость при воспроизведении.")
    if tp > -0.5:
        recs.append(f"Пик сигнала высокий ({tp:.1f} dBFS). Рекомендуется запас до −1 dBTP при мастеринге.")
    if not recs:
        recs.append("Громкость в разумном диапазоне для многих сценариев.")
    recs.append("Исправить баланс и лимитирование можно в один клик — мастеринг Magic Master.")

    spotify_ok = -16.5 <= integrated <= -11.5
    apple_ok = -18.5 <= integrated <= -13.5
    youtube_ok = -16.5 <= integrated <= -11.5
    podcast_ok = -18.5 <= integrated <= -13.5

    return LufsAnalyzeResponse(
        integrated_lufs=round(float(integrated), 2),
        true_peak_dbfs=round(tp, 2),
        loudness_range_lu=round(lra, 2),
        short_term_max_lufs=round(st_max, 2) if st_max is not None else None,
        sample_rate=int(sr),
        channels=ch,
        duration_sec=round(duration, 2),
        bit_depth=None,
        format=ext.upper(),
        spotify_ready=spotify_ok,
        apple_ready=apple_ok,
        youtube_ready=youtube_ok,
        podcast_ready=podcast_ok,
        recommendations=recs,
    )
