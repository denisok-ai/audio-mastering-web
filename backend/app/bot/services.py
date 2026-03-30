"""Синхронные операции для бота: мастеринг и анализ аудио."""
from __future__ import annotations

import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Any, Optional, Tuple

import numpy as np

from ..helpers import check_audio_magic_bytes
from ..mastering_trace import TraceContext, trace_job_done, trace_job_error, trace_job_start
from ..pipeline import (
    STYLE_CONFIGS,
    compute_spectrum_bars,
    export_audio,
    load_audio_from_bytes,
    measure_lufs,
    measure_stereo_correlation,
    run_mastering_pipeline,
)


TELEGRAM_MAX_DOWNLOAD_MB = 20


def run_mastering_sync(
    data: bytes,
    filename: str,
    style: str = "standard",
    target_lufs: Optional[float] = None,
    out_format: str = "wav",
) -> Tuple[bytes, str, Optional[float], Optional[float]]:
    """Полный проход мастеринга; возвращает (out_bytes, out_filename, before_lufs, after_lufs)."""
    if target_lufs is None:
        style_key = style.lower() if style.lower() in STYLE_CONFIGS else "standard"
        target_lufs = float(STYLE_CONFIGS[style_key].get("lufs", -14.0))
    style_key = style.lower() if style.lower() in STYLE_CONFIGS else "standard"
    trace_job_id = uuid.uuid4().hex[:12]
    ctx = TraceContext.build(
        trace_job_id,
        filename or "audio.wav",
        "telegram",
        style=style_key,
        target_lufs=target_lufs,
    )
    trace_job_start(ctx)
    try:
        audio, sr = load_audio_from_bytes(data, filename or "audio.wav")

        def _noop(_p: int, _m: str) -> None:
            pass

        before = measure_lufs(audio, sr)
        mastered = run_mastering_pipeline(
            audio,
            sr,
            target_lufs=target_lufs,
            style=style_key,
            progress_callback=_noop,
            trace_ctx=ctx,
        )
        after = measure_lufs(mastered, sr)
        channels = 1 if mastered.ndim == 1 else mastered.shape[1]
        out_bytes = export_audio(mastered, sr, channels, out_format.lower())
        base = (filename or "track").rsplit(".", 1)[0]
        ext = "m4a" if out_format.lower() == "aac" else out_format.lower()
        out_name = f"{base}_mastered.{ext}"
        trace_job_done(
            ctx,
            before_lufs=float(before) if before == before else None,
            after_lufs=float(after) if after == after else None,
            out_format=out_format.lower(),
        )
        return out_bytes, out_name, float(before) if before == before else None, float(after) if after == after else None
    except Exception as e:
        trace_job_error(ctx, e)
        raise


def analyze_audio_sync(data: bytes, filename: str) -> dict[str, Any]:
    """Анализ как для AI/измерений (без LLM)."""
    audio, sr = load_audio_from_bytes(data, filename or "wav")
    try:
        lufs = measure_lufs(audio, sr)
    except Exception:
        lufs = float("nan")
    peak = float(np.max(np.abs(audio))) if audio.size > 0 else 0.0
    peak_dbfs = float(20 * np.log10(max(peak, 1e-12)))
    duration_sec = float(len(audio) / sr)
    channels = 1 if audio.ndim == 1 else int(audio.shape[1])
    correlation = None
    if channels == 2 and audio.ndim == 2:
        correlation = measure_stereo_correlation(audio)
    result: dict[str, Any] = {
        "lufs": round(lufs, 2) if not np.isnan(lufs) else None,
        "peak_dbfs": round(peak_dbfs, 2),
        "duration_sec": round(duration_sec, 3),
        "sample_rate": sr,
        "channels": channels,
        "stereo_correlation": round(correlation, 4) if correlation is not None else None,
    }
    if audio.size >= 4096:
        try:
            result["spectrum_bars"] = compute_spectrum_bars(audio, sr)
        except Exception:
            pass
    return result


def validate_telegram_audio(data: bytes, filename: str, max_mb: int = TELEGRAM_MAX_DOWNLOAD_MB) -> Optional[str]:
    """None если ок, иначе текст ошибки."""
    if len(data) > max_mb * 1024 * 1024:
        return f"file_too_large:{max_mb}"
    ext = (filename or "").lower().split(".")[-1] if "." in (filename or "") else ""
    if ext not in ("wav", "mp3", "flac", "ogg", "oga"):
        return "bad_format"
    if ext in ("mp3",) and not shutil.which("ffmpeg"):
        return "need_ffmpeg"
    if not check_audio_magic_bytes(data, filename or "x.wav"):
        return "bad_magic"
    return None


def convert_voice_to_wav_if_needed(data: bytes, filename: str) -> Tuple[bytes, str]:
    """Голосовые OGG/OPUS -> WAV через ffmpeg."""
    low = (filename or "").lower()
    if not low.endswith((".ogg", ".oga")) and "audio/ogg" not in low:
        return data, filename
    if not shutil.which("ffmpeg"):
        return data, filename
    with tempfile.TemporaryDirectory(prefix="mm_tg_") as tmp:
        src = Path(tmp) / "in.ogg"
        dst = Path(tmp) / "out.wav"
        src.write_bytes(data)
        import subprocess

        r = subprocess.run(
            ["ffmpeg", "-y", "-i", str(src), str(dst)],
            capture_output=True,
            timeout=120,
        )
        if r.returncode != 0 or not dst.is_file():
            return data, filename
        return dst.read_bytes(), "voice.wav"
