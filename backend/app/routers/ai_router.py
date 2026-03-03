# @file routers/ai_router.py
# @description AI-эндпоинты: рекомендация пресета, отчёт, NL→config, чат.
# @created 2026-03-01

import logging
import shutil
from typing import List, Optional

import numpy as np
from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile
from pydantic import BaseModel

logger = logging.getLogger(__name__)

from .. import ai as ai_module
from .. import settings_store
from ..database import log_ai_usage
from ..deps import (
    get_ai_identifier,
    get_current_user_optional,
    get_tier_for_ai,
    require_feature_ai,
)
from ..helpers import allowed_file, check_audio_magic_bytes
from ..pipeline import (
    compute_spectrum_bars,
    load_audio_from_bytes,
    measure_lufs,
    measure_stereo_correlation,
)

router = APIRouter()


@router.get("/api/ai/limits")
async def api_ai_limits(
    request: Request,
    user: Optional[dict] = Depends(get_current_user_optional),
):
    """Лимиты AI-запросов по тарифу: Free 5/день, Pro 50, Studio без лимита."""
    require_feature_ai()
    tier = get_tier_for_ai(user, request)
    ident = get_ai_identifier(request, user)
    info = ai_module.check_ai_rate_limit(ident, tier)
    return {
        "tier": tier,
        "ai_used": info["used"],
        "ai_limit": info["limit"],
        "ai_remaining": info["remaining"],
        "ai_reset_at": info["reset_at"],
        "ai_backend": ai_module.get_ai_backend_name(),
    }


class AIRecommendRequest(BaseModel):
    analysis: Optional[dict] = None


@router.post("/api/ai/recommend")
async def api_ai_recommend(
    request: Request,
    user: Optional[dict] = Depends(get_current_user_optional),
    file: Optional[UploadFile] = File(None),
    body: Optional[AIRecommendRequest] = None,
):
    """Рекомендация пресета по анализу трека (file или body.analysis)."""
    try:
        require_feature_ai()
        tier = get_tier_for_ai(user, request)
        ident = get_ai_identifier(request, user)
        limit_info = ai_module.check_ai_rate_limit(ident, tier)
        if not limit_info["ok"]:
            raise HTTPException(
                429,
                f"Лимит AI-запросов исчерпан: {limit_info['limit']}/день. Сброс: {limit_info['reset_at']}.",
            )

        analysis = None
        if body and body.analysis:
            analysis = body.analysis
        elif file and file.filename and allowed_file(file.filename):
            analysis = await _analyze_file_for_ai(file)

        if not analysis:
            raise HTTPException(400, "Передайте file (аудиофайл) или body.analysis.")

        result = ai_module.recommend_preset(analysis)
        ai_module.record_ai_usage(ident)
        _log_ai(user, tier, "recommend")
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("api_ai_recommend: unhandled error")
        raise HTTPException(500, detail=f"Ошибка сервера при рекомендации пресета: {e!s}") from e


class AIReportRequest(BaseModel):
    analysis: Optional[dict] = None


@router.post("/api/ai/report")
async def api_ai_report(
    request: Request,
    user: Optional[dict] = Depends(get_current_user_optional),
    file: Optional[UploadFile] = File(None),
    body: Optional[AIReportRequest] = None,
):
    """Текстовый отчёт по анализу трека + рекомендации."""
    try:
        require_feature_ai()
        tier = get_tier_for_ai(user, request)
        ident = get_ai_identifier(request, user)
        limit_info = ai_module.check_ai_rate_limit(ident, tier)
        if not limit_info["ok"]:
            raise HTTPException(
                429,
                f"Лимит AI-запросов исчерпан: {limit_info['limit']}/день. Сброс: {limit_info['reset_at']}.",
            )

        analysis = None
        if body and body.analysis:
            analysis = body.analysis
        elif file and file.filename and allowed_file(file.filename):
            analysis = await _analyze_file_for_ai(file, extended=False)

        if not analysis:
            raise HTTPException(400, "Передайте file (аудиофайл) или body.analysis.")

        result = ai_module.report_with_recommendations(analysis)
        ai_module.record_ai_usage(ident)
        _log_ai(user, tier, "report")
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("api_ai_report: unhandled error")
        raise HTTPException(500, detail=f"Ошибка сервера при формировании AI-отчёта: {e!s}") from e


class AINlConfigRequest(BaseModel):
    text: str
    current_config: Optional[dict] = None


@router.post("/api/ai/nl-config")
async def api_ai_nl_config(
    request: Request,
    body: AINlConfigRequest,
    user: Optional[dict] = Depends(get_current_user_optional),
):
    """Естественный язык → параметры цепочки мастеринга и target_lufs."""
    try:
        require_feature_ai()
        tier = get_tier_for_ai(user, request)
        ident = get_ai_identifier(request, user)
        limit_info = ai_module.check_ai_rate_limit(ident, tier)
        if not limit_info["ok"]:
            raise HTTPException(
                429,
                f"Лимит AI-запросов исчерпан: {limit_info['limit']}/день. Сброс: {limit_info['reset_at']}.",
            )
        result = ai_module.nl_to_config(body.text or "", body.current_config)
        if result.get("error"):
            raise HTTPException(400, result["error"])
        ai_module.record_ai_usage(ident)
        _log_ai(user, tier, "nl_config")
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("api_ai_nl_config: unhandled error")
        raise HTTPException(500, detail=f"Ошибка сервера при разборе настроек: {e!s}") from e


class AIChatRequest(BaseModel):
    messages: List[dict]
    context: Optional[dict] = None


@router.post("/api/ai/chat")
async def api_ai_chat(
    request: Request,
    body: AIChatRequest,
    user: Optional[dict] = Depends(get_current_user_optional),
):
    """Чат-помощник по мастерингу."""
    try:
        require_feature_ai()
        tier = get_tier_for_ai(user, request)
        ident = get_ai_identifier(request, user)
        limit_info = ai_module.check_ai_rate_limit(ident, tier)
        if not limit_info["ok"]:
            raise HTTPException(
                429,
                f"Лимит AI-запросов исчерпан: {limit_info['limit']}/день. Сброс: {limit_info['reset_at']}.",
            )
        reply = ai_module.chat_assistant(body.messages or [], body.context)
        ai_module.record_ai_usage(ident)
        _log_ai(user, tier, "chat")
        return {"reply": reply}
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("api_ai_chat: unhandled error")
        raise HTTPException(500, detail=f"Ошибка сервера в AI-чате: {e!s}") from e


# ─── Helpers ──────────────────────────────────────────────────────────────────

async def _analyze_file_for_ai(file: UploadFile, extended: bool = True) -> dict:
    """Читает аудиофайл и возвращает словарь анализа для AI-эндпоинтов."""
    fname = file.filename or ""
    if fname.lower().endswith(".mp3") and not shutil.which("ffmpeg"):
        raise HTTPException(400, "Чтение MP3 требует ffmpeg.")
    data = await file.read()
    max_mb = settings_store.get_setting_int("max_upload_mb", 100)
    if len(data) > max_mb * 1024 * 1024:
        raise HTTPException(400, f"Файл больше {max_mb} МБ.")
    if not check_audio_magic_bytes(data, fname):
        raise HTTPException(400, "Содержимое файла не соответствует формату. Ожидается WAV, MP3 или FLAC.")
    try:
        audio, sr = load_audio_from_bytes(data, fname)
    except Exception as e:
        raise HTTPException(400, f"Не удалось прочитать аудио: {e}")
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
    result: dict = {
        "lufs": round(lufs, 2) if not np.isnan(lufs) else None,
        "peak_dbfs": round(peak_dbfs, 2),
        "duration_sec": round(duration_sec, 3),
        "sample_rate": sr,
        "channels": channels,
        "stereo_correlation": round(correlation, 4) if correlation is not None else None,
    }
    if extended and audio.size >= 4096:
        try:
            result["spectrum_bars"] = compute_spectrum_bars(audio, sr)
        except Exception:  # noqa: BLE001
            pass
    return result


def _log_ai(user: Optional[dict], tier: str, action: str) -> None:
    try:
        uid = user.get("sub") if user else None
        uid_int = int(uid) if uid is not None else None
        log_ai_usage(action, uid_int, tier)
    except Exception:  # noqa: BLE001
        pass
