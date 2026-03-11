# @file routers/misc.py
# @description Вспомогательные эндпоинты: новости, debug-mode, лимиты, пресеты, стили, прогресс, measure.
# @created 2026-03-01

import json
import logging
import shutil
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

import numpy as np
from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile
from fastapi.responses import Response

from ..config import settings
from ..database import (
    DB_AVAILABLE,
    get_db,
    get_news_posts,
    get_user_tokens_balance,
    count_mastering_jobs_today,
)
from ..deps import check_rate_limit, get_current_user_optional
from ..helpers import allowed_file, check_audio_magic_bytes, json_safe_float
from ..pipeline import PRESET_LUFS, STYLE_CONFIGS, load_audio_from_bytes, measure_lufs
from .. import settings_store

router = APIRouter()

# Корень проекта: routers/misc.py → routers → app → backend → project root
_PROGRESS_PATH = Path(__file__).resolve().parent.parent.parent.parent / "PROGRESS.md"


@router.get("/api/news")
def api_news_public(limit: int = 5):
    """Последние опубликованные новости для лендинга (без авторизации)."""
    if not DB_AVAILABLE:
        return {"posts": []}
    from ..database import SessionLocal as _SL
    if _SL is None:
        return {"posts": []}
    db = _SL()
    try:
        posts = get_news_posts(db, published_only=True, limit=max(1, min(limit, 20)))
        return {
            "posts": [
                {
                    "id": p.id,
                    "title": p.title,
                    "body": p.body,
                    "published_at": p.published_at,
                }
                for p in posts
            ]
        }
    finally:
        db.close()


@router.get("/api/debug-mode")
def api_debug_mode():
    """Режим отладки: при MAGIC_MASTER_DEBUG=1 возвращает {"debug": true}."""
    return {"debug": getattr(settings, "debug_mode", False)}


@router.get("/api/limits")
async def api_limits(
    request: Request,
    user: Optional[dict] = Depends(get_current_user_optional),
    db=Depends(get_db),
):
    """Текущий тариф и лимиты мастеринга. Free: 1/неделю; Pro/Studio: токены + дневной лимит."""
    if getattr(settings, "debug_mode", False):
        return {
            "tier": "pro",
            "used": 0,
            "limit": -1,
            "remaining": 999,
            "tokens_balance": 999,
            "daily_used": 0,
            "daily_limit": 30,
            "reset_at": None,
            "debug": True,
            "priority_queue": True,
        }
    if user:
        tier = (user.get("tier") or "pro").lower()
        uid = user.get("sub")
        try:
            uid = int(uid) if uid is not None else None
        except (TypeError, ValueError):
            uid = None
        tokens = 0
        daily_used = 0
        daily_limit = 10 if tier == "pro" else 30 if tier == "studio" else 0
        if db and uid and tier in ("pro", "studio"):
            tokens = get_user_tokens_balance(db, uid)
            daily_used = count_mastering_jobs_today(db, uid)
        remaining_today = max(0, daily_limit - daily_used) if daily_limit else 0
        remaining = min(tokens, remaining_today) if tier in ("pro", "studio") else tokens
        return {
            "tier": tier,
            "used": daily_used,
            "limit": daily_limit,
            "remaining": remaining,
            "tokens_balance": tokens,
            "daily_used": daily_used,
            "daily_limit": daily_limit,
            "reset_at": None,
            "email": user.get("email"),
            "priority_queue": tier in ("pro", "studio"),
        }
    from ..helpers import get_client_ip
    ip = get_client_ip(request)
    info = check_rate_limit(ip)
    return {
        "tier": "free",
        "used": info["used"],
        "limit": info["limit"],
        "remaining": info["remaining"],
        "reset_at": info["reset_at"],
        "priority_queue": False,
    }


@router.get("/api/progress", response_class=Response)
def api_progress():
    """Содержимое PROGRESS.md (статус плана разработки)."""
    if not _PROGRESS_PATH.is_file():
        return Response(
            content="# Статус плана\n\nФайл PROGRESS.md отсутствует на сервере.\n",
            media_type="text/markdown; charset=utf-8",
        )
    return Response(
        content=_PROGRESS_PATH.read_text(encoding="utf-8"),
        media_type="text/markdown; charset=utf-8",
    )


@router.get("/api/presets")
def get_presets():
    """Список пресетов целевой громкости (LUFS) для API.

    Возвращает словарь имя_пресета → целевой LUFS (например spotify: -14, club: -9, broadcast: -24).
    Используется при вызове POST /api/master с параметром preset. Фронт основного приложения
    использует /api/presets/community и /api/auth/presets; этот эндпоинт — для интеграций и скриптов.
    """
    return {"presets": PRESET_LUFS}


def _load_community_presets() -> list:
    """Загружает пресеты сообщества: app/presets_community.json + опционально extra (файл или каталог)."""
    base = Path(__file__).resolve().parent.parent
    main_path = base / "presets_community.json"
    out: list = []
    seen_ids: set = set()

    def _append_valid(items: list) -> None:
        nonlocal out, seen_ids
        if not isinstance(items, list):
            return
        for item in items:
            if not isinstance(item, dict):
                continue
            pid = item.get("id")
            if not pid or pid in seen_ids:
                continue
            if "name" in item and "target_lufs" in item:
                seen_ids.add(pid)
                out.append(item)

    if main_path.is_file():
        try:
            data = json.loads(main_path.read_text(encoding="utf-8"))
            _append_valid(data if isinstance(data, list) else [])
        except (json.JSONDecodeError, OSError):
            pass

    extra = getattr(settings, "community_presets_extra", None) or ""
    if extra:
        extra_path = Path(extra)
        if not extra_path.is_absolute():
            extra_path = base / extra
        if extra_path.is_file() and extra_path.suffix.lower() == ".json":
            try:
                data = json.loads(extra_path.read_text(encoding="utf-8"))
                _append_valid(data if isinstance(data, list) else [])
            except (json.JSONDecodeError, OSError):
                pass
        elif extra_path.is_dir():
            for p in sorted(extra_path.glob("*.json")):
                try:
                    data = json.loads(p.read_text(encoding="utf-8"))
                    _append_valid(data if isinstance(data, list) else [])
                except (json.JSONDecodeError, OSError):
                    pass

    return out


@router.get("/api/presets/community")
def get_presets_community():
    """Пресеты сообщества (P64). Поддержка расширений: MAGIC_MASTER_COMMUNITY_PRESETS_EXTRA."""
    return {"presets": _load_community_presets()}


@router.get("/api/extensions")
def get_extensions_status():
    """Статус расширений (для API и мониторинга).

    Возвращает community_presets_extra_configured и community_presets_extra_loaded.
    Фронт не вызывает этот эндпоинт; предназначен для /docs и внешних интеграций.
    """
    extra = getattr(settings, "community_presets_extra", None) or ""
    base = Path(__file__).resolve().parent.parent
    extra_path = Path(extra) if extra else None
    if extra_path and not extra_path.is_absolute():
        extra_path = base / extra
    extra_loaded = bool(
        extra_path
        and extra_path.exists()
        and (extra_path.is_file() and extra_path.suffix.lower() == ".json" or extra_path.is_dir())
    )
    return {
        "community_presets_extra_configured": bool(extra),
        "community_presets_extra_loaded": extra_loaded,
    }


@router.get("/api/styles")
def get_styles():
    """Список жанровых стилей и целевой LUFS по каждому (для API).

    Возвращает словарь имя_стиля → { lufs }. Полная цепочка модулей по стилю — через
    GET /api/v2/chain/default?style=... . Фронт использует chain/default; этот эндпоинт
    удобен для скриптов и отображения списка стилей в документации.
    """
    return {"styles": {k: {"lufs": v["lufs"]} for k, v in STYLE_CONFIGS.items()}}


@router.post("/api/measure")
async def api_measure(
    file: UploadFile = File(...),
    user: Optional[dict] = Depends(get_current_user_optional),
):
    """Загрузить файл и вернуть текущую громкость в LUFS. Форматы: WAV, MP3, FLAC. Лимит по тарифу и формату (WAV до 800 МБ, MP3 до 300 МБ)."""
    try:
        if not allowed_file(file.filename or ""):
            raise HTTPException(400, "Формат не поддерживается. Разрешены: WAV, MP3, FLAC.")
        fname = file.filename or ""
        if fname.lower().endswith(".mp3") and not shutil.which("ffmpeg"):
            raise HTTPException(
                400,
                "Чтение MP3 требует ffmpeg, который не найден на сервере. "
                "Установите: sudo apt-get install -y ffmpeg",
            )
        data = await file.read()
        tier = (user.get("tier") or "free").lower() if user else "free"
        max_mb = settings_store.get_max_upload_mb(fname, tier)
        if len(data) > max_mb * 1024 * 1024:
            raise HTTPException(400, f"Файл больше {max_mb} МБ.")
        if not check_audio_magic_bytes(data, fname):
            raise HTTPException(400, "Содержимое файла не соответствует формату. Ожидается WAV, MP3 или FLAC.")
        try:
            audio, sr = load_audio_from_bytes(data, fname or "wav")
        except Exception as e:
            raise HTTPException(400, f"Не удалось прочитать аудио: {e}") from e
        lufs = measure_lufs(audio, sr)
        peak = float(np.max(np.abs(audio))) if audio.size > 0 else 0.0
        peak_dbfs = float(20 * np.log10(max(peak, 1e-12)))
        duration = float(len(audio) / sr)
        channels = 1 if audio.ndim == 1 else int(audio.shape[1])
        return {
            "lufs": json_safe_float(lufs),
            "sample_rate": sr,
            "peak_dbfs": round(peak_dbfs, 2),
            "duration": round(duration, 3),
            "channels": channels,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("api_measure: unhandled error")
        raise HTTPException(500, detail=f"Ошибка сервера при измерении громкости: {e!s}") from e
