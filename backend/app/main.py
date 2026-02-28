# @file main.py
# @description FastAPI-приложение: загрузка, мастеринг, экспорт, замер LUFS
# @dependencies app.pipeline, app.chain, app.config
# @created 2026-02-26

import asyncio
import datetime
import json
import os
import shutil
import time
import uuid

import numpy as np
from pathlib import Path
from typing import Any, List, Optional, Tuple

from fastapi import BackgroundTasks, Depends, FastAPI, File, Form, HTTPException, Header, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, field_validator

from .auth import (
    AUTH_AVAILABLE,
    create_access_token,
    decode_token,
    extract_bearer_token,
    get_password_hash,
    verify_password,
)
from .chain import MasteringChain
from .config import settings
from .database import (
    DB_AVAILABLE,
    create_mastering_record,
    create_saved_preset,
    create_tables,
    create_user,
    delete_mastering_record,
    delete_saved_preset,
    get_db,
    get_saved_preset_by_id,
    get_user_by_email,
    get_user_history,
    get_user_presets,
    get_user_stats,
)
from .version import __version__
from .pipeline import (
    PRESET_LUFS,
    STYLE_CONFIGS,
    apply_deesser,
    apply_dynamic_eq,
    apply_parallel_compression,
    apply_reference_match,
    apply_spectral_denoise,
    apply_transient_designer,
    compute_lufs_timeline,
    compute_spectrum_bars,
    compute_spectral_envelope,
    compute_vectorscope_points,
    export_audio,
    load_audio_from_bytes,
    measure_lufs,
    measure_stereo_correlation,
    run_mastering_pipeline,
)

# Фоновые задачи мастеринга: job_id -> { status, progress, message, created_at, done_at?, ... }
_jobs: dict[str, dict] = {}

# P14: приоритетная очередь — семафоры слотов (Pro: 2 одновременных, Free: 1)
_sem_priority = asyncio.Semaphore(2)
_sem_normal = asyncio.Semaphore(1)


def _is_priority_user(user: Optional[dict]) -> bool:
    """Pro/Studio или режим отладки — доступ к приоритетным слотам."""
    if getattr(settings, "debug_mode", False):
        return True
    if not user:
        return False
    return (user.get("tier") or "").lower() in ("pro", "studio")


# ─── Rate limiting (Free tier: 3 мастеринга / день по IP) ─────────────────────
_FREE_DAILY_LIMIT = 3
_rate_limits: dict[str, dict] = {}  # ip -> {"count": int, "day": str (ISO)}


def _get_client_ip(request: Request) -> str:
    """Извлекает IP клиента (учитывает Nginx X-Forwarded-For)."""
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return (request.client.host if request.client else "unknown")


def _check_rate_limit(ip: str) -> dict:
    """Возвращает {"ok": bool, "used": int, "limit": int, "remaining": int, "reset_at": str}."""
    today = datetime.date.today().isoformat()
    entry = _rate_limits.get(ip)
    used = entry["count"] if (entry and entry.get("day") == today) else 0
    remaining = max(0, _FREE_DAILY_LIMIT - used)
    tomorrow = (datetime.date.today() + datetime.timedelta(days=1)).isoformat()
    return {
        "ok": used < _FREE_DAILY_LIMIT,
        "used": used,
        "limit": _FREE_DAILY_LIMIT,
        "remaining": remaining,
        "reset_at": tomorrow,
    }


def _record_usage(ip: str) -> None:
    """Увеличивает счётчик использований для IP за сегодня."""
    _record_usage_n(ip, 1)


def _record_usage_n(ip: str, n: int) -> None:
    """Увеличивает счётчик использований для IP на n."""
    today = datetime.date.today().isoformat()
    entry = _rate_limits.get(ip)
    if not entry or entry.get("day") != today:
        _rate_limits[ip] = {"count": n, "day": today}
    else:
        _rate_limits[ip]["count"] = entry["count"] + n

app = FastAPI(
    title="Magic Master — автоматический мастеринг",
    description="Загрузите трек → нажмите Magic Master → скачайте результат с целевой громкостью LUFS.",
    version=__version__,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs(settings.temp_dir, exist_ok=True)

# Создаём таблицы SQLite при запуске (idempotent)
create_tables()


# ─── Pydantic-схемы для auth ──────────────────────────────────────────────────
class RegisterRequest(BaseModel):
    email: str
    password: str

    @field_validator("email")
    @classmethod
    def email_lower(cls, v: str) -> str:
        v = v.strip().lower()
        if "@" not in v or len(v) < 5:
            raise ValueError("Некорректный email")
        return v

    @field_validator("password")
    @classmethod
    def password_min_len(cls, v: str) -> str:
        if len(v) < 6:
            raise ValueError("Пароль минимум 6 символов")
        return v


class LoginRequest(BaseModel):
    email: str
    password: str


# ─── Auth helper ─────────────────────────────────────────────────────────────
def _get_current_user_optional(authorization: Optional[str] = Header(None)) -> Optional[dict]:
    """Dependency: декодирует Bearer токен, возвращает payload или None."""
    token = extract_bearer_token(authorization)
    if not token:
        return None
    return decode_token(token)


def _allowed_file(filename: str) -> bool:
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    return ext in settings.allowed_extensions


def _prune_jobs() -> None:
    """Удаляет старые завершённые задачи и ограничивает число записей в _jobs."""
    now = time.time()
    ttl = settings.jobs_done_ttl_seconds
    max_entries = settings.jobs_max_entries
    to_remove = []
    for jid, job in _jobs.items():
        if job.get("status") in ("done", "error") and job.get("done_at"):
            if now - job["done_at"] > ttl:
                to_remove.append(jid)
    for jid in to_remove:
        del _jobs[jid]
    if len(_jobs) <= max_entries:
        return
    by_created = sorted(_jobs.items(), key=lambda x: x[1].get("created_at", 0))
    for jid, _ in by_created[: len(_jobs) - max_entries]:
        if jid in _jobs:
            del _jobs[jid]


@app.get("/api")
def api_root():
    return {"service": "Magic Master API", "docs": "/docs", "version": __version__}


@app.get("/api/version")
def api_version():
    """Версия приложения для интерфейса и мониторинга."""
    return {"version": __version__}


@app.get("/api/health")
def api_health():
    """Проверка живости сервиса для деплоя и мониторинга."""
    return {"status": "ok"}


@app.get("/api/debug-mode")
def api_debug_mode():
    """
    Режим отладки: при MAGIC_MASTER_DEBUG=1 возвращает {"debug": true}.
    Фронт использует для разблокировки всех функций без авторизации.
    """
    return {"debug": getattr(settings, "debug_mode", False)}


@app.get("/api/limits")
async def api_limits(
    request: Request,
    user: Optional[dict] = Depends(_get_current_user_optional),
):
    """
    Текущий тариф и лимиты мастеринга.
    Залогиненные пользователи: tier=pro/studio, безлимитные мастеринги.
    Режим отладки (MAGIC_MASTER_DEBUG=1): без входа — tier=pro, без лимитов.
    Гости (Free): 3 мастеринга в день по IP.
    """
    if user:
        tier = (user.get("tier") or "pro").lower()
        return {
            "tier": tier,
            "used": 0,
            "limit": -1,
            "remaining": -1,
            "reset_at": None,
            "email": user.get("email"),
            "priority_queue": tier in ("pro", "studio"),
        }
    if getattr(settings, "debug_mode", False):
        return {
            "tier": "pro",
            "used": 0,
            "limit": -1,
            "remaining": -1,
            "reset_at": None,
            "debug": True,
            "priority_queue": True,
        }
    ip = _get_client_ip(request)
    info = _check_rate_limit(ip)
    return {
        "tier": "free",
        "used": info["used"],
        "limit": info["limit"],
        "remaining": info["remaining"],
        "reset_at": info["reset_at"],
        "priority_queue": False,
    }


# ─── Auth endpoints ───────────────────────────────────────────────────────────
def _require_auth_available() -> None:
    """Вызвать HTTPException 503 если auth/DB пакеты не установлены."""
    if not DB_AVAILABLE or not AUTH_AVAILABLE:
        raise HTTPException(
            503,
            "Авторизация недоступна: установите пакеты sqlalchemy, passlib, python-jose "
            "(подключите интернет и запустите: pip install sqlalchemy passlib python-jose[cryptography])"
        )


@app.post("/api/auth/register")
def api_auth_register(body: RegisterRequest, db=Depends(get_db)):
    """Зарегистрировать нового пользователя. Возвращает JWT токен."""
    _require_auth_available()
    if get_user_by_email(db, body.email):
        raise HTTPException(400, "Пользователь с таким email уже существует")
    hashed = get_password_hash(body.password)
    user = create_user(db, body.email, hashed, tier="pro")
    token = create_access_token(user.id, user.email, user.tier)
    return {
        "access_token": token,
        "token_type": "bearer",
        "email": user.email,
        "tier": user.tier,
    }


@app.post("/api/auth/login")
def api_auth_login(body: LoginRequest, db=Depends(get_db)):
    """Войти. Возвращает JWT токен."""
    _require_auth_available()
    user = get_user_by_email(db, body.email)
    if not user or not verify_password(body.password, user.hashed_password):
        raise HTTPException(401, "Неверный email или пароль")
    import time as _time
    user.last_login_at = _time.time()
    db.commit()
    token = create_access_token(user.id, user.email, user.tier)
    return {
        "access_token": token,
        "token_type": "bearer",
        "email": user.email,
        "tier": user.tier,
    }


@app.get("/api/auth/me")
def api_auth_me(user: Optional[dict] = Depends(_get_current_user_optional)):
    """Информация о текущем пользователе по JWT токену."""
    if not user:
        raise HTTPException(401, "Не авторизован. Передайте заголовок Authorization: Bearer <token>")
    return {
        "email": user.get("email"),
        "tier": user.get("tier", "free"),
        "user_id": user.get("sub"),
    }


@app.post("/api/auth/logout")
def api_auth_logout():
    """Logout — клиент просто удаляет токен из localStorage. Endpoint для единообразия."""
    return {"message": "Токен удалён на стороне клиента"}


class ChangePasswordRequest(BaseModel):
    old_password: str
    new_password: str

    @field_validator("new_password")
    @classmethod
    def new_password_min(cls, v: str) -> str:
        if len(v) < 6:
            raise ValueError("Новый пароль минимум 6 символов")
        return v


@app.post("/api/auth/change-password")
def api_auth_change_password(
    body: ChangePasswordRequest,
    db=Depends(get_db),
    user: Optional[dict] = Depends(_get_current_user_optional),
):
    """Сменить пароль залогиненного пользователя."""
    _require_auth_available()
    if not user:
        raise HTTPException(401, "Требуется авторизация")
    db_user = get_user_by_email(db, user["email"])
    if not db_user or not verify_password(body.old_password, db_user.hashed_password):
        raise HTTPException(400, "Неверный текущий пароль")
    db_user.hashed_password = get_password_hash(body.new_password)
    db.commit()
    return {"message": "Пароль успешно изменён"}


# ─── History endpoints ────────────────────────────────────────────────────────
class RecordRequest(BaseModel):
    filename: str = ""
    style: str = "standard"
    out_format: str = "wav"
    before_lufs: Optional[float] = None
    after_lufs: Optional[float] = None
    target_lufs: Optional[float] = None
    duration_sec: Optional[float] = None


@app.post("/api/auth/record")
def api_auth_record(
    body: RecordRequest,
    db=Depends(get_db),
    user: Optional[dict] = Depends(_get_current_user_optional),
):
    """Сохранить запись о завершённом мастеринге в историю пользователя."""
    _require_auth_available()
    if not user:
        raise HTTPException(401, "Требуется авторизация")
    user_id = int(user["sub"])
    rec = create_mastering_record(
        db,
        user_id=user_id,
        filename=body.filename,
        style=body.style,
        out_format=body.out_format,
        before_lufs=body.before_lufs,
        after_lufs=body.after_lufs,
        target_lufs=body.target_lufs,
        duration_sec=body.duration_sec,
    )
    if rec is None:
        raise HTTPException(503, "База данных недоступна")
    return {"id": rec.id, "created_at": rec.created_at}


@app.get("/api/auth/history")
def api_auth_history(
    db=Depends(get_db),
    user: Optional[dict] = Depends(_get_current_user_optional),
):
    """История мастерингов + статистика залогиненного пользователя."""
    _require_auth_available()
    if not user:
        raise HTTPException(401, "Требуется авторизация")
    user_id = int(user["sub"])
    records = get_user_history(db, user_id, limit=30)
    stats = get_user_stats(db, user_id)
    return {
        "stats": stats,
        "records": [
            {
                "id": r.id,
                "filename": r.filename,
                "style": r.style,
                "out_format": r.out_format,
                "before_lufs": r.before_lufs,
                "after_lufs": r.after_lufs,
                "target_lufs": r.target_lufs,
                "duration_sec": r.duration_sec,
                "created_at": r.created_at,
            }
            for r in records
        ],
    }


@app.delete("/api/auth/history/{record_id}")
def api_auth_history_delete(
    record_id: int,
    db=Depends(get_db),
    user: Optional[dict] = Depends(_get_current_user_optional),
):
    """Удалить запись из истории."""
    _require_auth_available()
    if not user:
        raise HTTPException(401, "Требуется авторизация")
    user_id = int(user["sub"])
    if not delete_mastering_record(db, record_id, user_id):
        raise HTTPException(404, "Запись не найдена")
    return {"deleted": record_id}


# --- P10: сохранённые пресеты цепочки ---

class SavedPresetCreate(BaseModel):
    name: str
    config: dict
    style: str = "standard"
    target_lufs: float = -14.0


@app.get("/api/auth/presets")
def api_auth_presets_list(
    db=Depends(get_db),
    user: Optional[dict] = Depends(_get_current_user_optional),
):
    """Список сохранённых пресетов цепочки для залогиненного пользователя."""
    _require_auth_available()
    if not user:
        raise HTTPException(401, "Требуется авторизация")
    user_id = int(user["sub"])
    presets = get_user_presets(db, user_id, limit=50)
    return {
        "presets": [
            {
                "id": p.id,
                "name": p.name,
                "config": json.loads(p.config) if p.config else {},
                "style": p.style,
                "target_lufs": p.target_lufs,
                "created_at": p.created_at,
            }
            for p in presets
        ],
    }


@app.post("/api/auth/presets")
def api_auth_presets_create(
    body: SavedPresetCreate,
    db=Depends(get_db),
    user: Optional[dict] = Depends(_get_current_user_optional),
):
    """Сохранить текущую цепочку как пресет (имя + config + style + target_lufs)."""
    _require_auth_available()
    if not user:
        raise HTTPException(401, "Требуется авторизация")
    user_id = int(user["sub"])
    config_str = json.dumps(body.config) if isinstance(body.config, dict) else "{}"
    preset = create_saved_preset(
        db, user_id, body.name, config_str,
        style=body.style,
        target_lufs=body.target_lufs,
    )
    return {"id": preset.id, "name": preset.name, "created_at": preset.created_at}


@app.get("/api/auth/presets/{preset_id}")
def api_auth_presets_get(
    preset_id: int,
    db=Depends(get_db),
    user: Optional[dict] = Depends(_get_current_user_optional),
):
    """Получить один пресет по id (для загрузки в цепочку)."""
    _require_auth_available()
    if not user:
        raise HTTPException(401, "Требуется авторизация")
    user_id = int(user["sub"])
    preset = get_saved_preset_by_id(db, preset_id, user_id)
    if not preset:
        raise HTTPException(404, "Пресет не найден")
    return {
        "id": preset.id,
        "name": preset.name,
        "config": json.loads(preset.config) if preset.config else {},
        "style": preset.style,
        "target_lufs": preset.target_lufs,
        "created_at": preset.created_at,
    }


@app.delete("/api/auth/presets/{preset_id}")
def api_auth_presets_delete(
    preset_id: int,
    db=Depends(get_db),
    user: Optional[dict] = Depends(_get_current_user_optional),
):
    """Удалить сохранённый пресет."""
    _require_auth_available()
    if not user:
        raise HTTPException(401, "Требуется авторизация")
    user_id = int(user["sub"])
    if not delete_saved_preset(db, preset_id, user_id):
        raise HTTPException(404, "Пресет не найден")
    return {"deleted": preset_id}


# Корень проекта: backend/app/main.py -> app -> backend -> корень
_PROGRESS_PATH = Path(__file__).resolve().parent.parent.parent / "PROGRESS.md"


@app.get("/api/progress", response_class=Response)
def api_progress():
    """Вернуть содержимое PROGRESS.md (статус выполнения плана). Если файла нет — возвращаем краткое сообщение."""
    if not _PROGRESS_PATH.is_file():
        return Response(
            content="# Статус плана\n\nФайл PROGRESS.md на сервере отсутствует. Добавьте его в корень проекта при деплое (см. pack_for_deploy.sh).\n",
            media_type="text/markdown; charset=utf-8",
        )
    return Response(
        content=_PROGRESS_PATH.read_text(encoding="utf-8"),
        media_type="text/markdown; charset=utf-8",
    )


@app.get("/api/presets")
def get_presets():
    """Список пресетов целевой громкости (LUFS)."""
    return {"presets": PRESET_LUFS}


@app.post("/api/measure")
async def api_measure(file: UploadFile = File(...)):
    """
    Загрузить файл и вернуть текущую громкость в LUFS.
    Форматы: WAV, MP3, FLAC.
    """
    if not _allowed_file(file.filename or ""):
        raise HTTPException(400, "Формат не поддерживается. Разрешены: WAV, MP3, FLAC.")
    # MP3 требует ffmpeg — проверяем заранее
    fname = file.filename or ""
    if fname.lower().endswith(".mp3") and not shutil.which("ffmpeg"):
        raise HTTPException(
            400,
            "Чтение MP3 требует ffmpeg, который не найден на сервере. "
            "Установите: sudo apt-get install -y ffmpeg",
        )
    data = await file.read()
    if len(data) > settings.max_upload_mb * 1024 * 1024:
        raise HTTPException(400, f"Файл больше {settings.max_upload_mb} МБ.")
    try:
        audio, sr = load_audio_from_bytes(data, file.filename or "wav")
    except Exception as e:
        raise HTTPException(400, f"Не удалось прочитать аудио: {e}")
    lufs = measure_lufs(audio, sr)
    peak = float(np.max(np.abs(audio))) if audio.size > 0 else 0.0
    peak_dbfs = float(20 * np.log10(max(peak, 1e-12)))
    duration = float(len(audio) / sr)
    channels = 1 if audio.ndim == 1 else int(audio.shape[1])
    return {
        "lufs": lufs,
        "sample_rate": sr,
        "peak_dbfs": round(peak_dbfs, 2),
        "duration": round(duration, 3),
        "channels": channels,
    }


@app.get("/api/styles")
def get_styles():
    """Список жанровых пресетов с параметрами."""
    return {"styles": {k: {"lufs": v["lufs"]} for k, v in STYLE_CONFIGS.items()}}


def _run_mastering_job(
    job_id: str,
    data: bytes,
    filename: str,
    target_lufs: float,
    out_format: str,
    style: str = "standard",
):
    """Синхронный мастеринг в потоке; обновляет _jobs[job_id] (progress, result или error)."""
    job = _jobs[job_id]
    try:
        job["progress"] = 2
        job["message"] = "Загрузка аудио…"
        audio, sr = load_audio_from_bytes(data, filename or "wav")

        job["progress"] = 4
        job["message"] = "Измерение исходного уровня…"
        job["before_lufs"] = measure_lufs(audio, sr)

        def on_progress(pct: int, msg: str):
            job["progress"] = pct
            job["message"] = msg

        job["progress"] = 5
        job["message"] = "Мастеринг…"
        mastered = run_mastering_pipeline(
            audio, sr, target_lufs=target_lufs, style=style, progress_callback=on_progress
        )
        job["after_lufs"] = measure_lufs(mastered, sr)
        job["progress"] = 98
        job["message"] = "Экспорт файла…"
        channels = 1 if mastered.ndim == 1 else mastered.shape[1]
        out_bytes = export_audio(mastered, sr, channels, out_format.lower())
        out_name = (filename or "master").rsplit(".", 1)[0] + f"_mastered.{out_format}"
        job["status"] = "done"
        job["progress"] = 100
        job["message"] = "Готово"
        job["result_bytes"] = out_bytes
        job["filename"] = out_name
        job["done_at"] = time.time()
    except Exception as e:
        job["status"] = "error"
        job["progress"] = 0
        job["message"] = ""
        job["error"] = str(e)
        job["done_at"] = time.time()


def _run_mastering_job_v2(
    job_id: str,
    data: bytes,
    filename: str,
    target_lufs: float,
    out_format: str,
    style: str = "standard",
    chain_config: Optional[dict[str, Any]] = None,
    dither_type: Optional[str] = None,
    auto_blank_sec: Optional[float] = None,
    pro_params: Optional[dict] = None,
):
    """Мастеринг через MasteringChain (v2) + PRO-модули. Если chain_config is None — используется default_chain."""
    job = _jobs[job_id]
    pro_params = pro_params or {}
    try:
        job["progress"] = 2
        job["message"] = "Загрузка аудио…"
        audio, sr = load_audio_from_bytes(data, filename or "wav")

        job["progress"] = 4
        job["message"] = "Измерение исходного уровня…"
        job["before_lufs"] = measure_lufs(audio, sr)

        # PRO: Spectral Denoiser (до основного мастеринга)
        if pro_params.get("denoise_strength", 0) > 0:
            job["message"] = "Spectral Denoiser…"
            audio = apply_spectral_denoise(audio, sr, strength=pro_params["denoise_strength"])

        # PRO: De-esser (до основного мастеринга)
        if pro_params.get("deesser_enabled"):
            job["message"] = "De-esser…"
            audio = apply_deesser(audio, sr, threshold_db=pro_params.get("deesser_threshold", -10.0))

        def on_progress(pct: int, msg: str):
            job["progress"] = pct
            job["message"] = msg

        job["progress"] = 5
        job["message"] = "Мастеринг (v2)…"
        if chain_config:
            chain = MasteringChain.from_config(chain_config)
        else:
            chain = MasteringChain.default_chain(target_lufs=target_lufs, style=style)
        mastered = chain.process(
            audio,
            sr,
            target_lufs=target_lufs,
            style=style,
            progress_callback=on_progress,
        )

        # PRO: Transient Designer (после мастеринга)
        if pro_params.get("transient_attack") is not None:
            job["message"] = "Transient Designer…"
            mastered = apply_transient_designer(
                mastered, sr,
                attack_gain=pro_params.get("transient_attack", 1.0),
                sustain_gain=pro_params.get("transient_sustain", 1.0),
            )

        # PRO: Parallel Compression (после мастеринга)
        if pro_params.get("parallel_mix", 0) > 0:
            job["message"] = "Parallel Compression…"
            mastered = apply_parallel_compression(mastered, sr, mix=pro_params["parallel_mix"])

        # PRO: Dynamic EQ (после мастеринга)
        if pro_params.get("dynamic_eq_enabled"):
            job["message"] = "Dynamic EQ…"
            mastered = apply_dynamic_eq(mastered, sr)

        job["after_lufs"] = measure_lufs(mastered, sr)
        job["progress"] = 98
        job["message"] = "Экспорт файла…"
        channels = 1 if mastered.ndim == 1 else mastered.shape[1]
        dt = dither_type or (chain_config or {}).get("dither_type") or "tpdf"
        ab = auto_blank_sec if auto_blank_sec is not None else float((chain_config or {}).get("auto_blank_sec", 0) or 0)
        if dt not in ("tpdf", "ns_e", "ns_itu"):
            dt = "tpdf"
        out_bytes = export_audio(
            mastered, sr, channels, out_format.lower(),
            dither_type=dt,
            auto_blank_sec=max(0.0, ab),
        )
        out_name = (filename or "master").rsplit(".", 1)[0] + f"_mastered.{out_format}"
        job["status"] = "done"
        job["progress"] = 100
        job["message"] = "Готово"
        job["result_bytes"] = out_bytes
        job["filename"] = out_name
        job["done_at"] = time.time()
    except Exception as e:
        job["status"] = "error"
        job["progress"] = 0
        job["message"] = ""
        job["error"] = str(e)
        job["done_at"] = time.time()


@app.post("/api/master")
async def api_master(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    target_lufs: Optional[float] = Form(None),
    preset: Optional[str] = Form(None),
    out_format: str = Form("wav"),
    style: str = Form("standard"),
    user: Optional[dict] = Depends(_get_current_user_optional),
):
    """
    Запуск мастеринга: возвращает job_id. Статус — GET /api/master/status/{job_id}.
    Результат — GET /api/master/result/{job_id} после status=done.
    """
    if not _allowed_file(file.filename or ""):
        raise HTTPException(400, "Формат не поддерживается. Разрешены: WAV, MP3, FLAC.")
    if out_format.lower() not in ("wav", "mp3", "flac"):
        raise HTTPException(400, "Формат экспорта: wav, mp3 или flac.")
    # MP3 требует ffmpeg — проверяем заранее, чтобы не запускать задачу впустую
    if out_format.lower() == "mp3" and not shutil.which("ffmpeg"):
        raise HTTPException(
            400,
            "Экспорт в MP3 требует ffmpeg, который не найден на сервере. "
            "Установите: sudo apt-get install -y ffmpeg",
        )
    # Rate limit только для гостей (Free); залогиненные и режим отладки — без ограничений
    if not user and not getattr(settings, "debug_mode", False):
        ip = _get_client_ip(request)
        limit_info = _check_rate_limit(ip)
        if not limit_info["ok"]:
            raise HTTPException(
                429,
                f"Лимит Free-тарифа исчерпан: {limit_info['limit']} мастеринга в день. "
                f"Сброс: {limit_info['reset_at']}. Перейдите на Pro для безлимитного доступа.",
            )

    if target_lufs is None:
        target_lufs = settings.default_target_lufs
    if preset and preset.lower() in PRESET_LUFS:
        target_lufs = PRESET_LUFS[preset.lower()]
    data = await file.read()
    if len(data) > settings.max_upload_mb * 1024 * 1024:
        raise HTTPException(400, f"Файл больше {settings.max_upload_mb} МБ.")
    try:
        load_audio_from_bytes(data, file.filename or "wav")
    except Exception as e:
        raise HTTPException(400, f"Не удалось прочитать аудио: {e}")

    if not user and not getattr(settings, "debug_mode", False):
        _record_usage(_get_client_ip(request))
    _prune_jobs()
    job_id = str(uuid.uuid4())
    style_key = style.lower() if style.lower() in STYLE_CONFIGS else "standard"
    _jobs[job_id] = {
        "status": "running",
        "progress": 0,
        "message": "Ожидание…",
        "created_at": time.time(),
        "result_bytes": None,
        "filename": None,
        "error": None,
        "before_lufs": None,
        "after_lufs": None,
        "target_lufs": target_lufs,
        "style": style_key,
    }
    sem = _sem_priority if _is_priority_user(user) else _sem_normal
    async def run_job():
        async with sem:
            await asyncio.to_thread(
                _run_mastering_job,
                job_id,
                data,
                file.filename or "audio.wav",
                target_lufs,
                out_format.lower(),
                style_key,
            )

    background_tasks.add_task(run_job)
    return {"job_id": job_id, "status": "running"}


@app.post("/api/v2/master")
async def api_master_v2(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    config: Optional[str] = Form(None),
    target_lufs: Optional[float] = Form(None),
    style: str = Form("standard"),
    out_format: str = Form("wav"),
    dither_type: Optional[str] = Form(None),
    auto_blank_sec: Optional[float] = Form(None),
    # PRO processing modules
    denoise_strength: Optional[float] = Form(None),
    deesser_enabled: Optional[str] = Form(None),
    deesser_threshold: Optional[float] = Form(None),
    transient_attack: Optional[float] = Form(None),
    transient_sustain: Optional[float] = Form(None),
    parallel_mix: Optional[float] = Form(None),
    dynamic_eq_enabled: Optional[str] = Form(None),
    user: Optional[dict] = Depends(_get_current_user_optional),
):
    """
    Мастеринг v2: цепочка из JSON-конфига.
    Форма: file (обязательно), config (JSON-строка, опционально), target_lufs, style, out_format,
    dither_type (tpdf | ns_e | ns_itu), auto_blank_sec (сек, обрезка тишины в конце).
    Ответ: job_id; статус и результат — те же GET /api/master/status/{job_id}, GET /api/master/result/{job_id}.
    """
    if not _allowed_file(file.filename or ""):
        raise HTTPException(400, "Формат не поддерживается. Разрешены: WAV, MP3, FLAC.")
    if out_format.lower() not in ("wav", "mp3", "flac"):
        raise HTTPException(400, "Формат экспорта: wav, mp3 или flac.")
    if out_format.lower() == "mp3" and not shutil.which("ffmpeg"):
        raise HTTPException(
            400,
            "Экспорт в MP3 требует ffmpeg. Установите: sudo apt-get install -y ffmpeg",
        )

    # Rate limit только для гостей; залогиненные и режим отладки — без ограничений
    if not user and not getattr(settings, "debug_mode", False):
        ip = _get_client_ip(request)
        limit_info = _check_rate_limit(ip)
        if not limit_info["ok"]:
            raise HTTPException(
                429,
                f"Лимит Free-тарифа исчерпан: {limit_info['limit']} мастеринга в день. "
                f"Сброс: {limit_info['reset_at']}. Перейдите на Pro для безлимитного доступа.",
            )

    if target_lufs is None:
        target_lufs = settings.default_target_lufs
    data = await file.read()
    if len(data) > settings.max_upload_mb * 1024 * 1024:
        raise HTTPException(400, f"Файл больше {settings.max_upload_mb} МБ.")
    try:
        load_audio_from_bytes(data, file.filename or "wav")
    except Exception as e:
        raise HTTPException(400, f"Не удалось прочитать аудио: {e}")

    chain_config: Optional[dict[str, Any]] = None
    if config and config.strip():
        try:
            chain_config = json.loads(config)
        except json.JSONDecodeError as e:
            raise HTTPException(400, f"Неверный JSON в config: {e}")

    if not user and not getattr(settings, "debug_mode", False):
        _record_usage(_get_client_ip(request))
    _prune_jobs()
    job_id = str(uuid.uuid4())
    style_key = style.lower() if style.lower() in STYLE_CONFIGS else "standard"
    _jobs[job_id] = {
        "status": "running",
        "progress": 0,
        "message": "Ожидание…",
        "created_at": time.time(),
        "result_bytes": None,
        "filename": None,
        "error": None,
        "before_lufs": None,
        "after_lufs": None,
        "target_lufs": target_lufs,
        "style": style_key,
    }

    pro_params: dict = {}
    if denoise_strength is not None and denoise_strength > 0:
        pro_params["denoise_strength"] = float(denoise_strength)
    if deesser_enabled and deesser_enabled.lower() in ("true", "1", "yes"):
        pro_params["deesser_enabled"] = True
        pro_params["deesser_threshold"] = float(deesser_threshold) if deesser_threshold is not None else -10.0
    if transient_attack is not None and transient_sustain is not None:
        pro_params["transient_attack"] = float(transient_attack)
        pro_params["transient_sustain"] = float(transient_sustain)
    if parallel_mix is not None and parallel_mix > 0:
        pro_params["parallel_mix"] = float(parallel_mix)
    if dynamic_eq_enabled and dynamic_eq_enabled.lower() in ("true", "1", "yes"):
        pro_params["dynamic_eq_enabled"] = True

    sem = _sem_priority if _is_priority_user(user) else _sem_normal
    async def run_job_v2():
        async with sem:
            await asyncio.to_thread(
                _run_mastering_job_v2,
                job_id,
                data,
                file.filename or "audio.wav",
                target_lufs,
                out_format.lower(),
                style_key,
                chain_config,
                dither_type=dither_type,
                auto_blank_sec=auto_blank_sec,
                pro_params=pro_params,
            )

    background_tasks.add_task(run_job_v2)
    return {"job_id": job_id, "status": "running", "version": "v2"}


_BATCH_MAX_FILES = 10


@app.post("/api/v2/batch")
async def api_v2_batch(
    request: Request,
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    config: Optional[str] = Form(None),
    target_lufs: Optional[float] = Form(None),
    style: str = Form("standard"),
    out_format: str = Form("wav"),
    dither_type: Optional[str] = Form(None),
    auto_blank_sec: Optional[float] = Form(None),
    user: Optional[dict] = Depends(_get_current_user_optional),
):
    """
    Пакетный мастеринг: несколько файлов с одинаковыми параметрами.
    Возвращает список { job_id, filename }. Статус и результат — как у одиночного: GET /api/master/status/{id}, GET /api/master/result/{id}.
    Максимум файлов: 10. Для Free каждый файл считается за 1 использование в дневном лимите.
    """
    if not files:
        raise HTTPException(400, "Отправьте хотя бы один файл.")
    if len(files) > _BATCH_MAX_FILES:
        raise HTTPException(400, f"Максимум {_BATCH_MAX_FILES} файлов за один запрос.")

    for f in files:
        if not _allowed_file(f.filename or ""):
            raise HTTPException(400, f"Формат не поддерживается: {f.filename}. Разрешены: WAV, MP3, FLAC.")
    if out_format.lower() not in ("wav", "mp3", "flac"):
        raise HTTPException(400, "Формат экспорта: wav, mp3 или flac.")
    if out_format.lower() == "mp3" and not shutil.which("ffmpeg"):
        raise HTTPException(400, "Экспорт в MP3 требует ffmpeg.")

    is_pro = user or getattr(settings, "debug_mode", False)
    if not is_pro:
        ip = _get_client_ip(request)
        limit_info = _check_rate_limit(ip)
        if limit_info["remaining"] < len(files):
            raise HTTPException(
                429,
                f"Недостаточно лимита. Осталось {limit_info['remaining']} мастерингов, файлов — {len(files)}. "
                f"Сброс: {limit_info['reset_at']}.",
            )

    if target_lufs is None:
        target_lufs = settings.default_target_lufs
    style_key = style.lower() if style.lower() in STYLE_CONFIGS else "standard"
    chain_config: Optional[dict[str, Any]] = None
    if config and config.strip():
        try:
            chain_config = json.loads(config)
        except json.JSONDecodeError as e:
            raise HTTPException(400, f"Неверный JSON в config: {e}")

    # Читаем и валидируем все файлы
    payloads: List[Tuple[bytes, str]] = []
    for f in files:
        data = await f.read()
        if len(data) > settings.max_upload_mb * 1024 * 1024:
            raise HTTPException(400, f"Файл {f.filename} больше {settings.max_upload_mb} МБ.")
        try:
            load_audio_from_bytes(data, f.filename or "wav")
        except Exception as e:
            raise HTTPException(400, f"Не удалось прочитать аудио {f.filename}: {e}")
        payloads.append((data, f.filename or "audio.wav"))

    if not is_pro:
        _record_usage_n(_get_client_ip(request), len(payloads))

    _prune_jobs()
    dt = dither_type or "tpdf"
    ab = float(auto_blank_sec or 0)
    jobs_created: List[dict] = []

    for data, filename in payloads:
        job_id = str(uuid.uuid4())
        _jobs[job_id] = {
            "status": "running",
            "progress": 0,
            "message": "Ожидание…",
            "created_at": time.time(),
            "result_bytes": None,
            "filename": None,
            "error": None,
            "before_lufs": None,
            "after_lufs": None,
            "target_lufs": target_lufs,
            "style": style_key,
        }
        jobs_created.append({"job_id": job_id, "filename": filename})

        batch_sem = _sem_priority if _is_priority_user(user) else _sem_normal
        async def run_one(
            jid: str,
            d: bytes,
            fname: str,
        ):
            async with batch_sem:
                await asyncio.to_thread(
                    _run_mastering_job_v2,
                    jid,
                    d,
                    fname,
                    target_lufs,
                    out_format.lower(),
                    style_key,
                    chain_config,
                    dither_type=dt,
                    auto_blank_sec=ab,
                )

        background_tasks.add_task(run_one, job_id, data, filename)

    return {"version": "v2", "batch": True, "jobs": jobs_created}


# Человекочитаемые названия модулей цепочки (для UI)
CHAIN_MODULE_LABELS = {
    "dc_offset": "Удаление DC-смещения",
    "peak_guard": "Защита от пиков",
    "target_curve": "Студийный EQ (Ozone 5 Equalizer)",
    "dynamics": "Многополосная динамика (Ozone 5 Dynamics)",
    "maximizer": "Максимайзер",
    "normalize_lufs": "Нормализация LUFS",
    "final_spectral_balance": "Финальная частотная коррекция",
    "style_eq": "Жанровый EQ",
    "exciter": "Гармонический эксайтер (Ozone 5 Exciter)",
    "imager": "Стерео-расширение (Ozone 5 Imager)",
    "reverb": "Ревербератор (plate/room/hall/theater/cathedral)",
}


@app.get("/api/v2/chain/default")
def api_v2_chain_default(style: str = "standard", target_lufs: float = -14.0):
    """
    Конфиг цепочки по умолчанию: полный список модулей с id, label, enabled и параметрами.
    Ответ можно отправить в POST /api/v2/master (поле config) — в т.ч. с изменённым порядком модулей.
    При отправке поле label можно не передавать (игнорируется бэкендом).
    """
    config = MasteringChain.default_config(target_lufs=target_lufs, style=style)
    modules = []
    for m in config["modules"]:
        m = dict(m)
        mid = m.get("id")
        if mid:
            m["label"] = CHAIN_MODULE_LABELS.get(mid, mid)
            modules.append(m)
    return {"version": "v2", "style": style, "target_lufs": target_lufs, "modules": modules}


@app.post("/api/v2/analyze")
async def api_v2_analyze(file: UploadFile = File(...), extended: bool = Form(False)):
    """
    Загрузить файл и вернуть анализ: LUFS, peak dBFS, длительность, sample rate, stereo_correlation.
    extended=true: дополнительно spectrum_bars, lufs_timeline, timeline_step_sec; для стерео — vectorscope_points (массив [l, r] до 1000 точек).
    """
    if not _allowed_file(file.filename or ""):
        raise HTTPException(400, "Формат не поддерживается. Разрешены: WAV, MP3, FLAC.")
    fname = file.filename or ""
    if fname.lower().endswith(".mp3") and not shutil.which("ffmpeg"):
        raise HTTPException(
            400,
            "Чтение MP3 требует ffmpeg. Установите: sudo apt-get install -y ffmpeg",
        )
    data = await file.read()
    if len(data) > settings.max_upload_mb * 1024 * 1024:
        raise HTTPException(400, f"Файл больше {settings.max_upload_mb} МБ.")
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
    out = {
        "version": "v2",
        "lufs": round(lufs, 2) if not np.isnan(lufs) else None,
        "peak_dbfs": round(peak_dbfs, 2),
        "duration_sec": round(duration_sec, 3),
        "sample_rate": sr,
        "channels": channels,
    }
    if correlation is not None:
        out["stereo_correlation"] = round(correlation, 4)
    # Streaming Loudness Preview — penalty для каждой платформы
    if not np.isnan(lufs):
        _streaming_platforms = {
            "Spotify":              -14.0,
            "YouTube":              -14.0,
            "Apple Music":          -16.0,
            "Tidal":                -14.0,
            "Amazon Music":         -14.0,
            "Broadcast (EBU R128)": -23.0,
        }
        streaming_preview = {}
        for platform, target in _streaming_platforms.items():
            penalty = round(max(0.0, lufs - target), 2)
            gain_applied = round(min(0.0, target - lufs), 2)
            if penalty > 6.0:
                status = "loud"
            elif penalty > 1.0:
                status = "ok"
            else:
                status = "optimal"
            streaming_preview[platform] = {
                "target_lufs": target,
                "penalty_db": penalty,
                "gain_applied_db": gain_applied,
                "status": status,
            }
        out["streaming_preview"] = streaming_preview

    if extended:
        if audio.size >= 4096:
            try:
                out["spectrum_bars"] = compute_spectrum_bars(audio, sr)
            except Exception:
                pass
            # Separate Mid/Side spectrum for M/S analysis
            if channels == 2 and audio.ndim == 2:
                try:
                    mid = ((audio[:, 0] + audio[:, 1]) * 0.5).astype(np.float32)
                    side = ((audio[:, 0] - audio[:, 1]) * 0.5).astype(np.float32)
                    out["spectrum_bars_mid"] = compute_spectrum_bars(mid, sr)
                    out["spectrum_bars_side"] = compute_spectrum_bars(side, sr)
                except Exception:
                    pass
        try:
            lufs_timeline, timeline_step_sec = compute_lufs_timeline(audio, sr)
            out["lufs_timeline"] = lufs_timeline
            out["timeline_step_sec"] = timeline_step_sec
        except Exception:
            pass
        if channels == 2 and audio.ndim == 2:
            try:
                out["vectorscope_points"] = compute_vectorscope_points(audio, max_points=1000)
            except Exception:
                pass
    return out


@app.post("/api/v2/reference-match")
async def api_v2_reference_match(
    file: UploadFile = File(...),
    reference: UploadFile = File(...),
    strength: float = Form(0.8),
    out_format: str = Form("wav"),
):
    """
    Reference Track Mastering: подгоняет спектральный баланс треку к эталону.
    Принимает два файла (file — основной трек, reference — эталон) и параметр strength (0.0–1.0).
    Возвращает обработанный файл в формате out_format (wav/mp3/flac).
    """
    if not _allowed_file(file.filename or ""):
        raise HTTPException(400, "Формат основного файла не поддерживается.")
    if not _allowed_file(reference.filename or ""):
        raise HTTPException(400, "Формат эталонного файла не поддерживается.")
    if out_format.lower() not in ("wav", "mp3", "flac"):
        raise HTTPException(400, "Формат экспорта: wav, mp3 или flac.")
    strength = float(np.clip(strength, 0.0, 1.0))
    data_src = await file.read()
    data_ref = await reference.read()
    if len(data_src) > 200 * 1024 * 1024:
        raise HTTPException(400, "Основной файл больше 200 МБ.")
    if len(data_ref) > 200 * 1024 * 1024:
        raise HTTPException(400, "Эталонный файл больше 200 МБ.")
    try:
        audio_src, sr_src = load_audio_from_bytes(data_src, file.filename or "wav")
        audio_ref, sr_ref = load_audio_from_bytes(data_ref, reference.filename or "wav")
    except Exception as e:
        raise HTTPException(400, f"Не удалось прочитать аудио: {e}")
    try:
        result = apply_reference_match(audio_src, sr_src, audio_ref, sr_ref, strength=strength)
    except Exception as e:
        raise HTTPException(500, f"Ошибка обработки: {e}")
    channels = 1 if result.ndim == 1 else int(result.shape[1])
    out_bytes = export_audio(result, sr_src, channels, out_format.lower())
    out_name = (file.filename or "track").rsplit(".", 1)[0] + f"_ref_matched.{out_format}"
    return Response(
        content=out_bytes,
        media_type=f"audio/{out_format}",
        headers={"Content-Disposition": f'attachment; filename="{out_name}"'},
    )


def _json_safe_float(v):
    """Для ответа JSON: inf/-inf/nan заменяем на None."""
    if v is None:
        return None
    try:
        f = float(v)
        if np.isnan(f) or np.isinf(f):
            return None
        return f
    except (TypeError, ValueError):
        return None


@app.get("/api/master/status/{job_id}")
async def api_master_status(job_id: str):
    """Статус задачи мастеринга: progress 0–100, message."""
    _prune_jobs()
    if job_id not in _jobs:
        raise HTTPException(404, "Задача не найдена")
    job = _jobs[job_id]
    return {
        "status": job["status"],
        "progress": int(job.get("progress", 0)),
        "message": job.get("message") or "",
        "error": job.get("error"),
        "before_lufs": _json_safe_float(job.get("before_lufs")),
        "after_lufs": _json_safe_float(job.get("after_lufs")),
        "target_lufs": _json_safe_float(job.get("target_lufs")),
        "style": job.get("style", "standard"),
    }


@app.get("/api/master/result/{job_id}")
async def api_master_result(job_id: str):
    """Скачать результат мастеринга (после status=done)."""
    if job_id not in _jobs:
        raise HTTPException(404, "Задача не найдена")
    job = _jobs[job_id]
    if job["status"] != "done" or not job.get("result_bytes"):
        raise HTTPException(400, "Результат ещё не готов или задача с ошибкой")
    out_bytes = job["result_bytes"]
    filename = job["filename"] or "mastered.wav"
    del _jobs[job_id]
    return Response(
        content=out_bytes,
        media_type="application/octet-stream",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


# Раздача фронтенда — после всех API-маршрутов, иначе POST /api/measure отдаёт 405 Method Not Allowed
_frontend = Path(__file__).resolve().parent.parent.parent / "frontend"

if _frontend.is_dir():
    from fastapi.responses import FileResponse

    @app.get("/", include_in_schema=False)
    async def landing():
        """Лендинг-страница."""
        p = _frontend / "landing.html"
        return FileResponse(str(p) if p.is_file() else str(_frontend / "index.html"))

    @app.get("/app", include_in_schema=False)
    async def mastering_app():
        """Страница мастеринга. Встраивает флаг режима отладки в HTML до загрузки JS."""
        path = _frontend / "index.html"
        html = path.read_text(encoding="utf-8")
        debug_on = getattr(settings, "debug_mode", False) or (
            os.environ.get("MAGIC_MASTER_DEBUG", "").strip().lower() in ("1", "true", "yes", "on")
        )
        if debug_on:
            html = html.replace(
                "window.__MAGIC_MASTER_DEBUG__=false",
                "window.__MAGIC_MASTER_DEBUG__=true",
            )
        return HTMLResponse(content=html)

    @app.get("/pricing", include_in_schema=False)
    async def pricing():
        """Страница тарифов."""
        p = _frontend / "pricing.html"
        return FileResponse(str(p) if p.is_file() else str(_frontend / "landing.html"))

    @app.get("/login", include_in_schema=False)
    async def login_page():
        """Страница входа."""
        p = _frontend / "login.html"
        return FileResponse(str(p) if p.is_file() else str(_frontend / "index.html"))

    @app.get("/register", include_in_schema=False)
    async def register_page():
        """Страница регистрации."""
        p = _frontend / "register.html"
        return FileResponse(str(p) if p.is_file() else str(_frontend / "index.html"))

    @app.get("/dashboard", include_in_schema=False)
    async def dashboard_page():
        """Дашборд пользователя: история и профиль."""
        p = _frontend / "dashboard.html"
        return FileResponse(str(p) if p.is_file() else str(_frontend / "index.html"))

    @app.get("/favicon.ico", include_in_schema=False)
    async def favicon():
        """Убирает 404 в консоли: отдаём 204 No Content (иконки нет)."""
        return Response(status_code=204)

    @app.get("/progress.html", include_in_schema=False)
    async def progress_page():
        """Страница «Статус плана» — содержимое PROGRESS.md в HTML."""
        import html as html_module
        body = ""
        if _PROGRESS_PATH.is_file():
            raw = _PROGRESS_PATH.read_text(encoding="utf-8")
            body = f"<pre id='progress-content'>{html_module.escape(raw)}</pre>"
        else:
            body = "<p>Файл PROGRESS.md на сервере отсутствует.</p>"
        html = (
            "<!DOCTYPE html><html lang='ru'><head><meta charset='UTF-8'>"
            "<meta name='viewport' content='width=device-width,initial-scale=1'>"
            "<title>Статус плана — Magic Master</title>"
            "<style>body{background:#040408;color:#eee;font-family:system-ui,sans-serif;padding:1.5rem;max-width:900px;margin:0 auto;line-height:1.5;} "
            "pre{white-space:pre-wrap;word-break:break-word;}</style></head><body>"
            "<h1>Статус плана разработки</h1>" + body + "</body></html>"
        )
        return HTMLResponse(content=html)

    app.mount("/", StaticFiles(directory=str(_frontend), html=True), name="frontend")
