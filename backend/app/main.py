# @file main.py
# @description FastAPI-приложение: загрузка, мастеринг, экспорт, замер LUFS
# @dependencies app.pipeline, app.chain, app.config
# @created 2026-02-26

import asyncio
import base64
import datetime
import json
import logging
import os
import shutil
import time
import uuid

logger = logging.getLogger(__name__)

# Минимальный 1×1 PNG (прозрачный) для PWA-иконок, если файлы отсутствуют
_PNG_1X1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)

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
from . import settings_store
from .database import (
    DB_AVAILABLE,
    log_ai_usage,
    log_mastering_job_end,
    log_mastering_job_start,
    NewsPost,
    check_and_expire_subscription,
    create_api_key,
    create_mastering_record,
    create_saved_preset,
    create_tables,
    create_user,
    delete_mastering_record,
    delete_saved_preset,
    get_api_keys_for_user,
    get_db,
    get_news_posts,
    get_saved_preset_by_id,
    get_user_by_api_key,
    get_user_by_email,
    get_user_history,
    get_user_presets,
    get_user_stats,
    revoke_api_key,
)
from .version import __version__, __build_date__
from .helpers import (
    allowed_file as _allowed_file,
    check_audio_magic_bytes as _check_audio_magic_bytes,
    get_client_ip as _get_client_ip,
)
from . import ai as ai_module
from .admin import router as admin_router
from .payments import router as payments_router
from .pipeline import (
    DENOISE_PRESETS,
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

# ─── Auth brute-force protection (P33): 10 попыток входа/регистрации в минуту ─
_AUTH_LIMIT_PER_MINUTE = 10
_auth_attempts: dict[str, dict] = {}  # ip -> {"count": int, "window_start": float}


def _check_auth_rate_limit(ip: str) -> bool:
    """True = разрешено, False = заблокировано (слишком много попыток)."""
    now = time.time()
    entry = _auth_attempts.get(ip)
    if entry is None or now - entry["window_start"] >= 60:
        _auth_attempts[ip] = {"count": 1, "window_start": now}
        return True
    entry["count"] += 1
    return entry["count"] <= _AUTH_LIMIT_PER_MINUTE


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


# ─── AI rate limiting (по тарифу: Free 5/день, Pro 50, Studio без лимита) ─────
def _get_ai_identifier(request: Request, user: Optional[dict]) -> str:
    """Ключ для учёта AI-запросов: user_id если авторизован, иначе IP."""
    if user and user.get("sub"):
        return f"user:{user['sub']}"
    return f"ip:{_get_client_ip(request)}"


def _get_tier_for_ai(user: Optional[dict], request: Request) -> str:
    """Тариф для лимитов AI: debug/pro/studio без лимита, иначе free."""
    if getattr(settings, "debug_mode", False):
        return "studio"
    if user:
        return (user.get("tier") or "pro").lower()
    return "free"


app = FastAPI(
    title="Magic Master — автоматический мастеринг",
    description="Загрузите трек → нажмите Magic Master → скачайте результат с целевой громкостью LUFS.",
    version=__version__,
)

# P56: CORS — настраиваемый список origins (MAGIC_MASTER_CORS_ORIGINS); пусто = ["*"]
_cors_origins_str = getattr(settings, "cors_origins", "") or ""
_cors_origins = (
    [o.strip() for o in _cors_origins_str.split(",") if o.strip()]
    if _cors_origins_str.strip()
    else ["*"]
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── P46: Глобальный rate limit per-IP ────────────────────────────────────────
# 300 запросов/минуту с одного IP (исключая статику и SSE).
# В режиме debug_mode лимит отключён.
_GLOBAL_RATE_LIMIT = getattr(settings, "global_rate_limit", 300)
_GLOBAL_RATE_WINDOW = 60        # секунд
_GLOBAL_RATE_EXEMPT_PREFIXES = ("/api/master/progress/",)  # SSE — исключаем
_global_rate: dict[str, dict] = {}  # ip → {count, window_start}


@app.middleware("http")
async def global_rate_limit_middleware(request: Request, call_next):
    if not getattr(settings, "debug_mode", False):
        ip = _get_client_ip(request)
        path = request.url.path
        # Пропускаем статику и SSE
        is_static = not path.startswith("/api/")
        is_exempt = any(path.startswith(p) for p in _GLOBAL_RATE_EXEMPT_PREFIXES)
        if not is_static and not is_exempt:
            now = time.time()
            entry = _global_rate.get(ip)
            if entry is None or now - entry["window_start"] >= _GLOBAL_RATE_WINDOW:
                _global_rate[ip] = {"count": 1, "window_start": now}
            else:
                entry["count"] += 1
                if entry["count"] > _GLOBAL_RATE_LIMIT:
                    from fastapi.responses import JSONResponse
                    retry_after = int(_GLOBAL_RATE_WINDOW - (now - entry["window_start"])) + 1
                    return JSONResponse(
                        {"detail": f"Слишком много запросов. Повторите через {retry_after} сек."},
                        status_code=429,
                        headers={"Retry-After": str(retry_after)},
                    )
            # Периодически очищаем старые записи
            if len(_global_rate) > 5000:
                cutoff = time.time() - _GLOBAL_RATE_WINDOW
                for old_ip in [k for k, v in _global_rate.items() if v["window_start"] < cutoff]:
                    _global_rate.pop(old_ip, None)
    return await call_next(request)


# ─── Режим обслуживания: 503 для не-админских маршрутов ───────────────────────
_MAINTENANCE_EXEMPT_PREFIXES = ("/api/admin", "/admin", "/api/auth/login", "/docs", "/redoc", "/openapi.json")


@app.middleware("http")
async def maintenance_mode_middleware(request: Request, call_next):
    """При maintenance_mode=True возвращает 503 для всех маршрутов кроме админки и входа."""
    try:
        if settings_store.get_setting_bool("maintenance_mode", False):
            path = request.url.path.rstrip("/") or "/"
            if not any(path == p or path.startswith(p + "/") for p in _MAINTENANCE_EXEMPT_PREFIXES):
                from fastapi.responses import HTMLResponse
                html = """<!DOCTYPE html><html><head><meta charset="utf-8"><title>Обслуживание</title>
<style>body{font-family:system-ui;display:flex;align-items:center;justify-content:center;min-height:100vh;margin:0;background:#1a1a2e;color:#eee;}
.box{text-align:center;padding:2rem;}</style></head><body><div class="box">
<h1>Сайт на обслуживании</h1><p>Мы скоро вернёмся. Попробуйте позже.</p></div></body></html>"""
                return HTMLResponse(html, status_code=503, headers={"Retry-After": "300"})
    except Exception:  # noqa: BLE001
        pass
    return await call_next(request)
# ──────────────────────────────────────────────────────────────────────────────

os.makedirs(settings_store.get_setting_str("temp_dir") or getattr(settings, "temp_dir", "/tmp/masterflow"), exist_ok=True)

# Создаём таблицы SQLite при запуске (idempotent)
create_tables()


def _ensure_initial_admin() -> None:
    """Создать первого администратора если задан MAGIC_MASTER_ADMIN_EMAIL."""
    admin_email = getattr(settings, "admin_email", "").strip()
    if not admin_email:
        logger.info("Initial admin: не задан MAGIC_MASTER_ADMIN_EMAIL в .env — вход в /admin недоступен")
        return
    if not DB_AVAILABLE or not AUTH_AVAILABLE:
        logger.warning("Initial admin: БД или auth недоступны, администратор не создан")
        return
    try:
        from .database import SessionLocal, User
        if SessionLocal is None or User is None:
            return
        db = SessionLocal()
        try:
            existing = db.query(User).filter(User.email == admin_email.lower()).first()
            if existing:
                if not existing.is_admin:
                    existing.is_admin = True
                admin_pwd = (getattr(settings, "admin_password", "") or "").strip()
                if admin_pwd:
                    from .auth import get_password_hash as _hash
                    existing.hashed_password = _hash(admin_pwd)
                db.commit()
                logger.info("Initial admin: обновлён существующий пользователь %s", admin_email.lower())
            else:
                from .auth import get_password_hash as _hash
                default_pwd = (getattr(settings, "admin_password", "") or "").strip() or "changeMe123!"
                hashed = _hash(default_pwd)
                admin = User(
                    email=admin_email.lower(),
                    hashed_password=hashed,
                    tier="studio",
                    is_admin=True,
                    created_at=time.time(),
                )
                db.add(admin)
                db.commit()
                logger.info("Initial admin: создан администратор %s (пароль из MAGIC_MASTER_ADMIN_PASSWORD или changeMe123!)", admin_email.lower())
        finally:
            db.close()
    except Exception as e:  # noqa: BLE001
        logger.exception("Initial admin: ошибка при создании/обновлении администратора: %s", e)


_ensure_initial_admin()

# P51: Telegram-уведомление о запуске сервера (асинхронно)
try:
    from .notifier import notify_server_startup
    _host = f"{os.environ.get('MAGIC_MASTER_HOST', '0.0.0.0')}:{os.environ.get('MAGIC_MASTER_PORT', '8000')}"
    notify_server_startup(__version__, _host)
except Exception:  # noqa: BLE001
    pass


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
def _get_current_user_optional(
    authorization: Optional[str] = Header(None),
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
) -> Optional[dict]:
    """Dependency: декодирует Bearer JWT или X-API-Key, возвращает payload или None.

    P28: если подписка истекла — понижает tier до 'free'.
    P52: поддержка X-API-Key заголовка для Pro/Studio пользователей.
    """
    # P52: X-API-Key как альтернатива Bearer JWT
    if x_api_key and DB_AVAILABLE:
        try:
            db_gen = get_db()
            db = next(db_gen)
            try:
                api_user = get_user_by_api_key(db, x_api_key.strip())
                if api_user and not getattr(api_user, "is_blocked", False):
                    return {
                        "sub": str(api_user.id),
                        "email": api_user.email,
                        "tier": api_user.tier,
                        "is_admin": bool(getattr(api_user, "is_admin", False)),
                        "auth_method": "api_key",
                    }
            finally:
                try:
                    next(db_gen)
                except StopIteration:
                    pass
        except Exception:
            pass
        return None  # неверный ключ → не пробуем Bearer

    token = extract_bearer_token(authorization)
    if not token:
        return None
    payload = decode_token(token)
    if payload is None:
        return None
    # Проверяем истечение подписки для платных тарифов
    uid = payload.get("sub")
    tier_in_token = payload.get("tier", "free")
    if DB_AVAILABLE and uid and tier_in_token != "free":
        try:
            db_gen = get_db()
            db = next(db_gen)
            try:
                actual_tier = check_and_expire_subscription(db, int(uid))
                if actual_tier is not None and actual_tier != tier_in_token:
                    payload = dict(payload, tier=actual_tier)
            finally:
                try:
                    next(db_gen)
                except StopIteration:
                    pass
        except Exception:
            pass
    return payload


def _prune_jobs() -> None:
    """Удаляет старые завершённые задачи и ограничивает число записей в _jobs."""
    now = time.time()
    ttl = settings_store.get_setting_int("jobs_done_ttl_seconds", 3600)
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
    """Версия приложения и дата сборки для интерфейса и мониторинга."""
    return {"version": __version__, "build_date": __build_date__}


@app.get("/api/health")
def api_health():
    """Расширенный health-check для мониторинга и страницы статуса. P47."""
    import shutil as _shutil
    import platform as _platform

    # База данных
    db_status = "ok"
    db_detail = ""
    try:
        if DB_AVAILABLE:
            from .database import SessionLocal as _SL
            if _SL:
                _db = _SL()
                _db.execute(__import__("sqlalchemy").text("SELECT 1"))
                _db.close()
        else:
            db_status = "unavailable"
            db_detail = "SQLAlchemy не установлен"
    except Exception as e:
        db_status = "error"
        db_detail = str(e)[:120]

    # Дисковое пространство temp_dir
    disk_status = "ok"
    disk_free_mb = None
    try:
        usage = _shutil.disk_usage(settings_store.get_setting_str("temp_dir") or getattr(settings, "temp_dir", "/tmp/masterflow"))
        disk_free_mb = round(usage.free / 1024 / 1024)
        if disk_free_mb < 200:
            disk_status = "low"
    except Exception:
        disk_status = "unknown"

    # ffmpeg
    ffmpeg_ok = bool(_shutil.which("ffmpeg"))

    # Активные задачи мастеринга
    jobs_running = sum(1 for j in _jobs.values() if j.get("status") == "running")
    jobs_total   = len(_jobs)

    overall = "ok"
    if db_status != "ok":
        overall = "degraded"
    if disk_status == "low":
        overall = "degraded"

    # Флаги функций (из админки): по умолчанию True, если не заданы
    feature_ai = settings_store.get_setting("feature_ai_enabled")
    feature_batch = settings_store.get_setting("feature_batch_enabled")
    feature_registration = settings_store.get_setting("feature_registration_enabled")
    features = {
        "ai_enabled": feature_ai is None or feature_ai is True or (isinstance(feature_ai, str) and feature_ai.lower() in ("true", "1", "yes")),
        "batch_enabled": feature_batch is None or feature_batch is True or (isinstance(feature_batch, str) and feature_batch.lower() in ("true", "1", "yes")),
        "registration_enabled": feature_registration is None or feature_registration is True or (isinstance(feature_registration, str) and feature_registration.lower() in ("true", "1", "yes")),
    }

    return {
        "status": overall,
        "version": __version__,
        "build_date": __build_date__,
        "uptime_since": _APP_START_TIME,
        "python": _platform.python_version(),
        "components": {
            "database":   {"status": db_status, "detail": db_detail},
            "disk":       {"status": disk_status, "free_mb": disk_free_mb},
            "ffmpeg":     {"status": "ok" if ffmpeg_ok else "missing"},
        },
        "jobs": {"running": jobs_running, "total_cached": jobs_total},
        "features": features,
    }


_APP_START_TIME = datetime.datetime.utcnow().isoformat() + "Z"
_APP_START_TS = time.time()


@app.get("/api/metrics")
def api_metrics():
    """
    Метрики для внешнего мониторинга (скрапинг, дашборды). P58.
    Возвращает плоский JSON с числовыми и строковыми показателями.
    """
    uptime_seconds = max(0, int(time.time() - _APP_START_TS))
    jobs_running = sum(1 for j in _jobs.values() if j.get("status") == "running")
    jobs_total = len(_jobs)
    return {
        "uptime_seconds": uptime_seconds,
        "jobs_running": jobs_running,
        "jobs_total": jobs_total,
        "version": __version__,
        "build_date": __build_date__,
    }


@app.get("/api/locale")
def api_locale():
    """
    Доступные локали для интерфейса (i18n). P59.
    Строки переводов загружаются фронтендом из /locales/{lang}.json.
    """
    return {"available": ["ru", "en"], "default": "ru"}


# ─── P54: Кастомные страницы ошибок ──────────────────────────────────────────
from fastapi.responses import FileResponse as _FRE
from starlette.requests import Request as _SR
from starlette.exceptions import HTTPException as _SHTTPException

def _error_html_path(code: int) -> Optional[Path]:
    """Возвращает путь к frontend/<code>.html если файл существует."""
    try:
        p = Path(__file__).resolve().parent.parent.parent / "frontend" / f"{code}.html"
        return p if p.is_file() else None
    except Exception:
        return None


@app.exception_handler(404)
async def handler_404(request: _SR, exc: _SHTTPException):
    if request.url.path.startswith("/api/"):
        return Response(
            json.dumps({"detail": str(getattr(exc, "detail", "Not found"))}),
            status_code=404, media_type="application/json",
        )
    p = _error_html_path(404)
    if p:
        return _FRE(str(p), status_code=404)
    return HTMLResponse("<h1>404 Not Found</h1>", status_code=404)


@app.exception_handler(429)
async def handler_429(request: _SR, exc):
    detail = str(getattr(exc, "detail", "Too Many Requests"))
    retry = getattr(exc, "headers", {}).get("Retry-After", "60") if hasattr(exc, "headers") else "60"
    if request.url.path.startswith("/api/"):
        return Response(
            json.dumps({"detail": detail}),
            status_code=429, media_type="application/json",
            headers={"Retry-After": str(retry)},
        )
    p = _error_html_path(429)
    if p:
        from fastapi.responses import RedirectResponse
        return RedirectResponse(url=f"/429?retry={retry}", status_code=302)
    return HTMLResponse(f"<h1>429 Too Many Requests</h1><p>{detail}</p>", status_code=429)


@app.exception_handler(500)
async def handler_500(request: _SR, exc: Exception):
    if request.url.path.startswith("/api/"):
        return Response(
            json.dumps({"detail": "Внутренняя ошибка сервера"}),
            status_code=500, media_type="application/json",
        )
    p = _error_html_path(500)
    if p:
        return _FRE(str(p), status_code=500)
    return HTMLResponse("<h1>500 Internal Server Error</h1>", status_code=500)


# ─── Admin router (P19) ──────────────────────────────────────────────────────
app.include_router(admin_router)

# ─── Payments router (P23) ───────────────────────────────────────────────────
app.include_router(payments_router)


# ─── Публичные новости (P21) ─────────────────────────────────────────────────
@app.get("/api/news")
def api_news_public(limit: int = 5):
    """Последние опубликованные новости для лендинга (без авторизации)."""
    if not DB_AVAILABLE:
        return {"posts": []}
    from .database import SessionLocal as _SL
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


# ─── Флаги функций (из админки) ───────────────────────────────────────────────
def _require_feature_ai() -> None:
    """503 если AI отключён в настройках."""
    if not settings_store.get_setting_bool("feature_ai_enabled", True):
        raise HTTPException(503, "Функции AI временно отключены администратором.")


def _require_feature_registration() -> None:
    """503 если регистрация отключена."""
    if not settings_store.get_setting_bool("feature_registration_enabled", True):
        raise HTTPException(503, "Регистрация новых пользователей отключена.")


def _require_feature_batch() -> None:
    """503 если пакетная обработка отключена."""
    if not settings_store.get_setting_bool("feature_batch_enabled", True):
        raise HTTPException(503, "Пакетный мастеринг временно отключён.")


# ─── AI endpoints ────────────────────────────────────────────────────────────
@app.get("/api/ai/limits")
async def api_ai_limits(
    request: Request,
    user: Optional[dict] = Depends(_get_current_user_optional),
):
    """Лимиты AI-запросов по тарифу: Free 5/день, Pro 50, Studio без лимита."""
    _require_feature_ai()
    tier = _get_tier_for_ai(user, request)
    ident = _get_ai_identifier(request, user)
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
    """Тело запроса: либо analysis (JSON от /api/v2/analyze), либо пусто — тогда нужен file."""
    analysis: Optional[dict] = None


@app.post("/api/ai/recommend")
async def api_ai_recommend(
    request: Request,
    user: Optional[dict] = Depends(_get_current_user_optional),
    file: Optional[UploadFile] = File(None),
    body: Optional[AIRecommendRequest] = None,
):
    """
    Рекомендация пресета по анализу трека.
    Вариант 1: передайте file — будет выполнен анализ (extended) и по нему рекомендация.
    Вариант 2: передайте body.analysis (JSON от GET /api/v2/analyze) — рекомендация по нему.
    Ответ: style, target_lufs, chain_config (опц.), reason.
    """
    _require_feature_ai()
    tier = _get_tier_for_ai(user, request)
    ident = _get_ai_identifier(request, user)
    limit_info = ai_module.check_ai_rate_limit(ident, tier)
    if not limit_info["ok"]:
        raise HTTPException(
            429,
            f"Лимит AI-запросов исчерпан: {limit_info['limit']}/день. Сброс: {limit_info['reset_at']}.",
        )

    analysis = None
    if body and body.analysis:
        analysis = body.analysis
    elif file and file.filename and _allowed_file(file.filename):
        # Выполнить анализ как в api_v2_analyze (extended=True)
        fname = file.filename or ""
        if fname.lower().endswith(".mp3") and not shutil.which("ffmpeg"):
            raise HTTPException(400, "Чтение MP3 требует ffmpeg.")
        data = await file.read()
        max_mb = settings_store.get_setting_int("max_upload_mb", 100)
        if len(data) > max_mb * 1024 * 1024:
            raise HTTPException(400, f"Файл больше {max_mb} МБ.")
        if not _check_audio_magic_bytes(data, fname):
            raise HTTPException(400, "Содержимое файла не соответствует формату. Ожидается WAV, MP3 или FLAC.")
        try:
            audio, sr = load_audio_from_bytes(data, fname)
        except Exception as e:
            logger.error("ai/recommend: load_audio failed filename=%s error=%s", fname, str(e)[:200])
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
        analysis = {
            "lufs": round(lufs, 2) if not np.isnan(lufs) else None,
            "peak_dbfs": round(peak_dbfs, 2),
            "duration_sec": round(duration_sec, 3),
            "sample_rate": sr,
            "channels": channels,
            "stereo_correlation": round(correlation, 4) if correlation is not None else None,
        }
        if audio.size >= 4096:
            try:
                analysis["spectrum_bars"] = compute_spectrum_bars(audio, sr)
            except Exception:
                pass

    if not analysis:
        raise HTTPException(
            400,
            "Передайте file (аудиофайл) или body.analysis (JSON от /api/v2/analyze).",
        )

    result = ai_module.recommend_preset(analysis)
    ai_module.record_ai_usage(ident)
    try:
        uid = user.get("sub") if user else None
        uid_int = int(uid) if uid is not None else None
        log_ai_usage("recommend", uid_int, tier)
    except (TypeError, ValueError, Exception):  # noqa: BLE001
        pass
    return result


class AIReportRequest(BaseModel):
    """Тело запроса: analysis (JSON от /api/v2/analyze) или пусто — тогда нужен file."""
    analysis: Optional[dict] = None


@app.post("/api/ai/report")
async def api_ai_report(
    request: Request,
    user: Optional[dict] = Depends(_get_current_user_optional),
    file: Optional[UploadFile] = File(None),
    body: Optional[AIReportRequest] = None,
):
    """
    Текстовый отчёт по анализу трека + рекомендации на русском.
    Вариант 1: передайте file — выполнится анализ и по нему отчёт.
    Вариант 2: передайте body.analysis (JSON от /api/v2/analyze).
    Ответ: summary, recommendations (список строк).
    """
    _require_feature_ai()
    tier = _get_tier_for_ai(user, request)
    ident = _get_ai_identifier(request, user)
    limit_info = ai_module.check_ai_rate_limit(ident, tier)
    if not limit_info["ok"]:
        raise HTTPException(
            429,
            f"Лимит AI-запросов исчерпан: {limit_info['limit']}/день. Сброс: {limit_info['reset_at']}.",
        )

    analysis = None
    if body and body.analysis:
        analysis = body.analysis
    elif file and file.filename and _allowed_file(file.filename):
        fname = file.filename or ""
        if fname.lower().endswith(".mp3") and not shutil.which("ffmpeg"):
            raise HTTPException(400, "Чтение MP3 требует ffmpeg.")
        data = await file.read()
        max_mb = settings_store.get_setting_int("max_upload_mb", 100)
        if len(data) > max_mb * 1024 * 1024:
            raise HTTPException(400, f"Файл больше {max_mb} МБ.")
        if not _check_audio_magic_bytes(data, fname):
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
        analysis = {
            "lufs": round(lufs, 2) if not np.isnan(lufs) else None,
            "peak_dbfs": round(peak_dbfs, 2),
            "duration_sec": round(duration_sec, 3),
            "sample_rate": sr,
            "channels": channels,
            "stereo_correlation": round(correlation, 4) if correlation is not None else None,
        }

    if not analysis:
        raise HTTPException(
            400,
            "Передайте file (аудиофайл) или body.analysis (JSON от /api/v2/analyze).",
        )

    result = ai_module.report_with_recommendations(analysis)
    ai_module.record_ai_usage(ident)
    try:
        uid = user.get("sub") if user else None
        uid_int = int(uid) if uid is not None else None
        log_ai_usage("report", uid_int, tier)
    except (TypeError, ValueError, Exception):  # noqa: BLE001
        pass
    return result


class AINlConfigRequest(BaseModel):
    """Запрос на преобразование естественного языка в настройки цепочки."""
    text: str
    current_config: Optional[dict] = None


@app.post("/api/ai/nl-config")
async def api_ai_nl_config(
    request: Request,
    body: AINlConfigRequest,
    user: Optional[dict] = Depends(_get_current_user_optional),
):
    """
    Преобразование запроса на естественном языке в настройки цепочки и target_lufs.
    Пример: «потише, больше воздуха, меньше сибилянтов» → chain_config и/или target_lufs.
    """
    _require_feature_ai()
    tier = _get_tier_for_ai(user, request)
    ident = _get_ai_identifier(request, user)
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
    try:
        uid = user.get("sub") if user else None
        uid_int = int(uid) if uid is not None else None
        log_ai_usage("nl_config", uid_int, tier)
    except (TypeError, ValueError, Exception):  # noqa: BLE001
        pass
    return result


class AIChatRequest(BaseModel):
    """Запрос к чат-помощнику."""
    messages: List[dict]
    context: Optional[dict] = None


@app.post("/api/ai/chat")
async def api_ai_chat(
    request: Request,
    body: AIChatRequest,
    user: Optional[dict] = Depends(_get_current_user_optional),
):
    """
    Чат-помощник по мастерингу: ответы на вопросы с учётом контекста (анализ трека, настройки).
    messages: список { "role": "user"|"assistant", "content": "..." }. context: опционально текущий анализ/настройки.
    """
    _require_feature_ai()
    tier = _get_tier_for_ai(user, request)
    ident = _get_ai_identifier(request, user)
    limit_info = ai_module.check_ai_rate_limit(ident, tier)
    if not limit_info["ok"]:
        raise HTTPException(
            429,
            f"Лимит AI-запросов исчерпан: {limit_info['limit']}/день. Сброс: {limit_info['reset_at']}.",
        )
    reply = ai_module.chat_assistant(body.messages or [], body.context)
    ai_module.record_ai_usage(ident)
    try:
        uid = user.get("sub") if user else None
        uid_int = int(uid) if uid is not None else None
        log_ai_usage("chat", uid_int, tier)
    except (TypeError, ValueError, Exception):  # noqa: BLE001
        pass
    return {"reply": reply}


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
def api_auth_register(body: RegisterRequest, request: Request, db=Depends(get_db)):
    """Зарегистрировать нового пользователя. P41: при REQUIRE_EMAIL_VERIFY=true — отправляет письмо."""
    _require_feature_registration()
    _require_auth_available()
    if not _check_auth_rate_limit(_get_client_ip(request)):
        raise HTTPException(429, "Слишком много попыток. Подождите 1 минуту.")
    if get_user_by_email(db, body.email):
        raise HTTPException(400, "Пользователь с таким email уже существует")
    hashed = get_password_hash(body.password)
    need_verify = getattr(settings, "require_email_verify", False)
    user = create_user(db, body.email, hashed, tier="pro")

    # P41: если верификация включена — помечаем пользователя как неподтверждённого
    if need_verify and DB_AVAILABLE:
        try:
            user.is_verified = False
            db.commit()
        except Exception:  # noqa: BLE001
            pass

    # Письмо верификации или приветствия
    try:
        import asyncio as _asyncio
        if need_verify:
            import secrets as _sec
            _cleanup_verify_tokens()
            vtoken = _sec.token_urlsafe(32)
            _verify_tokens[vtoken] = {"email": user.email, "exp": time.time() + _VERIFY_TOKEN_TTL}
            base = str(request.base_url).rstrip("/")
            verify_url = f"{base}/verify-email?token={vtoken}"
            from .mailer import send_email_verification
            _asyncio.get_event_loop().run_in_executor(None, send_email_verification, user.email, verify_url)
        else:
            from .mailer import send_welcome_email
            _asyncio.get_event_loop().run_in_executor(None, send_welcome_email, user.email, user.email)
    except Exception:  # noqa: BLE001
        pass

    # P51: Telegram-уведомление о новом пользователе
    try:
        from .notifier import notify_new_user
        notify_new_user(user.email, user.tier)
    except Exception:  # noqa: BLE001
        pass

    if need_verify:
        return {
            "message": "Аккаунт создан. Проверьте почту и подтвердите email для входа.",
            "email": user.email,
            "requires_verification": True,
        }

    token = create_access_token(user.id, user.email, user.tier, is_admin=bool(getattr(user, "is_admin", False)))
    return {
        "access_token": token,
        "token_type": "bearer",
        "email": user.email,
        "tier": user.tier,
        "is_admin": bool(getattr(user, "is_admin", False)),
    }


@app.post("/api/auth/login")
def api_auth_login(body: LoginRequest, request: Request, db=Depends(get_db)):
    """Войти. Возвращает JWT токен."""
    _require_auth_available()
    if not _check_auth_rate_limit(_get_client_ip(request)):
        raise HTTPException(429, "Слишком много попыток входа. Подождите 1 минуту.")
    user = get_user_by_email(db, body.email)
    if not user or not verify_password(body.password, user.hashed_password):
        raise HTTPException(401, "Неверный email или пароль")
    # P41: блокируем вход неподтверждённым если верификация включена
    if getattr(settings, "require_email_verify", False) and not getattr(user, "is_verified", True):
        raise HTTPException(
            403,
            "Email не подтверждён. Проверьте почту и перейдите по ссылке из письма."
        )
    if getattr(user, "is_blocked", False):
        raise HTTPException(403, "Аккаунт заблокирован. Обратитесь в поддержку.")
    import time as _time
    user.last_login_at = _time.time()
    db.commit()
    token = create_access_token(user.id, user.email, user.tier, is_admin=bool(getattr(user, "is_admin", False)))
    return {
        "access_token": token,
        "token_type": "bearer",
        "email": user.email,
        "tier": user.tier,
        "is_admin": bool(getattr(user, "is_admin", False)),
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
        "is_admin": bool(user.get("is_admin", False)),
    }


@app.post("/api/auth/logout")
def api_auth_logout():
    """Logout — клиент просто удаляет токен из localStorage. Endpoint для единообразия."""
    return {"message": "Токен удалён на стороне клиента"}


@app.get("/api/auth/verify-email")
def api_auth_verify_email(token: str, db=Depends(get_db)):
    """Подтвердить email по токену из письма. P41."""
    _require_auth_available()
    _cleanup_verify_tokens()
    entry = _verify_tokens.get(token)
    if not entry or entry["exp"] < time.time():
        raise HTTPException(400, "Ссылка недействительна или истекла. Запросите письмо повторно.")
    db_user = get_user_by_email(db, entry["email"])
    if not db_user:
        raise HTTPException(404, "Пользователь не найден")
    db_user.is_verified = True
    db.commit()
    _verify_tokens.pop(token, None)
    # Отправляем приветственное письмо
    try:
        from .mailer import send_welcome_email
        import asyncio as _asyncio
        _asyncio.get_event_loop().run_in_executor(None, send_welcome_email, db_user.email, db_user.email)
    except Exception:  # noqa: BLE001
        pass
    return {"message": "Email подтверждён! Теперь вы можете войти.", "email": db_user.email}


class ForgotPasswordRequest(BaseModel):
    email: str


class ResetPasswordRequest(BaseModel):
    token: str
    new_password: str

    @field_validator("new_password")
    @classmethod
    def pwd_min(cls, v: str) -> str:
        if len(v) < 6:
            raise ValueError("Пароль минимум 6 символов")
        return v


@app.post("/api/auth/resend-verification")
def api_auth_resend_verification(body: ForgotPasswordRequest, request: Request, db=Depends(get_db)):
    """Повторно отправить письмо верификации. P41."""
    _require_auth_available()
    if not _check_auth_rate_limit(_get_client_ip(request)):
        raise HTTPException(429, "Слишком много попыток. Подождите 1 минуту.")
    db_user = get_user_by_email(db, body.email.strip().lower())
    if db_user and not getattr(db_user, "is_verified", True):
        import secrets as _sec
        _cleanup_verify_tokens()
        vtoken = _sec.token_urlsafe(32)
        _verify_tokens[vtoken] = {"email": db_user.email, "exp": time.time() + _VERIFY_TOKEN_TTL}
        base = str(request.base_url).rstrip("/")
        verify_url = f"{base}/verify-email?token={vtoken}"
        try:
            from .mailer import send_email_verification
            import asyncio as _asyncio
            _asyncio.get_event_loop().run_in_executor(None, send_email_verification, db_user.email, verify_url)
        except Exception:  # noqa: BLE001
            pass
    return {"message": "Если аккаунт ожидает верификации — письмо отправлено."}


@app.get("/api/auth/profile")
def api_auth_profile(
    db=Depends(get_db),
    user: Optional[dict] = Depends(_get_current_user_optional),
):
    """Полная информация профиля: тариф, подписка, статистика. P31."""
    if not user:
        raise HTTPException(401, "Требуется авторизация")
    db_user = get_user_by_email(db, user.get("email", ""))
    if not db_user:
        raise HTTPException(404, "Пользователь не найден")
    stats = get_user_stats(db, int(user["sub"]))
    return {
        "email": db_user.email,
        "tier": db_user.tier,
        "is_admin": bool(getattr(db_user, "is_admin", False)),
        "is_blocked": bool(getattr(db_user, "is_blocked", False)),
        "subscription_status": getattr(db_user, "subscription_status", "none"),
        "subscription_expires_at": getattr(db_user, "subscription_expires_at", None),
        "created_at": db_user.created_at,
        "last_login_at": getattr(db_user, "last_login_at", None),
        "stats": stats,
    }


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


# ─── Email verification (P41) ─────────────────────────────────────────────────
# Хранилище токенов верификации: {token: {"email": str, "exp": float}}
_verify_tokens: dict[str, dict] = {}
_VERIFY_TOKEN_TTL = 86400  # 24 часа


def _cleanup_verify_tokens() -> None:
    now = time.time()
    for t in [k for k, v in _verify_tokens.items() if v["exp"] < now]:
        _verify_tokens.pop(t, None)


# ─── Password reset (P35) ─────────────────────────────────────────────────────
# Хранилище токенов: {token: {"email": str, "exp": float}}
_reset_tokens: dict[str, dict] = {}
_RESET_TOKEN_TTL = 3600  # 1 час


def _cleanup_reset_tokens() -> None:
    """Удаляет просроченные токены."""
    now = time.time()
    expired = [t for t, d in _reset_tokens.items() if d["exp"] < now]
    for t in expired:
        _reset_tokens.pop(t, None)


@app.post("/api/auth/forgot-password")
def api_auth_forgot_password(body: ForgotPasswordRequest, request: Request, db=Depends(get_db)):
    """Запросить ссылку для сброса пароля. P35.

    Всегда возвращает 200 (не раскрывает, есть ли пользователь с таким email).
    """
    _require_auth_available()
    if not _check_auth_rate_limit(_get_client_ip(request)):
        raise HTTPException(429, "Слишком много запросов. Подождите 1 минуту.")
    _cleanup_reset_tokens()
    db_user = get_user_by_email(db, body.email.strip().lower())
    if db_user:
        import secrets as _sec
        token = _sec.token_urlsafe(32)
        _reset_tokens[token] = {"email": db_user.email, "exp": time.time() + _RESET_TOKEN_TTL}
        # Определяем base URL из запроса
        base = str(request.base_url).rstrip("/")
        reset_url = f"{base}/reset-password?token={token}"
        try:
            from .mailer import send_password_reset_email
            import asyncio as _asyncio
            _asyncio.get_event_loop().run_in_executor(
                None, send_password_reset_email, db_user.email, reset_url
            )
        except Exception:  # noqa: BLE001
            pass
    return {"message": "Если аккаунт с таким email существует, письмо со ссылкой отправлено."}


@app.post("/api/auth/reset-password")
def api_auth_reset_password(body: ResetPasswordRequest, db=Depends(get_db)):
    """Сбросить пароль по токену из email. P35."""
    _require_auth_available()
    _cleanup_reset_tokens()
    entry = _reset_tokens.get(body.token)
    if not entry or entry["exp"] < time.time():
        raise HTTPException(400, "Ссылка недействительна или истекла. Запросите новую.")
    db_user = get_user_by_email(db, entry["email"])
    if not db_user:
        raise HTTPException(404, "Пользователь не найден")
    db_user.hashed_password = get_password_hash(body.new_password)
    db.commit()
    _reset_tokens.pop(body.token, None)
    return {"message": "Пароль успешно изменён. Войдите с новым паролем."}


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


@app.get("/api/auth/history/export.csv")
def api_auth_history_export_csv(
    db=Depends(get_db),
    user: Optional[dict] = Depends(_get_current_user_optional),
):
    """Скачать историю мастерингов как CSV-файл. P42."""
    from fastapi.responses import StreamingResponse as _SR
    import csv as _csv
    import io as _io
    import datetime as _dt
    _require_auth_available()
    if not user:
        raise HTTPException(401, "Требуется авторизация")
    user_id = int(user["sub"])
    records = get_user_history(db, user_id, limit=10000)

    buf = _io.StringIO()
    w = _csv.writer(buf)
    w.writerow(["id", "filename", "style", "out_format",
                "before_lufs", "after_lufs", "target_lufs", "duration_sec", "date"])
    for r in records:
        w.writerow([
            r.id, r.filename or "", r.style or "", r.out_format or "",
            f"{r.before_lufs:.2f}" if r.before_lufs is not None else "",
            f"{r.after_lufs:.2f}" if r.after_lufs is not None else "",
            f"{r.target_lufs:.2f}" if r.target_lufs is not None else "",
            f"{r.duration_sec:.1f}" if r.duration_sec is not None else "",
            _dt.datetime.fromtimestamp(r.created_at).strftime("%Y-%m-%d %H:%M") if r.created_at else "",
        ])
    content = b"\xef\xbb\xbf" + buf.getvalue().encode("utf-8")
    return _SR(
        _io.BytesIO(content),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": "attachment; filename=mastering_history.csv"},
    )


# ─── P52: API Keys ────────────────────────────────────────────────────────────

class ApiKeyCreate(BaseModel):
    name: str = "My API Key"


@app.get("/api/auth/api-keys")
def api_keys_list(
    db=Depends(get_db),
    user: Optional[dict] = Depends(_get_current_user_optional),
):
    """Список API-ключей текущего пользователя. P52."""
    _require_auth_available()
    if not user:
        raise HTTPException(401, "Требуется авторизация")
    if user.get("tier", "free") not in ("pro", "studio") and not user.get("is_admin"):
        raise HTTPException(403, "API-ключи доступны только для тарифов Pro и Studio")
    user_id = int(user["sub"])
    keys = get_api_keys_for_user(db, user_id)
    return {
        "keys": [
            {
                "id": k.id,
                "name": k.name,
                "prefix": k.key_prefix + "…",
                "is_active": k.is_active,
                "created_at": k.created_at,
                "last_used_at": k.last_used_at,
            }
            for k in keys
        ]
    }


@app.post("/api/auth/api-keys", status_code=201)
def api_keys_create(
    body: ApiKeyCreate,
    db=Depends(get_db),
    user: Optional[dict] = Depends(_get_current_user_optional),
):
    """Создать новый API-ключ. Plaintext ключ возвращается ТОЛЬКО один раз. P52."""
    _require_auth_available()
    if not user:
        raise HTTPException(401, "Требуется авторизация")
    if user.get("tier", "free") not in ("pro", "studio") and not user.get("is_admin"):
        raise HTTPException(403, "API-ключи доступны только для тарифов Pro и Studio")
    user_id = int(user["sub"])
    # Лимит: не более 10 активных ключей
    existing = [k for k in get_api_keys_for_user(db, user_id) if k.is_active]
    if len(existing) >= 10:
        raise HTTPException(400, "Достигнут лимит: не более 10 активных API-ключей")
    key_obj, raw_key = create_api_key(db, user_id, body.name.strip() or "My API Key")
    return {
        "id": key_obj.id,
        "name": key_obj.name,
        "prefix": key_obj.key_prefix + "…",
        "key": raw_key,  # показывается только один раз
        "created_at": key_obj.created_at,
        "warning": "Сохраните ключ сейчас — он больше не будет показан.",
    }


@app.delete("/api/auth/api-keys/{key_id}")
def api_keys_revoke(
    key_id: int,
    db=Depends(get_db),
    user: Optional[dict] = Depends(_get_current_user_optional),
):
    """Отозвать (деактивировать) API-ключ. P52."""
    _require_auth_available()
    if not user:
        raise HTTPException(401, "Требуется авторизация")
    user_id = int(user["sub"])
    if not revoke_api_key(db, key_id, user_id):
        raise HTTPException(404, "Ключ не найден")
    return {"revoked": True, "key_id": key_id}


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


def _load_community_presets() -> list:
    """Загружает пресеты сообщества из app/presets_community.json. P64."""
    path = Path(__file__).resolve().parent / "presets_community.json"
    if not path.is_file():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Failed to load community presets: %s", e)
        return []


@app.get("/api/presets/community")
def get_presets_community():
    """
    Пресеты сообщества (готовые цепочки + LUFS). P64.
    Источник: backend/app/presets_community.json. Формат: [{ id, name, target_lufs, style, chain_config? }].
    """
    return {"presets": _load_community_presets()}


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
    max_mb = settings_store.get_setting_int("max_upload_mb", 100)
    if len(data) > max_mb * 1024 * 1024:
        raise HTTPException(400, f"Файл больше {max_mb} МБ.")
    if not _check_audio_magic_bytes(data, file.filename or ""):
        raise HTTPException(400, "Содержимое файла не соответствует формату. Ожидается WAV, MP3 или FLAC.")
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
        out_ext = "m4a" if out_format.lower() == "aac" else out_format.lower()
        out_name = (filename or "master").rsplit(".", 1)[0] + f"_mastered.{out_ext}"
        job["status"] = "done"
        job["progress"] = 100
        job["message"] = "Готово"
        job["result_bytes"] = out_bytes
        job["filename"] = out_name
        job["done_at"] = time.time()
        try:
            log_mastering_job_end(job_id, "done")
        except Exception:  # noqa: BLE001
            pass
    except Exception as e:
        job["status"] = "error"
        job["progress"] = 0
        job["message"] = ""
        job["error"] = str(e)
        job["done_at"] = time.time()
        try:
            log_mastering_job_end(job_id, "error")
        except Exception:  # noqa: BLE001
            pass
        logger.error("mastering job failed job_id=%s filename=%s error=%s", job_id, filename, str(e)[:200])
        # P51: Telegram-уведомление об ошибке мастеринга
        try:
            from .notifier import notify_mastering_error
            notify_mastering_error(filename, str(e)[:200])
        except Exception:  # noqa: BLE001
            pass


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

        # PRO: Spectral Denoiser (до основного мастеринга). Пресет или strength + noise_percentile.
        denoise_strength = pro_params.get("denoise_strength", 0) or 0
        denoise_preset = (pro_params.get("denoise_preset") or "").strip().lower()
        if denoise_preset in DENOISE_PRESETS:
            strength, noise_pct = DENOISE_PRESETS[denoise_preset]
        elif denoise_strength > 0:
            strength = float(denoise_strength)
            noise_pct = float(pro_params.get("denoise_noise_percentile", 15.0))
        else:
            strength = 0.0
        if strength > 0.01:
            job["message"] = "Spectral Denoiser…"
            audio = apply_spectral_denoise(audio, sr, strength=strength, noise_percentile=noise_pct)

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
        out_ext = "m4a" if out_format.lower() == "aac" else out_format.lower()
        out_name = (filename or "master").rsplit(".", 1)[0] + f"_mastered.{out_ext}"
        job["status"] = "done"
        job["progress"] = 100
        job["message"] = "Готово"
        job["result_bytes"] = out_bytes
        job["filename"] = out_name
        job["done_at"] = time.time()
        try:
            log_mastering_job_end(job_id, "done")
        except Exception:  # noqa: BLE001
            pass
    except Exception as e:
        job["status"] = "error"
        job["progress"] = 0
        job["message"] = ""
        job["error"] = str(e)
        job["done_at"] = time.time()
        try:
            log_mastering_job_end(job_id, "error")
        except Exception:  # noqa: BLE001
            pass
        logger.error("mastering v2 job failed job_id=%s filename=%s error=%s", job_id, filename, str(e)[:200])


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
    if out_format.lower() not in ("wav", "mp3", "flac", "opus", "aac"):
        raise HTTPException(400, "Формат экспорта: wav, mp3, flac, opus или aac.")
    if out_format.lower() in ("mp3", "opus", "aac") and not shutil.which("ffmpeg"):
        raise HTTPException(
            400,
            f"Экспорт в {out_format.upper()} требует ffmpeg, который не найден на сервере. "
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
        target_lufs = settings_store.get_setting_float("default_target_lufs", -14.0)
    if preset and preset.lower() in PRESET_LUFS:
        target_lufs = PRESET_LUFS[preset.lower()]
    data = await file.read()
    max_mb = settings_store.get_setting_int("max_upload_mb", 100)
    if len(data) > max_mb * 1024 * 1024:
        raise HTTPException(400, f"Файл больше {max_mb} МБ.")
    if not _check_audio_magic_bytes(data, file.filename or ""):
        raise HTTPException(400, "Содержимое файла не соответствует формату. Ожидается WAV, MP3 или FLAC.")
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
        # P45: оригинал для A/B сравнения
        "original_bytes": data,
        "original_filename": file.filename or "audio.wav",
        "out_format": out_format.lower(),
    }
    try:
        log_mastering_job_start(job_id, int(user["sub"]) if user and user.get("sub") else None, style_key)
    except Exception:  # noqa: BLE001
        pass
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
    denoise_preset: Optional[str] = Form(None),
    denoise_noise_percentile: Optional[float] = Form(None),
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
    if out_format.lower() not in ("wav", "mp3", "flac", "opus", "aac"):
        raise HTTPException(400, "Формат экспорта: wav, mp3, flac, opus или aac.")
    if out_format.lower() in ("mp3", "opus", "aac") and not shutil.which("ffmpeg"):
        raise HTTPException(
            400,
            f"Экспорт в {out_format.upper()} требует ffmpeg. Установите: sudo apt-get install -y ffmpeg",
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
        target_lufs = settings_store.get_setting_float("default_target_lufs", -14.0)
    data = await file.read()
    max_mb = settings_store.get_setting_int("max_upload_mb", 100)
    if len(data) > max_mb * 1024 * 1024:
        raise HTTPException(400, f"Файл больше {max_mb} МБ.")
    if not _check_audio_magic_bytes(data, file.filename or ""):
        raise HTTPException(400, "Содержимое файла не соответствует формату. Ожидается WAV, MP3 или FLAC.")
    try:
        load_audio_from_bytes(data, file.filename or "wav")
    except Exception as e:
        logger.error("v2/master: load_audio failed filename=%s error=%s", file.filename, str(e)[:200])
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
        # P45: оригинал для A/B сравнения
        "original_bytes": data,
        "original_filename": file.filename or "audio.wav",
        "out_format": out_format.lower(),
    }
    try:
        log_mastering_job_start(job_id, int(user["sub"]) if user and user.get("sub") else None, style_key)
    except Exception:  # noqa: BLE001
        pass

    pro_params: dict = {}
    if denoise_preset and (denoise_preset.strip().lower() in ("light", "medium", "aggressive")):
        pro_params["denoise_preset"] = denoise_preset.strip().lower()
    elif denoise_strength is not None and denoise_strength > 0:
        pro_params["denoise_strength"] = float(denoise_strength)
    if denoise_noise_percentile is not None and 5 <= denoise_noise_percentile <= 40:
        pro_params["denoise_noise_percentile"] = float(denoise_noise_percentile)
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
    _require_feature_batch()
    if not files:
        raise HTTPException(400, "Отправьте хотя бы один файл.")
    if len(files) > _BATCH_MAX_FILES:
        raise HTTPException(400, f"Максимум {_BATCH_MAX_FILES} файлов за один запрос.")

    for f in files:
        if not _allowed_file(f.filename or ""):
            raise HTTPException(400, f"Формат не поддерживается: {f.filename}. Разрешены: WAV, MP3, FLAC.")
    if out_format.lower() not in ("wav", "mp3", "flac", "opus", "aac"):
        raise HTTPException(400, "Формат экспорта: wav, mp3, flac, opus или aac.")
    if out_format.lower() in ("mp3", "opus", "aac") and not shutil.which("ffmpeg"):
        raise HTTPException(400, "Экспорт в MP3/OPUS/AAC требует ffmpeg.")

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
        target_lufs = settings_store.get_setting_float("default_target_lufs", -14.0)
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
        max_mb = settings_store.get_setting_int("max_upload_mb", 100)
        if len(data) > max_mb * 1024 * 1024:
            raise HTTPException(400, f"Файл {f.filename} больше {max_mb} МБ.")
        if not _check_audio_magic_bytes(data, f.filename or ""):
            raise HTTPException(400, f"Содержимое файла {f.filename} не соответствует формату. Ожидается WAV, MP3 или FLAC.")
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
        try:
            log_mastering_job_start(job_id, int(user["sub"]) if user and user.get("sub") else None, style_key)
        except Exception:  # noqa: BLE001
            pass
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


@app.post("/api/v2/master/auto")
async def api_v2_master_auto(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    out_format: str = Form("wav"),
    user: Optional[dict] = Depends(_get_current_user_optional),
):
    """
    Авто-мастеринг: анализ трека → AI подбор пресета/настроек → мастеринг.
    Принимает file, опционально out_format. Возвращает job_id; статус и результат — как у POST /api/v2/master.
    """
    if not file.filename or not _allowed_file(file.filename):
        raise HTTPException(400, "Формат не поддерживается. Разрешены: WAV, MP3, FLAC.")
    if out_format.lower() not in ("wav", "mp3", "flac", "opus", "aac"):
        raise HTTPException(400, "Формат экспорта: wav, mp3, flac, opus или aac.")
    if out_format.lower() in ("mp3", "opus", "aac") and not shutil.which("ffmpeg"):
        raise HTTPException(400, "Экспорт в MP3/OPUS/AAC требует ffmpeg.")

    if not user and not getattr(settings, "debug_mode", False):
        ip = _get_client_ip(request)
        limit_info = _check_rate_limit(ip)
        if not limit_info["ok"]:
            raise HTTPException(
                429,
                f"Лимит Free-тарифа исчерпан: {limit_info['limit']} мастеринга в день. Сброс: {limit_info['reset_at']}.",
            )

    tier = _get_tier_for_ai(user, request)
    ident = _get_ai_identifier(request, user)
    ai_limit_info = ai_module.check_ai_rate_limit(ident, tier)
    if not ai_limit_info["ok"]:
        raise HTTPException(
            429,
            f"Лимит AI-запросов исчерпан: {ai_limit_info['limit']}/день. Сброс: {ai_limit_info['reset_at']}.",
        )

    data = await file.read()
    max_mb = settings_store.get_setting_int("max_upload_mb", 100)
    if len(data) > max_mb * 1024 * 1024:
        raise HTTPException(400, f"Файл больше {max_mb} МБ.")
    if not _check_audio_magic_bytes(data, file.filename or ""):
        raise HTTPException(400, "Содержимое файла не соответствует формату. Ожидается WAV, MP3 или FLAC.")
    fname = file.filename or "audio.wav"
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
    analysis = {
        "lufs": round(lufs, 2) if not np.isnan(lufs) else None,
        "peak_dbfs": round(peak_dbfs, 2),
        "duration_sec": round(duration_sec, 3),
        "sample_rate": sr,
        "channels": channels,
        "stereo_correlation": round(correlation, 4) if correlation is not None else None,
    }
    if audio.size >= 4096:
        try:
            analysis["spectrum_bars"] = compute_spectrum_bars(audio, sr)
        except Exception:
            pass

    rec = ai_module.recommend_preset(analysis)
    style_key = (rec.get("style") or "standard").lower()
    if style_key not in STYLE_CONFIGS:
        style_key = "standard"
    target_lufs = float(rec.get("target_lufs", -14))
    target_lufs = max(-24, min(-6, target_lufs))
    chain_config = rec.get("chain_config")

    if not user and not getattr(settings, "debug_mode", False):
        _record_usage(_get_client_ip(request))
    ai_module.record_ai_usage(ident)

    _prune_jobs()
    job_id = str(uuid.uuid4())
    _jobs[job_id] = {
        "status": "running",
        "progress": 0,
        "message": "Авто-мастеринг…",
        "created_at": time.time(),
        "result_bytes": None,
        "filename": None,
        "error": None,
        "before_lufs": None,
        "after_lufs": None,
        "target_lufs": target_lufs,
        "style": style_key,
    }
    try:
        log_mastering_job_start(job_id, int(user["sub"]) if user and user.get("sub") else None, style_key)
    except Exception:  # noqa: BLE001
        pass

    async def run_auto_job():
        async with _sem_priority if _is_priority_user(user) else _sem_normal:
            await asyncio.to_thread(
                _run_mastering_job_v2,
                job_id,
                data,
                fname,
                target_lufs,
                out_format.lower(),
                style_key,
                chain_config,
                pro_params={},
            )

    background_tasks.add_task(run_auto_job)
    return {
        "job_id": job_id,
        "status": "running",
        "version": "v2",
        "auto": True,
        "recommendation": {"style": style_key, "target_lufs": target_lufs, "reason": rec.get("reason")},
    }


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
    max_mb = settings_store.get_setting_int("max_upload_mb", 100)
    if len(data) > max_mb * 1024 * 1024:
        raise HTTPException(400, f"Файл больше {max_mb} МБ.")
    if not _check_audio_magic_bytes(data, fname):
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
    if out_format.lower() not in ("wav", "mp3", "flac", "opus", "aac"):
        raise HTTPException(400, "Формат экспорта: wav, mp3, flac, opus или aac.")
    if out_format.lower() in ("mp3", "opus", "aac") and not shutil.which("ffmpeg"):
        raise HTTPException(400, "Экспорт в MP3/OPUS/AAC требует ffmpeg.")
    strength = float(np.clip(strength, 0.0, 1.0))
    data_src = await file.read()
    data_ref = await reference.read()
    if len(data_src) > 200 * 1024 * 1024:
        raise HTTPException(400, "Основной файл больше 200 МБ.")
    if len(data_ref) > 200 * 1024 * 1024:
        raise HTTPException(400, "Эталонный файл больше 200 МБ.")
    if not _check_audio_magic_bytes(data_src, file.filename or ""):
        raise HTTPException(400, "Содержимое основного файла не соответствует формату. Ожидается WAV, MP3 или FLAC.")
    if not _check_audio_magic_bytes(data_ref, reference.filename or ""):
        raise HTTPException(400, "Содержимое эталонного файла не соответствует формату. Ожидается WAV, MP3 или FLAC.")
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
    out_ext = "m4a" if out_format.lower() == "aac" else out_format.lower()
    out_name = (file.filename or "track").rsplit(".", 1)[0] + f"_ref_matched.{out_ext}"
    media = "audio/mp4" if out_format.lower() == "aac" else f"audio/{out_format}"
    return Response(
        content=out_bytes,
        media_type=media,
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


@app.get("/api/master/progress/{job_id}")
async def api_master_progress_sse(job_id: str):
    """SSE-стрим прогресса мастеринга. P37.

    Клиент: new EventSource('/api/master/progress/<job_id>')
    Закрывается сам после status=done|error.
    Формат события: data: {"status":"running","progress":42,"message":"..."}
    """
    from fastapi.responses import StreamingResponse

    async def event_stream():
        import json as _json
        poll_interval = 0.4  # секунд между проверками
        max_wait = 600        # максимум 10 минут
        elapsed = 0.0
        last_progress = -1

        while elapsed < max_wait:
            job = _jobs.get(job_id)
            if job is None:
                payload = _json.dumps({"status": "error", "progress": 0, "message": "Задача не найдена"})
                yield f"data: {payload}\n\n"
                return

            progress = int(job.get("progress", 0))
            status = job.get("status", "running")
            message = job.get("message") or ""

            # Отправляем обновление только при изменении или финальном статусе
            if progress != last_progress or status in ("done", "error"):
                payload = _json.dumps({
                    "status": status,
                    "progress": progress,
                    "message": message,
                    "error": job.get("error"),
                    "before_lufs": _json_safe_float(job.get("before_lufs")),
                    "after_lufs": _json_safe_float(job.get("after_lufs")),
                    "target_lufs": _json_safe_float(job.get("target_lufs")),
                    "style": job.get("style", "standard"),
                })
                yield f"data: {payload}\n\n"
                last_progress = progress

                if status in ("done", "error"):
                    return

            await asyncio.sleep(poll_interval)
            elapsed += poll_interval

        # Тайм-аут
        yield f"data: {_json.dumps({'status':'error','progress':0,'message':'Тайм-аут ожидания'})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",      # Nginx: отключить буферизацию
            "Connection": "keep-alive",
        },
    )


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


@app.get("/api/master/preview/{job_id}")
async def api_master_preview(job_id: str, src: str = "mastered"):
    """
    Стриминг аудио для A/B-сравнения в браузере. P45.
    src=original — оригинал до мастеринга.
    src=mastered  — обработанный вариант (только после status=done).
    Поддерживает Range-запросы для перемотки.
    """
    if job_id not in _jobs:
        raise HTTPException(404, "Задача не найдена")
    job = _jobs[job_id]

    if src == "original":
        audio_bytes = job.get("original_bytes")
        orig_name = job.get("original_filename", "audio.wav")
        ext = Path(orig_name).suffix.lower().lstrip(".")
    else:
        if job.get("status") != "done" or not job.get("result_bytes"):
            raise HTTPException(400, "Обработанный вариант ещё не готов")
        audio_bytes = job.get("result_bytes")
        orig_name = job.get("filename", "mastered.wav")
        ext = job.get("out_format", "wav")

    if not audio_bytes:
        raise HTTPException(404, "Аудио недоступно")

    _mime_map = {"wav": "audio/wav", "mp3": "audio/mpeg", "flac": "audio/flac", "opus": "audio/ogg", "aac": "audio/mp4", "m4a": "audio/mp4"}
    media_type = _mime_map.get(ext, "audio/wav")

    return Response(
        content=audio_bytes,
        media_type=media_type,
        headers={
            "Accept-Ranges": "bytes",
            "Content-Length": str(len(audio_bytes)),
            "Cache-Control": "no-cache",
        },
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

    @app.get("/admin", include_in_schema=False)
    async def admin_page():
        """Панель администратора."""
        p = _frontend / "admin.html"
        if not p.is_file():
            return HTMLResponse("<h1>Панель администратора недоступна</h1>", status_code=404)
        return FileResponse(str(p))

    @app.get("/profile", include_in_schema=False)
    async def profile_page():
        """Страница профиля пользователя. P31."""
        p = _frontend / "profile.html"
        if not p.is_file():
            return HTMLResponse("<h1>Страница профиля недоступна</h1>", status_code=404)
        return FileResponse(str(p))

    @app.get("/429", include_in_schema=False)
    async def too_many_page():
        p = _frontend / "429.html"
        if p.is_file():
            return FileResponse(str(p))
        return HTMLResponse("<h1>429 Too Many Requests</h1>", status_code=429)

    @app.get("/sw.js", include_in_schema=False)
    async def service_worker():
        """Service Worker с правильными заголовками. P49."""
        p = _frontend / "sw.js"
        if not p.is_file():
            return HTMLResponse("/* sw not found */", media_type="application/javascript")
        from fastapi.responses import FileResponse as _FR
        return _FR(
            str(p),
            media_type="application/javascript; charset=utf-8",
            headers={"Service-Worker-Allowed": "/", "Cache-Control": "no-cache"},
        )

    @app.get("/manifest.json", include_in_schema=False)
    async def web_manifest():
        """Web App Manifest. P49."""
        p = _frontend / "manifest.json"
        if not p.is_file():
            return HTMLResponse("{}", media_type="application/manifest+json")
        from fastapi.responses import FileResponse as _FR
        return _FR(str(p), media_type="application/manifest+json",
                   headers={"Cache-Control": "max-age=86400"})

    @app.get("/status", include_in_schema=False)
    async def status_page():
        """Публичная страница статуса сервиса. P47."""
        p = _frontend / "status.html"
        if not p.is_file():
            return HTMLResponse("<h1>Status page not found</h1>", status_code=404)
        return FileResponse(str(p))

    @app.get("/verify-email", include_in_schema=False)
    async def verify_email_page():
        """Страница подтверждения email. P41."""
        p = _frontend / "verify-email.html"
        if not p.is_file():
            return HTMLResponse("<h1>Страница недоступна</h1>", status_code=404)
        return FileResponse(str(p))

    @app.get("/forgot-password", include_in_schema=False)
    async def forgot_password_page():
        """Форма запроса сброса пароля. P35."""
        p = _frontend / "forgot-password.html"
        if not p.is_file():
            return HTMLResponse("<h1>Страница недоступна</h1>", status_code=404)
        return FileResponse(str(p))

    @app.get("/reset-password", include_in_schema=False)
    async def reset_password_page():
        """Форма ввода нового пароля по токену. P35."""
        p = _frontend / "reset-password.html"
        if not p.is_file():
            return HTMLResponse("<h1>Страница недоступна</h1>", status_code=404)
        return FileResponse(str(p))

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

    # PWA: иконки из manifest (если frontend/icons/ отсутствуют — отдаём минимальный PNG, чтобы не было 404)
    @app.get("/icons/icon-192.png", response_class=Response)
    def _icon_192():
        return Response(content=_PNG_1X1, media_type="image/png")

    @app.get("/icons/icon-512.png", response_class=Response)
    def _icon_512():
        return Response(content=_PNG_1X1, media_type="image/png")

    app.mount("/", StaticFiles(directory=str(_frontend), html=True), name="frontend")
