# @file main.py
# @description FastAPI-приложение: загрузка, мастеринг, экспорт, замер LUFS
# @dependencies app.pipeline, app.chain, app.config
# @created 2026-02-26

import base64
import datetime
import json
import logging
import os
import struct
import time
import zlib

logger = logging.getLogger(__name__)

# Минимальный 1×1 PNG (прозрачный) для PWA-иконок, если файлы отсутствуют
_PNG_1X1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)


def _make_png_placeholder(size: int, r: int = 0x6c, g: int = 0x4b, b: int = 0xff) -> bytes:
    """Генерирует PNG размером size×size (однотонный) для PWA — без зависимости от Pillow."""
    def png_chunk(chunk_type: bytes, data: bytes) -> bytes:
        chunk = chunk_type + data
        crc = zlib.crc32(chunk) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + chunk + struct.pack(">I", crc)
    signature = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", size, size, 8, 6, 0, 0, 0)  # 8bit RGBA
    raw = b""
    for _ in range(size):
        raw += b"\x00"  # filter
        raw += (struct.pack(">BBBB", r, g, b, 255) * size)
    idat_data = zlib.compress(raw, 9)
    return signature + png_chunk(b"IHDR", ihdr) + png_chunk(b"IDAT", idat_data) + png_chunk(b"IEND", b"")


# PWA-иконки правильного размера (manifest требует 192x192 и 512x512)
_PNG_192 = _make_png_placeholder(192)
_PNG_512 = _make_png_placeholder(512)

from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles

from .auth import AUTH_AVAILABLE
from .config import settings
from . import settings_store
from .database import DB_AVAILABLE, create_tables
from .version import __version__, __build_date__
from .helpers import get_client_ip as _get_client_ip
from .admin import router as admin_router
from .payments import router as payments_router
from . import jobs_store as _jobs_store
from .routers.misc import router as misc_router
from .routers.ai_router import router as ai_router_module
from .routers.auth import router as auth_router
from .routers.mastering import router as mastering_router

# Алиас для health/metrics (read-only view на jobs)
_jobs = _jobs_store.all_jobs()


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

# Восстановление задач из SQLite после рестарта
_restored = _jobs_store.restore_from_db()
if _restored:
    logger.info("jobs_store: восстановлено %d задач из SQLite", _restored)

# P51: Telegram-уведомление о запуске сервера (асинхронно)
try:
    from .notifier import notify_server_startup
    _host = f"{os.environ.get('MAGIC_MASTER_HOST', '0.0.0.0')}:{os.environ.get('MAGIC_MASTER_PORT', '8000')}"
    notify_server_startup(__version__, _host)
except Exception:  # noqa: BLE001
    pass


# ─── Admin router (P19) ──────────────────────────────────────────────────────
app.include_router(admin_router)

# ─── Payments router (P23) ───────────────────────────────────────────────────
app.include_router(payments_router)

# ─── Вынесенные роутеры ──────────────────────────────────────────────────────
app.include_router(misc_router)
app.include_router(ai_router_module)
app.include_router(auth_router)
app.include_router(mastering_router)


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
    headers = getattr(exc, "headers", None) or {}
    retry = headers.get("Retry-After", "60")
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

    # Алерт мониторинга при деградации (очередь 3.3)
    _alert_enabled = settings_store.get_setting("alert_monitoring_enabled")
    if overall != "ok" and (_alert_enabled is True or (isinstance(_alert_enabled, str) and _alert_enabled.strip().lower() in ("1", "true", "yes", "on")) or getattr(settings, "alert_monitoring_enabled", False)):
        try:
            from .notifier import notify_alert_health_degraded
            reasons = []
            if db_status != "ok":
                reasons.append(f"БД: {db_status}")
            if disk_status == "low":
                reasons.append(f"диск: {disk_status} (свободно {disk_free_mb} МБ)")
            notify_alert_health_degraded("; ".join(reasons), db_detail or None)
        except Exception:
            pass

    # Флаги функций (из админки): по умолчанию True, если не заданы
    feature_ai = settings_store.get_setting("feature_ai_enabled")
    feature_batch = settings_store.get_setting("feature_batch_enabled")
    feature_registration = settings_store.get_setting("feature_registration_enabled")
    features = {
        "ai_enabled": feature_ai is None or feature_ai is True or (isinstance(feature_ai, str) and feature_ai.lower() in ("true", "1", "yes")),
        "batch_enabled": feature_batch is None or feature_batch is True or (isinstance(feature_batch, str) and feature_batch.lower() in ("true", "1", "yes")),
        "registration_enabled": feature_registration is None or feature_registration is True or (isinstance(feature_registration, str) and feature_registration.lower() in ("true", "1", "yes")),
        "vocal_isolation_enabled": getattr(settings, "enable_vocal_isolation", False),
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
    # Алерт при превышении порога очереди (очередь 3.3)
    _alert_en = settings_store.get_setting("alert_monitoring_enabled")
    if _alert_en is True or (isinstance(_alert_en, str) and _alert_en and _alert_en.strip().lower() in ("1", "true", "yes", "on")) or getattr(settings, "alert_monitoring_enabled", False):
        try:
            from .notifier import notify_alert_queue_threshold
            notify_alert_queue_threshold(jobs_total, jobs_running)
        except Exception:
            pass
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


# Раздача фронтенда — после всех API-маршрутов, иначе POST /api/measure отдаёт 405 Method Not Allowed
_frontend = Path(__file__).resolve().parent.parent.parent / "frontend"
_PROGRESS_PATH = Path(__file__).resolve().parent.parent.parent / "PROGRESS.md"

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

    @app.get("/robots.txt", include_in_schema=False)
    async def robots_txt():
        p = _frontend / "robots.txt"
        if p.is_file():
            return FileResponse(str(p), media_type="text/plain")
        return Response("User-agent: *\nAllow: /\n", media_type="text/plain")

    @app.get("/sitemap.xml", include_in_schema=False)
    async def sitemap_xml():
        p = _frontend / "sitemap.xml"
        if p.is_file():
            return FileResponse(str(p), media_type="application/xml")
        return Response(status_code=404)

    @app.get("/favicon.ico", include_in_schema=False)
    async def favicon():
        p = _frontend / "favicon.ico"
        if p.is_file():
            return FileResponse(str(p), media_type="image/x-icon")
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

    # PWA: иконки правильного размера (manifest требует 192x192 и 512x512 — иначе предупреждение)
    @app.get("/icons/icon-192.png", response_class=Response)
    def _icon_192():
        return Response(content=_PNG_192, media_type="image/png")

    @app.get("/icons/icon-512.png", response_class=Response)
    def _icon_512():
        return Response(content=_PNG_512, media_type="image/png")

    app.mount("/", StaticFiles(directory=str(_frontend), html=True), name="frontend")
