# @file deps.py
# @description Общие FastAPI-зависимости и вспомогательные функции для всех роутеров.
#   Выделено из main.py для устранения дублирования при разбивке на роутеры.
# @created 2026-03-01

import datetime
import time
from typing import Optional

from fastapi import Depends, Header, HTTPException, Request

from .auth import AUTH_AVAILABLE, decode_token, extract_bearer_token
from .config import settings
from .database import DB_AVAILABLE, get_db, get_user_by_api_key
from .helpers import get_client_ip
from . import settings_store

# ─── Auth dependency ──────────────────────────────────────────────────────────

def get_current_user_optional(
    authorization: Optional[str] = Header(None),
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
) -> Optional[dict]:
    """Dependency: декодирует Bearer JWT или X-API-Key, возвращает payload или None.

    P28: если подписка истекла — понижает tier до 'free'.
    P52: поддержка X-API-Key заголовка для Pro/Studio пользователей.
    """
    from .database import check_and_expire_subscription
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
        except Exception:  # noqa: BLE001
            pass
        return None

    token = extract_bearer_token(authorization)
    if not token:
        return None
    payload = decode_token(token)
    if payload is None:
        return None
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
        except Exception:  # noqa: BLE001
            pass
    return payload


# ─── Rate limiting (Free tier: 3 мастеринга / день по IP) ────────────────────
_FREE_DAILY_LIMIT = 3
_rate_limits: dict[str, dict] = {}


def check_rate_limit(ip: str) -> dict:
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


def record_usage(ip: str, n: int = 1) -> None:
    """Увеличивает счётчик использований для IP за сегодня."""
    today = datetime.date.today().isoformat()
    entry = _rate_limits.get(ip)
    if not entry or entry.get("day") != today:
        _rate_limits[ip] = {"count": n, "day": today}
    else:
        _rate_limits[ip]["count"] = entry["count"] + n


# ─── Auth brute-force protection (P33) ───────────────────────────────────────
_AUTH_LIMIT_PER_MINUTE = 10
_auth_attempts: dict[str, dict] = {}


def check_auth_rate_limit(ip: str) -> bool:
    """True = разрешено, False = заблокировано (слишком много попыток)."""
    now = time.time()
    entry = _auth_attempts.get(ip)
    if entry is None or now - entry["window_start"] >= 60:
        _auth_attempts[ip] = {"count": 1, "window_start": now}
        return True
    entry["count"] += 1
    return entry["count"] <= _AUTH_LIMIT_PER_MINUTE


# ─── AI rate limiting helpers ─────────────────────────────────────────────────

def get_ai_identifier(request: Request, user: Optional[dict]) -> str:
    """Ключ для учёта AI-запросов: user_id если авторизован, иначе IP."""
    if user and user.get("sub"):
        return f"user:{user['sub']}"
    return f"ip:{get_client_ip(request)}"


def get_tier_for_ai(user: Optional[dict], request: Request) -> str:
    """Тариф для лимитов AI: debug/pro/studio без лимита, иначе free."""
    if getattr(settings, "debug_mode", False):
        return "studio"
    if user:
        return (user.get("tier") or "pro").lower()
    return "free"


# ─── Feature flags ────────────────────────────────────────────────────────────

def require_feature_ai() -> None:
    """503 если AI отключён в настройках."""
    if not settings_store.get_setting_bool("feature_ai_enabled", True):
        raise HTTPException(503, "Функции AI временно отключены администратором.")


def require_feature_registration() -> None:
    """503 если регистрация отключена."""
    if not settings_store.get_setting_bool("feature_registration_enabled", True):
        raise HTTPException(503, "Регистрация новых пользователей отключена.")


def require_feature_batch() -> None:
    """503 если пакетная обработка отключена."""
    if not settings_store.get_setting_bool("feature_batch_enabled", True):
        raise HTTPException(503, "Пакетный мастеринг временно отключён.")


def require_auth_available() -> None:
    """503 если auth/DB недоступны."""
    if not DB_AVAILABLE or not AUTH_AVAILABLE:
        raise HTTPException(
            503,
            "Авторизация недоступна: установите пакеты sqlalchemy, passlib, python-jose[cryptography]"
        )


def is_priority_user(user: Optional[dict]) -> bool:
    """Pro/Studio или режим отладки — доступ к приоритетным слотам."""
    if getattr(settings, "debug_mode", False):
        return True
    if not user:
        return False
    return (user.get("tier") or "").lower() in ("pro", "studio")
