"""Проверки квот и вспомогательные функции для бота."""
from __future__ import annotations

from typing import Optional, Tuple

from ..config import settings
from ..database import (
    DB_AVAILABLE,
    User,
    count_mastering_jobs_today,
    deduct_tokens,
    get_user_by_telegram_id,
    get_user_tokens_balance,
)
from ..deps import check_rate_limit, record_usage


def public_url() -> str:
    return (getattr(settings, "public_base_url", "") or "").strip().rstrip("/") or "https://magicmaster.pro"


def get_linked_user(db, telegram_id: int) -> Optional[User]:
    if not DB_AVAILABLE or db is None or telegram_id is None:
        return None
    return get_user_by_telegram_id(db, int(telegram_id))


def precheck_mastering(db, user: User) -> Optional[str]:
    """None если можно запускать; иначе код ошибки для texts."""
    if getattr(user, "is_blocked", False):
        return "blocked"
    tier = (user.tier or "free").lower()
    tid = int(user.telegram_id) if user.telegram_id else 0
    if tier in ("pro", "studio"):
        if get_user_tokens_balance(db, user.id) < 1:
            return "no_tokens"
        cap = 30 if tier == "studio" else 10
        if count_mastering_jobs_today(db, user.id) >= cap:
            return "daily_cap"
        if not deduct_tokens(db, user.id, 1):
            return "no_tokens"
        return None
    info = check_rate_limit(f"tg:{tid}")
    if not info["ok"]:
        return "free_limit"
    record_usage(f"tg:{tid}")
    return None


def refund_token_if_pro(db, user: User) -> None:
    """Если Pro/Studio — вернуть 1 токен при ошибке мастеринга."""
    from ..database import add_tokens

    tier = (user.tier or "free").lower()
    if tier in ("pro", "studio") and user:
        add_tokens(db, user.id, 1)
