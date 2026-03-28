"""Welcome-серия и еженедельные DM (cron: python -m app.bot.run_engagement)."""
from __future__ import annotations

import time

from ..database import (
    DB_AVAILABLE,
    MasteringRecord,
    SessionLocal,
    TelegramEngagement,
    get_user_by_telegram_id,
)
from .helpers import public_url
from .notify_user import send_user_bot_text_sync


def _now() -> float:
    return time.time()


def process_engagement_due() -> dict:
    """Welcome 1/3/7 дней и еженедельный DM с токенами."""
    out = {"welcome_1": 0, "welcome_3": 0, "welcome_7": 0, "weekly": 0}
    if not DB_AVAILABLE or SessionLocal is None:
        return out
    db = SessionLocal()
    try:
        now = _now()
        week_sec = 7 * 86400
        base_url = public_url()
        for row in db.query(TelegramEngagement).all():
            tid = int(row.telegram_id)
            u = get_user_by_telegram_id(db, tid)
            lang = ((getattr(u, "telegram_lang", None) or "ru")[:2] if u else "ru")
            lang = "en" if lang == "en" else "ru"

            if row.welcome_day_1_sent_at is None and now - row.first_start_at >= 86400:
                msg = (
                    "👋 Напоминание: попробуйте мастеринг — /master"
                    if lang != "en"
                    else "👋 Try mastering — /master"
                )
                if send_user_bot_text_sync(tid, msg):
                    row.welcome_day_1_sent_at = now
                    out["welcome_1"] += 1

            if row.welcome_day_3_sent_at is None and now - row.first_start_at >= 3 * 86400:
                msg = (
                    "💡 Совет: после генерации музыки LLM мастеринг выравнивает громкость."
                    if lang != "en"
                    else "💡 After LLM music, mastering balances loudness."
                )
                if send_user_bot_text_sync(tid, msg):
                    row.welcome_day_3_sent_at = now
                    out["welcome_3"] += 1

            if row.welcome_day_7_sent_at is None and now - row.first_start_at >= 7 * 86400:
                msg = "🎵 Ваш трек ждёт — /master" if lang != "en" else "🎵 Your track is waiting — /master"
                if send_user_bot_text_sync(tid, msg):
                    row.welcome_day_7_sent_at = now
                    out["welcome_7"] += 1

            if u and (row.last_weekly_digest_at is None or now - row.last_weekly_digest_at >= week_sec):
                n_m = (
                    db.query(MasteringRecord)
                    .filter(
                        MasteringRecord.user_id == u.id,
                        MasteringRecord.created_at >= now - week_sec,
                    )
                    .count()
                )
                tok = int(getattr(u, "tokens_balance", 0) or 0)
                msg = (
                    f"📈 За неделю мастерингов: {n_m}. Токены: {tok}. {base_url}/app"
                    if lang != "en"
                    else f"📈 Masterings this week: {n_m}. Tokens: {tok}. {base_url}/app"
                )
                if send_user_bot_text_sync(tid, msg):
                    row.last_weekly_digest_at = now
                    out["weekly"] += 1

        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()
    return out


def post_channel_weekly_stats() -> bool:
    """Один пост в канал со сводкой (вызывать из cron раз в неделю)."""
    if not DB_AVAILABLE or SessionLocal is None:
        return False
    from ..services.stats_service import get_dashboard_stats
    from .channel import post_to_channel_html

    db = SessionLocal()
    try:
        st = get_dashboard_stats(db)
        m = st.get("masterings", {}).get("month", 0)
        u = st.get("users", {}).get("total", 0)
    finally:
        db.close()
    line = f"📊 Magic Master: пользователей {u}, мастерингов за месяц {m}. {public_url()}"
    return post_to_channel_html(line)
