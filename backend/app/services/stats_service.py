"""Сервис статистики для admin dashboard.

Агрегирует метрики: пользователи, транзакции, мастеринги, новости.
"""
from __future__ import annotations

import datetime as _dt
import time
from typing import Optional

from ..database import (
    AiUsageLog,
    MasteringJobEvent,
    MasteringRecord,
    NewsPost,
    Transaction,
    User,
    DB_AVAILABLE,
)


def get_dashboard_stats(db) -> dict:
    """Метрики для Dashboard: пользователи, транзакции, мастеринги. P43: расширенная аналитика."""
    users = db.query(User).all()
    tier_counts: dict = {}
    for u in users:
        tier_counts[u.tier] = tier_counts.get(u.tier, 0) + 1

    total_users = len(users)
    blocked_count = sum(1 for u in users if getattr(u, "is_blocked", False))
    unverified_count = sum(1 for u in users if not getattr(u, "is_verified", True))

    now = time.time()
    month_start = now - 30 * 86400
    week_start = now - 7 * 86400
    new_users_month = sum(1 for u in users if u.created_at and u.created_at >= month_start)
    new_users_week = sum(1 for u in users if u.created_at and u.created_at >= week_start)

    today_date = _dt.date.today()
    new_users_by_day = []
    for i in range(6, -1, -1):
        day = today_date - _dt.timedelta(days=i)
        day_ts_start = _dt.datetime.combine(day, _dt.time.min).timestamp()
        day_ts_end = _dt.datetime.combine(day, _dt.time.max).timestamp()
        count = sum(1 for u in users if u.created_at and day_ts_start <= u.created_at <= day_ts_end)
        new_users_by_day.append({"date": day.isoformat(), "count": count})

    all_txs = db.query(Transaction).filter(Transaction.status == "succeeded").all()
    revenue_total = round(sum(getattr(t, "amount", 0) or 0 for t in all_txs), 2)
    txs_month = [t for t in all_txs if t.created_at and t.created_at >= month_start]
    revenue_month = round(sum(getattr(t, "amount", 0) or 0 for t in txs_month), 2)
    tx_count_month = len(txs_month)

    revenue_by_day = []
    for i in range(6, -1, -1):
        day = today_date - _dt.timedelta(days=i)
        day_ts_start = _dt.datetime.combine(day, _dt.time.min).timestamp()
        day_ts_end = _dt.datetime.combine(day, _dt.time.max).timestamp()
        amount = sum(
            getattr(t, "amount", 0) or 0
            for t in all_txs
            if t.created_at and day_ts_start <= t.created_at <= day_ts_end
        )
        revenue_by_day.append({"date": day.isoformat(), "amount": round(amount, 2)})

    day_start = now - 86400
    masters_today = db.query(MasteringRecord).filter(MasteringRecord.created_at >= day_start).count()
    masters_month = db.query(MasteringRecord).filter(MasteringRecord.created_at >= month_start).count()
    masters_total = db.query(MasteringRecord).count()

    masters_by_day = []
    for i in range(6, -1, -1):
        day = today_date - _dt.timedelta(days=i)
        day_ts_start = _dt.datetime.combine(day, _dt.time.min).timestamp()
        day_ts_end = _dt.datetime.combine(day, _dt.time.max).timestamp()
        count = db.query(MasteringRecord).filter(
            MasteringRecord.created_at >= day_ts_start,
            MasteringRecord.created_at <= day_ts_end,
        ).count()
        masters_by_day.append({"date": day.isoformat(), "count": count})

    news_total = db.query(NewsPost).count()
    news_published = db.query(NewsPost).filter(NewsPost.is_published == True).count()  # noqa: E712

    active_subs = sum(
        1 for u in users
        if getattr(u, "subscription_status", "none") == "active"
        and getattr(u, "subscription_expires_at", None)
        and u.subscription_expires_at > now
    )

    return {
        "users": {
            "total": total_users,
            "blocked": blocked_count,
            "unverified": unverified_count,
            "by_tier": tier_counts,
            "new_month": new_users_month,
            "new_week": new_users_week,
            "by_day": new_users_by_day,
            "active_subscriptions": active_subs,
        },
        "revenue": {
            "total_rub": revenue_total,
            "month_rub": revenue_month,
            "transactions_month": tx_count_month,
            "by_day": revenue_by_day,
        },
        "masterings": {
            "today": masters_today,
            "month": masters_month,
            "total": masters_total,
            "by_day": masters_by_day,
        },
        "news": {
            "total": news_total,
            "published": news_published,
        },
    }
