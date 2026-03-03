"""Сервис генерации аналитических отчётов для admin-панели.

Вся логика построения отчётов вынесена сюда. Роутер (admin.py) только
вызывает run_report() и отдаёт результат клиенту.
"""
from __future__ import annotations

import time
from datetime import datetime
from typing import Optional

from ..database import (
    AiUsageLog,
    DB_AVAILABLE,
    MasteringJobEvent,
    MasteringRecord,
    PromptTemplate,
    Transaction,
    User,
    get_active_prompt_body,
)


REPORTS_META = [
    {
        "id": "registrations_by_day",
        "name": "Регистрации по дням",
        "description": "Число новых пользователей по дате",
        "params": ["date_from", "date_to"],
    },
    {
        "id": "tier_distribution",
        "name": "Распределение по тарифам",
        "description": "Количество пользователей по tier",
        "params": [],
    },
    {
        "id": "revenue_by_period",
        "name": "Выручка за период",
        "description": "Сумма успешных транзакций по дням",
        "params": ["date_from", "date_to"],
    },
    {
        "id": "masterings_by_day",
        "name": "Мастеринги по дням",
        "description": "Число завершённых задач по дате",
        "params": ["date_from", "date_to"],
    },
    {
        "id": "avg_lufs_by_style",
        "name": "Средний LUFS по стилю",
        "description": "До/после по стилю мастеринга",
        "params": [],
    },
    {
        "id": "ai_usage_by_type",
        "name": "Использование AI по типу и тарифу",
        "description": "Запросы AI по типу (recommend, report, chat и т.д.)",
        "params": ["date_from", "date_to"],
    },
    {
        "id": "errors_failures",
        "name": "Ошибки мастеринга",
        "description": "Задачи со статусом error",
        "params": ["date_from", "date_to"],
    },
    {
        "id": "popular_styles",
        "name": "Популярные стили",
        "description": "Топ стилей по количеству использований",
        "params": [],
    },
    {
        "id": "user_activity",
        "name": "Активность пользователей",
        "description": "Уникальные пользователи по дням (мастеринг/AI)",
        "params": ["date_from", "date_to"],
    },
    {
        "id": "export_raw",
        "name": "Экспорт для аналитики",
        "description": "Сырые данные для внешнего BI (CSV)",
        "params": ["date_from", "date_to"],
    },
    {
        "id": "prompt_recommendations",
        "name": "Рекомендации по промптам",
        "description": "Использование AI по типам и активные промпты (recommend, report, nl_config, chat). Для советов нажмите «Резюме LLM».",
        "params": ["date_from", "date_to"],
    },
]


def parse_dates(date_from: Optional[str], date_to: Optional[str]):
    """Вернуть (ts_from, ts_to) в секундах или (None, None)."""
    def _parse(s: Optional[str]) -> Optional[float]:
        if not s:
            return None
        try:
            dt = datetime.strptime(s[:10], "%Y-%m-%d")
            return time.mktime(dt.timetuple())
        except Exception:
            return None
    return _parse(date_from), _parse(date_to)


def run_report(
    db,
    report_id: str,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
) -> dict:
    """Выполнить отчёт по id. Возвращает dict с полями отчёта."""
    ts_from, ts_to = parse_dates(date_from, date_to)

    if report_id == "registrations_by_day":
        q = db.query(User).filter(User.created_at != None)
        if ts_from:
            q = q.filter(User.created_at >= ts_from)
        if ts_to:
            q = q.filter(User.created_at <= ts_to + 86400)
        users = q.all()
        by_day: dict = {}
        for u in users:
            day = datetime.fromtimestamp(u.created_at).strftime("%Y-%m-%d")
            by_day[day] = by_day.get(day, 0) + 1
        rows = [{"date": k, "count": v} for k, v in sorted(by_day.items())]
        return {"rows": rows, "total": len(users)}

    if report_id == "tier_distribution":
        q = db.query(User.tier).distinct()
        tiers = [r[0] for r in q.all()]
        out = []
        for t in tiers or ["free", "pro", "studio"]:
            cnt = db.query(User).filter(User.tier == t).count()
            out.append({"tier": t, "count": cnt})
        return {"rows": out}

    if report_id == "revenue_by_period":
        q = db.query(Transaction).filter(Transaction.status == "succeeded")
        if ts_from:
            q = q.filter(Transaction.created_at >= ts_from)
        if ts_to:
            q = q.filter(Transaction.created_at <= ts_to + 86400)
        txs = q.all()
        by_day: dict = {}
        for t in txs:
            day = datetime.fromtimestamp(t.created_at).strftime("%Y-%m-%d")
            by_day[day] = by_day.get(day, 0) + t.amount
        rows = [{"date": k, "amount": round(v, 2)} for k, v in sorted(by_day.items())]
        return {"rows": rows, "total_amount": round(sum(t.amount for t in txs), 2)}

    if report_id == "masterings_by_day":
        by_day: dict = {}
        if DB_AVAILABLE and MasteringJobEvent is not None:
            q = db.query(MasteringJobEvent).filter(
                MasteringJobEvent.status == "done",
                MasteringJobEvent.completed_at != None,
            )
            if ts_from:
                q = q.filter(MasteringJobEvent.completed_at >= ts_from)
            if ts_to:
                q = q.filter(MasteringJobEvent.completed_at <= ts_to + 86400)
            for r in q.all():
                day = datetime.fromtimestamp(r.completed_at).strftime("%Y-%m-%d")
                by_day[day] = by_day.get(day, 0) + 1
        q = db.query(MasteringRecord).filter(MasteringRecord.created_at != None)
        if ts_from:
            q = q.filter(MasteringRecord.created_at >= ts_from)
        if ts_to:
            q = q.filter(MasteringRecord.created_at <= ts_to + 86400)
        for r in q.all():
            day = datetime.fromtimestamp(r.created_at).strftime("%Y-%m-%d")
            by_day[day] = by_day.get(day, 0) + 1
        rows = [{"date": k, "count": v} for k, v in sorted(by_day.items())]
        return {"rows": rows}

    if report_id == "avg_lufs_by_style":
        q = db.query(MasteringRecord).filter(
            MasteringRecord.before_lufs != None,
            MasteringRecord.after_lufs != None,
        )
        recs = q.all()
        by_style: dict = {}
        for r in recs:
            s = r.style or "standard"
            if s not in by_style:
                by_style[s] = {"before": [], "after": [], "target": []}
            by_style[s]["before"].append(r.before_lufs)
            by_style[s]["after"].append(r.after_lufs)
            if r.target_lufs is not None:
                by_style[s]["target"].append(r.target_lufs)
        rows = []
        for style, data in by_style.items():
            rows.append({
                "style": style,
                "avg_before_lufs": round(sum(data["before"]) / len(data["before"]), 2) if data["before"] else None,
                "avg_after_lufs": round(sum(data["after"]) / len(data["after"]), 2) if data["after"] else None,
                "count": len(data["before"]),
            })
        return {"rows": sorted(rows, key=lambda x: -x["count"])}

    if report_id == "ai_usage_by_type":
        if not DB_AVAILABLE or AiUsageLog is None:
            return {"rows": [], "message": "Таблица ai_usage_log недоступна"}
        q = db.query(AiUsageLog).filter(AiUsageLog.type != None)
        if ts_from:
            q = q.filter(AiUsageLog.created_at >= ts_from)
        if ts_to:
            q = q.filter(AiUsageLog.created_at <= ts_to + 86400)
        logs = q.all()
        by_type_tier: dict = {}
        for L in logs:
            key = (L.type or "unknown", L.tier or "free")
            by_type_tier[key] = by_type_tier.get(key, 0) + 1
        rows = [
            {"type": k[0], "tier": k[1], "count": v}
            for k, v in sorted(by_type_tier.items(), key=lambda x: -x[1])
        ]
        return {"rows": rows, "total": len(logs)}

    if report_id == "errors_failures":
        if not DB_AVAILABLE or MasteringJobEvent is None:
            return {"rows": [], "message": "Таблица mastering_job_events недоступна"}
        q = db.query(MasteringJobEvent).filter(MasteringJobEvent.status == "error")
        if ts_from:
            q = q.filter(MasteringJobEvent.completed_at >= ts_from)
        if ts_to:
            q = q.filter(MasteringJobEvent.completed_at <= ts_to + 86400)
        recs = q.all()
        rows = [{"job_id": r.job_id, "completed_at": r.completed_at, "style": r.style} for r in recs[:500]]
        return {"rows": rows, "total": len(recs)}

    if report_id == "popular_styles":
        q = db.query(MasteringRecord.style).filter(MasteringRecord.style != None)
        recs = q.all()
        cnt: dict = {}
        for r in recs:
            s = r[0] or "standard"
            cnt[s] = cnt.get(s, 0) + 1
        rows = [{"style": k, "count": v} for k, v in sorted(cnt.items(), key=lambda x: -x[1])]
        return {"rows": rows}

    if report_id == "user_activity":
        by_day: dict = {}
        if DB_AVAILABLE and AiUsageLog is not None:
            q = db.query(AiUsageLog.created_at, AiUsageLog.user_id).filter(
                AiUsageLog.created_at != None
            )
            if ts_from:
                q = q.filter(AiUsageLog.created_at >= ts_from)
            if ts_to:
                q = q.filter(AiUsageLog.created_at <= ts_to + 86400)
            for r in q.all():
                if ts_from and r.created_at < ts_from:
                    continue
                if ts_to and r.created_at > ts_to + 86400:
                    continue
                day = datetime.fromtimestamp(r.created_at).strftime("%Y-%m-%d")
                if day not in by_day:
                    by_day[day] = set()
                by_day[day].add(r.user_id)
        q = db.query(MasteringRecord.created_at, MasteringRecord.user_id).filter(
            MasteringRecord.created_at != None
        )
        if ts_from:
            q = q.filter(MasteringRecord.created_at >= ts_from)
        if ts_to:
            q = q.filter(MasteringRecord.created_at <= ts_to + 86400)
        for r in q.all():
            day = datetime.fromtimestamp(r.created_at).strftime("%Y-%m-%d")
            if day not in by_day:
                by_day[day] = set()
            by_day[day].add(r.user_id)
        rows = [{"date": k, "unique_users": len(v)} for k, v in sorted(by_day.items())]
        return {"rows": rows}

    if report_id == "export_raw":
        return {
            "message": "Используйте GET /api/admin/reports/export_raw.csv?date_from=&date_to= для выгрузки CSV"
        }

    if report_id == "prompt_recommendations":
        slug_names = {"recommend": "Рекомендатор пресета", "report": "AI отчёт", "nl_config": "Настройки голосом", "chat": "Чат"}
        rows = []
        if not DB_AVAILABLE:
            return {"rows": [], "message": "БД недоступна"}
        if AiUsageLog is None or PromptTemplate is None:
            return {"rows": [], "message": "Таблицы ai_usage_log или prompt_templates недоступны"}
        for slug in ("recommend", "report", "nl_config", "chat"):
            q = db.query(AiUsageLog).filter(AiUsageLog.type == slug)
            if ts_from:
                q = q.filter(AiUsageLog.created_at >= ts_from)
            if ts_to:
                q = q.filter(AiUsageLog.created_at <= ts_to + 86400)
            usage_count = q.count()
            active_body = get_active_prompt_body(db, slug)
            preview = (active_body or "")[:200] + ("…" if len(active_body or "") > 200 else "")
            rows.append({
                "slug": slug,
                "name": slug_names.get(slug, slug),
                "usage_count": usage_count,
                "active_preview": preview,
            })
        return {"rows": rows}

    return {"error": "Неизвестный отчёт"}
