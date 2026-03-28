"""Форматирование админ-отчётов для Telegram (русский язык)."""
from __future__ import annotations

import time
from typing import Any, Optional

from sqlalchemy import text as sql_text

from .. import jobs_store as js
from .. import settings_store
from ..config import settings
from ..database import MasteringJobEvent, User
from .server_metrics import format_server_oneline, format_server_report, get_system_metrics


def _sem_in_use(sem: Any, default_n: int) -> tuple[int, int]:
    """(занято слотов, всего слотов) для asyncio.Semaphore."""
    try:
        n = int(default_n)
        rem = int(getattr(sem, "_value", n))
        return max(0, n - rem), n
    except (TypeError, ValueError):
        return 0, int(default_n)


def format_stats_ru(st: dict) -> str:
    u = st.get("users", {})
    m = st.get("masterings", {})
    lines = [
        "📊 <b>Статистика</b>",
        "",
        "<b>Пользователи</b>",
        f"Всего: <b>{u.get('total', 0)}</b>",
        f"Новых за неделю: <b>{u.get('new_week', 0)}</b>",
        f"Новых за 30 дней: <b>{u.get('new_month', 0)}</b>",
        f"Заблокировано: {u.get('blocked', 0)}, не вериф.: {u.get('unverified', 0)}",
        f"Активных подписок: <b>{u.get('active_subscriptions', 0)}</b>",
    ]
    bt = u.get("by_tier") or {}
    if bt:
        tier_s = ", ".join(f"{k}: {v}" for k, v in sorted(bt.items()))
        lines.append(f"По тарифам: <code>{tier_s}</code>")
    lines.extend(
        [
            "",
            "<b>Мастеринги</b>",
            f"Сегодня: <b>{m.get('today', 0)}</b>",
            f"За 30 дней: <b>{m.get('month', 0)}</b>",
            f"Всего записей: {m.get('total', 0)}",
        ]
    )
    return "\n".join(lines)


def format_jobs_ru() -> str:
    jobs = js.all_jobs()
    run = sum(1 for j in jobs.values() if j.get("status") == "running")
    pend = sum(1 for j in jobs.values() if j.get("status") in ("pending", "queued"))
    done = sum(1 for j in jobs.values() if j.get("status") == "done")
    err = sum(1 for j in jobs.values() if j.get("status") == "error")
    total = len(jobs)
    ip, np = _sem_in_use(js.sem_priority, getattr(settings, "semaphore_priority", 2))
    inorm, nnorm = _sem_in_use(js.sem_normal, getattr(settings, "semaphore_normal", 1))
    lines = [
        "⚙️ <b>Задачи мастеринга</b>",
        f"В работе (running): <b>{run}</b>",
        f"В очереди / ожидание: <b>{pend}</b>",
        f"Успешно (в кэше): {done}, ошибок (в кэше): {err}",
        f"Всего в памяти: <b>{total}</b>",
        "",
        "<b>Семафоры</b>",
        f"Pro/Studio: занято <b>{ip}</b> из {np}",
        f"Free/гость: занято <b>{inorm}</b> из {nnorm}",
    ]
    running_ids = [jid for jid, j in jobs.items() if j.get("status") == "running"][:8]
    if running_ids:
        lines.append("")
        lines.append("<b>ID в работе:</b>")
        for jid in running_ids:
            lines.append(f"• <code>{jid[:20]}…</code>" if len(jid) > 20 else f"• <code>{jid}</code>")
    return "\n".join(lines)


def format_health_ru(db) -> str:
    import shutil

    db_ok = "ok"
    try:
        db.execute(sql_text("SELECT 1"))
    except Exception as e:  # noqa: BLE001
        db_ok = str(e)[:120]
    jobs = js.all_jobs()
    run = sum(1 for j in jobs.values() if j.get("status") == "running")
    temp_dir = settings_store.get_setting_str("temp_dir") or getattr(settings, "temp_dir", "/tmp")
    try:
        du = shutil.disk_usage(temp_dir)
        free_mb = du.free // (1024 * 1024)
    except OSError:
        free_mb = -1
    ff = "да" if shutil.which("ffmpeg") else "нет"
    m = get_system_metrics()
    dr = m.get("disk_root") or {}
    lines = [
        "❤️ <b>Состояние сервиса</b>",
        f"БД: <b>{db_ok}</b>",
        f"ffmpeg в PATH: <b>{ff}</b>",
        f"Задач в работе: <b>{run}</b>",
        f"Свободно на temp ({temp_dir[:40]}): <b>{free_mb}</b> МБ",
        f"Диск / (занято): <b>{dr.get('used_percent', '—')}%</b>, свободно {dr.get('free_gb', '—')} ГБ",
        "",
        format_server_oneline(m),
    ]
    return "\n".join(lines)


def format_users_ru(db, st: dict) -> str:
    u = st.get("users", {})
    n = db.query(User).count()
    tg_n = db.query(User).filter(User.telegram_id.isnot(None)).count()
    lines = [
        "👥 <b>Пользователи</b>",
        f"Всего в БД: <b>{n}</b>",
        f"С привязкой Telegram: <b>{tg_n}</b>",
        f"Новых за неделю: <b>{u.get('new_week', 0)}</b>, за 30 дней: <b>{u.get('new_month', 0)}</b>",
        f"Активных подписок: <b>{u.get('active_subscriptions', 0)}</b>",
    ]
    bt = u.get("by_tier") or {}
    if bt:
        lines.append("")
        lines.append("<b>По тарифам:</b>")
        for k, v in sorted(bt.items()):
            lines.append(f"• {k}: <b>{v}</b>")
    return "\n".join(lines)


def format_revenue_ru(st: dict) -> str:
    r = st.get("revenue", {})
    lines = [
        "💰 <b>Выручка (успешные платежи)</b>",
        f"Всего: <b>{r.get('total_rub', 0)}</b> ₽",
        f"За 30 дней: <b>{r.get('month_rub', 0)}</b> ₽",
        f"Транзакций за 30 дней: <b>{r.get('transactions_month', 0)}</b>",
    ]
    return "\n".join(lines)


def format_errors_ru(limit: int = 12) -> str:
    items = js.list_recent_error_jobs(limit)
    lines = ["🔴 <b>Последние ошибки мастеринга</b>", ""]
    if not items:
        lines.append("Записей с текстом ошибки не найдено.")
        return "\n".join(lines)
    for it in items:
        jid = it.get("job_id", "?")
        err = (it.get("error") or "")[:350]
        ts = it.get("done_at") or it.get("created_at") or 0
        tss = time.strftime("%d.%m %H:%M", time.localtime(float(ts))) if ts else "—"
        lines.append(f"• <code>{jid[:24]}</code> ({tss})")
        lines.append(f"  <code>{err}</code>")
        lines.append("")
    return "\n".join(lines).rstrip()


def format_full_report_ru(db, st: dict) -> str:
    parts = [
        format_stats_ru(st),
        "",
        "──────────",
        "",
        format_jobs_ru(),
        "",
        "──────────",
        "",
        format_health_ru(db),
        "",
        "──────────",
        "",
        format_users_ru(db, st),
        "",
        "──────────",
        "",
        format_revenue_ru(st),
        "",
        "──────────",
        "",
        format_errors_ru(8),
        "",
        "──────────",
        "",
        format_server_report(),
    ]
    return "\n".join(parts)


def mastering_error_rate_last_hour(db) -> Optional[tuple[int, int]]:
    """(ошибки, всего) за последний час по MasteringJobEvent; None если БД недоступна."""
    if db is None or MasteringJobEvent is None:
        return None
    try:
        now = time.time()
        t0 = now - 3600
        total = (
            db.query(MasteringJobEvent)
            .filter(MasteringJobEvent.created_at >= t0)
            .count()
        )
        err = (
            db.query(MasteringJobEvent)
            .filter(MasteringJobEvent.created_at >= t0, MasteringJobEvent.status == "error")
            .count()
        )
        return err, total
    except Exception:  # noqa: BLE001
        return None
