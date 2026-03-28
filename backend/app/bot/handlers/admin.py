"""Админ-команды в user-боте (только is_admin). Отчёты на русском."""
import asyncio

from aiogram import F, Router
from aiogram.filters import Command
from aiogram.types import CallbackQuery, Message

from ...database import DB_AVAILABLE, SessionLocal, list_telegram_ids_for_broadcast
from ..admin_reports import (
    format_errors_ru,
    format_full_report_ru,
    format_health_ru,
    format_jobs_ru,
    format_revenue_ru,
    format_server_oneline,
    format_stats_ru,
    format_users_ru,
)
from ..helpers import get_linked_user
from ..keyboards import admin_menu_inline
from ..notify_user import send_user_bot_text_sync
from ..server_metrics import format_server_report, get_system_metrics
from ..texts import txt

router = Router(name="bot_admin")

_TELEGRAM_MSG_MAX = 4000


def _split_telegram_html(s: str, max_len: int = _TELEGRAM_MSG_MAX) -> list[str]:
    s = (s or "").strip()
    parts: list[str] = []
    while s:
        if len(s) <= max_len:
            parts.append(s)
            break
        cut = s.rfind("\n", 0, max_len)
        if cut < max_len // 2:
            cut = max_len
        parts.append(s[:cut])
        s = s[cut:].lstrip()
    return parts


def _is_bot_admin(message: Message) -> bool:
    if not DB_AVAILABLE or SessionLocal is None:
        return False
    db = SessionLocal()
    try:
        u = get_linked_user(db, message.from_user.id)
        return bool(u and getattr(u, "is_admin", False))
    finally:
        db.close()


async def _answer_admin_chunks(message: Message, text: str) -> None:
    for chunk in _split_telegram_html(text):
        await message.answer(chunk)


@router.message(Command("admin"))
async def cmd_admin(message: Message) -> None:
    if not _is_bot_admin(message):
        await message.answer(txt("ru", "admin_only"))
        return
    await message.answer("Панель администратора:", reply_markup=admin_menu_inline())


@router.callback_query(F.data.startswith("adm:"))
async def admin_cb(query: CallbackQuery) -> None:
    if not DB_AVAILABLE or SessionLocal is None:
        await query.answer()
        return
    db = SessionLocal()
    try:
        u = get_linked_user(db, query.from_user.id)
        if not u or not getattr(u, "is_admin", False):
            await query.answer("Нет доступа", show_alert=True)
            return
        action = (query.data or "").split(":")[1]

        if action == "server":
            await query.message.answer(format_server_report(get_system_metrics()))
        elif action == "stats":
            from ...services.stats_service import get_dashboard_stats

            st = get_dashboard_stats(db)
            await query.message.answer(format_stats_ru(st))
        elif action == "jobs":
            await query.message.answer(format_jobs_ru())
        elif action == "health":
            await query.message.answer(format_health_ru(db))
        elif action == "users":
            from ...services.stats_service import get_dashboard_stats

            st = get_dashboard_stats(db)
            await query.message.answer(format_users_ru(db, st))
        elif action == "revenue":
            from ...services.stats_service import get_dashboard_stats

            st = get_dashboard_stats(db)
            await query.message.answer(format_revenue_ru(st))
        elif action == "errors":
            await query.message.answer(format_errors_ru(12))
        elif action == "report":
            from ...services.stats_service import get_dashboard_stats

            st = get_dashboard_stats(db)
            full = format_full_report_ru(db, st)
            await _answer_admin_chunks(query.message, full)

        await query.answer()
    finally:
        db.close()


@router.message(Command("stats"))
async def cmd_stats(message: Message) -> None:
    if not _is_bot_admin(message):
        await message.answer(txt("ru", "admin_only"))
        return
    if not DB_AVAILABLE or SessionLocal is None:
        return
    db = SessionLocal()
    try:
        from ...services.stats_service import get_dashboard_stats

        st = get_dashboard_stats(db)
        await message.answer(format_stats_ru(st))
    finally:
        db.close()


@router.message(Command("server"))
async def cmd_server(message: Message) -> None:
    if not _is_bot_admin(message):
        await message.answer(txt("ru", "admin_only"))
        return
    m = get_system_metrics()
    await message.answer(format_server_oneline(m))


@router.message(Command("jobs"))
async def cmd_jobs(message: Message) -> None:
    if not _is_bot_admin(message):
        await message.answer(txt("ru", "admin_only"))
        return
    await message.answer(format_jobs_ru())


@router.message(Command("errors"))
async def cmd_errors(message: Message) -> None:
    if not _is_bot_admin(message):
        await message.answer(txt("ru", "admin_only"))
        return
    await message.answer(format_errors_ru(15))


@router.message(Command("report"))
async def cmd_report(message: Message) -> None:
    if not _is_bot_admin(message):
        await message.answer(txt("ru", "admin_only"))
        return
    if not DB_AVAILABLE or SessionLocal is None:
        return
    db = SessionLocal()
    try:
        from ...services.stats_service import get_dashboard_stats

        st = get_dashboard_stats(db)
        full = format_full_report_ru(db, st)
        await _answer_admin_chunks(message, full)
    finally:
        db.close()


@router.message(Command("broadcast"))
async def cmd_broadcast(message: Message) -> None:
    if not _is_bot_admin(message):
        await message.answer(txt("ru", "admin_only"))
        return
    parts = (message.text or "").split(maxsplit=1)
    body = parts[1].strip() if len(parts) > 1 else ""
    if not body:
        await message.answer(txt("ru", "broadcast_usage"))
        return
    if not DB_AVAILABLE or SessionLocal is None:
        return
    db = SessionLocal()
    try:
        ids = list_telegram_ids_for_broadcast(db)
    finally:
        db.close()
    n = 0
    for chat_id in ids:
        if send_user_bot_text_sync(chat_id, body[:4000]):
            n += 1
        await asyncio.sleep(0.04)
    await message.answer(txt("ru", "broadcast_done", n=n))
