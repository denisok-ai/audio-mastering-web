"""Админ-команды в user-боте (только is_admin)."""
import asyncio
import shutil
import time

from aiogram import F, Router
from aiogram.filters import Command
from aiogram.types import CallbackQuery, Message

from ...database import DB_AVAILABLE, SessionLocal, User, list_telegram_ids_for_broadcast
from ... import jobs_store as js
from ... import settings_store
from ..helpers import get_linked_user, public_url
from ..keyboards import admin_menu_inline
from ..notify_user import send_user_bot_text_sync
from ..texts import lang_for_user, txt

router = Router(name="bot_admin")


def _is_bot_admin(message: Message) -> bool:
    if not DB_AVAILABLE or SessionLocal is None:
        return False
    db = SessionLocal()
    try:
        u = get_linked_user(db, message.from_user.id)
        return bool(u and getattr(u, "is_admin", False))
    finally:
        db.close()


@router.message(Command("admin"))
async def cmd_admin(message: Message) -> None:
    if not _is_bot_admin(message):
        await message.answer(txt("ru", "admin_only"))
        return
    await message.answer("Admin panel:", reply_markup=admin_menu_inline())


@router.callback_query(F.data.startswith("adm:"))
async def admin_cb(query: CallbackQuery) -> None:
    if not DB_AVAILABLE or SessionLocal is None:
        await query.answer()
        return
    db = SessionLocal()
    try:
        u = get_linked_user(db, query.from_user.id)
        if not u or not getattr(u, "is_admin", False):
            await query.answer("Denied", show_alert=True)
            return
        action = (query.data or "").split(":")[1]
        if action == "stats":
            from ...services.stats_service import get_dashboard_stats

            st = get_dashboard_stats(db)
            users = st.get("users", {})
            text = (
                f"📊 <b>Stats</b>\n"
                f"Users: {users.get('total', 0)}\n"
                f"New week: {users.get('new_week', 0)}\n"
                f"Masterings today: {st.get('masterings', {}).get('today', 0)}\n"
            )
            await query.message.answer(text)
        elif action == "health":
            from sqlalchemy import text as _sql_text

            db_ok = "ok"
            try:
                db.execute(_sql_text("SELECT 1"))
            except Exception as e:  # noqa: BLE001
                db_ok = str(e)[:80]
            jobs = js.all_jobs()
            run = sum(1 for j in jobs.values() if j.get("status") == "running")
            du = shutil.disk_usage(settings_store.get_setting_str("temp_dir") or "/tmp")
            free_mb = du.free // (1024 * 1024)
            ff = "yes" if shutil.which("ffmpeg") else "no"
            await query.message.answer(
                f"❤️ <b>Health</b>\nDB: {db_ok}\nJobs running: {run}\nDisk free MB: {free_mb}\nffmpeg: {ff}"
            )
        elif action == "users":
            n = db.query(User).count()
            tg_n = db.query(User).filter(User.telegram_id.isnot(None)).count()
            await query.message.answer(f"👥 Users total: {n}, with Telegram: {tg_n}")
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
        users = st.get("users", {})
        await message.answer(
            f"Users: {users.get('total', 0)}, masterings month: {st.get('masterings', {}).get('month', 0)}"
        )
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
