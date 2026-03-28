"""Обработчики бота уведомлений: админское нижнее меню (отчёты); пользовательский бот — по ссылке."""
import asyncio
import logging

from aiogram import F, Router
from aiogram.exceptions import TelegramBadRequest
from aiogram.filters import Command, CommandStart
from aiogram.types import Message

from ..config import settings
from ..database import DB_AVAILABLE, SessionLocal, list_telegram_ids_for_broadcast
from .admin_reports import (
    format_errors_ru,
    format_full_report_ru,
    format_health_ru,
    format_jobs_ru,
    format_revenue_ru,
    format_stats_ru,
    format_users_ru,
)
from .keyboards import admin_menu_reply, all_admin_menu_button_texts
from .notify_user import send_user_bot_text_sync
from .server_metrics import format_server_report, get_system_metrics
from .texts import txt

router = Router(name="notify_bot")

logger = logging.getLogger(__name__)

_TELEGRAM_MSG_MAX = 4000


async def _safe_answer(message: Message, text: str, **kwargs: object) -> None:
    """В Topics-чатах message.answer() может дать message thread not found — повтор без топика."""
    try:
        await message.answer(text, **kwargs)  # type: ignore[arg-type]
    except TelegramBadRequest as exc:
        logger.debug("notify_bot: answer failed (%s), retry send_message without topic", exc)
        safe = {k: v for k, v in kwargs.items() if k != "message_thread_id"}
        await message.bot.send_message(chat_id=message.chat.id, text=text, **safe)  # type: ignore[arg-type]


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


async def _answer_chunks(message: Message, text: str, *, lang: str) -> None:
    for chunk in _split_telegram_html(text):
        await _safe_answer(message, chunk, reply_markup=admin_menu_reply(lang))


def _hint_ru() -> str:
    url = (getattr(settings, "user_bot_telegram_url", "") or "https://t.me/magicmasterpro_user_bot").strip()
    return (
        "Здесь приходят <b>служебные уведомления</b> (запуск сервера, оплаты, ошибки).\n\n"
        "Мастеринг, анализ, AI и баланс — в клиентском боте:\n"
        f'<a href="{url}">перейти в рабочий бот</a>'
    )


def _hint_en() -> str:
    url = (getattr(settings, "user_bot_telegram_url", "") or "https://t.me/magicmasterpro_user_bot").strip()
    return (
        "This chat is for <b>service notifications</b> and admin reports.\n\n"
        "For mastering, analysis, AI — open the client bot:\n"
        f'<a href="{url}">open client bot</a>'
    )


def _lang(message: Message) -> str:
    lc = (message.from_user.language_code or "ru").lower()[:2]
    return "en" if lc == "en" else "ru"


@router.message(CommandStart())
async def notify_start(message: Message) -> None:
    lang = _lang(message)
    await _safe_answer(
        message,
        _hint_en() if lang == "en" else _hint_ru(),
        reply_markup=admin_menu_reply(lang),
        disable_web_page_preview=True,
    )


@router.message(Command("help"))
async def notify_help(message: Message) -> None:
    lang = _lang(message)
    await _safe_answer(
        message,
        _hint_en() if lang == "en" else _hint_ru(),
        reply_markup=admin_menu_reply(lang),
        disable_web_page_preview=True,
    )


@router.message(Command("broadcast"))
async def notify_broadcast(message: Message) -> None:
    """Рассылка в user bot всем подписчикам (чат уведомлений = доверенный)."""
    lang = _lang(message)
    parts = (message.text or "").split(maxsplit=1)
    body = parts[1].strip() if len(parts) > 1 else ""
    if not body:
        await _safe_answer(
            message,
            txt("ru", "broadcast_usage") if lang == "ru" else "Usage: /broadcast &lt;message&gt;",
            reply_markup=admin_menu_reply(lang),
        )
        return
    if not DB_AVAILABLE or SessionLocal is None:
        await _safe_answer(message, "DB недоступна.", reply_markup=admin_menu_reply(lang))
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
    await _safe_answer(
        message,
        txt("ru", "broadcast_done", n=n) if lang == "ru" else f"Sent to {n} chats.",
        reply_markup=admin_menu_reply(lang),
    )


# Кнопки RU
_BTN_SERVER_RU = "🖥 Сервер"
_BTN_STATS_RU = "📊 Статистика"
_BTN_JOBS_RU = "⚙️ Задачи"
_BTN_ERRORS_RU = "🔴 Ошибки"
_BTN_HEALTH_RU = "❤️ Здоровье"
_BTN_USERS_RU = "👥 Пользователи"
_BTN_REVENUE_RU = "💰 Выручка"
_BTN_REPORT_RU = "📋 Отчёт"
_BTN_BROADCAST_RU = "📢 Рассылка"
_BTN_HELP_RU = "❓ Помощь"

# Кнопки EN
_BTN_SERVER_EN = "🖥 Server"
_BTN_STATS_EN = "📊 Stats"
_BTN_JOBS_EN = "⚙️ Jobs"
_BTN_ERRORS_EN = "🔴 Errors"
_BTN_HEALTH_EN = "❤️ Health"
_BTN_USERS_EN = "👥 Users"
_BTN_REVENUE_EN = "💰 Revenue"
_BTN_REPORT_EN = "📋 Report"
_BTN_BROADCAST_EN = "📢 Broadcast"
_BTN_HELP_EN = "❓ Help"


@router.message(F.text.in_(all_admin_menu_button_texts()))
async def notify_menu_button(message: Message) -> None:
    lang = _lang(message)
    text = (message.text or "").strip()

    if text in (_BTN_HELP_RU, _BTN_HELP_EN):
        await _safe_answer(
            message,
            _hint_en() if lang == "en" else _hint_ru(),
            reply_markup=admin_menu_reply(lang),
            disable_web_page_preview=True,
        )
        return

    if text in (_BTN_BROADCAST_RU, _BTN_BROADCAST_EN):
        await _safe_answer(
            message,
            (
                "Отправьте команду в формате:\n<code>/broadcast текст сообщения</code>"
                if lang == "ru"
                else "Send:\n<code>/broadcast your message</code>"
            ),
            reply_markup=admin_menu_reply(lang),
        )
        return

    if text in (_BTN_SERVER_RU, _BTN_SERVER_EN):
        await _safe_answer(
            message,
            format_server_report(get_system_metrics()),
            reply_markup=admin_menu_reply(lang),
        )
        return

    if text in (_BTN_STATS_RU, _BTN_STATS_EN):
        if not DB_AVAILABLE or SessionLocal is None:
            await _safe_answer(message, "DB недоступна.", reply_markup=admin_menu_reply(lang))
            return
        db = SessionLocal()
        try:
            from ..services.stats_service import get_dashboard_stats

            st = get_dashboard_stats(db)
            await _safe_answer(message, format_stats_ru(st), reply_markup=admin_menu_reply(lang))
        finally:
            db.close()
        return

    if text in (_BTN_JOBS_RU, _BTN_JOBS_EN):
        await _safe_answer(message, format_jobs_ru(), reply_markup=admin_menu_reply(lang))
        return

    if text in (_BTN_ERRORS_RU, _BTN_ERRORS_EN):
        await _safe_answer(message, format_errors_ru(15), reply_markup=admin_menu_reply(lang))
        return

    if text in (_BTN_HEALTH_RU, _BTN_HEALTH_EN):
        if not DB_AVAILABLE or SessionLocal is None:
            await _safe_answer(message, "DB недоступна.", reply_markup=admin_menu_reply(lang))
            return
        db = SessionLocal()
        try:
            await _safe_answer(message, format_health_ru(db), reply_markup=admin_menu_reply(lang))
        finally:
            db.close()
        return

    if text in (_BTN_USERS_RU, _BTN_USERS_EN):
        if not DB_AVAILABLE or SessionLocal is None:
            await _safe_answer(message, "DB недоступна.", reply_markup=admin_menu_reply(lang))
            return
        db = SessionLocal()
        try:
            from ..services.stats_service import get_dashboard_stats

            st = get_dashboard_stats(db)
            await _safe_answer(message, format_users_ru(db, st), reply_markup=admin_menu_reply(lang))
        finally:
            db.close()
        return

    if text in (_BTN_REVENUE_RU, _BTN_REVENUE_EN):
        if not DB_AVAILABLE or SessionLocal is None:
            await _safe_answer(message, "DB недоступна.", reply_markup=admin_menu_reply(lang))
            return
        db = SessionLocal()
        try:
            from ..services.stats_service import get_dashboard_stats

            st = get_dashboard_stats(db)
            await _safe_answer(message, format_revenue_ru(st), reply_markup=admin_menu_reply(lang))
        finally:
            db.close()
        return

    if text in (_BTN_REPORT_RU, _BTN_REPORT_EN):
        if not DB_AVAILABLE or SessionLocal is None:
            await _safe_answer(message, "DB недоступна.", reply_markup=admin_menu_reply(lang))
            return
        db = SessionLocal()
        try:
            from ..services.stats_service import get_dashboard_stats

            st = get_dashboard_stats(db)
            full = format_full_report_ru(db, st)
            await _answer_chunks(message, full, lang=lang)
        finally:
            db.close()
        return
