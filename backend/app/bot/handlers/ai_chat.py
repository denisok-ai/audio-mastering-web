"""AI-ассистент: /ask и свободный текст."""
import asyncio
import logging

from aiogram import F, Router
from aiogram.filters import Command, StateFilter
from aiogram.types import Message

from ... import ai as ai_module
from ... import settings_store
from ...config import settings
from ...database import DB_AVAILABLE, SessionLocal, log_ai_usage
from ..helpers import get_linked_user
from ..texts import lang_for_user, txt

router = Router(name="ai_chat")
logger = logging.getLogger(__name__)


def _tier_for_bot(u) -> str:
    if getattr(settings, "debug_mode", False):
        return "studio"
    if u:
        return (u.tier or "free").lower()
    return "free"


def _ident(u, telegram_id: int) -> str:
    if u:
        return f"user:{u.id}"
    return f"tg:{telegram_id}"


@router.message(Command("ask"))
async def cmd_ask(message: Message) -> None:
    parts = (message.text or "").split(maxsplit=1)
    if len(parts) < 2:
        await message.answer("Usage: /ask your question\nИспользование: /ask ваш вопрос")
        return
    await _do_chat(message, parts[1].strip())


@router.message(
    StateFilter(None),
    F.text
    & ~F.text.startswith("/")
    & ~F.text.in_(
        [
            "🎛 Master",
            "🎛 Мастеринг",
            "📊 Analyze",
            "📊 Анализ",
            "🎨 Presets",
            "🎨 Пресеты",
            "💬 AI chat",
            "💬 AI чат",
            "💰 Balance",
            "💰 Баланс",
            "❓ Help",
            "❓ Помощь",
        ]
    ),
)
async def free_text_ai(message: Message) -> None:
    """Свободный текст → AI (если не похоже на кнопку меню)."""
    t = (message.text or "").strip()
    if len(t) < 3 or len(t) > 1800:
        return
    await _do_chat(message, t)


@router.message(F.text.in_(["💬 AI chat", "💬 AI чат"]))
async def ai_menu_hint(message: Message) -> None:
    await message.answer(
        "Напишите вопрос текстом или: /ask …\nSend a message or: /ask …"
    )


async def _do_chat(message: Message, user_text: str) -> None:
    if not settings_store.get_setting_bool("feature_ai_enabled", True):
        await message.answer(txt("ru", "ai_disabled"))
        return
    if not DB_AVAILABLE or SessionLocal is None:
        return

    db = SessionLocal()
    try:
        u = get_linked_user(db, message.from_user.id)
        lang = lang_for_user(u) if u else "ru"
        tier = _tier_for_bot(u)
        ident = _ident(u, message.from_user.id)
        lim = ai_module.check_ai_rate_limit(ident, tier)
        if not lim["ok"]:
            await message.answer(txt(lang, "ai_limit"))
            return
        ctx = {"product": "Magic Master", "lang": lang}
        user_id_for_log = u.id if u else None
    finally:
        db.close()

    try:
        reply = await asyncio.to_thread(
            ai_module.chat_assistant,
            [{"role": "user", "content": user_text}],
            ctx,
        )
    except Exception:  # noqa: BLE001
        logger.exception("AI chat error for %s", ident)
        await message.answer(txt(lang, "error"))
        return

    ai_module.record_ai_usage(ident)
    db2 = SessionLocal()
    try:
        log_ai_usage("chat", user_id_for_log, tier)
    finally:
        db2.close()

    await message.answer((reply or "…")[:4090])
