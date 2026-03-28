"""Старт, помощь, язык, главное меню."""
from aiogram import F, Router
from aiogram.filters import Command, CommandStart
from aiogram.types import Message

from ...database import DB_AVAILABLE, SessionLocal, upsert_telegram_engagement
from ..helpers import get_linked_user, public_url
from ..keyboards import main_menu_reply
from ..texts import lang_for_user, txt

router = Router(name="start")


def _lang(message: Message, db_user) -> str:
    if db_user:
        return lang_for_user(db_user)
    lc = (message.from_user.language_code or "ru").lower()[:2]
    return "en" if lc == "en" else "ru"


@router.message(CommandStart())
async def cmd_start(message: Message) -> None:
    if not DB_AVAILABLE or SessionLocal is None:
        await message.answer("Database unavailable.")
        return
    db = SessionLocal()
    try:
        u = get_linked_user(db, message.from_user.id)
        upsert_telegram_engagement(db, message.from_user.id)
        lang = _lang(message, u)
        await message.answer(
            txt(lang, "start_welcome"),
            reply_markup=main_menu_reply(lang),
        )
    finally:
        db.close()


@router.message(Command("help"))
@router.message(F.text.in_(["❓ Help", "❓ Помощь"]))
async def cmd_help(message: Message) -> None:
    if not DB_AVAILABLE or SessionLocal is None:
        return
    db = SessionLocal()
    try:
        u = get_linked_user(db, message.from_user.id)
        lang = _lang(message, u)
        await message.answer(txt(lang, "help"))
    finally:
        db.close()


@router.message(Command("lang"))
async def cmd_lang(message: Message) -> None:
    if not DB_AVAILABLE or SessionLocal is None:
        return
    db = SessionLocal()
    try:
        u = get_linked_user(db, message.from_user.id)
        if not u:
            await message.answer("Use /link first. /lang после привязки.")
            return
        cur = (u.telegram_lang or "ru").lower()
        new = "en" if cur == "ru" else "ru"
        u.telegram_lang = new
        db.commit()
        await message.answer(txt(new, "lang_ok", lang=new.upper()), reply_markup=main_menu_reply(new))
    finally:
        db.close()


@router.message(F.text.in_(["🎨 Presets", "🎨 Пресеты"]))
async def text_presets(message: Message) -> None:
    from .presets import send_presets_overview

    await send_presets_overview(message)
