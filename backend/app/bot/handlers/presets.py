"""Просмотр стилей мастеринга."""
from aiogram import Router
from aiogram.filters import Command
from aiogram.types import Message

from ...pipeline import STYLE_CONFIGS
from ...database import DB_AVAILABLE, SessionLocal
from ..helpers import get_linked_user
from ..texts import lang_for_user

router = Router(name="presets")


@router.message(Command("presets"))
async def cmd_presets(message: Message) -> None:
    await send_presets_overview(message)


async def send_presets_overview(message: Message) -> None:
    lines = []
    lang = "ru"
    if DB_AVAILABLE and SessionLocal is not None:
        db = SessionLocal()
        try:
            u = get_linked_user(db, message.from_user.id)
            lang = lang_for_user(u) if u else "ru"
        finally:
            db.close()
    for name, cfg in STYLE_CONFIGS.items():
        lu = cfg.get("lufs", -14)
        if lang == "en":
            lines.append(f"• <b>{name}</b> — target {lu} LUFS")
        else:
            lines.append(f"• <b>{name}</b> — цель {lu} LUFS")
    text = "\n".join(lines) if lines else "—"
    header = "🎨 Styles" if lang == "en" else "🎨 Стили мастеринга"
    await message.answer(f"{header}\n{text}")
