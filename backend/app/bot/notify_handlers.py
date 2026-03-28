"""Обработчики бота уведомлений: то же нижнее меню, что у user bot; действия — в клиентском боте."""
from aiogram import F, Router
from aiogram.filters import Command, CommandStart
from aiogram.types import Message

from ..config import settings
from .keyboards import all_main_menu_button_texts, main_menu_reply

router = Router(name="notify_bot")


def _hint_ru() -> str:
    url = (getattr(settings, "user_bot_telegram_url", "") or "https://t.me/magicmasterpro_user_bot").strip()
    return (
        "Здесь приходят <b>служебные уведомления</b> (запуск сервера, оплаты, ошибки).\n\n"
        "Мастеринг, анализ, AI и баланс — откройте клиентский бот:\n"
        f'<a href="{url}">перейти в рабочий бот</a>'
    )


def _hint_en() -> str:
    url = (getattr(settings, "user_bot_telegram_url", "") or "https://t.me/magicmasterpro_user_bot").strip()
    return (
        "This chat is for <b>service notifications</b> only.\n\n"
        "For mastering, analysis, AI — open the client bot:\n"
        f'<a href="{url}">open client bot</a>'
    )


def _lang(message: Message) -> str:
    lc = (message.from_user.language_code or "ru").lower()[:2]
    return "en" if lc == "en" else "ru"


@router.message(CommandStart())
async def notify_start(message: Message) -> None:
    lang = _lang(message)
    await message.answer(
        _hint_en() if lang == "en" else _hint_ru(),
        reply_markup=main_menu_reply(lang),
        disable_web_page_preview=True,
    )


@router.message(Command("help"))
async def notify_help(message: Message) -> None:
    lang = _lang(message)
    await message.answer(
        _hint_en() if lang == "en" else _hint_ru(),
        reply_markup=main_menu_reply(lang),
        disable_web_page_preview=True,
    )


@router.message(F.text.in_(all_main_menu_button_texts()))
async def notify_menu_button(message: Message) -> None:
    """Те же подписи, что у user bot — поясняем, где реальная обработка."""
    lang = _lang(message)
    await message.answer(
        _hint_en() if lang == "en" else _hint_ru(),
        reply_markup=main_menu_reply(lang),
        disable_web_page_preview=True,
    )
