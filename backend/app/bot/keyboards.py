"""Клавиатуры Telegram."""
from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup, ReplyKeyboardMarkup, KeyboardButton

from ..pipeline import STYLE_CONFIGS


def main_menu_reply(lang: str) -> ReplyKeyboardMarkup:
    if lang == "en":
        rows = [
            [KeyboardButton(text="🎛 Master"), KeyboardButton(text="📊 Analyze")],
            [KeyboardButton(text="🎨 Presets"), KeyboardButton(text="💬 AI chat")],
            [KeyboardButton(text="💰 Balance"), KeyboardButton(text="❓ Help")],
        ]
    else:
        rows = [
            [KeyboardButton(text="🎛 Мастеринг"), KeyboardButton(text="📊 Анализ")],
            [KeyboardButton(text="🎨 Пресеты"), KeyboardButton(text="💬 AI чат")],
            [KeyboardButton(text="💰 Баланс"), KeyboardButton(text="❓ Помощь")],
        ]
    return ReplyKeyboardMarkup(keyboard=rows, resize_keyboard=True)


def presets_inline(prefix: str = "p") -> InlineKeyboardMarkup:
    """prefix:master -> callback p:m:standard"""
    styles = list(STYLE_CONFIGS.keys())
    buttons = []
    row: list[InlineKeyboardButton] = []
    for i, s in enumerate(styles):
        row.append(InlineKeyboardButton(text=s, callback_data=f"{prefix}:m:{s}"))
        if len(row) >= 2:
            buttons.append(row)
            row = []
    if row:
        buttons.append(row)
    return InlineKeyboardMarkup(inline_keyboard=buttons)


def admin_menu_inline() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(text="🖥 Сервер", callback_data="adm:server"),
                InlineKeyboardButton(text="❤️ Здоровье", callback_data="adm:health"),
            ],
            [
                InlineKeyboardButton(text="📊 Статистика", callback_data="adm:stats"),
                InlineKeyboardButton(text="⚙️ Задачи", callback_data="adm:jobs"),
            ],
            [
                InlineKeyboardButton(text="👥 Пользователи", callback_data="adm:users"),
                InlineKeyboardButton(text="💰 Выручка", callback_data="adm:revenue"),
            ],
            [
                InlineKeyboardButton(text="🔴 Ошибки", callback_data="adm:errors"),
                InlineKeyboardButton(text="📋 Полный отчёт", callback_data="adm:report"),
            ],
        ]
    )
