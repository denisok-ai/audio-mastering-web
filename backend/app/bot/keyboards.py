"""Клавиатуры Telegram."""
from functools import lru_cache

from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup, ReplyKeyboardMarkup, KeyboardButton

from ..pipeline import STYLE_CONFIGS


def main_menu_button_rows(lang: str) -> list[list[str]]:
    """Тексты кнопок нижнего меню (RU/EN) — единый источник для ReplyKeyboard и sendMessage API."""
    if (lang or "ru").lower()[:2] == "en":
        return [
            ["🎛 Master", "📊 Analyze"],
            ["🎨 Presets", "💬 AI chat"],
            ["💰 Balance", "❓ Help"],
        ]
    return [
        ["🎛 Мастеринг", "📊 Анализ"],
        ["🎨 Пресеты", "💬 AI чат"],
        ["💰 Баланс", "❓ Помощь"],
    ]


def main_menu_reply_markup_dict(lang: str = "ru") -> dict:
    """Словарь reply_markup для Telegram Bot API (urllib / notifier), без aiogram."""
    rows = main_menu_button_rows(lang)
    return {
        "keyboard": [[{"text": t} for t in row] for row in rows],
        "resize_keyboard": True,
    }


@lru_cache
def all_main_menu_button_texts() -> frozenset[str]:
    """Все подписи кнопок главного меню (RU+EN) для матчинга входящих сообщений."""
    s: set[str] = set()
    for lang in ("ru", "en"):
        for row in main_menu_button_rows(lang):
            s.update(row)
    return frozenset(s)


def main_menu_reply(lang: str) -> ReplyKeyboardMarkup:
    rows_txt = main_menu_button_rows(lang)
    rows = [[KeyboardButton(text=t) for t in row] for row in rows_txt]
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
