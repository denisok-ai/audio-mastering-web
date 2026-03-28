"""Сборка Bot + Dispatcher."""
from __future__ import annotations

from typing import Any, Optional, Tuple

from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.fsm.storage.memory import MemoryStorage

from ..config import settings


def build_bot_dp() -> Tuple[Optional[Bot], Optional[Dispatcher]]:
    token = (getattr(settings, "user_bot_token", "") or "").strip()
    if not token:
        return None, None
    bot = Bot(token=token, default=DefaultBotProperties(parse_mode=ParseMode.HTML))
    dp = Dispatcher(storage=MemoryStorage())
    from .handlers import register_handlers

    register_handlers(dp)
    return bot, dp


_bot_singleton: Any = None
_dp_singleton: Any = None


def get_bot_dp() -> Tuple[Optional[Bot], Optional[Dispatcher]]:
    global _bot_singleton, _dp_singleton
    if _bot_singleton is None and (getattr(settings, "user_bot_token", "") or "").strip():
        _bot_singleton, _dp_singleton = build_bot_dp()
    return _bot_singleton, _dp_singleton
