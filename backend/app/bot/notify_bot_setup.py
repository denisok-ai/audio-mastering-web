"""Бот уведомлений (TELEGRAM_BOT_TOKEN): отдельный от user bot, своё webhook /bot/notify/webhook."""
from __future__ import annotations

from typing import Any, Optional, Tuple

from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.fsm.storage.memory import MemoryStorage

from ..config import settings


def build_notify_bot_dp() -> Tuple[Optional[Bot], Optional[Dispatcher]]:
    """Сборка бота уведомлений. Если токен совпадает с user bot или пуст — не поднимаем."""
    token = (getattr(settings, "telegram_bot_token", "") or "").strip()
    user_tok = (getattr(settings, "user_bot_token", "") or "").strip()
    if not token or (user_tok and token == user_tok):
        return None, None
    try:
        bot = Bot(token=token, default=DefaultBotProperties(parse_mode=ParseMode.HTML))
    except Exception:
        import logging

        logging.getLogger(__name__).warning(
            "TELEGRAM_BOT_TOKEN невалиден — уведомительный webhook не будет поднят."
        )
        return None, None
    dp = Dispatcher(storage=MemoryStorage())
    from .notify_handlers import router as notify_router

    dp.include_router(notify_router)
    return bot, dp


_notify_bot_singleton: Any = None
_notify_dp_singleton: Any = None


def get_notify_bot_dp() -> Tuple[Optional[Bot], Optional[Dispatcher]]:
    global _notify_bot_singleton, _notify_dp_singleton
    tok = (getattr(settings, "telegram_bot_token", "") or "").strip()
    ut = (getattr(settings, "user_bot_token", "") or "").strip()
    if not tok or (ut and tok == ut):
        return None, None
    if _notify_bot_singleton is None:
        _notify_bot_singleton, _notify_dp_singleton = build_notify_bot_dp()
    return _notify_bot_singleton, _notify_dp_singleton
