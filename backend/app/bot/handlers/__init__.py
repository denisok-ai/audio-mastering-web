"""Регистрация роутеров бота."""
from aiogram import Dispatcher

from .start import router as start_router
from .account import router as account_router
from .master import router as master_router
from .analyze import router as analyze_router
from .ai_chat import router as ai_router
from .presets import router as presets_router
from .admin import router as admin_router


def register_handlers(dp: Dispatcher) -> None:
    dp.include_router(admin_router)
    dp.include_router(account_router)
    dp.include_router(master_router)
    dp.include_router(analyze_router)
    dp.include_router(ai_router)
    dp.include_router(presets_router)
    dp.include_router(start_router)
