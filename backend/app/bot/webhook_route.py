"""FastAPI-маршрут webhook для aiogram."""
import logging

from aiogram.types import Update
from fastapi import APIRouter, HTTPException, Request

from ..config import settings
from .setup import get_bot_dp

logger = logging.getLogger(__name__)

router = APIRouter(tags=["telegram-bot"])


@router.post("/bot/webhook")
async def telegram_webhook(request: Request) -> dict:
    secret_cfg = (getattr(settings, "user_bot_webhook_secret", "") or "").strip()
    if secret_cfg:
        got = request.headers.get("X-Telegram-Bot-Api-Secret-Token") or ""
        if got != secret_cfg:
            raise HTTPException(status_code=403, detail="Invalid secret")
    bot, dp = get_bot_dp()
    if bot is None or dp is None:
        raise HTTPException(status_code=503, detail="User bot not configured")
    try:
        data = await request.json()
        update = Update.model_validate(data)
        await dp.feed_update(bot, update)
    except Exception:  # noqa: BLE001
        logger.exception("telegram_webhook feed_update failed")
        raise HTTPException(status_code=400, detail="Bad update")
    return {"ok": True}
