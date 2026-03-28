"""Webhook для бота уведомлений (TELEGRAM_BOT_TOKEN)."""
import logging

from aiogram.types import Update
from fastapi import APIRouter, HTTPException, Request

from ..config import settings
from .notify_bot_setup import get_notify_bot_dp

logger = logging.getLogger(__name__)

router = APIRouter(tags=["telegram-notify-bot"])


@router.post("/bot/notify/webhook")
async def notify_bot_webhook(request: Request) -> dict:
    secret_cfg = (getattr(settings, "telegram_bot_webhook_secret", "") or "").strip()
    if secret_cfg:
        got = request.headers.get("X-Telegram-Bot-Api-Secret-Token") or ""
        if got != secret_cfg:
            raise HTTPException(status_code=403, detail="Invalid secret")
    bot, dp = get_notify_bot_dp()
    if bot is None or dp is None:
        raise HTTPException(status_code=503, detail="Notify bot not configured")
    try:
        data = await request.json()
        update = Update.model_validate(data)
        await dp.feed_update(bot, update)
    except Exception:  # noqa: BLE001
        logger.exception("notify_bot_webhook feed_update failed")
    return {"ok": True}
