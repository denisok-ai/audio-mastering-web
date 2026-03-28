"""Установка / снятие webhook при старте приложения."""
import logging

from ..config import settings
from .setup import get_bot_dp

logger = logging.getLogger(__name__)


async def bot_startup() -> None:
    token = (getattr(settings, "user_bot_token", "") or "").strip()
    base = (getattr(settings, "public_base_url", "") or "").strip().rstrip("/")
    if not token or not base:
        logger.info("User bot: skipped (no USER_BOT_TOKEN or PUBLIC_BASE_URL)")
        return
    bot, _ = get_bot_dp()
    if not bot:
        return
    secret = (getattr(settings, "user_bot_webhook_secret", "") or "").strip()
    url = f"{base}/bot/webhook"
    try:
        await bot.set_webhook(
            url=url,
            secret_token=secret or None,
            drop_pending_updates=True,
        )
        logger.info("User bot webhook set: %s", url)
    except Exception as e:  # noqa: BLE001
        logger.warning("User bot set_webhook failed: %s", e)


async def bot_shutdown() -> None:
    bot, _ = get_bot_dp()
    if not bot:
        return
    try:
        await bot.delete_webhook(drop_pending_updates=False)
        await bot.session.close()
    except Exception as e:  # noqa: BLE001
        logger.debug("bot shutdown: %s", e)
