"""Установка / снятие webhook при старте приложения."""
import logging

from aiogram.types import BotCommand

from ..config import settings
from .setup import get_bot_dp

logger = logging.getLogger(__name__)


def _user_bot_commands() -> list[BotCommand]:
    """Команды меню «/» для клиентского бота (RU, кратко)."""
    return [
        BotCommand(command="start", description="Старт и главное меню"),
        BotCommand(command="master", description="Мастеринг аудио"),
        BotCommand(command="analyze", description="Анализ громкости и спектра"),
        BotCommand(command="presets", description="Стили мастеринга"),
        BotCommand(command="ask", description="AI-консультант"),
        BotCommand(command="balance", description="Токены и тариф"),
        BotCommand(command="history", description="История мастеринга"),
        BotCommand(command="pricing", description="Тарифы и оплата"),
        BotCommand(command="status", description="Статус сервиса"),
        BotCommand(command="link", description="Привязка аккаунта сайта"),
        BotCommand(command="lang", description="Язык интерфейса"),
        BotCommand(command="help", description="Список команд"),
        BotCommand(command="admin", description="Панель администратора"),
    ]


async def bot_startup() -> None:
    token = (getattr(settings, "user_bot_token", "") or "").strip()
    base = (getattr(settings, "public_base_url", "") or "").strip().rstrip("/")
    if not token or not base:
        logger.info("User bot: skipped (no USER_BOT_TOKEN or PUBLIC_BASE_URL)")
        return
    bot, _ = get_bot_dp()
    if not bot:
        logger.warning("User bot: get_bot_dp() returned None — проверьте USER_BOT_TOKEN")
        return

    cmds = _user_bot_commands()
    try:
        await bot.set_my_commands(cmds)
        logger.info("User bot: зарегистрировано команд в меню: %d", len(cmds))
    except Exception:
        logger.exception("User bot: не удалось set_my_commands")

    secret = (getattr(settings, "user_bot_webhook_secret", "") or "").strip()
    url = f"{base}/bot/webhook"
    try:
        await bot.set_webhook(
            url=url,
            secret_token=secret or None,
            drop_pending_updates=True,
        )
        logger.info("User bot webhook set: %s (secret=%s)", url, "yes" if secret else "no")
    except Exception:
        logger.exception("User bot set_webhook FAILED — входящие обновления не будут работать")


async def bot_shutdown() -> None:
    bot, _ = get_bot_dp()
    if not bot:
        return
    try:
        await bot.delete_webhook(drop_pending_updates=False)
        await bot.session.close()
    except Exception as e:  # noqa: BLE001
        logger.debug("bot shutdown: %s", e)
