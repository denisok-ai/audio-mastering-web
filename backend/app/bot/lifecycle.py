"""Установка / снятие webhook при старте приложения."""
import logging

from aiogram.types import BotCommand, BotCommandScopeChat, BotCommandScopeDefault

from ..config import settings
from ..database import DB_AVAILABLE, SessionLocal, get_user_by_telegram_id, list_admin_telegram_ids
from .notify_bot_setup import get_notify_bot_dp
from .setup import get_bot_dp

logger = logging.getLogger(__name__)

# Uvicorn по умолчанию не выводит INFO сторонних логгеров в journald — дублируем в stderr.
_root_bot_log = logging.getLogger("app.bot")
if not _root_bot_log.handlers:
    _h = logging.StreamHandler()
    _h.setLevel(logging.INFO)
    _h.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    _root_bot_log.addHandler(_h)
    _root_bot_log.setLevel(logging.INFO)


def _regular_user_bot_commands() -> list[BotCommand]:
    """Меню «/» для обычных пользователей (без админских команд)."""
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
    ]


def _admin_bot_commands() -> list[BotCommand]:
    """Расширенное меню для админов сайта (привязанный Telegram + is_admin)."""
    cmds = list(_regular_user_bot_commands())
    cmds.extend(
        [
            BotCommand(command="admin", description="Панель администратора"),
            BotCommand(command="server", description="Сервер (кратко)"),
            BotCommand(command="stats", description="Статистика"),
            BotCommand(command="jobs", description="Задачи мастеринга"),
            BotCommand(command="errors", description="Ошибки мастеринга"),
            BotCommand(command="report", description="Полный отчёт"),
            BotCommand(command="broadcast", description="Рассылка в Telegram"),
        ]
    )
    return cmds


async def refresh_menu_for_telegram_chat(telegram_id: int) -> None:
    """
    Обновить меню «/» для одного приватного чата после /link, /code или /unlink.
    Админы — расширенный список; остальные — сброс scope чата (действует меню по умолчанию).
    """
    bot, _ = get_bot_dp()
    if not bot or not telegram_id:
        return
    tid = int(telegram_id)
    if not DB_AVAILABLE or SessionLocal is None:
        return
    db = SessionLocal()
    try:
        u = get_user_by_telegram_id(db, tid)
        scope = BotCommandScopeChat(chat_id=tid)
        if u and getattr(u, "is_admin", False):
            await bot.set_my_commands(_admin_bot_commands(), scope=scope)
            logger.info("User bot: меню админа обновлено для chat_id=%s", tid)
        else:
            try:
                await bot.delete_my_commands(scope=scope)
            except Exception:  # noqa: BLE001
                logger.debug("delete_my_commands chat_id=%s", tid)
    finally:
        db.close()


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

    regular = _regular_user_bot_commands()
    admin_cmds = _admin_bot_commands()
    try:
        await bot.set_my_commands(regular, scope=BotCommandScopeDefault())
        logger.info("User bot: меню по умолчанию (пользователи): %d команд", len(regular))
    except Exception:
        logger.exception("User bot: не удалось set_my_commands (default scope)")

    n_admins = 0
    if DB_AVAILABLE and SessionLocal is not None:
        db = SessionLocal()
        try:
            for chat_id in list_admin_telegram_ids(db):
                try:
                    await bot.set_my_commands(admin_cmds, scope=BotCommandScopeChat(chat_id=chat_id))
                    n_admins += 1
                except Exception:
                    logger.exception("User bot: set_my_commands для admin chat_id=%s", chat_id)
        finally:
            db.close()
    logger.info("User bot: расширенное меню для админов: %d чат(ов)", n_admins)

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


async def notify_bot_startup() -> None:
    """Webhook для бота уведомлений (отдельный токен от USER_BOT)."""
    token = (getattr(settings, "telegram_bot_token", "") or "").strip()
    user_tok = (getattr(settings, "user_bot_token", "") or "").strip()
    base = (getattr(settings, "public_base_url", "") or "").strip().rstrip("/")
    if not token or not base:
        logger.info("Notify bot: skipped (no TELEGRAM_BOT_TOKEN or PUBLIC_BASE_URL)")
        return
    if user_tok and token == user_tok:
        logger.info("Notify bot: skipped (TELEGRAM_BOT_TOKEN совпадает с USER_BOT_TOKEN)")
        return
    bot, _ = get_notify_bot_dp()
    if not bot:
        logger.warning("Notify bot: get_notify_bot_dp() вернул None")
        return
    secret = (getattr(settings, "telegram_bot_webhook_secret", "") or "").strip()
    url = f"{base}/bot/notify/webhook"
    try:
        await bot.set_webhook(
            url=url,
            secret_token=secret or None,
            drop_pending_updates=False,
        )
        logger.info("Notify bot webhook set: %s (secret=%s)", url, "yes" if secret else "no")
    except Exception:
        logger.exception("Notify bot set_webhook FAILED")


async def notify_bot_shutdown() -> None:
    bot, _ = get_notify_bot_dp()
    if not bot:
        return
    try:
        await bot.delete_webhook(drop_pending_updates=False)
        await bot.session.close()
    except Exception as e:  # noqa: BLE001
        logger.debug("notify bot shutdown: %s", e)
