"""
Telegram-уведомления для администратора. P51.

Отправляет сообщения через Telegram Bot API (HTTP).
Настройка через переменные окружения:
  MAGIC_MASTER_TELEGRAM_BOT_TOKEN    — токен бота от @BotFather
  MAGIC_MASTER_TELEGRAM_ADMIN_CHAT_ID — chat_id получателя (строка или число)

Если переменные не заданы — все вызовы тихо игнорируются.
"""

from __future__ import annotations

import urllib.request
import urllib.parse
import json
import threading
import time
import datetime
from typing import Optional

from .config import settings


def _is_configured() -> bool:
    return bool(
        getattr(settings, "telegram_bot_token", "") and
        getattr(settings, "telegram_admin_chat_id", "")
    )


def _send_raw(text: str) -> None:
    """Синхронная отправка; вызывается из фонового потока."""
    if not _is_configured():
        return
    token   = settings.telegram_bot_token.strip()
    chat_id = settings.telegram_admin_chat_id.strip()
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = json.dumps({
        "chat_id": chat_id,
        "text": text,
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    }).encode("utf-8")
    try:
        req = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=8):
            pass
    except Exception:
        pass  # не мешаем основному потоку


def notify(text: str) -> None:
    """Отправить произвольное сообщение в Telegram (асинхронно, в фоновом потоке)."""
    if not _is_configured():
        return
    t = threading.Thread(target=_send_raw, args=(text,), daemon=True)
    t.start()


# ─── Шаблоны уведомлений ──────────────────────────────────────────────────────

def _ts() -> str:
    return datetime.datetime.now().strftime("%d.%m.%Y %H:%M")


def notify_new_user(email: str, tier: str) -> None:
    """Новый пользователь зарегистрировался."""
    msg = (
        f"👤 <b>Новый пользователь</b>\n"
        f"Email: <code>{email}</code>\n"
        f"Тариф: <b>{tier}</b>\n"
        f"🕐 {_ts()}"
    )
    notify(msg)


def notify_payment(email: str, amount: float, currency: str, tier: str) -> None:
    """Успешный платёж."""
    msg = (
        f"💰 <b>Оплата получена</b>\n"
        f"Email: <code>{email}</code>\n"
        f"Сумма: <b>{amount:,.0f} {currency}</b>\n"
        f"Тариф: <b>{tier}</b>\n"
        f"🕐 {_ts()}"
    )
    notify(msg)


def notify_payment_failed(email: str, amount: float, currency: str) -> None:
    """Платёж не прошёл."""
    msg = (
        f"⚠️ <b>Платёж отклонён</b>\n"
        f"Email: <code>{email}</code>\n"
        f"Сумма: {amount:,.0f} {currency}\n"
        f"🕐 {_ts()}"
    )
    notify(msg)


def notify_mastering_error(filename: str, error: str, user_email: Optional[str] = None) -> None:
    """Ошибка мастеринга."""
    user_part = f"\nПользователь: <code>{user_email}</code>" if user_email else ""
    msg = (
        f"❌ <b>Ошибка мастеринга</b>{user_part}\n"
        f"Файл: <code>{filename[:60]}</code>\n"
        f"Ошибка: <code>{error[:200]}</code>\n"
        f"🕐 {_ts()}"
    )
    notify(msg)


def notify_server_startup(version: str, host: str) -> None:
    """Сервер запустился."""
    msg = (
        f"🚀 <b>Magic Master запущен</b>\n"
        f"Версия: <b>{version}</b>\n"
        f"Хост: <code>{host}</code>\n"
        f"🕐 {_ts()}"
    )
    notify(msg)


def notify_backup_done(filename: str, size_mb: float) -> None:
    """Бэкап БД создан."""
    msg = (
        f"💾 <b>Бэкап базы данных</b>\n"
        f"Файл: <code>{filename}</code>\n"
        f"Размер: {size_mb:.1f} МБ\n"
        f"🕐 {_ts()}"
    )
    notify(msg)


def notify_user_blocked(email: str, admin_email: str) -> None:
    """Пользователь заблокирован."""
    msg = (
        f"🔒 <b>Пользователь заблокирован</b>\n"
        f"Email: <code>{email}</code>\n"
        f"Администратор: <code>{admin_email}</code>\n"
        f"🕐 {_ts()}"
    )
    notify(msg)
