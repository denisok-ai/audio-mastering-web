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
from . import settings_store


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


# ─── Алерты мониторинга (очередь 3.3) ─────────────────────────────────────────
_alert_last_sent: dict[str, float] = {}
_ALERT_KEY_QUEUE = "queue"
_ALERT_KEY_HEALTH = "health"


def _alert_setting(name: str, default: Optional[object]) -> Optional[object]:
    """Читает настройку алертов: сначала из БД (админка), затем из config."""
    val = settings_store.get_setting(name)
    if val is not None:
        if name == "alert_monitoring_enabled":
            return val is True or (isinstance(val, str) and val.strip().lower() in ("1", "true", "yes", "on"))
        if name == "alert_queue_threshold":
            try:
                return int(val) if val is not None else 0
            except (TypeError, ValueError):
                return 0
        if name == "alert_throttle_minutes":
            try:
                return int(val) if val is not None else 60
            except (TypeError, ValueError):
                return 60
    return getattr(settings, name, default)


def _should_send_alert(key: str) -> bool:
    """Отправлять не чаще чем раз в alert_throttle_minutes."""
    if not _alert_setting("alert_monitoring_enabled", False):
        return False
    mins = _alert_setting("alert_throttle_minutes", 60) or 60
    now = time.time()
    last = _alert_last_sent.get(key, 0)
    if now - last < mins * 60:
        return False
    _alert_last_sent[key] = now
    return True


def notify_alert_queue_threshold(jobs_total: int, jobs_running: int) -> None:
    """Превышен порог очереди задач (отправка не чаще раз в throttle)."""
    if not _is_configured() or not _should_send_alert(_ALERT_KEY_QUEUE):
        return
    threshold = _alert_setting("alert_queue_threshold", 0) or 0
    if threshold <= 0 or jobs_total < threshold:
        return
    msg = (
        f"⚠️ <b>Очередь мастеринга</b>\n"
        f"Задач в кэше: <b>{jobs_total}</b> (порог: {threshold})\n"
        f"В работе: <b>{jobs_running}</b>\n"
        f"🕐 {_ts()}"
    )
    notify(msg)


def notify_alert_health_degraded(reason: str, details: Optional[str] = None) -> None:
    """Health-check в состоянии degraded/error (отправка не чаще раз в throttle)."""
    if not _is_configured() or not _should_send_alert(_ALERT_KEY_HEALTH):
        return
    detail_line = f"\nДетали: <code>{details[:150]}</code>" if details else ""
    msg = (
        f"⚠️ <b>Health деградация</b>\n"
        f"Причина: {reason}{detail_line}\n"
        f"🕐 {_ts()}"
    )
    notify(msg)


def notify_operational_anomaly(anomaly_key: str, title_ru: str, details_ru: str) -> None:
    """Аномалия эксплуатации: отдельный ключ троттлинга на каждый тип (anom_*)."""
    if not _is_configured():
        return
    key = f"anom_{anomaly_key}"
    if not _should_send_alert(key):
        return
    msg = f"🚨 <b>{title_ru}</b>\n{details_ru}\n🕐 {_ts()}"
    notify(msg)
