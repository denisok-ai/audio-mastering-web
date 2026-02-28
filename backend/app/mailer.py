"""SMTP email module for Magic Master (P22).

Отправка одиночных писем и массовых рассылок через встроенный smtplib.
Конфигурация через переменные окружения:
  MAGIC_MASTER_SMTP_HOST     — хост SMTP сервера (gmail: smtp.gmail.com)
  MAGIC_MASTER_SMTP_PORT     — порт (по умолчанию 587)
  MAGIC_MASTER_SMTP_USER     — логин (ваш email)
  MAGIC_MASTER_SMTP_PASSWORD — пароль или App Password
  MAGIC_MASTER_SMTP_FROM     — адрес отправителя (если отличается от user)
  MAGIC_MASTER_SMTP_USE_TLS  — true/false (по умолчанию true, STARTTLS)
"""
import logging
import smtplib
import ssl
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Optional

from .config import settings

logger = logging.getLogger("mailer")


def _is_configured() -> bool:
    """Проверить, настроен ли SMTP."""
    return bool(
        getattr(settings, "smtp_host", "").strip()
        and getattr(settings, "smtp_user", "").strip()
        and getattr(settings, "smtp_password", "").strip()
    )


def send_email(
    to: str,
    subject: str,
    html: str,
    text: str = "",
    from_addr: Optional[str] = None,
) -> None:
    """Отправить одно письмо через SMTP.

    Если SMTP не настроен (MAGIC_MASTER_SMTP_HOST пустой), функция ничего не делает
    и пишет предупреждение в лог.

    Args:
        to: email получателя
        subject: тема
        html: HTML-версия тела письма
        text: plain-text версия (необязательно)
        from_addr: адрес отправителя (если None — берётся из конфига)
    """
    if not _is_configured():
        logger.warning("SMTP не настроен, письмо не отправлено: %s", to)
        return

    host = settings.smtp_host.strip()
    port = int(getattr(settings, "smtp_port", 587))
    user = settings.smtp_user.strip()
    password = settings.smtp_password.strip()
    use_tls = bool(getattr(settings, "smtp_use_tls", True))
    sender = from_addr or getattr(settings, "smtp_from", "").strip() or user

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = to

    if text:
        msg.attach(MIMEText(text, "plain", "utf-8"))
    msg.attach(MIMEText(html, "html", "utf-8"))

    ctx = ssl.create_default_context()
    try:
        if port == 465:
            with smtplib.SMTP_SSL(host, port, context=ctx) as server:
                server.login(user, password)
                server.sendmail(sender, [to], msg.as_string())
        else:
            with smtplib.SMTP(host, port) as server:
                server.ehlo()
                if use_tls:
                    server.starttls(context=ctx)
                    server.ehlo()
                server.login(user, password)
                server.sendmail(sender, [to], msg.as_string())
        logger.info("Письмо отправлено: %s → %s", subject, to)
    except Exception as exc:  # noqa: BLE001
        logger.error("Ошибка отправки письма %s: %s", to, exc)
        raise


# ─── Transactional templates ──────────────────────────────────────────────────

def _base_html(title: str, body_html: str) -> str:
    return f"""<!DOCTYPE html>
<html lang="ru"><head><meta charset="UTF-8">
<style>
body{{font-family:system-ui,sans-serif;background:#f4f4f8;color:#1a1a2e;margin:0;padding:2rem}}
.box{{background:#fff;border-radius:10px;padding:2rem;max-width:520px;margin:0 auto;box-shadow:0 2px 16px rgba(0,0,0,.08)}}
h2{{color:#7c3aed;margin-top:0}}
a.btn{{display:inline-block;background:#7c3aed;color:#fff;padding:.6rem 1.4rem;border-radius:6px;text-decoration:none;font-weight:600;margin-top:1rem}}
.footer{{text-align:center;margin-top:1.5rem;font-size:.78rem;color:#888}}
</style></head>
<body><div class="box">
<h2>{title}</h2>
{body_html}
<div class="footer">Magic Master · Автоматический мастеринг</div>
</div></body></html>"""


def send_welcome_email(to: str, email: str) -> None:
    """Приветственное письмо после регистрации."""
    html = _base_html(
        "Добро пожаловать в Magic Master!",
        f"""<p>Привет, <strong>{email}</strong>!</p>
<p>Ваш аккаунт успешно создан. Начните мастерить прямо сейчас — первые 3 трека в день бесплатно.</p>
<p>Для неограниченного доступа и приоритетной обработки перейдите на тариф <strong>Pro</strong> или <strong>Studio</strong>.</p>
<a class="btn" href="https://magicmaster.app/app">Начать мастеринг →</a>""",
    )
    try:
        send_email(to=to, subject="Добро пожаловать в Magic Master!", html=html,
                   text=f"Привет, {email}! Ваш аккаунт создан. Начните мастеринг: https://magicmaster.app/app")
    except Exception:  # noqa: BLE001
        pass


def send_subscription_expiry_warning_email(to: str, tier: str, expires_at: float, days_left: int) -> None:
    """Предупреждение об истечении подписки за N дней. P32."""
    import datetime
    tier_label = {"pro": "Pro", "studio": "Studio"}.get(tier, tier)
    exp_str = datetime.datetime.fromtimestamp(expires_at).strftime('%d.%m.%Y')
    html = _base_html(
        f"Ваша подписка {tier_label} истекает через {days_left} д.",
        f"""<p>Привет! Ваш план <strong>{tier_label}</strong> истекает <strong>{exp_str}</strong> ({days_left} {'день' if days_left == 1 else 'дня' if 2 <= days_left <= 4 else 'дней'}).</p>
<p>Чтобы не потерять доступ к расширенным возможностям, продлите подписку прямо сейчас.</p>
<a class="btn" href="https://magicmaster.app/pricing">Продлить подписку →</a>
<p style="font-size:.85rem;color:#999;margin-top:1rem">Если вы не хотите продлевать, аккаунт автоматически перейдёт на тариф Free после истечения.</p>""",
    )
    try:
        send_email(
            to=to,
            subject=f"Подписка {tier_label} истекает через {days_left} д. — Magic Master",
            html=html,
            text=f"Ваша подписка {tier_label} истекает {exp_str}. Продлите: https://magicmaster.app/pricing",
        )
    except Exception:  # noqa: BLE001
        pass


def send_subscription_expired_email(to: str, tier: str) -> None:
    """Уведомление о том, что подписка истекла и тариф понижен. P32."""
    tier_label = {"pro": "Pro", "studio": "Studio"}.get(tier, tier)
    html = _base_html(
        f"Подписка {tier_label} истекла",
        f"""<p>Ваша подписка <strong>{tier_label}</strong> истекла. Ваш аккаунт автоматически переведён на тариф <strong>Free</strong>.</p>
<p>Мастеринг доступен в ограниченном режиме (3 трека/день).</p>
<a class="btn" href="https://magicmaster.app/pricing">Восстановить подписку →</a>""",
    )
    try:
        send_email(
            to=to,
            subject=f"Подписка {tier_label} истекла — Magic Master",
            html=html,
            text=f"Ваша подписка {tier_label} истекла. Восстановите: https://magicmaster.app/pricing",
        )
    except Exception:  # noqa: BLE001
        pass


def send_subscription_activated_email(to: str, tier: str, expires_at: Optional[float] = None) -> None:
    """Уведомление об активации подписки."""
    import time, datetime
    tier_label = {"pro": "Pro", "studio": "Studio", "free": "Free"}.get(tier, tier)
    exp_str = ""
    if expires_at:
        exp_str = f"<p>Действует до: <strong>{datetime.datetime.fromtimestamp(expires_at).strftime('%d.%m.%Y')}</strong></p>"
    html = _base_html(
        f"Подписка {tier_label} активирована",
        f"""<p>Поздравляем! Ваш тариф <strong>{tier_label}</strong> успешно активирован.</p>
{exp_str}
<p>Теперь вам доступны все возможности плана {tier_label}: приоритетная обработка, расширенные процессоры и {'без лимита мастерингов в день' if tier != 'free' else ''}.</p>
<a class="btn" href="https://magicmaster.app/app">Перейти в приложение →</a>""",
    )
    try:
        send_email(to=to, subject=f"Подписка {tier_label} активирована — Magic Master", html=html,
                   text=f"Ваша подписка {tier_label} активирована. Перейдите в приложение: https://magicmaster.app/app")
    except Exception:  # noqa: BLE001
        pass


def send_email_verification(to: str, verify_url: str) -> None:
    """Письмо для подтверждения email после регистрации. P41."""
    html = _base_html(
        "Подтвердите ваш email — Magic Master",
        f"""<p>Спасибо за регистрацию в <strong>Magic Master</strong>!</p>
<p>Нажмите кнопку ниже, чтобы подтвердить ваш email и активировать аккаунт.</p>
<a class="btn" href="{verify_url}">Подтвердить email →</a>
<p style="font-size:.85rem;color:#999;margin-top:1.2rem">Ссылка действительна <strong>24 часа</strong>.<br>
Если вы не регистрировались — проигнорируйте это письмо.</p>""",
    )
    try:
        send_email(
            to=to,
            subject="Подтвердите email — Magic Master",
            html=html,
            text=f"Подтвердите email по ссылке: {verify_url}\nСсылка действительна 24 часа.",
        )
    except Exception:  # noqa: BLE001
        pass


def send_password_reset_email(to: str, reset_url: str) -> None:
    """Письмо со ссылкой для сброса пароля. P35."""
    html = _base_html(
        "Сброс пароля Magic Master",
        f"""<p>Мы получили запрос на сброс пароля для этого аккаунта.</p>
<p>Нажмите кнопку ниже, чтобы задать новый пароль. Ссылка действительна <strong>1 час</strong>.</p>
<a class="btn" href="{reset_url}">Сбросить пароль →</a>
<p style="font-size:.85rem;color:#999;margin-top:1.2rem">Если вы не запрашивали сброс пароля, просто проигнорируйте это письмо.</p>""",
    )
    try:
        send_email(
            to=to,
            subject="Сброс пароля — Magic Master",
            html=html,
            text=f"Для сброса пароля перейдите по ссылке: {reset_url}\nСсылка действует 1 час.",
        )
    except Exception:  # noqa: BLE001
        pass
