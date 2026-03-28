"""Постинг в Telegram-канал (urllib, без aiogram session в потоках)."""
from __future__ import annotations

import json
import logging
import urllib.request

from ..config import settings

logger = logging.getLogger(__name__)


def post_to_channel_html(text: str) -> bool:
    """Отправить HTML в канал user_bot_channel_id."""
    token = (getattr(settings, "user_bot_token", "") or "").strip()
    chat = (getattr(settings, "user_bot_channel_id", "") or "").strip()
    if not token or not chat:
        return False
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = json.dumps(
        {
            "chat_id": chat,
            "text": text[:4090],
            "parse_mode": "HTML",
            "disable_web_page_preview": True,
        },
        ensure_ascii=False,
    ).encode("utf-8")
    try:
        req = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=20):
            pass
        return True
    except Exception as e:  # noqa: BLE001
        logger.debug("post_to_channel_html: %s", e)
        return False


def post_news_to_channel_sync(title: str, body_plain: str) -> None:
    """Новость для канала (кратко)."""
    esc = lambda s: (s or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")[:3500]
    text = f"📰 <b>{esc(title)}</b>\n\n{esc(body_plain)}"
    post_to_channel_html(text)
