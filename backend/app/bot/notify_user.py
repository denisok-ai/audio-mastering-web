"""Синхронная отправка сообщений через user bot (без aiogram, из потоков мастеринга)."""
from __future__ import annotations

import json
import logging
import urllib.request

logger = logging.getLogger(__name__)


def send_user_bot_text_sync(chat_id: int, text: str, parse_mode: str = "HTML") -> bool:
    from ..config import settings

    token = (getattr(settings, "user_bot_token", "") or "").strip()
    if not token or not chat_id:
        return False
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = json.dumps(
        {
            "chat_id": chat_id,
            "text": text[:4090],
            "parse_mode": parse_mode,
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
        logger.debug("send_user_bot_text_sync failed: %s", e)
        return False
