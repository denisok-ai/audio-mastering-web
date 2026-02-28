# @file helpers.py
# @description Общие хелперы для main: IP, валидация файлов, magic bytes. P62.
# @dependencies app.config, app.settings_store

from fastapi import Request

from .config import settings
from . import settings_store


def get_client_ip(request: Request) -> str:
    """Извлекает IP клиента (учитывает Nginx X-Forwarded-For)."""
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def allowed_file(filename: str) -> bool:
    """Проверяет, что расширение файла в списке разрешённых (из админки или .env)."""
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    allowed = settings_store.get_setting("allowed_extensions")
    if allowed is None:
        allowed = getattr(settings, "allowed_extensions", {"wav", "mp3", "flac"})
    return ext in (allowed if isinstance(allowed, (set, list)) else [])


def check_audio_magic_bytes(data: bytes, filename: str) -> bool:
    """
    Проверяет, что содержимое файла соответствует заявленному расширению (magic bytes).
    P60. Возвращает True если проверка пройдена или пропущена.
    """
    if not data:
        return True
    ext = (filename.rsplit(".", 1)[-1].lower() if "." in filename else "").strip()
    if ext == "wav":
        return len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WAVE"
    if ext == "flac":
        return len(data) >= 4 and data[:4] == b"fLaC"
    if ext == "mp3":
        if len(data) >= 3 and data[:3] == b"ID3":
            return True
        return len(data) >= 2 and data[0] == 0xFF and (data[1] & 0xE0) == 0xE0
    return True
