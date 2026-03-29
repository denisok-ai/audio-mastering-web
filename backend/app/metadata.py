# @file metadata.py
# @description Брендинг в метаданных экспортируемых файлов (MP3/FLAC).

from __future__ import annotations

import logging
import os
import tempfile
from io import BytesIO

logger = logging.getLogger(__name__)

_BRAND = "Magic Master"
_URL = "magicmaster.pro"


def embed_magic_master_branding(audio_bytes: bytes, out_format: str) -> bytes:
    """
    Добавляет теги MASTERED_BY / encoder в MP3 и FLAC.
    WAV без изменений (RIFF LIST — опционально позже).
    """
    fmt = (out_format or "wav").lower()
    if fmt == "wav" or not audio_bytes:
        return audio_bytes
    try:
        if fmt == "mp3":
            return _tag_mp3(audio_bytes)
        if fmt == "flac":
            return _tag_flac(audio_bytes)
    except Exception as e:  # noqa: BLE001
        logger.warning("metadata branding skipped: %s", e)
    return audio_bytes


def _tag_mp3(data: bytes) -> bytes:
    from mutagen.id3 import ID3, TENC, TXXX, ID3NoHeaderError
    from mutagen.mp3 import MP3

    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
        tmp.write(data)
        path = tmp.name
    try:
        try:
            audio = MP3(path, ID3=ID3)
        except ID3NoHeaderError:
            audio = MP3(path)
            audio.add_tags()
        if audio.tags is None:
            audio.add_tags()
        try:
            audio.tags.delall("TXXX:MASTERED_BY")
        except Exception:  # noqa: BLE001
            pass
        audio.tags.add(
            TXXX(encoding=3, desc="MASTERED_BY", text=_URL)
        )
        audio.tags.add(TENC(encoding=3, text=f"{_BRAND} ({_URL})"))
        audio.save(path)
        with open(path, "rb") as f:
            return f.read()
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def _tag_flac(data: bytes) -> bytes:
    from mutagen.flac import FLAC

    bio = BytesIO(data)
    audio = FLAC(bio)
    audio["ENCODER"] = f"{_BRAND} ({_URL})"
    audio["MASTERED_BY"] = _URL
    out = BytesIO()
    audio.save(out)
    return out.getvalue()
